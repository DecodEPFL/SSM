# python
# file: src/neural_ssm/ssm/selective_cells.py
"""
Time-varying / selective SSM cell implementations:
  RobustMambaDiagSSM, RobustMambaDiagLTI
"""
import math
import warnings
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan_utils import associative_scan, binary_operator_diag, diag_affine_scan, GraphedDiagScan
from .state_utils import (
    resolve_runtime_state as _resolve_runtime_state,
    reset_runtime_state as _reset_runtime_state,
)
from .cache_utils import EvalCacheMixin
from ..static_layers.lipschitz_mlps import L2BoundedLinearExact


def _normalize_to_3d(x: torch.Tensor) -> torch.Tensor:
    """Accept (D,), (T,D), or (B,T,D) and return (B,T,D)."""
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 2:
        return x[None, :, :]
    if x.dim() == 3:
        return x
    raise ValueError(f"Invalid input dimensions {x.dim()}, expected 1, 2, or 3.")


# ----------------------------
# Helpers: exact spectral norm of [[a,b],[c,0]]
# ----------------------------
def spectral_norm_2x2_a_b_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Exact spectral norm of K = [[a, b],
                                [c, 0]].
    a, b, c broadcastable (e.g. (B,T,N)). Returns tensor of same shape.
    """
    p = a * a + c * c
    r = b * b
    q = a * b
    disc = (p - r) * (p - r) + 4.0 * q * q
    lam_max = 0.5 * (p + r + torch.sqrt(disc + eps))
    return torch.sqrt(lam_max + eps)


def spectral_norm_2x2_abcd(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Exact || [[a,b],[c,d]] ||_2 for broadcastable real tensors.
    """
    tau = a * a + b * b + c * c + d * d
    det2 = (a * d - b * c) ** 2
    inner = (tau * tau - 4.0 * det2).clamp_min(0.0)
    lam_max = 0.5 * (tau + torch.sqrt(inner + eps))
    return torch.sqrt(lam_max + eps)


# ----------------------------
# Diagonal time-varying recurrence scan
# ----------------------------
def _diag_scan(
    a_tb: torch.Tensor, bu_tb: torch.Tensor, z0: torch.Tensor, scan_impl=None
) -> torch.Tensor:
    """
    Parallel prefix scan for the diagonal real recurrence:
        z_{t+1} = a_t ⊙ z_t + bu_t

    Uses binary_operator_diag from scan_utils — the same JIT-compiled operator
    used for LRU — via the work-efficient O(T) recursive associative_scan.
    This replaces the earlier odd-stride scan which had O(T log T) work and
    O(log T) intermediate tensor allocations.

    a_tb, bu_tb : (T, B, N)  — time-major
    z0          : (B, N)     — initial state
    scan_impl   : optional callable(a, b) -> inclusive scan (T,B,N). Defaults to
                  the eager :func:`diag_affine_scan`. A :class:`GraphedDiagScan`
                  instance can be passed to replay a captured CUDA graph instead
                  (same maths, far less launch overhead).
    Returns states : (T+1, B, N)  — [z_0, z_1, ..., z_T]
    """
    T, B, N = a_tb.shape
    if T == 0:
        return z0.unsqueeze(0)

    # Fold z0 into the first time step so that after the inclusive prefix scan
    # b_prefix[t] == z_{t+1} directly (mirrors _scan_diag_complex in scan_utils).
    bu_tb = bu_tb.clone()
    bu_tb[0] = bu_tb[0] + a_tb[0] * z0

    scan = scan_impl if scan_impl is not None else diag_affine_scan
    z_next = scan(a_tb, bu_tb)  # (T,B,N)

    states = torch.empty(T + 1, B, N, device=a_tb.device, dtype=a_tb.dtype)
    states[0] = z0
    states[1:] = z_next
    return states


def _spectral_cap(weight: torch.Tensor, bound: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return ``weight`` rescaled so its spectral norm is ``<= bound``.

    Like :meth:`DeepSSM._spectrally_capped_weight`, this only ever shrinks the
    weight (``divisor >= 1``); it never amplifies an under-sized matrix. ``bound``
    may be a (possibly grad-carrying) scalar tensor so the cap adapts to learnable
    gain budgets, in which case gradients flow into ``bound`` while the cap binds.
    """
    weight_for_norm = (
        weight.float() if weight.dtype in (torch.float16, torch.bfloat16) else weight
    )
    sigma = torch.linalg.matrix_norm(weight_for_norm, ord=2).to(
        device=weight.device, dtype=weight.dtype
    )
    if not torch.is_tensor(bound):
        bound = torch.as_tensor(bound, device=weight.device, dtype=weight.dtype)
    divisor = torch.clamp(sigma / bound.clamp_min(eps), min=1.0)
    return weight / divisor


# ----------------------------
# Robust Mamba-style selective diagonal SSM with prescribed ℓ2 gain
# ----------------------------
class RobustMambaDiagSSM(nn.Module):
    r"""
    Discrete-time selective diagonal SSM (Mamba-like), robustly ℓ2-bounded.

    Input:  u_t ∈ R^D,  shape (B,T,D)
    Projections (bounded):
        ũ_t = W_in u_t,    ||W_in||_2  <= proj_bound
        y_t  = W_out ŷ_t,  ||W_out||_2 <= proj_bound

    Selective params from u_t:
        (delta_t, b_t, c_t) = ParamNet(u_t)
        a_t = exp( -softplus(delta_t + bias) ⊙ softplus(alpha) )  ∈ (0,1)

    State/output (no direct feedthrough):
        z_{t+1} = a_t ⊙ z_t + b_t ⊙ (γ ũ_t)
        ŷ_t     = c_t ⊙ z_t
        y_t     = W_out ŷ_t

    L2 gain certificate:
        Per-coordinate: ||[[a_t, b_t],[c_t, 0]]||_2 ≤ 1  (exact closed-form normalization).
        Combined with non-expansive projections: ||y||_ℓ2 ≤ γ ||u||_ℓ2  for z_0=0.

    Design notes:
      - No direct feedthrough (D=0): output at time t depends only on state z_t, giving
        a one-step causal delay on direct input→output paths.
      - If d_state ≠ d_model, the overall gain is γ·||W_in||·||W_out|| ≤ γ (proj_bound ≤ 1).
      - bc_nonlinearity="identity": b,c are raw network outputs, unbounded before normalization.
        Gradient flow through large-scale factors can vanish; use "tanh" (default) for stability.
      - mode="scan" uses the package's associative_scan + binary_operator_diag (same as LRU).
      - exact_norm=True (default): exact SVD each forward, hard gain guarantee.
        exact_norm=False: power iteration, approximate bound, cheaper for large d_model.
    """

    def __init__(
        self,
        d_model: int,
        d_state: Optional[int] = None,
        d_out: Optional[int] = None,
        *,
        gamma: float = 1.0,
        train_gamma: bool = True,
        eps_a: float = 1e-4,
        param_net: Literal["linear", "mlp"] = "linear",
        hidden: int = 128,
        init_rho: float = 0.99,
        init_delta0: float = 1.0,
        init_param_scale: float = 0.02,
        bc_nonlinearity: Literal["tanh", "identity"] = "tanh",
        proj_bound: float = 1.0,
        exact_norm: bool = True,
        power_iters: int = 1,
        learn_x0: bool = False,
        use_cuda_graph: bool = False,
    ):
        super().__init__()
        self.D = int(d_model)
        self.N = int(d_state if d_state is not None else d_model)
        self.D_out = int(d_out if d_out is not None else d_model)
        self.state: Optional[torch.Tensor] = None
        # CUDA-graph cache for the diagonal scan (no-op when disabled / on CPU).
        self._graphed_scan = GraphedDiagScan(enabled=use_cuda_graph)

        if bc_nonlinearity == "identity":
            warnings.warn(
                "RobustMambaDiagSSM: bc_nonlinearity='identity' leaves b and c unbounded "
                "before normalization. Gradient flow through large-scale normalization can "
                "vanish. Use bc_nonlinearity='tanh' (default) for stable training.",
                UserWarning,
                stacklevel=2,
            )

        # gamma (>0)
        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        self.eps_a = float(eps_a)
        self.bc_nonlinearity = bc_nonlinearity

        self.in_proj  = L2BoundedLinearExact(self.D, self.N,     bound=proj_bound,
                                             exact_norm=exact_norm, power_iters=power_iters)
        self.out_proj = L2BoundedLinearExact(self.N, self.D_out, bound=proj_bound,
                                             exact_norm=exact_norm, power_iters=power_iters)

        # alpha > 0 controls base decay: a_t = exp(-delta_t * alpha)
        self.alpha_log = nn.Parameter(torch.zeros(self.N))

        out_dim = 3 * self.N
        if param_net == "linear":
            self.param_net = nn.Linear(self.D, out_dim)
        elif param_net == "mlp":
            self.param_net = nn.Sequential(
                nn.Linear(self.D, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            raise ValueError(param_net)

        self.delta_bias = nn.Parameter(torch.zeros(self.N))

        self.reset_parameters(
            init_rho=init_rho, init_delta0=init_delta0, init_param_scale=init_param_scale
        )

        # Learnable initial condition (real, shape (1, N))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, self.N))
        else:
            self.register_buffer('x0_param', None)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    @torch.no_grad()
    def reset_parameters(self, *, init_rho: float, init_delta0: float, init_param_scale: float):
        rho    = min(max(float(init_rho),    1e-4), 1 - 1e-6)
        delta0 = max(float(init_delta0), 1e-4)

        alpha0 = (-math.log(rho)) / delta0
        self.alpha_log.fill_(
            math.log(math.expm1(alpha0)) if alpha0 > 1e-6 else math.log(alpha0 + 1e-6)
        )
        self.delta_bias.fill_(
            math.log(math.expm1(delta0)) if delta0 > 1e-6 else math.log(delta0 + 1e-6)
        )

        # Only re-initialise param_net layers — in_proj and out_proj have their own init.
        for m in self.param_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_param_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_params(
        self, u_bt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u_bt (B,T,D) → (a_bt, b_bt, c_bt, u_scaled_bt), all (B,T,N).

        a ∈ (0, 1-eps_a), b and c normalised so ||[[a,b],[c,0]]||_2 ≤ 1.
        u_scaled = γ · W_in u.
        """
        B, T, D = u_bt.shape
        assert D == self.D

        u_tilde  = self.in_proj(u_bt)                                  # (B,T,N)
        g        = self.gamma.to(device=u_bt.device, dtype=u_bt.dtype)
        u_scaled = g * u_tilde                                         # (B,T,N)

        raw = self.param_net(u_bt)                                     # (B,T,3N)
        delta_raw, b_raw, c_raw = raw.split(self.N, dim=-1)

        delta = F.softplus(delta_raw + self.delta_bias)                # (B,T,N)
        alpha = F.softplus(self.alpha_log).view(1, 1, self.N)          # (1,1,N)
        a     = torch.exp(-delta * alpha).clamp(max=1.0 - self.eps_a) # (B,T,N)

        if self.bc_nonlinearity == "tanh":
            b, c = torch.tanh(b_raw), torch.tanh(c_raw)
        else:   # "identity"
            b, c = b_raw, c_raw

        sigma = spectral_norm_2x2_a_b_c(a, b, c)   # (B,T,N)
        scale = torch.clamp(sigma, min=1.0)
        a, b, c = a / scale, b / scale, c / scale

        return a, b, c, u_scaled

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        *,
        mode: Literal["scan", "loop"] = "scan",
        return_state: bool = True,
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        """
        u : (B,T,D) or (T,D) or (D,) — normalized to (B,T,D) internally.

        Returns:
          y      : (B,T,D_out)
          z_seq  : (B,T+1,N)   if return_state   [z_0..z_T]
          z_last : (B,N)       if return_last     [z_T]
        """
        u = _normalize_to_3d(u)
        B, T, D = u.shape
        assert D == self.D, f"Expected d_model={self.D}, got {D}"
        device, dtype = u.device, u.dtype

        z0 = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=B,
            n_state=self.N,
            device=device,
            dtype=dtype,
            x0=self.x0_param,
        )

        a_bt, b_bt, c_bt, u_scaled_bt = self._compute_params(u)   # (B,T,N)
        bu_bt = b_bt * u_scaled_bt                                 # (B,T,N)

        if mode == "loop":
            z = z0
            y_hat = torch.empty(B, T, self.N, device=device, dtype=dtype)
            z_seq = torch.empty(B, T + 1, self.N, device=device, dtype=dtype) if return_state else None
            if return_state:
                z_seq[:, 0, :] = z0

            for t in range(T):
                y_hat[:, t, :] = c_bt[:, t, :] * z
                z = a_bt[:, t, :] * z + bu_bt[:, t, :]
                if return_state:
                    z_seq[:, t + 1, :] = z

            z_last = z

        else:  # scan
            a_tb  = a_bt.transpose(0, 1).contiguous()   # (T,B,N)
            bu_tb = bu_bt.transpose(0, 1).contiguous()  # (T,B,N)

            states = _diag_scan(a_tb, bu_tb, z0, scan_impl=self._graphed_scan)  # (T+1,B,N)
            z_last = states[-1]                           # (B,N)

            if return_state:
                z_seq = states.transpose(0, 1).contiguous()   # (B,T+1,N)
                z_bt  = z_seq[:, :-1, :]                       # (B,T,N)
            else:
                z_seq = None
                z_bt  = states[:-1].transpose(0, 1).contiguous()

            y_hat = c_bt * z_bt   # (B,T,N)

        y = self.out_proj(y_hat)   # (B,T,D_out)
        self.state = z_last.detach() if detach_state else z_last

        if return_state and return_last:
            return y, z_seq, z_last
        if return_state:
            return y, z_seq
        if return_last:
            return y, z_last
        return y

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)


class RobustMambaDiagLTI(nn.Module):
    r"""
    Time-varying diagonal LTI core with explicit local l2 certificate.

    Per coordinate:
        z_{t+1} = a_t z_t + b_t (gamma * W_in u_t)
        y_t     = c_t z_t + d_t (gamma * W_in u_t)       [default]

    where [[a_t, b_t], [c_t, d_t]] is normalized so that its spectral norm
    is <= 1 exactly, coordinatewise.

    Compared to RobustMambaDiagSSM:
      - signed a_t in (-1,1), not only positive
      - direct feedthrough d_t
      - exact local normalization of [[a,b],[c,d]]
      - same fast diagonal scan
    """

    def __init__(
        self,
        d_model: int,
        d_state: Optional[int] = None,
        d_out: Optional[int] = None,
        *,
        gamma: float = 1.0,
        train_gamma: bool = True,
        eps_a: float = 1e-4,
        param_net: Literal["linear", "mlp"] = "linear",
        hidden: int = 128,
        init_rho: float = 0.995,
        init_delta0: float = 1.0,
        init_param_scale: float = 0.02,
        init_sign: float = 0.995,
        init_b: float = 0.10,
        init_c: float = 0.10,
        init_d: float = 0.10,
        bcd_nonlinearity: Literal["tanh", "identity"] = "tanh",
        proj_bound: float = 1.0,
        exact_norm: bool = True,
        power_iters: int = 1,
        output_uses_post_state: bool = False,
        learn_x0: bool = False,
        use_cuda_graph: bool = False,
    ):
        super().__init__()
        self.D = int(d_model)
        self.N = int(d_state if d_state is not None else d_model)
        self.D_out = int(d_out if d_out is not None else d_model)
        self.state: Optional[torch.Tensor] = None
        # CUDA-graph cache for the diagonal scan (no-op when disabled / on CPU).
        self._graphed_scan = GraphedDiagScan(enabled=use_cuda_graph)

        self.eps_a = float(eps_a)
        self.bcd_nonlinearity = bcd_nonlinearity
        self.output_uses_post_state = bool(output_uses_post_state)

        if output_uses_post_state:
            warnings.warn(
                "RobustMambaDiagLTI: output_uses_post_state=True may improve expressivity, "
                "but the simple exact local certificate based on [[a,b],[c,d]] is no longer "
                "the exact forward-map certificate.",
                UserWarning,
                stacklevel=2,
            )

        if bcd_nonlinearity == "identity":
            warnings.warn(
                "bcd_nonlinearity='identity' leaves b,c,d unbounded before normalization. "
                "This can train less stably; 'tanh' is usually better.",
                UserWarning,
                stacklevel=2,
            )

        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        self.in_proj = L2BoundedLinearExact(
            self.D, self.N,
            bound=proj_bound,
            exact_norm=exact_norm,
            power_iters=power_iters,
        )
        self.out_proj = L2BoundedLinearExact(
            self.N, self.D_out,
            bound=proj_bound,
            exact_norm=exact_norm,
            power_iters=power_iters,
        )

        # rho_t = exp(-delta_t * alpha), alpha > 0
        self.alpha_log = nn.Parameter(torch.zeros(self.N))
        self.delta_bias = nn.Parameter(torch.zeros(self.N))

        # a_t = rho_t * tanh(sign_raw + sign_bias)
        self.sign_bias = nn.Parameter(torch.empty(self.N))

        # Param net outputs: delta | sign | b | c | d
        out_dim = 5 * self.N
        if param_net == "linear":
            self.param_net = nn.Linear(self.D, out_dim)
        elif param_net == "mlp":
            self.param_net = nn.Sequential(
                nn.Linear(self.D, hidden),
                nn.GELU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            raise ValueError(f"param_net must be 'linear' or 'mlp', got {param_net!r}")

        self.reset_parameters(
            init_rho=init_rho,
            init_delta0=init_delta0,
            init_param_scale=init_param_scale,
            init_sign=init_sign,
            init_b=init_b,
            init_c=init_c,
            init_d=init_d,
        )

        # Learnable initial condition (real, shape (1, N))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, self.N))
        else:
            self.register_buffer('x0_param', None)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    @torch.no_grad()
    def reset_parameters(
        self,
        *,
        init_rho: float,
        init_delta0: float,
        init_param_scale: float,
        init_sign: float,
        init_b: float,
        init_c: float,
        init_d: float,
    ):
        rho = min(max(float(init_rho), 1e-4), 1.0 - 1e-6)
        delta0 = max(float(init_delta0), 1e-4)
        init_sign = max(min(float(init_sign), 1.0 - 1e-6), -1.0 + 1e-6)
        init_b = max(min(float(init_b), 0.99), -0.99)
        init_c = max(min(float(init_c), 0.99), -0.99)
        init_d = max(min(float(init_d), 0.99), -0.99)

        alpha0 = -math.log(rho) / delta0
        self.alpha_log.fill_(
            math.log(math.expm1(alpha0)) if alpha0 > 1e-6 else math.log(alpha0 + 1e-6)
        )
        self.delta_bias.fill_(
            math.log(math.expm1(delta0)) if delta0 > 1e-6 else math.log(delta0 + 1e-6)
        )
        self.sign_bias.fill_(0.5 * math.log((1 + init_sign) / (1 - init_sign)))

        for m in self.param_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_param_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_linear = None
        if isinstance(self.param_net, nn.Linear):
            last_linear = self.param_net
        elif isinstance(self.param_net, nn.Sequential):
            for m in reversed(self.param_net):
                if isinstance(m, nn.Linear):
                    last_linear = m
                    break

        if last_linear is not None and last_linear.bias is not None:
            bias = last_linear.bias.view(5, self.N)
            bias[2].fill_(math.atanh(init_b))  # b
            bias[3].fill_(math.atanh(init_c))  # c
            bias[4].fill_(math.atanh(init_d))  # d

    def _compute_params(
        self,
        u_bt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u_bt (B,T,D) -> (a,b,c,d,u_scaled), all (B,T,N), all real.

        Exact local normalization of [[a,b],[c,d]] so that ||.||_2 <= 1.
        """
        B, T, D = u_bt.shape
        assert D == self.D

        u_tilde = self.in_proj(u_bt)                                   # (B,T,N)
        g = self.gamma.to(device=u_bt.device, dtype=u_bt.dtype)
        u_scaled = g * u_tilde                                         # (B,T,N)

        raw = self.param_net(u_bt)                                     # (B,T,5N)
        delta_raw, sign_raw, b_raw, c_raw, d_raw = raw.split(self.N, dim=-1)

        delta = F.softplus(delta_raw + self.delta_bias)                # (B,T,N)
        alpha = F.softplus(self.alpha_log).view(1, 1, self.N)         # (1,1,N)
        rho = torch.exp(-delta * alpha).clamp(max=1.0 - self.eps_a)

        s = torch.tanh(sign_raw + self.sign_bias.view(1, 1, self.N))
        a = rho * s

        if self.bcd_nonlinearity == "tanh":
            b = torch.tanh(b_raw)
            c = torch.tanh(c_raw)
            d = torch.tanh(d_raw)
        else:
            b, c, d = b_raw, c_raw, d_raw

        sigma = spectral_norm_2x2_abcd(a, b, c, d)                    # (B,T,N)
        scale = torch.clamp(sigma, min=1.0)

        a = a / scale
        b = b / scale
        c = c / scale
        d = d / scale

        return a, b, c, d, u_scaled

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        *,
        mode: Literal["scan", "loop"] = "scan",
        return_state: bool = True,
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        """
        Same interface as your old RobustMambaDiagSSM.

        Returns:
          y      : (B,T,D_out)
          z_seq  : (B,T+1,N) if return_state
          z_last : (B,N)     if return_last
        """
        u = _normalize_to_3d(u)
        B, T, D = u.shape
        assert D == self.D, f"Expected d_model={self.D}, got {D}"
        device, dtype = u.device, u.dtype

        z0 = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=B,
            n_state=self.N,
            device=device,
            dtype=dtype,
            x0=self.x0_param,
        )

        a_bt, b_bt, c_bt, d_bt, u_scaled_bt = self._compute_params(u)
        bu_bt = b_bt * u_scaled_bt

        if mode == "loop":
            z = z0
            y_hat = torch.empty(B, T, self.N, device=device, dtype=dtype)
            z_seq = (
                torch.empty(B, T + 1, self.N, device=device, dtype=dtype)
                if return_state else None
            )
            if return_state:
                z_seq[:, 0, :] = z0

            for t in range(T):
                if self.output_uses_post_state:
                    z = a_bt[:, t, :] * z + bu_bt[:, t, :]
                    y_hat[:, t, :] = c_bt[:, t, :] * z + d_bt[:, t, :] * u_scaled_bt[:, t, :]
                    if return_state:
                        z_seq[:, t + 1, :] = z
                else:
                    y_hat[:, t, :] = c_bt[:, t, :] * z + d_bt[:, t, :] * u_scaled_bt[:, t, :]
                    z = a_bt[:, t, :] * z + bu_bt[:, t, :]
                    if return_state:
                        z_seq[:, t + 1, :] = z

            z_last = z

        else:  # scan
            a_tb = a_bt.transpose(0, 1).contiguous()     # (T,B,N)
            bu_tb = bu_bt.transpose(0, 1).contiguous()   # (T,B,N)

            states = _diag_scan(a_tb, bu_tb, z0, scan_impl=self._graphed_scan)  # (T+1,B,N)
            z_last = states[-1]

            if return_state:
                z_seq = states.transpose(0, 1).contiguous()  # (B,T+1,N)
                z_bt = z_seq[:, 1:, :] if self.output_uses_post_state else z_seq[:, :-1, :]
            else:
                z_seq = None
                z_bt = (
                    states[1:].transpose(0, 1).contiguous()
                    if self.output_uses_post_state
                    else states[:-1].transpose(0, 1).contiguous()
                )

            y_hat = c_bt * z_bt + d_bt * u_scaled_bt

        y = self.out_proj(y_hat)
        self.state = z_last.detach() if detach_state else z_last

        if return_state and return_last:
            return y, z_seq, z_last
        if return_state:
            return y, z_seq
        if return_last:
            return y, z_last
        return y

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)


# ----------------------------
# Raven-like selective slot-memory cell with a prescribed ℓ2 gain
# ----------------------------
class L2SelectiveRavenCell(EvalCacheMixin, nn.Module):
    r"""Scan-compatible Raven-style selective SSM cell with a certified ℓ2 gain.

    Interface-wise this is an "LTI cell": it has the same
    ``forward(z, state=None, *, mode=...)`` contract, the same ``loop``/``scan``
    modes, the same streaming-state semantics, and exposes ``.gamma`` so the
    certified :class:`DeepSSM` stack can compose it. Mathematically it is **not**
    LTI: its diagonal coefficients depend on the current input token ``z_t``
    through a top-K router, i.e. it is an input-dependent (selective) diagonal SSM

        S_{t+1} = A(z_t) ⊙ S_t + B(z_t).

    State is key/value slot memory ``S_k: (B,H,M,d_k)``, ``S_v: (B,H,M,d_v)``.
    For each token (all projections are bias-free, which is required for the
    zero-state gain certificate: ``z_t = 0`` must give ``y_t = 0``)::

        q_t = W_q z_t,  k_t = W_k z_t,  v_t = W_v z_t          # per head
        m_t = sigmoid(W_r z_t)                                 # (B,H,M)
        g_t = top_k(m_t, K)                                    # zero outside top-K
        r_t = g_t / (alpha * g_t.sum(-1, keepdim).clamp_min(eps))   # sum_M r_t <= 1/alpha

        lambda_t = rho * exp(a ⊙ r_t),   a = -softplus(a_raw) <= 0  =>  lambda_t <= rho < 1
        omega_t  = r_t ⊙ (1 - lambda_t)                        # r_t = 0  =>  omega_t = 0

        S^{k}_{t+1} = lambda_t ⊙ S^k_t + omega_t ⊙ k_t         # inactive slots decay as rho
        S^{v}_{t+1} = lambda_t ⊙ S^v_t + omega_t ⊙ v_t

    The readout is attention over slots, using the pre-write memory (strictly
    causal, ``S_0 = 0`` so a zero prefix reads out zero)::

        p_t = softmax( (S^k_t · q_t) / sqrt(d_k) )             # over slots
        o_t = sum_m p_{t,m} S^v_t[m]                           # convex combination
        y_t = W_o o_t  (+ D z_t  if use_skip)

    Conservative ℓ2-gain certificate (zero initial state)::

        gamma_layer <= ||W_o||_2 ||W_v||_2 / (alpha (1 - rho)) + ||D||_2.

    It is enforced by spectrally capping ``W_v``, ``W_o`` to ``c_v = c_o =
    sqrt(gamma_mem * alpha * (1 - rho))`` (so ``||W_o|| ||W_v|| <= gamma_mem *
    alpha * (1 - rho)``) and ``D`` to ``gamma_skip``, with ``gamma_mem = gamma -
    gamma_skip``. Keys, queries and the router are unconstrained because the
    softmax weights live in the simplex and never enter the magnitude bound.
    The certificate is rigorous but conservative; the realized gain is typically
    well below ``gamma``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_slots: int,
        key_dim: int,
        value_dim: int,
        top_k: int,
        *,
        gamma: float = 1.0,
        train_gamma: bool = True,
        gamma_skip: float = 0.0,
        alpha: float = 1.0,
        rho_max: float = 0.999,
        eps: float = 1e-6,
        use_skip: bool = False,
        learn_x0: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.H = int(num_heads)
        self.M = int(num_slots)
        self.d_k = int(key_dim)
        self.d_v = int(value_dim)
        self.K = int(top_k)
        if min(self.d_model, self.H, self.M, self.d_k, self.d_v) <= 0:
            raise ValueError("d_model, num_heads, num_slots, key_dim, value_dim must be positive.")
        if not (1 <= self.K <= self.M):
            raise ValueError(f"top_k must be in [1, num_slots={self.M}], got {self.K}.")
        self.alpha = float(alpha)
        if self.alpha <= 0.0 or not math.isfinite(self.alpha):
            raise ValueError(f"alpha must be finite and positive, got {alpha}.")
        self.rho_max = float(rho_max)
        if not (0.0 < self.rho_max < 1.0):
            raise ValueError(f"rho_max must be in (0, 1), got {rho_max}.")
        self.eps = float(eps)
        self.use_skip = bool(use_skip)
        self.gamma_skip = float(gamma_skip)
        if self.gamma_skip < 0.0:
            raise ValueError(f"gamma_skip must be non-negative, got {gamma_skip}.")

        # Total prescribed ℓ2-gain budget (matches the `.gamma`/`log_gamma`
        # convention used by the other certified cells so `_set_cell_gamma` works).
        g0 = torch.tensor(float(gamma)).clamp_min(1e-8)
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        # rho in (0, rho_max) via rho = rho_max * sigmoid(rho_logit).
        self.rho_logit = nn.Parameter(torch.tensor(0.0))
        # Per-(head, slot) decay shaping, a = -softplus(a_raw) <= 0.
        self.a_raw = nn.Parameter(torch.zeros(self.H, self.M))

        # Bias-free projections (bias would break the zero-input -> zero-output
        # property the gain certificate relies on).
        self.W_q = nn.Linear(self.d_model, self.H * self.d_k, bias=False)
        self.W_k = nn.Linear(self.d_model, self.H * self.d_k, bias=False)
        self.W_v = nn.Linear(self.d_model, self.H * self.d_v, bias=False)
        self.W_r = nn.Linear(self.d_model, self.H * self.M, bias=False)
        self.W_o = nn.Linear(self.H * self.d_v, self.d_model, bias=False)
        self.D = nn.Linear(self.d_model, self.d_model, bias=False) if self.use_skip else None

        self.learn_x0 = bool(learn_x0)
        if self.learn_x0:
            self.S_k0 = nn.Parameter(torch.zeros(1, self.H, self.M, self.d_k))
            self.S_v0 = nn.Parameter(torch.zeros(1, self.H, self.M, self.d_v))
        else:
            self.register_buffer("S_k0", None)
            self.register_buffer("S_v0", None)

        self.state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # ---- gain-control parameters -------------------------------------------------
    @property
    def rho(self) -> torch.Tensor:
        return self.rho_max * torch.sigmoid(self.rho_logit)

    @property
    def gamma(self) -> torch.Tensor:
        """Prescribed (enforced) ℓ2-gain upper bound used by the DeepSSM certificate."""
        return self.log_gamma.exp()

    @property
    def decay_shape(self) -> torch.Tensor:
        """``a = -softplus(a_raw) <= 0``; controls per-(head, slot) extra decay."""
        return -F.softplus(self.a_raw)

    def _gain_budget(self, device, dtype):
        """Return ``(c, gamma_skip_eff, rho)`` with ``c_v = c_o = c``.

        Enforces ``gamma_mem + gamma_skip_eff <= gamma`` and ``c^2 = gamma_mem *
        alpha * (1 - rho)`` so the capped weights satisfy the certificate.
        """
        gamma_total = self.gamma.to(device=device, dtype=dtype)
        rho = self.rho.to(device=device, dtype=dtype)
        gamma_skip = torch.as_tensor(self.gamma_skip, device=device, dtype=dtype)
        gamma_skip_eff = torch.minimum(gamma_skip, gamma_total)
        gamma_mem = (gamma_total - gamma_skip_eff).clamp_min(0.0)
        budget = gamma_mem * self.alpha * (1.0 - rho)
        c = torch.sqrt(budget.clamp_min(0.0) + self.eps)
        return c, gamma_skip_eff, rho

    # ---- state plumbing ----------------------------------------------------------
    def _resolve_state(self, state, reset_state, B, device, dtype):
        internal = self.state
        if reset_state:
            internal = (self.S_k0, self.S_v0) if self.learn_x0 else None
        # Explicit state always wins (matches resolve_runtime_state semantics).
        src = state if state is not None else internal
        if src is None:
            Sk0 = torch.zeros(B, self.H, self.M, self.d_k, device=device, dtype=dtype)
            Sv0 = torch.zeros(B, self.H, self.M, self.d_v, device=device, dtype=dtype)
            return Sk0, Sv0
        Sk0, Sv0 = src
        return (
            self._cast_state(Sk0, B, self.d_k, device, dtype),
            self._cast_state(Sv0, B, self.d_v, device, dtype),
        )

    def _cast_state(self, S, B, d, device, dtype):
        if S.dim() == 3:           # (H,M,d) -> add batch
            S = S.unsqueeze(0)
        if S.dim() != 4:
            raise ValueError(f"slot state must have 3 or 4 dims, got shape {tuple(S.shape)}.")
        if S.size(0) == 1 and B > 1:
            S = S.expand(B, -1, -1, -1)
        if S.shape != (B, self.H, self.M, d):
            raise ValueError(
                f"slot state has shape {tuple(S.shape)}, expected {(B, self.H, self.M, d)}."
            )
        return S.to(device=device, dtype=dtype)

    def _router(self, m: torch.Tensor) -> torch.Tensor:
        """Top-K, then normalize so ``sum_M r_t <= 1/alpha`` (the gain-budget input)."""
        if self.K >= self.M:
            g = m
        else:
            top_val, top_idx = torch.topk(m, self.K, dim=-1)
            g = torch.zeros_like(m).scatter(-1, top_idx, top_val)
        denom = self.alpha * g.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return g / denom

    def _scan_memory(self, A, Bx, S0, mode):
        """Run ``S_{t+1} = A_t ⊙ S_t + B_t``; return ``(post, pre)`` states.

        post[t] = S_{t+1} (after writing token t); pre[t] = S_t (before).
        A: (B,T,H,M,1), Bx: (B,T,H,M,d), S0: (B,H,M,d).
        """
        T = A.shape[1]
        if T == 0:
            empty = Bx[:, :0]
            return empty, empty
        if mode == "loop":
            posts = []
            S = S0
            for t in range(T):
                S = A[:, t] * S + Bx[:, t]
                posts.append(S)
            post = torch.stack(posts, dim=1)
        elif mode == "scan":
            Bx = Bx.clone()
            Bx[:, 0] = Bx[:, 0] + A[:, 0] * S0
            # Inclusive affine prefix scan along time (axis=1); same JIT operator as LRU.
            _, post = associative_scan(binary_operator_diag, (A, Bx), axis=1)
        else:
            raise ValueError(f"Unknown mode {mode!r}, expected 'loop' or 'scan'.")
        pre = torch.cat([S0.unsqueeze(1), post[:, :-1]], dim=1)
        return post, pre

    def forward(
        self,
        z: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        mode: Literal["scan", "loop"] = "scan",
        return_state: bool = True,
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        """``z: (B,L,d_model) -> y: (B,L,d_model)``; state is ``(S_k, S_v)``.

        Returns ``(y, (S_k_seq, S_v_seq))`` by default, where each slot-memory
        trajectory has shape ``(B, L, H, M, d)`` and its last time index is the
        streaming state ``(B, H, M, d)`` recovered by ``DeepSSM._last_runtime_state``.
        """
        z = _normalize_to_3d(z)
        B, T, Dm = z.shape
        if Dm != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {Dm}.")
        device, dtype = z.device, z.dtype

        S_k0, S_v0 = self._resolve_state(state, reset_state, B, device, dtype)

        c, gamma_skip_eff, rho = self._gain_budget(device, dtype)
        # SVD-based spectral caps on W_v/W_o (and the optional skip D); cache in
        # eval (fixed weights), recompute in training.
        W_v_eff = self._eval_cached("W_v_eff", lambda: _spectral_cap(self.W_v.weight, c))
        W_o_eff = self._eval_cached("W_o_eff", lambda: _spectral_cap(self.W_o.weight, c))

        q = self.W_q(z).view(B, T, self.H, self.d_k)
        k = self.W_k(z).view(B, T, self.H, self.d_k)
        v = F.linear(z, W_v_eff).view(B, T, self.H, self.d_v)
        m = torch.sigmoid(self.W_r(z)).view(B, T, self.H, self.M)

        r = self._router(m)                                       # (B,T,H,M), sum_M r <= 1/alpha
        a = self.decay_shape.to(device=device, dtype=dtype)       # (H,M) <= 0
        lam = rho * torch.exp(a[None, None] * r)                  # (B,T,H,M) <= rho
        omega = r * (1.0 - lam)                                   # (B,T,H,M), 0 where r == 0

        A = lam.unsqueeze(-1)                                     # (B,T,H,M,1)
        Bk = omega.unsqueeze(-1) * k.unsqueeze(3)                 # (B,T,H,M,d_k)
        Bv = omega.unsqueeze(-1) * v.unsqueeze(3)                 # (B,T,H,M,d_v)

        Sk_post, Sk_pre = self._scan_memory(A, Bk, S_k0, mode)
        Sv_post, Sv_pre = self._scan_memory(A, Bv, S_v0, mode)

        # Attention readout over slots from the pre-write memory.
        scores = (Sk_pre * q.unsqueeze(3)).sum(-1) / math.sqrt(self.d_k)   # (B,T,H,M)
        p = torch.softmax(scores, dim=-1)                                  # (B,T,H,M)
        o = (p.unsqueeze(-1) * Sv_pre).sum(dim=3)                          # (B,T,H,d_v)
        o = o.reshape(B, T, self.H * self.d_v)
        y = F.linear(o, W_o_eff)                                           # (B,T,d_model)
        if self.use_skip:
            y = y + F.linear(z, self._eval_cached("D_eff", lambda: _spectral_cap(self.D.weight, gamma_skip_eff)))

        Sk_last = Sk_post[:, -1] if T > 0 else S_k0
        Sv_last = Sv_post[:, -1] if T > 0 else S_v0
        self.state = (
            (Sk_last.detach(), Sv_last.detach()) if detach_state else (Sk_last, Sv_last)
        )

        state_seq = (Sk_post, Sv_post)
        last_state = (Sk_last, Sv_last)
        if return_state and return_last:
            return y, state_seq, last_state
        if return_state:
            return y, state_seq
        if return_last:
            return y, last_state
        return y, last_state

    def reset(self):
        self.state = None
