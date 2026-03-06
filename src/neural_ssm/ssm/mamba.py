import math
import warnings
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .scan_utils import associative_scan, binary_operator_diag
from .state_utils import (
    resolve_runtime_state as _resolve_runtime_state,
    reset_runtime_state as _reset_runtime_state,
)


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
# Spectral-norm-bounded linear map (exact or power-iteration)
# ----------------------------
class L2BoundedLinearExact(nn.Module):
    """
    Linear map  y = x @ W^T  with  ||W||_2 <= bound  (no bias).

    exact_norm=True  (default):
        SVD-based exact bound via torch.linalg.matrix_norm(ord=2) every forward.
        Hard guarantee at every step; O(min(d_in,d_out)*max(d_in,d_out)) cost.
    exact_norm=False:
        Power iteration (one matmul per iter) with cached singular vectors.
        Approximate: bound holds asymptotically as u,v converge; much faster for
        large dimensions.  power_iters controls how many iterations per forward.
    """

    def __init__(self, d_in: int, d_out: int, *, bound: float = 1.0,
                 exact_norm: bool = True, power_iters: int = 1):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.bound = float(bound)
        self.exact_norm = bool(exact_norm)
        self.power_iters = max(1, int(power_iters))
        self.W_raw = nn.Parameter(0.02 * torch.randn(self.d_out, self.d_in))
        if not exact_norm:
            # Singular vector cached across forward calls; updated in-place (no grad).
            self.register_buffer("_u", F.normalize(torch.randn(self.d_out), dim=0))

    def _sigma_power_iter(self, W: torch.Tensor) -> torch.Tensor:
        u = self._u   # (d_out,)
        for _ in range(self.power_iters):
            v = F.normalize(W.T @ u, dim=0)    # (d_in,)
            u = F.normalize(W @ v, dim=0)       # (d_out,)
        sigma = (u @ (W @ v)).abs()
        with torch.no_grad():
            self._u.copy_(u)
        return sigma.clamp(min=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_raw
        if self.exact_norm:
            sigma = torch.linalg.matrix_norm(W, ord=2).clamp(min=1e-5)
        else:
            sigma = self._sigma_power_iter(W)
        scale = torch.clamp(sigma / self.bound, min=1.0)
        return x @ (W / scale).T


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


# ----------------------------
# Diagonal time-varying recurrence scan
# ----------------------------
def _diag_scan(
    a_tb: torch.Tensor, bu_tb: torch.Tensor, z0: torch.Tensor
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
    Returns states : (T+1, B, N)  — [z_0, z_1, ..., z_T]
    """
    T, B, N = a_tb.shape
    if T == 0:
        return z0.unsqueeze(0)

    # Fold z0 into the first time step so that after the inclusive prefix scan
    # b_prefix[t] == z_{t+1} directly (mirrors _scan_diag_complex in scan_utils).
    bu_tb = bu_tb.clone()
    bu_tb[0] = bu_tb[0] + a_tb[0] * z0

    _, z_next = associative_scan(binary_operator_diag, (a_tb, bu_tb), axis=0)  # (T,B,N)

    states = torch.empty(T + 1, B, N, device=a_tb.device, dtype=a_tb.dtype)
    states[0] = z0
    states[1:] = z_next
    return states


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
    ):
        super().__init__()
        self.D = int(d_model)
        self.N = int(d_state if d_state is not None else d_model)
        self.D_out = int(d_out if d_out is not None else d_model)
        self.state: Optional[torch.Tensor] = None

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

            states = _diag_scan(a_tb, bu_tb, z0)         # (T+1,B,N)
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
        self.state = _reset_runtime_state(self.state)
