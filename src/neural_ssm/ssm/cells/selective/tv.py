"""Selective diagonal cell without a direct term (param="tv")."""

from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....static_layers.lipschitz_mlps import L2BoundedLinearExact
from ....utils.runtime import (
    normalize_to_3d as _normalize_to_3d,
    reset_runtime_state as _reset_runtime_state,
    resolve_runtime_state as _resolve_runtime_state,
    sequence_output,
)
from ....utils.scan import GraphedDiagScan
from ..common import (
    _diag_scan,
    _selector_input,
    build_selector,
    initialize_selector_decay,
    spectral_norm_2x2_a_b_c,
)


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
        d_state: int | None = None,
        d_out: int | None = None,
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
        context_dim: int = 0,
        select_input: Literal["both", "context"] = "both",
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
        self.state: torch.Tensor | None = None
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

        self.in_proj = L2BoundedLinearExact(
            self.D, self.N, bound=proj_bound, exact_norm=exact_norm, power_iters=power_iters
        )
        self.out_proj = L2BoundedLinearExact(
            self.N, self.D_out, bound=proj_bound, exact_norm=exact_norm, power_iters=power_iters
        )

        # alpha > 0 controls base decay: a_t = exp(-delta_t * alpha)
        self.alpha_log = nn.Parameter(torch.zeros(self.N))

        self.context_dim = int(context_dim)
        self.supports_select_context = True
        self.select_input = str(select_input)
        self.param_net = build_selector(
            self.D,
            self.context_dim,
            3 * self.N,
            select_input=self.select_input,
            kind=param_net,
            hidden=hidden,
        )

        self.delta_bias = nn.Parameter(torch.zeros(self.N))

        self.reset_parameters(
            init_rho=init_rho, init_delta0=init_delta0, init_param_scale=init_param_scale
        )

        # Learnable initial condition (real, shape (1, N))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, self.N))
        else:
            self.register_buffer("x0_param", None)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    @torch.no_grad()
    def reset_parameters(self, *, init_rho: float, init_delta0: float, init_param_scale: float):
        initialize_selector_decay(
            self.alpha_log,
            self.delta_bias,
            self.param_net,
            rho=init_rho,
            delta0=init_delta0,
            scale=init_param_scale,
        )

    def _compute_params(
        self, u_bt: torch.Tensor, ctx_bt: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u_bt (B,T,D) → (a_bt, b_bt, c_bt, u_scaled_bt), all (B,T,N).

        a ∈ (0, 1-eps_a), b and c normalised so ||[[a,b],[c,0]]||_2 ≤ 1.
        u_scaled = γ · W_in u.
        """
        u_tilde = self.in_proj(u_bt)  # (B,T,N)
        g = self.gamma.to(device=u_bt.device, dtype=u_bt.dtype)
        u_scaled = g * u_tilde  # (B,T,N)

        raw = self.param_net(
            _selector_input(self.context_dim, u_bt, ctx_bt, self.select_input)
        )  # (B,T,3N)
        delta_raw, b_raw, c_raw = raw.split(self.N, dim=-1)

        delta = F.softplus(delta_raw + self.delta_bias)  # (B,T,N)
        alpha = F.softplus(self.alpha_log).view(1, 1, self.N)  # (1,1,N)
        a = torch.exp(-delta * alpha).clamp(max=1.0 - self.eps_a)  # (B,T,N)

        if self.bc_nonlinearity == "tanh":
            b, c = torch.tanh(b_raw), torch.tanh(c_raw)
        else:  # "identity"
            b, c = b_raw, c_raw

        sigma = spectral_norm_2x2_a_b_c(a, b, c)  # (B,T,N)
        scale = torch.clamp(sigma, min=1.0)
        a, b, c = a / scale, b / scale, c / scale

        return a, b, c, u_scaled

    def forward(
        self,
        u: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        mode: Literal["scan", "loop"] = "scan",
        return_state: bool = True,
        return_last: bool = False,
        reset_state: bool = True,
        detach_state: bool = True,
        select_context: torch.Tensor | None = None,
    ):
        """
        u : (B,T,D) or (T,D) or (D,) — normalized to (B,T,D) internally.

        Returns:
          y      : (B,T,D_out)
          z_seq  : (B,T+1,N)   if return_state   [z_0..z_T]
          z_last : (B,N)       if return_last     [z_T]
        """
        if mode not in ("scan", "loop"):
            raise ValueError("mode must be 'scan' or 'loop'.")
        u = _normalize_to_3d(u)
        B, T, D = u.shape
        if D != self.D:
            raise ValueError(f"Expected d_model={self.D}, got {D}.")
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

        a_bt, b_bt, c_bt, u_scaled_bt = self._compute_params(u, select_context)  # (B,T,N)
        bu_bt = b_bt * u_scaled_bt  # (B,T,N)

        if mode == "loop":
            z = z0
            y_hat = torch.empty(B, T, self.N, device=device, dtype=dtype)
            z_seq = (
                torch.empty(B, T + 1, self.N, device=device, dtype=dtype) if return_state else None
            )
            if return_state:
                z_seq[:, 0, :] = z0

            for t in range(T):
                y_hat[:, t, :] = c_bt[:, t, :] * z
                z = a_bt[:, t, :] * z + bu_bt[:, t, :]
                if return_state:
                    z_seq[:, t + 1, :] = z

            z_last = z

        else:  # scan
            a_tb = a_bt.transpose(0, 1).contiguous()  # (T,B,N)
            bu_tb = bu_bt.transpose(0, 1).contiguous()  # (T,B,N)

            states = _diag_scan(a_tb, bu_tb, z0, scan_impl=self._graphed_scan)  # (T+1,B,N)
            z_last = states[-1]  # (B,N)

            if return_state:
                z_seq = states.transpose(0, 1).contiguous()  # (B,T+1,N)
                z_bt = z_seq[:, :-1, :]  # (B,T,N)
            else:
                z_seq = None
                z_bt = states[:-1].transpose(0, 1).contiguous()

            y_hat = c_bt * z_bt  # (B,T,N)

        y = self.out_proj(y_hat)  # (B,T,D_out)
        self.state = z_last.detach() if detach_state else z_last

        return sequence_output(y, z_seq, z_last, return_state=return_state, return_last=return_last)

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)
