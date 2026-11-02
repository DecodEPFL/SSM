"""Selective diagonal cell with a direct term (param="tvc").

The original class name is retained for compatibility; its default dynamics
are time-varying and input-selective."""

from __future__ import annotations

import math
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
    spectral_norm_2x2_abcd,
)


class RobustMambaDiagLTI(nn.Module):
    r"""
    Selective diagonal state-space core with explicit local l2 certificate.

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
        d_state: int | None = None,
        d_out: int | None = None,
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
        context_dim: int = 0,
        select_input: Literal["both", "context"] = "both",
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
        self.state: torch.Tensor | None = None
        # CUDA-graph cache for the diagonal scan (no-op when disabled / on CPU).
        self._graphed_scan = GraphedDiagScan(enabled=use_cuda_graph)

        self.eps_a = float(eps_a)
        self.bcd_nonlinearity = bcd_nonlinearity
        self.output_uses_post_state = bool(output_uses_post_state)

        if output_uses_post_state:
            warnings.warn(
                "RobustMambaDiagLTI: output_uses_post_state=True VOIDS the l2-gain "
                "certificate. Normalizing [[a,b],[c,d]] does not bound the realized "
                "matrix [[a,b],[c*a,c*b+d]] of the post-state output path, so gamma is "
                "not a gain bound here and gain_bound() reports inf. Use it only for "
                "uncertified experiments.",
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
            self.D,
            self.N,
            bound=proj_bound,
            exact_norm=exact_norm,
            power_iters=power_iters,
        )
        self.out_proj = L2BoundedLinearExact(
            self.N,
            self.D_out,
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
        self.context_dim = int(context_dim)
        self.supports_select_context = True
        self.select_input = str(select_input)
        self.param_net = build_selector(
            self.D,
            self.context_dim,
            5 * self.N,
            select_input=self.select_input,
            kind=param_net,
            hidden=hidden,
        )

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
            self.register_buffer("x0_param", None)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    def gain_bound(self) -> torch.Tensor:
        """The cell's zero-state l2-gain contract, or ``inf`` when it has none.

        Normalizing ``[[a, b], [c, d]]`` certifies the map that reads the output
        from the *pre-update* state. With ``output_uses_post_state=True`` the
        realized matrix is ``[[a, b], [c*a, c*b + d]]``, whose norm is not
        controlled by normalizing the original, so ``gamma`` is not a bound on
        that path. Reporting ``inf`` makes any stack built on this cell surface an
        infinite certificate rather than a number the forward map does not obey.
        """
        if self.output_uses_post_state:
            return torch.full_like(self.log_gamma, float("inf"))
        return self.gamma

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
        init_sign = max(min(float(init_sign), 1.0 - 1e-6), -1.0 + 1e-6)
        init_b = max(min(float(init_b), 0.99), -0.99)
        init_c = max(min(float(init_c), 0.99), -0.99)
        init_d = max(min(float(init_d), 0.99), -0.99)
        self.sign_bias.fill_(0.5 * math.log((1 + init_sign) / (1 - init_sign)))
        initialize_selector_decay(
            self.alpha_log,
            self.delta_bias,
            self.param_net,
            rho=init_rho,
            delta0=init_delta0,
            scale=init_param_scale,
        )

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
        ctx_bt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        u_bt (B,T,D) -> (a,b,c,d,u_scaled), all (B,T,N), all real.

        Exact local normalization of [[a,b],[c,d]] so that ||.||_2 <= 1.
        """
        u_tilde = self.in_proj(u_bt)  # (B,T,N)
        g = self.gamma.to(device=u_bt.device, dtype=u_bt.dtype)
        u_scaled = g * u_tilde  # (B,T,N)

        raw = self.param_net(
            _selector_input(self.context_dim, u_bt, ctx_bt, self.select_input)
        )  # (B,T,5N)
        delta_raw, sign_raw, b_raw, c_raw, d_raw = raw.split(self.N, dim=-1)

        delta = F.softplus(delta_raw + self.delta_bias)  # (B,T,N)
        alpha = F.softplus(self.alpha_log).view(1, 1, self.N)  # (1,1,N)
        rho = torch.exp(-delta * alpha).clamp(max=1.0 - self.eps_a)

        s = torch.tanh(sign_raw + self.sign_bias.view(1, 1, self.N))
        a = rho * s

        if self.bcd_nonlinearity == "tanh":
            b = torch.tanh(b_raw)
            c = torch.tanh(c_raw)
            d = torch.tanh(d_raw)
        else:
            b, c, d = b_raw, c_raw, d_raw

        sigma = spectral_norm_2x2_abcd(a, b, c, d)  # (B,T,N)
        scale = torch.clamp(sigma, min=1.0)

        a = a / scale
        b = b / scale
        c = c / scale
        d = d / scale

        return a, b, c, d, u_scaled

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
        Evaluate the recurrence with a loop or a parallel diagonal scan.

        Returns:
          y      : (B,T,D_out)
          z_seq  : (B,T+1,N) if return_state
          z_last : (B,N)     if return_last
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

        a_bt, b_bt, c_bt, d_bt, u_scaled_bt = self._compute_params(u, select_context)
        bu_bt = b_bt * u_scaled_bt

        if mode == "loop":
            z = z0
            y_hat = torch.empty(B, T, self.N, device=device, dtype=dtype)
            z_seq = (
                torch.empty(B, T + 1, self.N, device=device, dtype=dtype) if return_state else None
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
            a_tb = a_bt.transpose(0, 1).contiguous()  # (T,B,N)
            bu_tb = bu_bt.transpose(0, 1).contiguous()  # (T,B,N)

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

        return sequence_output(y, z_seq, z_last, return_state=return_state, return_last=return_last)

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)
