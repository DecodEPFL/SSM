import math
import warnings
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import _normalize_to_3d, L2BoundedLinearExact, _diag_scan
from .state_utils import (
    resolve_runtime_state as _resolve_runtime_state,
    reset_runtime_state as _reset_runtime_state,
)


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


class RobustMambaDiagLTI(nn.Module):
    r"""
    Time-varying diagonal LTI core with explicit local l2 certificate.

    Per coordinate:
        z_{t+1} = a_t z_t + b_t (gamma * W_in u_t)
        y_t     = c_t z_t + d_t (gamma * W_in u_t)       [default]

    where [[a_t, b_t], [c_t, d_t]] is normalized so that its spectral norm
    is <= 1 exactly, coordinatewise.

    This is just the TV-LTI core. It is meant to be used inside your SSL block,
    which then adds FF + residual outside.

    Compared to your old RobustMambaDiagSSM:
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
    ):
        super().__init__()
        self.D = int(d_model)
        self.N = int(d_state if d_state is not None else d_model)
        self.D_out = int(d_out if d_out is not None else d_model)
        self.state: Optional[torch.Tensor] = None

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

            states = _diag_scan(a_tb, bu_tb, z0)         # (T+1,B,N)
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
        self.state = _reset_runtime_state(self.state)