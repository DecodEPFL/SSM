import math
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Exact norm-bounded linear (no power iteration)
# ----------------------------
class L2BoundedLinearExact(nn.Module):
    """
    Linear map y = x @ W^T with an *exact* spectral-norm bound: ||W||_2 <= bound.
    Uses torch.linalg.matrix_norm(ord=2) each forward (SVD-based).
    Bias is disabled to preserve the gain interpretation (no output for zero input).
    """
    def __init__(self, d_in: int, d_out: int, *, bound: float = 1.0):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.bound = float(bound)
        # Kaiming-ish small init
        self.W_raw = nn.Parameter(0.02 * torch.randn(self.d_out, self.d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in)
        W = self.W_raw
        sigma = torch.linalg.matrix_norm(W, ord=2)  # scalar
        # scale so ||W||_2 <= bound
        scale = torch.clamp(sigma / self.bound, min=1.0)
        Wn = W / scale
        return x @ Wn.T


# ----------------------------
# Helpers: exact spectral norm of [[a,b],[c,0]] and fast diagonal scan
# ----------------------------
def spectral_norm_2x2_a_b_c(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Exact spectral norm of K = [[a, b],
                               [c, 0]].
    a,b,c broadcastable, e.g. (B,T,N). Returns same shape.
    """
    # K^T K = [[a^2 + c^2, a b],
    #          [a b,       b^2]]
    p = a * a + c * c
    r = b * b
    q = a * b
    disc = (p - r) * (p - r) + 4.0 * q * q
    lam_max = 0.5 * (p + r + torch.sqrt(disc + eps))
    return torch.sqrt(lam_max + eps)


def parallel_scan_diag_affine(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inclusive prefix scan for diagonal affine pairs (a_t, b_t) representing:
        x -> a_t ⊙ x + b_t
    a,b: (T,B,N)
    """
    T = a.shape[0]
    if T == 0:
        return a, b

    a_p = a.clone()
    b_p = b.clone()

    offset = 1
    while offset < T:
        a_left = a_p[offset:]     # (T-offset,B,N)
        a_right = a_p[:-offset]   # (T-offset,B,N)

        b_right = b_p[:-offset]
        b_left = b_p[offset:]

        new_a_tail = a_left * a_right
        new_b_tail = a_left * b_right + b_left

        a_p = torch.cat([a_p[:offset], new_a_tail], dim=0)
        b_p = torch.cat([b_p[:offset], new_b_tail], dim=0)
        offset <<= 1

    return a_p, b_p


def diag_recurrence_scan(a_tb: torch.Tensor, bu_tb: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
    """
    Solve z_{t+1} = a_t ⊙ z_t + bu_t using scan.
    a_tb, bu_tb: (T,B,N), z0: (B,N)
    returns states: (T+1,B,N)
    """
    T, B, N = a_tb.shape
    a_p, b_p = parallel_scan_diag_affine(a_tb, bu_tb)  # (T,B,N)

    z0_exp = z0.unsqueeze(0).expand(T, -1, -1)         # (T,B,N)
    z_next = a_p * z0_exp + b_p                        # (T,B,N)

    states = torch.empty(T + 1, B, N, device=a_tb.device, dtype=a_tb.dtype)
    states[0] = z0
    states[1:] = z_next
    return states


# ----------------------------
# Robust Mamba-style selective diagonal SSM (D=0), with norm-bounded projections
# ----------------------------
class RobustMambaDiagSSM(nn.Module):
    r"""
    Discrete-time selective diagonal SSM (Mamba-like), robustly ℓ2-bounded.

    Input:  u_t ∈ R^D (you use (B,T,D))
    Projections (bounded):
        ũ_t = W_in u_t,   ||W_in||_2 <= 1
        y_t  = W_out ŷ_t,  ||W_out||_2 <= 1

    Selective params from u_t:
        (delta_t, b_t, c_t) = ParamNet(u_t)
        a_t = exp( - softplus(delta_t + bias) ⊙ softplus(alpha) )  in (0,1)

    State/output (D=0):
        z_{t+1} = a_t ⊙ z_t + b_t ⊙ (γ ũ_t)
        ŷ_t     = c_t ⊙ z_t
        y_t     = W_out ŷ_t

    Robustness (exact per-coordinate contraction):
        For each (t,i), scale (a,b,c) so || [[a,b],[c,0]] ||_2 <= 1 (exact closed form).
    This implies ||y||_{ℓ2} <= γ ||u||_{ℓ2} for z0=0 (since projections are non-expansive).

    Notes:
      - Projections are *linear without bias* to preserve the gain interpretation.
      - ParamNet can be "linear" or "mlp".
      - mode="scan" uses the diagonal affine scan (fast elementwise ops).
    """

    def __init__(
        self,
        d_model: int,                 # D
        d_state: Optional[int] = None,# N (often = D)
        d_out: Optional[int] = None,  # output dimension
        *,
        gamma: float = 1.0,
        train_gamma: bool = True,
        eps_a: float = 1e-4,          # keep a_t <= 1 - eps_a
        param_net: Literal["linear", "mlp"] = "linear",
        hidden: int = 128,
        # init near stability boundary
        init_rho: float = 0.99,       # a ≈ rho initially
        init_delta0: float = 1.0,     # typical delta at init
        init_param_scale: float = 0.02,
        # keep b,c bounded before normalization (helps avoid shrinking too much)
        bc_nonlinearity: Literal["tanh", "identity"] = "tanh",
        # projection bounds (<=1 keeps overall bound = gamma)
        proj_bound: float = 1.0,
    ):
        super().__init__()
        self.D = int(d_model)
        self.N = int(d_state if d_state is not None else d_model)
        self.D_out = int(d_out if d_out is not None else d_model)

        # assert self.N == self.D, (
        #     "To keep the clean overall bound with proj_bound=1 and avoid extra bookkeeping, "
        #     "set d_state == d_model. If you really want N!=D, it still works, but you're "
        #     "mixing dimensions via projections; the bound stays <= gamma as long as "
        #     "||W_in||,||W_out|| <= 1."
        # )

        # gamma (>0)
        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        self.eps_a = float(eps_a)
        self.bc_nonlinearity = bc_nonlinearity

        # Non-expansive projections (exact, SVD-based each forward)
        self.in_proj = L2BoundedLinearExact(self.D, self.N, bound=proj_bound)
        self.out_proj = L2BoundedLinearExact(self.N, self.D_out, bound=proj_bound)

        # alpha > 0 controls base decay; a_t = exp(-delta_t * alpha)
        self.alpha_log = nn.Parameter(torch.zeros(self.N))  # alpha = softplus(alpha_log)

        # ParamNet: u_t -> (delta_raw, b_raw, c_raw) in R^{3N}
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

        # bias for delta (learned)
        self.delta_bias = nn.Parameter(torch.zeros(self.N))

        # init
        self.reset_parameters(init_rho=init_rho, init_delta0=init_delta0, init_param_scale=init_param_scale)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    @torch.no_grad()
    def reset_parameters(self, *, init_rho: float, init_delta0: float, init_param_scale: float):
        # Initialize alpha so that exp(-delta0 * alpha) ≈ rho
        rho = float(init_rho)
        rho = min(max(rho, 1e-4), 1 - 1e-6)
        delta0 = float(init_delta0)
        delta0 = max(delta0, 1e-4)

        alpha0 = (-math.log(rho)) / delta0  # scalar target
        # softplus^{-1}(alpha0)
        self.alpha_log.fill_(math.log(math.expm1(alpha0)) if alpha0 > 1e-6 else math.log(alpha0 + 1e-6))

        # delta_bias so softplus(delta_raw + delta_bias) ≈ delta0 when delta_raw≈0
        self.delta_bias.fill_(math.log(math.expm1(delta0)) if delta0 > 1e-6 else math.log(delta0 + 1e-6))

        # Small init for param_net weights so b,c start small
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_param_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_params(self, u_bt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given u (B,T,D), return:
          a_bt: (B,T,N)  diagonal A_t coefficients
          b_bt: (B,T,N)
          c_bt: (B,T,N)
          u_scaled_bt: (B,T,N) = gamma * (W_in u)
        with per-(t,i) exact normalization so ||[[a,b],[c,0]]||_2 <= 1.
        """
        B, T, D = u_bt.shape
        assert D == self.D

        # ũ = W_in u  (non-expansive)
        u_tilde = self.in_proj(u_bt)  # (B,T,N)

        # gamma scaling
        g = self.gamma.to(device=u_bt.device, dtype=u_bt.dtype)
        u_scaled = g * u_tilde        # (B,T,N)

        # raw params from original u (Mamba-style: params depend on input)
        raw = self.param_net(u_bt)                # (B,T,3N)
        delta_raw, b_raw, c_raw = raw.split(self.N, dim=-1)

        # delta >= 0
        delta = F.softplus(delta_raw + self.delta_bias)  # (B,T,N)

        # alpha > 0
        alpha = F.softplus(self.alpha_log).view(1, 1, self.N)  # (1,1,N)

        # diagonal a in (0,1), close to 1 if delta small
        a = torch.exp(-delta * alpha).clamp(max=1.0 - self.eps_a)  # (B,T,N)

        # b,c shaping before normalization
        if self.bc_nonlinearity == "tanh":
            b = torch.tanh(b_raw)
            c = torch.tanh(c_raw)
        elif self.bc_nonlinearity == "identity":
            b = b_raw
            c = c_raw
        else:
            raise ValueError(self.bc_nonlinearity)

        # Exact per-coordinate contraction: scale so ||[[a,b],[c,0]]||_2 <= 1
        sigma = spectral_norm_2x2_a_b_c(a, b, c)   # (B,T,N)
        scale = torch.clamp(sigma, min=1.0)
        a = a / scale
        b = b / scale
        c = c / scale

        return a, b, c, u_scaled

    def forward(
            self,
            u: torch.Tensor,  # (B,T,D) = (u_0,...,u_{T-1})
            state: Optional[torch.Tensor] = None,  # z0: (B,N) or (N,)
            *,
            mode: Literal["scan", "loop"] = "scan",
            return_state: bool = True,
            return_last: bool = False,
    ):
        """
        Returns:
          y:      (B,T,D_out)                 [y_0..y_{T-1}]
          z_seq:  (B,T+1,N) if return_state   [z_0..z_T]
          z_last: (B,N)    if return_last    [z_T]
        """
        assert u.dim() == 3, "Expected u of shape (B,T,D)"
        B, T, D = u.shape
        assert D == self.D
        device, dtype = u.device, u.dtype

        # init z0
        if state is None:
            z0 = torch.zeros(B, self.N, device=device, dtype=dtype)
        else:
            z0 = state.unsqueeze(0).expand(B, -1) if state.dim() == 1 else state
            z0 = z0.to(device=device, dtype=dtype)

        a_bt, b_bt, c_bt, u_scaled_bt = self._compute_params(u)  # all (B,T,N)
        bu_bt = b_bt * u_scaled_bt  # (B,T,N)

        if mode == "loop":
            z = z0
            y_hat = torch.empty(B, T, self.N, device=device, dtype=dtype)

            z_seq = torch.empty(B, T + 1, self.N, device=device, dtype=dtype) if return_state else None
            if return_state:
                z_seq[:, 0, :] = z0

            for t in range(T):
                # y_t uses z_t
                y_hat[:, t, :] = c_bt[:, t, :] * z
                # update to z_{t+1}
                z = a_bt[:, t, :] * z + bu_bt[:, t, :]

                if return_state:
                    z_seq[:, t + 1, :] = z  # store z_{t+1}

            z_last = z  # z_T

        else:
            # scan needs (T,B,N)
            a_tb = a_bt.transpose(0, 1).contiguous()  # (T,B,N)
            bu_tb = bu_bt.transpose(0, 1).contiguous()  # (T,B,N)

            # states: (T+1,B,N) = z_0..z_T
            states = diag_recurrence_scan(a_tb, bu_tb, z0)

            z_last = states[-1]  # (B,N) = z_T

            if return_state:
                z_seq = states.transpose(0, 1).contiguous()  # (B,T+1,N)
                z_bt = z_seq[:, :-1, :]  # (B,T,N) = z_0..z_{T-1}
            else:
                z_seq = None
                z_bt = states[:-1].transpose(0, 1).contiguous()

            # y_t uses z_t
            y_hat = c_bt * z_bt  # (B,T,N)

        # bounded output projection
        y = self.out_proj(y_hat)  # (B,T,D_out)

        if return_state and return_last:
            return y, z_seq, z_last
        if return_state:
            return y, z_seq
        if return_last:
            return y, z_last
        return y

