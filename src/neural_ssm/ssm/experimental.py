import math
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertSelectiveTimeVaryingSSM(nn.Module):
    r"""
    Time-varying / selective model in (potentially trainable) storage coordinates.

    We maintain the *x*-state (in R^{d_state}) but enforce the ℓ2-gain certificate
    in z-coordinates defined by z = S x, with P = S^T S.

    For each time t:
        w_t = [ z_t ; gamma * u_t ]          in R^{d_state + d_in}
        v_t = [ z_{t+1} ; y_t ]             in R^{d_state + d_out}
        v_t = K_t w_t

    Selectivity via expert mixture:
        K_t = sum_{m=1}^M pi_{t,m} K^(m),   pi_t = softmax(gate(xi_t))

    Contractivity (exact, per-expert, no power iteration):
        K^(m) = K_raw^(m) / max(1, ||K_raw^(m)||_2)
    so each ||K^(m)||_2 <= 1, hence each ||K_t||_2 <= 1 (convexity).

    Notes:
    - This module outputs y_t from the selective core only.
    - Optional D=0: forces K22=0 before normalization.
    """

    def __init__(
        self,
        d_state: int,
        d_in: int,
        d_out: int,
        n_experts: int = 8,
        gate: Literal["linear", "mlp"] = "mlp",
        gate_hidden: int = 64,
        gate_on: Literal["u", "ux", "uz"] = "u",
        *,
        gamma_init: float = 1.0,
        train_gamma: bool = True,
        D_zero: bool = True,
        gate_temperature: float = 1.0,
        S_trainable: bool = True,
        S_init: Literal["identity", "random_cholesky"] = "identity",
        S_diag_eps: float = 1e-3,
    ):
        super().__init__()
        self.d_state = int(d_state)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.n_experts = int(n_experts)
        self.gate_on = gate_on
        self.D_zero = bool(D_zero)
        self.gate_temperature = float(gate_temperature)

        ds, du, dy, M = self.d_state, self.d_in, self.d_out, self.n_experts
        m = ds + dy  # rows of K
        n = ds + du  # cols of K

        # Raw experts (unconstrained); we will exact-normalize to ||K||_2 <= 1
        self.K_raw = nn.Parameter(0.02 * torch.randn(M, m, n))

        # Trainable gamma (positive)
        log_g = math.log(max(gamma_init, 1e-8))
        if train_gamma:
            self.log_gamma = nn.Parameter(torch.tensor(log_g, dtype=torch.float32))
        else:
            self.register_buffer("log_gamma", torch.tensor(log_g, dtype=torch.float32))

        # Trainable S via (lower-triangular) Cholesky-like factor with positive diagonal.
        # This guarantees invertibility and P = S^T S ≻ 0.
        self.S_diag_eps = float(S_diag_eps)
        if S_init == "identity":
            S0 = torch.eye(ds)
        elif S_init == "random_cholesky":
            A = torch.randn(ds, ds) / math.sqrt(ds)
            P0 = A @ A.T + (1.0 * torch.eye(ds))
            S0 = torch.linalg.cholesky(P0)
        else:
            raise ValueError(f"Unknown S_init: {S_init}")

        # Store an unconstrained raw matrix, we will project to lower-triangular + positive diag
        self.S_raw = nn.Parameter(S0.clone()) if S_trainable else nn.Parameter(S0.clone(), requires_grad=False)

        # Gate network (very simple)
        gate_in_dim = {
            "u": du,
            "ux": du + ds,
            "uz": du + ds,
        }[gate_on]

        if gate == "linear":
            self.gate_net = nn.Linear(gate_in_dim, M)
        elif gate == "mlp":
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, M),
            )
        else:
            raise ValueError(f"Unknown gate type: {gate}")

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    def _build_S(self) -> torch.Tensor:
        """
        S = tril(S_raw) with positive diagonal via softplus.
        """
        S = torch.tril(self.S_raw)
        diag = torch.diagonal(S, 0)
        diag_pos = F.softplus(diag) + self.S_diag_eps
        S = S.clone()
        S.diagonal(0).copy_(diag_pos)
        return S

    def _contractive_experts_exact(self) -> torch.Tensor:
        """
        Returns K_experts with shape (M, ds+dy, ds+du) where each expert has ||K||_2 <= 1
        using exact spectral norm via torch.linalg.matrix_norm(ord=2).
        """
        K = self.K_raw

        if self.D_zero:
            # K block partition:
            # rows: [0:ds]=z_{t+1}, [ds:ds+dy]=y_t
            # cols: [0:ds]=z_t,     [ds:ds+du]=gamma*u_t
            ds, du, dy = self.d_state, self.d_in, self.d_out
            K = K.clone()
            K[:, ds:ds + dy, ds:ds + du] = 0.0  # K22 = 0  => D=0 in the induced A,B,C,D mapping

        # Exact spectral norm per expert (SVD-based internally)
        norms = torch.linalg.matrix_norm(K, ord=2, dim=(-2, -1))  # (M,)
        scale = torch.clamp(norms, min=1.0)
        Kc = K / scale[:, None, None]
        return Kc

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        *,
        time_first: bool = False,     # u is (T,B,du) if True else (B,T,du)
        return_state: bool = True,
        return_z: bool = False,       # if True and return_state, returns z_seq instead of x_seq
        mode: str = "scan",
    ):
        """
        Args:
            u:  (B,T,du) or (T,B,du)
            state: (B,ds) or (ds,) or None (zeros)
        Returns:
            y_seq: (B,T,dy) or (T,B,dy)
            x_last: (B,ds)
            (optional) seq of states: x_seq or z_seq with matching time_first
        """
        if time_first:
            u_bt = u.transpose(0, 1)  # (B,T,du)
        else:
            u_bt = u

        B, T, du = u_bt.shape
        assert du == self.d_in, f"u last dim must be d_in={self.d_in}"

        ds, dy = self.d_state, self.d_out
        device, dtype = u_bt.device, u_bt.dtype

        # Initial state x
        if state is None:
            x = torch.zeros(B, ds, device=device, dtype=dtype)
        else:
            if state.dim() == 1:
                assert state.shape[0] == ds
                x = state.unsqueeze(0).expand(B, -1).to(device=device, dtype=dtype).contiguous()
            else:
                assert state.shape == (B, ds)
                x = state.to(device=device, dtype=dtype)

        # Build S and experts
        S = self._build_S().to(device=device, dtype=dtype)                  # (ds,ds), lower-triangular
        Kexp = self._contractive_experts_exact().to(device=device, dtype=dtype)  # (M, ds+dy, ds+du)
        M = Kexp.shape[0]

        g = self.gamma.to(device=device, dtype=dtype)

        y_seq = torch.empty(B, T, dy, device=device, dtype=dtype)
        state_seq = torch.empty(B, T, ds, device=device, dtype=dtype) if return_state else None

        # Precompute for triangular solves: we will compute x_{t+1} from z_{t+1} via S x = z
        # torch.linalg.solve_triangular expects RHS shape (ds, B) or (B, ds) depending; we use transpose.

        for t in range(T):
            u_t = u_bt[:, t, :]  # (B,du)

            # z_t = S x_t  (row-vector convention: z = x @ S^T)
            z = x @ S.T  # (B,ds)

            # Gate input xi_t
            if self.gate_on == "u":
                xi = u_t
            elif self.gate_on == "ux":
                xi = torch.cat([u_t, x], dim=-1)
            elif self.gate_on == "uz":
                xi = torch.cat([u_t, z], dim=-1)
            else:
                raise ValueError(f"Unknown gate_on: {self.gate_on}")

            logits = self.gate_net(xi) / self.gate_temperature  # (B,M)
            pi = F.softmax(logits, dim=-1)                      # (B,M), convex weights

            # w_t = [z_t ; gamma u_t]
            w = torch.cat([z, g * u_t], dim=-1)                 # (B, ds+du)

            # Apply all experts: v_all[b,m,:] = Kexp[m] @ w[b]
            # Kexp: (M, mrows, ncols); w: (B, ncols) -> v_all: (B, M, mrows)
            v_all = torch.einsum("bn,man->bma", w, Kexp)        # (B,M, ds+dy)

            # Mix: v = Σ_m pi_{b,m} v_all[b,m,:]
            v = torch.einsum("bm,bma->ba", pi, v_all)           # (B, ds+dy)

            z_next = v[:, :ds]
            y_t = v[:, ds:ds + dy]

            y_seq[:, t, :] = y_t
            if return_state:
                state_seq[:, t, :] = (z if return_z else x)

            # Recover x_{t+1} from z_{t+1}:  S x_{t+1} = z_{t+1}
            # Solve lower-triangular system for each batch: x = S^{-1} z
            x = torch.linalg.solve_triangular(S, z_next.T, upper=False).T  # (B,ds)

        x_last = x

        if time_first:
            y_seq = y_seq.transpose(0, 1)  # (T,B,dy)
            if return_state:
                state_seq = state_seq.transpose(0, 1)  # (T,B,ds)

        if return_state:
            return y_seq, state_seq
        return y_seq, state_seq
