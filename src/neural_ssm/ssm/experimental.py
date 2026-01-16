import math
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .scan_utils import compute_linear_recurrence_parallel_block2x2

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






class Block2x2SelectiveBCDExpertsL2SSM(nn.Module):
    r"""
    Scan-friendly selective model:

        z_{t+1} = A_z z_t + B_z(u_t) u_t
        y_t     = C_z(u_t) z_t + D_z(u_t) u_t

    with:
      - A_z constant, block-diagonal in 2x2 blocks (complex eigenvalues),
      - (B,C,D) input-dependent via convex mixture of M experts,
      - L2 gain certificate via per-step contraction of K_t:
            [ z_{t+1} ]   [ K11   K12(u_t) ] [ z_t     ]
            [   y_t   ] = [ K21(u_t) K22(u_t)] [ gamma u_t ]

        K_t = sum_m pi_{t,m} K^(m),  pi_t = softmax(gate(u_t)),
        and we enforce ||K^(m)||_2 <= 1 for all m by *global* exact normalization:
            scale = max(1, max_m ||K_raw^(m)||_2),   K^(m)=K_raw^(m)/scale.

    Because K11 is shared across experts and we use a *single global* scale, A_z=K11/scale
    remains constant across time and scan-friendly.

    IMPORTANT for your scan:
      Your scan function expects x_{t+1} = A x_t + B u_t with constant B.
      Here B is time-varying, but we can precompute v_t := B_t u_t ∈ R^{d_state}
      and run scan on x_{t+1}=A x_t + v_t by calling your scan with:
          B_eff = I_{d_state},   u_eff[t] = v_t
      (so the internal v = B_eff @ u_eff equals u_eff).
    """

    def __init__(
        self,
        d_state: int,
        d_input: int,
        d_output: int,
        *,
        n_experts: int = 8,
        gamma: float = 1.0,
        train_gamma: bool = False,
        # A eigenvalues modulus <= 1 - eps_radius
        eps_radius: float = 1e-3,
        # exact global spectral normalization
        exact_norm: bool = True,
        # gating
        gate: Literal["linear", "mlp"] = "linear",
        gate_hidden: int = 64,
        gate_temperature: float = 1.0,
        # init A on circle
        init_rho: float = 0.99,
        init_max_phase: Optional[float] = 0.2,  # radians; None => Uniform(-pi,pi)
        phase_center: float = 0.0,
        same_phase_across_blocks: bool = False,
        # init scale for offdiagonal blocks
        offdiag_scale: float = 0.02,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even (2x2 blocks)."

        self.d_state = int(d_state)
        self.d_input = int(d_input)
        self.d_output = int(d_output)
        self.n_experts = int(n_experts)

        self.eps_radius = float(eps_radius)
        self.exact_norm = bool(exact_norm)
        self.gate_temperature = float(gate_temperature)

        n_pairs = self.d_state // 2

        # Optional change of basis if you later want dense x-basis matrices
        self.S = nn.Parameter(0.1 * torch.randn(self.d_state, self.d_state))

        # A_z parameters: 2x2 blocks rho_i R(theta_i)
        self.rho_raw = nn.Parameter(torch.zeros(n_pairs))
        self.theta = nn.Parameter(torch.zeros(n_pairs))

        # Expert raw blocks (only B,C,D parts => K12,K21,K22)
        self.K12_raw = nn.Parameter(offdiag_scale * torch.randn(self.n_experts, self.d_state, self.d_input))
        self.K21_raw = nn.Parameter(offdiag_scale * torch.randn(self.n_experts, self.d_output, self.d_state))
        self.K22_raw = nn.Parameter(offdiag_scale * torch.randn(self.n_experts, self.d_output, self.d_input))

        # gamma (>0)
        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        # Gate network: u_t -> logits in R^M
        if gate == "linear":
            self.gate_net = nn.Linear(self.d_input, self.n_experts)
        elif gate == "mlp":
            self.gate_net = nn.Sequential(
                nn.Linear(self.d_input, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.n_experts),
            )
        else:
            raise ValueError(f"Unknown gate: {gate}")

        # Init A near stability boundary + small phases
        self.init_on_circle(
            rho=init_rho,
            max_phase=init_max_phase,
            phase_center=phase_center,
            same_phase_across_blocks=same_phase_across_blocks,
        )

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    # -------------------------
    # A_z = K11: block diagonal 2x2 blocks
    # -------------------------
    def _K11_structured(self) -> torch.Tensor:
        n_pairs = self.d_state // 2
        rho = torch.sigmoid(self.rho_raw) * (1.0 - self.eps_radius)  # (n_pairs,)
        th = self.theta
        c, s = torch.cos(th), torch.sin(th)

        K11 = torch.zeros(
            self.d_state,
            self.d_state,
            device=rho.device,
            dtype=rho.dtype,
        )
        for i in range(n_pairs):
            r = rho[i]
            K11[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = r * torch.stack(
                [
                    torch.stack([c[i], -s[i]]),
                    torch.stack([s[i],  c[i]]),
                ],
                dim=0,
            )
        return K11

    # -------------------------
    # Global exact normalization across experts
    # -------------------------
    def _build_expert_blocks_normalized(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            A_z : (dx, dx)              constant (scan-friendly)
            K12 : (M, dx, du)
            K21 : (M, dy, dx)
            K22 : (M, dy, du)

        with a global scale ensuring ||K^(m)||_2 <= 1 for all experts m,
        and shared A_z across experts.
        """
        dx, du, dy, M = self.d_state, self.d_input, self.d_output, self.n_experts

        K11 = self._K11_structured()                                   # (dx,dx)
        K11_exp = K11.unsqueeze(0).expand(M, -1, -1)                   # (M,dx,dx)

        top = torch.cat([K11_exp, self.K12_raw], dim=2)                # (M, dx, dx+du)
        bot = torch.cat([self.K21_raw, self.K22_raw], dim=2)           # (M, dy, dx+du)
        K_raw = torch.cat([top, bot], dim=1)                           # (M, dx+dy, dx+du)

        if self.exact_norm:
            sigmas = torch.linalg.matrix_norm(K_raw, ord=2, dim=(-2, -1))  # (M,)
        else:
            # Frobenius is an upper bound (guaranteed, cheaper, more conservative)
            sigmas = torch.linalg.matrix_norm(K_raw, ord="fro", dim=(-2, -1))

        scale = torch.clamp(sigmas.max(), min=1.0)                     # scalar >= 1

        K = K_raw / scale

        A_z = K[0, :dx, :dx]   # shared across experts due to shared K11_exp + single scale
        K12 = K[:, :dx, dx:]
        K21 = K[:, dx:, :dx]
        K22 = K[:, dx:, dx:]
        return A_z, K12, K21, K22

    # -------------------------
    # Init A eigenvalues near unit circle
    # -------------------------
    @torch.no_grad()
    def init_on_circle(
        self,
        rho: float = 0.99,
        *,
        max_phase: Optional[float] = 0.2,
        phase_center: float = 0.0,
        same_phase_across_blocks: bool = False,
        random_phase: bool = True,
    ):
        """
        Initialize K11 blocks so eig(A_z) ≈ rho * exp(±j theta_i), with rho close to 1.
        """
        assert 0.0 < rho < 1.0, "rho must be in (0,1)"
        n_pairs = self.d_state // 2
        device = self.rho_raw.device
        dtype = self.rho_raw.dtype

        # Set radii via rho_i = sigmoid(rho_raw_i) * (1 - eps_radius)
        target = rho / (1.0 - self.eps_radius)
        target = float(max(min(target, 0.999), 0.001))
        t = torch.full((n_pairs,), target, device=device, dtype=dtype)
        self.rho_raw.copy_(torch.log(t) - torch.log(1.0 - t))  # logit(target)

        # Phases
        if not random_phase:
            self.theta.fill_(phase_center)
            return

        if max_phase is None:
            if same_phase_across_blocks:
                phi = (2 * math.pi * torch.rand(1, device=device, dtype=dtype) - math.pi)
                self.theta.copy_(phi.expand(n_pairs))
            else:
                self.theta.uniform_(-math.pi, math.pi)
        else:
            low = phase_center - max_phase
            high = phase_center + max_phase
            if same_phase_across_blocks:
                phi = (high - low) * torch.rand(1, device=device, dtype=dtype) + low
                self.theta.copy_(phi.expand(n_pairs))
            else:
                self.theta.uniform_(low, high)

    # -------------------------
    # Forward: loop or scan
    # -------------------------
    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,   # z0 in z-basis
        *,
        time_first: bool = False,
        return_state: bool = False,
        mode: Literal["loop", "scan"] = "scan",
    ):
        """
        Args:
            u: (B,T,du) if time_first=False else (T,B,du)
            state: z0 either (B,dx) or (dx,) or None => zeros
            mode: "loop" or "scan"
        Returns:
            y_seq: (B,T,dy) or (T,B,dy)
            z_last: (B,dx)
            (optional) z_seq: (B,T,dx) or (T,B,dx)
        """
        if u.dim() == 2:
            # single-step
            u = u.unsqueeze(1) if not time_first else u.unsqueeze(1)

        if time_first:
            u_bt = u.transpose(0, 1)  # (B,T,du)
        else:
            u_bt = u

        Bsz, T, du = u_bt.shape
        assert du == self.d_input
        dx, dy, M = self.d_state, self.d_output, self.n_experts

        device, dtype = u_bt.device, u_bt.dtype

        # init state z
        if state is None:
            z0 = torch.zeros(Bsz, dx, device=device, dtype=dtype)
        else:
            if state.dim() == 1:
                z0 = state.unsqueeze(0).expand(Bsz, -1).to(device=device, dtype=dtype).contiguous()
            else:
                z0 = state.to(device=device, dtype=dtype)

        # build normalized blocks (shared A_z)
        A_z, K12, K21, K22 = self._build_expert_blocks_normalized()
        g = self.gamma.to(device=device, dtype=dtype)

        # gate weights pi_{b,t,m} (depends only on u_t => scan-friendly later)
        logits = self.gate_net(u_bt.reshape(Bsz * T, du)).reshape(Bsz, T, M)
        logits = logits / self.gate_temperature
        pi = F.softmax(logits, dim=-1)  # (B,T,M)

        # Precompute v_t := B_t u_t = gamma * (sum_m pi_m K12_m) u_t
        # v_all: (B,T,M,dx)
        v_all = torch.einsum("btu,msu->btms", u_bt, (g * K12))  # (B,T,M,dx)
        v_bt = torch.einsum("btm,btms->bts", pi, v_all)         # (B,T,dx)

        if mode == "loop":
            y_bt = torch.empty(Bsz, T, dy, device=device, dtype=dtype)
            z_bt = torch.empty(Bsz, T, dx, device=device, dtype=dtype) if return_state else None

            z = z0
            for t in range(T):
                u_t = u_bt[:, t, :]      # (B,du)
                pi_t = pi[:, t, :]       # (B,M)

                if return_state:
                    z_bt[:, t, :] = z

                # y_state: sum_m pi_m (K21_m z)
                y_state_all = torch.einsum("bs,mys->bmy", z, K21)          # (B,M,dy)
                y_state = torch.einsum("bm,bmy->by", pi_t, y_state_all)    # (B,dy)

                # y_in: sum_m pi_m (gamma*K22_m u)
                y_in_all = torch.einsum("bu,myu->bmy", u_t, (g * K22))     # (B,M,dy)
                y_in = torch.einsum("bm,bmy->by", pi_t, y_in_all)          # (B,dy)

                y_bt[:, t, :] = y_state + y_in

                # z_{t+1} = A_z z_t + v_t
                z = torch.matmul(z, A_z.T) + v_bt[:, t, :]

            z_last = z

        else:
            # --- SCAN MODE ---
            # Use your scan code with B_eff = I and u_eff = v (time-major):
            # x_{t+1} = A x_t + I @ v_t.
            v_tb = v_bt.transpose(0, 1).contiguous()  # (T,B,dx)
            I = torch.eye(dx, device=device, dtype=dtype)

            states = compute_linear_recurrence_parallel_block2x2(A_z, I, v_tb, z0)  # (T+1,B,dx)
            z_tb = states[:-1]                           # (T,B,dx) = z_0..z_{T-1}
            z_last = states[-1]                          # (B,dx)
            z_bt_seq = z_tb.transpose(0, 1).contiguous() # (B,T,dx)

            # y_t uses z_t and u_t with the same pi_t
            y_state_all = torch.einsum("bts,mys->btmy", z_bt_seq, K21)      # (B,T,M,dy)
            y_state = torch.einsum("btm,btmy->bty", pi, y_state_all)        # (B,T,dy)

            y_in_all = torch.einsum("btu,myu->btmy", u_bt, (g * K22))       # (B,T,M,dy)
            y_in = torch.einsum("btm,btmy->bty", pi, y_in_all)              # (B,T,dy)

            y_bt = y_state + y_in

            if return_state:
                z_bt = z_bt_seq

        # restore time_first if requested
        if time_first:
            y_out = y_bt.transpose(0, 1)  # (T,B,dy)
            if return_state:
                z_out = z_bt.transpose(0, 1)  # (T,B,dx)
        else:
            y_out = y_bt
            if return_state:
                z_out = z_bt

        if return_state:
            return y_out, z_last, z_out
        return y_out, z_bt_seq
