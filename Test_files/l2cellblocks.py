import torch
from torch import nn

# Optional hook: if you already have a parallel scan kernel, plug it here.
_HAS_SCAN = False
try:
    from SSM.scan_utils import compute_linear_recurrence_parallel
    _HAS_SCAN = True
except Exception:
    _HAS_SCAN = False


class Block2x2DenseL2SSM(nn.Module):
    r"""
    L2-bounded SSM with:

      - internal **energy coordinates** z ∈ R^{d_state},
      - block-diagonal A_z = K11 with 2x2 real blocks (complex eigenvalues),
      - a contraction K on [z; γ u] → [z⁺; y], so ℓ₂ gain ≤ γ,
      - an extra change-of-basis S so you can recover a **dense (A,B,C,D)** via
            A = S^{-1} K11 S,
            B = γ S^{-1} K12,
            C = K21 S,
            D = γ K22.

    Core contraction:
        [ z_{t+1} ]   [ K11 K12 ] [ z_t     ]
        [   y_t   ] = [ K21 K22 ] [ γ u_t   ]
    with ||K||_2 ≤ 1.

    Forward recursion in z-coordinates:
        z_{t+1} = A_z z_t + B_z u_t
        y_t     = C_z z_t + D_z u_t
    where A_z = K11 (block 2x2), B_z = γ K12, C_z = K21, D_z = γ K22.

    The BRL inequality in x-coordinates holds with V(x) = ||S x||^2, P = S^T S.

    Args
    ----
    d_state : int
        Must be even (2x2 blocks).
    d_input : int
    d_output: int
    gamma   : float
        Prescribed L2 gain bound.
    train_gamma : bool
        If True, gamma is trainable (log-param).
    eps_radius : float
        Margin so |ρ_i| ≤ 1 - eps_radius.
    power_iters : int
        Power iterations for approximate spectral norm (if exact_norm=False).
    exact_norm : bool
        If True, use SVD for exact spectral norm (hard guarantee, slower).
    init_rho : float | None
        If not None, initialize |eig(K11)| ≈ init_rho (<1).
    """

    def __init__(
        self,
        d_state: int,
        d_input: int,
        d_output: int,
        *,
        gamma: float = 1.0,
        train_gamma: bool = False,
        eps_radius: float = 1e-3,
        power_iters: int = 1,
        exact_norm: bool = False,
        init_rho: float | None = None,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state must be even (2x2 blocks)."

        self.d_state = d_state
        self.d_input = d_input
        self.d_output = d_output
        self.eps_radius = eps_radius
        self.power_iters = power_iters
        self.exact_norm = exact_norm

        n_pairs = d_state // 2

        # --- change-of-basis S: lets you recover a dense (A,B,C,D) in x-basis
        self.S = nn.Parameter(0.1 * torch.randn(d_state, d_state))

        # --- structured K11 params (2x2 blocks): ρ_i, θ_i ---
        # ρ_i = sigmoid(rho_raw_i) * (1 - eps_radius) ∈ (0, 1 - eps)
        self.rho_raw = nn.Parameter(0.01 * torch.randn(n_pairs))
        # θ_i ∈ ℝ, used directly as angle
        self.theta = nn.Parameter(0.01 * torch.randn(n_pairs))

        # --- off-diagonal blocks of K are dense ---
        self.K12_raw = nn.Parameter(0.05 * torch.randn(d_state, d_input))
        self.K21_raw = nn.Parameter(0.05 * torch.randn(d_output, d_state))
        self.K22_raw = nn.Parameter(0.05 * torch.randn(d_output, d_input))

        # --- gamma (>0) ---
        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        # optional: put |eig(K11)| ≈ init_rho at init
        if init_rho is not None:
            self.init_near_identity(init_rho)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    # ----------------------------------------------------------------------
    # Structured K11: block-diagonal with 2x2 blocks ρ R(θ).
    # ----------------------------------------------------------------------
    def _K11_structured(self) -> torch.Tensor:
        n_pairs = self.d_state // 2
        rho = torch.sigmoid(self.rho_raw) * (1.0 - self.eps_radius)
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
            block = torch.tensor(
                [[c[i], -s[i]],
                 [s[i],  c[i]]],
                device=K11.device,
                dtype=K11.dtype,
            )
            K11[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = r * block
        return K11

    # ----------------------------------------------------------------------
    # Contract K via spectral normalization
    # ----------------------------------------------------------------------
    def _spectral_normalize(self, M: torch.Tensor) -> torch.Tensor:
        """
        Scale M so that ||M||_2 <= 1 (or ≈ 1 if exact_norm=True).
        """
        if self.exact_norm:
            sigma = torch.linalg.svdvals(M)[0].clamp(min=1e-12)
            scale = torch.maximum(sigma, M.new_tensor(1.0))
            return M / scale

        # Approximate norm via power iteration
        with torch.no_grad():
            u = torch.randn(M.shape[0], device=M.device, dtype=M.dtype)
            u = u / (u.norm() + 1e-12)
            for _ in range(self.power_iters):
                v = M.t().matmul(u)
                v = v / (v.norm() + 1e-12)
                u = M.matmul(v)
                u = u / (u.norm() + 1e-12)
            sigma = u.dot(M.matmul(v)).clamp(min=1e-12)
        scale = torch.maximum(sigma, M.new_tensor(1.0))
        return M / scale

    def _build_K_blocks(self):
        """
        Build contraction K = [[K11,K12],[K21,K22]] and return its blocks.
        """
        K11_struct = self._K11_structured()
        top = torch.cat([K11_struct, self.K12_raw], dim=1)      # (dx, dx+du)
        bottom = torch.cat([self.K21_raw, self.K22_raw], dim=1) # (dy, dx+du)
        K_raw = torch.cat([top, bottom], dim=0)                 # (dx+dy, dx+du)

        K = self._spectral_normalize(K_raw)

        dx, du, dy = self.d_state, self.d_input, self.d_output
        K11 = K[:dx, :dx]
        K12 = K[:dx, dx:]
        K21 = K[dx:, :dx]
        K22 = K[dx:, dx:]
        return K11, K12, K21, K22

    # ----------------------------------------------------------------------
    # Matrices in **z-coordinates** (scan-friendly block 2x2 A_z)
    # ----------------------------------------------------------------------
    def compute_z_matrices(self):
        """
        Return (A_z, B_z, C_z, D_z, P_z) in z-coordinates, where P_z = I.

        z_{t+1} = A_z z_t + B_z u_t
        y_t     = C_z z_t + D_z u_t

        with A_z block-diag (2x2), and ||K||_2 <= 1 ⇒ ℓ₂ gain <= γ.
        """
        K11, K12, K21, K22 = self._build_K_blocks()
        gamma = self.gamma

        A_z = K11
        B_z = gamma * K12
        C_z = K21
        D_z = gamma * K22

        P_z = torch.eye(self.d_state, device=A_z.device, dtype=A_z.dtype)
        return A_z, B_z, C_z, D_z, P_z

    # ----------------------------------------------------------------------
    # Matrices in **x-coordinates** (dense A,B,C,D) if you care about them
    # ----------------------------------------------------------------------
    def compute_dense_matrices(self):
        """
        Return (A_x,B_x,C_x,D_x,P_x) in x-coordinates, using S as the state
        change-of-basis: z = S x.

            A_x = S^{-1} A_z S
            B_x = γ S^{-1} K12
            C_x = K21 S
            D_x = γ K22
            P_x = S^T S

        These satisfy the discrete bounded-real LMI with gain γ.
        """
        A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()
        S = self.S
        gamma = self.gamma

        Sinv = torch.linalg.inv(S)
        # A_x = S^{-1} A_z S
        A_x = Sinv @ A_z @ S
        # B_z = γ K12, so K12 = B_z / γ; B_x = γ S^{-1} K12 = S^{-1} B_z
        B_x = Sinv @ B_z
        # C_x = K21 S = C_z S / γ ?  (No: C_z = K21, D_z = γ K22)
        C_x = C_z @ S
        # D_x = D_z (since D_z = γ K22)
        D_x = D_z

        P_x = S.T @ S
        return A_x, B_x, C_x, D_x, P_x

    def bounded_real_matrix_x(self, gamma: float | None = None) -> torch.Tensor:
        """
        BRL matrix in x-basis, using P_x = S^T S, to sanity-check:

            [ A^T P A - P + C^T C    A^T P B + C^T D ]
            [ B^T P A + D^T C        B^T P B + D^T D - γ^2 I ]
        """
        A, B, C, D, P = self.compute_dense_matrices()
        if gamma is None:
            gamma = self.gamma
        else:
            gamma = torch.tensor(float(gamma), device=A.device, dtype=A.dtype)

        d_x = A.shape[0]
        d_u = B.shape[1]

        AtPA = A.T @ P @ A
        AtPB = A.T @ P @ B
        BtPA = B.T @ P @ A
        BtPB = B.T @ P @ B

        CtC = C.T @ C
        CtD = C.T @ D
        DtC = D.T @ C
        DtD = D.T @ D

        top_left = AtPA - P + CtC
        top_right = AtPB + CtD
        bot_left = BtPA + DtC
        bot_right = BtPB + DtD - (gamma**2) * torch.eye(d_u, device=A.device, dtype=A.dtype)

        top = torch.cat([top_left, top_right], dim=1)
        bottom = torch.cat([bot_left, bot_right], dim=1)
        M = torch.cat([top, bottom], dim=0)
        return M

    # ----------------------------------------------------------------------
    # Initialization: |eig(K11)| ≈ rho
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def init_near_identity(self, rho: float = 0.99, offdiag_scale: float = 1e-3):
        """
        Initialize so that each 2x2 block of A_z = K11 has eigenvalues
        |λ| ≈ rho (<1), and off-diagonal blocks (K12,K21,K22) are small.

        - ρ_i = rho for all i (via rho_raw),
        - θ_i ≈ 0 so blocks ≈ rho * I_2,
        - off-diagonals small so ||K_raw||_2 ≈ rho and spectral normalization
          does not rescale at init.

        S is initialized near identity, so the dense A_x ≈ block-diag A_z in x-basis.
        """
        assert 0.0 < rho < 1.0, "rho should be in (0,1)"

        n_pairs = self.d_state // 2
        device = self.rho_raw.device
        dtype = self.rho_raw.dtype

        # S ≈ I
        S_eye = torch.eye(self.d_state, device=device, dtype=dtype)
        S_pert = 0.01 * torch.randn(self.d_state, self.d_state, device=device, dtype=dtype)
        self.S.copy_(S_eye + S_pert)

        # ρ_i = sigmoid(rho_raw_i) * (1 - eps_radius) ≈ rho
        target = rho / (1.0 - self.eps_radius)
        target = float(max(min(target, 0.999), 0.001))
        t = torch.full((n_pairs,), target, device=device, dtype=dtype)
        self.rho_raw.copy_(torch.log(t) - torch.log(1 - t))

        # Small angles
        self.theta.zero_()

        # Small off-diagonal blocks
        self.K12_raw.normal_(mean=0.0, std=offdiag_scale)
        self.K21_raw.normal_(mean=0.0, std=offdiag_scale)
        self.K22_raw.normal_(mean=0.0, std=offdiag_scale)

    # ----------------------------------------------------------------------
    # One-step update in z-coordinates
    # ----------------------------------------------------------------------
    def step(self, z: torch.Tensor, u: torch.Tensor):
        """
        One step in z-coordinates:

            z_{t+1} = A_z z_t + B_z u_t
            y_t     = C_z z_t + D_z u_t

        z: (B, d_state), u: (B, d_input)
        """
        A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()
        z_next = z @ A_z.T + u @ B_z.T
        y = z @ C_z.T + u @ D_z.T
        return z_next, y

    # ----------------------------------------------------------------------
    # Forward: loop (for now) + scan hook
    # ----------------------------------------------------------------------
    def forward(
        self,
        u: torch.Tensor,
        z0: torch.Tensor | None = None,   # state in z-basis
        *,
        time_first: bool = False,
        return_state: bool = False,
        mode: str = "loop",               # "loop" or "scan"
    ):
        """
        Forward in z-coordinates (internal state). This is the thing you'd
        plug into your L2RU-like block; z is just the hidden state.

        Args
        ----
        u :  (B,T,d_input) if time_first=False
             (T,B,d_input) if time_first=True
        z0 : (B,d_state) or (d_state,) or None (zero init)
        time_first : if True, interpret first dim as time
        return_state : if True, also return z_seq
        mode : "loop" or "scan"

        Returns
        -------
        y_seq  : (B,T,d_output)
        z_last : (B,d_state)
        (optional) z_seq : (B,T,d_state)
        """
        if u.dim() == 2:
            if time_first:
                u = u.unsqueeze(1)  # (T,1,du)
            else:
                u = u.unsqueeze(0)  # (1,T,du)

        if time_first:
            u = u.transpose(0, 1)  # (B,T,du)

        B_sz, T, du = u.shape
        assert du == self.d_input

        A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()
        At, Bt, Ct, Dt = A_z.T, B_z.T, C_z.T, D_z.T

        # initial state
        if z0 is None:
            z = u.new_zeros(B_sz, self.d_state)
        else:
            if z0.dim() == 1:
                z = z0.unsqueeze(0).expand(B_sz, -1)
            else:
                z = z0
                assert z.shape == (B_sz, self.d_state)

        # LOOP mode
        if mode != "scan" or not _HAS_SCAN:
            y_seq = u.new_empty(B_sz, T, self.d_output)
            z_seq = u.new_empty(B_sz, T, self.d_state)

            for t in range(T):
                u_t = u[:, t, :]
                z_seq[:, t, :] = z
                y_t = z @ Ct + u_t @ Dt
                y_seq[:, t, :] = y_t
                z = z @ At + u_t @ Bt

            z_last = z

        else:
            # SCAN mode (if you have compute_linear_recurrence_parallel)
            # u_scan: (T,B,du)
            u_scan = u.transpose(0, 1)
            # states: (T+1,B,d_state)
            states = compute_linear_recurrence_parallel(A_z, B_z, u_scan, z)
            z_seq = states[:-1].transpose(0, 1)  # (B,T,d_state)
            z_last = states[-1]                  # (B,d_state)
            y_seq = z_seq @ Ct + u @ Dt          # (B,T,d_output)

        if time_first:
            y_seq = y_seq.transpose(0, 1)
            z_seq = z_seq.transpose(0, 1)

        if return_state:
            return y_seq, z_last, z_seq
        return y_seq, z_last
