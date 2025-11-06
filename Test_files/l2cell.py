import torch
from torch import nn


class L2BoundedLTICell(nn.Module):
    """
    Dense LTI cell with *hard* L2-gain bound via:

        [ S x_{k+1} ]   [ K11 K12 ] [ S x_k   ]
        [   y_k     ] = [ K21 K22 ] [ gamma u_k ]

    with ||K||_2 <= 1 (enforced via SVD).
    """

    def __init__(self, d_state, d_input, d_output, gamma=1.0, train_gamma=False):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input
        self.d_output = d_output

        # S: energy transform (invertible with prob. 1)
        self.S = nn.Parameter(0.1 * torch.randn(d_state, d_state))

        # Raw K (we'll spectral-normalize it)
        self.K_raw = nn.Parameter(
            1 * torch.randn(d_state + d_output, d_state + d_input)
        )

        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

    @property
    def gamma(self):
        return self.log_gamma.exp()

    # ---- enforce K contraction exactly via SVD ----
    def _build_contraction(self):
        K_raw = self.K_raw
        # exact largest singular value
        sigma = torch.linalg.matrix_norm(K_raw, ord=2)
        sigma = sigma.clamp(min=1e-5)
        K = K_raw / (sigma + 0.002)
        return K

    # ---- map (S,K) -> (A,B,C,D,P) correctly ----
    def compute_ssm_matrices(self):
        d_x = self.d_state
        d_u = self.d_input
        d_y = self.d_output

        S = self.S
        gamma = self.gamma
        K = self._build_contraction()  # (d_x + d_y, d_x + d_u)

        K11 = K[:d_x, :d_x]
        K12 = K[:d_x, d_x:]
        K21 = K[d_x:, :d_x]
        K22 = K[d_x:, d_x:]

        Sinv = torch.linalg.inv(S)

        # S A = K11 S     -> A = S^{-1} K11 S
        # S B = γ K12     -> B = γ S^{-1} K12
        # C   = K21 S
        # D   = γ K22
        A = Sinv @ K11 @ S
        B = gamma * (Sinv @ K12)
        C = K21 @ S
        D = gamma * K22

        P = S.T @ S

        return A, B, C, D, P

    def bounded_real_matrix(self, gamma=None):
        A, B, C, D, P = self.compute_ssm_matrices()
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
        bot_right = BtPB + DtD - (gamma ** 2) * torch.eye(d_u, device=A.device, dtype=A.dtype)

        top = torch.cat([top_left, top_right], dim=1)
        bottom = torch.cat([bot_left, bot_right], dim=1)
        M = torch.cat([top, bottom], dim=0)
        return M

        # ---------- NEW: initialization with eig(A) ≈ rho ----------

    @torch.no_grad()
    def init_orthogonal_spectrum(
            self,
            eigvals: torch.Tensor,
            offdiag_scale: float = 0.5,
    ):
        """
        Initialize such that:

            - A has a prescribed real spectrum at init (same as eigvals),
            - K has reasonably large off-diagonal blocks (so H∞ is not tiny),
            - K is still a contraction (||K||_2 <= 1).

        Steps:
          1. Choose eigvals (d_state,) with |eigvals[i]| < 1.
          2. Build K11_raw = Q diag(eigvals) Q^T with random orthogonal Q.
          3. Sample K12_raw, K21_raw, K22_raw at moderate scale (offdiag_scale).
          4. Scale K_raw so that ||K_raw||_2 <= 1 (contraction).
          5. Optionally re-scale once more so that max |eig(K11)| = max |eigvals|
             (so the spectral radius of A matches what you asked for).

        This gives:
            spec(A) = spec(K11) ≈ eigvals, with non-trivial input/output coupling.
        """
        d_x = self.d_state
        d_u = self.d_input
        d_y = self.d_output

        eigvals = eigvals.to(self.S.device, self.S.dtype)
        assert eigvals.shape == (d_x,), f"eigvals must have shape ({d_x},)"
        assert (eigvals.abs() < 1.0).all(), "All |eigvals| must be < 1."

        device = self.S.device
        dtype = self.S.dtype

        # 0) S ≈ I so A ≈ K11 in original coordinates
        S_eye = torch.eye(d_x, device=device, dtype=dtype)
        S_pert = 0.01 * torch.randn(d_x, d_x, device=device, dtype=dtype)
        self.S.copy_(S_eye + S_pert)

        # 1) Random orthogonal basis Q
        Q, _ = torch.linalg.qr(torch.randn(d_x, d_x, device=device, dtype=dtype))
        Lambda = torch.diag(eigvals)
        K11 = Q @ Lambda @ Q.T  # symmetric with desired eigenvalues

        # 2) Off-diagonal blocks with "healthy" scale
        K12 = offdiag_scale * torch.randn(d_x, d_u, device=device, dtype=dtype)
        K21 = offdiag_scale * torch.randn(d_y, d_x, device=device, dtype=dtype)
        K22 = offdiag_scale * torch.randn(d_y, d_u, device=device, dtype=dtype)

        # 3) Assemble K_raw_full
        top = torch.cat([K11, K12], dim=1)  # (d_x, d_x + d_u)
        bottom = torch.cat([K21, K22], dim=1)  # (d_y, d_x + d_u)
        K_full = torch.cat([top, bottom], dim=0)  # (d_x + d_y, d_x + d_u)

        # 4) First scaling: make K a contraction (||K||_2 <= 1)
        #    so that our parametrization is consistent with the BRL construction.
        sigma_init = torch.linalg.svdvals(K_full)[0].item()  # largest s.v.
        if sigma_init > 1.0:
            K_full /= sigma_init  # now ||K_full||_2 <= 1

        # 5) Optional second scaling: adjust spectral radius of K11
        #    so that max |eig(K11)| = max |eigvals| exactly.
        #    (This keeps K a contraction, because we only scale DOWN.)
        with torch.no_grad():
            # recompute K11 block after step 4
            K11_scaled = K_full[:d_x, :d_x]
            ev_K11 = torch.linalg.eigvals(K11_scaled)
            rho_current = ev_K11.abs().max().item()
        rho_target = eigvals.abs().max().item()
        if rho_current > 0:
            scale2 = min(1.0, rho_target / rho_current)
            K_full *= scale2  # still a contraction; K11 radius now ≈ rho_target

        # 6) Write back
        self.K_raw.copy_(K_full)

    def step(self, x: torch.Tensor, u: torch.Tensor):
        """
        One-step update:

            x_{t+1} = A x_t + B u_t
            y_t     = C x_t + D u_t

        Args
        ----
        x : (B, d_state)
        u : (B, d_input)

        Returns
        -------
        x_next : (B, d_state)
        y      : (B, d_output)
        """
        A, B, C, D, _ = self.compute_ssm_matrices()
        # Using row-batch convention: x @ A^T etc.
        x_next = x @ A.T + u @ B.T  # (B, d_state)
        y = x @ C.T + u @ D.T  # (B, d_output)
        return x_next, y

    # ---------- FORWARD: efficient loop over time ----------

    def forward(
            self,
            u: torch.Tensor,
            x0: torch.Tensor | None = None,
            *,
            time_first: bool = False,
            return_state: bool = True,
    ):
        """
        Sequential state-space recurrence (loop version), similar to your L2RU.

        Recurrence:
            x_{t+1} = A x_t + B u_t
            y_t     = C x_t + D u_t

        Args
        ----
        u :  (B, T, d_input) if time_first=False
             (T, B, d_input) if time_first=True
             (T, d_input) or (T,) will be promoted to batch size 1.
        x0 : (B, d_state) or (d_state,) or None (zero init).
        time_first : if True, interpret first dim as time.
        return_state : if True, also return full x_seq.

        Returns
        -------
        y_seq  : (B, T, d_output)
        x_last : (B, d_state)
        (optional) x_seq : (B, T, d_state)
        """
        # Normalize input shape
        if u.dim() == 2:
            # (T, d_input) -> (1, T, d_input)
            if time_first:
                u = u.unsqueeze(1)  # (T, 1, d_input)
            else:
                u = u.unsqueeze(0)  # (1, T, d_input)

        if time_first:
            # (T, B, d_in) -> (B, T, d_in)
            u = u.transpose(0, 1)

        B_sz, T, d_in = u.shape
        assert d_in == self.d_input, f"u.shape[-1]={d_in}, expected {self.d_input}"

        # Build A,B,C,D once per sequence (important for speed)
        A, Bm, C, D, _ = self.compute_ssm_matrices()

        # Prepare state
        if x0 is None:
            x = u.new_zeros(B_sz, self.d_state)
        else:
            if x0.dim() == 1:
                x = x0.unsqueeze(0).expand(B_sz, -1)
            else:
                x = x0
                assert x.shape == (B_sz, self.d_state)

        # Precompute transposes for efficient GEMV
        At = A.T
        Bt = Bm.T
        Ct = C.T
        Dt = D.T

        # Allocate output tensors
        y_seq = u.new_empty(B_sz, T, self.d_output)
        x_seq = u.new_empty(B_sz, T, self.d_state)

        # Efficient Python loop over time, batched matmuls inside
        for t in range(T):
            u_t = u[:, t, :]  # (B, d_in)
            x_seq[:, t, :] = x  # store current state
            # y_t = C x_t + D u_t
            y_t = x @ Ct + u_t @ Dt  # (B, d_out)
            y_seq[:, t, :] = y_t
            # x_{t+1} = A x_t + B u_t
            x = x @ At + u_t @ Bt  # (B, d_state)

        x_last = x

        if time_first:
            y_seq = y_seq.transpose(0, 1)
            x_seq = x_seq.transpose(0, 1)

        if return_state:
            return y_seq, x_seq
        return y_seq, x_last
