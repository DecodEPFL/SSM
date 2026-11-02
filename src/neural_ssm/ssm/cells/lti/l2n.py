"""Normalized small-block LTI systems, with identity or full state storage."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ....utils.lyapunov import block_left_multiply
from ....utils.runtime import (
    EvalCacheMixin,
    normalize_to_3d,
    reset_runtime_state as _reset_runtime_state,
    resolve_runtime_state as _resolve_runtime_state,
    sequence_output,
)
from ....utils.scan import compute_block_lti_states, compute_linear_recurrence_parallel_block2x2
from ..common import (
    _CONTRACTION_EPS,
    bounded_real_matrix as _bounded_real_matrix,
    block_diagonal,
    positive_upper_factor,
    rotation_block_diagonal,
)


class Block2x2DenseL2SSM(EvalCacheMixin, nn.Module):
    r"""
    Identity-metric L2-bounded SSM with:

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

    ``state_metric='full'`` instead learns P=L.T L in the scan coordinates
    themselves. Free real 2x2 blocks cover real poles and complex pairs, with
    an optional final scalar. Only the weighted norm sees L A L^-1; dividing
    the original matrices by one scalar preserves their block recurrence.
    There is no additional returned-state basis S in this mode. Preparation
    includes dense triangular solves and an SVD; gradient-free eval caches it.

    Args
    ----
    d_state : int
        Positive; identity mode requires an even size (rotation pairs).
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
    energy_only : bool
        Omit the external state basis S. MetricDeepSSM uses the core's own
        coordinates; full mode always omits this additional basis.
    state_metric : str
        'identity' retains the original rotation model; 'full' uses a dense
        certificate inside exact normalization. Full storage requires float32
        or float64 parameters and supports scan/loop, rather than the complex
        rotation model's convolution engine. eps_radius is identity-mode only.
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
        exact_norm: bool = True,
        learn_x0: bool = False,
        energy_only: bool = False,
        state_metric: str = "identity",
    ):
        super().__init__()
        if state_metric not in ("identity", "full"):
            raise ValueError("state_metric must be 'identity' or 'full'.")
        if d_state <= 0 or (state_metric == "identity" and d_state % 2):
            raise ValueError("d_state must be positive; identity storage requires an even size.")
        if d_input <= 0 or d_output <= 0:
            raise ValueError("Input and output dimensions must be positive.")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive.")
        if not 0 <= eps_radius < 1:
            raise ValueError("eps_radius must be in [0, 1).")
        if state_metric == "full" and not exact_norm:
            raise ValueError("Full state storage requires exact_norm=True for its certificate.")

        self.d_state = d_state
        self.d_input = d_input
        self.d_output = d_output
        self.eps_radius = eps_radius
        self.power_iters = power_iters
        self.exact_norm = exact_norm
        self.state_metric = state_metric
        self.state: torch.Tensor | None = None

        n_pairs = d_state // 2

        # --- change-of-basis S: lets you recover a dense (A,B,C,D) in x-basis
        if energy_only or state_metric == "full":
            # MetricDeepSSM already evolves the state in energy coordinates.
            # Avoid unused n-by-n parameters, storage, and optimizer work.
            self.register_buffer("S", None)
        else:
            self.S = nn.Parameter(0.1 * torch.randn(d_state, d_state))

        # --- structured K11 params (2x2 blocks): ρ_i, θ_i ---
        # ρ_i = sigmoid(rho_raw_i) * (1 - eps_radius) ∈ (0, 1 - eps)
        if state_metric == "identity":
            self.rho_raw = nn.Parameter(0.01 * torch.randn(n_pairs))
            # θ_i ∈ ℝ, used directly as angle
            self.theta = nn.Parameter(0.01 * torch.randn(n_pairs))
        else:
            # Unrestricted blocks avoid the rotation-only model's pole limits.
            # The weighted normalization below supplies stability and the gain.
            self.A_blocks_raw = nn.Parameter(.5 * torch.eye(2).expand(n_pairs, 2, 2).clone())
            if d_state % 2:
                self.A_last_raw = nn.Parameter(torch.tensor(.5))
            else:
                self.register_buffer("A_last_raw", None)
            self.register_buffer("metric_indices", torch.triu_indices(d_state, d_state, 1),
                                 persistent=False)
            self.metric_upper_raw = nn.Parameter(torch.zeros(d_state * (d_state - 1) // 2))
            self.metric_diag_raw = nn.Parameter(torch.full((d_state,), math.log(math.expm1(1.))))

        # --- off-diagonal blocks of K are dense ---
        self.K12_raw = nn.Parameter(0.5 * torch.randn(d_state, d_input))
        self.K21_raw = nn.Parameter(0.5 * torch.randn(d_output, d_state))
        self.K22_raw = nn.Parameter(0.5 * torch.randn(d_output, d_input))

        # --- gamma (>0) ---
        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        # Learnable initial condition in x-basis (real, shape (1, d_state))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, d_state))
        else:
            self.register_buffer("x0_param", None)

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    # ----------------------------------------------------------------------
    # Structured K11: block-diagonal with 2x2 blocks ρ R(θ).
    # ----------------------------------------------------------------------
    def _K11_structured(self) -> torch.Tensor:
        rho = torch.sigmoid(self.rho_raw) * (1.0 - self.eps_radius)
        return rotation_block_diagonal(rho, self.theta)

    # ----------------------------------------------------------------------
    # Contract K via spectral normalization
    # ----------------------------------------------------------------------
    def _spectral_norm(self, M: torch.Tensor) -> torch.Tensor:
        """
        Largest singular value, with a safe upper-bound fallback in exact mode.

        Power iteration is an estimate and does not give a hard certificate.
        """

        def _power_iteration_sigma(mat: torch.Tensor, iters: int) -> torch.Tensor:
            # Power iteration to estimate the spectral norm (largest singular value).
            # Both u and v are normalized at each step so estimates remain accurate
            # for any number of iterations.
            v = torch.randn(mat.shape[1], device=mat.device, dtype=mat.dtype)
            v = v / (v.norm() + 1e-12)
            iters = max(1, int(iters))
            for _ in range(iters):
                u = mat @ v
                u_norm = u.norm()
                if u_norm < 1e-12:
                    break
                u = u / u_norm
                v = mat.T @ u
                v_norm = v.norm()
                if v_norm < 1e-12:
                    break
                v = v / v_norm
            u = mat @ v
            return u.norm()

        if self.exact_norm:
            try:
                sigma = torch.linalg.matrix_norm(M, ord=2)
            except RuntimeError:
                # The Frobenius norm is an upper bound. Power iteration would
                # underestimate the norm and lose the exact-mode certificate.
                sigma = torch.linalg.vector_norm(M)
        else:
            sigma = _power_iteration_sigma(M, self.power_iters)

        return sigma

    def _spectral_normalize(self, M: torch.Tensor) -> torch.Tensor:
        # Preserve the original identity-mode normalization and checkpoints.
        sigma = self._spectral_norm(M).clamp(min=1e-5)
        return M / (sigma + _CONTRACTION_EPS)

    def _metric_factor(self):
        def build():
            if self.metric_diag_raw.dtype not in (torch.float32, torch.float64):
                raise TypeError("Full state storage requires float32 or float64 parameters.")
            return positive_upper_factor(self.metric_diag_raw, self.metric_upper_raw, self.metric_indices)
        return self._eval_cached("metric_factor", build)

    def _raw_state_blocks(self):
        blocks = self.A_blocks_raw
        if self.A_last_raw is not None:
            # A zero dummy mode lets the same real block scan handle odd n.
            scalar = torch.diag(torch.stack((self.A_last_raw, self.A_last_raw.new_zeros(()))))
            blocks = torch.cat((blocks, scalar.unsqueeze(0)))
        return blocks

    def _compute_full_metric_blocks(self):
        """Normalize in weighted energy, then run the original block coordinates.

        K_P = [[L A L^-1, L B_unit], [C L^-1, D_unit]]. A single right
        triangular solve prepares both state columns; no inverse or dense P
        is formed. Valid strict contractions are left unchanged, which is
        needed to retain the full interior of the gain-bounded family.
        """
        if self.K12_raw.dtype not in (torch.float32, torch.float64):
            raise TypeError("Full state storage requires float32 or float64 parameters.")
        with torch.autocast(device_type=self.K12_raw.device.type, enabled=False):
            a, l = self._raw_state_blocks(), self._metric_factor()
            la = block_left_multiply(a.mT, l.mT).mT
            state_columns = torch.cat((la, self.K21_raw), dim=0)
            state_columns = torch.linalg.solve_triangular(l.mT, state_columns.mT, upper=False).mT
            input_columns = torch.cat((l @ self.K12_raw, self.K22_raw), dim=0)
            sigma = self._spectral_norm(torch.cat((state_columns, input_columns), dim=1))
            # A machine-precision guard gives strict contraction at the boundary.
            # There is no fixed radius or .002 modelling margin in the full mode.
            alpha = (sigma * (1 + torch.finfo(sigma.dtype).eps)).clamp_min(1)
            return a / alpha, self.K12_raw / alpha, self.K21_raw / alpha, self.K22_raw / alpha

    def compute_block_matrices(self):
        """Return scan-coordinate (A_blocks,B,C,D), without a dense A or P."""
        def build():
            a, b, c, d = self.normalized_block_matrices()
            return a, self.gamma * b, c, self.gamma * d
        return self._eval_cached("blocks", build)

    def storage_matrix(self):
        """Certificate in the returned state coordinates, including the legacy S basis."""
        def build():
            if self.state_metric == "full":
                with torch.autocast(device_type=self.K12_raw.device.type, enabled=False):
                    l = self._metric_factor()
                    return l.mT @ l
            if self.S is not None:
                return self.S.mT @ self.S
            return torch.eye(self.d_state, device=self.K12_raw.device, dtype=self.K12_raw.dtype)
        return self._eval_cached("storage", build)

    def state_energy(self, state):
        """Evaluate certified state energy using its factor, without forming dense P."""
        if state is None:
            return None
        factor = self._metric_factor() if self.state_metric == "full" else self.S
        if factor is None:
            return state.square().sum(dim=-1)
        state = state.to(device=factor.device, dtype=factor.dtype)
        with torch.autocast(device_type=factor.device.type, enabled=False):
            return (state @ factor.mT).square().sum(dim=-1)

    def scan_state_energy(self, state):
        """Storage for normalized_block_matrices(), before the legacy S conversion."""
        if self.state_metric == "full":
            return self.state_energy(state)
        return None if state is None else state.square().sum(dim=-1)

    def _build_K_blocks(self):
        """
        Build contraction K = [[K11,K12],[K21,K22]] and return its blocks.
        """
        K11_struct = self._K11_structured()
        top = torch.cat([K11_struct, self.K12_raw], dim=1)  # (dx, dx+du)
        bottom = torch.cat([self.K21_raw, self.K22_raw], dim=1)  # (dy, dx+du)
        K_raw = torch.cat([top, bottom], dim=0)  # (dx+dy, dx+du)

        K = self._spectral_normalize(K_raw)

        dx = self.d_state
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
        Return dense (A,B,C,D,P) in scan coordinates, for diagnostics/export.

        Identity mode has P_z=I. Full mode returns the learned dense P in the
        recurrence's own coordinates. Forward uses the block matrices directly.

        z_{t+1} = A_z z_t + B_z u_t
        y_t     = C_z z_t + D_z u_t

        with A_z block-diag (2x2), and ||K||_2 <= 1 ⇒ ℓ₂ gain <= γ.

        The contraction K needs an exact spectral norm (SVD); cache the result in
        eval (fixed weights), recompute in training.
        """
        return self._eval_cached("z", self._compute_z_matrices)

    def _compute_z_matrices(self):
        if self.state_metric == "full":
            a, b, c, d = self.compute_block_matrices()
            return block_diagonal(a, self.d_state), b, c, d, self.storage_matrix()
        K11, K12, K21, K22 = self._build_K_blocks()
        gamma = self.gamma

        A_z = K11
        B_z = gamma * K12
        C_z = K21
        D_z = gamma * K22

        P_z = torch.eye(self.d_state, device=A_z.device, dtype=A_z.dtype)
        return A_z, B_z, C_z, D_z, P_z

    def normalized_block_matrices(self):
        """Unit input/output gain with the core's certified state storage."""
        if not self.exact_norm:
            raise ValueError("Metric transfer requires l2n exact_norm=True.")
        if self.state_metric == "full":
            return self._eval_cached("normalized_blocks", self._compute_full_metric_blocks)

        def build():
            a, b, c, d = self._build_K_blocks()
            pairs = self.d_state // 2
            blocks = a.reshape(pairs, 2, pairs, 2).diagonal(dim1=0, dim2=2).movedim(-1, 0)
            return blocks, b, c, d

        return self._eval_cached("normalized_blocks", build)

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
        return self._eval_cached("dense", self._compute_dense_matrices)

    def _solve_state_basis(self, rhs, S=None):
        """Solve S x = rhs; reuse its LU factors only in gradient-free eval."""
        S = self.S if S is None else S
        if S is None:
            return rhs
        if self.training or torch.is_grad_enabled():
            return torch.linalg.solve(S, rhs)
        lu, pivots = self._eval_cached(
            ("state_basis_lu", S.dtype, S.device), lambda: torch.linalg.lu_factor(S)
        )
        return torch.linalg.lu_solve(lu, pivots, rhs)

    def _compute_dense_matrices(self):
        A_z, B_z, C_z, D_z, P_z = self.compute_z_matrices()
        S = self.S
        if S is None:
            return A_z, B_z, C_z, D_z, P_z
        AB_x = self._solve_state_basis(torch.cat((A_z @ S, B_z), dim=1))
        A_x, B_x = AB_x.split((self.d_state, self.d_input), dim=1)
        C_x = C_z @ S
        D_x = D_z

        P_x = S.T @ S
        return A_x, B_x, C_x, D_x, P_x

    def bounded_real_matrix_x(self, gamma=None):
        """Return the discrete bounded-real LMI in the physical state basis."""
        A, B, C, D, P = self.compute_dense_matrices()
        return _bounded_real_matrix(A, B, C, D, P, self.gamma if gamma is None else gamma)

    # ----------------------------------------------------------------------
    # Initialization of the raw rotation template
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def init_on_circle(
        self,
        rho: float = 0.99,
        *,
        # phase control
        max_phase: float | None = None,  # if not None, sample θ in [-max_phase, max_phase]
        phase_center: float = 0.0,  # center of the phase window
        same_phase_across_blocks: bool = False,
        random_phase: bool = True,  # if False -> all θ_i = phase_center
        # off-diagonal scale
        offdiag_scale: float = 0.05,
    ):
        """Initialize raw rotations at radius rho with small input/output coupling.

        With random_phase=True, angles are sampled within max_phase of
        phase_center, or over the full circle when max_phase=None. Set
        same_phase_across_blocks=True to share that angle; random_phase=False
        uses phase_center exactly. offdiag_scale sets the coupling's standard
        deviation. Normalizing the complete contraction rescales the raw
        radii, so rho is not an exact promise for the final transition poles.
        """
        if not (0.0 <= rho < 1.0 if self.state_metric == "full" else 0.0 < rho < 1.0):
            raise ValueError("rho must be in [0,1) for full storage, or (0,1) for identity.")

        if self.state_metric == "full":
            # Initialize free blocks as rotations; training can change every entry.
            blocks = self.A_blocks_raw
            phase = blocks.new_full((blocks.shape[0],), phase_center)
            if random_phase:
                width = math.pi if max_phase is None else max_phase
                if same_phase_across_blocks:
                    phase = phase_center + (2 * torch.rand(1, device=blocks.device, dtype=blocks.dtype) - 1) * width
                    phase = phase.expand(blocks.shape[0])
                else:
                    phase.uniform_(phase_center - width, phase_center + width)
            c, s = rho * phase.cos(), rho * phase.sin()
            blocks.copy_(torch.stack((c, -s, s, c), -1).reshape_as(blocks))
            if self.A_last_raw is not None:
                self.A_last_raw.fill_(rho)
            self.metric_diag_raw.fill_(math.log(math.expm1(1.)))
            self.metric_upper_raw.zero_()
            self.K12_raw.normal_(0, offdiag_scale)
            self.K21_raw.normal_(0, offdiag_scale)
            self.K22_raw.normal_(0, offdiag_scale)
            return

        n_pairs = self.d_state // 2
        device = self.rho_raw.device
        dtype = self.rho_raw.dtype

        # 1) S ≈ I so x- and z-coordinates are initially close
        if self.S is not None:
            S_eye = torch.eye(self.d_state, device=device, dtype=dtype)
            S_pert = 0.01 * torch.randn(self.d_state, self.d_state, device=device, dtype=dtype)
            self.S.copy_(S_eye + S_pert)

        # 2) Set radii ρ_i ≈ rho via the sigmoid parametrization:
        #    rho_i = sigmoid(rho_raw_i) * (1 - eps_radius)
        # => sigmoid(rho_raw_i) = rho / (1 - eps_radius)
        target = rho / (1.0 - self.eps_radius)
        # Clamp to avoid logit blow-up
        target = float(max(min(target, 0.999), 0.001))
        t = torch.full((n_pairs,), target, device=device, dtype=dtype)
        self.rho_raw.copy_(torch.log(t) - torch.log(1 - t))  # logit(target)

        # 3) Set phases θ_i
        if not random_phase:
            # deterministic: all θ_i = phase_center
            self.theta.fill_(phase_center)
        else:
            if max_phase is None:
                # full random circle: θ_i ~ U(-π, π)
                if same_phase_across_blocks:
                    phi = 2 * math.pi * torch.rand(1, device=device, dtype=dtype) - math.pi
                    self.theta.copy_(phi.expand(n_pairs))
                else:
                    self.theta.uniform_(-math.pi, math.pi)
            else:
                # small/random window around phase_center
                # θ ∈ [phase_center - max_phase, phase_center + max_phase]
                low = phase_center - max_phase
                high = phase_center + max_phase
                if same_phase_across_blocks:
                    phi = (high - low) * torch.rand(1, device=device, dtype=dtype) + low
                    self.theta.copy_(phi.expand(n_pairs))
                else:
                    self.theta.uniform_(low, high)

        # 4) Keep input/output coupling small relative to the rotation blocks.
        # Global contraction normalization also rescales their initial radii.
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
        if self.state_metric == "full":
            a, b, c, d = self.compute_block_matrices()
            return block_left_multiply(a, z.mT).mT + F.linear(u, b), F.linear(z, c) + F.linear(u, d)
        A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()
        z_next = z @ A_z.T + u @ B_z.T
        y = z @ C_z.T + u @ D_z.T
        return z_next, y

    def _full_metric_trajectory(self, inputs, initial, mode):
        """Shared real block scan; dense metric factors stay outside the recurrence."""
        a, b, c, d = self.compute_block_matrices()
        drives = F.linear(inputs, b)
        if mode == "scan":
            states = compute_block_lti_states(a, drives.transpose(0, 1), initial).transpose(0, 1)
        else:
            current, trajectory = initial, [initial]
            for drive in drives.unbind(1):
                current = block_left_multiply(a, current.mT).mT + drive
                trajectory.append(current)
            states = torch.stack(trajectory, dim=1)
        outputs = F.linear(states[:, :-1], c) + F.linear(inputs, d)
        return outputs, states

    # ----------------------------------------------------------------------
    # Sequence evaluation: reference loop, parallel scan, and FFT convolution
    # ----------------------------------------------------------------------
    def forward(
        self,
        u: torch.Tensor,
        state: torch.Tensor | None = None,  # x0 in x-basis
        *,
        time_first: bool = False,
        return_state: bool = True,  # if True return x_seq (x0..x_{T+1})
        mode: str = "scan",  # "loop" or "scan"
        return_last: bool = False,  # if True return x_last (= x_{T+1})
        reset_state: bool = True,
        detach_state: bool = False,
    ):
        """
        Input:
            u contains (u_0,...,u_T)  -> length L = T+1

        Returns:
            y_seq: (B,L,dy) (or (L,B,dy) if time_first)          [y_0..y_T]
            x_seq: (B,L+1,dx) (or (L+1,B,dx) if time_first)      [x_0..x_{T+1}]   if return_state
            x_last: (B,dx)                                      [x_{T+1}]        if return_last

        Design:
            - scan: stays in z-basis exactly as before (fast)
            - loop: runs recurrence directly in x-basis to match step-by-step feeding x_last
        """
        if mode not in ("scan", "conv", "loop", "loop_efficient"):
            raise ValueError("mode must be 'scan', 'conv', 'loop', or 'loop_efficient'.")
        if self.state_metric == "full" and mode == "conv":
            raise ValueError("Full-metric l2n uses general real blocks; choose mode='scan' or 'loop'.")
        u_bt = normalize_to_3d(u, time_first=time_first)
        if time_first:
            u_bt = u_bt.contiguous()
        B_sz, L, du = u_bt.shape
        if du != self.d_input:
            raise ValueError(f"Expected d_input={self.d_input}, got {du}.")
        dx = self.d_state

        # ---- initial x0 in x-basis ----
        x0 = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=B_sz,
            n_state=dx,
            device=u_bt.device,
            dtype=u_bt.dtype,
            x0=self.x0_param,
        )

        # S used for x<->z
        S = self.S.to(device=u_bt.device, dtype=u_bt.dtype) if self.S is not None else None

        if self.state_metric == "full":
            y_seq, x_seq = self._full_metric_trajectory(u_bt, x0, mode)
            x_last = x_seq[:, -1]
            if not return_state:
                x_seq = None
        elif mode not in ("scan", "conv"):
            # Row convention:
            #   z = x @ S^T
            #   z_{t+1} = z_t @ A_z^T + u_t @ B_z^T
            #   y_t = z_t @ C_z^T + u_t @ D_z^T
            #
            # Convert to x-dynamics:
            #   x_{t+1} = x_t @ (S^T A_z^T (S^T)^{-1}) + u_t @ (B_z^T (S^T)^{-1})
            #   y_t     = x_t @ (S^T C_z^T) + u_t @ D_z^T
            A, B, C, D, _ = self.compute_dense_matrices()
            A_x, B_x, C_x, D_x = A.T, B.T, C.T, D.T

            y_seq = u_bt.new_empty(B_sz, L, self.d_output)
            x_seq = u_bt.new_empty(B_sz, L + 1, dx) if return_state else None

            x = x0
            if return_state:
                x_seq[:, 0, :] = x

            for t in range(L):
                u_t = u_bt[:, t, :]  # (B,du)
                y_seq[:, t, :] = x @ C_x + u_t @ D_x
                x = x @ A_x + u_t @ B_x
                if return_state:
                    x_seq[:, t + 1, :] = x

            x_last = x

        else:
            # ==========================================================
            # SCAN / CONV (in z-basis) — identical maths, different engine
            # ==========================================================
            A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()
            z0 = x0 @ S.T if S is not None else x0  # (B,dx)

            # scan expects time-major (L,B,du)
            u_tb = u_bt.transpose(0, 1).contiguous()  # (L,B,du)

            # states: (L+1,B,dx) = z_0..z_{T+1}
            solver = "conv" if mode == "conv" else "scan"
            states = compute_linear_recurrence_parallel_block2x2(A_z, B_z, u_tb, z0, solver=solver)

            # batch-first
            z_seq = states.transpose(0, 1).contiguous()  # (B,L+1,dx)
            z_last = z_seq[:, -1, :]  # (B,dx)

            # outputs use z_t for t=0..T (exclude last state)
            y_seq = z_seq[:, :-1, :] @ C_z.T + u_bt @ D_z.T  # (B,L,dy)

            # Convert states to x-basis only if needed
            if return_state:
                z_flat_T = z_seq.reshape(B_sz * (L + 1), dx).T
                x_flat = self._solve_state_basis(z_flat_T, S).T
                x_seq = x_flat.reshape(B_sz, L + 1, dx)
                x_last = x_seq[:, -1, :]
            else:
                x_seq = None
                x_last = self._solve_state_basis(z_last.T, S).T

        self.state = x_last.detach() if detach_state else x_last

        # ---- restore time_first ----
        if time_first:
            y_seq = y_seq.transpose(0, 1).contiguous()  # (L,B,dy)
            if return_state:
                x_seq = x_seq.transpose(0, 1).contiguous()  # (L+1,B,dx)

        return sequence_output(
            y_seq, x_seq, x_last, return_state=return_state, return_last=return_last
        )

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)
