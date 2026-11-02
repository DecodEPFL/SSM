"""Legacy complex ZAK construction, retained for comparisons (param="zak")."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils.runtime import (
    EvalCacheMixin,
    normalize_to_3d as _normalize_to_3d,
    reset_runtime_state as _reset_runtime_state,
    resolve_runtime_state as _resolve_runtime_state,
)
from ..common import _scan_diag_linear, complex_diagonal_to_real, lru_forward_loop


class lruz(EvalCacheMixin, nn.Module):
    """LRU (complex-diagonal ZAK-style parametrization) with fixed or learnable l2-gain bound gamma.

    This version uses a learning-oriented default initialization:
      - poles initialized close to the unit circle for long memory,
      - small initial phases,
      - less conservative D/Y margins,
      - larger off-diagonal template scale.

    The certification step still follows the corrected Schur-complement scaling:
        || Lw^{-1} Y Lz^{-H} ||_2 < 1
    with
        W = [[P, PA], [A^* P, P]]
        Z = [[gamma I, D^T], [D, gamma I]]
        Y = [[P B, 0], [0, C^*]]
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        state_features: int,
        rmin: float = 0.96,
        rmax: float = 0.999,
        max_phase: float = 0.1,
        gamma: float = None,
        d_margin: float = 0.05,
        x2_margin: float = 0.9,
        x2_init_scale: float = 0.6,
        init: str = "eye",
        learn_x0: bool = False,
    ):
        super().__init__()
        self.state_features = state_features
        self.input_features = input_features
        self.output_features = output_features
        self.d_margin = float(d_margin)
        self.x2_margin = float(x2_margin)
        self.x2_init_scale = float(x2_init_scale)
        self.init = str(init)
        self.rmin = float(rmin)
        self.rmax = float(rmax)
        self.max_phase = float(max_phase)

        if not (0.0 < rmin < rmax < 1.0):
            raise ValueError(f"Expected 0 < rmin < rmax < 1, got rmin={rmin}, rmax={rmax}.")
        if not (max_phase > 0.0):
            raise ValueError(f"Expected max_phase > 0, got {max_phase}.")

        if self.init == "eye":
            # Structured long-memory prior:
            # poles clustered very close to the unit circle with a small phase spread.
            radius_hi = min(rmax, 0.999)
            radius_lo = max(rmin, radius_hi - 0.02)
            if radius_hi <= radius_lo + 1e-8:
                radius0 = torch.full((state_features,), radius_hi)
            else:
                radius0 = torch.linspace(radius_hi, radius_lo, state_features)
        elif self.init == "rand":
            # Radius initialization: sample |lambda| uniformly in [rmin, rmax]
            # and map to nu_log so that exp(-exp(nu_log)) = |lambda|.
            u_r = torch.rand(state_features)
            radius0 = rmin + (rmax - rmin) * u_r
        else:
            raise ValueError(f"Unknown lruz init: {init}")
        self.nu_log = nn.Parameter(torch.log(-torch.log(radius0)))

        if self.init == "eye":
            phase_hi = min(max_phase, 0.1)
            phase_lo = min(1e-3, phase_hi)
            if phase_hi <= phase_lo + 1e-12:
                phase0 = torch.full((state_features,), max(phase_hi, 1e-6))
            else:
                phase0 = torch.linspace(phase_lo, phase_hi, state_features)
        else:
            # Phase initialization: small positive phases in (0, max_phase)
            u_theta = torch.rand(state_features)
            phase0 = max_phase * u_theta + 1e-6
        self.theta_log = nn.Parameter(torch.log(phase0))

        # Fixed gamma: exact user value
        # Learnable gamma: unconstrained raw parameter, mapped through softplus
        if gamma is not None:
            self.register_buffer("gamma_fixed", torch.tensor(float(gamma)))
            self.gamma_raw = None
        else:
            self.register_buffer("gamma_fixed", None)
            init_gamma = 2.2
            self.gamma_raw = nn.Parameter(torch.tensor(float(init_gamma)))

        self.state: torch.Tensor | None = None

        self.register_buffer("ID", torch.eye(state_features))
        self.register_buffer("IDu", torch.eye(input_features))
        self.register_buffer("IDy", torch.eye(output_features))
        self.register_buffer("Inu", torch.ones((state_features, input_features)))
        self.register_buffer("Iny", torch.ones((state_features, output_features)))
        self.register_buffer("Znu", torch.zeros((state_features, input_features)))
        self.register_buffer("Zny", torch.zeros((state_features, output_features)))

        # Free real block for the structured Y matrix.
        if self.init == "eye":
            x2b_init = torch.zeros(2 * state_features, input_features + output_features)
            diag_u = min(state_features, input_features)
            diag_y = min(state_features, output_features)
            if diag_u > 0:
                x2b_init[torch.arange(diag_u), torch.arange(diag_u)] = self.x2_init_scale
            if diag_y > 0:
                x2b_init[
                    state_features + torch.arange(diag_y), input_features + torch.arange(diag_y)
                ] = self.x2_init_scale
        else:
            x2b_init = self.x2_init_scale * torch.randn(
                2 * state_features, input_features + output_features
            )
        self.X2b = nn.Parameter(x2b_init)

        # Free real D template.
        if self.init == "eye":
            dp_init = torch.zeros(output_features, input_features)
            diag_d = min(output_features, input_features)
            if diag_d > 0:
                dp_init[torch.arange(diag_d), torch.arange(diag_d)] = 1.0
        else:
            dp_init = torch.randn(output_features, input_features)
        self.Dp = nn.Parameter(dp_init)

        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, state_features, dtype=torch.complex64))
        else:
            self.register_buffer("x0_param", None)

        self._T_cache: dict[str, torch.Tensor] = {}

        # Runtime SSM params initialization
        self.set_param()

    def _current_gamma(self, eps: float = 1e-6) -> torch.Tensor:
        """Return a strictly positive gamma."""
        if self.gamma_fixed is not None:
            return self.gamma_fixed.clamp_min(eps)
        return F.softplus(self.gamma_raw) + eps

    @property
    def gamma(self) -> torch.Tensor:
        """Backward-compatible public gamma handle used elsewhere in the codebase."""
        return self._current_gamma()

    @staticmethod
    def _right_solve_upper_conj_transpose(Y: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Compute Y @ L^{-H} for lower-triangular L by solving on the transposed system."""
        return torch.linalg.solve_triangular(L, Y.conj().mT, upper=False, left=True).conj().mT

    def _balanced_offdiag_norm(
        self, W: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Return ||Lw^{-1} Y Lz^{-H}||_2."""
        Lw = torch.linalg.cholesky(W)
        Lz = torch.linalg.cholesky(Z)

        Y_left = torch.linalg.solve_triangular(Lw, Y, upper=False, left=True)
        Y_bal = self._right_solve_upper_conj_transpose(Y_left, Lz)

        return torch.linalg.matrix_norm(Y_bal, ord=2)

    def ss_real_matrices(self, to_numpy: bool = True):
        """Export the equivalent real state-space matrices."""
        A, B, C, D = self.set_param()
        lambdas = torch.diagonal(A)
        D = D.real
        return complex_diagonal_to_real(lambdas, B, C, D, self._T_cache, to_numpy=to_numpy)

    def set_param(self):
        # The certified (A, B, C, D) require Cholesky-balanced spectral norms;
        # cache them in eval (fixed weights), recompute in training.
        A, B, C, D = self._eval_cached("ssm", self._compute_ssm)
        self.A, self.B, self.C, self.D = A, B, C, D
        return A, B, C, D

    def _compute_ssm(self):
        nx, nu = self.state_features, self.input_features
        eps = 1e-2

        gamma = self._current_gamma(eps=1e-6).to(self.ID.dtype)
        alpha_d = min(max(self.d_margin, 1e-6), 1.0 - 1e-6)
        alpha_y = min(max(self.x2_margin, 1e-6), 1.0 - 1e-6)

        # A = diag(lambda_j), with |lambda_j| < 1
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        A = torch.diag(lambda_abs * torch.exp(1j * lambda_phase))

        # P = A^* A + eps I
        P = A.conj().mT @ A + eps * self.ID.to(dtype=A.dtype, device=A.device)

        # W = [[P, PA], [A^* P, P]]
        W_top = torch.cat((P, P @ A), dim=1)
        W_bot = torch.cat((A.conj().mT @ P, P), dim=1)
        W = torch.cat((W_top, W_bot), dim=0)

        # D = gamma * alpha_d * Dp / (||Dp|| + eps), so ||D|| < gamma
        Dp = self.Dp.to(dtype=self.ID.dtype, device=A.device)
        dp_norm = torch.linalg.matrix_norm(Dp, ord=2)
        D = gamma * alpha_d * Dp / (dp_norm + eps)

        # Z = [[gamma I, D^T], [D, gamma I]]
        Z_top = torch.cat((gamma * self.IDu.to(D.dtype).to(D.device), D.mT), dim=1)
        Z_bot = torch.cat((D, gamma * self.IDy.to(D.dtype).to(D.device)), dim=1)
        Z = torch.cat((Z_top, Z_bot), dim=0).to(dtype=A.dtype)

        # Structured free Y template: [[Y21, 0], [0, Y22]]
        M1 = torch.cat((self.Inu, self.Zny), dim=1)
        M2 = torch.cat((self.Znu, self.Iny), dim=1)
        M = torch.cat((M1, M2), dim=0).to(device=A.device, dtype=self.X2b.dtype)

        Y_tilde = (self.X2b * M).to(dtype=A.dtype, device=A.device)

        # Correct Schur-complement scaling
        y_norm = self._balanced_offdiag_norm(W, Y_tilde, Z)
        eta = 1.0 + y_norm
        Y = alpha_y * Y_tilde / eta

        # Recover B and C from Y = [[P B, 0], [0, C^*]]
        Y21 = Y[:nx, :nu]
        Y22 = Y[nx:, nu:]

        B = torch.linalg.solve(P, Y21)
        C = Y22.conj().mT

        # Diagnostic intermediates (not cached; valid as long as params are fixed).
        self.P, self.W, self.Z, self.Y = P, W, Z, Y
        return A, B, C, D

    def forward_loop(
        self,
        input: torch.Tensor,
        state: torch.Tensor | None = None,
        set_param: bool = True,
        detach_state: bool = True,
    ):
        if set_param:
            self.set_param()
        lambdas = torch.diagonal(self.A)
        output, states = lru_forward_loop(input, state, lambdas, self.B, self.C, self.D)
        self.state = states[:, -1].detach() if detach_state else states[:, -1]
        return output, states

    @torch.compiler.disable
    def forward_scan(
        self,
        input: torch.Tensor,
        state: torch.Tensor | None = None,
        set_param: bool = True,
        detach_state: bool = True,
    ):
        A, B, C, D = self.set_param() if set_param else (self.A, self.B, self.C, self.D)
        lambdas = torch.diagonal(A)

        x0 = state.to(B.dtype)
        Bu = input.to(B.dtype) @ B.mT
        states = _scan_diag_linear(lambdas, Bu, x0)  # (B, L+1, N)

        self.state = states[:, -1, :].detach() if detach_state else states[:, -1, :]
        output = (states[:, :-1, :] @ C.mT).real + input @ D.mT
        return output, states

    def forward(
        self,
        input: torch.Tensor,
        gamma=None,
        state: torch.Tensor | None = None,
        set_param: bool = True,
        mode: str = "scan",
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        input = _normalize_to_3d(input)

        _cplx_dtype = (
            torch.complex64 if input.dtype in (torch.float16, torch.float32) else torch.complex128
        )

        self.state = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=input.shape[0],
            n_state=self.state_features,
            device=input.device,
            dtype=_cplx_dtype,
            x0=self.x0_param,
        )

        if mode == "scan":
            return self.forward_scan(input, self.state, set_param, detach_state=detach_state)
        if mode in ("loop", "loop_efficient"):
            return self.forward_loop(input, self.state, set_param, detach_state=detach_state)
        raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)
