"""Legacy dense normalized LTI construction (param="l2nt")."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....utils.runtime import (
    EvalCacheMixin,
    normalize_to_3d as _normalize_to_3d,
    reset_runtime_state as _reset_runtime_state,
    resolve_runtime_state as _resolve_runtime_state,
)
from ..common import _CONTRACTION_EPS, bounded_real_matrix as _bounded_real_matrix, lru_forward_loop


class L2BoundedLTICell(EvalCacheMixin, nn.Module):
    """
    Dense LTI cell with *hard* L2-gain bound via:

        [ S x_{k+1} ]   [ K11 K12 ] [ S x_k   ]
        [   y_k     ] = [ K21 K22 ] [ gamma u_k ]

    with ||K||_2 <= 1 (enforced via SVD).
    """

    def __init__(
        self, d_state, d_input, d_output, gamma=1.0, train_gamma=False, learn_x0: bool = False
    ):
        super().__init__()
        self.d_state = d_state
        self.d_input = d_input
        self.d_output = d_output
        self.state: torch.Tensor | None = None

        # S: energy transform (invertible with prob. 1)
        self.S = nn.Parameter(0.1 * torch.randn(d_state, d_state))

        # Raw K (we'll spectral-normalize it)
        self.K_raw = nn.Parameter(1 * torch.randn(d_state + d_output, d_state + d_input))

        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())

        # Learnable initial condition (real, shape (1, d_state))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, d_state))
        else:
            self.register_buffer("x0_param", None)

        eigvals_target = 0.98 * torch.ones(d_state, dtype=torch.float64)
        self.init_orthogonal_spectrum(eigvals_target, offdiag_scale=0.8)

    @property
    def gamma(self):
        return self.log_gamma.exp()

    # ---- enforce K contraction exactly via SVD ----
    def _build_contraction(self):
        K_raw = self.K_raw
        # exact largest singular value
        sigma = torch.linalg.matrix_norm(K_raw, ord=2)
        sigma = sigma.clamp(min=1e-5)
        K = K_raw / (sigma + _CONTRACTION_EPS)
        return K

    # ---- map (S,K) -> (A,B,C,D,P) correctly ----
    def compute_ssm_matrices(self):
        # Exact spectral contraction and state-basis solves; cache in eval.
        return self._eval_cached("ssm", self._compute_ssm_matrices)

    def _compute_ssm_matrices(self):
        d_x = self.d_state
        d_u = self.d_input

        S = self.S
        gamma = self.gamma
        K = self._build_contraction()  # (d_x + d_y, d_x + d_u)

        K11 = K[:d_x, :d_x]
        K12 = K[:d_x, d_x:]
        K21 = K[d_x:, :d_x]
        K22 = K[d_x:, d_x:]

        # S A = K11 S     -> A = S^{-1} K11 S
        # S B = γ K12     -> B = γ S^{-1} K12
        # C   = K21 S
        # D   = γ K22
        # Solve both right-hand sides together; do not form a dense inverse.
        AB = torch.linalg.solve(S, torch.cat((K11 @ S, gamma * K12), dim=1))
        A, B = AB.split((d_x, d_u), dim=1)
        C = K21 @ S
        D = gamma * K22

        P = S.T @ S

        return A, B, C, D, P

    def bounded_real_matrix(self, gamma=None):
        """Return the discrete bounded-real LMI in the physical state basis."""
        A, B, C, D, P = self.compute_ssm_matrices()
        return _bounded_real_matrix(A, B, C, D, P, self.gamma if gamma is None else gamma)

    # ---------- initialization with eig(A) ≈ eigvals ----------
    @torch.no_grad()
    def init_orthogonal_spectrum(
        self,
        eigvals: torch.Tensor,
        offdiag_scale: float = 0.5,
    ):
        """Initialize an orthogonal spectral template with input/output coupling.

        ``eigvals`` specifies the raw K11 spectrum. Normalizing the complete K
        to a contraction can reduce these radii; the requested spectrum is not
        an exact promise for the final A. The historical initialization keeps
        a second, downward-only radius adjustment for checkpoint reproducibility.
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

        # 5) Cap K11's radius at the target using downward scaling only.
        with torch.no_grad():
            # recompute K11 block after step 4
            K11_scaled = K_full[:d_x, :d_x]
            ev_K11 = torch.linalg.eigvals(K11_scaled)
            rho_current = ev_K11.abs().max().item()
        rho_target = eigvals.abs().max().item()
        if rho_current > 0:
            scale2 = min(1.0, rho_target / rho_current)
            K_full *= scale2

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
        state: torch.Tensor | None = None,
        *,
        mode: str = None,
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        u = _normalize_to_3d(u)
        state = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=u.shape[0],
            n_state=self.d_state,
            device=u.device,
            dtype=u.dtype,
            x0=self.x0_param,
        )

        A, B, C, D, _ = self.compute_ssm_matrices()
        output, states = lru_forward_loop(u, state, A, B, C, D)
        self.state = states[:, -1, :].detach() if detach_state else states[:, -1, :]
        return output, states

    def forward_original(
        self,
        u: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        time_first: bool = False,
        return_state: bool = True,
        mode: str | None = None,
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        """Compatibility interface returning pre-step states, without x_T.

        The canonical forward returns x_0..x_T; this historical interface returns
        x_0..x_{T-1}, or x_T when return_state=False. Both use the same kernel.
        """
        inputs = _normalize_to_3d(u, time_first=time_first)
        if inputs.shape[-1] != self.d_input:
            raise ValueError(f"Expected d_input={self.d_input}, got {inputs.shape[-1]}.")
        output, states = self.forward(
            inputs, state, mode=mode, reset_state=reset_state, detach_state=detach_state
        )
        last = states[:, -1]
        pre_states = states[:, :-1]
        if time_first:
            output, pre_states = output.transpose(0, 1), pre_states.transpose(0, 1)
        return output, pre_states if return_state else last

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)
