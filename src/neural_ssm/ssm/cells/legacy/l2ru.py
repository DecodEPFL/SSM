"""Legacy L2RU certificate construction, retained for comparisons (param="l2ru")."""

from __future__ import annotations

import torch
import torch.nn as nn

from ....utils.runtime import (
    EvalCacheMixin,
    normalize_to_3d as _normalize_to_3d,
    reset_runtime_state as _reset_runtime_state,
    resolve_runtime_state as _resolve_runtime_state,
)
from ....utils.scan import compute_linear_recurrence_parallel_scan
from ..common import lru_forward_loop


class L2RU(EvalCacheMixin, nn.Module):
    """LRU with learnable or fixed l2 gain gamma."""

    def __init__(
        self,
        state_features: int,
        gamma: float = None,
        init: str = "eye",
        q: int = 1,
        eye_scale=0.01,
        rand_scale=1,
        learn_x0: bool = False,
    ):
        super().__init__()
        self.state_features = state_features
        if gamma is not None:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        else:
            self.gamma = nn.Parameter(torch.tensor(2.2))

        self.register_buffer("ID", torch.eye(state_features))
        self.alpha = nn.Parameter(torch.tensor(4.1))
        self.register_buffer("epsilon", torch.tensor(-0.0))
        self.q = q

        # Precompute triangle indices
        self.register_buffer(
            "triu_indices", torch.triu_indices(state_features, state_features, offset=1)
        )
        self.register_buffer(
            "tril_indices", torch.tril_indices(state_features, state_features, offset=0)
        )

        n = state_features
        if init == "eye":
            X11_full = eye_scale * torch.eye(n)
            X22_full = eye_scale * torch.eye(n)
            X21_init = 0.1 * torch.eye(n)
        elif init == "rand":
            X11_full = rand_scale * torch.randn(n, n)
            X22_full = rand_scale * torch.randn(n, n)
            X21_init = rand_scale * torch.randn(n, n)
        else:
            raise ValueError(init)

        self.X11_params = nn.Parameter(X11_full[self.tril_indices[0], self.tril_indices[1]])
        self.X22_params = nn.Parameter(X22_full[self.tril_indices[0], self.tril_indices[1]])

        if q == 1:
            Skew_init = 0.01 * torch.randn(n, n)
            Skew_init = Skew_init - Skew_init.T
            Skew_params = Skew_init[self.triu_indices[0], self.triu_indices[1]]
            self.Skew_params = nn.Parameter(Skew_params)

        self.X21 = nn.Parameter(X21_init)
        self.C = nn.Parameter(torch.eye(state_features))
        self.Dt = nn.Parameter(torch.eye(state_features))

        # Runtime LTI
        self.state: torch.Tensor | None = None

        # Learnable initial condition (real, shape (1, N))
        if learn_x0:
            self.x0_param = nn.Parameter(torch.zeros(1, state_features))
        else:
            self.register_buffer("x0_param", None)

        self.set_param()

    def _get_lower_triangular(self, params: torch.Tensor) -> torch.Tensor:
        L = torch.zeros(
            self.state_features, self.state_features, device=params.device, dtype=params.dtype
        )
        L[self.tril_indices[0], self.tril_indices[1]] = params
        return L

    def _get_skew_symmetric(self, params: torch.Tensor) -> torch.Tensor:
        Sk = torch.zeros(
            self.state_features, self.state_features, device=params.device, dtype=params.dtype
        )
        Sk[self.triu_indices[0], self.triu_indices[1]] = params
        Sk[self.triu_indices[1], self.triu_indices[0]] = -params
        return Sk

    def set_param(self):
        # The certified (A, B, D) require an SVD + two Cholesky factorizations +
        # solves; cache them in eval (fixed weights), recompute in training.
        A, B, C, D = self._eval_cached("ssm", self._compute_ssm)
        self.A, self.B, self.D = A, B, D
        return A, B, C, D

    def _compute_ssm(self):
        ID = self.ID

        X11 = self._get_lower_triangular(self.X11_params)
        X22 = self._get_lower_triangular(self.X22_params)

        if self.q == 1:
            Sk = self._get_skew_symmetric(self.Skew_params)
            # Right solve implements the Cayley transform without an inverse.
            Qm = torch.linalg.solve((ID + Sk).T, (ID - Sk).T).T
        else:
            Qm = ID

        gamma = self.gamma
        Z = self.X21 @ self.X21.T + X22 @ X22.T + self.Dt.T @ self.Dt + torch.exp(self.epsilon) * ID
        beta = gamma**2 * torch.sigmoid(self.alpha) / torch.linalg.matrix_norm(Z, 2)

        H11 = X11 @ X11.T + self.C.T @ self.C + beta * torch.exp(self.epsilon) * ID
        H12 = torch.sqrt(beta) * (X11 @ self.X21.T + self.C.T @ self.Dt)
        V = Z * beta - gamma**2 * ID

        # Safer solves and light symmetrization
        S = torch.linalg.solve(V.T, H12.T)  # solves V^T X = H12^T
        R = H12 @ S
        R = 0.5 * (R + R.T)

        negR = -R + 1e-6 * ID
        try:
            CR = torch.linalg.cholesky(negR)
            CRH = torch.linalg.cholesky(negR + H11)
        except RuntimeError as exc:
            raise RuntimeError(
                "L2RU.set_param: Cholesky decomposition failed — the LMI constraint may be "
                "violated. Try reducing gamma or allowing alpha to adjust further."
            ) from exc

        A = torch.linalg.solve_triangular(CRH.T, Qm @ CR.T, upper=True)
        Xsolve = torch.linalg.solve(H12.T, V.T)  # (H12^T) X = V^T
        B = A @ Xsolve
        C = self.C
        D = torch.sqrt(beta) * self.Dt

        return A, B, C, D

    def forward(
        self,
        input: torch.Tensor,
        state: torch.Tensor | None = None,
        set_param: bool = True,
        mode: str = "scan",
        reset_state: bool = True,
        detach_state: bool = True,
    ):
        input = _normalize_to_3d(input)
        # real-valued state for L2RU
        self.state = _resolve_runtime_state(
            explicit_state=state,
            internal_state=self.state,
            reset_state=reset_state,
            batch_size=input.shape[0],
            n_state=self.state_features,
            device=input.device,
            dtype=input.dtype,
            x0=self.x0_param,
        )

        x0 = self.state
        if set_param:
            self.set_param()

        if mode == "scan":
            u = input.permute(1, 0, 2)  # (L,B,H) == (T,B,D)
            states = compute_linear_recurrence_parallel_scan(self.A, self.B, u, x0).transpose(0, 1)
            self.state = states[:, -1, :].detach() if detach_state else states[:, -1, :]
            outputs = states[:, :-1, :] @ self.C.transpose(-1, -2) + input @ self.D.transpose(
                -1, -2
            )
            return outputs, states
        elif mode in ("loop", "loop_efficient"):
            output, states = lru_forward_loop(input, self.state, self.A, self.B, self.C, self.D)
            self.state = states[:, -1].detach() if detach_state else states[:, -1]
            return output, states
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = _reset_runtime_state(self.state, x0=self.x0_param)
