# python
from __future__ import annotations

import math
from typing import TypedDict, Dict
from torch import Tensor
from dataclasses import fields
from .scan_utils import *
from ..static_layers.generic_layers import *
from ..static_layers.lipschitz_mlps import *
from .experimental import ExpertSelectiveTimeVaryingSSM, Block2x2SelectiveBCDExpertsL2SSM


# --------- Small utilities (DRY helpers) ---------

def _normalize_to_3d(x: torch.Tensor) -> torch.Tensor:
    # Returns (B, L, H)
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 2:
        return x[None, :, :]
    if x.dim() == 3:
        return x
    raise ValueError(f"Invalid input dimensions {x.dim()}, expected 1, 2, or 3.")


def _init_or_cast_state(
        state: Optional[torch.Tensor],
        batch_size: int,
        n_state: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    if state is not None:
        return state.to(device=device, dtype=dtype)
    return torch.zeros(batch_size, n_state, device=device, dtype=dtype)


def _scan_diag_linear(
        lambdas: torch.Tensor,  # (N,)
        Bu: torch.Tensor,  # (B, L, N) = B u_t already
        x0: torch.Tensor,  # (B, N)
) -> torch.Tensor:
    """
    Diagonal linear recurrence via parallel scan:

        x_{t+1} = lambdas * x_t + Bu[:, t]

    Args:
        lambdas: (N,)
        Bu:      (B, L, N)  precomputed B @ u_t
        x0:      (B, N)     initial state x_0

    Returns:
        states:  (B, L+1, N) with
                 states[:, 0]   = x_0
                 states[:, t+1] = x_{t+1} for t = 0..L-1
    """
    Bsz, L, N = Bu.shape
    Bu = Bu.clone()
    x0 = x0.squeeze(1)
    # fold x0 into the first step
    Bu[:, 0, :] += lambdas * x0

    lam_seq = lambdas.expand(L, -1)  # (L, N)

    def _scan_fn(bu_seq):
        # returns sequence x_1..x_L, shape (L, N)
        return associative_scan(binary_operator_diag, (lam_seq, bu_seq))[1]

    x_next = torch.vmap(_scan_fn)(Bu)  # (B, L, N): x_1..x_L

    # assemble full trajectory [x_0, ..., x_L]
    states = torch.empty(Bsz, L + 1, N, device=Bu.device, dtype=Bu.dtype)
    states[:, 0] = x0
    states[:, 1:] = x_next
    return states


@torch.jit.script
def lru_forward_loop(
        input: Tensor,
        state: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Sequential state-space recurrence (loop version).

    Recurrence:
        x_{t+1} = A x_t + B u_t
        y_t     = Re(C x_t) + D u_t

    Supports:
        - A: (N,)   -> diagonal (elementwise multiplication)
        - A: (N,N)  -> full constant matrix

    Args:
        input:  (B, L, H)
        state:  (B, N)        initial state x_0
        A:      (N,) or (N,N) state transition
        B:      (N, H)
        C:      (H_out, N)
        D:      (H_out, H)

    Returns:
        output: (B, L, H_out)   y_t = Re(C x_t) + D u_t
        states: (B, L+1, N)     full trajectory [x_0, ..., x_L]
    """
    BATCH, SEQ, H = input.shape
    N = state.size(1)

    # Basic shape sanity checks
    assert state.size(0) == BATCH
    assert B.size(0) == N and B.size(1) == H
    assert C.size(1) == N
    assert D.size(1) == H

    # Use input's dtype/device as reference
    dtype = input.dtype
    device = input.device

    # Cast everything once, up front (no per-step .to calls)
    state = state.to(dtype=dtype, device=device)
    B = B.to(dtype=dtype, device=device)
    C = C.to(dtype=dtype, device=device)
    D = D.to(dtype=dtype, device=device)

    # A might be complex in LRU; keep dtype consistent but don't force device move
    A = A.to(dtype=dtype)

    # Precompute transposes (matrix-layout friendly)
    BT = B.mT  # (H, N)
    CT = C.mT  # (N, H_out)
    DT = D.mT  # (H, H_out)

    # Allocate full trajectory [x_0, ..., x_L]
    states = torch.empty(
        (BATCH, SEQ + 1, N),
        device=device,
        dtype=dtype,
    )
    x = state  # (B, N)
    states[:, 0] = x

    if A.dim() == 1:
        # Diagonal A (vector of lambdas)
        lambdas = A  # (N,)
        for t in range(SEQ):
            u_t = input[:, t, :]  # (B, H)
            # x = lambdas * x + u_t @ BT
            x = x * lambdas  # broadcasts over batch
            x = x + u_t @ BT  # (B, N)
            states[:, t + 1] = x
    elif A.dim() == 2:
        # Full constant A
        A_T = A.mT  # (N, N)
        for t in range(SEQ):
            u_t = input[:, t, :]  # (B, H)
            # x = x @ A_T + u_t @ BT
            x = x @ A_T  # (B, N)
            x = x + u_t @ BT  # (B, N)
            states[:, t + 1] = x
    else:
        # TorchScript only supports RuntimeError, not ValueError etc.
        raise RuntimeError("Unsupported A.dim(), expected 1 or 2")

    # pre-update states for output: x_t
    pre_states = states[:, :-1, :]  # (B, L, N)

    # Vectorized output computation over time
    # (B, L, N) @ (N, H_out) --> (B, L, H_out)
    output = (pre_states @ CT).real + input @ DT

    return output, states


def _complex_real_transform_blocks(
        n: int,
        dtype: torch.dtype,
        device: torch.device,
        cache: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Cache 2x2 block and its inverse per dtype/device to avoid reallocation
    key = f"{str(dtype)}@{device.type}:{device.index}"
    T_key, Ti_key = f"T_{key}", f"Tinv_{key}"
    if T_key not in cache or Ti_key not in cache:
        T = torch.tensor([[1, 1], [1j, -1j]], device=device, dtype=dtype)
        cache[T_key] = T
        cache[Ti_key] = torch.linalg.inv(T)
    Tblk = torch.block_diag(*([cache[T_key]] * n))
    Tiblk = torch.block_diag(*([cache[Ti_key]] * n))
    return Tblk, Tiblk


""" Linear Recurrent Units ----------------------------------------- """


# python
class LRU(nn.Module):
    """Linear Recurrent Unit with loop or parallel-scan simulation."""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            state_features: int,
            internal_state_init=None,
            rmin: float = 0.9,
            rmax: float = 1.0,
            max_phase: float = 6.283,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        # Pre-compute scalars
        self._sqrt_in_features = math.sqrt(in_features)
        self._sqrt_2_in_features = math.sqrt(2 * in_features)
        self._sqrt_state_features = math.sqrt(state_features)
        self._rmin_rmax_diff = rmax - rmin
        self._rmin_rmax_sum = rmax + rmin
        self._rmin_squared = rmin ** 2

        # Real output projection
        self.D = nn.Parameter(torch.randn(out_features, in_features) / self._sqrt_in_features)

        # Complex SSM params (diagonal A via magnitudes+phases)
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * self._rmin_rmax_sum * self._rmin_rmax_diff + self._rmin_squared))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1.0 - lambda_abs.square())))

        B_complex = torch.complex(
            torch.randn(state_features, in_features) / self._sqrt_2_in_features,
            torch.randn(state_features, in_features) / self._sqrt_2_in_features,
        )
        self.Bp = nn.Parameter(B_complex)  # (N, U)

        C_complex = torch.complex(
            torch.randn(out_features, state_features) / self._sqrt_state_features,
            torch.randn(out_features, state_features) / self._sqrt_state_features,
        )
        self.C = nn.Parameter(C_complex)  # (H, N)

        # Runtime state
        self.state: Optional[torch.Tensor] = None

        # Small cache for complex->real transform 2x2 blocks
        self._T_cache: Dict[str, torch.Tensor] = {}

        self.set_param()  # initialize SSM params

    def set_param(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        self.lambdas = lambda_abs * torch.exp(1j * lambda_phase)  # (N,) complex
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)  # (N,1) real
        self.B = gammas * self.Bp  # (N,U) complex
        return self.lambdas, self.B, self.C, self.D

    def ss_real_matrices(self, to_numpy: bool = True):
        lambdas, B, C, D = self.set_param()
        device, dtype = lambdas.device, lambdas.dtype
        n2 = 2 * self.state_features

        lambdas_conj = torch.stack([lambdas, lambdas.conj()], dim=1).flatten()
        A_full = torch.diag(lambdas_conj)  # (2N,2N)
        B_full = torch.stack([B, B.conj()], dim=1).view(n2, self.in_features)
        C_half = 0.5 * C
        C_full = torch.stack([C_half, C_half.conj()], dim=2).view(self.out_features, n2)

        T, Tinv = _complex_real_transform_blocks(self.state_features, dtype, device, self._T_cache)
        A_real = (T @ A_full @ Tinv).real
        B_real = (T @ B_full).real
        C_real = (C_full @ Tinv).real
        D_real = D

        mats = [A_real, B_real, C_real, D_real]
        if to_numpy:
            mats = [m.detach().cpu().numpy() for m in mats]
        return tuple(mats)

    def forward_loop(self, input: torch.Tensor, state: torch.Tensor):
        self.set_param()
        output, states = lru_forward_loop(input, state, self.lambdas, self.B, self.C, self.D)
        self.state = states[:, -1].detach()
        return output, states

    @torch.compiler.disable
    def forward_scan(self, input: torch.Tensor, state: Optional[torch.Tensor] = None):
        lambdas, B, C, D = self.set_param()

        x0 = state.to(B.dtype)
        Bu = input.to(B.dtype) @ B.mT
        # compute state trajectory [x_0, ..., x_L]
        states = _scan_diag_linear(lambdas, Bu, x0)  # (B, L+1, N)

        self.state = states[:, -1, :].detach()
        output = (states[:, :-1, :] @ C.mT).real + input @ D.T
        return output, states

    def forward(self, input: torch.Tensor, gamma: Optional[float] = None, state: Optional[torch.Tensor] = None,
                mode: str = "loop"):
        input = _normalize_to_3d(input)
        self.state = _init_or_cast_state(state, input.shape[0], self.state_features, input.device, self.B.dtype)
        if mode == "scan":
            return self.forward_scan(input, self.state)
        if mode in ("loop", "loop_efficient"):
            return self.forward_loop(input, self.state)
        raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = None


# python
class L2RU(nn.Module):
    """LRU with learnable or fixed l2 gain gamma."""

    def __init__(self, state_features: int, gamma: float = None, init: str = "eye", q: int = 1, eye_scale=0.01,
                 rand_scale=1):
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
        self.register_buffer("triu_indices", torch.triu_indices(state_features, state_features, offset=1))
        self.register_buffer("tril_indices", torch.tril_indices(state_features, state_features, offset=0))

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
        self.state: Optional[torch.Tensor] = None
        self.set_param()

    def _get_lower_triangular(self, params: torch.Tensor) -> torch.Tensor:
        L = torch.zeros(self.state_features, self.state_features, device=params.device, dtype=params.dtype)
        L[self.tril_indices[0], self.tril_indices[1]] = params
        return L

    def _get_skew_symmetric(self, params: torch.Tensor) -> torch.Tensor:
        Sk = torch.zeros(self.state_features, self.state_features, device=params.device, dtype=params.dtype)
        Sk[self.triu_indices[0], self.triu_indices[1]] = params
        Sk[self.triu_indices[1], self.triu_indices[0]] = -params
        return Sk

    def set_param(self):
        ID = self.ID
        n = self.state_features

        X11 = self._get_lower_triangular(self.X11_params)
        X22 = self._get_lower_triangular(self.X22_params)

        if self.q == 1:
            Sk = self._get_skew_symmetric(self.Skew_params)
            Qm = (ID - Sk) @ torch.linalg.inv(ID + Sk)
        else:
            Qm = ID

        gamma = self.gamma
        Z = self.X21 @ self.X21.T + X22 @ X22.T + self.Dt.T @ self.Dt + torch.exp(self.epsilon) * ID
        beta = gamma ** 2 * torch.sigmoid(self.alpha) / torch.linalg.matrix_norm(Z, 2)

        H11 = X11 @ X11.T + self.C.T @ self.C + beta * torch.exp(self.epsilon) * ID
        H12 = torch.sqrt(beta) * (X11 @ self.X21.T + self.C.T @ self.Dt)
        V = Z * beta - gamma ** 2 * ID

        # Safer solves and light symmetrization
        S = torch.linalg.solve(V.T, H12.T)  # solves V^T X = H12^T
        R = H12 @ S
        R = 0.5 * (R + R.T)

        negR = -R + 1e-6 * ID
        CR = torch.linalg.cholesky(negR)
        CRH = torch.linalg.cholesky(negR + H11)

        A = torch.linalg.inv(CRH).T @ Qm @ CR.T
        Xsolve = torch.linalg.solve(H12.T, V.T)  # (H12^T) X = V^T
        B = A @ Xsolve
        C = self.C
        D = torch.sqrt(beta) * self.Dt

        self.A, self.B, self.D = A, B, D
        return A, B, C, D

    def forward(self, input: torch.Tensor, state: Optional[torch.Tensor] = None, set_param: bool = True,
                mode: str = "scan"):
        input = _normalize_to_3d(input)
        # real-valued state for L2RU
        self.state = _init_or_cast_state(state, input.shape[0], self.state_features, input.device, input.dtype)

        x0 = self.state
        if set_param:
            self.set_param()

        if mode == "scan":
            u = input.permute(1, 0, 2)  # (L,B,H) == (T,B,D)
            states = compute_linear_recurrence_parallel_scan(self.A, self.B, u, x0).transpose(0, 1)
            self.state = states[:, -1, :].detach()
            outputs = states[:, :-1, :] @ self.C.transpose(-1, -2) + input @ self.D.transpose(-1, -2)
            return outputs, states
        elif mode in ("loop", "loop_efficient"):
            output, states = lru_forward_loop(input, self.state, self.A, self.B, self.C, self.D)
            self.state = states[:, -1].detach()
            return output, states
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = None


# python
class lruz(nn.Module):
    """LRU (ZAK parametrization) with learnable or fixed l2 gain gamma."""

    def __init__(self, input_features: int, output_features: int, state_features: int, rmin=0.9, rmax=1.0,
                 max_phase=6.283, gamma: float = None):
        super().__init__()
        self.state_features = state_features
        self.input_features = input_features
        self.output_features = output_features

        self._rmin_rmax_diff = rmax - rmin
        self._rmin_rmax_sum = rmax + rmin
        self._rmin_squared = rmin ** 2
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * self._rmin_rmax_sum * self._rmin_rmax_diff + self._rmin_squared))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        if gamma is not None:
            self.register_buffer("gamma", torch.tensor(float(gamma)))
        else:
            self.gamma = nn.Parameter(torch.tensor(2.2))

        self.state: Optional[torch.Tensor] = None
        self.register_buffer("ID", torch.eye(state_features))
        self.register_buffer("IDu", torch.eye(input_features))
        self.register_buffer("IDy", torch.eye(output_features))
        self.register_buffer("Inu", torch.ones((state_features, input_features)))
        self.register_buffer("Iny", torch.ones((state_features, output_features)))
        self.register_buffer("Znu", torch.zeros((state_features, input_features)))
        self.register_buffer("Zny", torch.zeros((state_features, output_features)))

        self.X2b = nn.Parameter(torch.randn(2 * state_features, input_features + output_features))
        self.Dp = nn.Parameter(torch.randn(output_features, input_features))

        # Runtime SSM params initialization
        self.set_param()

        # Small 2x2 transform cache like LRU for reuse
        self._T_cache: Dict[str, torch.Tensor] = {}

    def ss_real_matrices(self, to_numpy: bool = True):
        A, B, C, D = self.set_param()
        lambdas = torch.diagonal(A)
        device, dtype = lambdas.device, lambdas.dtype
        n2 = 2 * self.state_features

        lambdas_conj = torch.stack([lambdas, lambdas.conj()], dim=1).flatten()
        A_full = torch.diag(lambdas_conj)
        B_full = torch.stack([B, B.conj()], dim=1).view(n2, self.input_features)
        C_half = 0.5 * C
        C_full = torch.stack([C_half, C_half.conj()], dim=2).view(self.output_features, n2)

        T, Tinv = _complex_real_transform_blocks(self.state_features, dtype, device, self._T_cache)
        A_real = (T @ A_full @ Tinv).real
        B_real = (T @ B_full).real
        C_real = (C_full @ Tinv).real
        D_real = D

        mats = [A_real, B_real, C_real, D_real]
        if to_numpy:
            mats = [m.detach().cpu().numpy() for m in mats]
        return tuple(mats)

    def set_param(self):
        nx, nu, ny = self.state_features, self.input_features, self.output_features
        epsilon = 0.01
        alpha = 1 - epsilon

        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        A = torch.diag(lambda_abs * torch.exp(1j * lambda_phase))

        Q = torch.conj(A).T @ A + epsilon * self.ID

        X11 = torch.cat((Q, Q @ A), dim=1)
        X12 = torch.cat((torch.conj(A).T @ Q, Q), dim=1)
        X1 = torch.cat((X11, X12), dim=0)

        X4_off = self.gamma * alpha * self.Dp.T / torch.linalg.matrix_norm(self.Dp, 2)

        X4_r1 = torch.cat((self.gamma * self.IDu, X4_off), dim=1)
        X4_r2 = torch.cat((X4_off.T, self.gamma * self.IDy), dim=1)
        X4 = torch.cat((X4_r1, X4_r2), dim=0)

        M1 = torch.cat((self.Inu, self.Zny), dim=1)
        M2 = torch.cat((self.Znu, self.Iny), dim=1)
        M = torch.cat((M1, M2), dim=0)

        X2t = self.X2b * M

        # Norm-based scaling (move complex ops where needed)
        eta_1 = torch.linalg.matrix_norm(torch.linalg.inv(X1) @ X2t.to(torch.complex64), ord=2)
        eta_2 = torch.linalg.matrix_norm(X2t @ torch.linalg.inv(X4), ord=2)
        eta = torch.maximum(torch.maximum(eta_1, eta_2), torch.tensor(1.0, device=X2t.device))

        X2 = X2t / eta

        B = torch.linalg.inv(Q) @ X2[:nx, :nu].to(torch.complex64)
        C = torch.conj(X2[-nx:, -ny:]).T.to(torch.complex64)
        D = X4_off.T

        self.A, self.B, self.C, self.D = A, B, C, D
        return A, B, C, D

    def forward_loop(self, input: torch.Tensor, state: Optional[torch.Tensor] = None, set_param: bool = True):
        if set_param:
            self.set_param()
        lambdas = torch.diagonal(self.A)
        output, states = lru_forward_loop(input, state, lambdas, self.B, self.C, self.D)
        self.state = states[:, -1].detach()
        return output, states

    @torch.compiler.disable
    def forward_scan(self, input: torch.Tensor, state: Optional[torch.Tensor] = None, set_param: bool = True):
        BATCH, SEQ, _ = input.shape
        A, B, C, D = self.set_param() if set_param else (self.A, self.B, self.C, self.D)
        lambdas = torch.diagonal(A)

        x0 = state.to(B.dtype)
        Bu = input.to(B.dtype) @ B.mT
        # compute state trajectory [x_0, ..., x_L]
        states = _scan_diag_linear(lambdas, Bu, x0)  # (B, L+1, N)

        self.state = states[:, -1, :].detach()
        output = (states[:, :-1, :] @ C.mT).real + input @ D.T
        return output, states

    def forward(self, input: torch.Tensor, gamma=None, state: Optional[torch.Tensor] = None, set_param: bool = True,
                mode: str = "scan"):
        input = _normalize_to_3d(input)
        # complex-valued state for ZAK
        self.state = _init_or_cast_state(state, input.shape[0], self.state_features, input.device, torch.complex64)

        if mode == "scan":
            return self.forward_scan(input, self.state, set_param)
        if mode in ("loop", "loop_efficient"):
            return self.forward_loop(input, self.state, set_param)
        raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = None


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

        # ---------- initialization with eig(A) ≈ eigvals ----------

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
            state: torch.Tensor | None = None,
            *,
            mode: str = None
    ):

        # Prepare state
        if state is None:
            state = torch.zeros(u.shape[0], self.d_state, device=u.device, dtype=u.dtype)

        A, B, C, D, _ = self.compute_ssm_matrices()
        output, states = lru_forward_loop(u, state, A, B, C, D)
        return output, states
    def forward_original(
            self,
            u: torch.Tensor,
            state: torch.Tensor | None = None,
            *,
            time_first: bool = False,
            return_state: bool = True,
            mode: str = None
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
        state : (B, d_state) or (d_state,) or None (zero init).
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
        if state is None:
            x = u.new_zeros(B_sz, self.d_state)
        else:
            if state.dim() == 1:
                x = state.unsqueeze(0).expand(B_sz, -1)
            else:
                x = state
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
        exact_norm: bool = True,
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
        self.K12_raw = nn.Parameter(0.5 * torch.randn(d_state, d_input))
        self.K21_raw = nn.Parameter(0.5 * torch.randn(d_output, d_state))
        self.K22_raw = nn.Parameter(0.5 * torch.randn(d_output, d_input))

        # --- gamma (>0) ---
        g0 = torch.tensor(float(gamma))
        if train_gamma:
            self.log_gamma = nn.Parameter(g0.log())
        else:
            self.register_buffer("log_gamma", g0.log())


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
        # exact largest singular value
        sigma = torch.linalg.matrix_norm(M, ord=2)
        sigma = sigma.clamp(min=1e-5)
        M = M / (sigma + 0.002)
        return M

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
            offdiag_scale: float = .05,
    ):
        """
        Initialize such that A_z = K11 has eigenvalues

            λ_i^± ≈ rho * exp(± j θ_i),

        with |λ_i| ≈ rho (< 1) and θ_i controlled.

        Args
        ----
        rho : desired modulus of eigenvalues (must be < 1).
        max_phase : if not None, sample θ_i in
                    [phase_center - max_phase, phase_center + max_phase].
                    For "small phase", use something like max_phase = 0.1 (≈ 6 degrees).
        phase_center : center of the phase interval (default 0.0).
        same_phase_across_blocks : if True and random_phase is True,
                    all blocks share the same θ.
        random_phase : if False, set all θ_i = phase_center exactly
                       (purely deterministic angle).
        offdiag_scale : std of K12, K21, K22 at init (kept small so spectral
                        normalization barely shrinks K at init).
        """
        assert 0.0 < rho < 1.0, "rho must be in (0,1)"

        n_pairs = self.d_state // 2
        device = self.rho_raw.device
        dtype = self.rho_raw.dtype

        # 1) S ≈ I so x- and z-coordinates are initially close
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
                    phi = (2 * math.pi * torch.rand(1, device=device, dtype=dtype) - math.pi)
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

        # 4) Small off-diagonal blocks so ||K_raw||_2 ≈ rho and spectral
        #    normalization is almost inactive at init.
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
        state: torch.Tensor | None = None,   # state in z-basis
        *,
        time_first: bool = False,
        return_state: bool = False,
        mode: str = "scan",               # "loop" or "scan"
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
        if state is None:
            z = u.new_zeros(B_sz, self.d_state)
        else:
            if state.dim() == 1:
                x0_ = state.unsqueeze(0).expand(B_sz, -1)
            else:
                x0_ = state
            # z = S x in column convention → row-wise: z = x @ S^T
            z = x0_ @ self.S.T

        # LOOP mode
        if mode != "scan":
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
            # states: (T+1,B,D_state) in z-coordinates
            states = compute_linear_recurrence_parallel_block2x2(A_z, B_z, u_scan, z)
            z_seq = states[:-1].transpose(0, 1)  # (B,T,d_state)
            z_last = states[-1]                  # (B,d_state)
            y_seq = z_seq @ Ct + u @ Dt          # (B,T,d_output)

        if time_first:
            y_seq = y_seq.transpose(0, 1)
            z_seq = z_seq.transpose(0, 1)

        if return_state:
            return y_seq, z_seq
        return y_seq, z_seq



""" SSM models ----------------------------------------- """

""" Optional data class to set up the SSM model (values here are used just to initialize all fields) """


@dataclass
class SSMConfig:
    d_model: int = 10  # input/output size of the LRU after the decoding phase (n_u = n_y)
    d_state: int = 32  # state size of the LRU (n_x)
    n_layers: int = 2  # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0  # set it different from 0 if you want to introduce dropout regularization
    bias: bool = False  # bias of MLP static_layers
    rmin: float = 0.0  # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = 1.0  # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi  # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP"  # non-linear block used in the scaffolding
    scale: float = 1  # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4  # controls the hidden layer's dimension of the MLP
    d_hidden: int = 4  # controls the hidden layer's dimension of the non-linear layer
    nl_layers: int = 2 # number of hidden layers of the non-linear layer
    param: str = None  # pick the parametrization you want to use for the LRU. Default = LRU, other options are L2RU
    # and ZAK
    gamma: float = None  # set the overall l2 gain value in case you want to keep it fixed and not trainable, if set to
    # None, the gain will be trainable.
    init: str = 'eye'  # controls the initialization of the parameters when the L2RU param is chosen.
    # parameters needed for the prescribed-gain parametrization
    rho: float = 0.9
    max_phase_b: float = 0.04          # small spread
    phase_center: float = 0        # center angle
    random_phase: bool = True

    # Parallel scan must be selected in the forward call of the SSM.

    # Generate TypedDict automatically


SSMConfigDict = TypedDict('SSMConfigDict',
                          {f.name: f.type for f in fields(SSMConfig)},
                          total=False)

""" SSMs blocks ----------------------------------------- """


# python
class SSL(nn.Module):
    """State Space Layer: LRU --> FF --> residual"""

    def __init__(self, config: SSMConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        if config.param is None or config.param == "lru":
            self.lru = LRU(
                in_features=config.d_model,
                out_features=config.d_model,
                state_features=config.d_state,
                rmin=config.rmin,
                rmax=config.rmax,
                max_phase=config.max_phase,
            )
        elif config.param == "l2ru":
            self.lru = L2RU(state_features=config.d_model, init=config.init)
        elif config.param == "zak":
            self.lru = lruz(
                input_features=config.d_model,
                output_features=config.d_model,
                state_features=config.d_state,
                rmin=config.rmin,
                rmax=config.rmax,
                max_phase=config.max_phase,
            )
        elif config.param == "l2n":
            self.lru = Block2x2DenseL2SSM(
                d_state=config.d_state,
                d_input=config.d_model,
                d_output=config.d_model,
                train_gamma=True,
            )
            # initialization of matrix A
            self.lru.init_on_circle(rho=config.rho, max_phase=config.max_phase_b, phase_center=config.phase_center, random_phase=config.random_phase)
        elif config.param == "l2nt":
            self.lru = L2BoundedLTICell(
                d_state=config.d_state,
                d_input=config.d_model,
                d_output=config.d_model,
                train_gamma=True,
                )
        elif config.param == "tv":
            self.lru = Block2x2SelectiveBCDExpertsL2SSM(
                d_state=config.d_state,
            d_input=config.d_model,
            d_output=config.d_model,
            train_gamma=True,
            )
            self.lru.init_on_circle(rho=config.rho, max_phase=config.max_phase_b, phase_center=config.phase_center, random_phase=config.random_phase)
        else:
            raise ValueError("Invalid parametrization")

        l_config = LayerConfig()
        l_config.d_input = config.d_model
        l_config.d_output = config.d_model
        l_config.d_hidden = config.d_hidden
        l_config.n_layers = config.nl_layers

        ff_layers = {
            "GLU": lambda: GLU(l_config),
            "MLP": lambda: MLP(l_config),
            "LGLU":lambda :L2BoundedGLU(l_config),
            "LMLP": lambda: LMLP(l_config),
            "TLIP": lambda: TLIP(l_config),
        }
        if config.ff not in ff_layers:
            raise ValueError(f"Unknown feedforward type: {config.ff}")

        self.ff = ff_layers[config.ff]()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, mode: str = "loop"):
        z, st = self.lru(_normalize_to_3d(x), state=state, mode=mode)  # LTI
        z = self.ff(z)  # nonlinearity
        z = self.dropout(z)
        return z + x, st


# python
class DeepSSM(nn.Module):
    """Deep SSM: encoder -> n blocks -> decoder."""

    def __init__(
            self,
            d_input: int,
            d_output: int,
            *,
            # explicit keyword-only params mirroring SSMConfig (purely for type hints)
            d_model: int = 10,
            d_state: int = 32,
            n_layers: int = 2,
            dropout: float = 0.0,
            bias: bool = False,
            rmin: float = 0.0,
            rmax: float = 1.0,
            max_phase: float = 2 * math.pi,
            ff: str = "MLP",
            scale: float = 1,
            dim_amp: int = 4,
            d_hidden: int = 4,
            nl_layers: int =3,
            param: Optional[str] = None,
            gamma: Optional[float] = None,
            init: str = "eye",
            rho: float =0.9,
            max_phase_b: float=0.5,            # small spread
            phase_center: float=0,          # center angle ≈ 0°
            random_phase=True,
            config: Optional[SSMConfig] = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        # prefer an explicit config instance, otherwise create one from kwargs
        if config is not None:
            self.config = config
        else:
            self.config = SSMConfig(
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                dropout=dropout,
                bias=bias,
                rmin=rmin,
                rmax=rmax,
                max_phase=max_phase,
                ff=ff,
                scale=scale,
                dim_amp=dim_amp,
                d_hidden=d_hidden,
                nl_layers = nl_layers,
                param=param,
                gamma=gamma,
                init=init,
                rho=rho,
                max_phase_b=max_phase_b,
                phase_center=phase_center,
                random_phase=random_phase,
            )

        if self.config.param is not None and self.config.gamma is not None:
            self.register_buffer("gamma_t", torch.tensor(self.config.gamma))
            self.encoder = nn.Parameter(torch.randn(self.config.d_model, self.d_input))
            self.decoder = nn.Parameter(torch.randn(self.d_output, self.config.d_model))
        else:
            self.encoder = nn.Linear(d_input, self.config.d_model, bias=False)
            self.decoder = nn.Linear(self.config.d_model, d_output, bias=False)

        self.blocks = nn.ModuleList([SSL(self.config) for _ in range(self.config.n_layers)])

    def forward(self, u: torch.Tensor, state: Optional[List[torch.Tensor]] = None, gamma=None, mode: str = "scan"):
        # Initialize per-layer states
        layer_states: List[Optional[torch.Tensor]]
        if state is None:
            layer_states = [None] * len(self.blocks)
        else:
            layer_states = state if isinstance(state, list) else [state] * len(self.blocks)

        # Encode
        if isinstance(self.encoder, nn.Linear):
            x = self.encoder(_normalize_to_3d(u))
        else:
            x = _normalize_to_3d(u) @ self.encoder.T

        # Cascade blocks
        for i, block in enumerate(self.blocks):
            x, st = block(x, state=layer_states[i], mode=mode)
            layer_states[i] = st[:, -1, :]  # keep only the final state

        # Decode
        if self.config.param is not None and self.config.gamma is not None:
            gamma_t = torch.abs(self.gamma_t) if gamma is None else gamma
            gammaLRU = [
                torch.abs(block.lru.gamma * block.ff.lip) if hasattr(block.ff, "lip")
                else torch.abs(block.lru.gamma)
                for block in self.blocks
            ]
            if len(gammaLRU) > 0:
                gammaLRU_tensor = torch.stack(gammaLRU)
                enc_norm = torch.linalg.matrix_norm(self.encoder, 2)
                dec_norm = torch.linalg.matrix_norm(self.decoder, 2)
                gamma_prod = torch.prod(gammaLRU_tensor+1)
                decoder_scaled = (gamma_t * self.decoder) / (enc_norm * dec_norm * gamma_prod)
                outputs = x @ decoder_scaled.T
            else:
                outputs = x @ self.decoder.T
        else:
            outputs = self.decoder(x) if isinstance(self.decoder, nn.Linear) else x @ self.decoder.T

        return outputs, layer_states

    def reset(self):
        for block in self.blocks:
            block.lru.reset()


# Pure LRU blocks -----------------------------------------------

# python
class PureLRUR(nn.Module):
    """Pure LRU block without scaffolding."""

    def __init__(self, n: int, gamma: float = None, param: str = "l2ru", init: str = "eye"):
        super().__init__()
        if param == "l2ru":
            self.lru = L2RU(state_features=n, gamma=gamma, init=init)
        elif param == "zak":
            self.lru = lruz(input_features=n, output_features=n, state_features=n, gamma=gamma)
        else:
            raise ValueError("Unsupported param")

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, mode: str = "scan"):
        y, st = self.lru(_normalize_to_3d(x), state=state, mode=mode)
        return y, st
