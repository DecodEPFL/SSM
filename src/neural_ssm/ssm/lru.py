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
from .mamba import  RobustMambaDiagSSM


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
    if state is None:
        return torch.zeros(batch_size, n_state, device=device, dtype=dtype)

    # accept (N,) or (B,N)
    if state.dim() == 1:
        state = state.unsqueeze(0).expand(batch_size, -1)
    elif state.dim() == 2:
        if state.size(0) == 1 and batch_size > 1:
            state = state.expand(batch_size, -1)
    else:
        raise ValueError("state must have shape (N,) or (B,N)")

    if state.shape != (batch_size, n_state):
        raise ValueError(f"state has shape {tuple(state.shape)}, expected {(batch_size, n_state)}")

    return state.to(device=device, dtype=dtype)



def _scan_diag_linear(lambdas: torch.Tensor, Bu: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    Bsz, L, N = Bu.shape

    # ensure x0 is (B,N)
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0).expand(Bsz, -1)
    elif x0.dim() == 2 and x0.size(0) == 1 and Bsz > 1:
        x0 = x0.expand(Bsz, -1)
    if x0.shape != (Bsz, N):
        raise ValueError(f"x0 has shape {tuple(x0.shape)}, expected {(Bsz, N)}")

    if L == 0:
        return x0.unsqueeze(1)  # (B,1,N) = [x0]

    Bu = Bu.clone()
    Bu[:, 0, :] = Bu[:, 0, :] + lambdas * x0   # fold x0 into first step

    lam_seq = lambdas.expand(L, -1)            # (L,N)

    def _scan_fn(bu_seq):
        return associative_scan(binary_operator_diag, (lam_seq, bu_seq))[1]

    x_next = torch.vmap(_scan_fn)(Bu)          # (B,L,N) = x_1..x_L

    states = torch.empty(Bsz, L + 1, N, device=Bu.device, dtype=Bu.dtype)
    states[:, 0] = x0
    states[:, 1:] = x_next
    return states


@torch.jit.script
def lru_forward_loop(
    input: Tensor,  # (B,L,H) real (usually)
    state: Tensor,  # (B,N) complex for LRU
    A: Tensor,      # (N,) or (N,N) complex
    B: Tensor,      # (N,H) complex
    C: Tensor,      # (H_out,N) complex
    D: Tensor,      # (H_out,H) real
) -> Tuple[Tensor, Tensor]:
    BATCH, SEQ, H = input.shape
    N = state.size(1)

    # compute recurrence in the state dtype (complex)
    sdtype = state.dtype
    device = input.device

    x = state.to(device=device, dtype=sdtype)
    A = A.to(device=device, dtype=sdtype)
    B = B.to(device=device, dtype=sdtype)
    C = C.to(device=device, dtype=sdtype)

    # D stays in input dtype (real), so y is real like in scan
    D = D.to(device=device, dtype=input.dtype)

    BT = B.mT          # (H,N) complex
    CT = C.mT          # (N,H_out) complex
    DT = D.mT          # (H,H_out) real

    states = torch.empty((BATCH, SEQ + 1, N), device=device, dtype=sdtype)
    states[:, 0] = x

    if A.dim() == 1:
        lambdas = A  # (N,) complex
        for t in range(SEQ):
            u_t_c = input[:, t, :].to(dtype=sdtype)   # complex for state update
            x = x * lambdas + u_t_c @ BT
            states[:, t + 1] = x
    elif A.dim() == 2:
        A_T = A.mT
        for t in range(SEQ):
            u_t_c = input[:, t, :].to(dtype=sdtype)
            x = x @ A_T + u_t_c @ BT
            states[:, t + 1] = x
    else:
        raise RuntimeError("Unsupported A.dim(), expected 1 or 2")

    pre_states = states[:, :-1, :]  # x_0..x_{L-1}

    # output is real (match scan): Re(C x_t) + D u_t
    y_lin = (pre_states @ CT).real.to(dtype=input.dtype)     # (B,L,H_out)
    y_dir = input @ DT                                       # (B,L,H_out)
    output = y_lin + y_dir

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
            state: torch.Tensor | None = None,  # x0 in x-basis
            *,
            time_first: bool = False,
            return_state: bool = True,  # if True return x_seq (x0..x_{T+1})
            mode: str = "scan",  # "loop" or "scan"
            return_last: bool = False,  # if True return x_last (= x_{T+1})
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
        # ---- normalize input to (B,L,du) ----
        if u.dim() == 2:
            u = u.unsqueeze(0) if not time_first else u.unsqueeze(1)  # (1,L,du) or (L,1,du)

        u_bt = u.transpose(0, 1).contiguous() if time_first else u
        B_sz, L, du = u_bt.shape
        assert du == self.d_input
        dx = self.d_state

        # ---- z-basis matrices (as before) ----
        A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()

        # ---- initial x0 in x-basis ----
        if state is None:
            x0 = u_bt.new_zeros(B_sz, dx)
        else:
            x0 = state.unsqueeze(0).expand(B_sz, -1) if state.dim() == 1 else state
            x0 = x0.to(device=u_bt.device, dtype=u_bt.dtype)

        # S used for x<->z
        S = self.S.to(device=u_bt.device, dtype=u_bt.dtype)

        if mode != "scan":
            # Row convention:
            #   z = x @ S^T
            #   z_{t+1} = z_t @ A_z^T + u_t @ B_z^T
            #   y_t = z_t @ C_z^T + u_t @ D_z^T
            #
            # Convert to x-dynamics:
            #   x_{t+1} = x_t @ (S^T A_z^T (S^T)^{-1}) + u_t @ (B_z^T (S^T)^{-1})
            #   y_t     = x_t @ (S^T C_z^T) + u_t @ D_z^T
            I = torch.eye(dx, device=u_bt.device, dtype=u_bt.dtype)
            ST_inv = torch.linalg.solve(S.T, I)  # (S^T)^{-1}

            At = A_z.T
            Bt = B_z.T
            Ct = C_z.T
            Dt = D_z.T

            A_x = S.T @ At @ ST_inv  # (dx, dx)
            B_x = Bt @ ST_inv  # (du, dx)
            C_x = S.T @ Ct  # (dx, dy)
            D_x = Dt  # (du, dy)

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
            # SCAN (keep exactly as before, in z-basis)
            # ==========================================================
            z0 = x0 @ S.T  # (B,dx)

            # scan expects time-major (L,B,du)
            u_tb = u_bt.transpose(0, 1).contiguous()  # (L,B,du)

            # states: (L+1,B,dx) = z_0..z_{T+1}
            states = compute_linear_recurrence_parallel_block2x2(A_z, B_z, u_tb, z0)

            # batch-first
            z_seq = states.transpose(0, 1).contiguous()  # (B,L+1,dx)
            z_last = z_seq[:, -1, :]  # (B,dx)

            # outputs use z_t for t=0..T (exclude last state)
            y_seq = z_seq[:, :-1, :] @ C_z.T + u_bt @ D_z.T  # (B,L,dy)

            # Convert states to x-basis only if needed
            if return_state:
                z_flat_T = z_seq.reshape(B_sz * (L + 1), dx).T
                x_flat = torch.linalg.solve(S.T, z_flat_T).T
                x_seq = x_flat.reshape(B_sz, L + 1, dx)
                x_last = x_seq[:, -1, :]
            else:
                x_seq = None
                x_last = torch.linalg.solve(S.T, z_last.T).T  # (B,dx)

        # ---- restore time_first ----
        if time_first:
            y_seq = y_seq.transpose(0, 1).contiguous()  # (L,B,dy)
            if return_state:
                x_seq = x_seq.transpose(0, 1).contiguous()  # (L+1,B,dx)

        # ---- returns ----
        if return_state and return_last:
            return y_seq, x_seq, x_last
        if return_state:
            return y_seq, x_seq
        if return_last:
            return y_seq, x_last
        return y_seq

""" SSM models ----------------------------------------- """

""" Optional data class to set up the SSM model (values here are used just to initialize all fields) """


@dataclass
class SSMConfig:
    d_model: int = 10  # input/output size of the LRU after the decoding phase (n_u = n_y)
    d_state: int = 32  # state size of the LRU (n_x)
    n_layers: int = 2  # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0  # set it different from 0 if you want to introduce dropout regularization
    bias: bool = False  # bias of MLP static_layers
    rmin: float = .8  # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = .95  # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi  # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP"  # non-linear static block used in the scaffolding
    scale: float = 1  # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4  # controls the hidden layer's dimension of the MLP
    d_hidden: int = 4  # controls the hidden layer's dimension of the non-linear layer
    nl_layers: int = 2 # number of hidden layers of the non-linear layers (static nonlinearities)
    param: str = None  # pick the parametrization you want to use for the LRU. Default = LRU, other options are L2RU
    # and ZAK
    gamma: float = 1  # set the overall l2 gain value for the SSM. If set to None, the l2 gain will be trainable and
    # not specified.
    train_gamma: bool = True # choose weather the l2 gain of the dynamical systems should be trainable or fixed.
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


class SSL(nn.Module):
    """State Space Layer: LRU -> FF -> dropout (optional) -> residual
    y = FF(LRU(u))+u """
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        # --- SSM selection ---
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
                train_gamma=config.train_gamma,
            )
            self.lru.init_on_circle(
                rho=config.rho,
                max_phase=config.max_phase_b,
                phase_center=config.phase_center,
                random_phase=config.random_phase,
            )
        elif config.param == "l2nt":
            self.lru = L2BoundedLTICell(
                d_state=config.d_state,
                d_input=config.d_model,
                d_output=config.d_model,
                train_gamma=config.train_gamma,
            )
        elif config.param == "tv":
            self.lru = RobustMambaDiagSSM(
                d_state=config.d_state,
                d_model=config.d_model,
                d_out=config.d_model,
                train_gamma=config.train_gamma,
            )
        else:
            raise ValueError("Invalid parametrization")

        # --- FF selection ---
        l_config = LayerConfig()
        l_config.d_input = config.d_model
        l_config.d_output = config.d_model
        l_config.d_hidden = config.d_hidden
        l_config.n_layers = config.nl_layers

        ff_layers = {
            "GLU": GLU,
            "MLP": MLP,
            "LGLU": L2BoundedGLU,
            "LMLP": LMLP,
            "TLIP": TLIP,
        }
        if config.ff not in ff_layers:
            raise ValueError(f"Unknown feedforward type: {config.ff}")
        self.ff = ff_layers[config.ff](l_config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x3d: torch.Tensor, state: Optional[torch.Tensor] = None, mode: str = "loop"):
        # assume x is already [B,T,D]
        #xn = self.ln(x3d)
        z, st = self.lru(x3d, state=state, mode=mode)
        z = self.ff(z)
        z = self.dropout(z)
        return x3d + z, st




class DeepSSM(nn.Module):
    """Deep SSM: encoder -> n blocks -> decoder."""
    def __init__(
        self,
        d_input: int,
        d_output: int,
        *,
        d_model: int = 10,
        d_state: int = 32,
        n_layers: int = 2,
        dropout: float = 0.0,
        bias: bool = False,
        rmin: float = .8,
        rmax: float = .95,
        max_phase: float = 2 * math.pi,
        ff: str = "LGLU",
        scale: float = 1,
        dim_amp: int = 4,
        d_hidden: int = 4,
        nl_layers: int = 3,
        param: Optional[str] = 'lru',
        gamma: Optional[float] = None,
        train_gamma: Optional[bool] = True,
        init: str = "eye",
        rho: float = 0.9,
        max_phase_b: float = 0.5,
        phase_center: float = 0,
        random_phase: bool = True,
        config: Optional[SSMConfig] = None,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output

        self.config = config if config is not None else SSMConfig(
            d_model=d_model, d_state=d_state, n_layers=n_layers, dropout=dropout, bias=bias,
            rmin=rmin, rmax=rmax, max_phase=max_phase, ff=ff, scale=scale, dim_amp=dim_amp,
            d_hidden=d_hidden, nl_layers=nl_layers, param=param, gamma=gamma,
            train_gamma=train_gamma, init=init, rho=rho, max_phase_b=max_phase_b,
            phase_center=phase_center, random_phase=random_phase,
        )

        self.use_cert_scaling = (self.config.param != "lru") and (self.config.gamma is not None)

        if self.use_cert_scaling:
            self.register_buffer("gamma_t", torch.tensor(float(self.config.gamma)))
            self.encoder_w = nn.Parameter(torch.randn(self.config.d_model, self.d_input))
            self.decoder_w = nn.Parameter(torch.randn(self.d_output, self.config.d_model))

            # config.ff is shared, so hasattr is constant across blocks
            self.ff_has_lip = (self.config.ff in {"LGLU", "TLIP"}) or True  # or safer: set after blocks are built
        else:
            self.encoder = nn.Linear(d_input, self.config.d_model, bias=False)
            self.decoder = nn.Linear(self.config.d_model, d_output, bias=False)

        self.blocks = nn.ModuleList([SSL(self.config) for _ in range(self.config.n_layers)])

        if self.use_cert_scaling:
            # now that blocks exist, we can check once
            self.ff_has_lip = hasattr(self.blocks[0].ff, "lip")


    def forward(self, u: torch.Tensor, state: Optional[List[torch.Tensor]] = None, gamma=None, mode: str = "scan"):
        u3d = _normalize_to_3d(u)

        if state is None:
            layer_states: List[Optional[torch.Tensor]] = [None] * len(self.blocks)
        else:
            layer_states = state if isinstance(state, list) else [state] * len(self.blocks)

        # Encode
        if self.use_cert_scaling:
            x = F.linear(u3d, self.encoder_w, bias=None)
        else:
            x = self.encoder(u3d)

        # Blocks
        for i, block in enumerate(self.blocks):
            x, st = block(x, state=layer_states[i], mode=mode)
            layer_states[i] = st[:, -1, :]

        # Decode
        if self.use_cert_scaling:
            gamma_t = self.gamma_t.abs() if gamma is None else torch.as_tensor(gamma, device=x.device, dtype=x.dtype).abs()

            # collect per-block gains
            g = torch.stack([block.lru.gamma.abs() for block in self.blocks])
            if self.ff_has_lip:
                l = torch.stack([block.ff.lip.abs() for block in self.blocks])
                g = g * l

            # prod(1+g) in log-space
            log_gamma_prod = torch.log1p(g).sum()
            gamma_prod = torch.exp(log_gamma_prod)

            # EXACT SVD norms with backward
            enc_norm = torch.linalg.svdvals(self.encoder_w).max()
            dec_norm = torch.linalg.svdvals(self.decoder_w).max()

            scale = gamma_t / (enc_norm * dec_norm * gamma_prod + 1e-12)
            decoder_scaled = self.decoder_w * scale

            outputs = F.linear(x, decoder_scaled, bias=None)
        else:
            outputs = self.decoder(x)

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
        elif param == "lru":
            self.lru = LRU(in_features=n, out_features=n, state_features=n)
        else:
            raise ValueError("Unsupported param")

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, mode: str = "scan"):
        y, st = self.lru(_normalize_to_3d(x), state=state, mode=mode)
        return y, st




class SimpleRNN(nn.Module):
    """
    Thin wrapper around nn.RNN that first normalizes the input to 3D
    using the same convention as DeepSSM.

    Input:
      u: (T, d_input) or (B, T, d_input)

    State:
      state (h0): (d_hidden,) or (B, d_hidden) or (1, B, d_hidden)

    Returns (batch-first):
      y: (B, T, d_output)
      h_seq (optional): (B, T+1, d_hidden) = [h0, h1, ..., hT]
      h_last (optional): (B, d_hidden)     = hT
    """

    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        *,
        num_layers: int = 1,
        nonlinearity: str = "tanh",  # "tanh" or "relu"
        bias: bool = True,
        dropout: float = 0.0,        # only applied if num_layers > 1 (PyTorch behavior)
        bidirectional: bool = False, # for simplicity, we keep return shapes as-is
    ):
        super().__init__()
        self.d_input = int(d_input)
        self.d_hidden = int(d_hidden)
        self.d_output = int(d_output)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        self.rnn = nn.RNN(
            input_size=self.d_input,
            hidden_size=self.d_hidden,
            num_layers=self.num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )

        # Project RNN outputs to d_output
        self.out_proj = nn.Linear(self.d_hidden * self.num_directions, self.d_output, bias=bias)

    def _format_h0(self, h0: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        """
        nn.RNN expects h0: (num_layers * num_directions, B, d_hidden)
        Accept:
          - None
          - (d_hidden,)
          - (B, d_hidden)
          - (1, B, d_hidden)  [for convenience if user already has RNN shape]
          - (L, B, d_hidden)  [full shape]
        """
        L = self.num_layers * self.num_directions

        if h0 is None:
            return torch.zeros(L, B, self.d_hidden, device=device, dtype=dtype)

        if h0.dim() == 1:
            h0 = h0.unsqueeze(0).unsqueeze(0)  # (1,1,H)
        elif h0.dim() == 2:
            h0 = h0.unsqueeze(0)               # (1,B,H)
        elif h0.dim() == 3:
            pass
        else:
            raise ValueError(f"h0 must have dim 1,2,3; got shape {tuple(h0.shape)}")

        # Broadcast batch if needed
        if h0.size(1) == 1 and B > 1:
            h0 = h0.expand(h0.size(0), B, h0.size(2))

        # Broadcast layers if needed
        if h0.size(0) == 1 and L > 1:
            h0 = h0.expand(L, h0.size(1), h0.size(2))

        if h0.shape != (L, B, self.d_hidden):
            raise ValueError(f"h0 has shape {tuple(h0.shape)}, expected {(L, B, self.d_hidden)}")

        return h0.to(device=device, dtype=dtype)

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,  # h0
        *,
        return_state: bool = False,
        return_last: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor],
    ]:
        u3d = _normalize_to_3d(u)  # (B,T,D)
        B, T, D = u3d.shape
        if D != self.d_input:
            raise ValueError(f"Expected input dim {self.d_input}, got {D}")

        h0 = self._format_h0(state, B, u3d.device, u3d.dtype)

        # Run RNN
        out, hT = self.rnn(u3d, h0)  # out: (B,T,H*num_dir), hT: (L,B,H)

        # Project to output dim
        y = self.out_proj(out)       # (B,T,d_output)

        # Build full hidden trajectory if requested: [h0, h1, ..., hT]
        # nn.RNN doesn't return all hidden states per step directly,
        # but `out` *is* the last-layer hidden state at each time.
        # For multi-layer RNNs, this corresponds to the top layer only.
        a = 0
        h_seq = None
        if return_state:
            # top-layer h_t sequence is out; prepend the top-layer initial state
            top_layer_idx = self.num_directions * (self.num_layers - 1)
            h0_top = h0[top_layer_idx: top_layer_idx + self.num_directions]  # (num_dir,B,H)
            # If bidirectional, top "initial" is two directions; we pack them consistently
            # by taking the forward direction state as "the" h0 for the sequence.
            h0_seq = h0_top[0].transpose(0, 0)  # (B,H) no-op, explicit
            h_seq = torch.empty(B, T + 1, out.size(-1), device=u3d.device, dtype=u3d.dtype)
            h_seq[:, 0, :] = torch.cat([h0_top[d] for d in range(self.num_directions)], dim=-1) if self.num_directions > 1 else h0_top[0]
            h_seq[:, 1:, :] = out

        # last hidden (top layer)
        h_last = None
        if return_last:
            top_layer = hT[-self.num_directions:]  # (num_dir,B,H)
            h_last = torch.cat([top_layer[d] for d in range(self.num_directions)], dim=-1) if self.num_directions > 1 else top_layer[0]

        if return_state and return_last:
            return y, h_seq, h_last
        if return_state:
            return y, h_seq
        if return_last:
            return y, h_last
        return y, a
