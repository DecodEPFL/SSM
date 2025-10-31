# python
import math
from typing import Optional, Tuple, List, Dict
from src.neural_ssm.ssm.scan_utils import *
from src.neural_ssm.static_layers.generic_layers import *
from src.neural_ssm.static_layers.lipschitz_mlps import *


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


def _scan_diag_linear(lambdas: torch.Tensor, Bu: torch.Tensor, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Bu: (B, L, N) already includes the input * B^T. We seed first step with lambdas * x0.
    B, L, N = Bu.shape
    Bu = Bu.clone()
    Bu[:, 0, :] += lambdas * x0
    lam_seq = lambdas.expand(L, -1)

    def _scan_fn(bu_seq):
        return associative_scan(binary_operator_diag, (lam_seq, bu_seq))[1]

    scanned = torch.vmap(_scan_fn)(Bu)             # (B, L, N)
    inner = torch.cat([x0.unsqueeze(1), scanned[:, :-1, :]], dim=1)  # (B, L, N)
    return scanned, inner


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
        self.B = nn.Parameter(B_complex)  # (N, U)

        C_complex = torch.complex(
            torch.randn(out_features, state_features) / self._sqrt_state_features,
            torch.randn(out_features, state_features) / self._sqrt_state_features,
        )
        self.C = nn.Parameter(C_complex)  # (H, N)

        # Runtime state
        self.state: Optional[torch.Tensor] = None

        # Small cache for complex->real transform 2x2 blocks
        self._T_cache: Dict[str, torch.Tensor] = {}

    def ss_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)
        lambdas = lambda_abs * torch.exp(1j * lambda_phase)  # (N,) complex
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)     # (N,1) real
        B = gammas * self.B                                  # (N,U) complex
        return lambdas, B, self.C, self.D

    def ss_real_matrices(self, to_numpy: bool = True):
        lambdas, B, C, D = self.ss_params()
        device, dtype = lambdas.device, lambdas.dtype
        n2 = 2 * self.state_features

        lambdas_conj = torch.stack([lambdas, lambdas.conj()], dim=1).flatten()
        A_full = torch.diag(lambdas_conj)                                # (2N,2N)
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
        BATCH, SEQ, _ = input.shape
        lambdas, B, C, D = self.ss_params()

        x = state.to(B.dtype)
        uB = input.to(B.dtype)
        BT = B.mT

        inner = torch.empty(BATCH, SEQ, self.state_features, device=input.device, dtype=B.dtype)
        for t, u_t in enumerate(uB.unbind(dim=1)):
            inner[:, t] = x
            x = lambdas * x + u_t @ BT

        self.state = x.detach()
        output = (inner @ C.mT).real + input @ D.T
        return output, inner

    @torch.compiler.disable
    def forward_scan(self, input: torch.Tensor, state: Optional[torch.Tensor] = None):
        BATCH, SEQ, _ = input.shape
        lambdas, B, C, D = self.ss_params()

        x0 = state.to(B.dtype)
        Bu = input.to(B.dtype) @ B.mT
        scanned, inner = _scan_diag_linear(lambdas, Bu, x0)

        self.state = scanned[:, -1, :].detach()
        output = (inner @ C.mT).real + input @ D.T
        return output, inner

    def forward(self, input: torch.Tensor, gamma: Optional[float] = None, state: Optional[torch.Tensor] = None, mode: str = "loop"):
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
    def __init__(self, state_features: int, gamma: float = None, init: str = "eye", q: int = 1, eye_scale=0.01, rand_scale=1):
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
        self.A = torch.zeros(state_features, state_features)
        self.B = torch.zeros(state_features, state_features)
        self.D = torch.zeros(state_features, state_features)
        self.state: Optional[torch.Tensor] = None

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

    def forward(self, input: torch.Tensor, state: Optional[torch.Tensor] = None, set_param: bool = True, mode: str = "scan"):
        input = _normalize_to_3d(input)
        # real-valued state for L2RU
        self.state = _init_or_cast_state(state, input.shape[0], self.state_features, input.device, input.dtype)

        x0 = self.state
        if set_param:
            self.set_param()

        if mode == "scan":
            u = input.permute(1, 0, 2)  # (L,B,H)
            states = compute_linear_recurrence_parallel(self.A, self.B, u, x0).permute(1, 0, 2)
            self.state = states[:, -1, :].detach()
            outputs = states @ self.C.transpose(-1, -2) + input @ self.D.transpose(-1, -2)
            return outputs, states
        elif mode in ("loop", "loop_efficient"):
            BT = self.B.transpose(-1, -2)
            AT = self.A.transpose(-1, -2)
            x = x0
            inner = []
            for u_t in input.unbind(dim=1):
                x = x @ AT + u_t @ BT
                inner.append(x)
            states = torch.stack(inner, dim=1)
            self.state = states[:, -1, :].detach()
            outputs = states @ self.C.transpose(-1, -2) + input @ self.D.transpose(-1, -2)
            return outputs, states
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = None



# python
class lruz(nn.Module):
    """LRU (ZAK parametrization) with learnable or fixed l2 gain gamma."""
    def __init__(self, input_features: int, output_features: int, state_features: int, rmin=0.9, rmax=1.0, max_phase=6.283, gamma: float = None):
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
        BATCH, SEQ, _ = input.shape
        if set_param:
            self.set_param()
        lambdas = torch.diagonal(self.A)

        x = state.to(self.B.dtype)
        uB = input.to(self.B.dtype)
        BT = self.B.mT

        inner = torch.empty(BATCH, SEQ, self.state_features, device=input.device, dtype=self.B.dtype)
        for t, u_t in enumerate(uB.unbind(dim=1)):
            inner[:, t] = x
            x = lambdas * x + u_t @ BT

        self.state = x.detach()
        output = (inner @ self.C.mT).real + input @ self.D.T
        return output, inner

    @torch.compiler.disable
    def forward_scan(self, input: torch.Tensor, state: Optional[torch.Tensor] = None, set_param: bool = True):
        BATCH, SEQ, _ = input.shape
        A, B, C, D = self.set_param() if set_param else (self.A, self.B, self.C, self.D)
        lambdas = torch.diagonal(A)

        x0 = state.to(B.dtype)
        Bu = input.to(B.dtype) @ B.mT
        scanned, inner = _scan_diag_linear(lambdas, Bu, x0)

        self.state = scanned[:, -1, :].detach()
        output = (inner @ C.mT).real + input @ D.T
        return output, inner

    def forward(self, input: torch.Tensor, gamma=None, state: Optional[torch.Tensor] = None, set_param: bool = True, mode: str = "scan"):
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



""" SSM models ----------------------------------------- """

""" Data class to set up the SSM model (values here are used just to initialize all fields) """


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
    param: str = None  # pick the parametrization you want to use for the LRU. Default = LRU, other options are L2RU
    # and ZAK
    gamma: float = None  # set the overall l2 gain value in case you want to keep it fixed and not trainable, if set to
    # None, the gain will be trainable.
    init: str = 'eye'  # controls the initialization of the parameters when the L2RU param is chosen.

    # Parallel scan must be selected in the forward call of the SSM.

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
        else:
            raise ValueError("Invalid parametrization")

        l_config = LayerConfig()
        l_config.d_input = config.d_model
        l_config.d_output = config.d_model
        l_config.d_hidden = config.d_hidden

        ff_layers = {
            "GLU": lambda: GLU(l_config),
            "MLP": lambda: MLP(l_config),
            "LMLP": lambda: LMLP(l_config),
            "TLIP": lambda: TLIP(l_config),
        }
        if config.ff not in ff_layers:
            raise ValueError(f"Unknown feedforward type: {config.ff}")

        self.ff = ff_layers[config.ff]()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, mode: str = "loop"):
        z, st = self.lru(_normalize_to_3d(x), state=state, mode=mode)  # LTI
        z = self.ff(z)                                # nonlinearity
        z = self.dropout(z)
        return z + x, st



# python
class DeepSSM(nn.Module):
    """Deep SSM: encoder -> n blocks -> decoder."""
    def __init__(self, n_u: int, n_y: int, config: SSMConfig):
        super().__init__()
        self.config = config

        if config.param is not None and config.gamma is not None:
            self.register_buffer("gamma_t", torch.tensor(config.gamma))
            self.encoder = nn.Parameter(torch.randn(config.d_model, n_u))
            self.decoder = nn.Parameter(torch.randn(n_y, config.d_model))
        else:
            self.encoder = nn.Linear(n_u, config.d_model, bias=False)
            self.decoder = nn.Linear(config.d_model, n_y, bias=False)

        self.blocks = nn.ModuleList([SSL(config) for _ in range(config.n_layers)])

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
            layer_states[i] = st

        # Decode
        if self.config.param is not None and self.config.gamma is not None:
            gamma_t = torch.abs(self.gamma_t) if gamma is None else gamma
            gammaLRU = [torch.abs(block.lru.gamma) for block in self.blocks if hasattr(block.lru, "gamma")]
            if len(gammaLRU) > 0:
                gammaLRU_tensor = torch.stack(gammaLRU)
                enc_norm = torch.linalg.matrix_norm(self.encoder, 2)
                dec_norm = torch.linalg.matrix_norm(self.decoder, 2)
                gamma_prod = torch.prod(gammaLRU_tensor) + 1  # kept as in original
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

