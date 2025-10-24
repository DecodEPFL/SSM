import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from collections import OrderedDict
from SSM.scan_utils import associative_scan, binary_operator_diag, compute_linear_recurrence_parallel
import torch.jit as jit
from SSM.L_bounded_MLPs import FirstChannel, SandwichFc, SandwichLin
from .ssm_zak import LRUZ

""" Linear Recurrent Units ----------------------------------------- """


class LRU(nn.Module):
    """ Linear Recurrent Unit. The LRU is simulated using Parallel Scan (fast!) when
     "scan" is set to True (default) in the forward pass, otherwise recursively (slow)."""

    def __init__(
            self, in_features: int, out_features: int, state_features: int, internal_state_init=None, rmin=0.9,
            rmax=1.0, max_phase=6.283
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.state_features = state_features

        # Pre-compute constants for efficiency
        self._sqrt_in_features = math.sqrt(in_features)
        self._sqrt_2_in_features = math.sqrt(2 * in_features)
        self._sqrt_state_features = math.sqrt(state_features)
        self._rmin_rmax_diff = rmax - rmin
        self._rmin_rmax_sum = rmax + rmin
        self._rmin_squared = rmin ** 2

        self.D = nn.Parameter(
            torch.randn([out_features, in_features]) / self._sqrt_in_features
        )

        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * self._rmin_rmax_sum * self._rmin_rmax_diff + self._rmin_squared))
        )
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(
            torch.log(torch.sqrt(1.0 - lambda_abs.square()))  # More efficient than torch.ones_like and torch.square
        )

        # More efficient initialization using a single complex tensor creation
        B_complex = torch.complex(
            torch.randn([state_features, in_features]) / self._sqrt_2_in_features,
            torch.randn([state_features, in_features]) / self._sqrt_2_in_features
        )
        self.B = nn.Parameter(B_complex)  # N, U

        C_complex = torch.complex(
            torch.randn([out_features, state_features]) / self._sqrt_state_features,
            torch.randn([out_features, state_features]) / self._sqrt_state_features
        )
        self.C = nn.Parameter(C_complex)  # H, N

        # initialize internal state
        self.state = None

        # Pre-compute transformation matrices for ss_real_matrices method
        self._T_block = None
        self._T_block_inv = None

    def ss_params(self):
        lambda_abs = torch.exp(-torch.exp(self.nu_log))
        lambda_phase = torch.exp(self.theta_log)

        # More efficient complex number creation
        lambdas = lambda_abs * torch.exp(1j * lambda_phase)
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        B = gammas * self.B
        return lambdas, B, self.C, self.D

    def ss_real_matrices(self, to_numpy=True):
        lambdas, B, C, D = self.ss_params()

        # Pre-allocate with the correct dtype and device
        device, dtype = lambdas.device, lambdas.dtype
        state_features_2 = 2 * self.state_features

        # More efficient tensor creation using stack instead of manual indexing
        lambdas_conjugate = torch.stack([lambdas, lambdas.conj()], dim=1).flatten()
        A_full = torch.diag(lambdas_conjugate)

        # More efficient B_full creation
        B_conjugate = torch.stack([B, B.conj()], dim=1).view(state_features_2, self.in_features)

        # More efficient C_full creation
        C_half = 0.5 * C
        C_conjugate = torch.stack([C_half, C_half.conj()], dim=2).view(self.out_features, state_features_2)

        # Cache transformation matrices
        if self._T_block is None or self._T_block.device != device:
            self._T_block = torch.tensor([[1, 1], [1j, -1j]], device=device, dtype=dtype)
            self._T_block_inv = torch.linalg.inv(self._T_block)

        T_full = torch.block_diag(*([self._T_block] * self.state_features))
        T_full_inv = torch.block_diag(*([self._T_block_inv] * self.state_features))

        # More efficient matrix operations using @ operator consistently
        A_real = (T_full @ A_full @ T_full_inv).real
        B_real = (T_full @ B_conjugate).real
        C_real = (C_conjugate @ T_full_inv).real
        D_real = D

        ss_real_params = [A_real, B_real, C_real, D_real]
        if to_numpy:
            ss_real_params = [param.detach().cpu().numpy() for param in ss_real_params]

        return tuple(ss_real_params)

    def forward_loop(self, input, state=None):
        batch_size, seq_len, _ = input.shape

        # State management
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.state_features,
                                     device=input.device, dtype=torch.complex64)

        lambdas, B, C, D = self.ss_params()

        # State computation using pre-converted input
        input_B_dtype = input.to(B.dtype)
        B_T = B.mT  # Cache transpose

        # Optimized loop with pre-allocated tensor for states
        inner_states = torch.empty(batch_size, seq_len, self.state_features,
                                   device=input.device, dtype=torch.complex64)

        # Vectorized state updates
        current_state = self.state
        for t, u_step in enumerate(input_B_dtype.unbind(dim=1)):
            inner_states[:, t] = current_state
            current_state = lambdas * current_state + u_step @ B_T

        self.state = current_state.detach()  # Update the internal state

        # Output computation using all inner states
        output = (inner_states @ C.mT).real + input @ D.T

        return output, inner_states

    @torch.compiler.disable
    def forward_scan(self, input, state=None):
        """
        Computes the LRU output using a parallel scan.

        Args:
            input (torch.Tensor): (B, L, H) input sequence.
            state (torch.Tensor, optional): (B, N) initial state. If None, a zero state is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - (B, L, H_out) output sequence.
                - (B, L, N) sequence of internal states.
        """
        batch_size, seq_len, _ = input.shape
        lambdas, B, C, D = self.ss_params()

        # If no initial state is provided, initialize it to zeros.
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.state_features,
                                     device=input.device, dtype=torch.complex64)

        # Pre-compute input transformation
        Bu_elements = input.to(B.dtype) @ B.mT

        # Incorporate the initial state into the first element of the sequence
        Bu_elements[:, 0, :] += lambdas * self.state

        # Define the scan function for vmap
        lambda_elements = lambdas.expand(seq_len, -1)

        def scan_fn(Bu_seq):
            return associative_scan(binary_operator_diag, (lambda_elements, Bu_seq))[1]

        # Apply the scan over the batch dimension
        scanned_states = torch.vmap(scan_fn)(Bu_elements)

        # Prepend the initial state to get the full state sequence
        inner_states = torch.cat([self.state.unsqueeze(1), scanned_states[:, :-1, :]], dim=1)

        # Update the internal state of the LRU module to the last state
        self.state = scanned_states[:, -1, :].detach()

        # Compute the final output
        output = (inner_states @ C.mT).real + input @ D.T
        return output, inner_states

    def forward(self, input, gamma=None, state=None, mode="loop"):

        if input.dim() == 1:
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 2:
            input = input.unsqueeze(0)

        if mode == "scan":
            return self.forward_scan(input, self.state)
        elif mode in ["loop", "loop_efficient"]:
            return self.forward_loop(input, self.state)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'scan', 'loop', or 'loop_efficient'.")

    def reset(self):
        self.state = None  # reset the SSM state to the initial value

    """ L2RU parametrization: Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma. """


class LRU_Robust(jit.ScriptModule):
    """ Implements a Linear Recurrent Unit (LRU) with trainable or prescribed l2 gain gamma. """

    def __init__(self, state_features: int, gamma: float = None, init='eye', q: int = 1, eye_scale=0.01, rand_scale=1):
        super().__init__()
        self.state_features = state_features
        if gamma is not None:
            self.gamma = torch.tensor(gamma)
        else:
            self.gamma = nn.Parameter(torch.tensor(2.2))

        self.state = torch.zeros(state_features)
        self.register_buffer('ID', torch.eye(state_features))
        self.alpha = nn.Parameter(torch.tensor(4.1))
        self.epsilon = torch.tensor(-.0)
        self.q = q

        # Store upper triangular indices for efficient computation
        self.register_buffer('triu_indices', torch.triu_indices(state_features, state_features, offset=1))
        self.register_buffer('tril_indices', torch.tril_indices(state_features, state_features, offset=0))

        n = state_features

        if init == 'eye':
            # Initialize with identity-like structure
            X11_full = eye_scale * torch.eye(n)
            X22_full = eye_scale * torch.eye(n)
            X21_init = 0.1 * torch.eye(n)
        elif init == 'rand':
            X11_full = rand_scale * torch.randn(n, n)
            X22_full = rand_scale * torch.randn(n, n)
            X21_init = rand_scale * torch.randn(n, n)
        else:
            raise ValueError(init)

        # Extract lower triangular elements as parameters
        X11_params = X11_full[self.tril_indices[0], self.tril_indices[1]]
        X22_params = X22_full[self.tril_indices[0], self.tril_indices[1]]

        self.X11_params = nn.Parameter(X11_params)
        self.X22_params = nn.Parameter(X22_params)

        # For Skew: store only upper triangular parameters (excluding diagonal)
        if q == 1:
            Skew_init = 0.01 * torch.randn(n, n)
            Skew_init = Skew_init - Skew_init.T  # Make it skew-symmetric
            Skew_params = Skew_init[self.triu_indices[0], self.triu_indices[1]]
            self.Skew_params = nn.Parameter(Skew_params)

        # Other parameters
        self.X21 = nn.Parameter(X21_init)
        self.C = nn.Parameter(torch.eye(state_features))
        self.Dt = nn.Parameter(torch.eye(state_features))

        # Initialize remaining LTI matrices
        self.A = torch.zeros(state_features, state_features)
        self.B = torch.zeros(state_features, state_features)
        self.D = torch.zeros(state_features, state_features)

    def _get_lower_triangular(self, params: torch.Tensor) -> torch.Tensor:
        """Reconstruct lower triangular matrix from parameters"""
        L = torch.zeros(self.state_features, self.state_features,
                        device=params.device, dtype=params.dtype)
        L[self.tril_indices[0], self.tril_indices[1]] = params
        return L

    def _get_skew_symmetric(self, params: torch.Tensor) -> torch.Tensor:
        """Reconstruct skew-symmetric matrix from parameters"""
        Sk = torch.zeros(self.state_features, self.state_features,
                         device=params.device, dtype=params.dtype)
        Sk[self.triu_indices[0], self.triu_indices[1]] = params
        Sk[self.triu_indices[1], self.triu_indices[0]] = -params
        return Sk

    # @jit.script_method
    def set_param(self):
        # Reconstruct parametrized matrices
        X11 = self._get_lower_triangular(self.X11_params)
        X22 = self._get_lower_triangular(self.X22_params)

        if self.q == 1:
            Sk = self._get_skew_symmetric(self.Skew_params)
            Q = (self.ID - Sk) @ torch.linalg.inv(self.ID + Sk)
        else:
            Q = self.ID  # just the identity

        # Parameter update for l2 gain (free param)
        gamma = self.gamma

        # Auxiliary Parameters
        Z = self.X21 @ self.X21.T + X22 @ X22.T + self.Dt.T @ self.Dt + torch.exp(self.epsilon) * self.ID
        beta = gamma ** 2 * torch.sigmoid(self.alpha) / torch.linalg.matrix_norm(Z, 2)
        H11 = X11 @ X11.T + self.C.T @ self.C + beta * torch.exp(self.epsilon) * self.ID
        H12 = torch.sqrt(beta) * (X11 @ self.X21.T + self.C.T @ self.Dt)
        V = Z * beta - gamma ** 2 * self.ID
        R = H12 @ torch.linalg.inv(V.T) @ H12.T
        CR = torch.linalg.cholesky(-R)
        CRH = torch.linalg.cholesky(-R + H11)

        # LTI system matrices
        A = torch.linalg.inv(CRH).T @ Q @ CR.T
        B = A @ torch.linalg.inv(H12.T) @ V.T
        C = self.C
        D = torch.sqrt(beta) * self.Dt

        self.A = A
        self.B = B
        self.D = D

        return A, B, C, D

    def forward(self, input: torch.Tensor, state=None, mode: str = "loop") -> tuple:
        if input.dim() == 1:
            input = input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 2:
            input = input.unsqueeze(0)
        batch_size = input.shape[0]

        if self.state.shape[0] != batch_size:
            self.state = torch.zeros(batch_size, self.state_features, device=input.device)
        x0 = self.state
        A, B, C, D = self.set_param()

        if mode == "scan":
            u = input.permute(1, 0, 2)
            states = compute_linear_recurrence_parallel(A, B, u, x0)
            states = states.permute(1, 0, 2)
            last_state = states[:, -1, :].detach()
            self.state = last_state
            outputs = torch.matmul(states, C.transpose(-1, -2)) + torch.matmul(input, D.transpose(-1, -2))
            return outputs, states
        else:
            x = x0
            states_list = []
            for u_step in input.split(1, dim=1):
                u = u_step.squeeze(1)
                x = torch.matmul(x, A.transpose(-1, -2)) + torch.matmul(u, B.transpose(-1, -2))
                states_list.append(x)
            states = torch.stack(states_list, dim=1)
            self.state = states[:, -1, :].detach()
            outputs = torch.matmul(states, C.transpose(-1, -2)) + torch.matmul(input, D.transpose(-1, -2))
            return outputs, states

    def reset(self, batch_size: int = 1, device: torch.device = None):
        if device is None:
            device = torch.device('cpu')
        self.state = torch.zeros(batch_size, self.state_features, device=device)


""" SSM models ----------------------------------------- """

""" Data class to set up the SSM model (values here are used just to initialize all fields) """


@dataclass
class SSMConfig:
    d_model: int = 10  # input/output size of the LRU after the decoding phase (n_u = n_y)
    d_state: int = 32  # state size of the LRU (n_x)
    n_layers: int = 2  # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0  # set it different from 0 if you want to introduce dropout regularization
    bias: bool = False  # bias of MLP layers
    rmin: float = 0.0  # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = 1.0  # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi  # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP"  # non-linear block used in the scaffolding
    scale: float = 1  # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4  # controls the hidden layer's dimension of the MLP
    param: str = None  # pick the parametrization you want to use for the LRU. Default = LRU, other options are L2RU
    # and ZAK
    gamma: float = None  # set the overall l2 gain value in case you want to keep it fixed and not trainable, if set to
    # None, the gain will be trainable.
    init: str = 'eye'  # controls the initialization of the parameters when the L2RU param is chosen.

    # Parallel scan must be selected in the forward call of the SSM.

    """ Scaffolding Layers """


class MLP(nn.Module):
    """ Standard Transformer MLP """

    def __init__(self, config: SSMConfig):
        super().__init__()
        # Pre-compute hidden dimension for efficiency
        self.hidden_dim = config.dim_amp * config.d_model

        self.c_fc = nn.Linear(config.d_model, self.hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(self.hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class LMLP(nn.Module):
    """ Implements a Lipschitz.-bounded MLP with sandwich layers. The square root
    # of the Lipschitz bound is given by the scale parameter, by default, set to 1. """

    def __init__(self, config: SSMConfig):
        super().__init__()
        # Pre-compute hidden dimension for efficiency
        hidden_dim = config.dim_amp * config.d_model

        # Layer construction using list comprehension
        layers = [
            FirstChannel(config.d_model, scale=config.scale),
            SandwichFc(config.d_model, hidden_dim, bias=False, scale=config.scale),
            SandwichFc(hidden_dim, hidden_dim, bias=False, scale=config.scale),
            SandwichFc(hidden_dim, hidden_dim, bias=False, scale=config.scale),
            SandwichLin(hidden_dim, config.d_model, bias=False, scale=config.scale)
        ]

        # Only add dropout if needed
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)


class GLU(nn.Module):
    """ The static non-linearity used in the S4 paper """

    def __init__(self, config: SSMConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # More efficient sequential construction
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        return self.output_linear(x)

    """ SSMs blocks ----------------------------------------- """


class SSL(nn.Module):
    """ State Space Layer: LRU --> MLP + skip connection """

    def __init__(self, config: SSMConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        # LRU initialization depending on the chosen architecture
        if config.param is None or config.param == "lru":
            self.lru = LRU(in_features=config.d_model, out_features=config.d_model, state_features=config.d_state,
                           rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        elif config.param == "l2ru":
            self.lru = LRU_Robust(state_features=config.d_model, init=config.init)
        elif config.param == "zak":
            self.lru = LRUZ(input_features=config.d_model, output_features=config.d_model,
                            state_features=config.d_state,
                            rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        else:
            raise ValueError(f"Invalid parametrization")

        # Dictionary for layer selection
        ff_layers = {
            "GLU": lambda: GLU(config),
            "MLP": lambda: MLP(config),
            "LMLP": lambda: LMLP(config)
        }

        if config.ff not in ff_layers:
            raise ValueError(f"Unknown feedforward type: {config.ff}")

        self.ff = ff_layers[config.ff]()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, state=None, mode: str = "loop"):
        z = x
        # z = self.ln(z)  # pre-norm

        z, st = self.lru(z, state=state, mode=mode)
        z = self.ff(z)  # MLP, GLU or LMLP
        z = self.dropout(z)  # inactive by default

        # Residual connection
        return z + x, st


class DeepSSM(nn.Module):
    """ Deep SSM: encoder --> cascade of n SSM blocks --> decoder  """

    def __init__(self, n_u: int, n_y: int, config: SSMConfig):
        super().__init__()

        self.config = config

        # Simplified initialization - only handle trainable gamma for LRU_Robust
        if config.param is not None and config.gamma is not None:
            # Fixed-γ: register buffer and use raw Parameters
            self.register_buffer('gamma_t', torch.tensor(config.gamma))
            self.encoder = nn.Parameter(torch.randn(config.d_model, n_u))
            self.decoder = nn.Parameter(torch.randn(n_y, config.d_model))
        else:
            # All other cases (no γ or trainable γ): simple Linear layers
            self.encoder = nn.Linear(n_u, config.d_model, bias=False)
            self.decoder = nn.Linear(config.d_model, n_y, bias=False)

        self.blocks = nn.ModuleList([SSL(config) for _ in range(config.n_layers)])

    def forward(self, u, state=None, gamma=None, mode="scan"):
        """
            Initial pre-processing common to all methods.
        """

        # Initialize states for all layers if not provided
        if state is None:
            layer_states = [None] * len(self.blocks)
        else:
            layer_states = state if isinstance(state, list) else [state] * len(self.blocks)

        # Pre-allocate output tensor with correct dimensions
        if isinstance(self.decoder, nn.Linear):
            output_dim = self.decoder.out_features
        else:
            output_dim = self.decoder.shape[0]


        # Process encoder once for the entire input sequence
        if isinstance(self.encoder, nn.Linear):
            x = self.encoder(u)  # (B, L, d_model)
        else:
            x = u @ self.encoder.T

        # Layer processing (Deep LRU cascade)
        for layer_idx, block in enumerate(self.blocks):
            # Pass through the SSL block
            x, st = block(x, state=layer_states[layer_idx], mode=mode)
            layer_states[layer_idx] = st

        # Final decoding step: handle fixed gamma case if needed (decoder rescaling)

        if self.config.param is not None and self.config.gamma is not None:
            """
            This is the case where we use a fixed gamma for LRU_Robust: need to rescale the decoder
            according to the product of the individual LRU gammas to ensure the overall gain is gamma_t.
            Note that this is only valid when using LRU_Robust blocks, otherwise the gamma values are not defined.
            """
            # Handle the fixed gamma case for LRU_Robust
            gamma_t = torch.abs(self.gamma_t) if gamma is None else gamma
            gammaLRU = [torch.abs(block.lru.gamma) for block in self.blocks]
            gammaLRU_tensor = torch.stack(gammaLRU)
            encoder_norm = torch.linalg.matrix_norm(self.encoder, 2)
            decoder_norm = torch.linalg.matrix_norm(self.decoder, 2)
            gamma_prod = torch.prod(gammaLRU_tensor) + 1
            decoder_scaled = (gamma_t * self.decoder) / (encoder_norm * decoder_norm * gamma_prod)
            outputs = x @ decoder_scaled.T
        else:
            if isinstance(self.decoder, nn.Linear):
                outputs = self.decoder(x)
            else:
                outputs = x @ self.decoder.T

        return outputs, layer_states

    def reset(self):
        # Reset initial states of LTI systems in the LRU blocks
        for block in self.blocks:
            block.lru.reset()


# Pure LRU blocks -----------------------------------------------

class PureLRUR(nn.Module):
    """ A pure robust LRU block without any scaffolding. """

    def __init__(self, n: int, gamma: float = None, param: str = 'l2ru', init: str = "eye"):
        super().__init__()
        if param == 'l2ru':  # In this case the LTI system is necessarily square
            self.lru = LRU_Robust(state_features=n, gamma=gamma, init=init)
        elif param == 'zak':  # Can handle non-square LTI systems
            self.lru = LRUZ(input_features=n, output_features=n, state_features=n, gamma=gamma)

    def forward(self, x, mode: str = "scan"):
        y, st = self.lru(x, mode=mode)

        return y, st
