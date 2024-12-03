import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from .lru import LRU, LRU_Robust
from .L_bounded_MLPs import FirstChannel, SandwichFc, SandwichLin


""" Data class to set up the model (values here are used just to initialize all fields) """


@dataclass
class DWNConfig:
    d_model: int = 10 # input/output size of the LRU (u and y)
    d_state: int = 64 # state size of the LRU (n)
    n_layers: int = 6 # number of SSMs blocks in cascade for deep structures
    dropout: float = 0.0 # set it different from 0 if you want to introduce dropout regularization
    bias: bool = True # bias of MLP layers
    rmin: float = 0.0 # min. magnitude of the eigenvalues at initialization in the complex parametrization
    rmax: float = 1.0 # max. magnitude of the eigenvalues at initialization in the complex parametrization
    max_phase: float = 2 * math.pi # maximum phase of the eigenvalues at initialization in the complex parametrization
    ff: str = "MLP" # non-linear block used in the scaffolding
    scale: float = 1 # Lipschitz constant of the Lipschitz bounded MLP (LMLP)
    dim_amp: int = 4 # controls the hidden layer's dimension of the MLP
    gamma: bool = True # set this to true if you want to use the l2 gain parametrization for the SSM. If set to false,
    # the complex diagonal parametrization of the LRU will be used instead.
    gain: float = 8 # set the overall l2 gain in case you want to keep it fixed and not trainable
    trainable: bool = True # set this to true if you want a trainable l2 gain.

    # Parallel scan must be selected in the forward call. It will be disabled when gamma is set to True.


    """ Scaffolding Layers """


class MLP(nn.Module):
    """ Standard Transformer MLP """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.dim_amp * config.d_model, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.dim_amp * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LMLP(nn.Module):
    """ Implements a Lipschitz.-bounded MLP with sandwich layers. The square root
    # of the Lipschitz bound is given by scale """

    def __init__(self, config: DWNConfig):
        super().__init__()
        layers = [FirstChannel(config.d_model, scale=config.scale),
                  SandwichFc(config.d_model, config.dim_amp * config.d_model, bias=False, scale=config.scale),
                  SandwichFc(config.dim_amp * config.d_model, config.dim_amp * config.d_model, bias=False,
                             scale=config.scale),
                  SandwichFc(config.dim_amp * config.d_model, config.dim_amp * config.d_model, bias=False,
                             scale=config.scale),
                  SandwichLin(config.dim_amp * config.d_model, config.d_model, bias=False, scale=config.scale),
                  nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        x = self.model(input)
        return x


class GLU(nn.Module):
    """ The static non-linearity used in the S4 paper """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_model, 2 * config.d_model),
            # nn.Conv1d(config.d_model, 2 * config.d_model, kernel_size=1),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        x = self.output_linear(x)
        return x


    """ SSMs blocks """


class DWNBlock(nn.Module):
    """ SSM block: LRU --> MLP + skip connection """

    def __init__(self, config: DWNConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        if config.gamma:
            self.lru = LRU_Robust(config.d_model, config.trainable)

        else:
            self.lru = LRU(config.d_model, config.d_model, config.d_state,
                           rmin=config.rmin, rmax=config.rmax, max_phase=config.max_phase)
        match config.ff:
            case "GLU":
                self.ff = GLU(config)
            case "MLP":
                self.ff = MLP(config)
            case "LMLP":
                self.ff = LMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, gamma=None, state=None, mode: str ="scan"):

        z = x
        #  z = self.ln(z)  # prenorm

        z = self.lru(z, gamma, state, mode)

        z = self.ff(z)  # MLP, GLU or LMLP
        z = self.dropout(z)

        # Residual connection
        x = z + x

        return x


class DWN(nn.Module):
    """ Deep SSMs block: encoder --> cascade of n SSMs --> decoder  """

    def __init__(self, n_u: int, n_y: int, config: DWNConfig):
        super().__init__()

        self.config = config

        self.encoder = nn.Linear(n_u, config.d_model, bias=False)
        self.decoder = nn.Linear(config.d_model, n_y, bias=False)

        if not config.trainable: # parameters needed for when the l2 gain is fixed and prescribed
            self.alpha = nn.Parameter(torch.randn(1))
            self.gamma_e = nn.Parameter(torch.randn(1))
            self.register_buffer('gamma_t', torch.tensor(config.gain))
            self.encoder = nn.Parameter(torch.randn(n_u, config.d_model))
            self.decoder = nn.Parameter(torch.randn(config.d_model, n_y))




        self.blocks = nn.ModuleList([DWNBlock(config) for _ in range(config.n_layers)])

    def forward_fixed_gamma(self, u, state=None, mode="scan"):

        gamma_t = torch.abs(self.gamma_t)
        gamma_e = torch.abs(self.gamma_e)
        gamma_mid = torch.pow(1 / torch.sigmoid(self.alpha), 1 / self.config.n_layers) * torch.ones(
            self.config.n_layers)
        gamma_d = torch.sigmoid(self.alpha) * gamma_t / self.gamma_e
        encoder = gamma_e * self.encoder / torch.norm(self.encoder, 2)
        decoder = gamma_d * self.decoder / torch.norm(self.decoder, 2)
        x = u@encoder
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x = block(x, gamma_mid[layer]-1, state=state_block, mode=mode)
        x = x@decoder

        return x

    def forward_trainable_gamma(self, u, state=None, mode="scan"):

        x = self.encoder(u)
        for layer, block in enumerate(self.blocks):
            state_block = state[layer] if state is not None else None
            x = block(x, state=state_block, mode=mode)
        x = self.decoder(x)

        return x


    def forward(self, u, state=None, mode="scan"):

        if not self.config.trainable:
            x = self.forward_fixed_gamma(u, state, mode)
        else:
            x = self.forward_trainable_gamma(u, state, mode)

        return x
