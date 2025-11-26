from dataclasses import dataclass
import torch.nn as nn


@dataclass  # generic dataclass to handle custom nn.modules
class LayerConfig:
    d_input: int = 10  # input size
    d_hidden: int = 32  # hidden size
    d_output: int = 10  # output size
    n_layers: int = 2  # number of static_layers
    dropout: float = 0.0  # set it different from 0 if you want to introduce dropout regularization
    lip: float = 1.0  # Lipschitz bound for lip. bounded MLPs


class GLU(nn.Module):
    """ The static non-linearity used in the S4 paper """

    def __init__(self, config: LayerConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Sequential construction
        self.output_linear = nn.Sequential(
            nn.Linear(config.d_input, 2 * config.d_input),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        return self.output_linear(x)


class MLP(nn.Module):
    """ pretty generic MLP """

    def __init__(self, config: LayerConfig):
        super().__init__()
        # Pre-compute hidden dimension for efficiency
        self.hidden_dim = config.d_hidden
        self.output_dim = config.d_output
        self.n_layers = config.n_layers
        self.input_dim = config.d_input
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        layers = nn.ModuleList()
        layers.append(nn.Linear(self.input_dim, self.hidden_dim, bias=False))
        for i in range(config.n_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
        layers.append(nn.GELU())
        layers.append(nn.Linear(self.hidden_dim, config.d_output, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return self.dropout(x)
