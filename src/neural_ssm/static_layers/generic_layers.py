"""Ordinary feedforward layers, with no global Lipschitz certificate.

LayerConfig supplies dimensions and optional gain settings to both this module
and lipschitz_mlps.py. GLU and MLP are the conventional baseline choices.
"""
from dataclasses import dataclass

import torch.nn as nn


# Shared settings

@dataclass
class LayerConfig:
    """Shared dimensions; lip/train_lip are used by bounded constructions."""

    d_input: int = 10  # input size
    d_hidden: int = 32  # hidden size
    d_output: int = 10  # output size
    n_layers: int = 2  # additional hidden layers
    dropout: float = 0.0
    lip: float = 1.0  # scale/budget interpreted by the chosen bounded construction
    train_lip: bool = True  # if False, keep the FF Lipschitz scale fixed at `lip`

# Ordinary gated layer

class GLU(nn.Module):
    """Pointwise GELU, optional dropout, then a learned GLU projection."""

    def __init__(self, config: LayerConfig):
        super().__init__()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.output_linear = nn.Sequential(
            nn.Linear(config.d_input, 2 * config.d_input),
            nn.GLU(dim=-1),
        )

    def forward(self, x):
        x = self.dropout(self.activation(x))
        return self.output_linear(x)

# Ordinary multilayer perceptron

class MLP(nn.Module):
    """Linear hidden stack, one GELU, then an output projection and dropout."""

    def __init__(self, config: LayerConfig):
        super().__init__()
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

__all__ = ["LayerConfig", "GLU", "MLP"]
