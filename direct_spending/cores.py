"""Spending cores predict dimensionless z, independently of storage and outputs."""
from __future__ import annotations

import torch
from torch import nn

from energy_bounded.cores import GRUCore, TCNCore, _CompactMaskedCore
from energy_bounded.state import valid_mask
from energy_bounded.transformer import TransformerCore


class StaticMLPCore(nn.Module):
    """Shared pointwise map; no neural dynamic state. Not an arbitrary dynamical model."""

    def __init__(self, d_input, d_output, *, width=16, layers=2):
        super().__init__()
        self.d_input, self.d_output = d_input, d_output
        modules, incoming = [], d_input
        for _ in range(layers):
            modules.extend((nn.Linear(incoming, width), nn.GELU()))
            incoming = width
        modules.append(nn.Linear(incoming, d_output))
        self.net = nn.Sequential(*modules)

    def init_state(self, batch_size, *, device=None, dtype=None):
        return None

    def forward(self, u, state=None, *, mask=None):
        valid = valid_mask(u, mask)
        if u.shape[-1] != self.d_input:
            raise ValueError("Wrong input feature dimension")
        z = self.net(torch.where(valid[..., None], u, 0))
        return torch.where(valid[..., None], z, 0), None

    def step(self, state, u, *, mask=None):
        z, state = self(u[:, None], state, mask=None if mask is None else mask[:, None])
        return z[:, 0], state


class LSTMCore(_CompactMaskedCore):
    """Native fused sequence LSTM; state is the explicit (hidden, cell) tuple.

    Reuses the same padding compaction adapter as GRU/TCN. Dense execution calls
    nn.LSTM once for the complete chunk, with no Python loop over time.
    """

    def __init__(self, d_input, d_output, *, width=16, layers=2):
        super().__init__()
        self.d_input, self.d_output = d_input, d_output
        self.width, self.layers = width, layers
        self.in_proj = nn.Linear(d_input, width)
        self.lstm = nn.LSTM(width, width, layers, batch_first=True)
        self.out_proj = nn.Linear(width, d_output)

    def init_state(self, batch_size, *, device=None, dtype=None):
        parameter = next(self.parameters())
        zero = lambda: torch.zeros(self.layers, batch_size, self.width,
                                   device=device or parameter.device, dtype=dtype or parameter.dtype)
        return zero(), zero()

    def _dense(self, u, state):
        x, state = self.lstm(torch.nn.functional.gelu(self.in_proj(u)), state)
        return self.out_proj(x), state

    def _slice(self, state, b):
        return tuple(part[:, b:b + 1].contiguous() for part in state)

    def _merge(self, states):
        return tuple(torch.cat(parts, 1) for parts in zip(*states))


def build_core(d_input, d_output, config):
    common = {"width": config.width}
    if config.core == "static_mlp":
        return StaticMLPCore(d_input, d_output, layers=config.layers, **common)
    if config.core == "gru":
        return GRUCore(d_input, d_output, layers=config.layers, **common)
    if config.core == "lstm":
        return LSTMCore(d_input, d_output, layers=config.layers, **common)
    if config.core == "tcn":
        return TCNCore(d_input, d_output, kernel_size=config.tcn_kernel,
                       dilations=config.tcn_dilations, **common)
    return TransformerCore(d_input, d_output, layers=config.layers, heads=config.heads,
                           dropout=config.dropout, **common)
