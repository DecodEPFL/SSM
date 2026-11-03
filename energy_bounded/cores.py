"""Small causal GRU/TCN cores using native PyTorch kernels, with explicit state."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .state import valid_mask


class _CompactMaskedCore(nn.Module):
    """Fast dense path; masked reference compacts valid tokens independently per row.

    Invalid positions pause logical time, even for interior/left padding. The
    masked path loops over batch rows, never over time. No state is detached.
    """

    def forward(self, u, state=None, *, mask=None):
        valid = valid_mask(u, mask)
        if u.shape[-1] != self.d_input:
            raise ValueError("Wrong input feature dimension")
        state = self.init_state(u.shape[0], device=u.device, dtype=u.dtype) if state is None else state
        if u.shape[1] == 0:
            return u.new_empty(u.shape[0], 0, self.d_output), state
        if bool(valid.all()):
            return self._dense(u, state)
        outputs, states = [], []
        for b in range(u.shape[0]):
            indices = valid[b].nonzero().squeeze(-1)
            initial = self._slice(state, b)
            row = u.new_zeros(1, u.shape[1], self.d_output)
            if indices.numel():
                compact, last = self._dense(u[b:b + 1, indices], initial)
                row = row.index_copy(1, indices, compact)
            else:
                last = initial
            outputs.append(row)
            states.append(last)
        return torch.cat(outputs, 0), self._merge(states)

    def step(self, state, u, *, mask=None):
        if u.ndim != 2:
            raise ValueError("step expects (batch,features), with explicit state first")
        y, state = self(u[:, None], state, mask=None if mask is None else mask[:, None])
        return y[:, 0], state


class GRUCore(_CompactMaskedCore):
    def __init__(self, d_input, d_output, *, width=24, layers=1):
        super().__init__()
        self.d_input, self.d_output = d_input, d_output
        self.width, self.layers = width, layers
        self.in_proj = nn.Linear(d_input, width)
        self.gru = nn.GRU(width, width, layers, batch_first=True)
        self.out_proj = nn.Linear(width, d_output)

    def init_state(self, batch_size, *, device=None, dtype=None):
        p = next(self.parameters())
        return torch.zeros(self.layers, batch_size, self.width,
                           device=device or p.device, dtype=dtype or p.dtype)

    def _dense(self, u, state):
        x, state = self.gru(F.gelu(self.in_proj(u)), state)
        return self.out_proj(x), state

    def _slice(self, state, b):
        return state[:, b:b + 1].contiguous()

    def _merge(self, states):
        return torch.cat(states, 1)


class TCNCore(_CompactMaskedCore):
    """Residual dilated convolutions, one temporal convolution per layer.

    Receptive field = 1 + (kernel_size-1) * sum(dilations), in valid tokens.
    Caches contain each layer's INPUT, including the projected zero-history
    boundary. No normalization over time and no output/energy feedback.
    """

    def __init__(self, d_input, d_output, *, width=24, dilations=(1, 2, 4), kernel_size=3):
        super().__init__()
        if kernel_size < 1 or not dilations or any(d < 1 for d in dilations):
            raise ValueError("Positive kernel size and nonempty positive dilations required")
        self.d_input, self.d_output, self.width = d_input, d_output, width
        self.history = tuple((kernel_size - 1) * d for d in dilations)
        self.receptive_field = 1 + sum(self.history)
        self.in_proj = nn.Linear(d_input, width)
        self.convs = nn.ModuleList(nn.Conv1d(width, width, kernel_size, dilation=d)
                                   for d in dilations)
        self.out_proj = nn.Linear(width, d_output)

    def init_state(self, batch_size, *, device=None, dtype=None):
        p = next(self.parameters())
        return tuple(torch.zeros(batch_size, self.width, h,
                                 device=device or p.device, dtype=dtype or p.dtype)
                     for h in self.history)

    def _dense(self, u, state):
        x = F.gelu(self.in_proj(u)).transpose(1, 2)
        buffers = []
        for conv, previous, history in zip(self.convs, state, self.history):
            joined = torch.cat((previous, x), -1)
            buffers.append(joined[..., -history:].clone() if history else joined[..., :0])
            x = x + F.gelu(conv(joined))
        return self.out_proj(x.transpose(1, 2)), tuple(buffers)

    def _slice(self, state, b):
        return tuple(t[b:b + 1] for t in state)

    def _merge(self, states):
        return tuple(torch.cat(parts, 0) for parts in zip(*states))
