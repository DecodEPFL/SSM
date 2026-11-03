"""Explicit streaming state, with no implicit truncated-BPTT detach."""
from dataclasses import dataclass
import torch


def accumulation_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


@dataclass(frozen=True)
class SplitMemoryState:
    m: torch.Tensor  # (batch, heads, 2)
    c: torch.Tensor  # (batch, computation_width)


def detach_state(state):
    return SplitMemoryState(state.m.detach(), state.c.detach())


def state_to(state, *, device=None, dtype=None):
    dtype = accumulation_dtype(dtype or state.m.dtype)
    return SplitMemoryState(state.m.to(device=device, dtype=dtype), state.c.to(device=device, dtype=dtype))
