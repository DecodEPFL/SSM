"""Explicit state utilities; no implicit detach or module-owned trajectory state."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Protocol

import torch
from torch import Tensor


class CausalCore(Protocol):
    d_input: int
    d_output: int

    def init_state(self, batch_size: int, *, device=None, dtype=None) -> Any: ...

    def forward(self, u: Tensor, state: Any = None, *, mask=None) -> tuple: ...

    def step(self, state: Any, u: Tensor, *, mask=None) -> tuple: ...


def map_state(state: Any, fn: Callable[[Tensor], Tensor]) -> Any:
    """Map tensor trees, including KV-cache dataclasses. Custom objects need an adapter."""
    if isinstance(state, Tensor):
        return fn(state)
    if state is None or isinstance(state, (str, int, float, bool)):
        return state
    if is_dataclass(state):
        return type(state)(**{f.name: map_state(getattr(state, f.name), fn) for f in fields(state)})
    if isinstance(state, dict):
        return {k: map_state(v, fn) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(map_state(v, fn) for v in state)
    raise TypeError(f"Provide a state adapter for {type(state).__name__}")


def detach_state(state: Any) -> Any:
    """Explicit truncated BPTT boundary, including reservoir energy."""
    return map_state(state, Tensor.detach)


def state_to(state: Any, *, device=None, dtype=None, energy_dtype=None) -> Any:
    """Move cache tensors; preserve integer positions and boolean padding masks."""
    if isinstance(state, WrapperState):
        tank_dtype = energy_dtype or (torch.float64 if dtype == torch.float64 else state.energy.dtype)
        if tank_dtype not in (torch.float32, torch.float64):
            raise ValueError("Reservoir energy must retain at least FP32 precision")
        if device is not None and torch.device(device).type == "mps" and tank_dtype == torch.float64:
            raise ValueError("MPS requires explicit energy_dtype=torch.float32 when moving FP64 state")
        return WrapperState(state_to(state.core, device=device, dtype=dtype),
                            state.energy.to(device=device, dtype=tank_dtype))
    return map_state(state, lambda t: t.to(device=device, dtype=dtype if t.is_floating_point() else None))


def check_finite(state: Any) -> None:
    def check(t):
        if t.is_floating_point() and not bool(torch.isfinite(t).all()):
            raise FloatingPointError("Nonfinite input, core output/state, or reservoir arithmetic")
        return t
    map_state(state, check)


def state_bytes(state: Any) -> int:
    """Logical persistent tensor bytes, excluding allocator and autograd graph overhead."""
    sizes = []
    map_state(state, lambda t: sizes.append(t.numel() * t.element_size()) or t)
    return sum(sizes)


def valid_mask(u: Tensor, mask=None) -> Tensor:
    if u.ndim != 3 or not u.is_floating_point() or min(u.shape[0], u.shape[2]) <= 0:
        raise ValueError("Expected floating (batch, time, features), with nonempty batch/features")
    if mask is None:
        return torch.ones(u.shape[:2], device=u.device, dtype=torch.bool)
    if mask.shape != u.shape[:2] or mask.dtype != torch.bool or mask.device != u.device:
        raise ValueError("mask must be boolean (batch,time) on the input device; True means valid")
    return mask


@dataclass
class WrapperState:
    core: Any
    energy: Tensor
