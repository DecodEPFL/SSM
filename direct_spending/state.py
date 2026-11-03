"""Explicit detach/device conversion, including FP32-minimum storage precision."""
import torch

from energy_bounded.state import detach_state, map_state

from .model import SpendingState


def state_to(state, *, device=None, dtype=None, energy_dtype=None):
    tank_dtype = energy_dtype or (torch.float64 if dtype == torch.float64 else state.storage.dtype)
    if tank_dtype not in (torch.float32, torch.float64):
        raise ValueError("Storage must retain at least FP32 precision")
    if device is not None and torch.device(device).type == "mps" and tank_dtype == torch.float64:
        raise ValueError("Moving FP64 storage to MPS requires explicit energy_dtype=torch.float32")
    core = map_state(state.core, lambda t: t.to(device=device, dtype=dtype if t.is_floating_point() else None))
    return SpendingState(core, state.storage.to(device=device, dtype=tank_dtype),
                         state.initial_beta.to(device=device, dtype=tank_dtype))


__all__ = ["detach_state", "state_to"]
