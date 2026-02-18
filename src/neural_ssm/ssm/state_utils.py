from __future__ import annotations

from typing import Optional

import torch


def init_or_cast_state(
    state: Optional[torch.Tensor],
    batch_size: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if state is None:
        return torch.zeros(batch_size, n_state, device=device, dtype=dtype)

    if state.dim() == 1:
        state = state.unsqueeze(0).expand(batch_size, -1)
    elif state.dim() == 2:
        if state.size(0) == 1 and batch_size > 1:
            state = state.expand(batch_size, -1)
    else:
        raise ValueError("state must have shape (N,) or (B,N)")

    if state.shape != (batch_size, n_state):
        raise ValueError(f"state has shape {tuple(state.shape)}, expected {(batch_size, n_state)}")

    return state.to(device=device, dtype=dtype)


def resolve_runtime_state(
    explicit_state: Optional[torch.Tensor],
    internal_state: Optional[torch.Tensor],
    *,
    reset_state: bool,
    batch_size: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if reset_state:
        internal_state = None
    source_state = explicit_state if explicit_state is not None else internal_state
    if explicit_state is None and source_state is not None:
        # Internal state is best-effort: if shape no longer matches current batch,
        # reinitialize from zeros instead of crashing.
        if source_state.dim() == 1:
            if source_state.shape[0] != n_state:
                source_state = None
        elif source_state.dim() == 2:
            if source_state.shape[1] != n_state or source_state.shape[0] not in (1, batch_size):
                source_state = None
        else:
            source_state = None
    return init_or_cast_state(source_state, batch_size, n_state, device, dtype)


def reset_runtime_state(state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if state is None:
        return None
    return torch.zeros_like(state)
