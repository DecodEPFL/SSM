"""Shared runtime conventions: input layout, streaming state, and inference caches.

These helpers do not implement model mathematics. Inputs use (batch, time,
features); state helpers resolve reset/continuation; EvalCacheMixin safely
reuses parameter-derived tensors only during gradient-free evaluation.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable

import torch

# Input shape


def normalize_to_3d(x: torch.Tensor, *, time_first: bool = False) -> torch.Tensor:
    """View an input as (batch, time, features), without copying its data.

    A vector is one sample; a matrix is one sequence. ``time_first`` only
    changes the interpretation of a 3D input, from (time, batch, features).
    Callers that need contiguous storage can request it at the kernel boundary.
    """
    if x.dim() == 1:
        return x[None, None, :]
    if x.dim() == 2:
        return x[None, :, :]
    if x.dim() == 3:
        return x.transpose(0, 1) if time_first else x
    raise ValueError(f"Invalid input dimensions {x.dim()}, expected 1, 2, or 3.")


def sequence_output(output, trajectory, last, *, return_state: bool, return_last: bool):
    """Use the same output/state return convention across sequence models.

    ``trajectory`` includes the initial state and the terminal state. The last
    state always stays batch-first, even when the sequence is time-first.
    """
    if return_state and return_last:
        return output, trajectory, last
    if return_state:
        return output, trajectory
    if return_last:
        return output, last
    return output


def last_runtime_state(state: Any) -> Any:
    """Extract final states from a tensor trajectory or a nested cell state."""
    if state is None:
        return None
    if torch.is_tensor(state):
        return state[:, -1, :] if state.ndim >= 3 else state
    if isinstance(state, tuple):
        return tuple(last_runtime_state(item) for item in state)
    if isinstance(state, list):
        return [last_runtime_state(item) for item in state]
    raise TypeError(f"Unsupported state trajectory type: {type(state).__name__}.")


# Streaming state


def init_or_cast_state(
    state: torch.Tensor | None,
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
        if state.size(0) == 1:
            # Always expand (even when batch_size==1) so that if x0 is an nn.Parameter,
            # expand() returns a plain Tensor view instead of the Parameter itself.
            # Autograd still tracks the view, so gradients reach x0_param.
            state = state.expand(batch_size, -1)
    else:
        raise ValueError("state must have shape (N,) or (B,N)")

    if state.shape != (batch_size, n_state):
        raise ValueError(f"state has shape {tuple(state.shape)}, expected {(batch_size, n_state)}")

    return state.to(device=device, dtype=dtype)


def resolve_runtime_state(
    explicit_state: torch.Tensor | None,
    internal_state: torch.Tensor | None,
    *,
    reset_state: bool,
    batch_size: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    if reset_state:
        # Use learnable x0 if provided, otherwise fall back to zeros via init_or_cast_state.
        internal_state = x0
    source_state = explicit_state if explicit_state is not None else internal_state
    if explicit_state is None and source_state is not None:
        # Internal state is best-effort: if shape no longer matches current batch,
        # fall back to x0 (or zeros) instead of crashing.
        if source_state.dim() == 1:
            if source_state.shape[0] != n_state:
                source_state = x0
        elif source_state.dim() == 2:
            if source_state.shape[1] != n_state or source_state.shape[0] not in (1, batch_size):
                source_state = x0
        else:
            source_state = x0
    return init_or_cast_state(source_state, batch_size, n_state, device, dtype)


def reset_runtime_state(
    state: torch.Tensor | None,
    x0: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Reset state to x0 (if learnable) or zeros."""
    if x0 is not None:
        return x0.detach()  # (1, n_state) — broadcast to (B, n_state) on next forward
    if state is None:
        return None
    return torch.zeros_like(state)


# Parameter-derived inference caches


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value, **{f.name: _detach_tree(getattr(value, f.name)) for f in fields(value)}
        )
    if isinstance(value, (tuple, list)):
        return type(value)(_detach_tree(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach_tree(v) for k, v in value.items()}
    return value


def _autocast_signature() -> tuple:
    # Matrix products can produce different derived tensors under autocast.
    # Keep the older API fallback for the package's PyTorch 2.2 minimum.
    if hasattr(torch, "get_autocast_dtype"):
        return tuple(
            (enabled, torch.get_autocast_dtype(device) if enabled else None)
            for device in ("cpu", "cuda")
            for enabled in (torch.is_autocast_enabled(device),)
        )
    cpu_enabled = torch.is_autocast_cpu_enabled()
    cuda_enabled = torch.is_autocast_enabled()
    return (
        (cpu_enabled, torch.get_autocast_cpu_dtype() if cpu_enabled else None),
        (cuda_enabled, torch.get_autocast_gpu_dtype() if cuda_enabled else None),
    )


class EvalCacheMixin:
    """Mixin providing eval-only caching of parameter-derived computations.

    Use ``self._eval_cached(key, compute)`` inside a method. ``compute`` is a
    zero-arg callable returning a tensor (or tuple of tensors).

      * Training mode, or any grad-enabled call: always recompute (and drop the
        cached entry) so autograd sees the live computation and the exact
        certificate is preserved bit-for-bit.
      * Eval mode with grad disabled: compute once, cache a detached copy, reuse.

    Entries track parameters and buffers, including identity, mutation version,
    dtype/device, and autocast settings. This covers optimizer updates, checkpoint
    loading, parameter replacement, and fixed gain changes. The whole cache is
    cleared on any ``train()`` / ``eval()`` toggle or device/dtype conversion. The cache
    lives in ``self.__dict__`` (not as a registered buffer), so it never enters
    ``state_dict`` and does not interfere with ``nn.Module.__setattr__``.
    """

    def _eval_cached(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        # Training or any grad-enabled call: recompute so autograd sees the live
        # computation (never serve a detached / possibly-stale-graph tensor).
        if self.training or torch.is_grad_enabled():
            cache = self.__dict__.get("_eval_cache")
            if cache is not None:
                cache.pop(key, None)
            return compute()
        return self._eval_cached_nograd(key, compute)

    @torch.compiler.disable
    def _eval_cached_nograd(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        # Version counters and Python cache mutation belong outside the compiled
        # numerical graph. Training bypasses this boundary and remains traceable.
        # Tensors created inside inference_mode have no mutation counter. A
        # module constructed there must recompute rather than risk stale values.
        param_version = self._param_version()
        if param_version is None:
            return compute()
        autocast = _autocast_signature()
        version = (param_version, autocast)
        cache = self.__dict__.setdefault("_eval_cache", {})
        hit = cache.get(key)
        if hit is not None and hit[0] == version:
            return hit[1]
        value = _detach_tree(compute())
        # Some normalizers update an auxiliary buffer while computing (power
        # iteration, for example). Record its final version so the next call hits.
        cache[key] = ((self._param_version(), autocast), value)
        return value

    def _param_version(self):
        tensors = (*self.parameters(), *self.buffers())
        if any(t.is_inference() for t in tensors):
            return None
        return tuple((id(t), t._version, t.dtype, t.device) for t in tensors)

    def _apply(self, fn, recurse=True):
        self.__dict__.pop("_eval_cache", None)
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True):
        self.__dict__.pop("_eval_cache", None)
        return super().train(mode)
