"""Shared gain and storage algebra for the SSM stacks.

These functions prepare certificates; they do not implement a recurrence.
Keeping them here leaves layers.py focused on the model and its forward pass.
Cell-specific algebra lives separately in cells/common.py.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def has_gain_contract(module: nn.Module) -> bool:
    return callable(getattr(module, "gain_bound", None)) or hasattr(module, "gamma")


def _module_bound(module, method, attribute, *, device, dtype):
    """Prefer an explicit bound method, while accepting historical attributes."""
    hook = getattr(module, method, None)
    bound = hook() if callable(hook) else getattr(module, attribute, None)
    if bound is None:
        return torch.full((), float("inf"), device=device, dtype=dtype)
    return torch.as_tensor(bound, device=device, dtype=dtype).abs()


def module_gain_bound(module: nn.Module, *, device, dtype) -> torch.Tensor:
    """Read a recurrent cell's gain_bound() or legacy gamma contract."""
    return _module_bound(module, "gain_bound", "gamma", device=device, dtype=dtype)


def module_lip_bound(module: nn.Module, *, device, dtype) -> torch.Tensor:
    """Read a feedforward branch's lip_bound() or legacy lip contract."""
    return _module_bound(module, "lip_bound", "lip", device=device, dtype=dtype)


def state_energy(state: Any, *, device, dtype) -> torch.Tensor | None:
    """Squared state norm per batch element, including nested/complex states."""
    if state is None:
        return None
    if torch.is_tensor(state):
        flat = state.reshape(state.shape[0], -1) if state.ndim > 1 else state.reshape(1, -1)
        # vector_norm uses magnitudes for complex states and returns real energy.
        return torch.linalg.vector_norm(flat, dim=-1).square().to(device=device, dtype=dtype)
    if isinstance(state, (tuple, list)):
        parts = [state_energy(item, device=device, dtype=dtype) for item in state]
        parts = [part for part in parts if part is not None]
        return torch.stack(parts).sum(dim=0) if parts else None
    raise TypeError(f"Unsupported state type: {type(state).__name__}.")


def spectrally_capped_weight(weight: torch.Tensor, *, bound: float = 1.0) -> torch.Tensor:
    """Cap the spectral norm without amplifying weights below the bound."""
    if bound <= 0.0:
        raise ValueError(f"bound must be positive, got {bound}.")
    norm_input = weight.float() if weight.dtype in (torch.float16, torch.bfloat16) else weight
    sigma = torch.linalg.matrix_norm(norm_input, ord=2).to(device=weight.device, dtype=weight.dtype)
    return weight / torch.clamp(sigma / float(bound), min=1.0)


def smooth_capped_log_scale_from_logs(gamma_t, log_gamma_prod, temperature):
    """Log of a decoder scale no larger than min(1, gamma_t / gamma_prod).

    Softplus overestimates max(0, log(gamma_prod / gamma_t)), so its negative
    gives a safe smooth cap. Stay in log space: deep stacks can underflow the
    scale, and taking log(clamp(scale)) would then invalidate the certificate.
    """
    tau = max(float(temperature), 1e-6)
    gamma_t = gamma_t.abs()
    log_gamma_t = torch.log(gamma_t.clamp_min(torch.finfo(gamma_t.dtype).tiny))
    log_scale = -tau * F.softplus((log_gamma_prod - log_gamma_t) / tau)
    return torch.where(gamma_t > 0, log_scale, torch.full_like(log_scale, float("-inf")))


def smooth_capped_scale_from_logs(gamma_t, log_gamma_prod, temperature):
    return torch.exp(smooth_capped_log_scale_from_logs(gamma_t, log_gamma_prod, temperature))


def smooth_capped_scale(gamma_t, gamma_prod, temperature):
    log_product = torch.log(gamma_prod.abs().clamp_min(torch.finfo(gamma_prod.dtype).tiny))
    return smooth_capped_scale_from_logs(gamma_t, log_product, temperature)
