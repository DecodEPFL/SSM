"""Exact unit-ball spending head and multiplicative scalar-storage dynamics."""
from __future__ import annotations

import math

import torch
from torch import nn

from energy_bounded.state import check_finite, valid_mask
from src.neural_ssm.utils.scan import associative_scan, binary_operator_diag


def safe_sqrt(x):
    """sqrt for nonnegative energy, with finite zero subgradient at zero."""
    positive = x > 0
    return torch.where(positive, torch.where(positive, x, torch.ones_like(x)).sqrt(), 0)


def unit_ball_head(z):
    """z/max(1,||z||), computed by scaling to avoid overflowing a finite raw norm.

    Returns p, its squared norm, and ||z||>1. Saturated coefficients are exactly
    zero: the mathematical squared norm of a normalized vector is exactly one.
    """
    z = z.to(torch.promote_types(z.dtype, torch.float32))
    scale = z.abs().amax(-1, keepdim=True).clamp(min=1)
    scaled = z / scale
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    inverse = scale.reciprocal()
    boundary = norm >= inverse
    # At equality choose the normalized branch for both p and its norm, giving
    # a consistent finite head-boundary gradient rather than mixed tie rules.
    p = scaled / torch.where(boundary, norm, inverse)
    fraction = torch.where(boundary, torch.ones_like(norm), p.square().sum(-1, keepdim=True))
    return p, fraction.clamp(0, 1), norm > inverse


def affine_prefix(a, b):
    """Work-efficient existing differentiable pair scan: O(BT) work, O(log T) depth.

    Reuses the repository's recursive adjacent-pair scan, not cumulative-product
    division or a prototype backend API. No CPU transfer or hidden fallback.
    """
    return associative_scan(binary_operator_diag, (a, b), axis=1)


class StorageLayer(nn.Module):
    def __init__(self, gamma=1., *, implementation="scan", precision="auto"):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        if implementation not in ("scan", "reference") or precision not in ("auto", "float32", "float64"):
            raise ValueError("Invalid storage implementation or precision")
        self.gamma, self.implementation, self.precision = float(gamma), implementation, precision

    def energy_dtype(self, device, input_dtype):
        if self.precision == "float64" or input_dtype == torch.float64:
            if torch.device(device).type == "mps":
                raise ValueError("MPS does not support FP64 energy; use float32 explicitly")
            return torch.float64
        if self.precision == "float32" or torch.device(device).type == "mps":
            return torch.float32
        return torch.float64

    def forward(self, v, z, storage=None, *, mask=None, implementation=None, diagnostics=False, gamma=None):
        valid = valid_mask(v, mask)
        if (z.ndim != 3 or z.shape[:2] != v.shape[:2] or z.shape[-1] < 1
                or z.device != v.device or not z.is_floating_point()):
            raise ValueError("z must be floating (batch,time,output features) on the input device")
        dtype = self.energy_dtype(v.device, torch.promote_types(v.dtype, z.dtype))
        output_dtype = torch.promote_types(z.dtype, torch.float32)
        vv = torch.where(valid[..., None], v, 0).to(dtype)
        zz = torch.where(valid[..., None], z, 0).to(dtype)
        initial = torch.zeros(v.shape[0], 1, device=v.device, dtype=dtype) if storage is None else storage
        if initial.shape != (v.shape[0], 1) or initial.device != v.device or bool((initial < 0).any()):
            raise ValueError("Incoming storage must be nonnegative (batch,1) on the input device")
        initial = initial.to(dtype)
        check_finite((vv, zz, initial))
        p, fraction, saturated = unit_ball_head(zz)
        gain = torch.as_tensor(self.gamma if gamma is None else gamma, dtype=dtype, device=v.device)
        if gain.ndim != 0 or not bool(torch.isfinite(gain)) or bool(gain < 0):
            raise ValueError("Gamma must be a finite nonnegative scalar")
        d = (vv * gain).square().sum(-1, keepdim=True)
        a = torch.where(valid[..., None], 1 - fraction, torch.ones_like(fraction))
        b = a * d
        check_finite((p, d, a, b))
        method = implementation or self.implementation
        if method not in ("scan", "reference"):
            raise ValueError("Unknown storage implementation")
        if v.shape[1] == 0:
            after = initial[:, None, :][:, :0]
            before = after
        elif method == "scan":
            prefix_a, prefix_b = affine_prefix(a, b)
            after = prefix_a * initial[:, None] + prefix_b
            before = torch.cat((initial[:, None], after[:, :-1]), 1)
        else:
            current, states = initial, []
            for t in range(v.shape[1]):
                current = a[:, t] * (current + d[:, t])
                states.append(current)
            after = torch.stack(states, 1)
            before = torch.cat((initial[:, None], after[:, :-1]), 1)
        available = before + d
        y = (p * safe_sqrt(available)).to(output_dtype)
        y = torch.where(valid[..., None], y, 0)
        last = after[:, -1] if v.shape[1] else initial
        check_finite((after, y, last))
        info = None
        if diagnostics:
            count = valid.sum().clamp(min=1)
            spent = y.to(dtype).square().sum(-1).cumsum(1)
            supplied = initial + d[..., 0].cumsum(1)
            residual = (spent - supplied) / (1 + supplied)
            info = {"storage": torch.cat((initial[:, None], after), 1),
                    "spending_fraction": torch.where(valid[..., None], fraction, 0),
                    "mean_spending_fraction": (fraction[..., 0] * valid).sum() / count,
                    "head_saturation_rate": (saturated[..., 0] & valid).sum() / count,
                    "max_normalized_energy_violation": residual.clamp(min=0).amax() if residual.numel() else initial.sum() * 0,
                    "max_normalized_identity_error": ((after[..., 0] + spent - supplied) / (1 + supplied)).abs().amax() if residual.numel() else initial.sum() * 0,
                    "implementation": method}
        return y, last, info

    def step(self, storage, v, z, *, mask=None, diagnostics=False, gamma=None):
        if v.ndim != 2 or z.ndim != 2:
            raise ValueError("step inputs must be (batch,features)")
        y, storage, info = self(v[:, None], z[:, None], storage,
                                mask=None if mask is None else mask[:, None],
                                implementation="reference", diagnostics=diagnostics, gamma=gamma)
        return y[:, 0], storage, info
