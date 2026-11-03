"""Generic output-energy reservoir, with online reference and portable prefix scan."""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
from torch import Tensor, nn

from .state import WrapperState, check_finite, valid_mask


class EnergyReservoir(nn.Module):
    """One scalar tank per sample. gamma is fixed, positive, and checkpointed.

    auto uses FP64 accumulation on CPU/CUDA and FP32 on MPS. Outputs retain at
    least FP32 precision. The scan has no time loop; an audited cancellation
    failure warns and falls back to the sequential reference (or raises).
    """

    def __init__(self, gamma: float = 1.0, *, accumulation: str = "auto",
                 cancellation: str = "reference", check_nonfinite: bool = True):
        super().__init__()
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and strictly positive")
        if accumulation not in ("auto", "float32", "float64"):
            raise ValueError("Unknown accumulation dtype")
        if cancellation not in ("reference", "raise"):
            raise ValueError("cancellation must be reference or raise")
        # Python scalar extra state avoids silently rounding gamma under model.half().
        self.gamma = float(gamma)
        self.accumulation, self.cancellation = accumulation, cancellation
        self.check_nonfinite = check_nonfinite

    def get_extra_state(self):
        return {"gamma": self.gamma, "accumulation": self.accumulation,
                "cancellation": self.cancellation}

    def set_extra_state(self, state):
        if not math.isfinite(state["gamma"]) or state["gamma"] <= 0:
            raise ValueError("Invalid checkpoint gamma")
        self.gamma = state["gamma"]
        self.accumulation, self.cancellation = state["accumulation"], state["cancellation"]

    def energy_dtype(self, device, input_dtype=torch.float32):
        if self.accumulation == "float64" or input_dtype == torch.float64:
            if torch.device(device).type == "mps":
                raise ValueError("MPS does not support FP64; use FP32 inputs/accumulation")
            return torch.float64
        if self.accumulation == "float32" or torch.device(device).type == "mps":
            return torch.float32
        return torch.float64

    def init_energy(self, batch_size, *, device=None, dtype=torch.float32, value=0.0):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Initial energy must be finite and nonnegative")
        return torch.full((batch_size, 1), value, device=device,
                          dtype=self.energy_dtype(device or "cpu", dtype))

    @staticmethod
    def _positive(x):
        # Zero derivative at exhaustion; identity on the positive half-line.
        return torch.where(x > 0, x, torch.zeros_like(x))

    @staticmethod
    def _project(v, q, budget, output_dtype):
        clipped = q > budget
        active = clipped & (budget > 0)
        # Both branches evaluated by autograd are finite, including at b=q=0.
        b_safe = torch.where(active, budget, torch.ones_like(budget))
        q_safe = torch.where(active, q, torch.ones_like(q))
        scale = b_safe.sqrt() / q_safe.sqrt()
        # Conservative rounding guard, only on clipped proposals. No added energy.
        scale = scale.clamp(max=1) * (1 - 4 * torch.finfo(output_dtype).eps)
        scale = torch.where(clipped, scale, torch.ones_like(scale))
        scale = torch.where(budget > 0, scale, torch.zeros_like(scale))
        return (v * scale).to(output_dtype), clipped

    def forward(self, u: Tensor, v: Tensor, energy: Optional[Tensor] = None, *,
                mask=None, mode="scan", diagnostics=False):
        valid = valid_mask(u, mask)
        if (v.ndim != 3 or v.shape[:2] != u.shape[:2] or v.shape[-1] <= 0
                or v.device != u.device or not v.is_floating_point()):
            raise ValueError("v must be floating (batch,time,output_features) on the input device")
        if mode not in ("scan", "reference"):
            raise ValueError("mode must be scan or reference")
        dtype = self.energy_dtype(u.device, torch.promote_types(u.dtype, v.dtype))
        output_dtype = torch.promote_types(v.dtype, torch.float32)
        # Mask BEFORE norms: padded NaNs must neither inject/spend energy nor spoil gradients.
        uu = torch.where(valid[..., None], u, 0).to(dtype)
        vv = torch.where(valid[..., None], v, 0).to(dtype)
        e = self.init_energy(u.shape[0], device=u.device, dtype=dtype) if energy is None else energy
        if e.shape != (u.shape[0], 1) or e.device != u.device or bool((e < 0).any()):
            raise ValueError("energy must be nonnegative (batch,1) on the input device")
        e = e.to(dtype)
        if self.check_nonfinite:
            check_finite((uu, vv, e))
        a = (uu * self.gamma).square().sum(-1, keepdim=True)
        q = vv.square().sum(-1, keepdim=True)
        if self.check_nonfinite:
            check_finite((a, q))
        if u.shape[1] == 0:
            info = self._diagnostics(q, q.bool(), valid, e[:, None], mode) if diagnostics else None
            return vv.to(output_dtype), e, info
        if mode == "reference":
            return self._reference(vv, a, q, e, valid, output_dtype, diagnostics)
        d = a - q
        sums = e[:, None] + d.cumsum(dim=1)
        # The initial zero is essential, particularly with E_init > 0.
        low = sums.cummin(dim=1).values.clamp(max=0)
        after = sums - low
        before = torch.cat((e[:, None], after[:, :-1]), dim=1)
        budget = before + a
        expected = self._positive(budget - q)
        # Audit local recurrence and an absolute roundoff envelope. A local-only
        # check could miss a long plateau of accumulated cancellation error.
        eps = torch.finfo(dtype).eps
        envelope = 8 * eps * (e[:, None] + (a + q).cumsum(1))
        tolerance = (1e-9 if dtype == torch.float64 else 2e-5) * (1 + budget)
        unreliable = ((after - expected).abs() > tolerance) | (envelope > tolerance)
        if self.check_nonfinite:
            check_finite((sums, after, budget))
        if bool(unreliable.any()):
            message = "Energy scan cancellation risk: using sequential reference; no energy refill"
            if self.cancellation == "raise":
                raise FloatingPointError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return self._reference(vv, a, q, e, valid, output_dtype, diagnostics,
                                   backend="reference_cancellation")
        y, clipped = self._project(vv, q, budget, output_dtype)
        info = self._diagnostics(q, clipped, valid, torch.cat((e[:, None], after), 1),
                                 "scan") if diagnostics else None
        return y, after[:, -1], info

    def _reference(self, v, a, q, e, valid, output_dtype, diagnostics,
                   backend="reference"):
        outputs, clipped_trace, energies = [], [], [e]
        for t in range(v.shape[1]):
            budget = e + a[:, t]
            y, clipped = self._project(v[:, t], q[:, t], budget, output_dtype)
            outputs.append(y)
            e = self._positive(budget - q[:, t])
            if diagnostics:
                clipped_trace.append(clipped)
                energies.append(e)
        y = torch.stack(outputs, 1)
        if self.check_nonfinite:
            check_finite((y, e))
        info = self._diagnostics(q, torch.stack(clipped_trace, 1), valid,
                                 torch.stack(energies, 1), backend) if diagnostics else None
        return y, e, info

    @staticmethod
    def _diagnostics(q, clipped, valid, energies, backend):
        count = valid.sum().clamp(min=1)
        norm = torch.where(q > 0, torch.where(q > 0, q, torch.ones_like(q)).sqrt(), 0)
        return {"clipping_fraction": (clipped.squeeze(-1) & valid).sum() / count,
                "energy": energies, "raw_output_norm": norm, "backend": backend}

    def step(self, energy: Tensor, u: Tensor, v: Tensor, *, mask=None, diagnostics=False):
        if u.ndim != 2 or v.ndim != 2:
            raise ValueError("step expects (batch,features)")
        y, energy, info = self(u[:, None], v[:, None], energy,
                               mask=None if mask is None else mask[:, None],
                               mode="reference", diagnostics=diagnostics)
        return y[:, 0], energy, info


class EnergyBoundedModel(nn.Module):
    """Place after the complete core readout/residual. Never feeds energy/output to core."""

    def __init__(self, core: nn.Module, gamma=1.0, *, check_core_state=False, **reservoir_options):
        super().__init__()
        self.core = core
        self.d_input, self.d_output = core.d_input, core.d_output
        self.reservoir = EnergyReservoir(gamma, **reservoir_options)
        self.check_core_state = check_core_state

    def init_state(self, batch_size, *, device=None, dtype=None, initial_energy=0.0):
        parameter = next(self.core.parameters(), None)
        device = device or (parameter.device if parameter is not None else "cpu")
        dtype = dtype or (parameter.dtype if parameter is not None else torch.get_default_dtype())
        return WrapperState(self.core.init_state(batch_size, device=device, dtype=dtype),
                            self.reservoir.init_energy(batch_size, device=device, dtype=dtype,
                                                       value=initial_energy))

    def forward(self, u, state=None, *, mask=None, mode="scan", diagnostics=False):
        valid = valid_mask(u, mask)
        state = self.init_state(u.shape[0], device=u.device, dtype=u.dtype) if state is None else state
        raw, core_state = self.core(u, state.core, mask=valid)
        if self.check_core_state:
            check_finite(core_state)
        y, energy, info = self.reservoir(u, raw, state.energy, mask=valid, mode=mode,
                                         diagnostics=diagnostics)
        result = (y, WrapperState(core_state, energy))
        return (*result, info) if diagnostics else result

    def step(self, state, u, *, mask=None, diagnostics=False):
        if u.ndim != 2:
            raise ValueError("step expects (batch,features)")
        raw, core_state = self.core.step(state.core, u, mask=mask)
        if self.check_core_state:
            check_finite(core_state)
        y, energy, info = self.reservoir.step(state.energy, u, raw, mask=mask,
                                              diagnostics=diagnostics)
        result = (y, WrapperState(core_state, energy))
        return (*result, info) if diagnostics else result
