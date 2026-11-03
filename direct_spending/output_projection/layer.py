"""Native linear readout and a stable reflected-energy scan.

The projection acts on r = NN(u, context) + D u. Context never supplies energy.
Only the sequential reference loops over time; scan uses the existing generic
associative scan with the max-plus pair composition.
"""
from __future__ import annotations

import warnings
import math

import torch
from torch import nn

from energy_bounded.state import check_finite, valid_mask
from src.neural_ssm.utils.scan import associative_scan
from ..storage import StorageLayer, safe_sqrt


class LinearProposal(nn.Module):
    def __init__(self, d_input, d_output, initialization="zero"):
        super().__init__()
        self.linear = nn.Linear(d_input, d_output, bias=False)
        if initialization == "zero":
            nn.init.zeros_(self.linear.weight)
        elif initialization == "identity":
            nn.init.eye_(self.linear.weight)
        else:
            raise ValueError("Linear path initialization must be zero or identity")

    def forward(self, u, neural_output, valid):
        return neural_output + self.linear(torch.where(valid[..., None], u, 0))


def _positive(value):
    # Zero subgradient at tank exhaustion, matching the online recurrence.
    return torch.where(value > 0, value, torch.zeros_like(value))


def _maximum(left, right):
    # Resolve equalities toward the more recent reset (right), rather than
    # distributing a gradient through an exhausted reservoir.
    return torch.where(left > right, left, right)


def _compose(first, second):
    a1, b1 = first
    a2, b2 = second
    return a1 + a2, _maximum(b1 + a2, b2)


class OutputProjection(StorageLayer):
    """Reuse storage precision/configuration and the native one-step adapter.

    E_next = max(E + gamma² ||u||² - ||r||², 0). For numerical robustness, cap
    proposal energy by the total incoming-plus-supplied energy at that prefix.
    That cap is an upper bound on every reachable available budget, so it does
    not change the real-arithmetic recurrence. Scaled norms avoid squaring a
    huge raw proposal. Max-plus scanning avoids cumsum/cummin subtraction.
    """

    @staticmethod
    def _geometry(proposal):
        scale = proposal.abs().amax(-1, keepdim=True).clamp(min=1)
        scaled = proposal / scale
        norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        direction = scaled / torch.where(norm > 0, norm, torch.ones_like(norm))
        limit = math.sqrt(torch.finfo(proposal.dtype).max / (4 * proposal.shape[-1]))
        ordinary = scale <= limit
        safe = torch.where(ordinary, proposal, 0)
        exact_energy = safe.square().sum(-1, keepdim=True)
        return scale, norm, direction, ordinary, exact_energy

    @staticmethod
    def _project(proposal, scale, norm, direction, ordinary, exact_energy, budget, output_dtype):
        radius = safe_sqrt(budget)
        clipped = torch.where(ordinary, exact_energy > budget, norm > radius / scale)
        # A conservative rounding guard only on clipped outputs. No budget
        # floor, refill, or modification of an under-budget proposal.
        guard = 1 - 4 * torch.finfo(output_dtype).eps
        projected = radius * direction * guard
        y = torch.where(clipped, projected, proposal)
        return torch.where(budget > 0, y, 0).to(output_dtype), clipped

    def forward(self, u, proposal, storage=None, *, mask=None, implementation=None,
                diagnostics=False, gamma=None):
        valid = valid_mask(u, mask)
        if (proposal.ndim != 3 or proposal.shape[:2] != u.shape[:2]
                or proposal.shape[-1] < 1 or proposal.device != u.device
                or not proposal.is_floating_point()):
            raise ValueError("Proposal must be floating (batch,time,output features) on the input device")
        dtype = self.energy_dtype(u.device, torch.promote_types(u.dtype, proposal.dtype))
        output_dtype = torch.promote_types(proposal.dtype, torch.float32)
        inputs = torch.where(valid[..., None], u, 0).to(dtype)
        raw = torch.where(valid[..., None], proposal, 0).to(dtype)
        initial = inputs.new_zeros(u.shape[0], 1) if storage is None else storage
        if initial.shape != (u.shape[0], 1) or initial.device != u.device or bool((initial < 0).any()):
            raise ValueError("Incoming energy must be nonnegative (batch,1) on the input device")
        initial = initial.to(dtype)
        gain = torch.as_tensor(self.gamma if gamma is None else gamma, device=u.device, dtype=dtype)
        if gain.ndim != 0 or not bool(torch.isfinite(gain)) or bool(gain < 0):
            raise ValueError("Gamma must be a finite nonnegative scalar")
        check_finite((inputs, raw, initial))
        deposit = (gain * inputs).square().sum(-1, keepdim=True)
        supplied = initial[:, None] + deposit.cumsum(1)
        check_finite((deposit, supplied))
        method = implementation or self.implementation
        if method not in ("scan", "reference"):
            raise ValueError("Unknown projection implementation")
        if not u.shape[1]:
            info = self._info(raw, valid, raw[..., :1], raw[..., :1].bool(), initial[:, None],
                              supplied, raw[..., :1], "scan_max_plus" if method == "scan" else method) if diagnostics else None
            return raw.to(output_dtype), initial, info
        scale, norm, direction, ordinary, exact_energy = self._geometry(raw)
        cap = safe_sqrt(supplied)
        capped_norm = scale * torch.minimum(norm, cap / scale)
        proposal_energy = torch.where(ordinary, torch.minimum(exact_energy, supplied),
                                       torch.minimum(capped_norm.square(), supplied))
        proposal_energy = torch.where(valid[..., None], proposal_energy, 0)
        check_finite(proposal_energy)
        if method == "reference":
            return self._reference_projection(raw, deposit, proposal_energy, initial, valid,
                                              supplied, scale, norm, direction, ordinary, exact_energy, output_dtype, diagnostics)
        delta = deposit - proposal_energy
        resets = torch.where(valid[..., None], torch.zeros_like(delta), torch.full_like(delta, -float("inf")))
        prefix_a, prefix_b = associative_scan(_compose, (delta, resets), axis=1)
        after = _maximum(initial[:, None] + prefix_a, prefix_b)
        before = torch.cat((initial[:, None], after[:, :-1]), 1)
        budget = before + deposit
        y, clipped = self._project(raw, scale, norm, direction, ordinary, exact_energy, budget, output_dtype)
        spent = y.to(dtype).square().sum(-1, keepdim=True)
        residual = after + spent.cumsum(1) - supplied
        tolerance = 1e-9 if dtype == torch.float64 else 2e-5
        output_tolerance = tolerance + 16 * torch.finfo(output_dtype).eps
        expected = torch.where(valid[..., None], _positive(budget - proposal_energy), before)
        unreliable = ((after - expected).abs() > tolerance * (1 + budget)).any()
        unreliable = unreliable | (residual.abs() > output_tolerance * (1 + supplied)).any()
        check_finite((after, y, spent))
        if bool(unreliable):
            warnings.warn("Projection scan roundoff audit failed; using sequential reference without refill",
                          RuntimeWarning, stacklevel=2)
            return self._reference_projection(raw, deposit, proposal_energy, initial, valid,
                supplied, scale, norm, direction, ordinary, exact_energy, output_dtype, diagnostics, "reference_roundoff")
        info = self._info(raw, valid, budget, clipped, torch.cat((initial[:, None], after), 1),
                          supplied, spent, "scan_max_plus") if diagnostics else None
        return y, after[:, -1], info

    def _reference_projection(self, raw, deposit, proposal_energy, initial, valid,
                              supplied, scale, norm, direction, ordinary, exact_energy, output_dtype, diagnostics,
                              backend="reference"):
        current, outputs, energies, clips, budgets = initial, [], [initial], [], []
        for t in range(raw.shape[1]):
            budget = current + deposit[:, t]
            y, clipped = self._project(raw[:, t], scale[:, t], norm[:, t], direction[:, t],
                                       ordinary[:, t], exact_energy[:, t], budget, output_dtype)
            current = torch.where(valid[:, t, None], _positive(budget - proposal_energy[:, t]), current)
            outputs.append(y)
            if diagnostics:
                energies.append(current); clips.append(clipped); budgets.append(budget)
        y = torch.stack(outputs, 1)
        check_finite((y, current))
        info = None
        if diagnostics:
            info = self._info(raw, valid, torch.stack(budgets, 1), torch.stack(clips, 1),
                torch.stack(energies, 1), supplied, y.to(initial.dtype).square().sum(-1, keepdim=True), backend)
        return y, current, info

    @staticmethod
    def _info(raw, valid, budget, clipped, energies, supplied, spent, backend):
        total = spent.cumsum(1)
        count = valid.sum().clamp(min=1)
        residual = (total - supplied) / (1 + supplied)
        balance = (energies[:, 1:] + total - supplied) / (1 + supplied)
        fraction = spent / torch.where(budget > 0, budget, torch.ones_like(budget))
        scale, norm, *_ = OutputProjection._geometry(raw) if raw.shape[1] else (None, None)
        return {"storage": energies, "clipping_fraction": (clipped[..., 0] & valid).sum() / count,
                "mean_spending_fraction": (fraction[..., 0] * valid).sum() / count,
                "raw_proposal_norm": scale * norm if scale is not None else raw[..., :1],
                "max_normalized_energy_violation": residual.clamp(min=0).amax() if residual.numel() else energies.sum() * 0,
                "max_normalized_identity_error": balance.abs().amax() if balance.numel() else energies.sum() * 0,
                "implementation": backend}
