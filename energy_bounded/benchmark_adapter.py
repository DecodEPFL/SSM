"""Contained bridge to Test_files.Benchmark.SystemIDTrainer's existing call/layout API."""
from __future__ import annotations

from torch import nn

from .reservoir import EnergyBoundedModel


class BenchmarkAdapter(nn.Module):
    """Adapt a raw or wrapped core to the legacy trainer's optional unbatched input.

    Use normalizer=None. Centering inputs or adding an output mean changes the
    zero-state gain statement in physical units. This adapter performs no such
    transforms. Each call starts a trajectory unless explicit state is supplied.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, u, state=None, *, mode="scan", mask=None):
        if mode not in ("scan", "reference"):
            raise ValueError("Use scan or reference for energy-bounded models")
        unbatched = u.ndim == 2
        if unbatched:
            u = u[None]
            mask = None if mask is None else mask[None]
        if isinstance(self.model, EnergyBoundedModel):
            y, state = self.model(u, state, mask=mask, mode=mode)
        else:
            y, state = self.model(u, state, mask=mask)
        return (y[0] if unbatched else y), state
