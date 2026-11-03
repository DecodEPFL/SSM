"""Independent spending core plus a scalar storage, with explicit initialization."""
from __future__ import annotations

import math
import contextlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from energy_bounded.state import check_finite, valid_mask

from .config import SpendingConfig
from .cores import build_core
from .storage import StorageLayer
from .gain import GainBound
from .output_projection import LinearProposal, OutputProjection


@dataclass
class SpendingState:
    core: Any
    storage: torch.Tensor
    initial_beta: torch.Tensor


class InitialStorage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.beta_mode == "learned":
            # inverse softplus(.1) by default. No cap or hidden regularization.
            logit = config.beta_initial + math.log(-math.expm1(-config.beta_initial))
            if config.initial_features:
                self.net = nn.Sequential(nn.Linear(len(config.initial_features), config.beta_width),
                                         nn.Tanh(), nn.Linear(config.beta_width, 1))
                nn.init.zeros_(self.net[-1].weight)
                nn.init.constant_(self.net[-1].bias, logit)
            else:
                self.logit = nn.Parameter(torch.tensor(logit))

    def forward(self, batch_size, *, device, dtype, features=None):
        cfg = self.config
        if cfg.beta_mode != "learned":
            if features is not None:
                raise ValueError("Initial features are used only by learned beta")
            value = 0. if cfg.beta_mode == "zero" else cfg.fixed_beta
            return torch.full((batch_size, 1), value, device=device, dtype=dtype)
        if not cfg.initial_features:
            if features is not None:
                raise ValueError("No context features were declared for the learned scalar beta")
            return F.softplus(self.logit.to(device=device, dtype=dtype)).expand(batch_size, 1)
        parameter = next(self.net.parameters())
        if features is None:
            features = torch.tensor(cfg.initial_features, device=device,
                                    dtype=parameter.dtype).expand(batch_size, -1)
        if features.shape != (batch_size, len(cfg.initial_features)) or features.device != torch.device(device):
            raise ValueError("Declared initial features must be (batch,context features) on the model device")
        check_finite(features)
        # Initial storage must not depend on whether reset happens inside an AMP
        # forward or explicitly before repeated steps.
        no_amp = (torch.autocast(device_type=parameter.device.type, enabled=False)
                  if parameter.device.type in ("cpu", "cuda") else contextlib.nullcontext())
        with no_amp:
            logits = self.net(features.to(parameter.dtype))
        return F.softplus(logits.to(dtype))


class DirectSpendingModel(nn.Module):
    """Selectable certified readout; no energy/output feedback into the core."""

    def __init__(self, d_input, d_output, config=None, *, core=None, check_core_state=False):
        super().__init__()
        self.config = config or SpendingConfig()
        self.d_input, self.d_output = d_input, d_output
        self.core = build_core(d_input + self.config.context_features, d_output, self.config) if core is None else core
        self.initializer = InitialStorage(self.config)
        self.gain = GainBound(self.config)
        layer_type = OutputProjection if self.config.readout == "projected" else StorageLayer
        self.storage_layer = layer_type(self.config.gamma, implementation=self.config.storage_impl,
                                          precision=self.config.energy_precision)
        self.proposal_readout = (LinearProposal(d_input, d_output, self.config.linear_init)
                                 if self.config.readout == "projected" else None)
        self.check_core_state = check_core_state

    @property
    def gamma(self):
        """Current differentiable global gain; hold weights fixed across chunks."""
        return self.gain()

    def _core_input(self, v, context):
        if v.shape[-1] != self.d_input:
            raise ValueError("Wrong external energy-input feature dimension")
        if not self.config.context_features:
            if context is not None:
                raise ValueError("Declare context_features to supply core-only context")
            return v
        expected = (*v.shape[:-1], self.config.context_features)
        if (not isinstance(context, torch.Tensor) or context.shape != expected
                or context.device != v.device or context.dtype != v.dtype):
            raise ValueError("Context must match input batch/time, declared features, device and dtype")
        return torch.cat((v, context), dim=-1)

    def init_state(self, batch_size, *, device=None, dtype=None, initial_features=None):
        parameter = next(self.core.parameters(), None)
        device = torch.device(device or (parameter.device if parameter is not None else "cpu"))
        dtype = dtype or (parameter.dtype if parameter is not None else torch.get_default_dtype())
        energy_dtype = self.storage_layer.energy_dtype(device, dtype)
        beta = self.initializer(batch_size, device=device, dtype=energy_dtype, features=initial_features)
        check_finite(beta)
        if self.config.beta_mode == "learned" and bool((beta <= 0).any()):
            raise FloatingPointError("Learned softplus beta underflowed to zero; no energy floor or refill was added")
        return SpendingState(self.core.init_state(batch_size, device=device, dtype=dtype), beta, beta)

    def forward(self, v, state=None, *, mask=None, initial_features=None,
                implementation=None, diagnostics=False, context=None):
        valid = valid_mask(v, mask)
        if state is None:
            state = self.init_state(v.shape[0], device=v.device, dtype=v.dtype,
                                    initial_features=initial_features)
        elif initial_features is not None:
            raise ValueError("Initial features/beta are computed once, never again on chunk continuation")
        z, core_state = self.core(self._core_input(v, context), state.core, mask=valid)
        if self.check_core_state:
            check_finite(core_state)
        if self.proposal_readout is not None:
            z = self.proposal_readout(v, z, valid)
        y, storage, info = self.storage_layer(v, z, state.storage, mask=valid,
                                               implementation=implementation, diagnostics=diagnostics,
                                               gamma=self.gamma)
        result = (y, SpendingState(core_state, storage, state.initial_beta))
        if diagnostics:
            info["initial_beta"] = state.initial_beta
            return *result, info
        return result

    def step(self, state, v, *, mask=None, diagnostics=False, context=None):
        if v.ndim != 2:
            raise ValueError("step expects explicit state first, then (batch,features)")
        z, core_state = self.core.step(state.core, self._core_input(v, context), mask=mask)
        if self.check_core_state:
            check_finite(core_state)
        if self.proposal_readout is not None:
            valid = valid_mask(v[:, None], None if mask is None else mask[:, None])[:, 0]
            z = self.proposal_readout(v, z, valid)
        y, storage, info = self.storage_layer.step(state.storage, v, z, mask=mask,
                                                   diagnostics=diagnostics, gamma=self.gamma)
        result = (y, SpendingState(core_state, storage, state.initial_beta))
        if diagnostics:
            info["initial_beta"] = state.initial_beta
            return *result, info
        return result
