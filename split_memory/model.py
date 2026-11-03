"""One split-memory block; all neural gradients use ordinary explicit autograd."""
import contextlib
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import SplitMemoryConfig, sufficient_sweeps
from .memory import BoundedMemory, memory_reference, memory_scan
from .solvers import (ConvergenceError, adaptive_jacobi, fixed_jacobi, residual,
                      sequential_reader)
from .state import SplitMemoryState, accumulation_dtype


class ContractiveWeight(nn.Module):
    def __init__(self, width, rho, delta, mixing_scale):
        super().__init__()
        self.rho, self.delta = rho, delta
        self.raw = nn.Parameter(torch.eye(width) + torch.randn(width, width) * mixing_scale / width)

    def forward(self, dtype=None):
        dtype = accumulation_dtype(dtype or self.raw.dtype)
        raw = self.raw.to(dtype)
        # Rescaling avoids overflow in row sums without changing the parametrization.
        scale = raw.abs().amax(-1, keepdim=True).clamp_min(1)
        scaled = raw / scale
        denominator = self.delta / scale + scaled.abs().sum(-1, keepdim=True)
        effective_rho = self.rho * (1 - 8 * torch.finfo(dtype).eps)
        return effective_rho * scaled / denominator


def _linear(layer, x):
    return F.linear(x, layer.weight.to(x.dtype), None if layer.bias is None else layer.bias.to(x.dtype))


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {name}; output bounds do not protect internal arithmetic")


def _last(values, valid, incoming):
    if values.shape[1] == 0:
        return incoming
    time = torch.arange(values.shape[1], device=values.device)[None]
    index = torch.where(valid, time, -1).amax(1)
    gathered = values[torch.arange(values.shape[0], device=values.device), index.clamp_min(0)]
    return torch.where((index >= 0).reshape(-1, *([1] * (incoming.ndim - 1))), gathered, incoming)


class SplitMemoryContractiveRNN(nn.Module):
    """Causal (B,T,D) core with explicit m/c state and an independent memory scan.

    ``forward`` returns (outputs, state), or adds a diagnostics dictionary when
    requested. ``step(state, x)`` evaluates the exact recurrent update once.
    ``solver``/``memory_backend`` overrides change evaluation, never model weights.
    """

    def __init__(self, d_input, d_output, config=None):
        super().__init__()
        if d_input < 1 or d_output < 1:
            raise ValueError("Positive input/output dimensions required")
        self.d_input, self.d_output = d_input, d_output
        self.config = config or SplitMemoryConfig()
        cfg = self.config
        self.memory = BoundedMemory(d_input, cfg)
        if cfg.reader != "linear":
            self.memory_projection = nn.Linear(2 * cfg.heads, cfg.computation_width, bias=False)
            self.input_projection = nn.Linear(d_input, cfg.computation_width, bias=False)
            self.bias = nn.Parameter(torch.zeros(cfg.computation_width))
            self.out_proj = nn.Linear(cfg.computation_width, d_output)
            for layer in (self.memory_projection, self.input_projection):
                nn.init.normal_(layer.weight, std=cfg.projection_scale / math.sqrt(layer.in_features))
        else:
            self.out_proj = nn.Linear(2 * cfg.heads, d_output)
        self.computation = (ContractiveWeight(cfg.computation_width, cfg.rho, cfg.row_delta, cfg.mixing_scale)
                            if cfg.reader == "recurrent" else None)
        nn.init.normal_(self.out_proj.weight, std=.5 / math.sqrt(self.out_proj.in_features))
        nn.init.zeros_(self.out_proj.bias)
        self.last_solver_info = None

    def init_state(self, batch_size, *, device=None, dtype=None):
        if batch_size < 1:
            raise ValueError("Positive batch size required")
        parameter = next(self.parameters())
        dtype = accumulation_dtype(dtype or parameter.dtype)
        device = device or parameter.device
        return SplitMemoryState(torch.zeros(batch_size, self.config.heads, 2, device=device, dtype=dtype),
                                torch.zeros(batch_size, self.config.computation_width, device=device, dtype=dtype))

    reset = init_state

    def _prepare(self, x, state, mask):
        if x.ndim != 3 or x.shape[-1] != self.d_input or x.shape[0] < 1 or not x.is_floating_point():
            raise ValueError("Expected floating inputs (positive batch,time,input_features)")
        parameter = next(self.parameters())
        if x.device != parameter.device:
            raise ValueError("Move model and inputs to the same device explicitly")
        dtype = torch.float64 if x.dtype == torch.float64 or parameter.dtype == torch.float64 else torch.float32
        if x.device.type == "mps" and dtype == torch.float64:
            raise ValueError("MPS does not support FP64; use FP32 or CPU explicitly")
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
        if valid.shape != x.shape[:2] or valid.dtype != torch.bool or valid.device != x.device:
            raise ValueError("mask must be boolean (batch,time) on the input device")
        x = torch.where(valid[..., None], x.to(dtype), 0)
        _finite(x, "valid input")
        state = self.init_state(x.shape[0], device=x.device, dtype=dtype) if state is None else state
        if (not isinstance(state, SplitMemoryState)
                or state.m.shape != (x.shape[0], self.config.heads, 2)
                or state.c.shape != (x.shape[0], self.config.computation_width)
                or state.m.device != x.device or state.c.device != x.device
                or not state.m.is_floating_point() or not state.c.is_floating_point()):
            raise ValueError("Incompatible incoming split-memory state")
        _finite(state.m, "incoming memory")
        _finite(state.c, "incoming computation state")
        state = SplitMemoryState(state.m.to(dtype), state.c.to(dtype))
        suffix = not bool(((~valid[:, :-1]) & valid[:, 1:]).any())
        return x, state, valid, suffix

    def reference(self, x, state=None, *, mask=None, **kwargs):
        return self(x, state, mask=mask, solver="sequential", memory_backend="sequential", **kwargs)

    def step(self, state, x, *, mask=None):
        if x.ndim != 2:
            raise ValueError("step expects (batch,features) and an explicit state first")
        # T=1 reference executes one memory update and one reader evaluation.
        output, state = self.reference(x[:, None], state, mask=None if mask is None else mask[:, None])
        return output[:, 0], state

    def forward(self, x, state=None, *, mask=None, solver=None, memory_backend=None,
                diagnostics=False, return_trajectory=False):
        cfg = replace(self.config, solver=solver or self.config.solver,
                      memory_backend=memory_backend or self.config.memory_backend)
        x, incoming, valid, suffix = self._prepare(x, state, mask)
        ctx = (torch.autocast(device_type=x.device.type, enabled=False)
               if x.device.type in ("cpu", "cuda", "mps") else contextlib.nullcontext())
        with ctx:
            return self._evaluate(x, incoming, valid, suffix, cfg, diagnostics, return_trajectory)

    def _evaluate(self, x, incoming, valid, suffix, cfg, diagnostics, return_trajectory):
        batch, length = x.shape[:2]
        if length == 0:
            info = dict(solver=cfg.solver, memory_backend=cfg.memory_backend, sweeps=0, cell_passes=0,
                        cell_evaluations=0, valid_cell_evaluations=0, residual=0., state_error_bound=0., converged=True,
                        tolerance=cfg.tolerance, mask_fallback=False, dtype=str(x.dtype))
            self.last_solver_info = info
            output = x.new_empty(batch, 0, self.d_output)
            if return_trajectory:
                info.update(memory=x.new_empty(batch, 0, cfg.heads, 2), computation=x.new_empty(batch, 0, cfg.computation_width))
            return (output, incoming, info) if diagnostics or return_trajectory else (output, incoming)

        # These coefficients and all input/memory projections are outside the sweep loop.
        a, b = self.memory.coefficients(x, valid)
        _finite(a, "memory transitions")
        _finite(b, "memory writes")
        backend = cfg.memory_backend if suffix else "sequential"
        memory = (memory_scan if backend == "scan" else memory_reference)(a, b, incoming.m)
        _finite(memory, "memory states")
        features = memory.flatten(-2)
        count = int(valid.sum())
        info = dict(solver=cfg.solver, memory_backend=backend, mask_fallback=not suffix,
                    tolerance=cfg.tolerance, dtype=str(x.dtype), gradient="ordinary explicit autograd")
        if cfg.reader == "linear":
            computation = incoming.c[:, None].expand(-1, length, -1)
            output = _linear(self.out_proj, features)
            info.update(solver="linear_readout", sweeps=0, cell_passes=0, cell_evaluations=0,
                        valid_cell_evaluations=0, residual=0., state_error_bound=0., converged=True, measured_row_sum=0.)
        else:
            u = _linear(self.memory_projection, features) + _linear(self.input_projection, x) + self.bias.to(x.dtype)
            _finite(u, "reader forcing")
            if cfg.reader == "pointwise":
                computation = torch.tanh(u)
                info.update(solver="pointwise", sweeps=0, cell_passes=1, cell_evaluations=batch * length, valid_cell_evaluations=count,
                            residual=0., state_error_bound=0., converged=True, measured_row_sum=0.)
            else:
                weight = self.computation(x.dtype)
                _finite(weight, "normalized computation weights")
                actual_solver = cfg.solver if suffix else "sequential_mask_fallback"
                verify = diagnostics or return_trajectory or cfg.require_convergence
                if actual_solver.startswith("sequential"):
                    computation = sequential_reader(u, weight, incoming.c, valid)
                    r = residual(computation, u, weight, incoming.c, valid) if diagnostics or return_trajectory else x.new_zeros(())
                    bound, converged = r / (1 - cfg.rho), True
                    sweeps, passes = 0, 1 + int(diagnostics or return_trajectory)
                elif actual_solver == "fixed":
                    computation = fixed_jacobi(u, weight, incoming.c, valid, cfg.fixed_sweeps)
                    r = residual(computation, u, weight, incoming.c, valid) if verify else None
                    bound = r / (1 - cfg.rho) if r is not None else None
                    converged = bool(bound <= cfg.tolerance) if bound is not None else None
                    sweeps, passes = cfg.fixed_sweeps, cfg.fixed_sweeps + int(verify)
                else:
                    solved = adaptive_jacobi(u, weight, incoming.c, valid, rho=cfg.rho,
                        tolerance=cfg.tolerance, max_sweeps=cfg.max_sweeps)
                    computation, r, bound, converged = solved.states, solved.measured_residual, solved.state_error_bound, solved.converged
                    sweeps, passes = solved.sweeps, solved.sweeps + solved.verification_passes
                info.update(solver=actual_solver, sweeps=sweeps, cell_passes=passes,
                            cell_evaluations=batch * length * passes, valid_cell_evaluations=count * passes,
                            residual=None if r is None else float(r.detach()),
                            state_error_bound=None if bound is None else float(bound.detach()),
                            converged=converged, measured_row_sum=float(weight.abs().sum(-1).amax().detach()),
                            sufficient_uniform_sweeps=sufficient_sweeps(cfg.rho, cfg.tolerance),
                            bound_scope="exact-arithmetic residual bound for the computed memory; not an interval proof")
                if cfg.require_convergence and converged is False:
                    self.last_solver_info = info
                    raise ConvergenceError(info)
            output = _linear(self.out_proj, computation)
        output = torch.where(valid[..., None], output, 0)
        _finite(output, "output")
        outgoing = SplitMemoryState(_last(memory, valid, incoming.m), _last(computation, valid, incoming.c))
        if diagnostics or return_trajectory:
            head_norms = torch.linalg.vector_norm(memory, dim=-1)
            bounds = torch.linalg.vector_norm(incoming.m, dim=-1).clamp_min(cfg.memory_radius)
            info.update(max_memory_head_norm=float(head_norms.amax().detach()),
                        max_memory_bound_deviation=float((head_norms - bounds[:, None]).clamp_min(0).amax().detach()),
                        memory_to_computation_lipschitz_inf=(0. if cfg.reader == "linear" else
                            float(self.memory_projection.weight.detach().abs().sum(-1).amax())),
                        rho=cfg.rho if cfg.reader == "recurrent" else 0.)
        self.last_solver_info = dict(info)
        if return_trajectory:
            info.update(memory=memory, computation=computation)
        return (output, outgoing, info) if diagnostics or return_trajectory else (output, outgoing)
