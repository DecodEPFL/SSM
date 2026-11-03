"""Adapters for the existing nonlinear suite, with tagged reconstruction."""
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from .config import SplitMemoryConfig
from .model import SplitMemoryContractiveRNN
from .solvers import ConvergenceError


class SplitMemorySim(nn.Module):
    def __init__(self, n_u, n_y, config):
        super().__init__()
        self.split_config = config
        self.core = SplitMemoryContractiveRNN(n_u, n_y, config)

    def forward(self, u, *, mask=None):
        # An explicitly approximate training run is always audited at evaluation.
        diagnose = self.split_config.diagnostics or (not self.training and not self.split_config.require_convergence)
        result = self.core(u, mask=mask, diagnostics=diagnose)
        return result[0]

    def solver_diagnostics(self):
        return dict(self.core.last_solver_info or {})

    @torch.no_grad()
    def diagnostics(self):
        weight = self.core.computation
        return dict(certified_gain_bound=None, global_gamma=None,
                    computation_rho=self.split_config.rho if weight is not None else 0.,
                    measured_row_sum=0. if weight is None else float(weight().abs().sum(-1).amax()),
                    claim="bounded gated memory and contractive computation; no prescribed l2-gain certificate",
                    solver=self.solver_diagnostics())


def config_for_model(global_config):
    return SplitMemoryConfig.from_dict(global_config.split_memory or {})


def build_suite_model(n_u, n_y, global_config, param_budget=0, *, closest_width=None):
    config = config_for_model(global_config)
    if param_budget:
        key = "heads" if config.reader == "linear" else "computation_width"
        with torch.random.fork_rng(devices=[]):
            width = closest_width(lambda value: sum(p.numel() for p in SplitMemorySim(
                n_u, n_y, replace(config, **{key: value})).parameters()), param_budget, lo=1)
        config = replace(config, **{key: width})
    return SplitMemorySim(n_u, n_y, config)


def add_arguments(parser):
    group = parser.add_argument_group("Scanned memory with a contractive reader")
    group.add_argument("--split-reader", choices=("recurrent", "pointwise", "linear"), default="recurrent")
    group.add_argument("--split-memory-heads", type=int, default=8)
    group.add_argument("--split-computation-width", type=int, default=16)
    group.add_argument("--split-rho", type=float, default=.7)
    group.add_argument("--split-memory-radius", type=float, default=1.)
    group.add_argument("--split-initial-retention", type=float, default=.95)
    group.add_argument("--split-row-delta", type=float, default=1e-4)
    group.add_argument("--split-solver", choices=("fixed", "adaptive", "sequential"), default="fixed")
    group.add_argument("--split-memory-backend", choices=("scan", "sequential"), default="scan")
    group.add_argument("--split-tolerance", type=float, default=1e-4)
    group.add_argument("--split-fixed-sweeps", type=int, default=32)
    group.add_argument("--split-max-sweeps", type=int, default=64)
    import argparse
    group.add_argument("--split-require-convergence", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--split-diagnostics", action="store_true")


def settings_from_args(args):
    return SplitMemoryConfig(reader=args.split_reader, heads=args.split_memory_heads,
        computation_width=args.split_computation_width, rho=args.split_rho,
        memory_radius=args.split_memory_radius, initial_retention=args.split_initial_retention,
        row_delta=args.split_row_delta, solver=args.split_solver, memory_backend=args.split_memory_backend,
        tolerance=args.split_tolerance, fixed_sweeps=args.split_fixed_sweeps,
        max_sweeps=args.split_max_sweeps, require_convergence=args.split_require_convergence,
        diagnostics=args.split_diagnostics).to_dict()


def save_checkpoint(path, model, norm, history, metadata):
    torch.save(dict(format="split_memory_v1", family="split_memory", n_u=model.core.d_input,
        n_y=model.core.d_output, config=model.split_config.to_dict(),
        parameter_dtype=str(next(model.parameters()).dtype), model_state_dict=model.state_dict(),
        normalizer={key: getattr(norm, key).detach().cpu() for key in ("u_mean", "u_std", "y_mean", "y_std")},
        history=history, metadata=metadata, gradient="ordinary autograd through the implemented finite-sweep graph"), Path(path))


def load_checkpoint(path, *, device="cpu", solver=None, memory_backend=None):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "split_memory_v1" or payload.get("family") != "split_memory":
        raise ValueError("Not a split_memory_v1 checkpoint; other model families are distinct")
    config = SplitMemoryConfig.from_dict(payload["config"])
    config = replace(config, solver=solver or config.solver, memory_backend=memory_backend or config.memory_backend)
    dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64,
             "torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16}[payload["parameter_dtype"]]
    model = SplitMemorySim(payload["n_u"], payload["n_y"], config).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval(), payload
