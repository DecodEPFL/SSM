"""Adapters and persisted certificates for the existing nonlinear benchmark runner."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import CORES, MODEL_SPECS, SpendingConfig
from .model import DirectSpendingModel


class SpendingSim(nn.Module):
    """Output-only suite adapter: returned predictions are always the storage output y."""

    def __init__(self, n_u, n_y, config, *, coordinates="standardized"):
        super().__init__()
        if coordinates not in ("standardized", "physical"):
            raise ValueError("Spending coordinates must be standardized or physical")
        self.spending_config = config
        self.coordinates = coordinates
        self.model = DirectSpendingModel(n_u, n_y, config)

    def forward(self, u, *, mask=None, context=None):
        return self.model(u, mask=mask, context=context)[0]

    @torch.no_grad()
    def certificate(self):
        parameter = next(self.parameters())
        beta = self.model.init_state(1, device=parameter.device, dtype=parameter.dtype).initial_beta.item()
        gamma = self.model.gamma
        gamma = gamma.detach().item() if torch.is_tensor(gamma) else gamma
        cfg = self.spending_config
        return {"family": "energy_projection" if cfg.readout == "projected" else "direct_spending", "version": 1,
                "readout": cfg.readout,
                "linear_path": cfg.readout == "projected",
                "linear_initialization": cfg.linear_init if cfg.readout == "projected" else None,
                "numerical_semantics": "real-arithmetic energy identity; conservative clipped-output rounding may dissipate energy" if cfg.readout == "projected" else "real-arithmetic energy identity, audited against returned outputs",
                "spending_core": cfg.core, "gamma": gamma, "gamma_mode": cfg.gamma_mode,
                "initial_gamma": cfg.gamma,
                "learned_gamma_bounds": [cfg.gamma_min, cfg.gamma_max] if cfg.gamma_mode == "learned" else None,
                "uniform_gamma_bound": cfg.gamma_max if cfg.gamma_mode == "learned" else gamma,
                "gamma_semantics": "global scalar; fixed weights/gamma throughout each trajectory; reset states after optimizer updates",
                "beta_mode": self.spending_config.beta_mode, "initial_beta": beta,
                "coordinates": self.coordinates,
                "label": self.spending_config.label + f" ({self.coordinates} coordinates)",
                "statement": "sum ||y||^2 + S_final = initial_beta + gamma^2 sum ||input||^2",
                "certified_interval": "complete trajectory from sample 0, including metric warm-up",
                "metric_suffix": "skipping initialization samples does not reset storage; a suffix has its carried-storage allowance",
                "context_features": cfg.context_features,
                "context_semantics": "optional sequence context conditions the causal core only, never supplies storage energy; beta context is separately declared constant initialization",
                "physical_zero_state_bound": self.coordinates == "physical" and self.spending_config.beta_mode == "zero"}

    @torch.no_grad()
    def diagnostics(self):
        certificate = self.certificate()
        return {"certified_gain_bound": certificate["gamma"] if certificate["physical_zero_state_bound"] else None,
                "global_gamma": certificate["gamma"], "certificate": certificate}

    @torch.no_grad()
    def storage_diagnostics(self, u, *, initialization_window=0, context=None):
        _, _, info = self.model(u, diagnostics=True, context=context)
        trace = info["storage"].detach().cpu()
        start = min(max(0, initialization_window), u.shape[1])
        result = {"gamma": self.certificate()["gamma"],
                "initial_beta": info["initial_beta"].detach().cpu().flatten().tolist(),
                "storage_at_metric_start": trace[:, start, 0].tolist(),
                "storage_trace": trace[..., 0].tolist(),
                "mean_spending_fraction": info["mean_spending_fraction"].item(),
                "max_normalized_energy_violation": info["max_normalized_energy_violation"].item(),
                "max_normalized_identity_error": info["max_normalized_identity_error"].item(),
                "implementation": info["implementation"]}
        for key in ("head_saturation_rate", "clipping_fraction"):
            if key in info:
                result[key] = info[key].item()
        if "raw_proposal_norm" in info:
            result["raw_proposal_norm"] = info["raw_proposal_norm"].detach().cpu()[..., 0].tolist()
        return result


def config_for_model(name, global_config):
    values = dict(global_config.spending or {})
    values.setdefault("width", global_config.d_model)
    values.setdefault("layers", global_config.n_layers)
    if "spending_core" in MODEL_SPECS[name]:
        values["core"] = MODEL_SPECS[name]["spending_core"]
    # Coordinate selection belongs to the suite adapter, not to the mathematical model.
    values.pop("coordinates", None)
    return SpendingConfig.from_dict(values)


def build_suite_model(name, n_u, n_y, global_config, param_budget=0, *, closest_width=None):
    config = config_for_model(name, global_config)
    coordinates = (global_config.spending or {}).get("coordinates", "standardized")
    if param_budget:
        def valid_width(width):
            stride = math.lcm(2, config.heads) if config.core == "transformer" else 1
            return max(stride, math.ceil(width / stride) * stride)
        with torch.random.fork_rng(devices=[]):
            width = closest_width(lambda w: sum(p.numel() for p in SpendingSim(
                n_u, n_y, replace(config, width=valid_width(w)), coordinates=coordinates).parameters()),
                param_budget, lo=2)
        config = replace(config, width=valid_width(width))
    return SpendingSim(n_u, n_y, config, coordinates=coordinates)


def add_arguments(parser):
    group = parser.add_argument_group("Energy-bounded sequence readouts")
    group.add_argument("--spending-readout", choices=("spending", "projected"), default="spending")
    group.add_argument("--spending-linear-init", choices=("zero", "identity"), default="zero")
    group.add_argument("--spending-core", choices=CORES, default="gru")
    group.add_argument("--spending-gamma", type=float, default=1., help="fixed gamma, or initial gamma in learned mode")
    group.add_argument("--spending-gamma-mode", choices=("fixed", "learned"), default="fixed")
    group.add_argument("--spending-gamma-min", type=float, default=1e-4)
    group.add_argument("--spending-gamma-max", type=float, default=100.)
    group.add_argument("--spending-beta-mode", choices=("zero", "fixed", "learned"), default="zero")
    group.add_argument("--spending-fixed-beta", type=float, default=0.)
    group.add_argument("--spending-beta-initial", type=float, default=.1)
    group.add_argument("--spending-beta-width", type=int, default=8)
    group.add_argument("--spending-initial-features", default="", help="comma-separated declared CONSTANT initial context; never extracted from inputs/targets")
    group.add_argument("--spending-tcn-kernel", type=int, default=3)
    group.add_argument("--spending-tcn-dilations", default="1,2,4")
    group.add_argument("--spending-heads", type=int, default=2)
    group.add_argument("--spending-dropout", type=float, default=0.)
    group.add_argument("--spending-storage", choices=("scan", "reference"), default="scan")
    group.add_argument("--spending-precision", choices=("auto", "float32", "float64"), default="auto")
    group.add_argument("--spending-coordinates", choices=("standardized", "physical"), default="standardized")
    group.add_argument("--spending-diagnostics", action="store_true")


def settings_from_args(args):
    config = SpendingConfig(core=args.spending_core, width=args.d_model, layers=args.n_layers,
        readout=args.spending_readout, linear_init=args.spending_linear_init,
        gamma=args.spending_gamma, gamma_mode=args.spending_gamma_mode,
        gamma_min=args.spending_gamma_min, gamma_max=args.spending_gamma_max,
        beta_mode=args.spending_beta_mode,
        fixed_beta=args.spending_fixed_beta, beta_initial=args.spending_beta_initial,
        beta_width=args.spending_beta_width,
        initial_features=tuple(float(x.strip()) for x in args.spending_initial_features.split(",") if x.strip()),
        tcn_kernel=args.spending_tcn_kernel,
        tcn_dilations=tuple(int(x.strip()) for x in args.spending_tcn_dilations.split(",") if x.strip()),
        heads=args.spending_heads, dropout=args.spending_dropout,
        storage_impl=args.spending_storage, energy_precision=args.spending_precision,
        diagnostics=args.spending_diagnostics)
    return {**config.to_dict(), "coordinates": args.spending_coordinates}


def identity_standardizer(n_u, n_y, standardizer_type):
    return standardizer_type(torch.zeros(n_u), torch.ones(n_u), torch.zeros(n_y), torch.ones(n_y))


def target_compatibility(bench, norm, gamma):
    """Informative feasibility audit; never changes gamma/beta or the dataset splits."""
    rows = []
    for split, trajectories in (("train", bench.train), ("validation", bench.validation)):
        for sequence in trajectories:
            u = norm.norm_u(torch.as_tensor(np.asarray(sequence.u), dtype=torch.float64)).double()
            y = norm.norm_y(torch.as_tensor(np.asarray(sequence.y), dtype=torch.float64)).double()
            required = (y.square().sum(-1) - (gamma * u).square().sum(-1)).cumsum(0).clamp(min=0).max().item()
            rows.append({"split": split, "trajectory": sequence.name,
                         "minimum_initial_energy_for_exact_target_prefixes": required})
    return rows


def save_checkpoint(path, model, norm, history, metadata):
    """New tagged format; never reinterpret projection-wrapper/SSM checkpoints."""
    projected = model.spending_config.readout == "projected"
    payload = {"format": "energy_projection_v1" if projected else "direct_spending_v1",
               "family": "energy_projection" if projected else "direct_spending",
               "parameter_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
               "n_u": model.model.d_input, "n_y": model.model.d_output,
               "config": model.spending_config.to_dict(), "coordinates": model.coordinates,
               "model_state_dict": model.state_dict(),
               "normalizer": {key: getattr(norm, key).detach().cpu()
                              for key in ("u_mean", "u_std", "y_mean", "y_std")},
               "history": history, "metadata": metadata, "certificate": model.certificate()}
    torch.save(payload, Path(path))


def load_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    tags = {("direct_spending_v1", "direct_spending"): "spending",
            ("energy_projection_v1", "energy_projection"): "projected"}
    readout = tags.get((payload.get("format"), payload.get("family")))
    if readout is None:
        raise ValueError("Not a supported direct_spending_v1/energy_projection_v1 checkpoint; older model formats remain distinct")
    config = SpendingConfig.from_dict(payload["config"])
    if config.readout != readout:
        raise ValueError("Checkpoint family and readout disagree; refusing to reinterpret weights")
    # Infer old checkpoints' dtype from their actual weights. Preserving gain
    # precision is necessary to reproduce the saved fitted-gamma certificate.
    dtype = next(value.dtype for value in payload["model_state_dict"].values()
                 if torch.is_tensor(value) and value.is_floating_point())
    model = SpendingSim(payload["n_u"], payload["n_y"], config,
                         coordinates=payload["coordinates"]).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload
