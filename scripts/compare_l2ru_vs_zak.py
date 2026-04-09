#!/usr/bin/env python3
"""
End-to-end comparison between the `l2ru` and `zak` parametrizations in DeepSSM.

What this script does
---------------------
1. Loads one or more nonlinear system-identification datasets.
2. Trains both parametrizations with the same surrounding DeepSSM scaffold in two regimes:
   - an accuracy regime with trainable global SSM gamma
   - a robustness regime with fixed prescribed gamma
3. Compares:
   - clean system-ID accuracy in the trainable-gamma regime
   - robustness under bounded-energy input perturbations in the fixed-gamma regime
   - empirical finite-horizon l2 gain vs the prescribed certificate in the fixed-gamma regime
   - forward / backward runtime in supported modes in the fixed-gamma regime
   - parameter counts, convergence, and scan-vs-loop consistency
4. Saves publication-style figures plus machine-readable JSON / CSV summaries.

Datasets
--------
- Requires `nonlinear_benchmarks`.
- Dataset names such as `Cascaded_Tanks`, `WienerHammer`, or `Silverbox`
  are supported.

Typical usage
-------------
Smoke test:
    python scripts/compare_l2ru_vs_zak.py \
        --datasets Cascaded_Tanks \
        --epochs 8 \
        --accuracy-gamma-init 1.0 \
        --robustness-gamma 2.0 \
        --out-dir runs/l2ru_vs_zak_smoke
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = REPO_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import nonlinear_benchmarks  # type: ignore
except Exception:
    nonlinear_benchmarks = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from src.neural_ssm import DeepSSM, SSMConfig


PLOT_COLORS = {
    "l2ru": "#0f766e",
    "zak": "#b45309",
    "scan": "#2563eb",
    "loop": "#dc2626",
    "gain": "#7c3aed",
    "target": "#111827",
    "truth": "#111827",
}


def supported_modes_for_param(param: str) -> tuple[str, ...]:
    if param == "l2ru":
        return ("loop",)
    if param == "zak":
        return ("scan", "loop")
    return ("scan", "loop")


def resolve_mode_for_param(param: str, requested_mode: str) -> str:
    supported = supported_modes_for_param(param)
    if requested_mode in supported:
        return requested_mode
    if "loop" in supported:
        return "loop"
    return supported[0]

METRIC_LABELS = {
    "rmse": "RMSE",
    "nrmse": "NRMSE",
    "mae": "MAE",
    "r2": r"$R^2$",
    "fit": "Fit Index (%)",
}


@dataclass
class DatasetBundle:
    name: str
    pretty_name: str
    u_train: torch.Tensor
    y_train: torch.Tensor
    u_val: torch.Tensor
    y_val: torch.Tensor
    u_test: torch.Tensor
    y_test: torch.Tensor
    init_window: int
    description: str

    @property
    def d_input(self) -> int:
        return int(self.u_train.shape[-1])

    @property
    def d_output(self) -> int:
        return int(self.y_train.shape[-1])


@dataclass
class TrainConfig:
    epochs: int = 2000
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 30
    min_delta: float = 1e-5
    val_interval: int = 1
    window_len: int = 256
    normalize_data: bool = True
    train_mode: str = "scan"
    eval_mode: str = "scan"
    use_early_stopping: bool = False


@dataclass
class ModelShape:
    d_model: int = 8
    d_state: int = 16
    n_layers: int = 4
    ff: str = "LGLU2"
    d_hidden: int = 12
    nl_layers: int = 3
    scale: float = 1.0
    init: str = "eye"
    rho: float = 0.95
    rmin: float = 0.95
    rmax: float = 0.98
    max_phase: float = 2 * math.pi
    max_phase_b: float = 2 * math.pi
    phase_center: float = 0.0
    random_phase: bool = True
    learn_x0: bool = True
    zak_d_margin: float = 0.25
    zak_x2_margin: float = 0.9
    zak_x2_init_scale: float = 0.1


@dataclass
class GainConfig:
    probe_samples: int = 128
    probe_batch_size: int = 32
    probe_radius: float = 1.0
    worst_case_restarts: int = 8
    worst_case_steps: int = 80
    worst_case_batch: int = 4
    worst_case_lr: float = 8e-2
    local_gain_windows: int = 8
    local_gain_eps: float = 0.25
    local_gain_steps: int = 60
    local_gain_restarts: int = 4
    local_gain_lr: float = 6e-2
    disturbance_radii: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
    disturbance_trials: int = 4


@dataclass
class RuntimeConfig:
    batch_size: int = 8
    seq_lens: tuple[int, ...] = (64, 128, 256, 512, 1024)
    repeats_forward: int = 20
    repeats_backward: int = 10
    warmup: int = 3


@dataclass(frozen=True)
class ExperimentRegime:
    name: str
    pretty_name: str
    gamma: float
    train_global_gamma: bool
    compute_gain: bool
    compute_runtime: bool
    compute_robustness: bool
    compute_consistency: bool


@dataclass
class EarlyStoppingState:
    best_score: float = float("inf")
    best_epoch: int = -1
    bad_epochs: int = 0
    best_state: Optional[dict[str, torch.Tensor]] = None


class ChannelwiseStandardizer:
    """Per-channel affine normalization over all sample dimensions."""

    def __init__(
        self,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ):
        self.u_mean = u_mean.detach().cpu()
        self.u_std = u_std.detach().cpu()
        self.y_mean = y_mean.detach().cpu()
        self.y_std = y_std.detach().cpu()

    @staticmethod
    def _reduce_dims(x: torch.Tensor) -> tuple[int, ...]:
        return tuple(range(x.dim() - 1))

    @classmethod
    def fit(cls, u: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> "ChannelwiseStandardizer":
        reduce_u = cls._reduce_dims(u)
        reduce_y = cls._reduce_dims(y)
        u_mean = u.mean(dim=reduce_u, keepdim=True)
        u_std = u.std(dim=reduce_u, keepdim=True, unbiased=False).clamp_min(eps)
        y_mean = y.mean(dim=reduce_y, keepdim=True)
        y_std = y.std(dim=reduce_y, keepdim=True, unbiased=False).clamp_min(eps)
        return cls(u_mean, u_std, y_mean, y_std)

    def _match(self, tensor: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        return stats.to(device=tensor.device, dtype=tensor.dtype)

    def transform_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self._match(u, self.u_mean)) / self._match(u, self.u_std)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self._match(y, self.y_mean)) / self._match(y, self.y_std)

    def inverse_transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self._match(y, self.y_std) + self._match(y, self.y_mean)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "u_mean": self.u_mean.flatten().tolist(),
            "u_std": self.u_std.flatten().tolist(),
            "y_mean": self.y_mean.flatten().tolist(),
            "y_std": self.y_std.flatten().tolist(),
        }


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "axes.linewidth": 0.9,
            "legend.frameon": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_bln(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    xt = torch.as_tensor(x, dtype=torch.float32)
    if xt.ndim == 1:
        return xt[None, :, None]
    if xt.ndim == 2:
        return xt[None, :, :]
    if xt.ndim == 3:
        return xt
    raise ValueError(f"unsupported tensor rank {xt.ndim}")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def regime_label(name: str, fixed_gamma: Optional[float] = None) -> str:
    if name == "accuracy_trainable_gamma":
        return r"Accuracy regime ($\gamma$ trainable)"
    if name == "robustness_fixed_gamma":
        gamma_text = "2" if fixed_gamma is None else f"{fixed_gamma:g}"
        return rf"Robustness regime ($\gamma={gamma_text}$ fixed)"
    return name.replace("_", " ")


def short_regime_label(name: str, fixed_gamma: Optional[float] = None) -> str:
    if name == "accuracy_trainable_gamma":
        return r"trainable $\gamma$"
    if name == "robustness_fixed_gamma":
        gamma_text = "2" if fixed_gamma is None else f"{fixed_gamma:g}"
        return rf"fixed $\gamma={gamma_text}$"
    return name.replace("_", " ")


def resolve_benchmark_fn(name: str, fallback: Optional[str] = None) -> tuple[Callable[..., Any], str]:
    if nonlinear_benchmarks is None:
        raise ImportError(
            "nonlinear_benchmarks is not installed. Install it to use this comparison script."
        )

    dataset_name = name.strip()
    if dataset_name.endswith("()"):
        dataset_name = dataset_name[:-2]

    def candidate_names(base: str) -> list[str]:
        if base.endswith("BenchMark") or base.endswith("Benchmark"):
            return [base]
        return [
            base,
            f"{base}BenchMark",
            f"{base}Benchmark",
            f"{base.capitalize()}BenchMark",
            f"{base.capitalize()}Benchmark",
        ]

    for candidate in candidate_names(dataset_name):
        if hasattr(nonlinear_benchmarks, candidate):
            return getattr(nonlinear_benchmarks, candidate), candidate

    for attr in dir(nonlinear_benchmarks):
        attr_lower = attr.lower()
        if attr_lower == dataset_name.lower():
            return getattr(nonlinear_benchmarks, attr), attr
        if attr_lower.startswith(dataset_name.lower()) and attr_lower.endswith("benchmark"):
            return getattr(nonlinear_benchmarks, attr), attr

    if fallback is not None:
        return resolve_benchmark_fn(fallback, fallback=None)

    raise ValueError(f"could not resolve benchmark `{name}`")


def load_nonlinear_benchmark_dataset(
    name: str,
    *,
    fallback: Optional[str],
) -> DatasetBundle:
    fn, resolved_name = resolve_benchmark_fn(name, fallback=fallback)
    train_split, test_split = fn()

    u_train_np, y_train_np = train_split
    u_test_np, y_test_np = test_split
    init_window = int(getattr(test_split, "state_initialization_window_length", 0))

    u_train_full = ensure_bln(u_train_np)
    y_train_full = ensure_bln(y_train_np)
    u_test = ensure_bln(u_test_np)
    y_test = ensure_bln(y_test_np)

    return DatasetBundle(
        name=name,
        pretty_name=resolved_name.replace("BenchMark", "").replace("_", " "),
        u_train=u_train_full,
        y_train=y_train_full,
        u_val=u_test.clone(),
        y_val=y_test.clone(),
        u_test=u_test,
        y_test=y_test,
        init_window=init_window,
        description=(
            f"Loaded from nonlinear_benchmarks.{resolved_name}(). "
            "No temporal slicing is applied: the benchmark train split is used as-is "
            "for optimization, and the benchmark test split is used for validation/reporting."
        ),
    )


def load_dataset(
    name: str,
    *,
    benchmark_fallback: Optional[str],
) -> DatasetBundle:
    return load_nonlinear_benchmark_dataset(
        name,
        fallback=benchmark_fallback,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_model(
    *,
    param: str,
    gamma: Optional[float],
    train_branch_gamma: bool,
    train_global_gamma: bool,
    d_input: int,
    d_output: int,
    shape: ModelShape,
    device: torch.device,
) -> DeepSSM:
    config = SSMConfig(
        d_model=shape.d_model,
        d_state=shape.d_state,
        n_layers=shape.n_layers,
        ff=shape.ff,
        rmin=shape.rmin,
        rmax=shape.rmax,
        max_phase=shape.max_phase,
        scale=shape.scale,
        d_hidden=shape.d_hidden,
        nl_layers=shape.nl_layers,
        param=param,
        gamma=gamma,
        train_gamma=train_branch_gamma,
        init=shape.init,
        rho=shape.rho,
        max_phase_b=shape.max_phase_b,
        phase_center=shape.phase_center,
        random_phase=shape.random_phase,
        learn_x0=shape.learn_x0,
        zak_d_margin=shape.zak_d_margin,
        zak_x2_margin=shape.zak_x2_margin,
        zak_x2_init_scale=shape.zak_x2_init_scale,
    )
    model = DeepSSM(d_input=d_input, d_output=d_output, config=config).to(device)
    if train_global_gamma:
        if gamma is None:
            raise ValueError("train_global_gamma=True requires an initial gamma value")
        if not getattr(model, "use_cert_scaling", False) or not hasattr(model, "gamma_t"):
            raise ValueError("global gamma can only be trained when certificate scaling is active")
        gamma_init = model.gamma_t.detach().clone().to(device=device)
        if "gamma_t" in model._buffers:
            del model._buffers["gamma_t"]
        model.register_parameter("gamma_t", nn.Parameter(gamma_init))
    return model


def maybe_match_param_budget(
    *,
    target_param: str,
    reference_param: str,
    gamma: Optional[float],
    train_branch_gamma: bool,
    train_global_gamma: bool,
    d_input: int,
    d_output: int,
    shape: ModelShape,
    device: torch.device,
    enabled: bool,
    search_min: int,
    search_max: int,
) -> tuple[ModelShape, dict[str, Any]]:

    reference_model = build_model(
        param=reference_param,
        gamma=gamma,
        train_branch_gamma=train_branch_gamma,
        train_global_gamma=train_global_gamma,
        d_input=d_input,
        d_output=d_output,
        shape=shape,
        device=device,
    )
    reference_params = count_parameters(reference_model)
    del reference_model

    if not enabled:
        target_shape = copy.deepcopy(shape)
        target_model = build_model(
            param=target_param,
            gamma=gamma,
            train_branch_gamma=train_branch_gamma,
            train_global_gamma=train_global_gamma,
            d_input=d_input,
            d_output=d_output,
            shape=target_shape,
            device=device,
        )
        target_params = count_parameters(target_model)
        del target_model
        abs_gap = abs(target_params - reference_params)
        return target_shape, {
            "enabled": False,
            "reference_param": reference_param,
            "target_param": target_param,
            "reference_parameter_count": int(reference_params),
            "target_parameter_count": int(target_params),
            "absolute_gap": int(abs_gap),
            "relative_gap": float(abs_gap / max(reference_params, 1)),
            "target_shape": asdict(target_shape),
        }

    best_shape = copy.deepcopy(shape)
    best_diff = float("inf")
    best_target_params = None

    for candidate_state in range(max(1, search_min), max(search_min, search_max) + 1):
        candidate_shape = copy.deepcopy(shape)
        candidate_shape.d_state = candidate_state
        candidate_model = build_model(
            param=target_param,
            gamma=gamma,
            train_branch_gamma=train_branch_gamma,
            train_global_gamma=train_global_gamma,
            d_input=d_input,
            d_output=d_output,
            shape=candidate_shape,
            device=device,
        )
        candidate_params = count_parameters(candidate_model)
        diff = abs(candidate_params - reference_params)
        del candidate_model
        if diff < best_diff:
            best_diff = diff
            best_shape = candidate_shape
            best_target_params = candidate_params

    return best_shape, {
        "enabled": True,
        "reference_param": reference_param,
        "target_param": target_param,
        "reference_parameter_count": int(reference_params),
        "target_parameter_count": int(best_target_params if best_target_params is not None else reference_params),
        "absolute_gap": int(best_diff),
        "relative_gap": float(best_diff / max(reference_params, 1)),
        "target_shape": asdict(best_shape),
    }


def flatten_metric_region(x: torch.Tensor, init_window: int) -> torch.Tensor:
    x = x.detach().cpu()
    if x.ndim != 3:
        raise ValueError("expected BLN tensor")
    if x.shape[1] <= init_window:
        return x.reshape(-1, x.shape[-1])
    return x[:, init_window:, :].reshape(-1, x.shape[-1])


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, init_window: int) -> dict[str, Any]:
    yt = flatten_metric_region(y_true, init_window).numpy()
    yp = flatten_metric_region(y_pred, init_window).numpy()
    err = yp - yt
    denom_std = np.std(yt, axis=0, ddof=0)
    denom_std = np.where(denom_std < 1e-12, 1.0, denom_std)
    centered = yt - np.mean(yt, axis=0, keepdims=True)
    centered_norm = np.linalg.norm(centered, axis=0)
    centered_norm = np.where(centered_norm < 1e-12, 1.0, centered_norm)
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum(centered ** 2, axis=0)
    ss_tot = np.where(ss_tot < 1e-12, 1.0, ss_tot)

    rmse_c = np.sqrt(np.mean(err ** 2, axis=0))
    nrmse_c = rmse_c / denom_std
    mae_c = np.mean(np.abs(err), axis=0)
    r2_c = 1.0 - ss_res / ss_tot
    fit_c = 100.0 * (1.0 - np.linalg.norm(err, axis=0) / centered_norm)

    summary = {
        "rmse": float(np.mean(rmse_c)),
        "nrmse": float(np.mean(nrmse_c)),
        "mae": float(np.mean(mae_c)),
        "r2": float(np.mean(r2_c)),
        "fit": float(np.mean(fit_c)),
        "rmse_per_channel": rmse_c.tolist(),
        "nrmse_per_channel": nrmse_c.tolist(),
        "mae_per_channel": mae_c.tolist(),
        "r2_per_channel": r2_c.tolist(),
        "fit_per_channel": fit_c.tolist(),
    }
    return summary


def masked_mse(y_pred: torch.Tensor, y_true: torch.Tensor, init_window: int) -> torch.Tensor:
    if y_pred.shape[1] > init_window:
        y_pred = y_pred[:, init_window:, :]
        y_true = y_true[:, init_window:, :]
    return torch.mean((y_pred - y_true) ** 2)


@torch.no_grad()
def predict(
    model: DeepSSM,
    u: torch.Tensor,
    *,
    normalizer: Optional[ChannelwiseStandardizer],
    mode: str,
) -> torch.Tensor:
    model.eval()
    u_norm = normalizer.transform_u(u) if normalizer is not None else u
    y_norm, _ = model(u_norm, mode=mode, reset_state=True, detach_state=True)
    if normalizer is not None:
        y_norm = normalizer.inverse_transform_y(y_norm)
    return y_norm.detach().cpu()


def train_model(
    *,
    model: DeepSSM,
    dataset: DatasetBundle,
    train_cfg: TrainConfig,
    device: torch.device,
    seed: int,
    progress_label: Optional[str] = None,
) -> tuple[dict[str, Any], Optional[ChannelwiseStandardizer]]:
    set_seed(seed)

    u_train = dataset.u_train.to(device)
    y_train = dataset.y_train.to(device)
    u_val = dataset.u_val.to(device)
    y_val = dataset.y_val.to(device)

    normalizer = ChannelwiseStandardizer.fit(u_train, y_train) if train_cfg.normalize_data else None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(train_cfg.epochs, 1),
        eta_min=train_cfg.lr * 0.1,
    )

    stopping = EarlyStoppingState()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "epoch_time_sec": [],
    }
    seq_count = int(u_train.shape[0])
    steps_per_epoch = max((seq_count + max(train_cfg.batch_size, 1) - 1) // max(train_cfg.batch_size, 1), 1)

    epoch_iter = range(train_cfg.epochs)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(
            epoch_iter,
            total=train_cfg.epochs,
            desc=progress_label or "training",
            dynamic_ncols=True,
            leave=True,
        )
        epoch_iter = pbar

    for epoch in epoch_iter:
        start_time = time.perf_counter()
        model.train()
        running_loss = 0.0
        n_batches = 0

        permutation = torch.randperm(seq_count, device=device)
        for start in range(0, seq_count, train_cfg.batch_size):
            batch_idx = permutation[start : start + train_cfg.batch_size]
            u_batch = u_train[batch_idx]
            y_batch = y_train[batch_idx]
            if normalizer is not None:
                u_batch_in = normalizer.transform_u(u_batch)
                y_batch_ref = normalizer.transform_y(y_batch)
            else:
                u_batch_in = u_batch
                y_batch_ref = y_batch

            optimizer.zero_grad(set_to_none=True)
            y_pred, _ = model(u_batch_in, mode=train_cfg.train_mode, reset_state=True, detach_state=True)
            loss = masked_mse(y_pred, y_batch_ref, dataset.init_window)
            loss.backward()
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            n_batches += 1

        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)
        history["epoch_time_sec"].append(float(time.perf_counter() - start_time))

        if (epoch + 1) % train_cfg.val_interval == 0 or epoch == train_cfg.epochs - 1:
            with torch.no_grad():
                y_val_pred = predict(model, u_val, normalizer=normalizer, mode=train_cfg.eval_mode)
                val_metrics = compute_metrics(y_val.cpu(), y_val_pred, dataset.init_window)
                val_loss = float(val_metrics["rmse"])
        else:
            val_metrics = None
            val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")

        history["val_loss"].append(float(val_loss))
        history["val_rmse"].append(float(val_loss))
        if pbar is not None:
            pbar.set_postfix(
                train_loss=f"{train_loss:.3e}",
                val_rmse=f"{val_loss:.3e}",
                steps=steps_per_epoch,
                refresh=False,
            )

        improved = val_loss < (stopping.best_score - train_cfg.min_delta)
        if improved:
            stopping.best_score = float(val_loss)
            stopping.best_epoch = epoch
            stopping.bad_epochs = 0
            stopping.best_state = copy.deepcopy(model.state_dict())
        else:
            stopping.bad_epochs += 1
            if train_cfg.use_early_stopping and stopping.bad_epochs >= train_cfg.patience:
                break

    if pbar is not None:
        pbar.close()

    if train_cfg.use_early_stopping and stopping.best_state is not None:
        model.load_state_dict(stopping.best_state)

    return {
        "history": history,
        "best_epoch": int(stopping.best_epoch),
        "best_val_rmse": float(stopping.best_score),
        "steps_per_epoch": int(steps_per_epoch),
        "train_sequence_count": int(seq_count),
        "train_sequence_length": int(u_train.shape[1]),
    }, normalizer


def summarize_certificate(model: DeepSSM) -> dict[str, Any]:
    block_entries: list[dict[str, Any]] = []
    block_gain_product = 1.0

    for idx, block in enumerate(model.blocks):
        entry: dict[str, Any] = {"index": idx}
        branch_gamma = None
        ff_lip = 1.0
        res_scale = None

        if hasattr(block.lru, "gamma"):
            branch_gamma = float(block.lru.gamma.detach().abs().cpu())
            entry["branch_gamma"] = branch_gamma
        if hasattr(block.ff, "lip"):
            ff_lip = float(block.ff.lip.detach().cpu())
            entry["ff_lip"] = ff_lip
        if hasattr(block, "res_scale"):
            res_scale = float(block.res_scale.detach().cpu())
            entry["res_scale"] = res_scale

        if branch_gamma is not None and res_scale is not None:
            effective_block_gain = 1.0 + res_scale * branch_gamma * ff_lip
            block_gain_product *= effective_block_gain
            entry["effective_block_gain"] = float(effective_block_gain)

        block_entries.append(entry)

    summary = {
        "available": bool(getattr(model, "use_cert_scaling", False) and hasattr(model, "gamma_t")),
        "blocks": block_entries,
        "block_gain_product": float(block_gain_product),
    }

    if summary["available"]:
        enc_norm = float(torch.linalg.matrix_norm(model.encoder_w.detach(), ord=2).cpu())
        dec_norm = float(torch.linalg.matrix_norm(model.decoder_w.detach(), ord=2).cpu())
        target_gamma = float(model.gamma_t.detach().cpu())
        decoder_scale = target_gamma / (enc_norm * dec_norm * block_gain_product + 1e-12)
        summary.update(
            {
                "target_gamma": target_gamma,
                "encoder_norm": enc_norm,
                "decoder_norm": dec_norm,
                "decoder_scale": float(decoder_scale),
            }
        )
    return summary


def flatten_l2(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).norm(p=2, dim=1)


def normalize_to_radius(x: torch.Tensor, radius: float) -> torch.Tensor:
    norms = flatten_l2(x).clamp_min(1e-12)
    view_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    return x * (float(radius) / norms).reshape(view_shape)


def run_model(
    model: DeepSSM,
    u: torch.Tensor,
    *,
    normalizer: Optional[ChannelwiseStandardizer],
    mode: str,
) -> torch.Tensor:
    u_norm = normalizer.transform_u(u) if normalizer is not None else u
    y_norm, _ = model(u_norm, mode=mode, reset_state=True, detach_state=True)
    if normalizer is not None:
        y_norm = normalizer.inverse_transform_y(y_norm)
    return y_norm


def run_model_normalized(
    model: DeepSSM,
    u_norm: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    y_norm, _ = model(u_norm, mode=mode, reset_state=True, detach_state=True)
    return y_norm


@torch.no_grad()
def random_probe_gains(
    model: DeepSSM,
    *,
    seq_len: int,
    radius: float,
    n_samples: int,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> np.ndarray:
    values = []
    remaining = int(n_samples)
    while remaining > 0:
        cur = min(remaining, int(batch_size))
        u_norm = torch.randn(cur, seq_len, model.d_input, device=device)
        u_norm = normalize_to_radius(u_norm, radius)
        y_norm = run_model_normalized(model, u_norm, mode=mode)
        values.append((flatten_l2(y_norm) / flatten_l2(u_norm).clamp_min(1e-12)).detach().cpu().numpy())
        remaining -= cur
    return np.concatenate(values, axis=0)


def optimize_finite_horizon_gain(
    model: DeepSSM,
    *,
    seq_len: int,
    radius: float,
    restarts: int,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    best_estimate = 0.0
    best_history: list[float] = []
    restart_bests: list[float] = []

    batches = math.ceil(max(restarts, 1) / max(batch_size, 1))
    for batch_idx in range(batches):
        cur = min(batch_size, restarts - batch_idx * batch_size)
        v = torch.randn(cur, seq_len, model.d_input, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([v], lr=lr)
        local_best = torch.zeros(cur, device=device)
        local_history: list[float] = []

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            u_norm = normalize_to_radius(v, radius)
            y_norm = run_model_normalized(model, u_norm, mode=mode)
            gains_sq = (flatten_l2(y_norm) ** 2) / (flatten_l2(u_norm).clamp_min(1e-12) ** 2)
            loss = -gains_sq.mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                u_eval = normalize_to_radius(v, radius)
                y_eval = run_model_normalized(model, u_eval, mode=mode)
                gains = flatten_l2(y_eval) / flatten_l2(u_eval).clamp_min(1e-12)
                local_best = torch.maximum(local_best, gains)
                local_history.append(float(gains.max().detach().cpu()))

        restart_bests.extend(local_best.detach().cpu().tolist())
        if float(local_best.max().item()) > best_estimate:
            best_estimate = float(local_best.max().item())
            best_history = local_history

    return {
        "estimate": float(best_estimate),
        "p95": float(np.percentile(restart_bests, 95)) if restart_bests else 0.0,
        "mean": float(np.mean(restart_bests)) if restart_bests else 0.0,
        "history_max": best_history,
    }


def sample_test_windows(
    u: torch.Tensor,
    *,
    n_windows: int,
    window_len: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    seq_count, seq_len, _ = u.shape
    length = min(window_len, seq_len)
    windows = []
    for _ in range(n_windows):
        seq_idx = int(rng.integers(0, seq_count))
        max_start = max(seq_len - length, 0)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        windows.append(u[seq_idx, start : start + length, :])
    return torch.stack(windows, dim=0)


def estimate_local_test_gain(
    model: DeepSSM,
    *,
    normalizer: Optional[ChannelwiseStandardizer],
    base_inputs: torch.Tensor,
    eps: float,
    steps: int,
    restarts: int,
    lr: float,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    base_inputs = base_inputs.to(device)
    base_inputs_norm = normalizer.transform_u(base_inputs) if normalizer is not None else base_inputs
    with torch.no_grad():
        y_base = run_model_normalized(model, base_inputs_norm, mode=mode)

    best_per_window = torch.zeros(base_inputs_norm.shape[0], device=device)
    for _ in range(restarts):
        v = torch.randn_like(base_inputs_norm, requires_grad=True)
        optimizer = torch.optim.Adam([v], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            delta = normalize_to_radius(v, eps)
            y_pert = run_model_normalized(model, base_inputs_norm + delta, mode=mode)
            gains = flatten_l2(y_pert - y_base) / max(float(eps), 1e-12)
            loss = -(gains ** 2).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta_eval = normalize_to_radius(v, eps)
                y_pert_eval = run_model_normalized(model, base_inputs_norm + delta_eval, mode=mode)
                gains_eval = flatten_l2(y_pert_eval - y_base) / max(float(eps), 1e-12)
                best_per_window = torch.maximum(best_per_window, gains_eval)

    best_np = best_per_window.detach().cpu().numpy()
    return {
        "eps": float(eps),
        "mean": float(best_np.mean()),
        "p95": float(np.percentile(best_np, 95)),
        "max": float(best_np.max()),
    }


def sample_energy_bounded_noise_like(x: torch.Tensor, radius: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return normalize_to_radius(noise, radius)


def raw_gain_upper_bound(
    *,
    certificate: dict[str, Any],
    normalizer: Optional[ChannelwiseStandardizer],
) -> Optional[float]:
    if normalizer is None or not certificate.get("available"):
        return None
    u_std = normalizer.u_std.reshape(-1).numpy()
    y_std = normalizer.y_std.reshape(-1).numpy()
    input_scale = 1.0 / max(float(np.min(np.abs(u_std))), 1e-12)
    output_scale = max(float(np.max(np.abs(y_std))), 1e-12)
    return float(certificate["target_gamma"] * output_scale * input_scale)


def evaluate_under_disturbance(
    model: DeepSSM,
    *,
    dataset: DatasetBundle,
    normalizer: Optional[ChannelwiseStandardizer],
    radii: tuple[float, ...],
    trials: int,
    seed: int,
    device: torch.device,
    mode: str,
) -> list[dict[str, Any]]:
    u_test = dataset.u_test.to(device)
    y_test = dataset.y_test.to(device)
    outputs = []

    for radius in radii:
        rmse_vals = []
        gain_vals = []
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + int(round(1000 * radius)))
        for _ in range(trials):
            noise = sample_energy_bounded_noise_like(u_test, radius, generator)
            with torch.no_grad():
                y_clean = run_model(model, u_test, normalizer=normalizer, mode=mode)
                y_noisy = run_model(model, u_test + noise, normalizer=normalizer, mode=mode)
            metrics = compute_metrics(y_test.cpu(), y_noisy.cpu(), dataset.init_window)
            gain = flatten_l2((y_noisy - y_clean).detach().cpu()) / flatten_l2(noise.detach().cpu()).clamp_min(1e-12)
            rmse_vals.append(metrics["rmse"])
            gain_vals.extend(gain.tolist())

        outputs.append(
            {
                "radius": float(radius),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals)),
                "gain_mean": float(np.mean(gain_vals)),
                "gain_p95": float(np.percentile(gain_vals, 95)),
            }
        )

    return outputs


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_runtime(
    model: DeepSSM,
    *,
    runtime_cfg: RuntimeConfig,
    device: torch.device,
    mode: str,
) -> list[dict[str, Any]]:
    results = []

    for seq_len in runtime_cfg.seq_lens:
        u = torch.randn(runtime_cfg.batch_size, seq_len, model.d_input, device=device)
        target = torch.randn(runtime_cfg.batch_size, seq_len, model.d_output, device=device)

        with torch.no_grad():
            for _ in range(runtime_cfg.warmup):
                model(u, mode=mode, reset_state=True, detach_state=True)
        synchronize_if_needed(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            for _ in range(runtime_cfg.repeats_forward):
                model(u, mode=mode, reset_state=True, detach_state=True)
            synchronize_if_needed(device)
            t1 = time.perf_counter()
        forward_sec = (t1 - t0) / max(runtime_cfg.repeats_forward, 1)

        model.train()
        for _ in range(runtime_cfg.warmup):
            model.zero_grad(set_to_none=True)
            y, _ = model(u, mode=mode, reset_state=True, detach_state=True)
            loss = torch.mean((y - target) ** 2)
            loss.backward()
        synchronize_if_needed(device)

        t0 = time.perf_counter()
        for _ in range(runtime_cfg.repeats_backward):
            model.zero_grad(set_to_none=True)
            y, _ = model(u, mode=mode, reset_state=True, detach_state=True)
            loss = torch.mean((y - target) ** 2)
            loss.backward()
        synchronize_if_needed(device)
        t1 = time.perf_counter()
        backward_sec = (t1 - t0) / max(runtime_cfg.repeats_backward, 1)
        model.eval()

        results.append(
            {
                "seq_len": int(seq_len),
                "mode": mode,
                "forward_sec": float(forward_sec),
                "backward_sec": float(backward_sec),
            }
        )

    return results


def safe_runtime_benchmark(
    model: DeepSSM,
    *,
    runtime_cfg: RuntimeConfig,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    try:
        return {"available": True, "points": benchmark_runtime(model, runtime_cfg=runtime_cfg, device=device, mode=mode)}
    except Exception as exc:
        return {"available": False, "error": repr(exc), "points": []}


def compare_scan_loop_consistency(
    model: DeepSSM,
    *,
    param: str,
    dataset: DatasetBundle,
    normalizer: Optional[ChannelwiseStandardizer],
) -> dict[str, Any]:
    supported = supported_modes_for_param(param)
    if not {"scan", "loop"}.issubset(set(supported)):
        return {"available": False, "error": f"{param} does not support both scan and loop"}
    try:
        y_scan = predict(model, dataset.u_test, normalizer=normalizer, mode="scan")
        y_loop = predict(model, dataset.u_test, normalizer=normalizer, mode="loop")
        diff = torch.max(torch.abs(y_scan - y_loop)).item()
        rel = torch.norm((y_scan - y_loop).reshape(-1)).item() / max(torch.norm(y_scan.reshape(-1)).item(), 1e-12)
        return {"available": True, "max_abs_diff": float(diff), "relative_l2_diff": float(rel)}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def aggregate_group(records: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec["dataset"]), str(rec["param"]), str(rec["regime"]))
        grouped.setdefault(key, []).append(rec)

    rows = []
    for (dataset, param, regime), group in grouped.items():
        row: dict[str, Any] = {
            "dataset": dataset,
            "param": param,
            "regime": regime,
            "n_runs": len(group),
        }
        for metric in metrics:
            vals = np.asarray([float(item[metric]) for item in group], dtype=np.float64)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std())
        learned_gamma = [item["trained_global_gamma"] for item in group if item.get("trained_global_gamma") is not None]
        fixed_gamma = [item["gamma_fixed"] for item in group if item.get("gamma_fixed") is not None]
        if learned_gamma:
            vals = np.asarray(learned_gamma, dtype=np.float64)
            row["trained_global_gamma_mean"] = float(vals.mean())
            row["trained_global_gamma_std"] = float(vals.std())
        if fixed_gamma:
            vals = np.asarray(fixed_gamma, dtype=np.float64)
            row["gamma_fixed_mean"] = float(vals.mean())
            row["gamma_fixed_std"] = float(vals.std())
        row["regime_pretty"] = regime_label(regime, row.get("gamma_fixed_mean"))
        rows.append(row)
    return rows


def aggregate_gain_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["param"]), str(row["regime"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    fields = [
        "target_gamma",
        "random_probe_mean",
        "random_probe_p95",
        "random_probe_max",
        "worst_case_estimate",
        "local_test_gain_mean",
        "local_test_gain_p95",
        "local_test_gain_max",
        "worst_case_over_gamma",
        "random_probe_max_over_gamma",
        "local_test_gain_max_over_gamma",
    ]
    for (dataset, param, regime), group in grouped.items():
        gamma_fixed = next((row.get("gamma_fixed") for row in group if row.get("gamma_fixed") is not None), None)
        item: dict[str, Any] = {
            "dataset": dataset,
            "param": param,
            "regime": regime,
            "gamma_fixed": gamma_fixed,
            "label": f"{param.upper()} | {dataset} | {short_regime_label(regime, gamma_fixed)}",
        }
        for field in fields:
            vals = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            item[field] = float(vals.mean())
            item[f"{field}_std"] = float(vals.std())
        summary.append(item)
    return summary


def aggregate_runtime_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["param"]), str(row["regime"]), str(row["mode"]), int(row["seq_len"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (param, regime, mode, seq_len), group in grouped.items():
        fwd = np.asarray([float(row["forward_sec"]) for row in group], dtype=np.float64)
        bwd = np.asarray([float(row["backward_sec"]) for row in group], dtype=np.float64)
        summary.append(
            {
                "param": param,
                "regime": regime,
                "mode": mode,
                "seq_len": seq_len,
                "forward_sec": float(fwd.mean()),
                "forward_sec_std": float(fwd.std()),
                "backward_sec": float(bwd.mean()),
                "backward_sec_std": float(bwd.std()),
            }
        )
    return summary


def aggregate_robustness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["param"]), str(row["regime"]), float(row["radius"]))
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dataset, param, regime, radius), group in grouped.items():
        gamma_fixed = next((row.get("gamma_fixed") for row in group if row.get("gamma_fixed") is not None), None)
        item: dict[str, Any] = {
            "dataset": dataset,
            "param": param,
            "regime": regime,
            "gamma_fixed": gamma_fixed,
            "radius": radius,
            "label": f"{param.upper()} | {dataset} | {short_regime_label(regime, gamma_fixed)}",
        }
        for field in ("rmse_mean", "rmse_std", "gain_mean", "gain_p95"):
            vals = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            item[field] = float(vals.mean())
            item[f"{field}_std"] = float(vals.std())
        summary.append(item)
    return summary


def summary_row(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    param: str,
    regime: str,
) -> Optional[dict[str, Any]]:
    return next(
        (
            row
            for row in rows
            if row["dataset"] == dataset and row["param"] == param and row["regime"] == regime
        ),
        None,
    )


def plot_clean_metric_summary(
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    focus_regime: str,
    reference_regime: Optional[str],
    out_dir: Path,
) -> dict[str, str]:
    datasets = sorted({row["dataset"] for row in summary_rows})
    params = ["l2ru", "zak"]
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(datasets)), 4.2))
    for idx, param in enumerate(params):
        means = []
        stds = []
        ref_means = []
        for dataset in datasets:
            match = summary_row(summary_rows, dataset=dataset, param=param, regime=focus_regime)
            means.append(match[f"{metric}_mean"] if match else np.nan)
            stds.append(match[f"{metric}_std"] if match else 0.0)
            ref_match = (
                summary_row(summary_rows, dataset=dataset, param=param, regime=reference_regime)
                if reference_regime is not None
                else None
            )
            ref_means.append(ref_match[f"{metric}_mean"] if ref_match else np.nan)
        offsets = x + (idx - 0.5) * width
        ax.bar(offsets, means, width=width, color=PLOT_COLORS[param], alpha=0.88, label=param.upper())
        ax.errorbar(offsets, means, yerr=stds, fmt="none", ecolor="#111827", capsize=3, lw=1.0)
        if reference_regime is not None:
            ax.scatter(
                offsets,
                ref_means,
                s=42,
                marker="D",
                color="white",
                edgecolor=PLOT_COLORS[param],
                linewidth=1.2,
                zorder=4,
                label=f"{param.upper()} ({short_regime_label(reference_regime)})",
            )

    ax.set_xticks(x, datasets)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(
        f"Clean System-ID Performance: {METRIC_LABELS.get(metric, metric)}\n"
        f"bars = {short_regime_label(focus_regime)}, markers = {short_regime_label(reference_regime) if reference_regime else 'n/a'}"
    )
    ax.legend(loc="best")
    return save_figure(fig, out_dir / f"clean_{metric}_summary")


def plot_prediction_panels(
    run_results: list[dict[str, Any]],
    *,
    metric_key: str,
    regime: str,
    out_dir: Path,
) -> dict[str, str]:
    best_per_group: dict[tuple[str, str], dict[str, Any]] = {}
    for result in run_results:
        if result["regime"] != regime:
            continue
        key = (str(result["dataset"]), str(result["param"]))
        current = best_per_group.get(key)
        if current is None or float(result[metric_key]) < float(current[metric_key]):
            best_per_group[key] = result

    datasets = sorted({key[0] for key in best_per_group})
    fig, axes = plt.subplots(len(datasets), 1, figsize=(9.0, max(3.5, 2.8 * len(datasets))), squeeze=False)

    for row_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, 0]
        l2ru = best_per_group[(dataset, "l2ru")]
        zak = best_per_group[(dataset, "zak")]
        y_true = np.asarray(l2ru["prediction_payload"]["y_true"])
        y_l2ru = np.asarray(l2ru["prediction_payload"]["y_pred"])
        y_zak = np.asarray(zak["prediction_payload"]["y_pred"])
        init_window = int(l2ru["init_window"])

        time_axis = np.arange(y_true.shape[0])
        channel = 0
        ax.plot(time_axis, y_true[:, channel], color=PLOT_COLORS["truth"], lw=1.35, label="ground truth")
        ax.plot(time_axis, y_l2ru[:, channel], color=PLOT_COLORS["l2ru"], lw=1.1, label="l2ru")
        ax.plot(time_axis, y_zak[:, channel], color=PLOT_COLORS["zak"], lw=1.1, ls="--", label="zak")
        if init_window > 0:
            ax.axvspan(0, init_window, color="#94a3b8", alpha=0.12)
        gamma_text = (
            rf"learned $\gamma$: L2RU={l2ru['trained_global_gamma']:.2f}, ZAK={zak['trained_global_gamma']:.2f}"
            if l2ru.get("trained_global_gamma") is not None and zak.get("trained_global_gamma") is not None
            else short_regime_label(regime)
        )
        ax.set_title(f"{dataset}\n{gamma_text}")
        ax.set_ylabel("output")
        if row_idx == len(datasets) - 1:
            ax.set_xlabel("time step")
        if row_idx == 0:
            ax.legend(loc="upper right", ncol=3)

    return save_figure(fig, out_dir / "prediction_panels")


def plot_training_curves(
    run_results: list[dict[str, Any]],
    *,
    focus_regime: str,
    reference_regime: Optional[str],
    out_dir: Path,
) -> dict[str, str]:
    best_per_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in run_results:
        if result["regime"] not in {focus_regime, reference_regime}:
            continue
        key = (str(result["dataset"]), str(result["param"]), str(result["regime"]))
        current = best_per_group.get(key)
        if current is None or float(result["val_rmse"]) < float(current["val_rmse"]):
            best_per_group[key] = result

    datasets = sorted({key[0] for key in best_per_group})
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8.4, max(3.3, 2.6 * len(datasets))), squeeze=False)

    for row_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, 0]
        for param in ("l2ru", "zak"):
            for regime_name, line_style in ((focus_regime, "-"), (reference_regime, "--")):
                if regime_name is None or (dataset, param, regime_name) not in best_per_group:
                    continue
                result = best_per_group[(dataset, param, regime_name)]
                history = result["training_history"]
                epochs = np.arange(1, len(history["val_rmse"]) + 1)
                ax.plot(
                    epochs,
                    history["val_rmse"],
                    color=PLOT_COLORS[param],
                    lw=1.6,
                    ls=line_style,
                    label=f"{param.upper()} ({short_regime_label(regime_name)})",
                )
        ax.set_yscale("log")
        ax.set_title(dataset)
        ax.set_ylabel("validation RMSE")
        if row_idx == len(datasets) - 1:
            ax.set_xlabel("epoch")
        if row_idx == 0:
            ax.legend(loc="upper right")
    return save_figure(fig, out_dir / "training_curves")


def plot_runtime_scaling(
    runtime_records: list[dict[str, Any]],
    *,
    regime: str,
    out_dir: Path,
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for panel_idx, key in enumerate(("forward_sec", "backward_sec")):
        ax = axes[panel_idx]
        for param in ("l2ru", "zak"):
            for mode in ("scan", "loop"):
                subset = [
                    rec for rec in runtime_records
                    if rec["regime"] == regime and rec["param"] == param and rec["mode"] == mode and np.isfinite(rec[key])
                ]
                if not subset:
                    continue
                subset.sort(key=lambda item: item["seq_len"])
                seq = [item["seq_len"] for item in subset]
                vals = [1000.0 * item[key] for item in subset]
                ax.plot(
                    seq,
                    vals,
                    marker="o",
                    lw=1.8,
                    color=PLOT_COLORS[param],
                    ls="-" if mode == "scan" else "--",
                    label=f"{param.upper()} / {mode}",
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("milliseconds")
        ax.set_title(
            ("Forward latency" if key == "forward_sec" else "Forward + backward latency")
            + f"\n{short_regime_label(regime)}"
        )
        ax.legend(loc="best", fontsize=8)
    return save_figure(fig, out_dir / "runtime_scaling")


def plot_gain_certificate(
    gain_rows: list[dict[str, Any]],
    *,
    regime: str,
    out_dir: Path,
) -> dict[str, str]:
    gain_rows = [row for row in gain_rows if row["regime"] == regime]
    labels = [row["label"] for row in gain_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    width = 0.22
    ax.bar(x - width, [row["random_probe_p95"] for row in gain_rows], width=width, color="#60a5fa", label="random p95")
    ax.bar(x, [row["worst_case_estimate"] for row in gain_rows], width=width, color=PLOT_COLORS["gain"], label="worst-case")
    ax.bar(x + width, [row["local_test_gain_p95"] for row in gain_rows], width=width, color="#f59e0b", label="local test p95")
    ax.plot(x, [row["target_gamma"] for row in gain_rows], color=PLOT_COLORS["target"], lw=2.0, marker="D", label="target gamma")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(r"Gain estimate $\|y\|_2 / \|u\|_2$")
    ax.set_title("Empirical Gain vs Prescribed Certificate\n(normalized coordinates, fixed-gamma regime)")
    ax.legend(loc="best")

    ax = axes[1]
    ratios = [row["worst_case_over_gamma"] for row in gain_rows]
    random_ratios = [row["random_probe_max_over_gamma"] for row in gain_rows]
    local_ratios = [row["local_test_gain_max_over_gamma"] for row in gain_rows]
    ax.plot(labels, random_ratios, marker="o", lw=1.8, color="#60a5fa", label="random max / gamma")
    ax.plot(labels, ratios, marker="o", lw=1.8, color=PLOT_COLORS["gain"], label="worst / gamma")
    ax.plot(labels, local_ratios, marker="o", lw=1.8, color="#f59e0b", label="local test max / gamma")
    ax.axhline(1.0, color=PLOT_COLORS["target"], lw=1.6, ls="--", label="certificate")
    ax.set_ylabel("ratio")
    ax.set_title("Certificate Tightness\n(normalized coordinates, fixed-gamma regime)")
    ax.legend(loc="best")

    return save_figure(fig, out_dir / "gain_certificate")


def plot_disturbance_robustness(
    robustness_rows: list[dict[str, Any]],
    *,
    regime: str,
    out_dir: Path,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in robustness_rows:
        if row["regime"] != regime:
            continue
        grouped.setdefault(row["label"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["radius"])
        param = str(rows[0]["param"])
        axes[0].plot(
            [row["radius"] for row in rows],
            [row["rmse_mean"] for row in rows],
            marker="o",
            lw=1.8,
            color=PLOT_COLORS[param],
            label=label,
        )
        axes[1].plot(
            [row["radius"] for row in rows],
            [row["gain_p95"] for row in rows],
            marker="o",
            lw=1.8,
            color=PLOT_COLORS[param],
            label=label,
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("disturbance radius")
    axes[0].set_ylabel("test RMSE")
    axes[0].set_title("Prediction Error Under Input Disturbance\n(fixed-gamma regime)")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].set_xscale("log")
    axes[1].set_xlabel("disturbance radius")
    axes[1].set_ylabel("output sensitivity p95")
    axes[1].set_title("Empirical Sensitivity on Test Inputs\n(fixed-gamma regime)")
    axes[1].legend(loc="best", fontsize=8)

    return save_figure(fig, out_dir / "disturbance_robustness")


def plot_summary_dashboard(
    clean_summary: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    *,
    accuracy_regime: str,
    robustness_regime: str,
    out_dir: Path,
) -> dict[str, str]:
    accuracy_rows = [row for row in clean_summary if row["regime"] == accuracy_regime]
    gain_rows = [row for row in gain_rows if row["regime"] == robustness_regime]
    runtime_rows = [row for row in runtime_rows if row["regime"] == robustness_regime]
    datasets = sorted({row["dataset"] for row in accuracy_rows})
    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(datasets))
    width = 0.35
    for idx, param in enumerate(("l2ru", "zak")):
        rows = [row for row in accuracy_rows if row["param"] == param]
        rows_map = {row["dataset"]: row for row in rows}
        rmse = [rows_map[d]["rmse_mean"] for d in datasets]
        fit = [rows_map[d]["fit_mean"] for d in datasets]
        learned_gamma = [rows_map[d].get("trained_global_gamma_mean", np.nan) for d in datasets]
        ax1.bar(x + (idx - 0.5) * width, rmse, width=width, color=PLOT_COLORS[param], alpha=0.9, label=param.upper())
        ax2.bar(x + (idx - 0.5) * width, fit, width=width, color=PLOT_COLORS[param], alpha=0.9, label=param.upper())
        ax3.bar(x + (idx - 0.5) * width, learned_gamma, width=width, color=PLOT_COLORS[param], alpha=0.9, label=param.upper())

    ax1.set_xticks(x, datasets, rotation=20, ha="right")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Accuracy Regime: RMSE")
    ax1.legend(loc="best")

    ax2.set_xticks(x, datasets, rotation=20, ha="right")
    ax2.set_ylabel("Fit Index (%)")
    ax2.set_title("Accuracy Regime: Fit")

    ax3.set_xticks(x, datasets, rotation=20, ha="right")
    ax3.set_ylabel(r"learned global $\gamma$")
    ax3.set_title("Accuracy Regime: Learned Global Gamma")

    longest_seq = max((row["seq_len"] for row in runtime_rows), default=None)
    runtime_points = [row for row in runtime_rows if row["seq_len"] == longest_seq and row["mode"] == "loop"]
    ax4.scatter(
        [row["forward_sec"] * 1000.0 for row in runtime_points],
        [row["backward_sec"] * 1000.0 for row in runtime_points],
        s=90,
        c=[PLOT_COLORS[row["param"]] for row in runtime_points],
        alpha=0.9,
    )
    max_val = max(
        [max(row["backward_sec"], row["forward_sec"]) * 1000.0 for row in runtime_points],
        default=1.0,
    )
    ax4.plot([0.0, max_val], [0.0, max_val], color="#94a3b8", ls="--", lw=1.2)
    for row in runtime_points:
        ax4.annotate(
            f"{row['param'].upper()}",
            (row["forward_sec"] * 1000.0, row["backward_sec"] * 1000.0),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax4.set_xlabel("forward latency [ms]")
    ax4.set_ylabel("backward latency [ms]")
    ax4.set_title(
        "Fixed-Gamma Runtime Footprint"
        + (f"\nloop mode, seq_len={longest_seq}" if longest_seq is not None else "")
    )

    return save_figure(fig, out_dir / "summary_dashboard")


def plot_learned_gamma_summary(
    clean_summary: list[dict[str, Any]],
    *,
    regime: str,
    out_dir: Path,
) -> dict[str, str]:
    rows = [row for row in clean_summary if row["regime"] == regime]
    datasets = sorted({row["dataset"] for row in rows})
    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6.0, 1.8 * len(datasets)), 4.0))
    for idx, param in enumerate(("l2ru", "zak")):
        param_rows = {row["dataset"]: row for row in rows if row["param"] == param}
        vals = [param_rows[d].get("trained_global_gamma_mean", np.nan) for d in datasets]
        errs = [param_rows[d].get("trained_global_gamma_std", 0.0) for d in datasets]
        offsets = x + (idx - 0.5) * width
        ax.bar(offsets, vals, width=width, color=PLOT_COLORS[param], alpha=0.9, label=param.upper())
        ax.errorbar(offsets, vals, yerr=errs, fmt="none", ecolor="#111827", capsize=3, lw=1.0)

    ax.set_xticks(x, datasets)
    ax.set_ylabel(r"learned global $\gamma$")
    ax.set_title("Accuracy Regime: Learned Global Gamma After Training")
    ax.legend(loc="best")
    return save_figure(fig, out_dir / "learned_gamma_summary")


def plot_accuracy_robustness_tradeoff(
    clean_summary: list[dict[str, Any]],
    gain_rows: list[dict[str, Any]],
    *,
    accuracy_regime: str,
    robustness_regime: str,
    out_dir: Path,
) -> dict[str, str]:
    accuracy_rows = [row for row in clean_summary if row["regime"] == accuracy_regime]
    gain_rows = [row for row in gain_rows if row["regime"] == robustness_regime]

    fig, ax = plt.subplots(figsize=(7.3, 5.2))
    markers = ["o", "s", "^", "D", "P", "X", "v"]
    datasets = sorted({row["dataset"] for row in accuracy_rows})
    dataset_markers = {dataset: markers[idx % len(markers)] for idx, dataset in enumerate(datasets)}

    for dataset in datasets:
        for param in ("l2ru", "zak"):
            acc = summary_row(accuracy_rows, dataset=dataset, param=param, regime=accuracy_regime)
            gain = summary_row(gain_rows, dataset=dataset, param=param, regime=robustness_regime)
            if acc is None or gain is None:
                continue
            x_val = float(acc["fit_mean"])
            y_val = float(gain["worst_case_over_gamma"])
            ax.scatter(
                x_val,
                y_val,
                s=95,
                color=PLOT_COLORS[param],
                marker=dataset_markers[dataset],
                alpha=0.92,
            )
            ax.annotate(
                f"{param.upper()} | {dataset}",
                (x_val, y_val),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.axhline(1.0, color="#111827", lw=1.4, ls="--")
    ax.set_xlabel("fit index [%] with trainable global gamma")
    ax.set_ylabel(r"worst-case gain / fixed $\gamma$")
    ax.set_title("Accuracy vs Certified Robustness Trade-off")
    return save_figure(fig, out_dir / "accuracy_robustness_tradeoff")


def flatten_prediction_payload(y: torch.Tensor) -> list[list[float]]:
    if y.shape[0] == 1:
        return y[0].cpu().tolist()
    return y.reshape(-1, y.shape[-1]).cpu().tolist()


def run_single_experiment(
    *,
    dataset: DatasetBundle,
    param: str,
    regime: ExperimentRegime,
    train_branch_gamma: bool,
    parameter_match: Optional[dict[str, Any]],
    shape: ModelShape,
    train_cfg: TrainConfig,
    gain_cfg: GainConfig,
    runtime_cfg: RuntimeConfig,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    effective_train_mode = resolve_mode_for_param(param, train_cfg.train_mode)
    effective_eval_mode = resolve_mode_for_param(param, train_cfg.eval_mode)
    gamma = float(regime.gamma)

    model = build_model(
        param=param,
        gamma=gamma,
        train_branch_gamma=train_branch_gamma,
        train_global_gamma=bool(regime.train_global_gamma),
        d_input=dataset.d_input,
        d_output=dataset.d_output,
        shape=shape,
        device=device,
    )

    train_info, normalizer = train_model(
        model=model,
        dataset=dataset,
        train_cfg=replace(train_cfg, train_mode=effective_train_mode, eval_mode=effective_eval_mode),
        device=device,
        seed=seed,
        progress_label=(
            f"{dataset.pretty_name} | {param.upper()} | "
            f"{'gamma~trainable' if regime.train_global_gamma else f'gamma={gamma:g}'} | seed={seed}"
        ),
    )

    y_test_pred = predict(model, dataset.u_test.to(device), normalizer=normalizer, mode=effective_eval_mode)
    clean_metrics = compute_metrics(dataset.y_test, y_test_pred, dataset.init_window)
    trained_global_gamma = (
        float(model.gamma_t.detach().abs().cpu())
        if regime.train_global_gamma and hasattr(model, "gamma_t")
        else None
    )
    reported_gamma = trained_global_gamma if regime.train_global_gamma else gamma
    consistency = (
        compare_scan_loop_consistency(model, param=param, dataset=dataset, normalizer=normalizer)
        if regime.compute_consistency
        else {"available": False, "skipped": True, "reason": f"skipped for regime={regime.name}"}
    )
    certificate = summarize_certificate(model)
    if regime.compute_gain:
        probe_values = random_probe_gains(
            model,
            seq_len=min(train_cfg.window_len, dataset.u_test.shape[1]),
            radius=gain_cfg.probe_radius,
            n_samples=gain_cfg.probe_samples,
            batch_size=gain_cfg.probe_batch_size,
            device=device,
            mode=effective_eval_mode,
        )
        worst_case = optimize_finite_horizon_gain(
            model,
            seq_len=min(train_cfg.window_len, dataset.u_test.shape[1]),
            radius=gain_cfg.probe_radius,
            restarts=gain_cfg.worst_case_restarts,
            steps=gain_cfg.worst_case_steps,
            lr=gain_cfg.worst_case_lr,
            batch_size=gain_cfg.worst_case_batch,
            device=device,
            mode=effective_eval_mode,
        )
        base_inputs = sample_test_windows(
            dataset.u_test,
            n_windows=gain_cfg.local_gain_windows,
            window_len=min(train_cfg.window_len, dataset.u_test.shape[1]),
            seed=seed,
        )
        local_gain = estimate_local_test_gain(
            model,
            normalizer=normalizer,
            base_inputs=base_inputs,
            eps=gain_cfg.local_gain_eps,
            steps=gain_cfg.local_gain_steps,
            restarts=gain_cfg.local_gain_restarts,
            lr=gain_cfg.local_gain_lr,
            device=device,
            mode=effective_eval_mode,
        )
        target_gamma = float(certificate.get("target_gamma", reported_gamma if reported_gamma is not None else gamma))
        raw_target_upper = raw_gain_upper_bound(certificate=certificate, normalizer=normalizer)
        gain_estimates = {
            "coordinate_system": "normalized_model_coordinates",
            "random_probe_mean": float(np.mean(probe_values)),
            "random_probe_p95": float(np.percentile(probe_values, 95)),
            "random_probe_max": float(np.max(probe_values)),
            "worst_case": worst_case,
            "local_test": local_gain,
            "target_gamma": float(target_gamma),
            "raw_target_gamma_upper_bound": raw_target_upper,
            "worst_case_over_gamma": float(worst_case["estimate"] / max(target_gamma, 1e-12)),
            "random_probe_max_over_gamma": float(np.max(probe_values) / max(target_gamma, 1e-12)),
            "local_test_max_over_gamma": float(local_gain["max"] / max(target_gamma, 1e-12)),
        }
    else:
        gain_estimates = None

    if regime.compute_robustness:
        disturbance = evaluate_under_disturbance(
            model,
            dataset=dataset,
            normalizer=normalizer,
            radii=gain_cfg.disturbance_radii,
            trials=gain_cfg.disturbance_trials,
            seed=seed,
            device=device,
            mode=effective_eval_mode,
        )
    else:
        disturbance = []

    if regime.compute_runtime:
        runtime_scan = (
            safe_runtime_benchmark(model, runtime_cfg=runtime_cfg, device=device, mode="scan")
            if "scan" in supported_modes_for_param(param)
            else {"available": False, "error": f"{param} does not support scan", "points": []}
        )
        runtime_loop = safe_runtime_benchmark(model, runtime_cfg=runtime_cfg, device=device, mode="loop")
    else:
        runtime_scan = {"available": False, "skipped": True, "error": f"skipped for regime={regime.name}", "points": []}
        runtime_loop = {"available": False, "skipped": True, "error": f"skipped for regime={regime.name}", "points": []}

    result = {
        "dataset": dataset.pretty_name,
        "dataset_key": dataset.name,
        "param": param,
        "regime": regime.name,
        "regime_pretty": regime.pretty_name,
        "gamma": None if reported_gamma is None else float(reported_gamma),
        "gamma_init": float(gamma),
        "gamma_fixed": None if regime.train_global_gamma else float(gamma),
        "train_global_gamma": bool(regime.train_global_gamma),
        "trained_global_gamma": None if trained_global_gamma is None else float(trained_global_gamma),
        "train_branch_gamma": bool(train_branch_gamma),
        "seed": int(seed),
        "shape": asdict(shape),
        "parameter_match": parameter_match,
        "init_window": int(dataset.init_window),
        "parameter_count": int(count_parameters(model)),
        "train_cfg": asdict(train_cfg),
        "effective_modes": {
            "train_mode_requested": train_cfg.train_mode,
            "eval_mode_requested": train_cfg.eval_mode,
            "train_mode_used": effective_train_mode,
            "eval_mode_used": effective_eval_mode,
            "supported_modes": list(supported_modes_for_param(param)),
        },
        "training_history": train_info["history"],
        "steps_per_epoch": int(train_info["steps_per_epoch"]),
        "train_sequence_count": int(train_info["train_sequence_count"]),
        "train_sequence_length": int(train_info["train_sequence_length"]),
        "best_epoch": int(train_info["best_epoch"]),
        "best_val_rmse": float(train_info["best_val_rmse"]),
        "normalizer": None if normalizer is None else normalizer.to_dict(),
        "clean_metrics": clean_metrics,
        "scan_loop_consistency": consistency,
        "certificate": certificate,
        "gain_estimates": gain_estimates,
        "disturbance_robustness": disturbance,
        "runtime": {
            "scan": runtime_scan,
            "loop": runtime_loop,
        },
        "prediction_payload": {
            "y_true": flatten_prediction_payload(dataset.y_test),
            "y_pred": flatten_prediction_payload(y_test_pred),
        },
    }

    run_dir = out_dir / dataset.name / regime.name / f"{param}_ginit{str(gamma).replace('.', 'p')}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "result.json", result)

    if consistency.get("available"):
        try:
            y_scan = predict(model, dataset.u_test.to(device), normalizer=normalizer, mode="scan")
            y_loop = predict(model, dataset.u_test.to(device), normalizer=normalizer, mode="loop")
            comparison = {
                "scan_pred": flatten_prediction_payload(y_scan),
                "loop_pred": flatten_prediction_payload(y_loop),
            }
            save_json(run_dir / "scan_loop_predictions.json", comparison)
        except Exception:
            pass

    return result


def results_to_flat_tables(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clean_rows = []
    gain_rows = []
    runtime_rows = []

    for result in results:
        clean_metrics = result["clean_metrics"]
        clean_rows.append(
            {
                "dataset": result["dataset"],
                "param": result["param"],
                "regime": result["regime"],
                "regime_pretty": result["regime_pretty"],
                "gamma": result["gamma"],
                "gamma_fixed": result["gamma_fixed"],
                "gamma_init": result["gamma_init"],
                "trained_global_gamma": result["trained_global_gamma"],
                "train_global_gamma": result["train_global_gamma"],
                "seed": result["seed"],
                "rmse": clean_metrics["rmse"],
                "nrmse": clean_metrics["nrmse"],
                "mae": clean_metrics["mae"],
                "r2": clean_metrics["r2"],
                "fit": clean_metrics["fit"],
                "best_val_rmse": result["best_val_rmse"],
                "parameter_count": result["parameter_count"],
            }
        )

        gains = result["gain_estimates"]
        if gains is not None:
            gain_rows.append(
                {
                    "dataset": result["dataset"],
                    "param": result["param"],
                    "regime": result["regime"],
                    "gamma": result["gamma"],
                    "gamma_fixed": result["gamma_fixed"],
                    "seed": result["seed"],
                    "label": f"{result['param'].upper()} | {result['dataset']}",
                    "target_gamma": gains["target_gamma"],
                    "random_probe_mean": gains["random_probe_mean"],
                    "random_probe_p95": gains["random_probe_p95"],
                    "random_probe_max": gains["random_probe_max"],
                    "worst_case_estimate": gains["worst_case"]["estimate"],
                    "local_test_gain_mean": gains["local_test"]["mean"],
                    "local_test_gain_p95": gains["local_test"]["p95"],
                    "local_test_gain_max": gains["local_test"]["max"],
                    "worst_case_over_gamma": gains["worst_case_over_gamma"],
                    "random_probe_max_over_gamma": gains["random_probe_max_over_gamma"],
                    "local_test_gain_max_over_gamma": gains["local_test_max_over_gamma"],
                }
            )

        for mode in ("scan", "loop"):
            mode_runtime = result["runtime"][mode]
            if not mode_runtime["available"]:
                continue
            for point in mode_runtime["points"]:
                runtime_rows.append(
                    {
                        "dataset": result["dataset"],
                        "param": result["param"],
                        "regime": result["regime"],
                        "gamma": result["gamma"],
                        "gamma_fixed": result["gamma_fixed"],
                        "seed": result["seed"],
                        "mode": point["mode"],
                        "seq_len": point["seq_len"],
                        "forward_sec": point["forward_sec"],
                        "backward_sec": point["backward_sec"],
                    }
                )

    return clean_rows, gain_rows, runtime_rows


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare l2ru vs zak with trainable-gamma accuracy runs and fixed-gamma robustness/runtime runs."
    )
    parser.add_argument("--datasets", type=str, default="Cascaded_Tanks",
                        help="Comma-separated nonlinear_benchmarks dataset names, e.g. Cascaded_Tanks,WienerHammer,Silverbox.")
    parser.add_argument("--benchmark-fallback", type=str, default="WienerHammer",
                        help="Fallback nonlinear benchmark if a requested one is unavailable.")

    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "l2ru_vs_zak")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--robustness-gamma", type=float, default=2.0,
                        help="Fixed global gamma used for certificate, robustness, and runtime comparisons.")
    parser.add_argument("--accuracy-gamma-init", type=float, default=1.0,
                        help="Initial global gamma for the accuracy-only regime, where gamma_t is trainable.")
    parser.add_argument("--gammas", type=str, default="2.0",
                        help="Backward-compatible alias for the fixed-gamma robustness regime. Only the first value is used.")
    parser.add_argument("--show-plots", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--ff", type=str, default="LGLU2")
    parser.add_argument("--d-hidden", type=int, default=12)
    parser.add_argument("--nl-layers", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--init", type=str, default="eye")
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--rmin", type=float, default=0.7)
    parser.add_argument("--rmax", type=float, default=0.98)
    parser.add_argument("--max-phase", type=float, default=2 * math.pi)
    parser.add_argument("--max-phase-b", type=float, default=2 * math.pi)
    parser.add_argument("--phase-center", type=float, default=0.0)
    parser.add_argument("--random-phase", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learn-x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zak-d-margin", type=float, default=0.25,
                        help="ZAK-only: initialize the direct term strictly inside the feasible set.")
    parser.add_argument("--zak-x2-margin", type=float, default=0.9,
                        help="ZAK-only: initialize the off-diagonal Gershgorin block strictly inside the feasible set.")
    parser.add_argument("--zak-x2-init-scale", type=float, default=0.1,
                        help="ZAK-only: scale of the free real X2 initialization.")

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--window-len", type=int, default=256)
    parser.add_argument("--train-mode", type=str, choices=["scan", "loop"], default="scan")
    parser.add_argument("--eval-mode", type=str, choices=["scan", "loop"], default="scan")
    parser.add_argument("--normalize-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-early-stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-branch-gamma", action=argparse.BooleanOptionalAction, default=True,
                        help="Train the internal LTI/block gamma parameters while keeping the overall DeepSSM target gamma fixed via certificate scaling.")

    parser.add_argument("--probe-samples", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-radius", type=float, default=1.0)
    parser.add_argument("--worst-case-restarts", type=int, default=8)
    parser.add_argument("--worst-case-steps", type=int, default=80)
    parser.add_argument("--worst-case-batch", type=int, default=4)
    parser.add_argument("--worst-case-lr", type=float, default=8e-2)
    parser.add_argument("--local-gain-windows", type=int, default=8)
    parser.add_argument("--local-gain-eps", type=float, default=0.25)
    parser.add_argument("--local-gain-steps", type=int, default=60)
    parser.add_argument("--local-gain-restarts", type=int, default=4)
    parser.add_argument("--local-gain-lr", type=float, default=6e-2)
    parser.add_argument("--disturbance-radii", type=str, default="0.05,0.1,0.25,0.5,1.0")
    parser.add_argument("--disturbance-trials", type=int, default=4)

    parser.add_argument("--runtime-batch-size", type=int, default=8)
    parser.add_argument("--runtime-seq-lens", type=str, default="64,128,256,512,1024")
    parser.add_argument("--runtime-repeats-forward", type=int, default=20)
    parser.add_argument("--runtime-repeats-backward", type=int, default=10)
    parser.add_argument("--runtime-warmup", type=int, default=3)

    parser.add_argument("--auto-match-zak-state", action=argparse.BooleanOptionalAction, default=True,
                        help="Adjust zak d_state to match the l2ru parameter count more closely. Enabled by default because l2ru uses d_model as its effective state size, while zak uses d_state.")
    parser.add_argument("--match-search-min", type=int, default=2)
    parser.add_argument("--match-search-max", type=int, default=128)

    return parser


def parse_int_list(text: str) -> list[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    set_plot_style()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    datasets = [chunk.strip() for chunk in args.datasets.split(",") if chunk.strip()]
    seeds = parse_int_list(args.seeds)
    legacy_gammas = parse_float_list(args.gammas)
    robustness_gamma = float(args.robustness_gamma)
    if legacy_gammas:
        robustness_gamma = float(legacy_gammas[0])
        if len(legacy_gammas) > 1:
            print(
                "warning: multiple values passed to --gammas. "
                f"Only the first one ({robustness_gamma:g}) is used in the fixed-gamma regime."
            )

    regimes = [
        ExperimentRegime(
            name="accuracy_trainable_gamma",
            pretty_name=regime_label("accuracy_trainable_gamma"),
            gamma=float(args.accuracy_gamma_init),
            train_global_gamma=True,
            compute_gain=False,
            compute_runtime=False,
            compute_robustness=False,
            compute_consistency=False,
        ),
        ExperimentRegime(
            name="robustness_fixed_gamma",
            pretty_name=regime_label("robustness_fixed_gamma", robustness_gamma),
            gamma=float(robustness_gamma),
            train_global_gamma=False,
            compute_gain=True,
            compute_runtime=True,
            compute_robustness=True,
            compute_consistency=True,
        ),
    ]

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        min_delta=args.min_delta,
        val_interval=args.val_interval,
        window_len=args.window_len,
        normalize_data=args.normalize_data,
        train_mode=args.train_mode,
        eval_mode=args.eval_mode,
        use_early_stopping=args.use_early_stopping,
    )
    model_shape = ModelShape(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        ff=args.ff,
        d_hidden=args.d_hidden,
        nl_layers=args.nl_layers,
        scale=args.scale,
        init=args.init,
        rho=args.rho,
        rmin=args.rmin,
        rmax=args.rmax,
        max_phase=args.max_phase,
        max_phase_b=args.max_phase_b,
        phase_center=args.phase_center,
        random_phase=args.random_phase,
        learn_x0=args.learn_x0,
        zak_d_margin=args.zak_d_margin,
        zak_x2_margin=args.zak_x2_margin,
        zak_x2_init_scale=args.zak_x2_init_scale,
    )
    gain_cfg = GainConfig(
        probe_samples=args.probe_samples,
        probe_batch_size=args.probe_batch_size,
        probe_radius=args.probe_radius,
        worst_case_restarts=args.worst_case_restarts,
        worst_case_steps=args.worst_case_steps,
        worst_case_batch=args.worst_case_batch,
        worst_case_lr=args.worst_case_lr,
        local_gain_windows=args.local_gain_windows,
        local_gain_eps=args.local_gain_eps,
        local_gain_steps=args.local_gain_steps,
        local_gain_restarts=args.local_gain_restarts,
        local_gain_lr=args.local_gain_lr,
        disturbance_radii=tuple(parse_float_list(args.disturbance_radii)),
        disturbance_trials=args.disturbance_trials,
    )
    runtime_cfg = RuntimeConfig(
        batch_size=args.runtime_batch_size,
        seq_lens=tuple(parse_int_list(args.runtime_seq_lens)),
        repeats_forward=args.runtime_repeats_forward,
        repeats_backward=args.runtime_repeats_backward,
        warmup=args.runtime_warmup,
    )

    experiment_manifest = {
        "datasets": datasets,
        "seeds": seeds,
        "robustness_gamma": robustness_gamma,
        "accuracy_gamma_init": float(args.accuracy_gamma_init),
        "device": args.device,
        "train_cfg": asdict(train_cfg),
        "model_shape": asdict(model_shape),
        "gain_cfg": asdict(gain_cfg),
        "runtime_cfg": asdict(runtime_cfg),
        "auto_match_zak_state": bool(args.auto_match_zak_state),
        "train_branch_gamma": bool(args.train_branch_gamma),
        "regimes": [asdict(regime) for regime in regimes],
    }
    save_json(out_dir / "manifest.json", experiment_manifest)

    all_results = []
    for dataset_name in datasets:
        dataset = load_dataset(
            dataset_name,
            benchmark_fallback=args.benchmark_fallback,
        )
        print(f"\n=== Dataset: {dataset.pretty_name} ===")
        print(dataset.description)
        print(
            f"train={tuple(dataset.u_train.shape)}  val={tuple(dataset.u_val.shape)}  "
            f"test={tuple(dataset.u_test.shape)}  init_window={dataset.init_window}"
        )
        dataset_steps_per_epoch = max(
            (int(dataset.u_train.shape[0]) + max(train_cfg.batch_size, 1) - 1) // max(train_cfg.batch_size, 1),
            1,
        )
        print(
            f"training uses full trajectories (no slicing): "
            f"{int(dataset.u_train.shape[0])} train sequence(s), "
            f"seq_len={int(dataset.u_train.shape[1])}, "
            f"batch_size={train_cfg.batch_size}, "
            f"steps_per_epoch={dataset_steps_per_epoch}"
        )
        if (not args.auto_match_zak_state) and (model_shape.d_state != model_shape.d_model):
            print(
                "note: L2RU uses d_model as its effective state size, while ZAK uses d_state. "
                f"With d_model={model_shape.d_model} and d_state={model_shape.d_state}, this is not an order-matched comparison. "
                "Set --auto-match-zak-state or choose d_state=d_model for a fairer comparison."
            )

        for regime in regimes:
            zak_shape, param_match = maybe_match_param_budget(
                target_param="zak",
                reference_param="l2ru",
                gamma=regime.gamma,
                train_branch_gamma=bool(args.train_branch_gamma),
                train_global_gamma=bool(regime.train_global_gamma),
                d_input=dataset.d_input,
                d_output=dataset.d_output,
                shape=model_shape,
                device=device,
                enabled=bool(args.auto_match_zak_state),
                search_min=args.match_search_min,
                search_max=args.match_search_max,
            )
            print(
                f"  [{regime.pretty_name}] parameter match | "
                f"L2RU params={param_match['reference_parameter_count']} | "
                f"ZAK params={param_match['target_parameter_count']} | "
                f"gap={param_match['absolute_gap']} ({100.0 * param_match['relative_gap']:.2f}%) | "
                f"zak d_state={param_match['target_shape']['d_state']}"
            )
            if float(param_match["relative_gap"]) > 0.05:
                print(
                    "  warning: parameter gap is above 5%. "
                    "Consider widening --match-search-max or adjusting d_model/d_hidden for a tighter match."
                )

            for seed in seeds:
                for param, shape in (("l2ru", model_shape), ("zak", zak_shape)):
                    train_mode_used = resolve_mode_for_param(param, train_cfg.train_mode)
                    eval_mode_used = resolve_mode_for_param(param, train_cfg.eval_mode)
                    print(
                        f"  -> training {param.upper()} | {regime.pretty_name} | seed={seed} | "
                        f"train_mode={train_mode_used} | eval_mode={eval_mode_used}"
                    )
                    result = run_single_experiment(
                        dataset=dataset,
                        param=param,
                        regime=regime,
                        train_branch_gamma=bool(args.train_branch_gamma),
                        parameter_match=param_match,
                        shape=shape,
                        train_cfg=train_cfg,
                        gain_cfg=gain_cfg,
                        runtime_cfg=runtime_cfg,
                        seed=seed,
                        device=device,
                        out_dir=out_dir,
                    )
                    all_results.append(result)
                    print(
                        f"     test RMSE={result['clean_metrics']['rmse']:.4e} | "
                        f"fit={result['clean_metrics']['fit']:.2f}% | "
                        f"params={result['parameter_count']} | "
                        f"gamma={result['gamma'] if result['gamma'] is not None else 'n/a'}"
                    )

    clean_rows, gain_rows, runtime_rows = results_to_flat_tables(all_results)
    clean_summary = aggregate_group(
        clean_rows,
        metrics=["rmse", "nrmse", "mae", "r2", "fit", "best_val_rmse", "parameter_count"],
    )
    gain_summary = aggregate_gain_rows(gain_rows)
    runtime_summary = aggregate_runtime_rows(runtime_rows)

    figures_dir = out_dir / "figures"
    figure_index = {
        "clean_rmse_summary": plot_clean_metric_summary(
            clean_summary,
            metric="rmse",
            focus_regime="accuracy_trainable_gamma",
            reference_regime="robustness_fixed_gamma",
            out_dir=figures_dir,
        ),
        "clean_fit_summary": plot_clean_metric_summary(
            clean_summary,
            metric="fit",
            focus_regime="accuracy_trainable_gamma",
            reference_regime="robustness_fixed_gamma",
            out_dir=figures_dir,
        ),
        "prediction_panels": plot_prediction_panels(
            all_results,
            metric_key="best_val_rmse",
            regime="accuracy_trainable_gamma",
            out_dir=figures_dir,
        ),
        "training_curves": plot_training_curves(
            all_results,
            focus_regime="accuracy_trainable_gamma",
            reference_regime="robustness_fixed_gamma",
            out_dir=figures_dir,
        ),
        "runtime_scaling": plot_runtime_scaling(runtime_summary, regime="robustness_fixed_gamma", out_dir=figures_dir),
        "gain_certificate": plot_gain_certificate(gain_summary, regime="robustness_fixed_gamma", out_dir=figures_dir),
        "learned_gamma_summary": plot_learned_gamma_summary(
            clean_summary,
            regime="accuracy_trainable_gamma",
            out_dir=figures_dir,
        ),
        "accuracy_robustness_tradeoff": plot_accuracy_robustness_tradeoff(
            clean_summary,
            gain_summary,
            accuracy_regime="accuracy_trainable_gamma",
            robustness_regime="robustness_fixed_gamma",
            out_dir=figures_dir,
        ),
        "summary_dashboard": plot_summary_dashboard(
            clean_summary,
            gain_summary,
            runtime_summary,
            accuracy_regime="accuracy_trainable_gamma",
            robustness_regime="robustness_fixed_gamma",
            out_dir=figures_dir,
        ),
    }

    robustness_rows = []
    for result in all_results:
        for point in result["disturbance_robustness"]:
            robustness_rows.append(
                {
                    "dataset": result["dataset"],
                    "param": result["param"],
                    "regime": result["regime"],
                    "gamma": result["gamma"],
                    "gamma_fixed": result["gamma_fixed"],
                    **point,
                }
            )
    robustness_summary = aggregate_robustness_rows(robustness_rows)
    figure_index["disturbance_robustness"] = plot_disturbance_robustness(
        robustness_summary,
        regime="robustness_fixed_gamma",
        out_dir=figures_dir,
    )

    save_json(out_dir / "all_results.json", {"runs": all_results, "figures": figure_index})
    save_csv(out_dir / "clean_metrics.csv", clean_rows)
    save_csv(out_dir / "clean_summary.csv", clean_summary)
    save_csv(out_dir / "gain_metrics.csv", gain_rows)
    save_csv(out_dir / "gain_summary.csv", gain_summary)
    save_csv(out_dir / "runtime_metrics.csv", runtime_rows)
    save_csv(out_dir / "runtime_summary.csv", runtime_summary)
    save_csv(out_dir / "disturbance_metrics.csv", robustness_rows)
    save_csv(out_dir / "disturbance_summary.csv", robustness_summary)

    print("\nSaved comparison artifacts:")
    print(f"  results: {out_dir / 'all_results.json'}")
    print(f"  clean summary: {out_dir / 'clean_summary.csv'}")
    for name, paths in figure_index.items():
        print(f"  {name}: {paths['png']}")

    if args.show_plots:
        image_paths = [Path(paths["png"]) for paths in figure_index.values()]
        for path in image_paths:
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.set_title(path.stem.replace("_", " "))
            ax.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
