#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GammaBench-DR (Delayed Retrieval) benchmark for SSMs with tunable l2-gain (gamma).

Models follow:
  core: (B,T,d_input) -> (B,T,d_output)

instantiated as:
  model = DeepSSM(
      d_input=1, d_output=1, d_model=10, d_state=8, n_layers=4,
      param='lru', ff='LGLU', gamma=4, max_phase_b=None
  ).to(device)

This script:
  - Generates synthetic sequences with a faint long-memory cue + bounded-energy disturbance.
  - Trains/evaluates:
      * baseline(s): param='lru' (gamma ignored)
      * L2RU sweep:  param='l2n'  with gamma in a list
  - Produces:
      * results.jsonl
      * heatmaps (clean + noisy + degradation + gain-sensitivity) over (T x eps)
      * combined degradation comparison bar chart
      * Pareto plot (mean clean vs mean noisy) with model-type colour/marker coding
      * summary table printed to console

Run example:
  python Robust_Benchmark.py --out_dir runs --grid_T 128,256,512 \
      --grid_eps 0,0.25,0.5,1.0 --baselines lru --l2ru_param_name l2n \
      --gammas 0.5,1,2,4 --disturbance burst --burst_frac 0.08
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.neural_ssm.ssm import DeepSSM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm as _tqdm
    HAVE_TQDM = True
except ImportError:
    _tqdm = None   # type: ignore[assignment]
    HAVE_TQDM = False


def _trange(n: int, **kwargs):
    if HAVE_TQDM and _tqdm is not None:
        return _tqdm(range(n), **kwargs)
    return range(n)


# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Disturbance generators
# ----------------------------

def sample_energy_bounded_noise(shape: Tuple[int, ...], eps: float, device: torch.device) -> torch.Tensor:
    """Gaussian noise rescaled per-sample so ||d||_2 = eps across time and channels."""
    if eps <= 0:
        return torch.zeros(shape, device=device)
    d = torch.randn(shape, device=device)
    B = d.shape[0]
    flat = d.view(B, -1)
    nrm = flat.norm(p=2, dim=1).clamp_min(1e-12)
    flat = flat * (eps / nrm).unsqueeze(1)
    return flat.view_as(d)


def sample_burst_energy_noise(
    shape: Tuple[int, ...],
    eps: float,
    burst_frac: float,
    device: torch.device,
) -> torch.Tensor:
    """Concentrate all energy into one contiguous burst window per sample."""
    if eps <= 0:
        return torch.zeros(shape, device=device)

    B, T = shape[0], shape[1]
    D = int(torch.tensor(shape[2:]).prod().item()) if len(shape) > 2 else 1

    d = torch.zeros((B, T, D), device=device)
    burst_len = max(1, int(round(T * burst_frac)))

    for b in range(B):
        start = int(torch.randint(0, T - burst_len + 1, (1,)).item())
        burst = torch.randn((burst_len, D), device=device)
        flat = burst.reshape(-1)
        nrm = flat.norm(p=2).clamp_min(1e-12)
        burst = burst * (eps / nrm)
        d[b, start:start + burst_len, :] = burst

    return d.view(shape)


# ----------------------------
# Synthetic dataset
# ----------------------------

@dataclass
class GammaBenchConfig:
    T: int = 256
    cue_len: int = 8
    rho: float = 0.99
    b: float = 0.5
    process_noise_std: float = 0.01
    meas_noise_std: float = 0.01

    eps: float = 0.5
    disturbance: str = "dense"  # {"dense","burst","none"}
    burst_frac: float = 0.1

    task: str = "classif"  # {"classif","regress"}
    regress_range: float = 1.0
    normalize_input: bool = True


class GammaBenchDataset(Dataset):
    """
    Pre-materialises all N sequences at construction time.

    The linear recurrence has a sequential dependency over T, so we loop over T
    once but vectorise each step over all N samples.  This replaces N×T Python
    iterations (old per-sample generator loop) with T vectorised tensor ops of
    size N — typically 50–100× faster for large N or T.

    Disturbance d is still added in collate_fn (per-batch, on device).
    """
    def __init__(self, n: int, cfg: GammaBenchConfig, seed: int):
        T, cue_len = cfg.T, cfg.cue_len
        rng = torch.Generator()
        rng.manual_seed(seed)

        if cfg.task == "classif":
            s = torch.randint(0, 2, (n,), generator=rng).float() * 2.0 - 1.0
        else:
            s = (2.0 * torch.rand((n,), generator=rng) - 1.0) * cfg.regress_range

        w   = cfg.process_noise_std * torch.randn((n, T), generator=rng)
        eta = cfg.meas_noise_std    * torch.randn((n, T), generator=rng)

        # Input cue: non-zero only for t < cue_len.
        u = torch.zeros(n, T)
        if cue_len > 0:
            u[:, :cue_len] = cfg.b * s.unsqueeze(1)

        # Linear recurrence — T sequential steps, each vectorised over N.
        z = torch.zeros(n, T)
        for t in range(T - 1):
            z[:, t + 1] = cfg.rho * z[:, t] + u[:, t] + w[:, t]

        x = (z + eta).unsqueeze(-1)  # (N, T, 1)
        if cfg.normalize_input:
            mu  = x.mean(dim=1, keepdim=True)
            sig = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x   = (x - mu) / sig

        self.x = x                   # (N, T, 1)  float32
        self.y = s.unsqueeze(-1)     # (N, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: GammaBenchConfig,
    device: torch.device,
    add_disturbance: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function — cfg and device are bound via functools.partial to avoid closure bugs."""
    xs, ys = zip(*batch)
    X = torch.stack(xs, dim=0).to(device)   # (B,T,1)
    Y = torch.stack(ys, dim=0).to(device)   # (B,1)

    if add_disturbance and cfg.eps > 0 and cfg.disturbance != "none":
        if cfg.disturbance == "dense":
            d = sample_energy_bounded_noise(X.shape, cfg.eps, device)
        elif cfg.disturbance == "burst":
            d = sample_burst_energy_noise(X.shape, cfg.eps, cfg.burst_frac, device)
        else:
            raise ValueError(f"Unknown disturbance={cfg.disturbance}")
        X = X + d

    return X, Y


def make_loader(
    ds: Dataset,
    cfg: GammaBenchConfig,
    device: torch.device,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    add_disturbance: bool,
) -> DataLoader:
    """Build a DataLoader with a safely-bound collate_fn (no closure over loop variable)."""
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=partial(_collate, cfg=cfg, device=device, add_disturbance=add_disturbance),
    )


# ----------------------------
# Model wrapper: take last time-step output
# ----------------------------

class LastStepWrapper(nn.Module):
    """Wrap a core SSM that returns (B,T,d_output) and return last step (B,d_output)."""
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_seq = self.core(x)
        if isinstance(y_seq, (tuple, list)):
            y_seq = y_seq[0]
        return y_seq[:, -1, :]   # (B,d_output)


def build_model(
    param: str,
    gamma: Optional[float],
    d_input: int,
    d_output: int,
    d_model: int,
    d_state: int,
    n_layers: int,
    device: torch.device,
    max_phase_b: Optional[float] = None,
    train_gamma: bool = False,
    ff_scale: float = 1.0,
) -> nn.Module:
    core = DeepSSM(
        d_input=d_input,
        d_output=d_output,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        param=param,
        ff='LGLU',
        gamma=gamma,
        max_phase_b=max_phase_b,
        train_gamma=train_gamma,
        scale=ff_scale,
    ).to(device)
    return LastStepWrapper(core).to(device)


# ----------------------------
# Train / Eval
# ----------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate model. Pass seed for deterministic disturbance sampling."""
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model.eval()
    total = 0
    correct = 0
    sum_loss = 0.0

    for X, Y in loader:
        pred = model(X)   # (B,1)

        if task == "classif":
            y01 = (Y > 0).float()
            loss = F.binary_cross_entropy_with_logits(pred, y01)
            yhat = (torch.sigmoid(pred) > 0.5).float()
            correct += (yhat == y01).sum().item()
            total += Y.numel()
            sum_loss += loss.item() * Y.shape[0]
        else:
            loss = F.mse_loss(pred, Y)
            total += Y.shape[0]
            sum_loss += loss.item() * Y.shape[0]

    if task == "classif":
        return {"acc": correct / max(1, total), "loss": sum_loss / max(1, total)}
    else:
        return {"mse": sum_loss / max(1, total), "loss": sum_loss / max(1, total)}


@torch.no_grad()
def evaluate_disturbance_gain(
    model: nn.Module,
    loader_clean: DataLoader,
    cfg: GammaBenchConfig,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate output sensitivity to bounded-energy disturbances:
        gain = ||y(x + d) - y(x)||_2 / ||d||_2
    aggregated over the test set.
    """
    if cfg.eps <= 0 or cfg.disturbance == "none":
        return {"gain_mean": 0.0, "gain_p95": 0.0, "gain_max": 0.0}

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    model.eval()
    ratios = []

    for X, _ in loader_clean:
        if cfg.disturbance == "dense":
            d = sample_energy_bounded_noise(X.shape, cfg.eps, X.device)
        elif cfg.disturbance == "burst":
            d = sample_burst_energy_noise(X.shape, cfg.eps, cfg.burst_frac, X.device)
        else:
            raise ValueError(f"Unknown disturbance={cfg.disturbance}")

        y0 = model(X)
        y1 = model(X + d)

        dy = (y1 - y0).reshape(y0.shape[0], -1)
        dn = d.reshape(d.shape[0], -1).norm(p=2, dim=1).clamp_min(1e-12)
        g = dy.norm(p=2, dim=1) / dn
        ratios.append(g.detach().cpu())

    if not ratios:
        return {"gain_mean": 0.0, "gain_p95": 0.0, "gain_max": 0.0}

    g_all = torch.cat(ratios, dim=0)
    return {
        "gain_mean": float(g_all.mean().item()),
        "gain_p95": float(torch.quantile(g_all, 0.95).item()),
        "gain_max": float(g_all.max().item()),
    }

def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader_clean: DataLoader,
    val_loader_noisy: DataLoader,
    task: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    grad_clip: float,
    val_noisy_weight: float,
    val_seed: int,
    val_interval: int = 5,
) -> Dict[str, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    best_metric = -1e18
    best_state = None

    for ep in _trange(epochs, desc="  epochs", leave=False):
        model.train()
        for X, Y in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(X)

            if task == "classif":
                y01 = (Y > 0).float()
                loss = F.binary_cross_entropy_with_logits(pred, y01)
            else:
                loss = F.mse_loss(pred, Y)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        scheduler.step()

        if (ep + 1) % val_interval == 0 or ep == epochs - 1:
            vc = evaluate(model, val_loader_clean, task, seed=val_seed)
            vn = evaluate(model, val_loader_noisy, task, seed=val_seed + 1)

            if task == "classif":
                metric = val_noisy_weight * vn["acc"] + (1.0 - val_noisy_weight) * vc["acc"]
            else:
                metric = -(val_noisy_weight * vn["mse"] + (1.0 - val_noisy_weight) * vc["mse"])

            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    vc = evaluate(model, val_loader_clean, task, seed=val_seed)
    vn = evaluate(model, val_loader_noisy, task, seed=val_seed + 1)
    out = {"val_clean_" + k: v for k, v in vc.items()}
    out.update({"val_noisy_" + k: v for k, v in vn.items()})
    return out


# ----------------------------
# Plotting helpers
# ----------------------------

def save_heatmap(
    grid_T: List[int],
    grid_eps: List[float],
    values,   # (len(T), len(eps)) tensor or ndarray
    title: str,
    outpath: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    if hasattr(values, "cpu"):
        values = values.cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(4.0, len(grid_eps) * 1.4), max(3.0, len(grid_T) * 1.0)))
    im = ax.imshow(values, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(grid_eps)))
    ax.set_xticklabels([str(e) for e in grid_eps], rotation=45)
    ax.set_yticks(range(len(grid_T)))
    ax.set_yticklabels([str(t) for t in grid_T])
    ax.set_xlabel("eps (disturbance energy)")
    ax.set_ylabel("T (horizon)")
    ax.set_title(title)

    # Annotate each cell with its value
    for i in range(len(grid_T)):
        for j in range(len(grid_eps)):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] < (values.min() + values.max()) / 2 else "black")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_degradation_comparison(
    grid_eps: List[float],
    all_degradations: Dict[str, List[float]],
    title: str,
    outpath: str,
) -> None:
    """Bar chart: mean degradation (clean - noisy) per model, grouped by eps."""
    labels = list(all_degradations.keys())
    n_eps = len(grid_eps)
    n_models = len(labels)
    x = range(n_eps)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(6, n_eps * 2), 4))
    for i, lbl in enumerate(labels):
        vals = all_degradations[lbl]
        offsets = [xi + (i - n_models / 2 + 0.5) * width for xi in x]
        ax.bar(offsets, vals, width=width * 0.9, label=lbl)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"eps={e}" for e in grid_eps], rotation=30)
    ax.set_ylabel("Mean degradation (clean - noisy perf)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_pareto(
    points: List[Tuple[float, float, str]],
    baseline_labels: List[str],
    l2ru_gammas: List[float],
    l2ru_param_name: str,
    title: str,
    outpath: str,
) -> None:
    """
    Pareto plot with:
      - baselines: red circles
      - L2RU variants: blue squares, sized by gamma, connected by an arrow
        from smallest to largest gamma to show the trade-off sweep direction
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    baseline_set = set(baseline_labels)
    l2ru_points = []   # (clean, noisy, label, gamma)

    for clean, noisy, lbl in points:
        if lbl in baseline_set:
            ax.scatter([clean], [noisy], color="crimson", marker="o", s=80, zorder=5)
            ax.annotate(lbl, (clean, noisy), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="crimson")
        else:
            # find the gamma for this label
            gamma = None
            for g in l2ru_gammas:
                gtag = str(g).replace(".", "p")
                if lbl == f"{l2ru_param_name}_g{gtag}":
                    gamma = g
                    break
            l2ru_points.append((clean, noisy, lbl, gamma if gamma is not None else 1.0))

    if l2ru_points:
        # sort by gamma to draw arrow in order
        l2ru_points.sort(key=lambda t: t[3])
        gammas_vals = [t[3] for t in l2ru_points]
        g_min, g_max = min(gammas_vals), max(gammas_vals)
        g_range = g_max - g_min if g_max > g_min else 1.0

        for clean, noisy, lbl, gamma in l2ru_points:
            size = 60 + 160 * (gamma - g_min) / g_range
            ax.scatter([clean], [noisy], color="steelblue", marker="s", s=size, zorder=5,
                       alpha=0.85)
            ax.annotate(lbl, (clean, noisy), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="steelblue")

        # Draw arrows along the gamma sweep
        for i in range(len(l2ru_points) - 1):
            x0, y0 = l2ru_points[i][0], l2ru_points[i][1]
            x1, y1 = l2ru_points[i + 1][0], l2ru_points[i + 1][1]
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.2),
            )

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
               markersize=9, label="LRU baseline"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="steelblue",
               markersize=9, label=f"{l2ru_param_name} (arrow = ↑ gamma)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("Clean performance (acc or −mse)", fontsize=10)
    ax.set_ylabel("Robust performance (acc or −mse)", fontsize=10)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_gamma_tradeoff_plot(
    gammas: List[float],
    clean_vals: List[float],
    noisy_vals: List[float],
    gain_vals: List[float],
    baseline_clean: Dict[str, float],
    baseline_noisy: Dict[str, float],
    baseline_gain: Dict[str, float],
    task: str,
    T: int,
    eps: float,
    outpath: str,
) -> None:
    """
    Twin-axis plot for a fixed (T, eps) cell.

    Left axis : clean & noisy performance vs prescribed gamma (L2RU line + LRU baselines).
    Right axis: empirical gain ||Dy||/||d|| vs gamma, with a y=gamma dashed bound line.

    This is the primary 'knob' plot: it makes it visually obvious that gamma
    controls a continuous robustness-performance tradeoff in L2RU, while
    LRU sits at a single uncontrolled operating point.
    """
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    metric = "Accuracy" if task == "classif" else u"\u2212MSE"

    ax1.plot(gammas, clean_vals, "o-",  color="steelblue", lw=2.0, label=f"L2RU clean {metric}")
    ax1.plot(gammas, noisy_vals, "s--", color="steelblue", lw=2.0, alpha=0.6,
             label=f"L2RU noisy {metric}")
    ax2.plot(gammas, gain_vals,  "^:",  color="darkorange", lw=1.5, label="L2RU empirical gain")
    ax2.plot(gammas, gammas,     "k--", lw=1.0, alpha=0.5, label=u"\u03b3 (prescribed bound)")

    _bl_colors = ["crimson", "forestgreen", "purple"]
    for i, bl in enumerate(baseline_clean):
        c = _bl_colors[i % len(_bl_colors)]
        ax1.axhline(baseline_clean[bl], ls="-.",              color=c, lw=1.5, label=f"{bl} clean")
        ax1.axhline(baseline_noisy[bl], ls=":",               color=c, lw=1.5, label=f"{bl} noisy")
        ax2.axhline(baseline_gain[bl],  ls=(0, (3, 1, 1, 1)), color=c, lw=1.2, alpha=0.7,
                    label=f"{bl} gain")

    ax1.set_xlabel(u"Prescribed \u03b3 (L2 gain bound)", fontsize=11)
    ax1.set_ylabel(metric, fontsize=10)
    ax2.set_ylabel(u"Empirical gain  \u2016\u0394y\u2016\u2082 / \u2016d\u2016\u2082",
                   fontsize=10, color="darkorange")
    ax2.tick_params(axis="y", colors="darkorange")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="best", ncol=2)
    ax1.set_title(f"L2RU \u03b3 tradeoff  |  T={T}, \u03b5={eps:.2f}  "
                  f"[trained clean, tested clean & noisy]", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_gain_certificate_plot(
    gammas: List[float],
    gain_mean: List[float],
    gain_p95: List[float],
    baseline_gains: Dict[str, Tuple[float, float]],   # label -> (mean, p95)
    T: int,
    eps: float,
    outpath: str,
) -> None:
    """
    Empirical gain vs prescribed gamma, with a y = gamma bound line.

    The central certificate plot: demonstrates that L2RU's empirical output
    sensitivity stays at or below its prescribed gain bound gamma, while
    the LRU baseline sits at an uncontrolled (and unverifiable) operating point.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(gammas, gain_mean, "o-",  color="steelblue", lw=2.0, label="L2RU gain (mean)")
    ax.fill_between(gammas, gain_mean, gain_p95, alpha=0.18, color="steelblue")
    ax.plot(gammas, gain_p95, "^--",  color="steelblue", lw=1.2, alpha=0.6, label="L2RU gain (p95)")
    ax.plot(gammas, gammas,   "k--",  lw=1.8, label="Theoretical bound  (y = \u03b3)")

    _bl_colors = ["crimson", "forestgreen", "purple"]
    for i, (bl, (bg_mean, bg_p95)) in enumerate(baseline_gains.items()):
        c = _bl_colors[i % len(_bl_colors)]
        ax.axhline(bg_mean, ls="-.", color=c, lw=1.5, label=f"{bl} gain (mean)")
        ax.axhline(bg_p95,  ls=":",  color=c, lw=1.0, alpha=0.7, label=f"{bl} gain (p95)")

    ax.set_xlabel(u"Prescribed \u03b3", fontsize=11)
    ax.set_ylabel(u"Empirical  \u2016\u0394y\u2016\u2082 / \u2016d\u2016\u2082", fontsize=10)
    ax.set_title(f"Gain certificate  |  T={T}, \u03b5={eps:.2f}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ----------------------------
# Summary table
# ----------------------------

def print_summary_table(
    pareto_points: List[Tuple[float, float, str]],
    task: str,
) -> None:
    metric_name = "acc" if task == "classif" else "-mse"
    header = f"{'Label':<24} {'Clean '+metric_name:>14} {'Noisy '+metric_name:>14} {'Degradation':>14} {'Deg %':>10}"
    print("\n" + "=" * len(header))
    print("  BENCHMARK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for clean, noisy, lbl in sorted(pareto_points, key=lambda t: -t[1]):
        deg = clean - noisy
        deg_pct = (deg / abs(clean) * 100) if abs(clean) > 1e-9 else float("nan")
        print(f"{lbl:<24} {clean:>14.4f} {noisy:>14.4f} {deg:>14.4f} {deg_pct:>9.1f}%")
    print("=" * len(header) + "\n")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="gammabench_runs")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--task", type=str, default="classif", choices=["classif", "regress"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Benchmark knobs
    ap.add_argument("--grid_T", type=str, default="128,256,512")
    ap.add_argument("--grid_eps", type=str, default="0.0,1.0,3.0,10.0")
    ap.add_argument("--rho", type=float, default=0.99)
    ap.add_argument("--cue_len", type=int, default=8)
    ap.add_argument("--b", type=float, default=0.5)
    ap.add_argument("--process_noise_std", type=float, default=0.01)
    ap.add_argument("--meas_noise_std", type=float, default=0.01)

    ap.add_argument("--disturbance", type=str, default="dense", choices=["dense", "burst", "none"])
    ap.add_argument("--burst_frac", type=float, default=0.1)

    # Models
    ap.add_argument("--baselines", type=str, default="lru")
    ap.add_argument("--l2ru_param_name", type=str, default="l2n")
    ap.add_argument("--gammas", type=str, default="0.5,1.0,2.0,4.0")
    ap.add_argument("--max_phase_b", type=float, default=float("nan"))
    ap.add_argument("--train_gamma", type=int, default=0, choices=[0, 1],
                    help="Set to 1 to make internal L2-SSM gammas trainable. "
                         "Default 0 keeps the prescribed-gain interpretation cleaner.")
    ap.add_argument("--ff_scale", type=float, default=1.0,
                    help="Lipschitz scale passed to FF blocks (e.g. LGLU/LMLP).")

    # DeepSSM hyperparams
    ap.add_argument("--d_model", type=int, default=8)
    ap.add_argument("--d_state", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=1)

    # Training
    ap.add_argument("--train_n", type=int, default=8000)
    ap.add_argument("--val_n", type=int, default=2000)
    ap.add_argument("--test_n", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # 0.0 = select on clean only (recommended: models trained clean), 1.0 = noisy only
    ap.add_argument("--val_noisy_weight", type=float, default=0.0,
                    help="Weight given to noisy val metric when selecting the best checkpoint "
                         "(0=clean only, 1=noisy only). Default 0.0: models are trained on "
                         "clean data and selected by clean val performance, so the robustness "
                         "difference is purely architectural (L2 gain bound).")
    ap.add_argument("--val_interval", type=int, default=5,
                    help="Evaluate on val sets every N epochs (default: 5). "
                         "Use 1 to evaluate every epoch.")
    ap.add_argument("--gamma_plot_T", type=int, default=-1,
                    help="T value to use for gamma-tradeoff / gain-certificate plots. "
                         "-1 (default) = largest T in grid_T.")
    ap.add_argument("--gamma_plot_eps", type=float, default=-1.0,
                    help="eps value for gamma-tradeoff / gain-certificate plots. "
                         "-1 (default) = largest non-zero eps in grid_eps.")

    args = ap.parse_args()

    # l2n / l2ru require even d_state (2×2 block structure). Round up silently.
    _l2_params = {"l2n", "l2ru", "lruz"}
    l2ru_needs_even = args.l2ru_param_name in _l2_params
    baseline_needs_even = any(p in _l2_params for p in args.baselines.split(","))
    if (l2ru_needs_even or baseline_needs_even) and args.d_state % 2 != 0:
        args.d_state += 1
        print(f"[warn] d_state rounded up to {args.d_state} (must be even for {args.l2ru_param_name})")

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    grid_T   = [int(x.strip())   for x in args.grid_T.split(",")   if x.strip()]
    grid_eps = [float(x.strip()) for x in args.grid_eps.split(",") if x.strip()]
    gammas   = [float(x.strip()) for x in args.gammas.split(",")   if x.strip()]
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]

    max_phase_b = None if math.isnan(args.max_phase_b) else args.max_phase_b

    jsonl_path = os.path.join(args.out_dir, "results.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    def write_result(d: Dict):
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d) + "\n")

    def perf_metric(metrics: Dict[str, float]) -> float:
        if args.task == "classif":
            return float(metrics["acc"])
        return -float(metrics["mse"])

    # Flat list of all (label, param, gamma) specs to run.
    all_model_specs: List[Tuple[str, str, Optional[float]]] = [
        (p, p, None) for p in baselines
    ] + [
        (f"{args.l2ru_param_name}_g{str(g).replace('.', 'p')}", args.l2ru_param_name, g)
        for g in gammas
    ]
    all_labels = [lbl for lbl, _, _ in all_model_specs]

    # Pre-allocate per-model result grids.
    perf_grids_clean: Dict[str, torch.Tensor] = {
        lbl: torch.zeros(len(grid_T), len(grid_eps)) for lbl in all_labels
    }
    perf_grids_noisy: Dict[str, torch.Tensor] = {
        lbl: torch.zeros(len(grid_T), len(grid_eps)) for lbl in all_labels
    }
    perf_grids_gain: Dict[str, torch.Tensor] = {
        lbl: torch.zeros(len(grid_T), len(grid_eps)) for lbl in all_labels
    }
    perf_grids_gain_p95: Dict[str, torch.Tensor] = {
        lbl: torch.zeros(len(grid_T), len(grid_eps)) for lbl in all_labels
    }

    # --- Main loop: cell-outer / model-inner so datasets are built once per cell ---
    cells = [
        (iT, T, ie, eps)
        for iT, T in enumerate(grid_T)
        for ie, eps in enumerate(grid_eps)
    ]
    n_cells = len(cells)
    cell_iter = _tqdm(cells, desc="cells", total=n_cells) if HAVE_TQDM else cells

    for iT, T, ie, eps in cell_iter:
        cfg = GammaBenchConfig(
            T=T,
            cue_len=args.cue_len,
            rho=args.rho,
            b=args.b,
            process_noise_std=args.process_noise_std,
            meas_noise_std=args.meas_noise_std,
            eps=eps,
            disturbance=args.disturbance,
            burst_frac=args.burst_frac,
            task=args.task,
            normalize_input=True,
        )

        # Datasets are deterministic — build once and share across all models.
        train_ds  = GammaBenchDataset(args.train_n, cfg, seed=args.seed + 1000)
        val_ds    = GammaBenchDataset(args.val_n,   cfg, seed=args.seed + 2000)
        test_ds   = GammaBenchDataset(args.test_n,  cfg, seed=args.seed + 3000)

        train_loader      = make_loader(train_ds, cfg, device, args.batch_size, True,  True,  False)
        val_loader_clean  = make_loader(val_ds,   cfg, device, args.batch_size, False, False, False)
        val_loader_noisy  = make_loader(val_ds,   cfg, device, args.batch_size, False, False, True)
        test_loader_clean = make_loader(test_ds,  cfg, device, args.batch_size, False, False, False)
        test_loader_noisy = make_loader(test_ds,  cfg, device, args.batch_size, False, False, True)

        val_seed  = args.seed + iT * 997 + ie * 101
        test_seed = args.seed + iT * 997 + ie * 101 + 50000

        for label, param, gamma in all_model_specs:
            model = build_model(
                param=param,
                gamma=gamma,
                d_input=1,
                d_output=1,
                d_model=args.d_model,
                d_state=args.d_state,
                n_layers=args.n_layers,
                device=device,
                max_phase_b=max_phase_b,
                train_gamma=bool(args.train_gamma),
                ff_scale=args.ff_scale,
            )
            train_metrics = train_one(
                model=model,
                train_loader=train_loader,
                val_loader_clean=val_loader_clean,
                val_loader_noisy=val_loader_noisy,
                task=args.task,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                grad_clip=args.grad_clip,
                val_noisy_weight=args.val_noisy_weight,
                val_seed=val_seed,
                val_interval=args.val_interval,
            )

            test_clean = evaluate(model, test_loader_clean, args.task, seed=test_seed)
            test_noisy = evaluate(model, test_loader_noisy, args.task, seed=test_seed + 1)
            test_gain = evaluate_disturbance_gain(model, test_loader_clean, cfg, seed=test_seed + 2)

            perf_grids_clean[label][iT, ie] = perf_metric(test_clean)
            perf_grids_noisy[label][iT, ie] = perf_metric(test_noisy)
            perf_grids_gain[label][iT, ie]     = float(test_gain["gain_mean"])
            perf_grids_gain_p95[label][iT, ie] = float(test_gain["gain_p95"])

            write_result({
                "label": label,
                "param": param,
                "gamma": gamma,
                "T": T,
                "eps": eps,
                "rho": args.rho,
                "task": args.task,
                "disturbance": args.disturbance,
                "burst_frac": args.burst_frac,
                "deepssm": {
                    "d_model": args.d_model,
                    "d_state": args.d_state,
                    "n_layers": args.n_layers,
                    "max_phase_b": max_phase_b,
                    "train_gamma": bool(args.train_gamma),
                    "ff_scale": args.ff_scale,
                },
                "train": train_metrics,
                "test_clean": test_clean,
                "test_noisy": test_noisy,
                "test_gain": test_gain,
            })

    # --- Aggregate: heatmaps, Pareto, summary ---
    pareto_points: List[Tuple[float, float, str]] = []
    all_degradations: Dict[str, List[float]] = {}

    for label, param, gamma in all_model_specs:
        pgc = perf_grids_clean[label]
        pgn = perf_grids_noisy[label]
        pgd = pgc - pgn
        pgg = perf_grids_gain[label]

        save_heatmap(grid_T, grid_eps, pgc,
                     title=f"{label}: clean performance",
                     outpath=os.path.join(args.out_dir, f"heatmap_{label}_clean.png"))
        save_heatmap(grid_T, grid_eps, pgn,
                     title=f"{label}: robust performance",
                     outpath=os.path.join(args.out_dir, f"heatmap_{label}_noisy.png"))
        save_heatmap(grid_T, grid_eps, pgd,
                     title=f"{label}: degradation (clean − noisy)",
                     outpath=os.path.join(args.out_dir, f"heatmap_{label}_degradation.png"),
                     vmin=0.0, cmap="Reds")

        save_heatmap(grid_T, grid_eps, pgg,
                     title=f"{label}: disturbance sensitivity ||Δy||₂ / ||d||₂ (mean)",
                     outpath=os.path.join(args.out_dir, f"heatmap_{label}_gain.png"),
                     vmin=0.0, cmap="magma")
        all_degradations[label] = pgd.mean(dim=0).tolist()
        pareto_points.append((float(pgc.mean()), float(pgn.mean()), label))

    save_degradation_comparison(
        grid_eps=grid_eps,
        all_degradations=all_degradations,
        title="Mean degradation per model (averaged over T)",
        outpath=os.path.join(args.out_dir, "degradation_comparison.png"),
    )

    save_pareto(
        points=pareto_points,
        baseline_labels=baselines,
        l2ru_gammas=gammas,
        l2ru_param_name=args.l2ru_param_name,
        title="Pareto (mean over grid): clean vs robust",
        outpath=os.path.join(args.out_dir, "pareto_clean_vs_robust.png"),
    )

    print_summary_table(pareto_points, args.task)

    # --- Gamma-sweep and gain-certificate plots ---
    # Resolve which (T, eps) cell to use for these plots.
    _req_T   = args.gamma_plot_T
    _req_eps = args.gamma_plot_eps

    iT_ps = (grid_T.index(_req_T)
             if _req_T in grid_T
             else len(grid_T) - 1)

    _nonzero_eps = [e for e in grid_eps if e > 0]
    _default_eps = _nonzero_eps[-1] if _nonzero_eps else grid_eps[-1]
    # Closest match for float eps (avoids exact float equality issues).
    if _req_eps > 0 and any(abs(e - _req_eps) < 1e-9 for e in grid_eps):
        ie_ps = min(range(len(grid_eps)), key=lambda i: abs(grid_eps[i] - _req_eps))
    else:
        ie_ps = min(range(len(grid_eps)), key=lambda i: abs(grid_eps[i] - _default_eps))

    T_ps   = grid_T[iT_ps]
    eps_ps = grid_eps[ie_ps]
    print(f"[gamma plots] Using T={T_ps}, eps={eps_ps}")

    l2ru_specs = [
        (lbl, g)
        for lbl, param, g in all_model_specs
        if param == args.l2ru_param_name and g is not None
    ]

    if len(l2ru_specs) >= 2:
        l2ru_specs_sorted = sorted(l2ru_specs, key=lambda t: t[1])
        ps_gammas   = [g   for _, g in l2ru_specs_sorted]
        ps_clean    = [float(perf_grids_clean[lbl][iT_ps, ie_ps])    for lbl, _ in l2ru_specs_sorted]
        ps_noisy    = [float(perf_grids_noisy[lbl][iT_ps, ie_ps])    for lbl, _ in l2ru_specs_sorted]
        ps_gain     = [float(perf_grids_gain[lbl][iT_ps, ie_ps])     for lbl, _ in l2ru_specs_sorted]
        ps_gain_p95 = [float(perf_grids_gain_p95[lbl][iT_ps, ie_ps]) for lbl, _ in l2ru_specs_sorted]

        bl_clean    = {bl: float(perf_grids_clean[bl][iT_ps, ie_ps])    for bl in baselines}
        bl_noisy    = {bl: float(perf_grids_noisy[bl][iT_ps, ie_ps])    for bl in baselines}
        bl_gain     = {bl: float(perf_grids_gain[bl][iT_ps, ie_ps])     for bl in baselines}
        bl_gain_p95 = {bl: float(perf_grids_gain_p95[bl][iT_ps, ie_ps]) for bl in baselines}

        save_gamma_tradeoff_plot(
            gammas=ps_gammas,
            clean_vals=ps_clean,
            noisy_vals=ps_noisy,
            gain_vals=ps_gain,
            baseline_clean=bl_clean,
            baseline_noisy=bl_noisy,
            baseline_gain=bl_gain,
            task=args.task,
            T=T_ps,
            eps=eps_ps,
            outpath=os.path.join(args.out_dir, "gamma_tradeoff.png"),
        )

        save_gain_certificate_plot(
            gammas=ps_gammas,
            gain_mean=ps_gain,
            gain_p95=ps_gain_p95,
            baseline_gains={bl: (bl_gain[bl], bl_gain_p95[bl]) for bl in baselines},
            T=T_ps,
            eps=eps_ps,
            outpath=os.path.join(args.out_dir, "gain_certificate.png"),
        )
    else:
        print("[gamma plots] Skipped: need at least 2 L2RU gamma values to draw sweep plots.")

    print(f"Results written to: {args.out_dir}/")
    print(f"  {jsonl_path}")
    print("  heatmap_<label>_{clean,noisy,degradation,gain}.png")
    print("  degradation_comparison.png")
    print("  pareto_clean_vs_robust.png")
    print("  gamma_tradeoff.png       <- tradeoff knob plot")
    print("  gain_certificate.png     <- empirical gain vs prescribed gamma")


if __name__ == "__main__":
    main()
















