#!/usr/bin/env python
# file: Test_files/run_benchmarks.py
"""Run repo models against the ``nonlinear_benchmarks`` system-identification suite.

This is a self-contained benchmark harness:

* loads any subset of the ``nonlinear_benchmarks`` datasets (SISO *and* MIMO),
  handling the tuple-of-datasets splits (CED, Silverbox, ParWH, F16, …) and the
  not-splitted MIMO ``Industrial_robot`` benchmark;
* trains models in **open-loop simulation** mode with windowed mini-batches
  (GPU-ready, all window tensors live on the device, no DataLoader overhead);
* evaluates with the official protocol (free-run simulation, skipping each test
  sequence's ``state_initialization_window_length``) and the package metrics
  (RMSE / NRMSE / R² / fit-index, per channel, in physical units);
* draws **animated training plots** — a reference-trajectory + loss view that is
  exported as a GIF (headless-safe) and optionally shown live; for MIMO outputs
  it plots the loss plus several output-channel trajectories;
* writes a neat report (Markdown + JSON + per-run figures) and a console table.

Examples
--------
    # One benchmark, two models
    python Test_files/run_benchmarks.py --benchmarks Cascaded_Tanks --models raven lstm

    # All (small) splitted benchmarks, every model, on GPU
    python Test_files/run_benchmarks.py --benchmarks all --device cuda --epochs 400

    # The MIMO industrial robot
    python Test_files/run_benchmarks.py --benchmarks Industrial_robot --models tv raven

    # List what is available
    python Test_files/run_benchmarks.py --list
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Headless-safe by default; the live monitor upgrades to a GUI backend on request.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn

# Make `src...` importable no matter the current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nonlinear_benchmarks as nlb
from nonlinear_benchmarks import Input_output_data
from nonlinear_benchmarks.error_metrics import RMSE, NRMSE, R_squared, MAE, fit_index
from nonlinear_benchmarks.not_splitted_benchmarks import (
    Industrial_robot,
    WienerHammerstein_Process_Noise,
)

from src.neural_ssm.ssm import DeepSSM, SSMConfig


# ============================================================================
# Benchmark registry
# ============================================================================

# Splitted loaders accept atleast_2d / always_return_tuples_of_datasets.
_SPLITTED: Dict[str, Callable] = {
    "EMPS": nlb.EMPS,
    "CED": nlb.CED,
    "Cascaded_Tanks": nlb.Cascaded_Tanks,
    "Silverbox": nlb.Silverbox,
    "WienerHammerBenchMark": nlb.WienerHammerBenchMark,
    "ParWH": nlb.ParWH,
    "F16": nlb.F16,
}
# Not-splitted loaders return (train_list, test_list) but lack those kwargs.
_NOT_SPLITTED: Dict[str, Callable] = {
    "Industrial_robot": Industrial_robot,                       # MIMO: 6-in / 6-out
    "WienerHammerstein_Process_Noise": WienerHammerstein_Process_Noise,  # large download
}
_ALL_LOADERS = {**_SPLITTED, **_NOT_SPLITTED}

# Named groups for the CLI.
_GROUPS: Dict[str, List[str]] = {
    "small": ["EMPS", "CED", "Cascaded_Tanks"],
    "all": list(_SPLITTED),                       # all officially-split benchmarks
    "mimo": ["Industrial_robot"],
    "heavy": ["WienerHammerBenchMark", "ParWH", "F16", "WienerHammerstein_Process_Noise"],
}


@dataclass
class BenchmarkData:
    """A loaded benchmark, normalized to lists of 2D ``Input_output_data``."""
    name: str
    train: List[Input_output_data]
    test: List[Input_output_data]
    n_u: int
    n_y: int
    sampling_time: Optional[float]
    init_window: int


def _as_dataset_list(obj) -> List[Input_output_data]:
    if isinstance(obj, Input_output_data):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise TypeError(f"Unexpected benchmark return element of type {type(obj).__name__}.")


def load_benchmark(name: str) -> BenchmarkData:
    """Load one benchmark and normalize it to ``(train_list, test_list)`` of 2D data."""
    if name in _SPLITTED:
        train, test = _SPLITTED[name](atleast_2d=True, always_return_tuples_of_datasets=True)
        train_list, test_list = _as_dataset_list(train), _as_dataset_list(test)
    elif name in _NOT_SPLITTED:
        train, test = _NOT_SPLITTED[name](train_test_split=True)
        train_list = [d.atleast_2d() for d in _as_dataset_list(train)]
        test_list = [d.atleast_2d() for d in _as_dataset_list(test)]
    else:
        raise KeyError(f"Unknown benchmark {name!r}. Available: {sorted(_ALL_LOADERS)}")

    if not train_list or not test_list:
        raise RuntimeError(f"Benchmark {name!r} produced an empty train/test split.")

    n_u = int(train_list[0].u.shape[-1])
    n_y = int(train_list[0].y.shape[-1])
    init_window = test_list[0].state_initialization_window_length or 0
    return BenchmarkData(
        name=name,
        train=train_list,
        test=test_list,
        n_u=n_u,
        n_y=n_y,
        sampling_time=getattr(train_list[0], "sampling_time", None),
        init_window=int(init_window),
    )


def resolve_benchmark_names(tokens: Sequence[str]) -> List[str]:
    """Expand group tokens (small/all/mimo/heavy) and validate explicit names."""
    out: List[str] = []
    for tok in tokens:
        if tok in _GROUPS:
            out.extend(_GROUPS[tok])
        elif tok in _ALL_LOADERS:
            out.append(tok)
        else:
            raise SystemExit(
                f"Unknown benchmark/group {tok!r}. "
                f"Benchmarks: {sorted(_ALL_LOADERS)}; groups: {sorted(_GROUPS)}."
            )
    # De-duplicate, preserve order.
    seen, unique = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


# ============================================================================
# Normalization (per-channel z-score, fit on training data only)
# ============================================================================

@dataclass
class Standardizer:
    u_mean: torch.Tensor
    u_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    @classmethod
    def fit(cls, train: List[Input_output_data], eps: float = 1e-8) -> "Standardizer":
        u = np.concatenate([d.u for d in train], axis=0)
        y = np.concatenate([d.y for d in train], axis=0)
        return cls(
            u_mean=torch.tensor(u.mean(0), dtype=torch.float32),
            u_std=torch.tensor(u.std(0), dtype=torch.float32).clamp_min(eps),
            y_mean=torch.tensor(y.mean(0), dtype=torch.float32),
            y_std=torch.tensor(y.std(0), dtype=torch.float32).clamp_min(eps),
        )

    def norm_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mean.to(u)) / self.u_std.to(u)

    def norm_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean.to(y)) / self.y_std.to(y)

    def denorm_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.to(y) + self.y_mean.to(y)


# ============================================================================
# Windowing for open-loop simulation training
# ============================================================================

def make_windows(
    sequences: List[Tuple[np.ndarray, np.ndarray]],
    *,
    seq_len: int,
    stride: int,
    washout: int,
    max_windows: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Cut (already-normalized) sequences into fixed-length windows for batched training.

    Each window starts from zero state; a per-window mask zeroes the loss over the
    ``washout`` warm-up and any zero-padding (for sequences shorter than ``seq_len``),
    so simulation training works uniformly across very different sequence lengths.

    Returns ``(U, Y, M, L)`` with ``U:(Nw,L,n_u)``, ``Y:(Nw,L,n_y)``, ``M:(Nw,L)``.
    """
    max_len = max(len(u) for u, _ in sequences)
    L = min(seq_len, max_len)
    n_u = sequences[0][0].shape[-1]
    n_y = sequences[0][1].shape[-1]

    u_wins, y_wins, masks = [], [], []
    for u, y in sequences:
        n = len(u)
        if n >= L:
            starts = list(range(0, n - L + 1, stride))
            if (n - L) % stride != 0:
                starts.append(n - L)
        else:
            starts = [0]
        for s in starts:
            seg = min(L, n - s)
            wo = min(washout, seg // 2)        # keep at least half the window supervised
            if seg - wo <= 0:
                continue
            uw = np.zeros((L, n_u), dtype=np.float32)
            yw = np.zeros((L, n_y), dtype=np.float32)
            mw = np.zeros((L,), dtype=np.float32)
            uw[:seg] = u[s:s + seg]
            yw[:seg] = y[s:s + seg]
            mw[wo:seg] = 1.0
            u_wins.append(uw)
            y_wins.append(yw)
            masks.append(mw)

    if not u_wins:
        raise RuntimeError("No training windows were produced (sequences too short?).")

    U = torch.from_numpy(np.stack(u_wins))
    Y = torch.from_numpy(np.stack(y_wins))
    M = torch.from_numpy(np.stack(masks))

    if max_windows is not None and U.shape[0] > max_windows:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(U.shape[0], generator=g)[:max_windows]
        dropped = U.shape[0] - max_windows
        print(f"    [windows] capping {U.shape[0]} -> {max_windows} windows "
              f"({dropped} dropped at random; raise --max-windows to keep all).")
        U, Y, M = U[idx], Y[idx], M[idx]

    return U, Y, M, L


# ============================================================================
# Models
# ============================================================================

class DeepSSMSim(nn.Module):
    """Wraps a ``DeepSSM`` to a uniform ``forward(u:(B,T,n_u)) -> y:(B,T,n_y)``."""

    def __init__(self, n_u: int, n_y: int, cfg: SSMConfig, mode: str = "scan"):
        super().__init__()
        self.core = DeepSSM(d_input=n_u, d_output=n_y, config=cfg)
        self.mode = mode

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        y, _ = self.core(u, mode=self.mode, reset_state=True)
        return y

    @torch.no_grad()
    def diagnostics(self) -> Optional[dict]:
        return self.core.gain_diagnostics() if self.core.use_cert_scaling else None


class RNNSim(nn.Module):
    """LSTM/GRU baseline with the same uniform interface."""

    def __init__(self, n_u: int, n_y: int, *, kind: str = "lstm",
                 hidden: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = rnn_cls(n_u, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden, n_y)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(u)
        return self.proj(h)

    def diagnostics(self) -> Optional[dict]:
        return None


# name -> spec. `kind` selects the builder; remaining keys are builder kwargs.
MODEL_ZOO: Dict[str, dict] = {
    "lru":   dict(kind="deepssm", param="lru",   gamma=None, ff="GLU"),     # uncertified baseline
    "l2ru":  dict(kind="deepssm", param="l2ru",  gamma=5.0,  ff="LGLU2"),   # certified LTI
    "tv":    dict(kind="deepssm", param="tv",    gamma=5.0,  ff="MBLIP"),   # certified selective
    "raven": dict(kind="deepssm", param="raven", gamma=5.0,  ff="MBLIP"),   # certified Raven cell
    "lstm":  dict(kind="lstm",    hidden=64, layers=2),
    "gru":   dict(kind="gru",     hidden=64, layers=2),
}


def build_model(name: str, n_u: int, n_y: int, gconf: "GlobalModelConfig") -> nn.Module:
    spec = dict(MODEL_ZOO[name])
    kind = spec.pop("kind")
    if kind == "deepssm":
        cfg = SSMConfig(
            d_model=gconf.d_model,
            d_state=gconf.d_state,
            n_layers=gconf.n_layers,
            d_hidden=gconf.d_hidden,
            nl_layers=gconf.nl_layers,
            scale=gconf.scale,
            param=spec["param"],
            gamma=spec.get("gamma"),
            ff=spec.get("ff", "LGLU2"),
            train_gamma=True,
            learn_x0=False,
            raven_heads=gconf.raven_heads,
            raven_slots=gconf.raven_slots,
            raven_top_k=gconf.raven_top_k,
        )
        return DeepSSMSim(n_u, n_y, cfg)
    if kind in ("lstm", "gru"):
        return RNNSim(n_u, n_y, kind=kind, hidden=spec["hidden"], layers=spec["layers"])
    raise ValueError(f"Unknown model kind {kind!r}.")


@dataclass
class GlobalModelConfig:
    d_model: int = 16
    d_state: int = 16
    n_layers: int = 2
    d_hidden: int = 16
    nl_layers: int = 3
    scale: float = 1.0
    raven_heads: int = 4
    raven_slots: int = 8
    raven_top_k: int = 2


# ============================================================================
# Device / training
# ============================================================================

def pick_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def simulate(model: nn.Module, u_np: np.ndarray, norm: Standardizer,
             device: torch.device) -> np.ndarray:
    """Free-run open-loop prediction over a full sequence; returns physical-unit y."""
    model.eval()
    u = torch.from_numpy(np.asarray(u_np, dtype=np.float32))
    u = norm.norm_u(u).unsqueeze(0).to(device)          # (1,T,n_u)
    y = model(u).squeeze(0).float().cpu()               # (T,n_y) normalized
    return norm.denorm_y(y).numpy()


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 3e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 40
    seq_len: int = 1024
    stride: Optional[int] = None       # default: non-overlapping (= L)
    washout: int = 50
    val_frac: float = 0.2
    max_windows: Optional[int] = 4000
    amp: bool = False
    seed: int = 0


def _split_tail(seqs, frac):
    """Per-sequence time split into (head, tail) by fraction; tail used for validation."""
    head, tail = [], []
    for u, y in seqs:
        n = len(u)
        cut = int(round(n * (1.0 - frac)))
        cut = min(max(cut, 1), n - 1) if n > 1 else n
        head.append((u[:cut], y[:cut]))
        if frac > 0 and cut < n:
            tail.append((u[cut:], y[cut:]))
    return head, tail


def train_model(
    model: nn.Module,
    bench: BenchmarkData,
    norm: Standardizer,
    tcfg: TrainConfig,
    device: torch.device,
    monitor: "Optional[TrainingMonitor]" = None,
) -> dict:
    """Windowed open-loop simulation training with validation early-stopping."""
    model.to(device)
    rng = np.random.default_rng(tcfg.seed)

    # Normalize all train sequences, then per-sequence tail-split into train/val.
    def _norm_pair(d):
        u = norm.norm_u(torch.from_numpy(d.u.astype(np.float32))).numpy()
        y = norm.norm_y(torch.from_numpy(d.y.astype(np.float32))).numpy()
        return u, y

    norm_seqs = [_norm_pair(d) for d in bench.train]
    train_seqs, val_seqs = _split_tail(norm_seqs, tcfg.val_frac)

    stride = tcfg.stride or tcfg.seq_len
    Ut, Yt, Mt, L = make_windows(
        train_seqs, seq_len=tcfg.seq_len, stride=stride,
        washout=tcfg.washout, max_windows=tcfg.max_windows,
    )
    Ut, Yt, Mt = Ut.to(device), Yt.to(device), Mt.to(device)

    has_val = len(val_seqs) > 0
    if has_val:
        Uv, Yv, Mv, _ = make_windows(
            val_seqs, seq_len=L, stride=L, washout=min(tcfg.washout, L // 2),
            max_windows=tcfg.max_windows,
        )
        Uv, Yv, Mv = Uv.to(device), Yv.to(device), Mv.to(device)

    n_y = bench.n_y
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tcfg.epochs, 1))
    use_amp = tcfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def masked_mse(pred, tgt, mask):
        diff2 = (pred - tgt) ** 2 * mask.unsqueeze(-1)
        return diff2.sum() / (mask.sum().clamp_min(1.0) * n_y)

    @torch.no_grad()
    def eval_windows(U, Y, M):
        model.eval()
        tot, denom = 0.0, 0.0
        for i in range(0, U.shape[0], tcfg.batch_size):
            pred = model(U[i:i + tcfg.batch_size])
            m = M[i:i + tcfg.batch_size]
            tot += (((pred - Y[i:i + tcfg.batch_size]) ** 2) * m.unsqueeze(-1)).sum().item()
            denom += m.sum().item() * n_y
        return tot / max(denom, 1.0)

    n_win = Ut.shape[0]
    train_losses, val_losses = [], []
    best_val, best_state, bad = math.inf, None, 0
    t0 = time.time()

    for epoch in range(tcfg.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(n_win)).to(device)
        running, seen = 0.0, 0
        for i in range(0, n_win, tcfg.batch_size):
            idx = perm[i:i + tcfg.batch_size]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(Ut[idx])
                loss = masked_mse(pred, Yt[idx], Mt[idx])
            scaler.scale(loss).backward()
            if tcfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * idx.numel()
            seen += idx.numel()
        sched.step()
        train_loss = running / max(seen, 1)
        train_losses.append(train_loss)

        val_loss = eval_windows(Uv, Yv, Mv) if has_val else train_loss
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-9:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if monitor is not None:
            monitor.update(epoch, train_losses, val_losses, model)

        if bad >= tcfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val,
        "epochs_run": len(train_losses),
        "train_time_s": time.time() - t0,
        "n_windows": int(n_win),
        "window_len": int(L),
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(model: nn.Module, bench: BenchmarkData, norm: Standardizer,
             device: torch.device) -> Tuple[List[dict], List[np.ndarray]]:
    """Per-test-sequence metrics (physical units) + the predictions for plotting."""
    rows, preds = [], []
    for d in bench.test:
        y_pred = simulate(model, d.u, norm, device)        # (T,n_y) physical
        # Package metrics auto-skip d.state_initialization_window_length.
        rmse = np.atleast_1d(RMSE(d, y_pred))
        nrmse = np.atleast_1d(NRMSE(d, y_pred))
        r2 = np.atleast_1d(R_squared(d, y_pred))
        mae = np.atleast_1d(MAE(d, y_pred))
        fit = np.atleast_1d(fit_index(d, y_pred))
        rows.append({
            "test_name": d.name or "test",
            "rmse": float(rmse.mean()), "rmse_per_channel": rmse.tolist(),
            "nrmse": float(nrmse.mean()),
            "r2": float(r2.mean()),
            "mae": float(mae.mean()),
            "fit": float(fit.mean()),
            "length": int(len(d)),
            "init_window": int(d.state_initialization_window_length or 0),
        })
        preds.append(y_pred)
    return rows, preds


# ============================================================================
# Animated training monitor
# ============================================================================

class TrainingMonitor:
    """Animated loss + reference-trajectory view (GIF export, optional live window).

    SISO -> one trajectory panel; MIMO -> up to ``max_channels`` channel panels,
    always alongside the train/val loss curve.
    """

    def __init__(self, *, out_dir: Path, bench: BenchmarkData, model_name: str,
                 norm: Standardizer, device: torch.device, plot_every: int = 10,
                 max_channels: int = 3, gif: bool = True, show: bool = False,
                 n_points: int = 1500):
        self.out_dir = out_dir
        self.bench = bench
        self.model_name = model_name
        self.norm = norm
        self.device = device
        self.plot_every = max(1, plot_every)
        self.gif = gif
        self.show = show
        self.frames: List[np.ndarray] = []

        # Preview the first test sequence (visualization only — never tunes anything).
        self.preview = bench.test[0]
        self.target = self.preview.y                      # (T, n_y)
        T = len(self.target)
        self.idx = (np.linspace(0, T - 1, n_points).astype(int)
                    if T > n_points else np.arange(T))
        self.n_show = min(max_channels, bench.n_y)
        self.init_window = self.preview.state_initialization_window_length or 0

        if show:
            for backend in ("MacOSX", "TkAgg", "QtAgg", "Qt5Agg"):
                try:
                    plt.switch_backend(backend)
                    break
                except Exception:
                    continue
            plt.ion()

        self.fig = plt.figure(figsize=(11, 2.4 + 1.9 * self.n_show), constrained_layout=True)
        gs = gridspec.GridSpec(self.n_show + 1, 1, height_ratios=[1.1] * self.n_show + [1.0],
                               figure=self.fig)
        self.traj_axes = [self.fig.add_subplot(gs[i, 0]) for i in range(self.n_show)]
        self.loss_ax = self.fig.add_subplot(gs[self.n_show, 0])
        self.fig.suptitle(f"{bench.name} · {model_name}", fontsize=11)

        self.pred_lines, self.tgt_lines = [], []
        ts = self.idx
        for c, ax in enumerate(self.traj_axes):
            ax.plot(ts, self.target[self.idx, c], color="#f0a202", lw=1.0, label="target")
            (pl,) = ax.plot(ts, np.full_like(ts, np.nan, dtype=float),
                            color="#2364aa", lw=1.1, ls="--", label="prediction")
            self.pred_lines.append(pl)
            if self.init_window > 0:
                ax.axvspan(0, self.init_window, color="#cccccc", alpha=0.3, lw=0)
            ax.set_ylabel(f"$y_{{{c + 1}}}$")
            ax.legend(loc="upper right", fontsize=7, frameon=False)
        self.traj_axes[-1].set_xlabel("time step")

        (self.tloss_line,) = self.loss_ax.plot([], [], color="#2364aa", lw=1.2, label="train")
        (self.vloss_line,) = self.loss_ax.plot([], [], color="#e63946", lw=1.2, ls="--", label="val")
        self.loss_ax.set_yscale("log")
        self.loss_ax.set_xlabel("epoch")
        self.loss_ax.set_ylabel("MSE (norm.)")
        self.loss_ax.legend(loc="upper right", fontsize=7, frameon=False)

    def _capture(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        self.frames.append(buf[..., :3].copy())

    def update(self, epoch: int, train_losses, val_losses, model):
        if (epoch % self.plot_every != 0) and (epoch != 0):
            return
        y_pred = simulate(model, self.preview.u, self.norm, self.device)
        for c, pl in enumerate(self.pred_lines):
            pl.set_ydata(y_pred[self.idx, c])
            ax = self.traj_axes[c]
            lo = min(self.target[:, c].min(), y_pred[:, c].min())
            hi = max(self.target[:, c].max(), y_pred[:, c].max())
            pad = max((hi - lo) * 0.1, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(int(self.idx[0]), int(self.idx[-1]))
        ex = np.arange(len(train_losses))
        self.tloss_line.set_data(ex, train_losses)
        self.vloss_line.set_data(ex, val_losses)
        self.loss_ax.relim()
        self.loss_ax.autoscale_view()
        self.traj_axes[0].set_title(f"epoch {epoch + 1}  ·  val {val_losses[-1]:.3e}",
                                    fontsize=9)
        if self.show:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        if self.gif:
            self._capture()

    def finalize(self) -> Dict[str, str]:
        artifacts = {}
        png = self.out_dir / "training.png"
        self.fig.savefig(png, dpi=130, bbox_inches="tight")
        artifacts["figure"] = str(png)
        if self.gif and self.frames:
            try:
                from PIL import Image
                gif_path = self.out_dir / "training.gif"
                imgs = [Image.fromarray(f) for f in self.frames]
                imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                             duration=120, loop=0, optimize=True)
                artifacts["gif"] = str(gif_path)
            except Exception as exc:  # pragma: no cover
                print(f"    [monitor] GIF export failed: {exc}")
        if self.show:
            plt.ioff()
        plt.close(self.fig)
        return artifacts


# ============================================================================
# Reporting
# ============================================================================

def _fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))


def console_table(results: List[dict]) -> None:
    cols = [("benchmark", 18), ("model", 8), ("params", 9), ("RMSE", 11),
            ("NRMSE", 9), ("fit%", 8), ("R2", 8), ("train s", 9), ("status", 8)]
    header = "  ".join(f"{c:<{w}}" for c, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        row = [
            r["benchmark"][:18], r["model"][:8], str(r.get("n_params", "")),
            _fmt(r.get("rmse")), _fmt(r.get("nrmse")), _fmt(r.get("fit")),
            _fmt(r.get("r2")), _fmt(r.get("train_time_s"), 3), r.get("status", "")[:8],
        ]
        print("  ".join(f"{v:<{w}}" for v, (_, w) in zip(row, cols)))
    print()


def write_reports(results: List[dict], out_dir: Path, meta: dict) -> None:
    (out_dir / "report.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    lines = ["# Nonlinear-benchmarks report", ""]
    lines.append(f"- generated: {meta['timestamp']}")
    lines.append(f"- device: `{meta['device']}`  · dtype: `{meta['dtype']}`  · seed: {meta['seed']}")
    lines.append(f"- epochs: {meta['epochs']}  · batch: {meta['batch_size']}  · "
                 f"seq_len: {meta['seq_len']}  · model dims: {meta['model_dims']}")
    lines.append("")

    by_bench: Dict[str, List[dict]] = {}
    for r in results:
        by_bench.setdefault(r["benchmark"], []).append(r)

    for bench, rows in by_bench.items():
        n_u = rows[0].get("n_u", "?")
        n_y = rows[0].get("n_y", "?")
        lines.append(f"## {bench}  (n_u={n_u}, n_y={n_y})")
        lines.append("")
        lines.append("| model | params | RMSE | NRMSE | fit % | R² | best val | epochs | train s | status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: (x.get("rmse") is None, x.get("rmse", math.inf))):
            lines.append("| {model} | {params} | {rmse} | {nrmse} | {fit} | {r2} | "
                         "{bv} | {ep} | {tt} | {st} |".format(
                model=r["model"], params=r.get("n_params", ""),
                rmse=_fmt(r.get("rmse")), nrmse=_fmt(r.get("nrmse")),
                fit=_fmt(r.get("fit")), r2=_fmt(r.get("r2")),
                bv=_fmt(r.get("best_val_loss")), ep=r.get("epochs_run", ""),
                tt=_fmt(r.get("train_time_s"), 3), st=r.get("status", "")))
        # Per-test-sequence detail when a split has several test signals.
        for r in rows:
            seqs = r.get("test_sequences", [])
            if len(seqs) > 1:
                lines.append("")
                lines.append(f"<details><summary>{r['model']}: per-test-sequence</summary>")
                lines.append("")
                lines.append("| test sequence | RMSE | NRMSE | fit % |")
                lines.append("|---|---|---|---|")
                for s in seqs:
                    lines.append(f"| {s['test_name']} | {_fmt(s['rmse'])} | "
                                 f"{_fmt(s['nrmse'])} | {_fmt(s['fit'])} |")
                lines.append("</details>")
        if rows[0].get("artifacts"):
            lines.append("")
            for r in rows:
                arts = r.get("artifacts", {})
                if "gif" in arts or "figure" in arts:
                    rel = Path(arts.get("gif", arts.get("figure"))).relative_to(out_dir)
                    lines.append(f"- {r['model']}: `{rel}`")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"Reports written to {out_dir}/report.md and report.json")


# ============================================================================
# Orchestration
# ============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(args) -> None:
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    gconf = GlobalModelConfig(
        d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
        d_hidden=args.d_hidden, nl_layers=args.nl_layers,
        raven_heads=args.raven_heads, raven_slots=args.raven_slots, raven_top_k=args.raven_top_k,
    )
    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip, patience=args.patience,
        seq_len=args.seq_len, stride=args.stride, washout=args.washout,
        val_frac=args.val_frac, max_windows=args.max_windows, amp=args.amp, seed=args.seed,
    )

    bench_names = resolve_benchmark_names(args.benchmarks)
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Benchmarks: {bench_names}\nModels: {args.models}\nOutput: {out_root}")

    results: List[dict] = []
    for bname in bench_names:
        print(f"\n=== {bname} ===")
        try:
            bench = load_benchmark(bname)
        except Exception as exc:
            print(f"  [skip] failed to load {bname}: {exc}")
            continue
        print(f"  loaded: n_u={bench.n_u} n_y={bench.n_y} "
              f"train_seqs={len(bench.train)} test_seqs={len(bench.test)} "
              f"init_window={bench.init_window} "
              f"train_len={[len(d) for d in bench.train][:6]}{'...' if len(bench.train) > 6 else ''}")
        norm = Standardizer.fit(bench.train)

        for mname in args.models:
            if mname not in MODEL_ZOO:
                print(f"  [skip] unknown model {mname!r} (have {sorted(MODEL_ZOO)})")
                continue
            print(f"  -- model: {mname}")
            run_dir = out_root / bname / mname
            run_dir.mkdir(parents=True, exist_ok=True)
            rec = {"benchmark": bname, "model": mname, "n_u": bench.n_u, "n_y": bench.n_y}
            try:
                set_seed(args.seed)
                model = build_model(mname, bench.n_u, bench.n_y, gconf)
                rec["n_params"] = sum(p.numel() for p in model.parameters())

                monitor = None
                if not args.no_plot:
                    monitor = TrainingMonitor(
                        out_dir=run_dir, bench=bench, model_name=mname, norm=norm,
                        device=device, plot_every=args.plot_every, gif=not args.no_gif,
                        show=args.show,
                    )

                hist = train_model(model, bench, norm, tcfg, device, monitor)
                rec.update({k: hist[k] for k in
                            ("best_val_loss", "epochs_run", "train_time_s", "n_windows", "window_len")})

                seq_rows, _ = evaluate(model, bench, norm, device)
                rec["test_sequences"] = seq_rows
                rec["rmse"] = float(np.mean([s["rmse"] for s in seq_rows]))
                rec["nrmse"] = float(np.mean([s["nrmse"] for s in seq_rows]))
                rec["r2"] = float(np.mean([s["r2"] for s in seq_rows]))
                rec["mae"] = float(np.mean([s["mae"] for s in seq_rows]))
                rec["fit"] = float(np.mean([s["fit"] for s in seq_rows]))

                diag = model.diagnostics() if hasattr(model, "diagnostics") else None
                if diag is not None:
                    rec["certified_gain_bound"] = diag.get("certified_gain_bound")
                    rec["global_gamma"] = diag.get("global_gamma")

                if monitor is not None:
                    monitor.update(hist["epochs_run"] - 1, hist["train_losses"],
                                   hist["val_losses"], model)
                    rec["artifacts"] = monitor.finalize()

                rec["status"] = "ok"
                print(f"     RMSE={rec['rmse']:.4g}  fit={rec['fit']:.3g}%  "
                      f"params={rec['n_params']}  {hist['epochs_run']} epochs in "
                      f"{hist['train_time_s']:.1f}s")
            except Exception as exc:
                import traceback
                rec["status"] = f"error: {type(exc).__name__}"
                rec["error"] = str(exc)
                print(f"     [error] {type(exc).__name__}: {exc}")
                if args.verbose:
                    traceback.print_exc()
            results.append(rec)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device), "dtype": "float16-amp" if tcfg.amp else "float32",
        "seed": args.seed, "epochs": tcfg.epochs, "batch_size": tcfg.batch_size,
        "seq_len": tcfg.seq_len,
        "model_dims": f"d_model={gconf.d_model},d_state={gconf.d_state},n_layers={gconf.n_layers}",
    }
    console_table(results)
    write_reports(results, out_root, meta)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark repo models on the nonlinear_benchmarks suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--benchmarks", nargs="+", default=["Cascaded_Tanks"],
                   help="benchmark names and/or groups: " + ", ".join(sorted(_GROUPS)))
    p.add_argument("--models", nargs="+", default=["raven", "tv", "lstm"],
                   help="model names: " + ", ".join(sorted(MODEL_ZOO)))
    p.add_argument("--out", default=str(_REPO_ROOT / "Test_files" / "benchmark_runs"))
    p.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--washout", type=int, default=50)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-windows", type=int, default=4000)
    p.add_argument("--amp", action="store_true", help="cuda mixed precision (loosens certified caps)")
    p.add_argument("--seed", type=int, default=0)
    # model dims
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-hidden", type=int, default=16)
    p.add_argument("--nl-layers", type=int, default=3)
    p.add_argument("--raven-heads", type=int, default=4)
    p.add_argument("--raven-slots", type=int, default=8)
    p.add_argument("--raven-top-k", type=int, default=2)
    # plotting
    p.add_argument("--no-plot", action="store_true", help="disable the training monitor entirely")
    p.add_argument("--no-gif", action="store_true", help="disable GIF export (keep final PNG)")
    p.add_argument("--show", action="store_true", help="show a live interactive window")
    p.add_argument("--plot-every", type=int, default=10)
    # misc
    p.add_argument("--list", action="store_true", help="list benchmarks/models and exit")
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.list:
        print("Benchmarks:")
        for n in _ALL_LOADERS:
            tag = " (MIMO)" if n == "Industrial_robot" else ""
            print(f"  {n}{tag}")
        print("Groups:", {g: v for g, v in _GROUPS.items()})
        print("Models:", list(MODEL_ZOO))
        return
    set_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
