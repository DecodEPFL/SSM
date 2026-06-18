#!/usr/bin/env python
# file: Test_files/run_benchmarks.py
"""Run repo models against the ``nonlinear_benchmarks`` system-identification suite.

This is a self-contained benchmark harness:

* loads any subset of the ``nonlinear_benchmarks`` datasets (SISO *and* MIMO),
  handling the tuple-of-datasets splits (CED, Silverbox, ParWH, F16, …) and the
  not-splitted MIMO ``Industrial_robot`` benchmark;
* trains models in **open-loop simulation** mode on every complete training
  trajectory, without temporal windows or a derived holdout split;
* validates with the benchmark-provided second split (free-run simulation, skipping each
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
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Headless-safe by default; the live monitor upgrades to a GUI backend on request.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Let MPS fall back unsupported ops (e.g. the SVD behind the certified spectral
# norms) to CPU rather than crashing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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

from src.neural_ssm.ssm import DeepSSM, SSMConfig, CertifiedTransformer


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
# Not-splitted loaders return (train_list, validation_list) but lack those kwargs.
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
    validation: List[Input_output_data]
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
    """Load native splits, treating the loader's second split as validation.

    The upstream package commonly calls this second split ``test``. This harness
    intentionally uses it for validation/checkpoint selection as an experiment
    protocol, rather than claiming strict benchmark-submission semantics.
    """
    if name in _SPLITTED:
        train, validation = _SPLITTED[name](
            atleast_2d=True, always_return_tuples_of_datasets=True
        )
        train_list = _as_dataset_list(train)
        validation_list = _as_dataset_list(validation)
    elif name in _NOT_SPLITTED:
        train, validation = _NOT_SPLITTED[name](train_test_split=True)
        train_list = [d.atleast_2d() for d in _as_dataset_list(train)]
        validation_list = [d.atleast_2d() for d in _as_dataset_list(validation)]
    else:
        raise KeyError(f"Unknown benchmark {name!r}. Available: {sorted(_ALL_LOADERS)}")

    if not train_list or not validation_list:
        raise RuntimeError(f"Benchmark {name!r} produced an empty train/validation split.")

    n_u = int(train_list[0].u.shape[-1])
    n_y = int(train_list[0].y.shape[-1])
    init_window = validation_list[0].state_initialization_window_length or 0
    return BenchmarkData(
        name=name,
        train=train_list,
        validation=validation_list,
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
# Complete-sequence batching for open-loop simulation training
# ============================================================================

NormalizedSequence = Tuple[np.ndarray, np.ndarray, int]


def pack_full_sequences(
    sequences: Sequence[NormalizedSequence],
    indices: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad complete trajectories for one batch and mask padding/initialization only."""
    selected = [sequences[int(i)] for i in indices]
    if not selected:
        raise ValueError("cannot pack an empty sequence batch")

    max_len = max(len(u) for u, _, _ in selected)
    n_u = selected[0][0].shape[-1]
    n_y = selected[0][1].shape[-1]
    U = torch.zeros((len(selected), max_len, n_u), dtype=torch.float32, device=device)
    Y = torch.zeros((len(selected), max_len, n_y), dtype=torch.float32, device=device)
    M = torch.zeros((len(selected), max_len), dtype=torch.float32, device=device)

    for row, (u, y, loss_start) in enumerate(selected):
        if len(u) != len(y):
            raise ValueError("input and output trajectory lengths differ")
        length = len(u)
        start = min(max(int(loss_start), 0), length)
        U[row, :length] = torch.as_tensor(u, dtype=torch.float32, device=device)
        Y[row, :length] = torch.as_tensor(y, dtype=torch.float32, device=device)
        M[row, start:length] = 1.0
    return U, Y, M


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
    "l2n":   dict(kind="deepssm", param="l2n",   gamma=5.0,  ff="LGLU2"),   # certified 2x2 dense LTI (needs even d_state)
    "tv":    dict(kind="deepssm", param="tv",    gamma=5.0,  ff="MBLIP"),   # certified selective
    "tvc":   dict(kind="deepssm", param="tvc",   gamma=5.0,  ff="MBLIP"),   # certified selective LTI
    "raven": dict(kind="deepssm", param="raven", gamma=5.0,  ff="MBLIP"),   # certified Raven cell
    "ctransformer": dict(kind="ctransformer", gamma_total=20.0),            # certified softmax transformer (conservative attn bound -> needs more budget)
    "lstm":  dict(kind="lstm",    hidden=64, layers=2),
    "gru":   dict(kind="gru",     hidden=64, layers=2),
}


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _effective_width(model: nn.Module) -> Optional[int]:
    """Return the width knob changed by parameter matching, when available."""
    core = getattr(model, "core", None)
    if core is not None:
        return int(core.config.d_model)
    rnn = getattr(model, "rnn", None)
    if rnn is not None:
        return int(rnn.hidden_size)
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "d_out"):
        return int(encoder.d_out)
    return None


def _make_ssm_config(spec: dict, gconf: "GlobalModelConfig") -> SSMConfig:
    gamma = spec.get("gamma")
    if str(gconf.gamma_override).lower() != "auto":
        ov = str(gconf.gamma_override).lower()
        gamma = None if ov in ("none", "") else float(gconf.gamma_override)
    ffo = gconf.ff_override
    ff = spec.get("ff", "LGLU2") if (not ffo or str(ffo).lower() == "auto") else ffo
    return SSMConfig(
        d_model=gconf.d_model, d_state=gconf.d_state, n_layers=gconf.n_layers,
        d_hidden=gconf.d_hidden, nl_layers=gconf.nl_layers, scale=gconf.scale,
        param=spec["param"], gamma=gamma, ff=ff, init=gconf.init,
        train_gamma=True, learn_x0=False,
        rmin=gconf.lru_rmin, rmax=gconf.lru_rmax, max_phase=gconf.lru_max_phase,
        l2ru_eye_scale=gconf.l2ru_eye_scale, l2ru_rand_scale=gconf.l2ru_rand_scale,
        rho=gconf.l2n_rho, max_phase_b=gconf.l2n_max_phase,
        phase_center=gconf.l2n_phase_center, random_phase=gconf.l2n_random_phase,
        offdiag_scale=gconf.l2n_offdiag_scale,
        tv_init_rho=gconf.tv_init_rho, tv_init_delta0=gconf.tv_init_delta0,
        tv_init_param_scale=gconf.tv_init_param_scale,
        tvc_init_rho=gconf.tvc_init_rho, tvc_init_delta0=gconf.tvc_init_delta0,
        tvc_init_param_scale=gconf.tvc_init_param_scale,
        tvc_init_sign=gconf.tvc_init_sign,
        tvc_init_b=gconf.tvc_init_b, tvc_init_c=gconf.tvc_init_c,
        tvc_init_d=gconf.tvc_init_d,
        raven_heads=gconf.raven_heads, raven_slots=gconf.raven_slots, raven_top_k=gconf.raven_top_k,
        per_channel_gates=gconf.per_channel_gates,
    )


def _closest_width(count_fn, target: int, lo: int = 2, cap: int = 4096) -> int:
    """Width whose model param count is closest to ``target``.

    ``count_fn(w)`` builds a model of width ``w`` and returns its trainable param
    count (or None if infeasible). Param count is ~monotonic in width, so bracket
    the target then check neighbours.
    """
    def c(w):
        try:
            return count_fn(int(w))
        except Exception:
            return None

    hi, chi = max(lo, 4), None
    chi = c(max(lo, 4))
    while (chi is None or chi < target) and hi < cap:
        hi *= 2
        chi = c(hi)
    a, b = lo, hi
    while a < b:
        mid = (a + b) // 2
        cm = c(mid)
        if cm is None or cm < target:
            a = mid + 1
        else:
            b = mid
    best, best_err = None, None
    for w in (a - 1, a, a + 1):
        if w < lo:
            continue
        cw = c(w)
        if cw is None:
            continue
        err = abs(cw - target)
        if best is None or err < best_err:
            best, best_err = w, err
    return best if best is not None else lo


def build_model(name: str, n_u: int, n_y: int, gconf: "GlobalModelConfig",
                param_budget: int = 0) -> nn.Module:
    """Build a model. With ``param_budget > 0`` the width knob is tuned so the
    trainable parameter count lands near the budget (d_model for DeepSSM cores,
    hidden size for the RNN baselines); other hyperparameters are left as set."""
    spec = dict(MODEL_ZOO[name])
    kind = spec.pop("kind")

    if kind == "deepssm":
        if param_budget and param_budget > 0:
            # Width probes construct temporary modules. Isolate their RNG use so
            # parameter matching does not change the final seeded initialization.
            with torch.random.fork_rng(devices=[]):
                best = _closest_width(
                    lambda w: _count_params(
                        DeepSSMSim(n_u, n_y, _make_ssm_config(spec, replace(gconf, d_model=w)))),
                    param_budget, lo=max(2, gconf.raven_heads),
                )
            gconf = replace(gconf, d_model=best)
        if spec.get("param") == "l2n" and gconf.d_state % 2 != 0:
            raise ValueError(
                f"param='l2n' (Block2x2DenseL2SSM) needs an even d_state (2x2 blocks); "
                f"got {gconf.d_state}. Pass an even --d-state (e.g. {gconf.d_state + 1})."
            )
        return DeepSSMSim(n_u, n_y, _make_ssm_config(spec, gconf))

    if kind in ("lstm", "gru"):
        if param_budget and param_budget > 0:
            with torch.random.fork_rng(devices=[]):
                spec["hidden"] = _closest_width(
                    lambda w: _count_params(
                        RNNSim(n_u, n_y, kind=kind, hidden=w, layers=spec["layers"])
                    ),
                    param_budget,
                )
        return RNNSim(n_u, n_y, kind=kind, hidden=spec["hidden"], layers=spec["layers"])

    if kind == "ctransformer":
        gt = float(spec.get("gamma_total", 5.0))
        ov = str(gconf.gamma_override).lower()
        if ov not in ("auto", "none", ""):
            gt = float(gconf.gamma_override)

        def _make_ct(dm):
            # Causal: simulation must not peek at future inputs. FFN width scales with
            # d_model (avoids a bottleneck when param-matching inflates d_model).
            return CertifiedTransformer(
                d_input=n_u, d_model=int(dm), d_output=n_y,
                n_layers=max(1, gconf.n_layers), n_heads=gconf.raven_heads,
                d_ff=max(int(gconf.d_hidden), 2 * int(dm)), gamma_total=gt, causal=True,
                max_len=16384,
            )

        if param_budget and param_budget > 0:
            with torch.random.fork_rng(devices=[]):
                best = _closest_width(lambda w: _count_params(_make_ct(w)), param_budget,
                                      lo=max(2, gconf.raven_heads))
            return _make_ct(best)
        return _make_ct(gconf.d_model)

    raise ValueError(f"Unknown model kind {kind!r}.")


def resolve_param_budget(value: object, n_u: int, n_y: int,
                         gconf: "GlobalModelConfig") -> Tuple[int, str]:
    """Resolve an integer/off/model-name parameter-budget CLI value.

    A model name means "use that model's unscaled trainable parameter count".
    This is resolved per benchmark because input/output dimensions affect it.
    """
    token = str(value).strip().lower()
    if token in ("", "0", "off", "none"):
        return 0, "off"
    if token in MODEL_ZOO:
        reference = build_model(token, n_u, n_y, gconf, param_budget=0)
        return _count_params(reference), f"reference:{token}"
    try:
        budget = int(token)
    except ValueError as exc:
        raise ValueError(
            "--param-budget must be a positive integer, 'off', or a model name "
            f"from {sorted(MODEL_ZOO)}; got {value!r}."
        ) from exc
    if budget <= 0:
        raise ValueError(f"--param-budget must be positive or 'off'; got {value!r}.")
    return budget, "fixed"


@dataclass
class GlobalModelConfig:
    d_model: int = 8
    d_state: int = 8
    n_layers: int = 4
    d_hidden: int = 16
    nl_layers: int = 3
    scale: float = 1.0
    raven_heads: int = 4
    raven_slots: int = 8
    raven_top_k: int = 2
    per_channel_gates: bool = False
    lru_rmin: float = 0.8
    lru_rmax: float = 0.95
    lru_max_phase: float = 2.0 * math.pi
    l2ru_eye_scale: float = 0.01
    l2ru_rand_scale: float = 1.0
    l2n_rho: float = 0.9
    l2n_max_phase: float = 0.04
    l2n_phase_center: float = 0.0
    l2n_random_phase: bool = True
    l2n_offdiag_scale: float = 0.05
    tv_init_rho: float = 0.99
    tv_init_delta0: float = 1.0
    tv_init_param_scale: float = 0.02
    tvc_init_rho: float = 0.9
    tvc_init_delta0: float = 1.0
    tvc_init_param_scale: float = 0.02
    tvc_init_sign: float = 0.995
    tvc_init_b: float = 0.10
    tvc_init_c: float = 0.10
    tvc_init_d: float = 0.10
    init: str = "eye"                      # recurrent-cell init for l2ru/zak/...
    gamma_override: str = "none"           # "auto" (per-model), "none", or a float string
    ff_override: Optional[str] = "GLU"      # override the per-model feedforward


def _validate_initialization(models: Sequence[str], cfg: GlobalModelConfig) -> None:
    """Validate only the initialization settings used by selected models."""
    selected = set(models)

    def finite(**values):
        for option, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"--{option.replace('_', '-')} must be finite, got {value}.")

    if "lru" in selected:
        finite(lru_rmin=cfg.lru_rmin, lru_rmax=cfg.lru_rmax,
               lru_max_phase=cfg.lru_max_phase)
        if not 0.0 < cfg.lru_rmin < cfg.lru_rmax < 1.0:
            raise ValueError("LRU requires 0 < --lru-rmin < --lru-rmax < 1.")
        if cfg.lru_max_phase <= 0.0:
            raise ValueError("--lru-max-phase must be positive.")

    if "l2ru" in selected:
        finite(l2ru_eye_scale=cfg.l2ru_eye_scale,
               l2ru_rand_scale=cfg.l2ru_rand_scale)
        if cfg.init not in ("eye", "rand"):
            raise ValueError("--init must be 'eye' or 'rand' for L2RU.")
        if cfg.l2ru_eye_scale < 0.0 or cfg.l2ru_rand_scale < 0.0:
            raise ValueError("L2RU initialization scales must be non-negative.")

    if "l2n" in selected:
        finite(l2n_rho=cfg.l2n_rho, l2n_max_phase=cfg.l2n_max_phase,
               l2n_phase_center=cfg.l2n_phase_center,
               l2n_offdiag_scale=cfg.l2n_offdiag_scale)
        if not 0.0 < cfg.l2n_rho < 1.0:
            raise ValueError("--l2n-rho must be in (0, 1).")
        if cfg.l2n_max_phase < 0.0 or cfg.l2n_offdiag_scale < 0.0:
            raise ValueError("L2N phase width and off-diagonal scale must be non-negative.")

    if "tv" in selected:
        finite(tv_init_rho=cfg.tv_init_rho, tv_init_delta0=cfg.tv_init_delta0,
               tv_init_param_scale=cfg.tv_init_param_scale)
        if not 0.0 < cfg.tv_init_rho < 1.0:
            raise ValueError("--tv-init-rho must be in (0, 1).")
        if cfg.tv_init_delta0 <= 0.0 or cfg.tv_init_param_scale < 0.0:
            raise ValueError("TV delta0 must be positive and parameter scale non-negative.")

    if "tvc" in selected:
        finite(tvc_init_rho=cfg.tvc_init_rho, tvc_init_delta0=cfg.tvc_init_delta0,
               tvc_init_param_scale=cfg.tvc_init_param_scale,
               tvc_init_sign=cfg.tvc_init_sign, tvc_init_b=cfg.tvc_init_b,
               tvc_init_c=cfg.tvc_init_c, tvc_init_d=cfg.tvc_init_d)
        if not 0.0 < cfg.tvc_init_rho < 1.0:
            raise ValueError("--tvc-init-rho must be in (0, 1).")
        if cfg.tvc_init_delta0 <= 0.0 or cfg.tvc_init_param_scale < 0.0:
            raise ValueError("TVC delta0 must be positive and parameter scale non-negative.")
        if not -1.0 < cfg.tvc_init_sign < 1.0:
            raise ValueError("--tvc-init-sign must be strictly between -1 and 1.")
        if any(abs(v) > 0.99 for v in (cfg.tvc_init_b, cfg.tvc_init_c, cfg.tvc_init_d)):
            raise ValueError("TVC initial b, c, and d must each be in [-0.99, 0.99].")


def _initialization_metadata(cfg: GlobalModelConfig) -> dict:
    return {
        "lru": {"rmin": cfg.lru_rmin, "rmax": cfg.lru_rmax,
                "max_phase": cfg.lru_max_phase},
        "l2ru": {"mode": cfg.init, "eye_scale": cfg.l2ru_eye_scale,
                 "rand_scale": cfg.l2ru_rand_scale},
        "l2n": {"rho": cfg.l2n_rho, "max_phase": cfg.l2n_max_phase,
                "phase_center": cfg.l2n_phase_center,
                "random_phase": cfg.l2n_random_phase,
                "offdiag_scale": cfg.l2n_offdiag_scale},
        "tv": {"rho": cfg.tv_init_rho, "delta0": cfg.tv_init_delta0,
               "param_scale": cfg.tv_init_param_scale},
        "tvc": {"rho": cfg.tvc_init_rho, "delta0": cfg.tvc_init_delta0,
                "param_scale": cfg.tvc_init_param_scale,
                "sign": cfg.tvc_init_sign, "b": cfg.tvc_init_b,
                "c": cfg.tvc_init_c, "d": cfg.tvc_init_d},
    }


# ============================================================================
# Device / training
# ============================================================================

def _make_grad_scaler(enabled: bool):
    """Construct a GradScaler across torch versions.

    torch>=2.4 exposes ``torch.amp.GradScaler``; older versions only have
    ``torch.cuda.amp.GradScaler`` (which exists even on CPU-only builds and is a
    no-op when ``enabled=False``).
    """
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - signature drift
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
    amp: bool = False
    seed: int = 0


def train_model(
    model: nn.Module,
    bench: BenchmarkData,
    norm: Standardizer,
    tcfg: TrainConfig,
    device: torch.device,
    monitor: "Optional[TrainingMonitor]" = None,
) -> dict:
    """Open-loop training on complete native trajectories for the full schedule.

    The loader-provided second-split trajectories are never sliced or used for
    optimization. Validation loss selects the returned checkpoint.
    """
    model.to(device)
    rng = np.random.default_rng(tcfg.seed)

    def _norm_pair(d, *, use_initialization_window: bool) -> NormalizedSequence:
        u = norm.norm_u(torch.from_numpy(d.u.astype(np.float32))).numpy()
        y = norm.norm_y(torch.from_numpy(d.y.astype(np.float32))).numpy()
        loss_start = int(d.state_initialization_window_length or 0) if use_initialization_window else 0
        return u, y, loss_start

    # Every training sample contributes to optimization. For validation, the
    # complete signal initializes the state while the official prefix is masked
    # from the metric, matching nonlinear_benchmarks.error_metrics.
    train_seqs = [_norm_pair(d, use_initialization_window=False) for d in bench.train]
    val_seqs = [
        _norm_pair(d, use_initialization_window=True) for d in bench.validation
    ]

    n_y = bench.n_y
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(tcfg.epochs, 1))
    use_amp = tcfg.amp and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    def masked_mse(pred, tgt, mask):
        diff2 = (pred - tgt) ** 2 * mask.unsqueeze(-1)
        return diff2.sum() / (mask.sum().clamp_min(1.0) * n_y)

    @torch.no_grad()
    def eval_sequences(sequences):
        model.eval()
        tot = torch.zeros((), device=device)
        denom = torch.zeros((), device=device)
        for i in range(0, len(sequences), tcfg.batch_size):
            batch_indices = range(i, min(i + tcfg.batch_size, len(sequences)))
            U, Y, M = pack_full_sequences(sequences, batch_indices, device)
            pred = model(U)
            tot = tot + (((pred - Y) ** 2) * M.unsqueeze(-1)).sum()
            denom = denom + M.sum() * n_y
        denom = denom.clamp_min(1.0)
        return (tot / denom).item()

    n_train = len(train_seqs)
    train_losses, val_losses = [], []
    best_val, best_state, best_epoch = float("inf"), None, 0
    t0 = time.time()

    for epoch in range(tcfg.epochs):
        model.train()
        perm = rng.permutation(n_train)
        running = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        for i in range(0, n_train, tcfg.batch_size):
            idx = perm[i:i + tcfg.batch_size]
            Ut, Yt, Mt = pack_full_sequences(train_seqs, idx, device)
            opt.zero_grad(set_to_none=True)
            # autocast only when AMP is active (CUDA). Avoid passing 'mps'/'cpu' to
            # torch.autocast, which can raise on some torch builds even when disabled.
            amp_ctx = torch.autocast(device_type="cuda") if use_amp else contextlib.nullcontext()
            with amp_ctx:
                pred = model(Ut)
                loss = masked_mse(pred, Yt, Mt)
            scaler.scale(loss).backward()
            if tcfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            valid_values = Mt.sum() * n_y
            running = running + loss.detach() * valid_values
            seen = seen + valid_values
        sched.step()
        train_loss = (running / seen.clamp_min(1.0)).item()
        train_losses.append(train_loss)

        val_loss = eval_sequences(val_seqs)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if monitor is not None:
            monitor.update(epoch, train_losses, val_losses, model)

    # No early stopping (full schedule runs), but restore the best-validation checkpoint.
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val if best_state is not None else float("inf"),
        "best_epoch": best_epoch + 1 if best_state is not None else 0,
        "final_val_loss": val_losses[-1] if val_losses else float("inf"),
        "epochs_run": len(train_losses),
        "train_time_s": time.time() - t0,
        "n_train_sequences": int(n_train),
        "n_validation_sequences": int(len(val_seqs)),
        "train_samples": int(sum(len(u) for u, _, _ in train_seqs)),
        "validation_samples": int(sum(len(u) for u, _, _ in val_seqs)),
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_validation(model: nn.Module, bench: BenchmarkData, norm: Standardizer,
                        device: torch.device) -> Tuple[List[dict], List[np.ndarray]]:
    """Per-validation-trajectory metrics in physical units, plus predictions."""
    rows, preds = [], []
    for d in bench.validation:
        y_pred = simulate(model, d.u, norm, device)        # (T,n_y) physical
        # Package metrics auto-skip d.state_initialization_window_length.
        rmse = np.atleast_1d(RMSE(d, y_pred))
        nrmse = np.atleast_1d(NRMSE(d, y_pred))
        r2 = np.atleast_1d(R_squared(d, y_pred))
        mae = np.atleast_1d(MAE(d, y_pred))
        fit = np.atleast_1d(fit_index(d, y_pred))
        rows.append({
            "validation_name": d.name or "validation",
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

# Backends that cannot show a live window. NB: TkAgg/QtAgg/GTK3Agg ARE interactive
# even though their names contain "agg" — only these exact names are non-interactive.
_NONINTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}


def _is_interactive_backend() -> bool:
    return matplotlib.get_backend().lower() not in _NONINTERACTIVE_BACKENDS

class TrainingMonitor:
    """Live + GIF training animation in the Benchmark.py house style.

    Dark theme, glow lines, blitted + tweened updates, a loss-trajectory panel
    with the best-val marker, and epoch/stats overlays. SISO shows a training-
    and a validation-sequence prediction panel; MIMO shows up to
    ``max_channels`` per-output-channel panels of the validation sequence. Each update
    also captures a frame so the same animation is exported as a GIF (headless).
    """

    _PALETTE = ("#5ec8e5", "#ff7a90", "#9bffb0", "#f6bd60", "#b8a1ff", "#ff9f1c")

    def __init__(self, *, out_dir: Path, bench: BenchmarkData, model_name: str,
                 norm: Standardizer, device: torch.device, total_epochs: int = 200,
                 plot_every: int = 10, gif: bool = True, show: bool = False,
                 max_channels: int = 3, n_points: int = 1200,
                 transition_frames: int = 5, transition_pause: float = 0.0015,
                 max_preview_len: int = 4000):
        self.out_dir = out_dir
        self.norm = norm
        self.device = device
        self.plot_every = max(1, plot_every)
        if max_preview_len <= 0:
            raise ValueError("max_preview_len must be positive")
        self.max_preview_len = int(max_preview_len)
        self.gif = gif
        self.show = show
        self.total_epochs = max(1, total_epochs)
        self.transition_frames = max(1, transition_frames)
        self.transition_pause = transition_pause
        self.frames: List[np.ndarray] = []
        self._loss_ylim = (np.inf, -np.inf)
        self._prev: Optional[List[np.ndarray]] = None
        self._active = False

        def _slice(d):
            u = np.asarray(d.u, dtype=np.float32)
            y = np.asarray(d.y, dtype=np.float32)
            if len(u) > self.max_preview_len:
                u, y = u[:self.max_preview_len], y[:self.max_preview_len]
            return u, y

        # Pick preview sequences + panels: SISO = train+validation, MIMO = validation channels.
        val_u, val_y = _slice(bench.validation[0])
        if bench.n_y == 1:
            train_u, train_y = _slice(bench.train[0])
            self._previews = {"train": (train_u, train_y), "validation": (val_u, val_y)}
            self._panels = [
                dict(source="train", channel=0, title="Training sequence",
                     color=self._PALETTE[0], init_window=0),
                dict(source="validation", channel=0, title="Validation sequence",
                     color=self._PALETTE[1], init_window=bench.init_window),
            ]
            self._layout = "pair"
        else:
            self._previews = {"validation": (val_u, val_y)}
            n_show = min(max_channels, bench.n_y)
            self._panels = [
                dict(source="validation", channel=c, title=f"output $y_{{{c + 1}}}$",
                     color=self._PALETTE[c % len(self._PALETTE)],
                     init_window=bench.init_window)
                for c in range(n_show)
            ]
            self._layout = "stack"

        self._idx = {}
        for name, (u, y) in self._previews.items():
            T = len(y)
            self._idx[name] = (np.linspace(0, T - 1, n_points).astype(int)
                               if T > n_points else np.arange(T))
        for p in self._panels:
            idx = self._idx[p["source"]]
            yt = self._previews[p["source"]][1][idx, p["channel"]]
            lo, hi = float(yt.min()), float(yt.max())
            margin = max((hi - lo) * 0.12, 1e-4)
            p["idx"], p["target"], p["ylim"] = idx, yt, (lo - margin, hi + margin)

        if self.show:
            chosen = None
            # If the current backend is already interactive, keep it; else try GUI ones.
            candidates = [matplotlib.get_backend(), "MacOSX", "TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg"]
            for backend in candidates:
                try:
                    plt.switch_backend(backend)
                except Exception:
                    continue
                if _is_interactive_backend():
                    chosen = matplotlib.get_backend()
                    break
            if chosen is None:
                print("    [monitor] no interactive GUI backend available (install tkinter or "
                      "PyQt for a live window); saving GIF/PNG only.")
                self.show = False
            else:
                print(f"    [monitor] live window via '{chosen}' backend.")

        try:
            if self.show:
                plt.ion()
            self._build_figure(bench, model_name)
            self.fig.canvas.draw()
            self._safe_flush()
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self.fig.canvas.mpl_connect("resize_event", lambda _e: self._redraw_background())
            if self.show:
                try:
                    self.fig.show()            # actually raise the window
                    plt.pause(0.05)            # let the window manager create it
                except Exception as exc:
                    print(f"    [monitor] could not raise the live window ({exc}); "
                          "saving GIF/PNG only.")
                    self.show = False
            self._active = True
        except Exception as exc:
            print(f"    [monitor] could not initialise plot: {exc}")
            self._active = False

    # ---- figure construction --------------------------------------------------
    def _style_axis(self, ax):
        ax.set_facecolor("#111925")
        ax.grid(True, color="#425066", alpha=0.24, linewidth=0.6)
        ax.tick_params(colors="#d7dde8", labelsize=8)
        ax.xaxis.label.set_color("#eef2f7")
        ax.yaxis.label.set_color("#eef2f7")
        ax.title.set_color("#f6f7fb")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#5d6c80")

    def _build_figure(self, bench, model_name):
        n = len(self._panels)
        if self._layout == "pair":
            self.fig = plt.figure(figsize=(13, 6))
            gs = self.fig.add_gridspec(2, 2, height_ratios=[2.2, 1.0], hspace=0.3, wspace=0.18)
            self._panel_axes = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])]
            self.loss_ax = self.fig.add_subplot(gs[1, :])
            self.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.07, right=0.97)
        else:
            self.fig = plt.figure(figsize=(12, 2.2 + 2.0 * n))
            gs = self.fig.add_gridspec(n + 1, 1, height_ratios=[2.0] * n + [1.3], hspace=0.5)
            self._panel_axes = [self.fig.add_subplot(gs[i, 0]) for i in range(n)]
            self.loss_ax = self.fig.add_subplot(gs[n, 0])
            self.fig.subplots_adjust(top=0.94, bottom=0.07, left=0.08, right=0.97)

        self.fig.patch.set_facecolor("#0b1118")
        self.fig.suptitle(f"Live Training · {bench.name} · {model_name}",
                          fontsize=11, color="#f6f7fb")

        self._glow, self._pred = [], []
        for p, ax in zip(self._panels, self._panel_axes):
            self._style_axis(ax)
            ax.set_title(p["title"], fontsize=10)
            ax.set_ylabel("output")
            ax.set_xlim(int(p["idx"][0]), int(p["idx"][-1]))
            ax.set_ylim(*p["ylim"])
            ax.plot(p["idx"], p["target"], color="#f6bd60", lw=1.1, alpha=0.9, label="target")
            if p["init_window"] > 0:
                ax.axvspan(0, p["init_window"], color="#d7dde8", alpha=0.08, lw=0)
            nan = np.full_like(p["target"], np.nan)
            g, = ax.plot(p["idx"], nan, color=p["color"], lw=4.5, alpha=0.14,
                         animated=True, solid_capstyle="round")
            l, = ax.plot(p["idx"], nan, color=p["color"], lw=1.8, alpha=0.97,
                         animated=True, solid_capstyle="round", label="prediction")
            self._glow.append(g)
            self._pred.append(l)
            ax.legend(fontsize=7, loc="upper right", facecolor="#111925",
                      edgecolor="#334155", labelcolor="#eef2f7")
        self._panel_axes[-1].set_xlabel("time step")

        self._style_axis(self.loss_ax)
        self.loss_ax.set_title("Loss trajectory", fontsize=10)
        self.loss_ax.set_xlabel("epoch")
        self.loss_ax.set_ylabel("MSE (norm.)")
        self.loss_ax.set_yscale("log")
        self.loss_ax.set_xlim(0, max(1, self.total_epochs - 1))
        self._tl_glow, = self.loss_ax.plot([], [], color="#5ec8e5", lw=4.0, alpha=0.12, animated=True)
        self._tl, = self.loss_ax.plot([], [], color="#5ec8e5", lw=1.3, animated=True, label="train")
        self._vl_glow, = self.loss_ax.plot([], [], color="#ff7a90", lw=4.0, alpha=0.12, animated=True)
        self._vl, = self.loss_ax.plot([], [], color="#ff7a90", lw=1.3, ls="--", animated=True, label="val")
        self._best, = self.loss_ax.plot([], [], ls="None", marker="o", ms=5.5, color="#9bffb0",
                                        markeredgecolor="#ffffff", markeredgewidth=0.5,
                                        animated=True, label="best val")
        self.loss_ax.legend(fontsize=7, loc="upper right", facecolor="#111925",
                            edgecolor="#334155", labelcolor="#eef2f7")
        self._epoch_text = self._panel_axes[0].text(
            0.02, 0.96, "", transform=self._panel_axes[0].transAxes, fontsize=9, va="top",
            color="#e8edf6", animated=True,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#101826", edgecolor="none", alpha=0.82))
        self._stats_text = self.loss_ax.text(
            0.015, 0.94, "", transform=self.loss_ax.transAxes, fontsize=8, va="top",
            color="#e8edf6", animated=True,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#101826", edgecolor="none", alpha=0.82))

    # ---- blitting helpers ------------------------------------------------------
    def _safe_flush(self):
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def _redraw_background(self):
        self.fig.canvas.draw()
        self._safe_flush()
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _animated_artists(self):
        return (self._glow + self._pred +
                [self._tl_glow, self._tl, self._vl_glow, self._vl, self._best,
                 self._epoch_text, self._stats_text])

    def _draw_artists(self, preds, epoch, train_losses, val_losses):
        self.fig.canvas.restore_region(self._bg)
        for i, (g, l) in enumerate(zip(self._glow, self._pred)):
            g.set_ydata(preds[i])
            l.set_ydata(preds[i])
            self._panel_axes[i].draw_artist(g)
            self._panel_axes[i].draw_artist(l)
        ex = np.arange(len(train_losses))
        self._tl_glow.set_data(ex, train_losses)
        self._tl.set_data(ex, train_losses)
        self._vl_glow.set_data(ex, val_losses)
        self._vl.set_data(ex, val_losses)
        best_e = int(np.argmin(val_losses)) if val_losses else 0
        best_v = float(val_losses[best_e]) if val_losses else float("nan")
        self._best.set_data([best_e], [best_v])
        self._epoch_text.set_text(f"epoch {epoch + 1}/{self.total_epochs}")
        self._stats_text.set_text(
            f"train {train_losses[-1]:.2e}   val {val_losses[-1]:.2e}\n"
            f"best {best_v:.2e} @ epoch {best_e + 1}")
        for art in (self._tl_glow, self._tl, self._vl_glow, self._vl, self._best):
            self.loss_ax.draw_artist(art)
        self._panel_axes[0].draw_artist(self._epoch_text)
        self.loss_ax.draw_artist(self._stats_text)
        self.fig.canvas.blit(self.fig.bbox)
        self._safe_flush()

    def _pause(self):
        if self.transition_pause <= 0:
            return
        try:
            self.fig.canvas.start_event_loop(self.transition_pause)
        except Exception:
            try:
                plt.pause(self.transition_pause)
            except Exception:
                pass

    def _capture(self):
        try:
            buf = np.asarray(self.fig.canvas.buffer_rgba())
            self.frames.append(buf[..., :3].copy())
        except Exception:
            print("    [monitor] frame capture unavailable on this backend; GIF disabled.")
            self.gif = False

    # ---- public API ------------------------------------------------------------
    def update(self, epoch: int, train_losses, val_losses, model, force: bool = False):
        if not self._active:
            return
        if not force and (epoch % self.plot_every != 0) and (epoch != 0):
            return

        preds = {name: simulate(model, u, self.norm, self.device)
                 for name, (u, _y) in self._previews.items()}
        currents = [preds[p["source"]][p["idx"], p["channel"]] for p in self._panels]

        # Expand the (log) loss y-range only; otherwise the cached background stays valid.
        finite = [v for v in list(train_losses) + list(val_losses) if np.isfinite(v) and v > 0]
        if finite:
            lo, hi = min(finite) * 0.5, max(finite) * 2.0
            clo, chi = self._loss_ylim
            if lo < clo or hi > chi:
                self._loss_ylim = (min(clo, lo), max(chi, hi))
                self.loss_ax.set_ylim(*self._loss_ylim)
                self._redraw_background()

        if self.show and self._prev is not None:
            for f in range(1, self.transition_frames + 1):
                a = f / self.transition_frames
                blended = [
                    self._prev[i] + a * (currents[i] - self._prev[i])
                    if self._prev[i].shape == currents[i].shape else currents[i]
                    for i in range(len(currents))
                ]
                self._draw_artists(blended, epoch, train_losses, val_losses)
                if f < self.transition_frames:
                    self._pause()
        else:
            self._draw_artists(currents, epoch, train_losses, val_losses)

        self._prev = [c.copy() for c in currents]
        if self.gif:
            self._capture()

    def finalize(self) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}
        if not self._active:
            return artifacts
        # Animated artists are skipped by a normal draw, so un-animate before saving.
        for art in self._animated_artists():
            art.set_animated(False)
        try:
            self.fig.canvas.draw()
            png = self.out_dir / "training.png"
            self.fig.savefig(png, dpi=130, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
            artifacts["figure"] = str(png)
        except Exception as exc:
            print(f"    [monitor] PNG save failed: {exc}")
        if self.gif and self.frames:
            try:
                from PIL import Image
                gif_path = self.out_dir / "training.gif"
                imgs = [Image.fromarray(f) for f in self.frames]
                imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                             duration=120, loop=0, optimize=True)
                artifacts["gif"] = str(gif_path)
            except Exception as exc:
                print(f"    [monitor] GIF export failed: {exc}")
        if self.show:
            try:
                plt.ioff()
            except Exception:
                pass
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


_METRIC_LABELS = {"rmse": "RMSE (physical units)", "nrmse": "NRMSE",
                  "fit": "fit index [%]", "r2": "R²", "mae": "MAE"}
_LOWER_BETTER = {"rmse", "nrmse", "mae"}


def write_summary_plots(results: List[dict], out_dir: Path, metric: str = "rmse") -> List[Path]:
    """Aggregate comparison bar charts.

    Figure 1: one bar panel per benchmark, ``metric`` per model.
    Figure 2 (multi-benchmark only): a scale-free NRMSE overview grouped by
    benchmark so a model's relative quality is comparable across datasets.
    Returns the saved file paths (for embedding in the report).
    """
    def _val(r, key):
        v = r.get(key)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return []

    benches, models = [], []
    for r in ok:
        if r["benchmark"] not in benches:
            benches.append(r["benchmark"])
        if r["model"] not in models:
            models.append(r["model"])
    cmap = plt.get_cmap("tab10")
    color = {m: cmap(i % 10) for i, m in enumerate(models)}
    lookup = {(r["benchmark"], r["model"]): r for r in ok}

    label = _METRIC_LABELS.get(metric, metric)
    better = "lower is better" if metric in _LOWER_BETTER else "higher is better"
    saved: List[Path] = []

    # Figure 1 — the chosen metric, one panel per benchmark.
    n = len(benches)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.3 * nrows),
                             squeeze=False, constrained_layout=True)
    for k, b in enumerate(benches):
        ax = axes[k // ncols][k % ncols]
        ms = [m for m in models if _val(lookup.get((b, m), {}), metric) is not None]
        vals = [_val(lookup[(b, m)], metric) for m in ms]
        xs = np.arange(len(ms))
        ax.bar(xs, vals, color=[color[m] for m in ms], width=0.7)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=8)
        r0 = lookup.get((b, ms[0]), {}) if ms else {}
        ax.set_title(f"{b}  (n_u={r0.get('n_u', '?')}, n_y={r0.get('n_y', '?')})", fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if vals:
            ax.set_ylim(min(0, min(vals)), max(vals) * 1.18 if max(vals) > 0 else 1.0)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"{label} by model   ({better})", fontsize=12)
    p1 = out_dir / f"summary_{metric}.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    saved.append(p1)

    # Figure 2 — scale-free overview across benchmarks.
    if n > 1:
        present = [m for m in models if any((b, m) in lookup for b in benches)]
        fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * n + 2), 4.0), constrained_layout=True)
        x = np.arange(n)
        w = 0.8 / max(1, len(present))
        for i, m in enumerate(present):
            vals = [(_val(lookup.get((b, m), {}), "nrmse") or np.nan) for b in benches]
            ax.bar(x + (i - (len(present) - 1) / 2) * w, vals, w, label=m, color=color[m])
        ax.set_xticks(x)
        ax.set_xticklabels(benches, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("NRMSE (scale-free)")
        ax.set_title("NRMSE across benchmarks (lower is better)", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=8, ncol=min(len(present), 4), frameon=False)
        p2 = out_dir / "summary_overview_nrmse.png"
        fig.savefig(p2, dpi=140)
        plt.close(fig)
        saved.append(p2)

    return saved


def write_reports(results: List[dict], out_dir: Path, meta: dict,
                  summary_paths: Optional[List[Path]] = None) -> None:
    (out_dir / "report.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    lines = ["# Nonlinear-benchmarks report", ""]
    lines.append(f"- generated: {meta['timestamp']}")
    lines.append(f"- device: `{meta['device']}`  · dtype: `{meta['dtype']}`  · seed: {meta['seed']}")
    lines.append(f"- epochs: {meta['epochs']}  · trajectory batch: {meta['batch_size']}  · "
                 f"model dims: {meta['model_dims']}")
    lines.append("- data protocol: complete native training and validation trajectories; no windows")
    lines.append(f"- live-plot trajectory preview: first {meta['plot_max_length']} samples")
    lines.append(f"- optimizer: AdamW  · lr: {meta['lr']}  · weight decay: {meta['weight_decay']}  · "
                 f"gradient clip: {meta['grad_clip']}")
    gate_mode = "per-channel" if meta["per_channel_gates"] else "scalar"
    lines.append(f"- model policy: parameter budget `{meta['param_budget']}`  · "
                 f"gamma `{meta['gamma']}`  · feedforward `{meta['ff']}`  · "
                 f"residual gates `{gate_mode}`")
    present_models = {r.get("model") for r in results}
    for model, settings in meta["model_initialization"].items():
        if model in present_models:
            values = "  · ".join(f"{key} {value}" for key, value in settings.items())
            lines.append(f"- {model.upper()} init: {values}")
    lines.append("")
    lines.append("> Parameter matching controls trainable model size, not optimization difficulty. "
                 "Gain-certified models remain more constrained than the unconstrained LSTM/GRU baselines.")
    lines.append("")

    if summary_paths:
        lines.append("## Summary")
        lines.append("")
        for p in summary_paths:
            rel = Path(p).relative_to(out_dir).as_posix()
            lines.append(f"![{Path(rel).stem}]({rel})")
            lines.append("")

    by_bench: Dict[str, List[dict]] = {}
    for r in results:
        by_bench.setdefault(r["benchmark"], []).append(r)

    for bench, rows in by_bench.items():
        n_u = rows[0].get("n_u", "?")
        n_y = rows[0].get("n_y", "?")
        lines.append(f"## {bench}  (n_u={n_u}, n_y={n_y})")
        lines.append("")
        lines.append("| model | params | mean RMSE | mean NRMSE | mean fit % | mean R² | best val | best ep | epochs | train s | status |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda x: (x.get("rmse") is None, x.get("rmse", math.inf))):
            lines.append("| {model} | {params} | {rmse} | {nrmse} | {fit} | {r2} | "
                         "{bv} | {be} | {ep} | {tt} | {st} |".format(
                model=r["model"], params=r.get("n_params", ""),
                rmse=_fmt(r.get("rmse")), nrmse=_fmt(r.get("nrmse")),
                fit=_fmt(r.get("fit")), r2=_fmt(r.get("r2")),
                bv=_fmt(r.get("best_val_loss")), be=r.get("best_epoch", ""),
                ep=r.get("epochs_run", ""),
                tt=_fmt(r.get("train_time_s"), 3), st=r.get("status", "")))
        # The model each run is evaluated at = its lowest-validation-loss checkpoint.
        ok_rows = [r for r in rows if r.get("status") == "ok"
                   and isinstance(r.get("best_val_loss"), (int, float))
                   and math.isfinite(r["best_val_loss"])]
        if ok_rows:
            bm = min(ok_rows, key=lambda r: r["best_val_loss"])
            lines.append("")
            lines.append(f"**Best model (by validation loss): `{bm['model']}`** — "
                         f"val {_fmt(bm['best_val_loss'])} @ epoch {bm.get('best_epoch', '?')}, "
                         f"validation RMSE {_fmt(bm.get('rmse'))}, fit {_fmt(bm.get('fit'))}%.")
        # Per-validation-sequence detail when a split has several signals.
        for r in rows:
            seqs = r.get("validation_sequences", [])
            if len(seqs) > 1:
                lines.append("")
                lines.append(f"<details><summary>{r['model']}: per-validation-sequence</summary>")
                lines.append("")
                lines.append("| validation sequence | RMSE | NRMSE | fit % |")
                lines.append("|---|---|---|---|")
                for s in seqs:
                    lines.append(f"| {s['validation_name']} | {_fmt(s['rmse'])} | "
                                 f"{_fmt(s['nrmse'])} | {_fmt(s['fit'])} |")
                lines.append("</details>")
        if rows[0].get("artifacts"):
            lines.append("")
            for r in rows:
                arts = r.get("artifacts", {})
                if "gif" in arts or "figure" in arts:
                    rel = Path(arts.get("gif", arts.get("figure"))).relative_to(out_dir).as_posix()
                    lines.append(f"- {r['model']}: [{Path(rel).name}]({rel})")
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


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def run(args) -> None:
    device = pick_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    if device.type == "mps":
        print("Note: MPS lacks the SVD op behind the certified spectral norms, so those "
              "ops fall back to CPU (often SLOWER than '--device cpu' for these small "
              "models). Prefer cpu unless you've confirmed MPS is faster for your setup.")

    gconf = GlobalModelConfig(
        d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
        d_hidden=args.d_hidden, nl_layers=args.nl_layers,
        raven_heads=args.raven_heads, raven_slots=args.raven_slots, raven_top_k=args.raven_top_k,
        per_channel_gates=args.per_channel_gates,
        lru_rmin=args.lru_rmin, lru_rmax=args.lru_rmax,
        lru_max_phase=args.lru_max_phase,
        l2ru_eye_scale=args.l2ru_eye_scale,
        l2ru_rand_scale=args.l2ru_rand_scale,
        l2n_rho=args.l2n_rho, l2n_max_phase=args.l2n_max_phase,
        l2n_phase_center=args.l2n_phase_center,
        l2n_random_phase=args.l2n_random_phase,
        l2n_offdiag_scale=args.l2n_offdiag_scale,
        tv_init_rho=args.tv_init_rho,
        tv_init_delta0=args.tv_init_delta0,
        tv_init_param_scale=args.tv_init_param_scale,
        tvc_init_rho=args.tvc_init_rho,
        tvc_init_delta0=args.tvc_init_delta0,
        tvc_init_param_scale=args.tvc_init_param_scale,
        tvc_init_sign=args.tvc_init_sign,
        tvc_init_b=args.tvc_init_b, tvc_init_c=args.tvc_init_c,
        tvc_init_d=args.tvc_init_d,
        init=args.init, gamma_override=args.gamma, ff_override=args.ff,
    )
    try:
        _validate_initialization(args.models, gconf)
    except ValueError as exc:
        raise SystemExit(f"Invalid initialization setting: {exc}") from exc
    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        amp=args.amp, seed=args.seed,
    )

    bench_names = resolve_benchmark_names(args.benchmarks)

    # Live interactive plots default ON for a single benchmark, OFF for a sweep.
    if args.no_show:
        show_live = False
    elif args.show:
        show_live = True
    else:
        show_live = len(bench_names) == 1

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Benchmarks: {bench_names}\nModels: {args.models}\nOutput: {out_root}")
    if show_live and not args.no_plot:
        print("Live training plots: ON (single benchmark). Override with --no-show / --show.")

    results: List[dict] = []
    for bname in bench_names:
        print(f"\n=== {bname} ===")
        try:
            bench = load_benchmark(bname)
        except Exception as exc:
            print(f"  [skip] failed to load {bname}: {exc}")
            continue
        print(f"  loaded: n_u={bench.n_u} n_y={bench.n_y} "
              f"train_seqs={len(bench.train)} validation_seqs={len(bench.validation)} "
              f"init_window={bench.init_window} "
              f"train_len={[len(d) for d in bench.train][:6]}{'...' if len(bench.train) > 6 else ''} "
              f"validation_len={[len(d) for d in bench.validation][:6]}"
              f"{'...' if len(bench.validation) > 6 else ''}")
        norm = Standardizer.fit(bench.train)
        try:
            param_budget, budget_policy = resolve_param_budget(
                args.param_budget, bench.n_u, bench.n_y, gconf,
            )
        except Exception as exc:
            raise SystemExit(f"Invalid parameter-budget policy: {exc}") from exc
        if param_budget:
            print(f"  parameter matching: {budget_policy} -> {param_budget} trainable parameters")
        else:
            print("  parameter matching: off (using each architecture's raw width defaults)")

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
                model = build_model(mname, bench.n_u, bench.n_y, gconf,
                                    param_budget=param_budget)
                rec["n_params"] = _count_params(model)
                rec["n_params_total"] = sum(p.numel() for p in model.parameters())
                rec["width_eff"] = _effective_width(model)
                rec["param_budget"] = param_budget or None
                rec["param_budget_policy"] = budget_policy
                if param_budget:
                    print(f"     [param-match] width={rec['width_eff']} -> "
                          f"{rec['n_params']} trainable params (target {param_budget})")

                monitor = None
                if not args.no_plot:
                    monitor = TrainingMonitor(
                        out_dir=run_dir, bench=bench, model_name=mname, norm=norm,
                        device=device, total_epochs=tcfg.epochs,
                        plot_every=args.plot_every, gif=not args.no_gif, show=show_live,
                        max_preview_len=args.plot_max_length,
                    )

                hist = train_model(model, bench, norm, tcfg, device, monitor)
                rec.update({k: hist[k] for k in
                            ("best_val_loss", "final_val_loss", "best_epoch", "epochs_run",
                             "train_time_s", "n_train_sequences", "n_validation_sequences",
                             "train_samples", "validation_samples")})

                seq_rows, _ = evaluate_validation(model, bench, norm, device)
                rec["validation_sequences"] = seq_rows
                rec["rmse"] = float(np.mean([s["rmse"] for s in seq_rows]))
                rec["nrmse"] = float(np.mean([s["nrmse"] for s in seq_rows]))
                rec["r2"] = float(np.mean([s["r2"] for s in seq_rows]))
                rec["mae"] = float(np.mean([s["mae"] for s in seq_rows]))
                rec["fit"] = float(np.mean([s["fit"] for s in seq_rows]))

                if len(seq_rows) > 1:
                    print("     validation trajectories:")
                    for split in seq_rows:
                        print(
                            f"       {split['validation_name']}: "
                            f"RMSE={split['rmse']:.4g}  "
                            f"NRMSE={split['nrmse']:.4g}  fit={split['fit']:.3g}%"
                        )

                diag = model.diagnostics() if hasattr(model, "diagnostics") else None
                if diag is not None:
                    rec["certified_gain_bound"] = diag.get("certified_gain_bound")
                    rec["global_gamma"] = diag.get("global_gamma")

                if monitor is not None:
                    monitor.update(hist["epochs_run"] - 1, hist["train_losses"],
                                   hist["val_losses"], model, force=True)
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
        "data_protocol": "complete_native_splits_no_windows",
        "plot_max_length": args.plot_max_length,
        "lr": tcfg.lr, "weight_decay": tcfg.weight_decay, "grad_clip": tcfg.grad_clip,
        "gamma": args.gamma, "ff": args.ff, "param_budget": args.param_budget,
        "per_channel_gates": gconf.per_channel_gates,
        "model_initialization": _initialization_metadata(gconf),
        "model_dims": f"d_model={gconf.d_model},d_state={gconf.d_state},n_layers={gconf.n_layers}",
    }
    console_table(results)
    # Best model (by validation loss) per benchmark — the checkpoint it is evaluated at.
    by_b: Dict[str, List[dict]] = {}
    for r in results:
        by_b.setdefault(r["benchmark"], []).append(r)
    for b, rs in by_b.items():
        ok = [r for r in rs if r.get("status") == "ok"
              and isinstance(r.get("best_val_loss"), (int, float)) and math.isfinite(r["best_val_loss"])]
        if ok:
            bm = min(ok, key=lambda r: r["best_val_loss"])
            print(f"Best on {b} (by val loss): {bm['model']}  "
                  f"(val {_fmt(bm['best_val_loss'])} @ epoch {bm.get('best_epoch', '?')}, "
                  f"validation RMSE {_fmt(bm.get('rmse'))}, fit {_fmt(bm.get('fit'))}%)")

    try:
        summary_paths = write_summary_plots(results, out_root, metric=args.report_metric)
    except Exception as exc:
        print(f"[summary] plot generation failed: {exc}")
        summary_paths = []
    write_reports(results, out_root, meta, summary_paths=summary_paths)
    for p in summary_paths:
        print(f"Summary plot: {p}")


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
    p.add_argument("--per-channel-gates", action="store_true",
                   help="Use per-channel (d_model-dim) residual gates instead of scalar gates "
                        "in DeepSSM layers. Certified models use the worst-channel gain.")
    # Model-specific initialization. The desktop UI reveals these conditionally.
    p.add_argument("--lru-rmin", type=float, default=0.8,
                   help="LRU minimum initial pole radius")
    p.add_argument("--lru-rmax", type=float, default=0.95,
                   help="LRU maximum initial pole radius")
    p.add_argument("--lru-max-phase", type=float, default=2.0 * math.pi,
                   help="LRU maximum initial pole phase in radians")
    p.add_argument("--l2ru-eye-scale", type=float, default=0.01,
                   help="L2RU triangular-factor scale for eye initialization")
    p.add_argument("--l2ru-rand-scale", type=float, default=1.0,
                   help="L2RU free-matrix standard deviation for random initialization")
    p.add_argument("--l2n-rho", type=float, default=0.9,
                   help="L2N target initial recurrent pole radius before contraction normalization")
    p.add_argument("--l2n-max-phase", type=float, default=0.04,
                   help="L2N initial phase half-width around --l2n-phase-center")
    p.add_argument("--l2n-phase-center", type=float, default=0.0,
                   help="L2N initial phase-window center in radians")
    p.add_argument("--l2n-random-phase", action=argparse.BooleanOptionalAction, default=True,
                   help="sample L2N phases in the configured window; disable to use the center")
    p.add_argument("--l2n-offdiag-scale", type=float, default=0.05,
                   help="initial standard deviation of L2N K12/K21/K22")
    p.add_argument("--tv-init-rho", type=float, default=0.99,
                   help="TV initial diagonal-state decay")
    p.add_argument("--tv-init-delta0", type=float, default=1.0,
                   help="TV initial selective step size")
    p.add_argument("--tv-init-param-scale", type=float, default=0.02,
                   help="TV parameter-network weight standard deviation")
    p.add_argument("--tvc-init-rho", type=float, default=0.9,
                   help="TVC initial unsigned state decay")
    p.add_argument("--tvc-init-delta0", type=float, default=1.0,
                   help="TVC initial selective step size")
    p.add_argument("--tvc-init-param-scale", type=float, default=0.02,
                   help="TVC parameter-network weight standard deviation")
    p.add_argument("--tvc-init-sign", type=float, default=0.995,
                   help="TVC initial signed-decay multiplier")
    p.add_argument("--tvc-init-b", type=float, default=0.10,
                   help="TVC initial input coupling")
    p.add_argument("--tvc-init-c", type=float, default=0.10,
                   help="TVC initial output coupling")
    p.add_argument("--tvc-init-d", type=float, default=0.10,
                   help="TVC initial direct feedthrough")
    p.add_argument("--init", default="eye", choices=["eye", "rand"],
                   help="L2RU initialization mode")
    p.add_argument("--gamma", default="auto",
                   help="prescribed gain: 'auto' (per-model), 'none' (uncertified), or a float")
    p.add_argument("--ff", default="auto",
                   help="feedforward: auto (per-model) | GLU|MLP|LGLU2|MBLIP|BLGLU2|TLIP|LMLP")
    # plotting
    p.add_argument("--no-plot", action="store_true", help="disable the training monitor entirely")
    p.add_argument("--no-gif", action="store_true", help="disable GIF export (keep final PNG)")
    p.add_argument("--show", action="store_true",
                   help="force the live interactive window on (default: on iff a single benchmark)")
    p.add_argument("--no-show", action="store_true",
                   help="force the live interactive window off (still saves GIF/PNG)")
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument(
        "--plot-max-length",
        type=_positive_int,
        default=4000,
        metavar="SAMPLES",
        help="maximum number of leading samples shown from both training and validation trajectories",
    )
    p.add_argument("--report-metric", default="rmse", choices=["rmse", "nrmse", "fit", "r2", "mae"],
                   help="metric for the per-benchmark summary bar charts")
    p.add_argument("--param-budget", default="lstm",
                   help="capacity matching target: a positive trainable-parameter count, "
                        "a model name whose raw count is used as the per-benchmark target, "
                        "or 'off'. The default matches every model to the LSTM baseline")
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
