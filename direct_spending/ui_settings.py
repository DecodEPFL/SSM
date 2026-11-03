"""Lightweight benchmark UI field policy and contained default-file persistence."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile

from .config import MODEL_SPECS
from split_memory.ui import DEFAULTS as SPLIT_DEFAULTS
from .ui import DEFAULTS as SPENDING_DEFAULTS

SSM_MODELS = frozenset(("lru", "l2n", "defect", "tv", "tvc", "l2ru", "zak", "l2nt"))
METRIC_MODELS = frozenset(("metric_defect", "metric_l2n"))
RNN_MODELS = frozenset(("lstm", "gru"))
UI_MODELS = ("lru", "l2n", "defect", "metric_defect", "metric_l2n", "tv", "tvc",
             "ren", "lstm", "gru", "l2ru", "zak", "l2nt", "direct_spending", "split_memory")

# Shared fields describe the run, never an architecture.
SHARED_DEFAULTS = dict(device="auto", epochs="200", batch_size="32", lr="3e-3",
    weight_decay="0", grad_clip="1", seed="0", report_metric="rmse", param_budget="lstm",
    plot_every="10", plot_max_length="4000", compile_mode="default")
MODEL_DEFAULTS = dict(d_model="16", d_state="16", n_layers="2", d_hidden="16", nl_layers="3",
    gamma="auto", ff="auto", mode="auto", init="eye", rnn_hidden="64", rnn_layers="2",
    lru_rmin="0.8", lru_rmax="0.95", lru_max_phase=str(2 * math.pi),
    l2ru_eye_scale="0.01", l2ru_rand_scale="1.0", l2n_rho="0.9", l2n_max_phase="0.04",
    l2n_phase_center="0.0", l2n_offdiag_scale="0.05", l2n_state_metric="identity",
    defect_block_size="2", defect_rho="0.9", defect_max_radius="0.999",
    defect_factor_margin="0.001", defect_init_scale="0.05", defect_max_phase="0.04",
    defect_phase_center="0.0", defect_state_metric="identity", metric_activation="tanh",
    metric_ff="residual", metric_init_feedthrough="0.95", tv_init_rho="0.99",
    tv_init_delta0="1.0", tv_init_param_scale="0.02", tvc_init_rho="0.9", tvc_init_delta0="1.0",
    tvc_init_param_scale="0.02", tvc_init_sign="0.995", tvc_init_b="0.10", tvc_init_c="0.10",
    tvc_init_d="0.10", ren_dim_internal="16", ren_dim_nl="16", **SPENDING_DEFAULTS, **SPLIT_DEFAULTS)
VALUE_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS}
TOGGLE_DEFAULTS = dict(show_var=True, gif_var=True, amp_var=False, compile_var=False,
    pcg_var=False, cudagraph_var=True, l2n_random_phase_var=True, defect_random_phase_var=True)


def architecture_keys(models, spending_core="gru", ff="auto", metric_ff="residual"):
    """Fields actually used by the selected runner factories (not their global config)."""
    selected = set(models)
    keys = set()
    if selected & (SSM_MODELS | METRIC_MODELS):
        keys.update(("d_model", "d_state", "n_layers", "gamma", "mode"))
    if selected & SSM_MODELS:
        keys.add("ff")
        for model in selected & SSM_MODELS:
            effective_ff = ({"lru": "GLU", "tv": "MBLIP", "tvc": "MBLIP"}.get(model, "LGLU2")
                            if ff == "auto" else ff)
            if effective_ff in ("MLP", "TLIP", "LMLP", "MBLIP", "BLGLU2", "BudgetedLGLU2"):
                keys.add("d_hidden")
            if effective_ff in ("MLP", "TLIP", "LMLP"):
                keys.add("nl_layers")
    if selected & METRIC_MODELS and metric_ff != "none":
        keys.add("d_hidden")
    if selected & RNN_MODELS:
        keys.update(("rnn_hidden", "rnn_layers"))
    if "ren" in selected:
        keys.update(("gamma", "ren_dim_internal", "ren_dim_nl"))
    if "direct_spending" in selected:
        keys.add("d_model")
        if spending_core != "tcn":
            keys.add("n_layers")
    # Legacy L2RU uses d_model as its state dimension.
    if not selected & ((SSM_MODELS - {"l2ru"}) | METRIC_MODELS):
        keys.discard("d_state")
    return keys


def normalize_configuration(config, benchmarks):
    """Validate a saved UI selection, migrating one old spending alias explicitly."""
    if not isinstance(config, dict) or config.get("format") != "nonlinear_benchmark_ui_v1":
        raise ValueError("Unsupported configuration format")
    result = copy.deepcopy(config)
    for key in ("models", "benchmarks"):
        if not isinstance(result.get(key), list) or not result[key] or not all(isinstance(x, str) for x in result[key]):
            raise ValueError(f"Configuration needs a nonempty {key} list")
    if not isinstance(result.get("values"), dict) or not isinstance(result.get("toggles"), dict):
        raise ValueError("Configuration values and toggles must be objects")
    if not isinstance(result.get("out"), str) or not result["out"].strip():
        raise ValueError("Configuration needs an output directory")
    if set(result["models"]) - (set(UI_MODELS) | set(MODEL_SPECS)):
        raise ValueError("Configuration contains an unknown model")
    if set(result["benchmarks"]) - set(benchmarks):
        raise ValueError("Configuration contains an unknown benchmark")
    if any(key not in TOGGLE_DEFAULTS or not isinstance(value, bool)
           for key, value in result["toggles"].items()):
        raise ValueError("Configuration contains an unknown or invalid toggle")
    if any(key not in VALUE_DEFAULTS or not isinstance(value, (str, int, float)) or isinstance(value, bool)
           for key, value in result["values"].items()):
        raise ValueError("Configuration contains an unknown or invalid parameter")
    values = {**VALUE_DEFAULTS, **result["values"]}
    cores = {MODEL_SPECS[name]["spending_core"] for name in result["models"]
             if name in MODEL_SPECS and name != "direct_spending"}
    if "direct_spending" in result["models"]:
        cores.add(values["spending_core"])
    if len(cores) > 1:
        raise ValueError("This file selects several spending cores. Load one core at a time in the UI; "
                         "the command-line runner still supports spending_* batch aliases.")
    if cores:
        values["spending_core"] = cores.pop()
    result["models"] = list(dict.fromkeys("direct_spending" if name in MODEL_SPECS else name
                                        for name in result["models"]))
    result["values"] = {key: str(value) for key, value in values.items()}
    result["toggles"] = {**TOGGLE_DEFAULTS, **result["toggles"]}
    # Rebuild argv from controls; the old argv may still contain an alias.
    result.pop("argv", None)
    return result


def write_configuration(path, config):
    """Replace defaults/configuration atomically, preserving a previous file on failure."""
    path = Path(path)
    payload = json.dumps(config, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
