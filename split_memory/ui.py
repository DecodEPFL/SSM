"""Conditional options embedded in the existing benchmark UI; no torch import."""
import tkinter as tk
from tkinter import ttk

from .config import SplitMemoryConfig

DEFAULTS = dict(split_reader="recurrent", split_memory_heads="8", split_computation_width="16",
    split_rho="0.7", split_memory_radius="1", split_initial_retention="0.95", split_row_delta="1e-4",
    split_solver="fixed", split_memory_backend="scan", split_tolerance="1e-4",
    split_fixed_sweeps="32", split_max_sweeps="64", split_require_convergence="1", split_diagnostics="0")
LABELS = dict(split_reader="Reader", split_memory_heads="Memory heads (2D)", split_computation_width="Computation width",
    split_rho="Contraction bound (rho)", split_memory_radius="Memory radius", split_initial_retention="Initial retention",
    split_solver="Reader solver", split_memory_backend="Memory evaluation", split_tolerance="State error tolerance",
    split_fixed_sweeps="Fixed sweeps", split_max_sweeps="Maximum sweeps")
CHOICES = dict(split_reader=("recurrent", "pointwise", "linear"),
    split_solver=("fixed", "adaptive", "sequential"), split_memory_backend=("scan", "sequential"))


def active_keys(get):
    keys = {"split_reader", "split_memory_heads", "split_memory_radius", "split_initial_retention", "split_memory_backend"}
    if get("split_reader") != "linear":
        keys.add("split_computation_width")
    if get("split_reader") == "recurrent":
        keys.update(("split_rho", "split_solver"))
        if get("split_solver") != "sequential":
            keys.update(("split_tolerance", "split_require_convergence"))
            keys.add("split_fixed_sweeps" if get("split_solver") == "fixed" else "split_max_sweeps")
    return keys


def command_arguments(variables):
    raw = lambda key: str(variables[key].get()).strip()
    active = active_keys(raw)
    values = {key: raw(key) if key in active or key == "split_diagnostics" else value for key, value in DEFAULTS.items()}
    config = SplitMemoryConfig(reader=values["split_reader"], heads=int(values["split_memory_heads"]),
        computation_width=int(values["split_computation_width"]), rho=float(values["split_rho"]),
        memory_radius=float(values["split_memory_radius"]), initial_retention=float(values["split_initial_retention"]),
        row_delta=float(values["split_row_delta"]), solver=values["split_solver"], memory_backend=values["split_memory_backend"],
        tolerance=float(values["split_tolerance"]), fixed_sweeps=int(values["split_fixed_sweeps"]),
        max_sweeps=int(values["split_max_sweeps"]), require_convergence=values["split_require_convergence"] == "1")
    if values["split_require_convergence"] not in ("0", "1") or values["split_diagnostics"] not in ("0", "1"):
        raise ValueError("Invalid split-memory checkbox value")
    output = []
    for key in DEFAULTS:
        if key in active and key != "split_require_convergence":
            output += ["--" + key.replace("_", "-"), values[key]]
    if "split_require_convergence" in active:
        output += ["--split-require-convergence" if config.require_convergence else "--no-split-require-convergence"]
    if values["split_diagnostics"] == "1":
        output += ["--split-diagnostics"]
    return output


class SplitMemoryOptions:
    def __init__(self, parent, variables):
        # Reuse the existing field renderer; importing here avoids UI-module cycles.
        from direct_spending.ui_layout import fields, CHOICES as common_choices, LABELS as common_labels
        common_choices.update(CHOICES)
        common_labels.update(LABELS)
        self.variables = variables
        self.frame = ttk.Frame(parent)
        for key, default in DEFAULTS.items():
            if key not in variables:
                variables[key] = tk.StringVar(value=default)
        grid = ttk.Frame(self.frame)
        grid.pack(fill="x")
        self.controls = fields(grid, list(LABELS), variables)
        switches = ttk.Frame(self.frame)
        switches.pack(fill="x", pady=6)
        self.require = ttk.Checkbutton(switches, text="Require tolerance (disable for explicit approximate runs)",
            variable=variables["split_require_convergence"], onvalue="1", offvalue="0")
        self.require.pack(side="left")
        ttk.Checkbutton(switches, text="Memory/solver diagnostics", variable=variables["split_diagnostics"],
            onvalue="1", offvalue="0").pack(side="left", padx=10)
        self.note = ttk.Label(self.frame, wraplength=950)
        self.note.pack(fill="x")
        self.frame.bind("<Configure>", lambda event: self.note.configure(wraplength=max(280, event.width - 20)))
        for key in ("split_reader", "split_solver", "split_require_convergence"):
            variables[key].trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def refresh(self):
        from direct_spending.ui_layout import reflow_fields
        get = lambda key: str(self.variables[key].get())
        active = active_keys(get)
        for key, widgets in self.controls.items():
            for widget in widgets:
                widget.grid() if key in active else widget.grid_remove()
        reflow_fields(self.controls, width=self.frame.winfo_width())
        if "split_require_convergence" in active:
            self.require.pack(side="left", before=self.require.master.winfo_children()[-1])
        else:
            self.require.pack_forget()
        message = "Bounded memory; the recurrent reader is contractive in its own state. No prescribed l2-gain guarantee. "
        if get("split_reader") == "recurrent":
            message += ("Streaming uses one exact recurrent evaluation per token. " if get("split_solver") == "sequential" else
                "Fixed/adaptive sweeps use ordinary autograd through their actual finite graph. ")
            if get("split_solver") != "sequential" and get("split_require_convergence") == "0":
                message += "Approximate runs are explicitly allowed; evaluation reports convergence and residuals. "
        else:
            message += "This ablation has no nonlinear recurrent computation update. "
        self.note.configure(text=message + "Memory heads and computation width are independent. Parameter matching may adjust reader width (heads for linear readout).")
