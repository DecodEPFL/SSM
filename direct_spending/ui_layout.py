"""Architecture-dependent controls for the existing Tk benchmark window."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui import SpendingOptions
from .ui_settings import SSM_MODELS, METRIC_MODELS, RNN_MODELS, architecture_keys
from .ui_numbers import learning_rate_preview

CHOICES = {
    "device": ("auto", "cpu", "cuda", "mps"),
    "report_metric": ("rmse", "nrmse", "fit", "r2", "mae"),
    "compile_mode": ("default", "reduce-overhead", "max-autotune"),
    "ff": ("auto", "GLU", "MLP", "LGLU2", "MBLIP", "BLGLU2", "BudgetedLGLU2", "TLIP", "LMLP"),
    "mode": ("auto", "loop", "scan", "conv"), "init": ("eye", "rand"),
    "gamma": ("auto", "none", "0.5", "1", "2", "5", "10", "20"),
    "l2n_state_metric": ("identity", "full"), "defect_state_metric": ("identity", "full"),
    "metric_activation": ("tanh", "relu", "arctan", "identity"),
    "metric_ff": ("residual", "none"),
}
LABELS = {
    "device": "Device", "epochs": "Epochs", "batch_size": "Trajectory batch size",
    "lr": "Learning rate", "weight_decay": "Weight decay", "grad_clip": "Gradient clip",
    "seed": "Random seed", "report_metric": "Report metric", "param_budget": "Parameter budget",
    "plot_every": "Plot every (epochs)", "plot_max_length": "Plot preview samples",
    "compile_mode": "Compilation mode", "d_model": "Model width", "d_state": "State dimension",
    "n_layers": "Layers", "d_hidden": "Feedforward width", "nl_layers": "Feedforward layers",
    "gamma": "Gain limit (gamma)", "ff": "Feedforward", "mode": "SSM execution",
    "rnn_hidden": "Hidden width", "rnn_layers": "Recurrent layers", "init": "Initialization",
    "lru_rmin": "Minimum pole radius", "lru_rmax": "Maximum pole radius",
    "lru_max_phase": "Maximum phase", "l2ru_eye_scale": "Identity scale", "l2ru_rand_scale": "Random scale",
    "l2n_rho": "Target pole radius", "l2n_max_phase": "Phase half-width",
    "l2n_phase_center": "Phase center", "l2n_offdiag_scale": "Off-diagonal std",
    "l2n_state_metric": "State storage", "defect_block_size": "Block size",
    "defect_rho": "Initial radius", "defect_max_radius": "Contraction limit",
    "defect_factor_margin": "Factor margin", "defect_init_scale": "Initial factor scale",
    "defect_max_phase": "Phase half-width (2x2)", "defect_phase_center": "Phase center (2x2)",
    "defect_state_metric": "State storage", "metric_activation": "Activation",
    "metric_ff": "Weighted feedforward", "metric_init_feedthrough": "Initial direct path",
    "tv_init_rho": "State decay", "tv_init_delta0": "Step size", "tv_init_param_scale": "Parameter std",
    "tvc_init_rho": "Unsigned decay", "tvc_init_delta0": "Step size",
    "tvc_init_param_scale": "Parameter std", "tvc_init_sign": "Decay sign",
    "tvc_init_b": "Input coupling", "tvc_init_c": "Output coupling", "tvc_init_d": "Feedthrough",
    "ren_dim_internal": "Internal state dimension", "ren_dim_nl": "Nonlinearity dimension",
}
INITIALIZATION = {
    "lru": ("lru_rmin", "lru_rmax", "lru_max_phase"),
    "l2ru": ("init", "l2ru_eye_scale", "l2ru_rand_scale"), "zak": ("init",),
    "l2n": ("l2n_rho", "l2n_max_phase", "l2n_phase_center", "l2n_offdiag_scale", "l2n_state_metric"),
    "defect": ("defect_block_size", "defect_rho", "defect_max_radius", "defect_factor_margin",
               "defect_init_scale", "defect_max_phase", "defect_phase_center", "defect_state_metric"),
    "tv": ("tv_init_rho", "tv_init_delta0", "tv_init_param_scale"),
    "tvc": ("tvc_init_rho", "tvc_init_delta0", "tvc_init_param_scale", "tvc_init_sign",
            "tvc_init_b", "tvc_init_c", "tvc_init_d"),
}
ARCHITECTURE_ORDER = ("d_model", "d_state", "n_layers", "d_hidden", "nl_layers", "gamma", "ff", "mode",
                      "rnn_hidden", "rnn_layers", "ren_dim_internal", "ren_dim_nl")


def fields(parent, keys, variables, *, columns=3):
    """One compact field grid; variables stay alive when their model is hidden."""
    controls = {}
    for index, key in enumerate(keys):
        row, column = divmod(index, columns)
        label = ttk.Label(parent, text=LABELS[key])
        label.grid(row=row, column=2 * column, sticky="e", padx=(8, 3), pady=3)
        options = dict(textvariable=variables[key], width=13)
        widget = (ttk.Combobox(parent, values=CHOICES[key],
                              state="normal" if key == "gamma" else "readonly", **options)
                  if key in CHOICES else ttk.Entry(parent, **options))
        widget.grid(row=row, column=2 * column + 1, sticky="ew", padx=(0, 12), pady=3)
        parent.columnconfigure(2 * column + 1, weight=1)
        controls[key] = (label, widget)
        if key == "lr":
            def preview(*_, label=label, variable=variables[key]):
                label.configure(text="Learning rate = " + learning_rate_preview(variable.get()))
            variables[key].trace_add("write", preview)
            preview()
    parent.bind("<Configure>", lambda event: reflow_fields(controls, columns, event.width))
    return controls


def reflow_fields(controls, columns=3, width=1120):
    count = columns if width <= 1 or width >= 1000 else min(columns, 2 if width >= 620 else 1)
    index = 0
    for widgets in controls.values():
        if widgets[1].winfo_manager():
            row, column = divmod(index, count)
            for offset, widget in enumerate(widgets):
                widget.grid(row=row, column=2 * column + offset)
            index += 1


class ScrollableSettings:
    """Keep action buttons/results reachable when conditional options grow."""
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.frame, highlightthickness=0, height=570)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.body = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window(0, 0, anchor="nw", window=self.body)
        self.body.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.window, width=event.width))
        # A private bind tag scrolls only settings, without stealing listbox/combobox wheels.
        self.wheel_tag = "BenchmarkSettingsWheel_" + str(self.frame).replace(".", "_")
        self.canvas.bind_class(self.wheel_tag, "<MouseWheel>", self._wheel)
        self.canvas.bind_class(self.wheel_tag, "<Button-4>", lambda _: self._scroll(-1))
        self.canvas.bind_class(self.wheel_tag, "<Button-5>", lambda _: self._scroll(1))

    def bind_wheel(self, widget=None):
        widget = self.body if widget is None else widget
        if not isinstance(widget, (tk.Listbox, ttk.Combobox, ttk.Scrollbar)):
            tags = widget.bindtags()
            if self.wheel_tag not in tags:
                widget.bindtags((self.wheel_tag, *tags))
        for child in widget.winfo_children():
            self.bind_wheel(child)

    def _scroll(self, units):
        if self.body.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(units, "units")
        return "break"

    def _wheel(self, event):
        # macOS sends small deltas; Windows commonly sends multiples of 120.
        delta = event.delta
        units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        return self._scroll(units)


class ModelOptions:
    def __init__(self, parent, owner):
        self.owner = owner
        self.frame = ttk.LabelFrame(parent, text="Model options", padding=8)
        self.frame.pack(fill="x", pady=(8, 0))
        self.parameter_button = ttk.Button(self.frame, text="Count parameters", command=owner._show_parameter_count)
        self.parameter_button.pack(anchor="w", pady=(0, 5))
        self.note = ttk.Label(self.frame, text="", wraplength=960)
        self.note.pack(fill="x", pady=(0, 5))
        self.frame.bind("<Configure>", lambda event: self.note.configure(wraplength=max(280, event.width - 20)))
        self.notebook = ttk.Notebook(self.frame)
        self.notebook.pack(fill="x")
        self.frames, self.controls = {}, {}
        self.spending = None
        self.selected = []
        for key in ("spending_core", "ff", "metric_ff"):
            owner.vars[key].trace_add("write", lambda *_: self.refresh())

    def _build(self, model):
        frame = ttk.Frame(self.notebook, padding=6)
        self.frames[model] = frame
        grid = ttk.Frame(frame)
        grid.pack(fill="x")
        # Build the union of this family's fields once. TCN hides its unused layer count.
        keys = architecture_keys([model], ff="MLP")
        self.controls[model] = fields(grid, [key for key in ARCHITECTURE_ORDER if key in keys], self.owner.vars)
        if model in SSM_MODELS:
            switches = ttk.Frame(frame)
            switches.pack(fill="x", pady=3)
            ttk.Checkbutton(switches, text="Per-channel residual gates", variable=self.owner.pcg_var).pack(side="left")
            if model in ("tv", "tvc"):
                ttk.Checkbutton(switches, text="CUDA graph", variable=self.owner.cudagraph_var).pack(side="left", padx=14)
        if model in RNN_MODELS:
            ttk.Label(frame, text="Parameter matching, when enabled, adjusts hidden width; layer count stays fixed.").pack(anchor="w", pady=4)
        if model == "direct_spending":
            self.spending = SpendingOptions(frame, self.owner.vars, initialize=False)
            self.spending.frame.pack(fill="x", pady=(6, 0))
            self.owner.spending_options = self.spending
        if model == "split_memory":
            from split_memory.ui import SplitMemoryOptions
            self.owner.split_memory_options = SplitMemoryOptions(frame, self.owner.vars)
            self.owner.split_memory_options.frame.pack(fill="x")
        base = {"metric_l2n": "l2n", "metric_defect": "defect"}.get(model, model)
        if model in METRIC_MODELS:
            group = ttk.LabelFrame(frame, text="Metric transfer", padding=4)
            group.pack(fill="x", pady=(6, 0))
            fields(group, ("metric_activation", "metric_ff", "metric_init_feedthrough"), self.owner.vars)
        if base in INITIALIZATION:
            group = ttk.LabelFrame(frame, text="Initialization", padding=4)
            group.pack(fill="x", pady=(6, 0))
            grid = ttk.Frame(group)
            grid.pack(fill="x")
            fields(grid, INITIALIZATION[base], self.owner.vars)
            if base in ("l2n", "defect"):
                ttk.Checkbutton(group, text="Random phases", variable=getattr(self.owner, base + "_random_phase_var")).pack(
                    anchor="w", padx=8, pady=3)

    def refresh(self):
        for model, controls in self.controls.items():
            active = architecture_keys([model], self.owner.vars["spending_core"].get(),
                                      self.owner.vars["ff"].get(), self.owner.vars["metric_ff"].get())
            for key, widgets in controls.items():
                for widget in widgets:
                    widget.grid() if key in active else widget.grid_remove()
            if controls:
                parent = next(iter(controls.values()))[1].master
                reflow_fields(controls, width=parent.winfo_width())

    def select(self, models):
        current = self.notebook.select()
        previous = set(self.selected)
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.selected = list(models)
        for model in models:
            if model not in self.frames:
                self._build(model)
            self.notebook.add(self.frames[model], text=model.upper() if model != "direct_spending" else "Direct spending")
        added = [name for name in models if name not in previous]
        if added:
            self.notebook.select(self.frames[added[0]])
        elif current in self.notebook.tabs():
            self.notebook.select(current)
        self.note.configure(text=("Select a model above to see its settings." if not models else
            "Each tab shows the selected model's options. Fields with the same name are shared across selected models."
            if len(models) > 1 else ""))
        if self.spending is not None:
            self.spending.set_models(["direct_spending"])
        self.refresh()
