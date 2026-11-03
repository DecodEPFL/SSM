"""Conditional controls embedded in the existing benchmark Tk window; no torch import."""
from __future__ import annotations

from dataclasses import replace

import tkinter as tk
from tkinter import ttk

from .config import CORES, MODEL_SPECS, SpendingConfig
from .ui_numbers import real_number, whole_number


DEFAULTS = {"spending_core": "gru", "spending_gamma": "1", "spending_beta_mode": "zero",
            "spending_readout": "Spending fractions", "spending_linear_init": "zero",
            "spending_gamma_mode": "fixed", "spending_gamma_min": "0.0001", "spending_gamma_max": "100",
            "spending_fixed_beta": "0", "spending_beta_initial": "0.1", "spending_beta_width": "8",
            "spending_initial_features": "", "spending_tcn_kernel": "3",
            "spending_tcn_dilations": "1,2,4", "spending_heads": "2", "spending_dropout": "0",
            "spending_storage": "scan", "spending_precision": "auto",
            "spending_coordinates": "standardized", "spending_diagnostics": "0"}

READOUTS = {"Spending fractions": "spending", "Output prediction + linear": "projected"}


def readout_mode(value):
    return READOUTS.get(value, value)


def command_arguments(values, models):
    """Validate operator settings without loading torch, then emit existing runner arguments."""
    raw = lambda key: values[key].get().strip()
    core, mode = raw("spending_core"), raw("spending_beta_mode")
    readout = readout_mode(raw("spending_readout"))
    active = {"spending_readout", "spending_core", "spending_gamma", "spending_gamma_mode", "spending_beta_mode", "spending_coordinates",
              "spending_storage", "spending_precision", "spending_diagnostics"}
    if readout == "projected":
        active.add("spending_linear_init")
    if raw("spending_gamma_mode") == "learned":
        active.update(("spending_gamma_min", "spending_gamma_max"))
    if mode == "fixed":
        active.add("spending_fixed_beta")
    if mode == "learned":
        active.update(("spending_beta_initial", "spending_initial_features"))
        if raw("spending_initial_features"):
            active.add("spending_beta_width")
    if core == "tcn":
        active.update(("spending_tcn_kernel", "spending_tcn_dilations"))
    if core == "transformer":
        active.update(("spending_heads", "spending_dropout"))
    get = lambda key: raw(key) if key in active else DEFAULTS[key]
    cfg = SpendingConfig(core=core, width=whole_number(raw("d_model"), "Model width"),
        readout=readout, linear_init=get("spending_linear_init"),
        layers=2 if core == "tcn" else whole_number(raw("n_layers"), "Layers"),
        gamma=real_number(get("spending_gamma"), "Spending gain (gamma)"),
        gamma_mode=get("spending_gamma_mode"),
        gamma_min=real_number(get("spending_gamma_min"), "Minimum gamma"),
        gamma_max=real_number(get("spending_gamma_max"), "Maximum gamma"),
        beta_mode=mode, fixed_beta=real_number(get("spending_fixed_beta"), "Fixed initial energy"),
        beta_initial=real_number(get("spending_beta_initial"), "Learned initial energy"),
        beta_width=whole_number(get("spending_beta_width"), "Initial energy network width"),
        initial_features=tuple(real_number(x.strip(), "Initial context") for x in get("spending_initial_features").split(",") if x.strip()),
        tcn_kernel=whole_number(get("spending_tcn_kernel"), "TCN kernel size"),
        tcn_dilations=tuple(whole_number(x.strip(), "TCN dilations") for x in get("spending_tcn_dilations").split(",") if x.strip()),
        heads=whole_number(get("spending_heads"), "Attention heads"),
        dropout=real_number(get("spending_dropout"), "Transformer dropout"),
        storage_impl=get("spending_storage"), energy_precision=get("spending_precision"))
    for model in set(models).intersection(MODEL_SPECS):
        replace(cfg, core=MODEL_SPECS[model].get("spending_core", cfg.core))
    output = []
    for key in DEFAULTS:
        if key in active and key != "spending_diagnostics" and not (key == "spending_initial_features" and not get(key)):
            flag = "--" + key.replace("_", "-")
            if key == "spending_initial_features":
                # A leading negative comma-separated context must stay one value.
                output.append(flag + "=" + get(key))
            else:
                output.extend((flag, readout if key == "spending_readout" else get(key)))
    if get("spending_diagnostics") == "1":
        output.append("--spending-diagnostics")
    return output


class SpendingOptions:
    def __init__(self, parent, variables, *, initialize=True):
        self.frame = ttk.LabelFrame(parent, text="Direct spending — scalar storage", padding=8)
        self.variables = variables
        self.selected_models = set()
        self.controls = {}
        for key, value in DEFAULTS.items():
            if initialize or key not in variables:
                variables[key] = tk.StringVar(value=value)

        def control(key, label, row, column, choices=None, width=12):
            text = ttk.Label(self.frame, text=label)
            text.grid(row=row, column=column * 2, sticky="e", padx=(5, 2), pady=3)
            options = dict(textvariable=variables[key], width=width)
            widget = (ttk.Combobox(self.frame, values=choices, state="readonly", **options)
                      if choices is not None else ttk.Entry(self.frame, **options))
            widget.grid(row=row, column=column * 2 + 1, sticky="w", padx=(0, 12), pady=3)
            self.controls[key] = (text, widget)

        control("spending_core", "Spending core", 0, 0, CORES)
        control("spending_readout", "Readout", 0, 0, tuple(READOUTS), width=25)
        control("spending_linear_init", "Linear path start", 0, 0, ("zero", "identity"))
        control("spending_gamma", "Fixed gamma", 0, 1)
        control("spending_gamma_mode", "Gamma", 0, 1, ("fixed", "learned"))
        control("spending_gamma_min", "Minimum gamma", 0, 1)
        control("spending_gamma_max", "Maximum gamma", 0, 1)
        control("spending_beta_mode", "Initial energy", 0, 2, ("zero", "fixed", "learned"))
        control("spending_coordinates", "Certificate units", 3, 0, ("standardized", "physical"))
        control("spending_fixed_beta", "Fixed beta", 1, 0)
        control("spending_beta_initial", "Initial learned beta", 1, 0)
        control("spending_beta_width", "Beta MLP width", 1, 1)
        control("spending_initial_features", "Initial context constants", 1, 2, width=20)
        control("spending_tcn_kernel", "TCN kernel", 2, 0)
        control("spending_tcn_dilations", "TCN dilations", 2, 1)
        control("spending_heads", "Attention heads", 2, 0)
        control("spending_dropout", "Attention dropout", 2, 1)
        control("spending_storage", "Storage evaluation", 3, 1, ("scan", "reference"))
        control("spending_precision", "Energy precision", 3, 2, ("auto", "float32", "float64"))
        self.diagnostics = ttk.Checkbutton(self.frame, text="Storage diagnostics", variable=variables["spending_diagnostics"],
                                          onvalue="1", offvalue="0")
        self.diagnostics.grid(row=4, column=0, columnspan=4, sticky="w")
        self.explanation = ttk.Label(self.frame, wraplength=1000)
        self.explanation.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.frame.bind("<Configure>", lambda event: self.explanation.configure(wraplength=max(280, event.width - 20)))
        for key in ("spending_core", "spending_readout", "spending_beta_mode", "spending_coordinates", "spending_initial_features", "spending_gamma_mode"):
            variables[key].trace_add("write", lambda *_: self.refresh())
        self.refresh()

    def set_models(self, models):
        self.selected_models = set(models).intersection(MODEL_SPECS)
        named = [name for name in MODEL_SPECS if name in self.selected_models and name != "direct_spending"]
        if "direct_spending" not in self.selected_models and named:
            self.variables["spending_core"].set(MODEL_SPECS[named[0]]["spending_core"])
        self.refresh()

    def refresh(self):
        core = self.variables["spending_core"].get()
        active_cores = {MODEL_SPECS[name].get("spending_core", core) for name in self.selected_models} or {core}
        mode = self.variables["spending_beta_mode"].get()
        learned_gamma = self.variables["spending_gamma_mode"].get() == "learned"
        projected = readout_mode(self.variables["spending_readout"].get()) == "projected"
        self.frame.configure(text="Output projection + linear path — scalar reservoir" if projected else "Direct spending — scalar storage")
        self.controls["spending_core"][0].configure(text="Neural core" if projected else "Spending core")
        self.controls["spending_gamma"][0].configure(text="Initial gamma" if learned_gamma else "Fixed gamma")
        for key, widgets in self.controls.items():
            show = not ((key == "spending_fixed_beta" and mode != "fixed")
                        or (key == "spending_linear_init" and not projected)
                        or (key in ("spending_gamma_min", "spending_gamma_max") and not learned_gamma)
                        or (key in ("spending_beta_initial", "spending_beta_width", "spending_initial_features") and mode != "learned")
                        or (key == "spending_beta_width" and not self.variables["spending_initial_features"].get().strip())
                        or (key.startswith("spending_tcn_") and "tcn" not in active_cores)
                        or (key in ("spending_heads", "spending_dropout") and "transformer" not in active_cores))
            for widget in widgets:
                widget.grid() if show else widget.grid_remove()
        self._layout()
        selectable = not self.selected_models or "direct_spending" in self.selected_models
        self.controls["spending_core"][1].configure(state="readonly" if selectable else "disabled")
        units = self.variables["spending_coordinates"].get()
        certificate = "Zero-bias gain bound" if mode == "zero" else "Gain bound with an additive initial-energy allowance"
        suffix = ("Standardization includes centering; this is not a physical-coordinate zero-state bound."
                  if units == "standardized" else "Physical coordinates use no centering or rescaling.")
        self.explanation.configure(text=f"{certificate} in {units} coordinates. {suffix} "
            "Warm-up inputs are included in the certificate. "
            + ("Predicts NN output + learned D × input; their sum is projected within the energy budget. " if projected else "")
            + ("Learned gamma stays within the declared limits; saved results report its fitted value. " if learned_gamma else "") +
            "Initial context is declared once and never taken from inputs or targets.")

    def _layout(self):
        # Two columns keep the operator controls readable in a narrow window.
        groups = (("spending_readout", "spending_core", "spending_linear_init"),
                  ("spending_gamma_mode", "spending_gamma", "spending_gamma_min", "spending_gamma_max", "spending_beta_mode"),
                  ("spending_fixed_beta", "spending_beta_initial", "spending_initial_features", "spending_beta_width"),
                  ("spending_tcn_kernel", "spending_tcn_dilations", "spending_heads", "spending_dropout"),
                  ("spending_coordinates", "spending_storage", "spending_precision"))
        row = 0
        for group in groups:
            visible = [key for key in group if self.controls[key][1].winfo_manager()]
            for index, key in enumerate(visible):
                offset_row, column = divmod(index, 2)
                for offset, widget in enumerate(self.controls[key]):
                    widget.grid(row=row + offset_row, column=2 * column + offset)
            row += (len(visible) + 1) // 2
        self.diagnostics.grid(row=row, column=0)
        self.explanation.grid(row=row + 1, column=0)
