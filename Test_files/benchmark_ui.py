#!/usr/bin/env python
# file: Test_files/benchmark_ui.py
"""A tiny Tkinter control panel for run_benchmarks.py.

Pick the benchmark(s), model(s) and all the knobs (n_layers, init, gamma, ...)
from dropdowns/spinboxes and hit Run. It builds the matching
``run_benchmarks.py`` command, launches it as a subprocess (so the live animated
plot still appears exactly as from the CLI), and streams the log here.

    python Test_files/benchmark_ui.py

No extra dependencies — uses the standard-library ``tkinter``. If tkinter is
missing, install it (e.g. ``conda install tk``).
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import json
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_RUNNER = _HERE / "run_benchmarks.py"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from direct_spending.config import MODEL_SPECS as SPENDING_MODELS
from direct_spending.ui import command_arguments as spending_arguments
from direct_spending.ui_layout import LABELS, ModelOptions, ScrollableSettings, fields
from direct_spending.ui_numbers import real_number, whole_number, learning_rate_preview
from direct_spending.ui_settings import (UI_MODELS, VALUE_DEFAULTS, TOGGLE_DEFAULTS,
    SSM_MODELS, architecture_keys, normalize_configuration, write_configuration)
from direct_spending.ui_results import RunResults
from direct_spending.ui_parameter_count import show_parameter_count
from split_memory.ui import command_arguments as split_arguments

# Keep the complete registry available to existing integrations, while the
# window shows one spending family with a core selector. No torch import here.
BENCHMARKS = [
    "EMPS", "CED", "Cascaded_Tanks", "Silverbox", "WienerHammerBenchMark",
    "ParWH", "F16", "Industrial_robot", "WienerHammerstein_Process_Noise",
]
VISIBLE_MODELS = list(UI_MODELS)
MODELS = VISIBLE_MODELS + [name for name in SPENDING_MODELS if name != "direct_spending"]
LEGACY_MODELS = {"l2ru", "zak", "l2nt"}
_DEFAULTS_PATH = _REPO_ROOT / "direct_spending" / "configs" / "ui-defaults.json"


class BenchmarkUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Nonlinear benchmarks")
        root.geometry("1120x920")
        root.minsize(900, 650)
        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str]" = queue.Queue()
        self.vars = {key: tk.StringVar(value=value) for key, value in VALUE_DEFAULTS.items()}
        for key, value in TOGGLE_DEFAULTS.items():
            setattr(self, key, tk.BooleanVar(value=value))
        self.spending_options = None

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(2, weight=2)
        self.settings = ScrollableSettings(outer)
        self.settings.frame.grid(row=0, column=0, sticky="nsew")
        body = self.settings.body

        # Everything in this first box describes the run, independently of architecture.
        shared = ttk.LabelFrame(body, text="Shared run settings", padding=8)
        shared.pack(fill="x")
        self.shared_frame = shared
        selections = ttk.Frame(shared)
        selections.pack(fill="x")
        for index, (title, names, default, attribute) in enumerate((
            ("Benchmarks", BENCHMARKS, "Cascaded_Tanks", "bench_list"),
            ("Models", VISIBLE_MODELS, "tv", "model_list"),
        )):
            group = ttk.LabelFrame(selections, text=title + "  (Cmd/Ctrl-click for several)", padding=5)
            group.pack(side="left", fill="both", expand=True, padx=(0, 4) if index == 0 else (4, 0))
            widget = tk.Listbox(group, selectmode="extended", height=5, exportselection=False)
            for name in names:
                widget.insert("end", f"{name} (legacy)" if name in LEGACY_MODELS else name)
            widget.selection_set(names.index(default))
            widget.see(names.index(default))
            scrollbar = ttk.Scrollbar(group, orient="vertical", command=widget.yview)
            widget.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            widget.pack(side="left", fill="both", expand=True)
            setattr(self, attribute, widget)

        common = ttk.Frame(shared, padding=(0, 6, 0, 0))
        common.pack(fill="x")
        self.common_controls = fields(common, ("device", "epochs", "batch_size", "lr",
            "weight_decay", "grad_clip", "seed", "report_metric", "param_budget",
            "plot_every", "plot_max_length"), self.vars)
        toggles = ttk.Frame(shared)
        toggles.pack(fill="x", pady=5)
        for text, key in (("Live training plot", "show_var"), ("Save GIF", "gif_var"),
                          ("Mixed precision (CUDA)", "amp_var"), ("Compile models", "compile_var")):
            ttk.Checkbutton(toggles, text=text, variable=getattr(self, key)).pack(side="left", padx=(0, 14))
        self.compile_controls = ttk.Frame(toggles)
        fields(self.compile_controls, ("compile_mode",), self.vars, columns=1)
        self.compile_var.trace_add("write", lambda *_: self._update_compile_options())
        self._update_compile_options()
        output = ttk.Frame(shared)
        output.pack(fill="x", pady=(3, 0))
        ttk.Label(output, text="Output directory").pack(side="left")
        self.out_var = tk.StringVar(value=str(_REPO_ROOT / "Test_files" / "benchmark_runs"))
        ttk.Entry(output, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(output, text="Browse", command=self._browse).pack(side="left")

        self.model_options = ModelOptions(body, self)
        self.model_list.bind("<<ListboxSelect>>", self._update_model_options)
        self._update_model_options()

        actions = ttk.Frame(outer, padding=(0, 8))
        actions.grid(row=1, column=0, sticky="ew")
        self.run_btn = ttk.Button(actions, text="▶  Run", command=self._run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="■  Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(actions, text="Save as defaults", command=self._save_defaults).pack(side="left", padx=(14, 6))
        ttk.Button(actions, text="Save configuration…", command=self._save_config).pack(side="left", padx=6)
        ttk.Button(actions, text="Load configuration…", command=self._load_config).pack(side="left", padx=6)
        self.status = ttk.Label(actions, text="idle")
        self.status.pack(side="right")

        tabs = ttk.Notebook(outer)
        tabs.grid(row=2, column=0, sticky="nsew")
        log_frame = ttk.Frame(tabs, padding=4)
        tabs.add(log_frame, text="Log / progress")
        toolbar = ttk.Frame(log_frame)
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(toolbar, text="Clear log", command=lambda: self._set_log("")).pack(side="left")
        log_body = ttk.Frame(log_frame)
        log_body.pack(fill="both", expand=True)
        self.log = tk.Text(log_body, wrap="none", height=8, bg="#0b1118", fg="#d7dde8",
                           insertbackground="#d7dde8", state="disabled")
        scrollbar = ttk.Scrollbar(log_body, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.results_panel = RunResults(tabs)
        results_toolbar = ttk.Frame(self.results_panel.results)
        results_toolbar.pack(fill="x", pady=(0, 4), before=self.results_panel.table)
        ttk.Button(results_toolbar, text="Open saved run…", command=self._open_report).pack(side="left")
        ttk.Button(results_toolbar, text="Storage trace", command=self.results_panel.show_storage_trace).pack(side="left", padx=6)
        tabs.select(log_frame)
        self._load_defaults()
        self.settings.bind_wheel()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    def _update_compile_options(self):
        if self.compile_var.get():
            self.compile_controls.pack(side="left")
        else:
            self.compile_controls.pack_forget()

    # ---- helpers --------------------------------------------------------------
    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get() or str(_REPO_ROOT))
        if d:
            self.out_var.set(d)

    def _snapshot(self):
        return {"format": "nonlinear_benchmark_ui_v1",
                "values": {key: var.get() for key, var in self.vars.items()},
                "benchmarks": self._selected(self.bench_list, BENCHMARKS),
                "models": self._selected(self.model_list, VISIBLE_MODELS), "out": self.out_var.get(),
                "toggles": {key: getattr(self, key).get() for key in TOGGLE_DEFAULTS}}

    def _configuration(self):
        return {**self._snapshot(), "argv": self._build_command()[3:]}

    def _save_config(self):
        try:
            config = self._configuration()
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Configuration", "*.json")])
            if path:
                write_configuration(path, config)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Configuration", str(exc))

    def _save_defaults(self):
        try:
            write_configuration(_DEFAULTS_PATH, self._configuration())
        except (ValueError, OSError) as exc:
            messagebox.showerror("Defaults", str(exc))
            return
        self.status.configure(text="defaults saved")
        self._append(f"[ui] Saved defaults for next launch: {_DEFAULTS_PATH}\n")

    def _load_defaults(self):
        if not _DEFAULTS_PATH.is_file():
            return
        try:
            self._apply_configuration(json.loads(_DEFAULTS_PATH.read_text()))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self._append(f"[ui] Could not load saved defaults; using built-in settings: {exc}\n")
        else:
            self.status.configure(text="saved defaults loaded")

    def _restore_controls(self, config):
        for key, value in config["values"].items():
            self.vars[key].set(value)
        for key, value in config["toggles"].items():
            getattr(self, key).set(value)
        for widget, names, chosen in ((self.model_list, VISIBLE_MODELS, config["models"]),
                                      (self.bench_list, BENCHMARKS, config["benchmarks"])):
            widget.selection_clear(0, "end")
            for name in chosen:
                widget.selection_set(names.index(name))
            if chosen:
                widget.see(names.index(chosen[0]))
        self.out_var.set(config["out"])
        self._update_model_options()

    def _apply_configuration(self, config):
        config = normalize_configuration(config, BENCHMARKS)
        previous = self._snapshot()
        try:
            self._restore_controls(config)
            self._build_command()
        except (ValueError, KeyError, TypeError):
            self._restore_controls(previous)
            raise

    def _load_config(self):
        path = filedialog.askopenfilename(filetypes=[("Configuration", "*.json")])
        if path:
            try:
                self._apply_configuration(json.loads(Path(path).read_text()))
            except (ValueError, OSError, KeyError, TypeError) as exc:
                messagebox.showerror("Configuration", str(exc))

    def _open_report(self):
        path = filedialog.askopenfilename(initialdir=self.out_var.get(),
                                         filetypes=[("Saved report", "report.json"), ("JSON", "*.json")])
        if path:
            try:
                self.results_panel.load(path)
            except (ValueError, OSError) as exc:
                messagebox.showerror("Saved run", str(exc))

    def _set_log(self, text):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", text)
        self.log.configure(state="disabled")

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _selected(self, listbox, names):
        return [names[i] for i in listbox.curselection()]

    def _update_model_options(self, _event=None):
        self.model_options.select(self._selected(self.model_list, VISIBLE_MODELS))
        self.settings.bind_wheel()

    def _value(self, key):
        # Defaults also support older programmatic control dictionaries.
        return str(self.vars[key].get()).strip() if key in self.vars else VALUE_DEFAULTS[key]

    def _show_parameter_count(self):
        try:
            command = self._build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid options", str(exc))
            return None
        return show_parameter_count(self.root, command, _REPO_ROOT)

    def _build_command(self):
        benches = self._selected(self.bench_list, BENCHMARKS)
        models = self._selected(self.model_list, VISIBLE_MODELS)
        if not benches:
            raise ValueError("Select at least one benchmark.")
        if not models:
            raise ValueError("Select at least one model.")

        g = self._value
        cmd = [sys.executable, "-u", str(_RUNNER),
               "--benchmarks", *benches,
               "--models", *models,
               "--epochs", g("epochs"), "--batch-size", g("batch_size"), "--lr", g("lr"),
               "--plot-every", g("plot_every"),
               "--plot-max-length", g("plot_max_length"),
               "--weight-decay", g("weight_decay"), "--grad-clip", g("grad_clip"), "--seed", g("seed"),
               "--report-metric", g("report_metric"),
               "--param-budget", g("param_budget"),
               "--out", self.out_var.get().strip()]
        if g("device") != "auto":
            cmd += ["--device", g("device")]
        active = architecture_keys(models, g("spending_core"), g("ff"), g("metric_ff"))
        self._validate_run_values(active)
        for key in ("d_model", "d_state", "n_layers", "d_hidden", "nl_layers", "gamma", "rnn_hidden", "rnn_layers"):
            if key in active:
                cmd += ["--" + key.replace("_", "-"), g(key)]
        if "ff" in active and g("ff") != "auto":
            cmd += ["--ff", g("ff")]
        if "mode" in active and g("mode") != "auto":
            cmd += ["--mode", g("mode")]
        if "lru" in models:
            cmd += [
                "--lru-rmin", g("lru_rmin"),
                "--lru-rmax", g("lru_rmax"),
                "--lru-max-phase", g("lru_max_phase"),
            ]
        if "l2ru" in models or "zak" in models:
            cmd += ["--init", g("init")]
        if "l2ru" in models:
            cmd += [
                "--l2ru-eye-scale", g("l2ru_eye_scale"),
                "--l2ru-rand-scale", g("l2ru_rand_scale"),
            ]
        if "l2n" in models or "metric_l2n" in models:
            cmd += [
                "--l2n-state-metric", g("l2n_state_metric"),
                "--l2n-rho", g("l2n_rho"),
                "--l2n-max-phase", g("l2n_max_phase"),
                "--l2n-phase-center", g("l2n_phase_center"),
                "--l2n-offdiag-scale", g("l2n_offdiag_scale"),
                ("--l2n-random-phase" if self.l2n_random_phase_var.get()
                 else "--no-l2n-random-phase"),
            ]
        if "defect" in models or "metric_defect" in models:
            for key in (
                "defect_block_size", "defect_rho", "defect_max_radius",
                "defect_factor_margin", "defect_init_scale", "defect_max_phase",
                "defect_phase_center",
                "defect_state_metric",
            ):
                cmd += ["--" + key.replace("_", "-"), g(key)]
            cmd += ["--defect-random-phase" if self.defect_random_phase_var.get()
                    else "--no-defect-random-phase"]
        if "metric_defect" in models or "metric_l2n" in models:
            for key in ("metric_activation", "metric_ff", "metric_init_feedthrough"):
                cmd += ["--" + key.replace("_", "-"), g(key)]
        if "tv" in models:
            cmd += [
                "--tv-init-rho", g("tv_init_rho"),
                "--tv-init-delta0", g("tv_init_delta0"),
                "--tv-init-param-scale", g("tv_init_param_scale"),
            ]
        if "tvc" in models:
            cmd += [
                "--tvc-init-rho", g("tvc_init_rho"),
                "--tvc-init-delta0", g("tvc_init_delta0"),
                "--tvc-init-param-scale", g("tvc_init_param_scale"),
                "--tvc-init-sign", g("tvc_init_sign"),
                "--tvc-init-b", g("tvc_init_b"),
                "--tvc-init-c", g("tvc_init_c"),
                "--tvc-init-d", g("tvc_init_d"),
            ]
        if "ren" in models:
            cmd += [
                "--ren-dim-internal", g("ren_dim_internal"),
                "--ren-dim-nl", g("ren_dim_nl"),
            ]
        if set(models).intersection(SPENDING_MODELS):
            cmd += spending_arguments(self.vars, models)
        if "split_memory" in models:
            cmd += split_arguments(self.vars)
        cmd += ["--show"] if self.show_var.get() else ["--no-show"]
        if not self.gif_var.get():
            cmd += ["--no-gif"]
        if self.amp_var.get():
            cmd += ["--amp"]
        if self.compile_var.get():
            cmd += ["--compile", "--compile-mode", g("compile_mode")]
        if set(models).intersection(SSM_MODELS) and self.pcg_var.get():
            cmd += ["--per-channel-gates"]
        if set(models).intersection(("tv", "tvc")):
            cmd += ["--use-cuda-graph"] if self.cudagraph_var.get() else ["--no-use-cuda-graph"]
        return cmd

    def _validate_run_values(self, active):
        """Catch malformed visible fields before saving defaults or launching a child."""
        get = self._value
        integer_minima = dict(epochs=1, batch_size=1, plot_every=1, plot_max_length=1, seed=0,
                             d_model=1, d_state=1, n_layers=0, d_hidden=1, nl_layers=1,
                             rnn_hidden=1, rnn_layers=1, ren_dim_internal=1, ren_dim_nl=1)
        for key, minimum in integer_minima.items():
            if key not in active and key not in ("epochs", "batch_size", "plot_every", "plot_max_length", "seed"):
                continue
            if whole_number(get(key), LABELS[key]) < minimum:
                raise ValueError(f"{LABELS[key]} must be at least {minimum}")
        for key in ("lr", "weight_decay", "grad_clip"):
            value = real_number(get(key), LABELS[key])
            if value < 0 or (key == "lr" and value == 0):
                raise ValueError(f"{LABELS[key]} must be {'positive' if key == 'lr' else 'nonnegative'}")
        if "gamma" in active and get("gamma").lower() not in ("auto", "none"):
            value = real_number(get("gamma"), LABELS["gamma"])
            if value <= 0:
                raise ValueError("Gain limit must be auto, none, or a positive number")
        budget = get("param_budget").lower()
        if budget not in ("off", "none", "0", "") and budget not in MODELS and budget not in SPENDING_MODELS:
            if whole_number(budget, "Parameter budget") <= 0:
                raise ValueError("Parameter budget must be a positive integer, off, or a model name")
        if not self.out_var.get().strip():
            raise ValueError("Choose an output directory")

    # ---- run / stream ---------------------------------------------------------
    def _run(self):
        if self.proc is not None:
            messagebox.showinfo("Busy", "A run is already in progress.")
            return
        try:
            cmd = self._build_command()
        except ValueError as exc:
            messagebox.showerror("Invalid options", str(exc))
            return

        self._set_log("Learning rate: " + learning_rate_preview(self._value("lr"))
                      + " per optimizer step\n$ " + " ".join(cmd) + "\n\n")
        self._run_started = time.time()
        output_path = Path(self.out_var.get()).expanduser()
        self._active_output = (output_path if output_path.is_absolute() else _REPO_ROOT / output_path).resolve()
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(_REPO_ROOT), env=env, text=True, bufsize=1,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            self.proc = None
            return

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text="running…")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        try:
            for line in iter(proc.stdout.readline, ""):
                self.q.put(line)
            proc.stdout.close()
        except Exception as exc:  # pragma: no cover
            self.q.put(f"\n[ui] reader error: {exc}\n")
        finally:
            proc.wait()
            self.q.put(None)  # sentinel: process finished

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:
                    rc = self.proc.returncode if self.proc else None
                    self._append(f"\n[done] exit code {rc}\n")
                    self.proc = None
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status.configure(text=f"done (exit {rc})")
                    report = self._active_output / "report.json"
                    if report.is_file() and report.stat().st_mtime >= self._run_started:
                        try:
                            self.results_panel.load(report)
                        except (ValueError, OSError) as exc:
                            self._append(f"[ui] report view: {exc}\n")
                else:
                    self._append(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _stop(self):
        if self.proc is not None:
            self.status.configure(text="stopping…")
            self.proc.terminate()

    def _on_close(self):
        if self.proc is not None and messagebox.askyesno(
                "Quit", "A run is in progress. Stop it and quit?"):
            self.proc.terminate()
        elif self.proc is not None:
            return
        self.root.destroy()


def main():
    if not _RUNNER.exists():
        raise SystemExit(f"Could not find runner at {_RUNNER}")
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    BenchmarkUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
