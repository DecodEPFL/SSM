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
import math
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_RUNNER = _HERE / "run_benchmarks.py"

# These mirror run_benchmarks.py (kept here so the UI starts instantly without
# importing torch). Update alongside the registries there.
BENCHMARKS = [
    "EMPS", "CED", "Cascaded_Tanks", "Silverbox", "WienerHammerBenchMark",
    "ParWH", "F16", "Industrial_robot", "WienerHammerstein_Process_Noise",
]
MODELS = ["lru", "l2ru", "l2n", "tv", "tvc", "raven", "ctransformer", "lstm", "gru"]
INITS = ["eye", "rand"]
FFS = ["auto", "GLU", "MLP", "LGLU2", "MBLIP", "BLGLU2", "BudgetedLGLU2", "TLIP", "LMLP"]
DEVICES = ["auto", "cpu", "cuda", "mps"]
GAMMAS = ["auto", "none", "0.5", "1", "2", "5", "10", "20"]
# SSM execution mode. "auto" = fastest per model (conv for lru/l2n, scan otherwise).
MODES = ["auto", "loop", "scan", "conv"]


class BenchmarkUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Nonlinear-benchmarks runner")
        root.geometry("1080x760")
        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str]" = queue.Queue()

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)

        # ---- selections (benchmarks + models side by side) --------------------
        top = ttk.Frame(outer)
        top.pack(fill="x")

        bf = ttk.LabelFrame(top, text="Benchmarks  (Cmd/Ctrl-click for several)", padding=6)
        bf.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.bench_list = tk.Listbox(bf, selectmode="extended", height=8, exportselection=False)
        for b in BENCHMARKS:
            self.bench_list.insert("end", b)
        self.bench_list.selection_set(BENCHMARKS.index("Cascaded_Tanks"))
        self.bench_list.pack(fill="both", expand=True)

        mf = ttk.LabelFrame(top, text="Models  (Cmd/Ctrl-click for several)", padding=6)
        mf.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.model_list = tk.Listbox(mf, selectmode="extended", height=8, exportselection=False)
        for m in MODELS:
            self.model_list.insert("end", m)
        self.model_list.selection_set(MODELS.index("raven"))
        self.model_list.pack(fill="both", expand=True)

        # ---- options grid -----------------------------------------------------
        opt = ttk.LabelFrame(outer, text="Options", padding=8)
        opt.pack(fill="x", pady=8)
        self.vars: dict[str, tk.Variable] = {}

        def combo(parent, key, label, values, default, width=10, editable=False):
            self._row_label(parent, label)
            var = tk.StringVar(value=str(default))
            cb = ttk.Combobox(parent, textvariable=var, values=values, width=width,
                              state="normal" if editable else "readonly")
            cb.grid(row=self._r, column=self._c * 2 + 1, sticky="w", padx=(2, 14), pady=3)
            self.vars[key] = var
            self._advance()

        def spin(parent, key, label, frm, to, default, inc=1, width=8):
            self._row_label(parent, label)
            var = tk.StringVar(value=str(default))
            sp = ttk.Spinbox(parent, from_=frm, to=to, increment=inc, textvariable=var, width=width)
            sp.grid(row=self._r, column=self._c * 2 + 1, sticky="w", padx=(2, 14), pady=3)
            self.vars[key] = var
            self._advance()

        def entry(parent, key, label, default, width=10):
            self._row_label(parent, label)
            var = tk.StringVar(value=str(default))
            en = ttk.Entry(parent, textvariable=var, width=width)
            en.grid(row=self._r, column=self._c * 2 + 1, sticky="w", padx=(2, 14), pady=3)
            self.vars[key] = var
            self._advance()

        self._r, self._c, self._ncols = 0, 0, 4
        combo(opt, "device", "device", DEVICES, "auto")
        spin(opt, "n_layers", "n_layers", 0, 16, 2)
        spin(opt, "d_model", "d_model", 2, 512, 16)
        spin(opt, "d_state", "d_state", 2, 512, 16, inc=2)
        spin(opt, "d_hidden", "d_hidden", 1, 512, 16)
        spin(opt, "nl_layers", "nl_layers", 1, 12, 3)
        combo(opt, "gamma", "gamma", GAMMAS, "auto", editable=True)
        combo(opt, "ff", "ff (override)", FFS, "auto")
        combo(opt, "mode", "exec mode", MODES, "auto")
        spin(opt, "epochs", "epochs", 1, 20000, 200)
        spin(opt, "batch_size", "batch_size", 1, 1024, 32)
        entry(opt, "lr", "lr", "3e-3")
        spin(opt, "plot_every", "plot_every", 1, 1000, 10)
        spin(opt, "plot_max_length", "plot max samples", 1, 10000000, 4000, inc=100)
        spin(opt, "raven_heads", "raven_heads", 1, 32, 4)
        spin(opt, "raven_slots", "raven_slots", 1, 256, 8)
        spin(opt, "raven_top_k", "raven_top_k", 1, 256, 2)
        combo(opt, "report_metric", "report metric", ["rmse", "nrmse", "fit", "r2", "mae"], "rmse")
        entry(opt, "param_budget", "param budget (model/int/off)", "lstm", width=10)

        # Model-specific controls are created once, then inserted only while the
        # corresponding model is selected so the main options stay compact.
        self.model_option_frames: dict[str, ttk.LabelFrame] = {}

        def model_panel(model, title):
            frame = ttk.LabelFrame(outer, text=title, padding=8)
            self.model_option_frames[model] = frame
            return frame

        def panel_entry(frame, index, key, label, default, width=9):
            row, column = divmod(index, 4)
            ttk.Label(frame, text=label).grid(
                row=row, column=2 * column, sticky="e", padx=(8, 2), pady=3,
            )
            var = tk.StringVar(value=str(default))
            ttk.Entry(frame, textvariable=var, width=width).grid(
                row=row, column=2 * column + 1, sticky="w", padx=(0, 14), pady=3,
            )
            self.vars[key] = var

        lru_panel = model_panel("lru", "LRU initialization")
        panel_entry(lru_panel, 0, "lru_rmin", "minimum radius", "0.8")
        panel_entry(lru_panel, 1, "lru_rmax", "maximum radius", "0.95")
        panel_entry(lru_panel, 2, "lru_max_phase", "maximum phase", str(2 * math.pi))

        l2ru_panel = model_panel("l2ru", "L2RU initialization")
        ttk.Label(l2ru_panel, text="mode").grid(row=0, column=0, sticky="e", padx=(8, 2), pady=3)
        self.vars["init"] = tk.StringVar(value="eye")
        ttk.Combobox(
            l2ru_panel, textvariable=self.vars["init"], values=INITS,
            width=9, state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=(0, 14), pady=3)
        panel_entry(l2ru_panel, 1, "l2ru_eye_scale", "eye scale", "0.01")
        panel_entry(l2ru_panel, 2, "l2ru_rand_scale", "random scale", "1.0")

        l2n_panel = model_panel("l2n", "L2N initialization")
        panel_entry(l2n_panel, 0, "l2n_rho", "target pole radius", "0.9")
        panel_entry(l2n_panel, 1, "l2n_max_phase", "phase half-width", "0.04")
        panel_entry(l2n_panel, 2, "l2n_phase_center", "phase center", "0.0")
        panel_entry(l2n_panel, 3, "l2n_offdiag_scale", "off-diagonal std", "0.05")
        self.l2n_random_phase_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            l2n_panel,
            text="random phases (otherwise all use the phase center)",
            variable=self.l2n_random_phase_var,
        ).grid(row=1, column=0, columnspan=8, sticky="w", padx=8, pady=(3, 0))

        tv_panel = model_panel("tv", "TV initialization")
        panel_entry(tv_panel, 0, "tv_init_rho", "state decay", "0.99")
        panel_entry(tv_panel, 1, "tv_init_delta0", "step size", "1.0")
        panel_entry(tv_panel, 2, "tv_init_param_scale", "parameter std", "0.02")

        tvc_panel = model_panel("tvc", "TVC initialization")
        panel_entry(tvc_panel, 0, "tvc_init_rho", "unsigned decay", "0.9")
        panel_entry(tvc_panel, 1, "tvc_init_delta0", "step size", "1.0")
        panel_entry(tvc_panel, 2, "tvc_init_param_scale", "parameter std", "0.02")
        panel_entry(tvc_panel, 3, "tvc_init_sign", "decay sign", "0.995")
        panel_entry(tvc_panel, 4, "tvc_init_b", "input coupling", "0.10")
        panel_entry(tvc_panel, 5, "tvc_init_c", "output coupling", "0.10")
        panel_entry(tvc_panel, 6, "tvc_init_d", "feedthrough", "0.10")

        # ---- toggles + output dir --------------------------------------------
        self.toggles = ttk.Frame(outer)
        self.toggles.pack(fill="x")
        self.show_var = tk.BooleanVar(value=True)
        self.gif_var = tk.BooleanVar(value=True)
        self.amp_var = tk.BooleanVar(value=False)
        self.pcg_var = tk.BooleanVar(value=False)
        self.cudagraph_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.toggles, text="live window (--show)", variable=self.show_var).pack(side="left")
        ttk.Checkbutton(self.toggles, text="save GIF", variable=self.gif_var).pack(side="left", padx=12)
        ttk.Checkbutton(self.toggles, text="mixed precision (--amp, CUDA)", variable=self.amp_var).pack(side="left")
        ttk.Checkbutton(self.toggles, text="per-channel gates", variable=self.pcg_var).pack(side="left", padx=12)
        ttk.Checkbutton(self.toggles, text="CUDA graph (tv/tvc)", variable=self.cudagraph_var).pack(side="left")

        self.model_list.bind("<<ListboxSelect>>", self._update_model_options)
        self._update_model_options()

        od = ttk.Frame(outer)
        od.pack(fill="x", pady=(6, 0))
        ttk.Label(od, text="output dir:").pack(side="left")
        self.out_var = tk.StringVar(value=str(_REPO_ROOT / "Test_files" / "benchmark_runs"))
        ttk.Entry(od, textvariable=self.out_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(od, text="Browse", command=self._browse).pack(side="left")

        # ---- action buttons ---------------------------------------------------
        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=8)
        self.run_btn = ttk.Button(btns, text="▶  Run", command=self._run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="■  Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Clear log", command=lambda: self._set_log("")).pack(side="left", padx=6)
        self.status = ttk.Label(btns, text="idle")
        self.status.pack(side="right")

        # ---- log --------------------------------------------------------------
        logf = ttk.LabelFrame(outer, text="Log", padding=4)
        logf.pack(fill="both", expand=True)
        self.log = tk.Text(logf, wrap="none", height=16, bg="#0b1118", fg="#d7dde8",
                           insertbackground="#d7dde8")
        ys = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.configure(state="disabled")

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll)

    # ---- grid bookkeeping -----------------------------------------------------
    def _row_label(self, parent, label):
        ttk.Label(parent, text=label).grid(row=self._r, column=self._c * 2, sticky="e", padx=(8, 0), pady=3)

    def _advance(self):
        self._c += 1
        if self._c >= self._ncols:
            self._c = 0
            self._r += 1

    # ---- helpers --------------------------------------------------------------
    def _browse(self):
        d = filedialog.askdirectory(initialdir=self.out_var.get() or str(_REPO_ROOT))
        if d:
            self.out_var.set(d)

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
        selected = set(self._selected(self.model_list, MODELS))
        for model, frame in self.model_option_frames.items():
            if model in selected:
                if not frame.winfo_manager():
                    frame.pack(fill="x", pady=(0, 8), before=self.toggles)
            elif frame.winfo_manager():
                frame.pack_forget()

    def _build_command(self):
        benches = self._selected(self.bench_list, BENCHMARKS)
        models = self._selected(self.model_list, MODELS)
        if not benches:
            raise ValueError("Select at least one benchmark.")
        if not models:
            raise ValueError("Select at least one model.")

        g = lambda k: self.vars[k].get().strip()
        cmd = [sys.executable, "-u", str(_RUNNER),
               "--benchmarks", *benches,
               "--models", *models,
               "--epochs", g("epochs"), "--batch-size", g("batch_size"), "--lr", g("lr"),
               "--plot-every", g("plot_every"),
               "--plot-max-length", g("plot_max_length"),
               "--d-model", g("d_model"), "--d-state", g("d_state"), "--n-layers", g("n_layers"),
               "--d-hidden", g("d_hidden"), "--nl-layers", g("nl_layers"),
               "--raven-heads", g("raven_heads"), "--raven-slots", g("raven_slots"),
               "--raven-top-k", g("raven_top_k"),
               "--gamma", g("gamma"),
               "--report-metric", g("report_metric"),
               "--param-budget", g("param_budget"),
               "--out", self.out_var.get().strip()]
        if g("device") != "auto":
            cmd += ["--device", g("device")]
        if g("ff") != "auto":
            cmd += ["--ff", g("ff")]
        if g("mode") != "auto":
            cmd += ["--mode", g("mode")]
        if "lru" in models:
            cmd += [
                "--lru-rmin", g("lru_rmin"),
                "--lru-rmax", g("lru_rmax"),
                "--lru-max-phase", g("lru_max_phase"),
            ]
        if "l2ru" in models:
            cmd += [
                "--init", g("init"),
                "--l2ru-eye-scale", g("l2ru_eye_scale"),
                "--l2ru-rand-scale", g("l2ru_rand_scale"),
            ]
        if "l2n" in models:
            cmd += [
                "--l2n-rho", g("l2n_rho"),
                "--l2n-max-phase", g("l2n_max_phase"),
                "--l2n-phase-center", g("l2n_phase_center"),
                "--l2n-offdiag-scale", g("l2n_offdiag_scale"),
                ("--l2n-random-phase" if self.l2n_random_phase_var.get()
                 else "--no-l2n-random-phase"),
            ]
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
        cmd += ["--show"] if self.show_var.get() else ["--no-show"]
        if not self.gif_var.get():
            cmd += ["--no-gif"]
        if self.amp_var.get():
            cmd += ["--amp"]
        if self.pcg_var.get():
            cmd += ["--per-channel-gates"]
        cmd += ["--use-cuda-graph"] if self.cudagraph_var.get() else ["--no-use-cuda-graph"]
        return cmd

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

        self._set_log("$ " + " ".join(cmd) + "\n\n")
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
