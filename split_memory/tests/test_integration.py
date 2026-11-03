from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from split_memory import SplitMemoryConfig
from split_memory.benchmark import SplitMemorySim, load_checkpoint, save_checkpoint, settings_from_args
from split_memory.ui import DEFAULTS, command_arguments
from Test_files import benchmark_ui as ui
from Test_files import run_benchmarks as runner
from direct_spending.ui_settings import VALUE_DEFAULTS, TOGGLE_DEFAULTS


class Value:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def ui_panel(models=("split_memory",)):
    panel = ui.BenchmarkUI.__new__(ui.BenchmarkUI)
    panel.vars = {key: Value(value) for key, value in VALUE_DEFAULTS.items()}
    panel.model_list = SimpleNamespace(curselection=lambda: tuple(ui.VISIBLE_MODELS.index(name) for name in models))
    panel.bench_list = SimpleNamespace(curselection=lambda: (ui.BENCHMARKS.index("Cascaded_Tanks"),))
    panel.out_var = Value("/tmp/split-memory-tests")
    for key, value in TOGGLE_DEFAULTS.items(): setattr(panel, key, Value(value))
    return panel


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def test_registry_factory_parameter_budget_and_no_gain_certificate(self):
        self.assertEqual(set(ui.MODELS), set(runner.MODEL_ZOO))
        self.assertEqual(ui.VISIBLE_MODELS.count("split_memory"), 1)
        for reader in ("recurrent", "pointwise", "linear"):
            cfg = runner.GlobalModelConfig(split_memory=SplitMemoryConfig(reader=reader, heads=2, computation_width=4).to_dict())
            model = runner.build_model("split_memory", 2, 3, cfg)
            matched = runner.build_model("split_memory", 2, 3, cfg, param_budget=200)
            self.assertGreater(runner._count_params(matched), 0)
            self.assertIsNone(model.diagnostics()["certified_gain_bound"])
            self.assertEqual(model(torch.randn(2, 9, 2)).shape, (2, 9, 3))
            self.assertEqual(runner._effective_width(model), 2 if reader == "linear" else 4)

    def test_ui_all_readers_solvers_configuration_defaults_and_hidden_fields(self):
        for reader in ("recurrent", "pointwise", "linear"):
            for solver in ("fixed", "adaptive", "sequential"):
                panel = ui_panel(("gru", "split_memory"))
                panel.vars["split_reader"].set(reader)
                panel.vars["split_solver"].set(solver)
                panel.vars["split_memory_heads"].set("3")
                panel.vars["split_computation_width"].set("7")
                args = runner.build_parser().parse_args(panel._build_command()[3:])
                cfg = settings_from_args(args)
                self.assertEqual(cfg["reader"], reader)
                self.assertEqual(cfg["heads"], 3)
                if reader == "recurrent": self.assertEqual(cfg["solver"], solver)
                config = panel._configuration()
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "ui.json"; path.write_text(json.dumps(config))
                    with patch("sys.argv", ["runner", "--config", str(path), "--epochs", "1"]), patch.object(runner, "run") as run:
                        runner.main()
                        self.assertEqual(run.call_args.args[0].epochs, 1)
                        self.assertEqual(run.call_args.args[0].split_reader, reader)
        panel = ui_panel()
        panel.vars["split_reader"].set("linear")
        for key in ("split_rho", "split_tolerance", "split_fixed_sweeps", "split_computation_width"):
            panel.vars[key].set("invalid hidden value")
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertEqual(args.split_reader, "linear")
        self.assertNotIn("--d-state", panel._build_command())

    def test_tagged_checkpoints_reconstruct_precision_and_solver_overrides(self):
        normalizer = runner.Standardizer(torch.zeros(2), torch.ones(2), torch.zeros(3), torch.ones(3))
        x = torch.randn(2, 11, 2, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            for reader in ("recurrent", "pointwise", "linear"):
                model = SplitMemorySim(2, 3, SplitMemoryConfig(reader=reader, heads=2, computation_width=4, fixed_sweeps=11)).double()
                expected = model(x)
                save_checkpoint(path, model, normalizer, {"train_losses": [1.]}, {"seed": 0})
                restored, payload = load_checkpoint(path)
                torch.testing.assert_close(restored(x), expected, atol=1e-13, rtol=1e-13)
                self.assertEqual(next(restored.parameters()).dtype, torch.float64)
                self.assertEqual(payload["config"], model.split_config.to_dict())
                sequential, _ = load_checkpoint(path, solver="sequential", memory_backend="sequential")
                torch.testing.assert_close(sequential(x), expected, atol=1e-12, rtol=1e-12)
            torch.save({"format": "direct_spending_v1", "family": "direct_spending"}, path)
            with self.assertRaises(ValueError): load_checkpoint(path)

    def test_prediction_plot_includes_outputs_outside_target_range(self):
        bench = runner.load_benchmark("Cascaded_Tanks")
        norm = runner.Standardizer.fit(bench.train)
        model = SplitMemorySim(bench.n_u, bench.n_y, SplitMemoryConfig(heads=2, computation_width=4))
        with torch.no_grad():
            model.core.out_proj.bias.fill_(100)
        with tempfile.TemporaryDirectory() as temp:
            monitor = runner.TrainingMonitor(out_dir=Path(temp), bench=bench, model_name="split_memory",
                norm=norm, device=torch.device("cpu"), total_epochs=1, plot_every=1,
                gif=False, show=False, max_preview_len=64)
            monitor.update(0, [1.], [1.], model)
            for ax, prediction in zip(monitor._panel_axes, monitor._prev):
                low, high = ax.get_ylim()
                self.assertLess(low, float(prediction.min()))
                self.assertGreater(high, float(prediction.max()))
            artifacts = monitor.finalize()
            self.assertTrue(Path(artifacts["figure"]).is_file())

    def test_native_dataset_training_evaluation_masks_reports_and_failure_status(self):
        with tempfile.TemporaryDirectory() as temp:
            args = runner.build_parser().parse_args(["--models", "split_memory", "--benchmarks", "Cascaded_Tanks",
                "--epochs", "1", "--batch-size", "1", "--param-budget", "off", "--split-memory-heads", "2",
                "--split-computation-width", "4", "--split-diagnostics", "--device", "cpu", "--no-plot",
                "--no-show", "--out", temp])
            runner.run(args)
            row = json.loads((Path(temp) / "report.json").read_text())["results"][0]
            self.assertEqual(row["status"], "ok", row.get("error"))
            self.assertEqual(row["epochs_run"], 1)
            self.assertIsNone(row["certified_gain_bound"])
            self.assertTrue(row["validation_sequences"][0]["solver_diagnostics"]["converged"])
            self.assertEqual(row["validation_sequences"][0]["init_window"], 50)
            model, payload = load_checkpoint(row["artifacts"]["checkpoint"])
            norm = runner.Standardizer(**payload["normalizer"])
            bench = runner.load_benchmark("Cascaded_Tanks")
            metrics, _ = runner.evaluate_validation(model, bench, norm, torch.device("cpu"))
            self.assertAlmostEqual(row["rmse"], metrics[0]["rmse"], places=6)
            args.split_fixed_sweeps, args.split_tolerance = 1, 1e-12
            runner.run(args)
            failed = json.loads((Path(temp) / "report.json").read_text())["results"][0]
            self.assertIn("ConvergenceError", failed["status"])
            self.assertFalse(failed["solver_failure"]["converged"])


@unittest.skipUnless(os.environ.get("BENCHMARK_UI_TK_TESTS") == "1", "native Tk test is opt-in")
class NativeUITests(unittest.TestCase):
    def test_real_run_button_configuration_results_and_checkpoint(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            output = Path(os.environ.get("SPLIT_MEMORY_UI_SMOKE_OUT", temp))
            output.mkdir(parents=True, exist_ok=True)
            root = tk.Tk(); root.withdraw()
            panel = None
            try:
                panel = ui.BenchmarkUI(root)
                panel.model_list.selection_clear(0, "end")
                panel.model_list.selection_set(ui.VISIBLE_MODELS.index("split_memory"))
                panel._update_model_options()
                for key, value in dict(epochs="1", batch_size="1", device="cpu", param_budget="off",
                    plot_every="1", plot_max_length="64", split_memory_heads="2", split_computation_width="4",
                    split_fixed_sweeps="32", split_solver="fixed", split_diagnostics="1").items():
                    panel.vars[key].set(value)
                panel.show_var.set(False); panel.gif_var.set(False)
                panel.out_var.set(str(output))
                (output / "ui-configuration.json").write_text(json.dumps(panel._configuration(), indent=2))
                panel.run_btn.invoke()
                self.assertIsNotNone(panel.proc)
                # Exercise the app's real event loop. Manual repeated update() can
                # wait indefinitely inside AppKit on this macOS/Tk build.
                def completed():
                    if panel.proc is None:
                        root.quit()
                    else:
                        root.after(50, completed)
                root.after(50, completed)
                root.after(75_000, root.quit)
                root.deiconify()
                root.mainloop()
                self.assertIsNone(panel.proc, "UI-launched job timed out")
                self.assertEqual(panel.status.cget("text"), "done (exit 0)")
                report = json.loads((output / "report.json").read_text())
                row = report["results"][0]
                self.assertEqual(row["status"], "ok", row.get("error"))
                validation = row["validation_sequences"][0]
                full_diag = row["solver_diagnostics"]["solver"]
                self.assertEqual(full_diag["cell_evaluations"], validation["length"] * full_diag["cell_passes"])
                self.assertEqual(full_diag, validation["solver_diagnostics"])
                self.assertEqual(panel.results_panel.report_path, output / "report.json")
                self.assertEqual(len(panel.results_panel.table.get_children()), 1)
                self.assertIsNotNone(panel.results_panel.image)
                self.assertIn("split_memory", panel.results_panel.text.get("1.0", "end"))
                model, payload = load_checkpoint(row["artifacts"]["checkpoint"])
                norm = runner.Standardizer(**payload["normalizer"])
                bench = runner.load_benchmark("Cascaded_Tanks")
                metrics, _ = runner.evaluate_validation(model, bench, norm, torch.device("cpu"))
                self.assertAlmostEqual(metrics[0]["rmse"], row["rmse"], places=6)
                (output / "ui-log.txt").write_text(panel.log.get("1.0", "end"))
            finally:
                if panel is not None and panel.proc is not None:
                    panel.proc.terminate(); panel.proc.wait(timeout=10)
                root.destroy()

    def test_model_options_and_new_parameters_saved_as_defaults(self):
        import tkinter as tk
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            root = tk.Tk(); root.withdraw()
            try:
                panel = ui.BenchmarkUI(root)
                panel.model_list.selection_clear(0, "end")
                panel.model_list.selection_set(ui.VISIBLE_MODELS.index("split_memory"))
                panel._update_model_options()
                options = panel.split_memory_options
                for reader in ("recurrent", "pointwise", "linear"):
                    panel.vars["split_reader"].set(reader)
                    for solver in ("fixed", "adaptive", "sequential"):
                        panel.vars["split_solver"].set(solver)
                        root.update_idletasks()
                        for key, widgets in options.controls.items():
                            self.assertEqual(bool(widgets[1].winfo_manager()), key in __import__("split_memory.ui", fromlist=["active_keys"]).active_keys(panel._value))
                        runner.build_parser().parse_args(panel._build_command()[3:])
                panel.vars["split_reader"].set("recurrent")
                panel.vars["split_solver"].set("adaptive")
                panel.vars["split_memory_heads"].set("3")
                panel.vars["split_rho"].set("0.6")
                panel._save_defaults()
                second = tk.Toplevel(root); second.withdraw()
                try:
                    restored = ui.BenchmarkUI(second)
                    self.assertEqual(restored._configuration(), panel._configuration())
                finally: second.destroy()
                for size in ("900x650", "1120x920"):
                    root.geometry(size); root.deiconify(); root.update_idletasks()
                    self.assertTrue(panel.run_btn.winfo_ismapped())
                    for key, widgets in options.controls.items():
                        widget = widgets[1]
                        if widget.winfo_manager():
                            self.assertLessEqual(widget.winfo_x()+widget.winfo_width(), widget.master.winfo_width()+1, key)
            finally: root.destroy()
