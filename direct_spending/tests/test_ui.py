"""Behavioral checks for model-dependent fields and reusable UI defaults."""
import copy
import json
import os
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

from direct_spending.ui_settings import (SHARED_DEFAULTS, MODEL_DEFAULTS, VALUE_DEFAULTS,
    TOGGLE_DEFAULTS, architecture_keys, normalize_configuration, write_configuration)
from direct_spending.ui_numbers import learning_rate_preview
from Test_files import benchmark_ui as ui
from Test_files import run_benchmarks as runner


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Selection:
    def __init__(self, indices):
        self.indices = list(indices)

    def curselection(self):
        return tuple(sorted(self.indices))

    def selection_clear(self, *_):
        self.indices.clear()

    def selection_set(self, index):
        self.indices.append(index)

    def see(self, index):
        pass


def panel_for(models):
    panel = ui.BenchmarkUI.__new__(ui.BenchmarkUI)
    panel.vars = {key: Value(value) for key, value in VALUE_DEFAULTS.items()}
    panel.model_list = Selection(ui.MODELS.index(name) for name in models)
    panel.bench_list = Selection([ui.BENCHMARKS.index("Cascaded_Tanks")])
    panel.out_var = Value("/tmp/benchmark-ui-tests")
    for key, value in TOGGLE_DEFAULTS.items():
        setattr(panel, key, Value(value))
    panel.status = type("Status", (), {"configure": lambda *_args, **_kw: None})()
    panel._append = lambda text: None
    panel._update_model_options = lambda: None
    return panel


class UISettingsTests(unittest.TestCase):
    def test_learned_gamma_controls_roundtrip_and_ignore_hidden_limits(self):
        panel = panel_for(["direct_spending"])
        panel.vars["spending_core"].set("lstm")
        panel.vars["spending_gamma_min"].set("unfinished")
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertEqual(args.spending_core, "lstm")
        self.assertEqual(args.spending_gamma_mode, "fixed")
        panel.vars["spending_gamma_mode"].set("learned")
        with self.assertRaisesRegex(ValueError, "Minimum gamma"):
            panel._build_command()
        panel.vars["spending_gamma_min"].set("0.2")
        panel.vars["spending_gamma_max"].set("3")
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertEqual((args.spending_gamma_mode, args.spending_gamma_min, args.spending_gamma_max), ("learned", .2, 3.))
        restored = normalize_configuration(panel._configuration(), ui.BENCHMARKS)
        self.assertEqual(restored["values"]["spending_gamma_mode"], "learned")
        self.assertEqual(restored["values"]["spending_core"], "lstm")

    def test_incomplete_numeric_entry_names_field_and_never_launches(self):
        panel = panel_for(["direct_spending"])
        panel.proc = None
        panel.vars["lr"].set("3e-")
        with patch.object(ui.messagebox, "showerror") as error, patch.object(ui.subprocess, "Popen") as launch:
            panel._run()
        launch.assert_not_called()
        self.assertIn("Learning rate", error.call_args.args[1])
        self.assertIn("3e-", error.call_args.args[1])
        self.assertIn("0.003", error.call_args.args[1])
        panel.vars["lr"].set("3e-3")
        panel.vars["spending_core"].set("transformer")
        panel.vars["spending_dropout"].set("2e-")
        with self.assertRaisesRegex(ValueError, "Transformer dropout.*2e-"):
            panel._build_command()
        panel.vars["spending_dropout"].set("0")
        panel.vars["epochs"].set("2.5")
        with self.assertRaisesRegex(ValueError, "Epochs.*whole number"):
            panel._build_command()

    def test_rate_preview_matches_parsed_runner_value_without_changing_entry(self):
        panel = panel_for(["direct_spending"])
        for entry, preview, parsed in (("3e-3", "0.003", .003), ("3e3", "3000", 3000.),
                                      ("0.0003", "0.0003", .0003)):
            panel.vars["lr"].set(entry)
            self.assertEqual(learning_rate_preview(entry), preview)
            self.assertEqual(runner.build_parser().parse_args(panel._build_command()[3:]).lr, parsed)
            self.assertEqual(panel.vars["lr"].get(), entry)
        for invalid in ("3e-", "nan", "inf"):
            panel.vars["lr"].set(invalid)
            with self.assertRaisesRegex(ValueError, "Learning rate"):
                panel._build_command()

    def test_shared_fields_are_architecture_independent(self):
        self.assertFalse(set(SHARED_DEFAULTS) & set(MODEL_DEFAULTS))
        for model in ui.VISIBLE_MODELS:
            self.assertFalse(set(SHARED_DEFAULTS) & architecture_keys([model]))
        self.assertEqual(architecture_keys(["gru"]), {"rnn_hidden", "rnn_layers"})
        self.assertEqual(architecture_keys(["ren"]), {"gamma", "ren_dim_internal", "ren_dim_nl"})

    def test_hidden_architecture_values_do_not_affect_other_models(self):
        panel = panel_for(["gru"])
        for key in ("d_state", "d_hidden", "nl_layers", "d_model", "n_layers", "gamma", "spending_gamma"):
            panel.vars[key].set("invalid hidden value")
        panel.pcg_var.set(True)
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertFalse(args.per_channel_gates)
        self.assertEqual(args.gamma, "auto")
        self.assertEqual(args.rnn_hidden, 64)
        panel.model_list = Selection([ui.MODELS.index("direct_spending")])
        panel.vars["d_model"].set("8")
        panel.vars["spending_core"].set("tcn")
        panel.vars["spending_gamma"].set("1")
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertEqual(args.spending_core, "tcn")
        self.assertEqual(args.n_layers, 2)  # parser fallback, unused by TCN
        self.assertEqual(args.d_state, 16)

    def test_hidden_spending_beta_and_core_parameters_are_not_emitted(self):
        panel = panel_for(["direct_spending"])
        for key in ("spending_fixed_beta", "spending_beta_width", "spending_beta_initial",
                    "spending_heads", "spending_dropout", "spending_tcn_kernel"):
            panel.vars[key].set("invalid hidden value")
        command = panel._build_command()[3:]
        self.assertNotIn("--gamma", command)
        self.assertNotIn("--spending-heads", command)
        panel.vars["spending_beta_mode"].set("fixed")
        with self.assertRaises(ValueError):
            panel._build_command()

    def test_feedforward_only_exposes_dimensions_it_uses(self):
        self.assertNotIn("d_hidden", architecture_keys(["lru"]))
        self.assertNotIn("nl_layers", architecture_keys(["tv"]))
        self.assertIn("d_hidden", architecture_keys(["tv"]))
        self.assertIn("nl_layers", architecture_keys(["lru"], ff="MLP"))
        self.assertNotIn("d_hidden", architecture_keys(["metric_l2n"], metric_ff="none"))

    def test_rnn_controls_change_actual_factory_and_keep_existing_defaults(self):
        for model in ("lstm", "gru"):
            default = runner.build_model(model, 1, 1, runner.GlobalModelConfig())
            self.assertEqual((default.rnn.hidden_size, default.rnn.num_layers), (64, 2))
            panel = panel_for([model])
            panel.vars["rnn_hidden"].set("7")
            panel.vars["rnn_layers"].set("1")
            args = runner.build_parser().parse_args(panel._build_command()[3:])
            changed = runner.build_model(model, 1, 1, runner.GlobalModelConfig(
                rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers))
            self.assertEqual((changed.rnn.hidden_size, changed.rnn.num_layers), (7, 1))

    def test_save_defaults_and_startup_restore_complete_configuration(self):
        original = panel_for(["direct_spending", "gru"])
        original.vars["spending_core"].set("transformer")
        original.vars["spending_beta_mode"].set("learned")
        original.vars["spending_initial_features"].set("-1,2")
        original.vars["rnn_hidden"].set("19")
        original.vars["seed"].set("21")
        original.out_var.set("a different output")
        original.gif_var.set(False)
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            original._save_defaults()
            self.assertEqual(json.loads(ui._DEFAULTS_PATH.read_text())["models"], ["gru", "direct_spending"])
            restored = panel_for(["tv"])
            restored._load_defaults()
            self.assertEqual(restored._configuration(), original._configuration())
            # Invalid visible parameters never overwrite a good default file.
            previous = ui._DEFAULTS_PATH.read_bytes()
            original.vars["epochs"].set("0")
            with patch.object(ui.messagebox, "showerror") as error:
                original._save_defaults()
                error.assert_called_once()
            self.assertEqual(ui._DEFAULTS_PATH.read_bytes(), previous)

    def test_failed_configuration_is_transactional_and_corrupt_defaults_are_ignored(self):
        panel = panel_for(["gru"])
        before = panel._snapshot()
        invalid = copy.deepcopy(before)
        invalid["values"]["epochs"] = "bad"
        with self.assertRaises(ValueError):
            panel._apply_configuration(invalid)
        self.assertEqual(panel._snapshot(), before)
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            ui._DEFAULTS_PATH.write_text("not json")
            panel._load_defaults()
            self.assertEqual(panel._snapshot(), before)

    def test_old_single_alias_configs_migrate_and_multi_core_batches_remain_explicit(self):
        config = panel_for(["gru"])._configuration()
        config["models"] = ["spending_tcn", "gru"]
        normalized = normalize_configuration(config, ui.BENCHMARKS)
        self.assertEqual(normalized["models"], ["direct_spending", "gru"])
        self.assertEqual(normalized["values"]["spending_core"], "tcn")
        panel = panel_for(["tv"])
        panel._apply_configuration(config)
        self.assertEqual(runner.build_parser().parse_args(panel._build_command()[3:]).spending_core, "tcn")
        config["models"] = ["spending_tcn", "spending_gru"]
        with self.assertRaisesRegex(ValueError, "several spending cores"):
            normalize_configuration(config, ui.BENCHMARKS)

    def test_atomic_file_failure_preserves_existing_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "defaults.json"
            write_configuration(path, {"previous": True})
            with patch("direct_spending.ui_settings.os.replace", side_effect=OSError("write denied")):
                with self.assertRaises(OSError):
                    write_configuration(path, {"new": True})
            self.assertEqual(json.loads(path.read_text()), {"previous": True})
            self.assertEqual(list(path.parent.iterdir()), [path])


@unittest.skipUnless(os.environ.get("BENCHMARK_UI_TK_TESTS") == "1", "opt-in native Tk widget check")
class TkWidgetTests(unittest.TestCase):
    """Programmatic Tk checks, without claiming visual verification."""
    def test_conditional_widgets_defaults_reload_and_small_window_access(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            root = tk.Tk()
            root.withdraw()
            try:
                panel = ui.BenchmarkUI(root)
                for entry, text in (("3e-3", "0.003"), ("3e3", "3000"), ("3e-", "complete number")):
                    panel.vars["lr"].set(entry)
                    self.assertIn(text, panel.common_controls["lr"][0].cget("text"))
                panel.vars["lr"].set("3e-3")
                self.assertEqual(tuple(panel.model_list.get(0, "end")), tuple(
                    name + " (legacy)" if name in ui.LEGACY_MODELS else name for name in ui.VISIBLE_MODELS))
                self.assertFalse(any(name.startswith("spending_") for name in panel.model_list.get(0, "end")))
                for model in ui.VISIBLE_MODELS:
                    panel.model_list.selection_clear(0, "end")
                    panel.model_list.selection_set(ui.MODELS.index(model))
                    panel._update_model_options()
                    root.update_idletasks()
                    self.assertEqual(len(panel.model_options.notebook.tabs()), 1)
                    self.assertEqual(panel.model_options.notebook.tabs()[0], str(panel.model_options.frames[model]))
                    runner.build_parser().parse_args(panel._build_command()[3:])
                options = panel.spending_options
                self.assertEqual(str(options.controls["spending_core"][1].cget("state")), "readonly")
                for core in ("gru", "lstm", "tcn", "transformer", "static_mlp"):
                    panel.vars["spending_core"].set(core)
                    root.update_idletasks()
                    self.assertEqual(bool(panel.model_options.controls["direct_spending"]["n_layers"][1].winfo_manager()), core != "tcn")
                    self.assertEqual(bool(options.controls["spending_heads"][1].winfo_manager()), core == "transformer")
                    self.assertEqual(bool(options.controls["spending_tcn_kernel"][1].winfo_manager()), core == "tcn")
                for mode in ("learned", "fixed"):
                    panel.vars["spending_gamma_mode"].set(mode)
                    self.assertEqual(bool(options.controls["spending_gamma_min"][1].winfo_manager()), mode == "learned")
                    self.assertEqual(bool(options.controls["spending_gamma_max"][1].winfo_manager()), mode == "learned")
                    self.assertEqual(options.controls["spending_gamma"][0].cget("text"), "Initial gamma" if mode == "learned" else "Fixed gamma")
                panel.vars["spending_beta_mode"].set("learned")
                self.assertFalse(options.controls["spending_beta_width"][1].winfo_manager())
                panel.vars["spending_initial_features"].set("-1,2")
                self.assertTrue(options.controls["spending_beta_width"][1].winfo_manager())
                panel.vars["epochs"].set("3")
                panel._save_defaults()
                other_root = tk.Toplevel(root)
                other_root.withdraw()
                try:
                    restored = ui.BenchmarkUI(other_root)
                    self.assertEqual(restored._configuration(), panel._configuration())
                    self.assertEqual(restored.status.cget("text"), "saved defaults loaded")
                finally:
                    other_root.destroy()
                root.geometry("900x650")
                root.deiconify()
                root.update_idletasks()
                self.assertTrue(panel.run_btn.winfo_ismapped())
                self.assertTrue(panel.log.winfo_ismapped())
                self.assertGreater(panel.settings.canvas.winfo_height(), 100)
                # Geometry checks exercise narrow layouts without claiming visual inspection.
                for size in ("900x650", "1120x920"):
                    root.geometry(size)
                    for model in ("l2n", "defect", "metric_l2n", "direct_spending"):
                        panel.model_list.selection_clear(0, "end")
                        panel.model_list.selection_set(ui.MODELS.index(model))
                        panel._update_model_options()
                        root.update_idletasks()
                        for key, widgets in panel.common_controls.items():
                            widget = widgets[1]
                            self.assertLessEqual(widget.winfo_x() + widget.winfo_width(), widget.master.winfo_width() + 1, (size, key))
                        if model == "direct_spending":
                            for key, widgets in panel.spending_options.controls.items():
                                widget = widgets[1]
                                if widget.winfo_manager():
                                    self.assertLessEqual(widget.winfo_x() + widget.winfo_width(), widget.master.winfo_width() + 1, (size, key))
            finally:
                root.destroy()

    def test_parameter_button_counts_real_selection_without_training(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            root = tk.Tk()
            root.withdraw()
            outcome = []
            try:
                panel = ui.BenchmarkUI(root)
                panel.model_list.selection_clear(0, "end")
                panel.model_list.selection_set(ui.VISIBLE_MODELS.index("direct_spending"))
                panel.vars["spending_core"].set("lstm")
                panel.vars["spending_gamma_mode"].set("learned")
                panel.vars["d_model"].set("8")
                panel.vars["n_layers"].set("1")
                panel.vars["param_budget"].set("off")
                panel._update_model_options()
                panel.model_options.parameter_button.invoke()
                window = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
                window.withdraw()
                body = next(child for child in window.winfo_children() if isinstance(child, ui.ttk.Frame))
                text = next(child for child in body.winfo_children() if isinstance(child, tk.Text))

                def inspect():
                    value = text.get("1.0", "end")
                    if "trainable parameters" in value or "Could not count" in value:
                        outcome.append(value)
                        root.quit()
                    else:
                        root.after(50, inspect)
                root.after(50, inspect)
                timeout = root.after(25000, root.quit)
                root.mainloop()
                root.after_cancel(timeout)
                self.assertTrue(outcome, "Parameter-count helper did not complete")
                self.assertIn("602 trainable parameters", outcome[0])
                self.assertIn("lstm", outcome[0])
                self.assertIsNone(panel.proc)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
