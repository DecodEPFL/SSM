import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from direct_spending import SpendingConfig
from direct_spending.benchmark import (SpendingSim, load_checkpoint, save_checkpoint,
                                      settings_from_args)
from direct_spending.config import CORES, MODEL_SPECS
from direct_spending.ui_settings import VALUE_DEFAULTS
from Test_files import benchmark_ui as ui
from Test_files import run_benchmarks as runner


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_registry_core_and_beta_paths_and_configuration_roundtrip(self):
        self.assertEqual(set(ui.MODELS), set(runner.MODEL_ZOO))
        self.assertEqual([name for name in ui.VISIBLE_MODELS if name in MODEL_SPECS], ["direct_spending"])
        x = torch.randn(2, 13, 2)
        for core in CORES:
            for beta_mode in ("zero", "fixed", "learned"):
                args = runner.build_parser().parse_args([
                    "--models", "direct_spending", "--d-model", "8", "--n-layers", "1",
                    "--spending-core", core, "--spending-beta-mode", beta_mode,
                    "--spending-fixed-beta", "2", "--spending-initial-features", "1,-2",
                    "--spending-storage", "reference"])
                settings = settings_from_args(args)
                cfg = runner.GlobalModelConfig(d_model=8, n_layers=1, spending=settings)
                runner._validate_initialization(["direct_spending"], cfg)
                model = runner.build_model("direct_spending", 2, 3, cfg)
                self.assertEqual(model.spending_config.core, core)
                self.assertEqual(model.spending_config.beta_mode, beta_mode)
                self.assertEqual(model(x).shape, (2, 13, 3))
                values = json.loads(json.dumps(model.spending_config.to_dict()))
                self.assertEqual(SpendingConfig.from_dict(values), model.spending_config)
        cfg = runner.GlobalModelConfig(d_model=8, n_layers=1)
        for name in MODEL_SPECS:
            model = runner.build_model(name, 2, 3, cfg)
            self.assertEqual(model.model.config.core, MODEL_SPECS[name].get("spending_core", "gru"))

    def test_padding_mask_keeps_genuine_warmup_and_pauses_only_padding(self):
        sequences = [(np.ones((7, 2), np.float32), np.ones((7, 3), np.float32), 3),
                     (np.ones((4, 2), np.float32), np.ones((4, 3), np.float32), 2)]
        u, y, loss_mask, valid = runner.pack_full_sequences(sequences, [0, 1], torch.device("cpu"), return_valid=True)
        self.assertTrue(valid[0, :3].all())
        self.assertEqual(loss_mask[0, :3].count_nonzero(), 0)
        self.assertEqual(valid[1, 4:].count_nonzero(), 0)
        u[~valid] = float("nan")
        model = SpendingSim(2, 3, SpendingConfig(core="gru", width=8, layers=1))
        pred = model(u, mask=valid)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertEqual(pred[~valid].count_nonzero(), 0)
        for row, length in ((0, 7), (1, 4)):
            torch.testing.assert_close(pred[row:row + 1, :length], model(u[row:row + 1, :length]))

    def test_tagged_checkpoint_all_cores_beta_modes_and_normalization(self):
        norm = runner.Standardizer(torch.tensor([1., 2.]), torch.tensor([3., 4.]),
                                   torch.tensor([7., 8., 9.]), torch.ones(3))
        x = torch.randn(2, 11, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            for core in CORES:
                for mode in ("zero", "fixed", "learned"):
                    model = SpendingSim(2, 3, SpendingConfig(core=core, width=8, layers=1,
                        beta_mode=mode, fixed_beta=3., initial_features=(2., -1.)))
                    model.eval()
                    expected = model(x)
                    save_checkpoint(path, model, norm, {"train_losses": [1.]}, {"model": "direct_spending"})
                    restored, payload = load_checkpoint(path)
                    torch.testing.assert_close(restored(x), expected)
                    self.assertEqual(payload["config"]["beta_mode"], mode)
                    self.assertEqual(payload["config"]["core"], core)
                    torch.testing.assert_close(payload["normalizer"]["y_mean"], norm.y_mean)
                    self.assertFalse(restored.certificate()["physical_zero_state_bound"])
            torch.save({"format": "energy_projection", "model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "remain distinct"):
                load_checkpoint(path)

    def test_coordinate_labels_and_target_feasibility_audit_do_not_change_policy(self):
        for coordinates in ("standardized", "physical"):
            for mode in ("zero", "fixed", "learned"):
                model = SpendingSim(2, 3, SpendingConfig(beta_mode=mode), coordinates=coordinates)
                info = model.certificate()
                self.assertEqual(info["physical_zero_state_bound"], coordinates == "physical" and mode == "zero")
                self.assertIn("complete trajectory", info["certified_interval"])
                self.assertIn("carried", info["metric_suffix"])

    def test_real_native_benchmark_minimal_job_standard_results_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            args = runner.build_parser().parse_args([
                "--models", "direct_spending", "--benchmarks", "Cascaded_Tanks", "--epochs", "1",
                "--batch-size", "1", "--d-model", "4", "--n-layers", "1", "--param-budget", "off",
                "--spending-core", "static_mlp", "--spending-beta-mode", "learned",
                "--spending-initial-features", "1,2", "--spending-diagnostics",
                "--device", "cpu", "--no-plot", "--no-show", "--out", temp])
            runner.run(args)
            report = json.loads((Path(temp) / "report.json").read_text())
            self.assertEqual(len(report["results"]), 1)
            row = report["results"][0]
            self.assertEqual(row["status"], "ok", row.get("error"))
            self.assertEqual(row["epochs_run"], 1)
            self.assertEqual(len(row["history"]["train_losses"]), 1)
            self.assertEqual(row["model_family"], "direct_spending")
            self.assertIn("static_mlp", row["model_label"])
            self.assertTrue(Path(row["artifacts"]["storage_figure"]).exists())
            model, payload = load_checkpoint(row["artifacts"]["checkpoint"])
            norm = runner.Standardizer(**payload["normalizer"])
            bench = runner.load_benchmark("Cascaded_Tanks")
            metrics, predictions = runner.evaluate_validation(model, bench, norm, torch.device("cpu"))
            self.assertAlmostEqual(row["rmse"], metrics[0]["rmse"], places=6)
            self.assertEqual(predictions[0].shape, bench.validation[0].y.shape)
            diagnostics = row["validation_sequences"][0]["storage_diagnostics"]
            self.assertLess(diagnostics["max_normalized_energy_violation"], 2e-6)
            self.assertEqual(len(diagnostics["storage_trace"][0]), len(bench.validation[0]) + 1)

    def make_ui(self, models):
        values = dict(VALUE_DEFAULTS, epochs="1", batch_size="1", lr=".001", plot_every="1",
            plot_max_length="64", d_model="8", d_state="8", n_layers="1", d_hidden="8",
            nl_layers="1", gamma="auto", report_metric="rmse", param_budget="off", device="cpu",
            ff="auto", mode="auto", compile_mode="default")
        variable = lambda value: SimpleNamespace(get=lambda: value)
        panel = ui.BenchmarkUI.__new__(ui.BenchmarkUI)
        panel.vars = {key: variable(value) for key, value in values.items()}
        panel.bench_list = SimpleNamespace(curselection=lambda: (ui.BENCHMARKS.index("Cascaded_Tanks"),))
        panel.model_list = SimpleNamespace(curselection=lambda: tuple(ui.MODELS.index(m) for m in models))
        panel.out_var = variable("/tmp/direct-spending-ui-test")
        for name in ("show_var", "gif_var", "amp_var", "compile_var", "pcg_var", "cudagraph_var",
                     "l2n_random_phase_var", "defect_random_phase_var"):
            setattr(panel, name, variable(False))
        return panel

    def test_ui_single_batch_commands_saved_configuration_and_cli_override(self):
        for selected in (("direct_spending",), ("direct_spending", "gru")):
            panel = self.make_ui(selected)
            command = panel._build_command()
            args = runner.build_parser().parse_args(command[3:])
            self.assertEqual(tuple(args.models), selected)
            settings_from_args(args)
            config = panel._configuration()
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "configuration.json"
                path.write_text(json.dumps(config))
                with patch("sys.argv", ["runner", "--config", str(path), "--epochs", "2"]), patch.object(runner, "run") as run:
                    runner.main()
                    restored = run.call_args.args[0]
                    self.assertEqual(restored.epochs, 2)
                    self.assertEqual(restored.models, list(selected))
                    self.assertEqual(restored.spending_core, "gru")

    def test_ui_command_all_core_and_beta_choices_and_parameter_matching(self):
        for core in CORES:
            for mode in ("zero", "fixed", "learned"):
                panel = self.make_ui(("direct_spending",))
                for key, value in (("spending_core", core), ("spending_beta_mode", mode),
                                   ("spending_fixed_beta", "3"), ("spending_initial_features", "-1,2")):
                    panel.vars[key] = SimpleNamespace(get=lambda value=value: value)
                args = runner.build_parser().parse_args(panel._build_command()[3:])
                settings = settings_from_args(args)
                self.assertEqual(settings["core"], core)
                self.assertEqual(settings["beta_mode"], mode)
                self.assertEqual(settings["initial_features"], (-1., 2.) if mode == "learned" else ())
                if mode == "zero":
                    cfg = runner.GlobalModelConfig(d_model=8, n_layers=1, spending=settings)
                    model = runner.build_model("direct_spending", 2, 3, cfg, param_budget=300)
                    self.assertGreater(runner._count_params(model), 0)
                    self.assertEqual(runner._effective_width(model), model.spending_config.width)

    def test_validation_rejects_negative_energy_nonfinite_and_bad_shapes(self):
        for kwargs in ({"gamma": -1}, {"gamma": float("nan")}, {"fixed_beta": -1},
                       {"beta_initial": 0}, {"core": "transformer", "width": 7},
                       {"tcn_dilations": (0,)}, {"initial_features": (float("inf"),)}):
            with self.assertRaises(ValueError):
                SpendingConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
