"""Behavioral checks for the combined neural/linear proposal and energy certificate."""
import copy
import json
import os
import tempfile
import tkinter as tk
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from direct_spending import DirectSpendingModel, SpendingConfig
from direct_spending.benchmark import SpendingSim, load_checkpoint, save_checkpoint
from direct_spending.config import CORES
from direct_spending.parameter_count import parameter_counts
from direct_spending.tests.test_models import tensor_leaves
from direct_spending.tests.test_storage import audit
from direct_spending.tests.test_ui import panel_for
from direct_spending.ui_settings import normalize_configuration
from Test_files import benchmark_ui as ui, run_benchmarks as runner
from .layer import OutputProjection
from . import layer as projection_layer


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(84)
        torch.set_num_threads(1)

    def test_independent_online_oracle_direction_budget_and_per_sample_accounting(self):
        u = torch.randn(3, 79, 2, dtype=torch.float64)
        raw = 3 * torch.randn(3, 79, 4, dtype=torch.float64)
        raw[:, ::4] = .01
        initial = torch.tensor([[0.], [2.], [40.]], dtype=u.dtype)
        expected, current = [], initial.clone()
        for t in range(u.shape[1]):
            budget = current + (.7 * u[:, t]).square().sum(-1, keepdim=True)
            q = raw[:, t].square().sum(-1, keepdim=True)
            scale = torch.ones_like(q)
            clipped = q > budget
            scale[clipped] = (budget[clipped] / q[clipped]).sqrt()
            y = raw[:, t] * scale
            expected.append(y)
            current = (budget - y.square().sum(-1, keepdim=True)).clamp(min=0)
        for method in ("reference", "scan"):
            y, end, info = OutputProjection(.7)(u, raw, initial, implementation=method, diagnostics=True)
            torch.testing.assert_close(y, torch.stack(expected, 1), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(end, current, atol=1e-11, rtol=1e-11)
            audit(self, u, y, info["storage"], initial, gamma=.7, tolerance=1e-12)
            self.assertGreater(info["clipping_fraction"], 0)
        # One row's drive cannot fund another row's proposal.
        u.zero_(); u[0, 0] = 10
        y, _, _ = OutputProjection()(u, torch.ones_like(raw))
        self.assertEqual(y[1:].count_nonzero(), 0)

    def test_huge_proposals_long_refill_and_saturation_without_scan_cancellation(self):
        for dtype, huge, length, tolerance in ((torch.float64, 1e200, 16385, 1e-10),
                                               (torch.float32, 1e30, 8193, 3e-6)):
            u = torch.ones(2, length, 1, dtype=dtype)
            raw = torch.full((2, length, 2), huge, dtype=dtype)
            raw[:, ::3] = 0
            u[1, 1::5] = 0
            initial = torch.tensor([[0.], [3.]], dtype=dtype)
            layer = OutputProjection(precision="float64" if dtype == torch.float64 else "float32")
            with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
                y, end, info = layer(u, raw, initial, diagnostics=True)
                ref, last, _ = layer(u, raw, initial, implementation="reference")
            self.assertFalse(caught)
            self.assertEqual(info["implementation"], "scan_max_plus")
            torch.testing.assert_close(y, ref, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(end, last, atol=tolerance, rtol=tolerance)
            audit(self, u, y, info["storage"], initial, tolerance=tolerance)

    def test_gradient_agreement_and_zero_switch_boundaries(self):
        values = (torch.randn(2, 29, 2, dtype=torch.float64),
                  .8 * torch.randn(2, 29, 3, dtype=torch.float64),
                  torch.tensor([[.4], [2.]], dtype=torch.float64),
                  torch.tensor(.9, dtype=torch.float64))
        results = []
        for method in ("reference", "scan"):
            u, raw, initial, gamma = [v.clone().requires_grad_() for v in values]
            y, last, _ = OutputProjection()(u, raw, initial, gamma=gamma, implementation=method)
            gradients = torch.autograd.grad(y.sum() + .1 * last.sum(), (u, raw, initial, gamma))
            results.append((y, last, gradients))
        torch.testing.assert_close(results[0][0], results[1][0], atol=1e-10, rtol=1e-10)
        for a, b in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-7)
            self.assertTrue(torch.isfinite(a).all())
        for u_value, r_value in ((0., 0.), (0., 2.), (1., 1.), (1., 0.), (1., 1e30)):
            for method in ("reference", "scan"):
                u = torch.full((1, 7, 1), u_value, requires_grad=True)
                raw = torch.full_like(u, r_value, requires_grad=True)
                y, last, _ = OutputProjection()(u, raw, implementation=method)
                (y.sum() + last.sum()).backward()
                self.assertTrue(torch.isfinite(u.grad).all())
                self.assertTrue(torch.isfinite(raw.grad).all())
                if not u_value:
                    self.assertEqual(y.count_nonzero(), 0)
        for method in ("reference", "scan"):
            initial = torch.zeros(2, 1, requires_grad=True)
            x = torch.full((2, 9, 1), float("nan"))
            _, end, _ = OutputProjection()(x, x, initial, mask=torch.zeros(2, 9, dtype=torch.bool), implementation=method)
            torch.testing.assert_close(torch.autograd.grad(end.sum(), initial)[0], torch.ones_like(initial))

    def test_linear_identity_exact_and_joint_projection_all_cores(self):
        x = torch.randn(2, 21, 2)
        x[:, 3] = 0
        for core in CORES:
            model = DirectSpendingModel(2, 2, SpendingConfig(core=core, width=8, layers=1,
                                       readout="projected", linear_init="identity"))
            with torch.no_grad():
                for parameter in model.core.parameters(): parameter.zero_()
            y, _, info = model(x, diagnostics=True)
            self.assertTrue(torch.equal(y, x))
            self.assertEqual(info["clipping_fraction"], 0)
            # The linear path is constrained together with the nonlinear output.
            with torch.no_grad(): model.proposal_readout.linear.weight.mul_(10)
            y, _, info = model(x, diagnostics=True)
            self.assertGreater(info["clipping_fraction"], 0)
            audit(self, x, y, info["storage"], info["initial_beta"])

    def test_finite_difference_gradcheck_away_from_switches(self):
        inputs = torch.tensor([[[.3, .6], [.1, -.2], [.4, .1]]], dtype=torch.float64, requires_grad=True)
        raw = torch.tensor([[[.2, .1], [2., 1.], [.1, .1]]], dtype=torch.float64, requires_grad=True)
        initial = torch.tensor([[.4]], dtype=torch.float64, requires_grad=True)
        gain = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
        for method in ("scan", "reference"):
            def outputs(u, r, e, gamma):
                y, last, _ = OutputProjection()(u, r, e, gamma=gamma, implementation=method)
                return y, last
            self.assertTrue(torch.autograd.gradcheck(outputs, (inputs, raw, initial, gain),
                            eps=1e-6, atol=2e-5, rtol=1e-4))

    def test_failed_scan_audit_reports_reference_fallback_without_refill(self):
        x, raw = torch.ones(2, 13, 1), torch.full((2, 13, 2), 3.)
        expected, last, _ = OutputProjection()(x, raw, implementation="reference")
        real_scan = projection_layer.associative_scan
        def corrupted_scan(*args, **kwargs):
            a, b = real_scan(*args, **kwargs)
            return a + 1000, b
        with patch.object(projection_layer, "associative_scan", side_effect=corrupted_scan):
            with self.assertWarnsRegex(RuntimeWarning, "without refill"):
                y, end, info = OutputProjection()(x, raw, diagnostics=True)
        self.assertEqual(info["implementation"], "reference_roundoff")
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(end, last)
        audit(self, x, y, info["storage"], torch.zeros(2, 1))

    def test_full_chunks_steps_state_gradients_context_and_padding_all_cores(self):
        x = torch.randn(3, 23, 2, dtype=torch.float64)
        context = torch.randn(3, 23, 2, dtype=torch.float64)
        mask = torch.rand(3, 23) > .2; mask[2] = False
        x[~mask] = float("nan"); context[~mask] = float("nan")
        for core in CORES:
            config = SpendingConfig(core=core, width=8, layers=1, readout="projected",
                gamma=.8, gamma_mode="learned", gamma_min=.1, gamma_max=2.,
                beta_mode="learned", context_features=2)
            model = DirectSpendingModel(2, 3, config).double().eval()
            streamed = copy.deepcopy(model)
            y, full, info = model(x, context=context, mask=mask, diagnostics=True)
            state, parts = None, []
            for start, stop in ((0, 0), (0, 4), (4, 9), (9, 23), (23, 23)):
                part, state = streamed(x[:, start:stop], state, context=context[:, start:stop], mask=mask[:, start:stop])
                parts.append(part)
            z = torch.cat(parts, 1)
            torch.testing.assert_close(z, y, atol=1e-9, rtol=1e-9)
            for a, b in zip(tensor_leaves(state), tensor_leaves(full)):
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
            (y.sum() + .1 * full.storage.sum()).backward()
            (z.sum() + .1 * state.storage.sum()).backward()
            for a, b in zip(model.parameters(), streamed.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=1e-8, rtol=1e-7)
                self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(model.proposal_readout.linear.weight.grad.abs().sum(), 0)
            with torch.no_grad():
                state, parts = model.init_state(3, dtype=x.dtype), []
                for t in range(x.shape[1]):
                    part, state = model.step(state, x[:, t], context=context[:, t], mask=mask[:, t])
                    parts.append(part)
                torch.testing.assert_close(torch.stack(parts, 1), y, atol=1e-9, rtol=1e-9)
                for row in (0, 1):
                    compact, _ = model(x[row:row + 1, mask[row]], context=context[row:row + 1, mask[row]])
                    torch.testing.assert_close(compact[0], y[row, mask[row]])
                changed = context.clone(); changed[:, 10:] = -70
                future_x = x.clone(); future_x[:, 10:] = 30
                other, _ = model(future_x, context=changed, mask=mask)
                torch.testing.assert_close(y[:, :10], other[:, :10])
                zero, _ = model(torch.zeros_like(x), context=torch.ones_like(context))
                # Learned beta can fund a free response; zero mode cannot.
                empty = DirectSpendingModel(2, 3, SpendingConfig(core=core, width=8, layers=1,
                    readout="projected", context_features=2)).double()
                zero, _ = empty(torch.zeros_like(x), context=torch.ones_like(context))
                self.assertEqual(zero.count_nonzero(), 0)
            audit(self, x, y, info["storage"], info["initial_beta"], gamma=model.gamma.item(), mask=mask, tolerance=1e-10)

    def test_checkpoint_family_dtype_and_real_parameter_count(self):
        norm = runner.Standardizer(torch.zeros(2), torch.ones(2), torch.zeros(3), torch.ones(3))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            for core in CORES:
                cfg = SpendingConfig(core=core, width=8, layers=1, readout="projected", gamma_mode="learned")
                model = SpendingSim(2, 3, cfg).double()
                with torch.no_grad(): model.model.proposal_readout.linear.weight.fill_(.3)
                save_checkpoint(path, model, norm, {}, {})
                restored, payload = load_checkpoint(path)
                self.assertEqual(payload["format"], "energy_projection_v1")
                self.assertEqual(restored.certificate(), model.certificate())
                x = torch.randn(2, 11, 2, dtype=torch.float64)
                torch.testing.assert_close(restored(x), model(x))
                plain = SpendingSim(2, 3, SpendingConfig(core=core, width=8, layers=1, gamma_mode="learned"))
                self.assertEqual(runner._count_params(model) - runner._count_params(plain), 6)
            payload["config"]["readout"] = "spending"; torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_checkpoint(path)

    def test_ui_cli_defaults_and_budget_count(self):
        panel = panel_for(["direct_spending"])
        panel.vars["spending_readout"].set("Output prediction + linear")
        panel.vars["spending_core"].set("lstm")
        panel.vars["spending_linear_init"].set("identity")
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertEqual((args.spending_readout, args.spending_linear_init), ("projected", "identity"))
        saved = normalize_configuration(panel._configuration(), ui.BENCHMARKS)
        self.assertEqual(saved["values"]["spending_readout"], "Output prediction + linear")
        for budget in ("off", "500"):
            args.param_budget = budget
            with patch.object(runner, "load_benchmark", return_value=SimpleNamespace(n_u=2, n_y=3)):
                row = parameter_counts(args)[0]
            cfg = runner.model_config_from_args(args)
            target, _ = runner.resolve_param_budget(budget, 2, 3, cfg)
            model = runner.build_model("direct_spending", 2, 3, cfg, param_budget=target)
            self.assertEqual(row["trainable"], runner._count_params(model))
        panel.vars["spending_readout"].set("Spending fractions")
        panel.vars["spending_linear_init"].set("invalid hidden value")
        panel._build_command()

    def test_cpu_autocast_and_available_accelerators(self):
        for core in CORES:
            model = DirectSpendingModel(2, 3, SpendingConfig(core=core, width=8, layers=1,
                readout="projected", gamma_mode="learned", context_features=2))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y, state = model(torch.zeros(2, 9, 2), context=torch.ones(2, 9, 2))
            (y.sum() + state.storage.sum()).backward()
            self.assertEqual(y.count_nonzero(), 0)
            self.assertEqual(y.dtype, torch.float32)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_accelerators(self):
        devices = [name for name, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]
        if not devices: self.skipTest("CUDA/MPS unavailable")
        for device in devices:
            for core in CORES:
                model = DirectSpendingModel(2, 3, SpendingConfig(core=core, width=8, layers=1,
                    readout="projected", context_features=2, gamma_mode="learned")).to(device)
                x = torch.randn(2, 31, 2, device=device)
                context = torch.ones(2, 31, 2, device=device, requires_grad=True)
                y, state, info = model(x, context=context, diagnostics=True)
                (y.square().sum() + .01 * state.storage.sum()).backward()
                self.assertTrue(torch.isfinite(context.grad).all())
                self.assertTrue(torch.isfinite(model.gain.logit.grad))
                self.assertTrue(torch.isfinite(model.proposal_readout.linear.weight.grad).all())
                self.assertLess(info["max_normalized_energy_violation"].item(), 2e-6)
                with torch.no_grad():
                    a, state = model(x[:, :13], context=context[:, :13])
                    b, state = model(x[:, 13:], state, context=context[:, 13:])
                    torch.testing.assert_close(torch.cat((a, b), 1), y, atol=3e-5, rtol=3e-4)
                    state, steps = model.init_state(2, device=device), []
                    for t in range(x.shape[1]):
                        value, state = model.step(state, x[:, t], context=context[:, t])
                        steps.append(value)
                    torch.testing.assert_close(torch.stack(steps, 1), y, atol=3e-5, rtol=3e-4)
            huge = torch.full((2, 4097, 2), 1e30, device=device, requires_grad=True)
            y, _, info = OutputProjection()(torch.ones(2, 4097, 1, device=device), huge, diagnostics=True)
            y.sum().backward()
            self.assertTrue(torch.isfinite(huge.grad).all())
            self.assertEqual(info["implementation"], "scan_max_plus")


@unittest.skipUnless(os.environ.get("BENCHMARK_UI_TK_TESTS") == "1", "opt-in native Tk check")
class ProjectionUI(unittest.TestCase):
    def test_mode_controls_count_button_defaults_and_small_window(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(ui, "_DEFAULTS_PATH", Path(temp) / "defaults.json"):
            root = tk.Tk(); root.withdraw(); result = []
            try:
                panel = ui.BenchmarkUI(root)
                panel.model_list.selection_clear(0, "end")
                panel.model_list.selection_set(ui.VISIBLE_MODELS.index("direct_spending"))
                panel.vars["spending_core"].set("lstm")
                panel.vars["d_model"].set("8"); panel.vars["n_layers"].set("1")
                panel.vars["param_budget"].set("off"); panel.vars["spending_gamma_mode"].set("learned")
                panel.vars["spending_readout"].set("Output prediction + linear")
                panel._update_model_options()
                options = panel.spending_options
                self.assertTrue(options.controls["spending_linear_init"][1].winfo_manager())
                for mode in ("Spending fractions", "Output prediction + linear"):
                    panel.vars["spending_readout"].set(mode)
                    self.assertEqual(bool(options.controls["spending_linear_init"][1].winfo_manager()), mode.startswith("Output"))
                panel._save_defaults()
                self.assertEqual(json.loads(ui._DEFAULTS_PATH.read_text())["values"]["spending_readout"], "Output prediction + linear")
                root.geometry("900x650"); root.deiconify(); root.update_idletasks()
                for key, widgets in options.controls.items():
                    widget = widgets[1]
                    if widget.winfo_manager():
                        self.assertLessEqual(widget.winfo_x() + widget.winfo_width(), widget.master.winfo_width() + 1, key)
                panel.model_options.parameter_button.invoke()
                window = next(child for child in root.winfo_children() if isinstance(child, tk.Toplevel))
                window.withdraw()
                body = next(child for child in window.winfo_children() if isinstance(child, ui.ttk.Frame))
                text = next(child for child in body.winfo_children() if isinstance(child, tk.Text))
                def inspect():
                    value = text.get("1.0", "end")
                    if "trainable parameters" in value or "Could not count" in value:
                        result.append(value); root.quit()
                    else: root.after(50, inspect)
                root.after(50, inspect); timeout = root.after(25000, root.quit)
                root.mainloop(); root.after_cancel(timeout)
                self.assertTrue(result)
                self.assertIn("603 trainable parameters", result[0])
                self.assertIn("Output projection", result[0])
                self.assertIsNone(panel.proc)
            finally:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
