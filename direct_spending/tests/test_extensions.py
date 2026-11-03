"""Native LSTM, learned-certificate and exact factory-count behavior."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from direct_spending import DirectSpendingModel, SpendingConfig
from direct_spending.benchmark import SpendingSim, load_checkpoint, save_checkpoint
from direct_spending.config import CORES
from direct_spending.cores import LSTMCore
from direct_spending.parameter_count import parameter_counts
from direct_spending.tests.test_storage import audit
from Test_files import run_benchmarks as runner


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51)
        torch.set_num_threads(1)

    def test_lstm_calls_native_sequence_module_and_carries_both_states(self):
        core = LSTMCore(2, 3, width=8, layers=2).double()
        self.assertIsInstance(core.lstm, torch.nn.LSTM)
        x = torch.randn(2, 17, 2, dtype=torch.float64)
        with patch.object(core.lstm, "forward", wraps=core.lstm.forward) as native:
            y, state = core(x)
            native.assert_called_once()
        self.assertEqual(tuple(state[0].shape), (2, 2, 8))
        self.assertEqual(tuple(state[1].shape), (2, 2, 8))
        a, carried = core(x[:, :6]); b, carried = core(x[:, 6:], carried)
        torch.testing.assert_close(torch.cat((a, b), 1), y)
        for actual, expected in zip(carried, state):
            torch.testing.assert_close(actual, expected)

    def test_learned_gamma_certificate_and_cross_chunk_gradients_all_cores(self):
        x = torch.randn(2, 19, 2, dtype=torch.float64)
        mask = torch.rand(2, 19) > .2
        for core in CORES:
            config = SpendingConfig(core=core, width=8, layers=1, gamma=.8,
                                    gamma_mode="learned", gamma_min=.2, gamma_max=2.)
            model = DirectSpendingModel(2, 3, config).double()
            other = copy.deepcopy(model)
            y, state, info = model(x, mask=mask, diagnostics=True)
            gain = model.gamma.item()
            self.assertAlmostEqual(gain, .8, places=6)
            audit(self, x, y, info["storage"], info["initial_beta"], gamma=gain, mask=mask, tolerance=1e-11)
            a, s = other(x[:, :7], mask=mask[:, :7]); b, s = other(x[:, 7:], s, mask=mask[:, 7:])
            torch.testing.assert_close(torch.cat((a, b), 1), y)
            loss = y.square().sum() + state.storage.sum()
            loss.backward()
            (torch.cat((a, b), 1).square().sum() + s.storage.sum()).backward()
            self.assertGreater(model.gain.logit.grad.abs(), 0)
            for first, second in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(first.grad, second.grad, atol=1e-8, rtol=1e-6)

    def test_gamma_derivative_and_extreme_logits_preserve_declared_bounds(self):
        config = SpendingConfig(core="lstm", width=4, layers=1, gamma=.5,
                                gamma_mode="learned", gamma_min=.1, gamma_max=1.5)
        model = DirectSpendingModel(1, 1, config).double()
        x = torch.randn(2, 11, 1, dtype=torch.float64)
        y, state = model(x)
        derivative = torch.autograd.grad(y.square().sum() + state.storage.sum(), model.gain.logit)[0]
        sigmoid = model.gain.logit.sigmoid()
        expected = 2 * model.gamma * x.square().sum() * (config.gamma_max - config.gamma_min) * sigmoid * (1 - sigmoid)
        torch.testing.assert_close(derivative, expected, atol=1e-9, rtol=1e-8)
        for dtype in (torch.float32, torch.float64, torch.float16):
            model = model.to(dtype=dtype)
            for logit in (-10000., 10000.):
                with torch.no_grad(): model.gain.logit.fill_(logit)
                value = model.gamma
                self.assertGreaterEqual(value.item(), config.gamma_min)
                self.assertLessEqual(value.item(), config.gamma_max)
                self.assertTrue(torch.isfinite(torch.autograd.grad(value, model.gain.logit)[0]))
        model = model.float()
        y, _ = model(torch.zeros(2, 9, 1))
        self.assertEqual(y.count_nonzero(), 0)

    def test_learned_gamma_updates_and_checkpoint_reports_actual_value(self):
        cfg = SpendingConfig(core="lstm", width=4, layers=1, gamma=1.,
                             gamma_mode="learned", gamma_min=.1, gamma_max=3.)
        model = SpendingSim(1, 1, cfg, coordinates="physical")
        x = torch.randn(2, 13, 1)
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        opt.zero_grad(); model(x).square().mean().backward(); opt.step()
        info = model.certificate()
        self.assertNotEqual(info["gamma"], cfg.gamma)
        self.assertEqual(info["uniform_gamma_bound"], 3.)
        self.assertEqual(info["learned_gamma_bounds"], [.1, 3.])
        norm = runner.Standardizer(torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            save_checkpoint(path, model, norm, {}, {})
            restored, payload = load_checkpoint(path)
            torch.testing.assert_close(restored(x), model(x))
            self.assertEqual(restored.certificate(), info)
            self.assertEqual(payload["certificate"]["gamma"], info["gamma"])
            model = model.double()
            save_checkpoint(path, model, norm, {}, {})
            restored, _ = load_checkpoint(path)
            self.assertEqual(next(restored.parameters()).dtype, torch.float64)
            self.assertEqual(restored.certificate(), model.certificate())
            torch.testing.assert_close(restored(x.double()), model(x.double()))
            # A legacy fixed-gamma configuration has no gain tensors to migrate.
            fixed = SpendingSim(1, 1, SpendingConfig(core="gru", width=4, layers=1))
            save_checkpoint(path, fixed, norm, {}, {})
            payload = torch.load(path, weights_only=True)
            for key in ("gamma_mode", "gamma_min", "gamma_max"):
                payload["config"].pop(key)
            torch.save(payload, path)
            restored, _ = load_checkpoint(path)
            torch.testing.assert_close(restored(x), fixed(x))

    def test_counts_match_real_factory_with_dimensions_budget_and_gain_parameter(self):
        for budget in ("off", "lstm", "500"):
            args = runner.build_parser().parse_args(["--models", "direct_spending", "gru",
                "--benchmarks", "Cascaded_Tanks", "--spending-core", "lstm", "--n-layers", "1",
                "--spending-gamma-mode", "learned", "--param-budget", budget])
            with patch.object(runner, "load_benchmark", return_value=SimpleNamespace(n_u=2, n_y=3)):
                rng = torch.get_rng_state()
                rows = parameter_counts(args)
                torch.testing.assert_close(torch.get_rng_state(), rng)
            cfg = runner.model_config_from_args(args)
            target, _ = runner.resolve_param_budget(budget, 2, 3, cfg)
            for row in rows:
                actual = runner.build_model(row["model"], 2, 3, cfg, param_budget=target)
                self.assertEqual(row["trainable"], runner._count_params(actual))
                self.assertEqual(row["effective_width"], runner._effective_width(actual))
                self.assertEqual((row["input_features"], row["output_features"]), (2, 3))

    def test_bad_learned_gamma_limits_rejected(self):
        for values in ({"gamma_min": 0}, {"gamma_max": 1}, {"gamma": 0},
                       {"gamma_min": 2}, {"gamma_max": float("inf")}):
            with self.assertRaises(ValueError):
                SpendingConfig(gamma_mode="learned", **values)

    def test_persistent_core_only_context_does_not_supply_energy(self):
        x = torch.randn(2, 23, 1, dtype=torch.float64)
        x[:, 4:] = 0  # Finite-energy drive, persistent nondecaying context.
        context = torch.ones(2, 23, 2, dtype=torch.float64, requires_grad=True)
        mask = torch.ones(2, 23, dtype=torch.bool); mask[0, 8:10] = False
        for core in CORES:
            model = DirectSpendingModel(1, 2, SpendingConfig(core=core, width=8, layers=1,
                context_features=2, gamma=.8, gamma_mode="learned", gamma_min=.1, gamma_max=2.)).double().eval()
            y, _, info = model(x, context=context, mask=mask, diagnostics=True)
            audit(self, x, y, info["storage"], info["initial_beta"], gamma=model.gamma.item(), mask=mask, tolerance=1e-11)
            a, state = model(x[:, :7], context=context[:, :7], mask=mask[:, :7])
            b, state = model(x[:, 7:], state, context=context[:, 7:], mask=mask[:, 7:])
            torch.testing.assert_close(torch.cat((a, b), 1), y)
            state, steps = model.init_state(2, dtype=x.dtype), []
            for t in range(x.shape[1]):
                value, state = model.step(state, x[:, t], context=context[:, t], mask=mask[:, t])
                steps.append(value)
            torch.testing.assert_close(torch.stack(steps, 1), y)
            zero, _ = model(torch.zeros_like(x), context=100 * context)
            self.assertEqual(zero.count_nonzero(), 0)
            gradient = torch.autograd.grad(y.square().sum(), context, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum(), 0)
            changed = context.detach().clone(); changed[:, 12:] *= -9
            other, _ = model(x, context=changed, mask=mask)
            torch.testing.assert_close(other[:, :12], y[:, :12])
            with self.assertRaisesRegex(ValueError, "Context"):
                model(x)

    def test_accelerator_learned_gain_context_forward_and_backward(self):
        devices = [name for name, available in (("cuda", torch.cuda.is_available()),
                    ("mps", torch.backends.mps.is_available())) if available]
        if not devices:
            self.skipTest("CUDA/MPS unavailable")
        for device in devices:
            for core in CORES:
                model = DirectSpendingModel(1, 2, SpendingConfig(core=core, width=8, layers=1,
                    context_features=2, gamma=.8, gamma_mode="learned", gamma_min=.1, gamma_max=2.)).to(device)
                x = torch.randn(2, 17, 1, device=device)
                context = torch.ones(2, 17, 2, device=device, requires_grad=True)
                y, state, info = model(x, context=context, diagnostics=True)
                (y.square().mean() + .01 * state.storage.mean()).backward()
                self.assertTrue(torch.isfinite(model.gain.logit.grad))
                self.assertTrue(torch.isfinite(context.grad).all())
                self.assertLess(info["max_normalized_energy_violation"].item(), 2e-6)


if __name__ == "__main__":
    unittest.main()
