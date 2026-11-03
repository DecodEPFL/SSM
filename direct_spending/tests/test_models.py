import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from direct_spending import DirectSpendingModel, SpendingConfig, detach_state, state_to
from energy_bounded.state import map_state

from .test_storage import audit


def tensor_leaves(state):
    leaves = []
    map_state(state, lambda t: leaves.append(t) or t)
    return leaves


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33)
        torch.set_num_threads(1)

    def configs(self):
        return [SpendingConfig(core=core, width=8, layers=2) for core in
                ("static_mlp", "gru", "lstm", "tcn", "transformer")]

    def test_all_cores_all_beta_modes_prefix_certificate(self):
        v = torch.randn(3, 57, 2)
        for config in self.configs():
            for mode in ("zero", "fixed", "learned"):
                cfg = replace(config, beta_mode=mode, fixed_beta=3., initial_features=(1., -2.))
                model = DirectSpendingModel(2, 3, cfg)
                y, state, info = model(v, diagnostics=True)
                audit(self, v, y, info["storage"], info["initial_beta"])
                if mode == "zero":
                    self.assertEqual(info["initial_beta"].count_nonzero(), 0)
                if mode == "fixed":
                    self.assertEqual(info["initial_beta"].flatten().tolist(), [3.] * 3)
                if mode == "learned":
                    torch.testing.assert_close(info["initial_beta"], torch.full_like(info["initial_beta"], .1))

    def test_sequence_chunks_steps_and_all_cache_tensors(self):
        v = torch.randn(3, 29, 2, dtype=torch.float64)
        for config in self.configs():
            model = DirectSpendingModel(2, 3, replace(config, beta_mode="fixed", fixed_beta=.7)).double().eval()
            with torch.no_grad():
                y, full = model(v)
                state, chunks = None, []
                for a, b in ((0, 0), (0, 1), (1, 7), (7, 13), (13, 29), (29, 29)):
                    part, state = model(v[:, a:b], state)
                    chunks.append(part)
                torch.testing.assert_close(torch.cat(chunks, 1), y)
                for x, z in zip(tensor_leaves(state), tensor_leaves(full)):
                    torch.testing.assert_close(x, z)
                state, steps = model.init_state(3, dtype=v.dtype), []
                for t in range(29):
                    part, state = model.step(state, v[:, t])
                    steps.append(part)
                torch.testing.assert_close(torch.stack(steps, 1), y)
                for x, z in zip(tensor_leaves(state), tensor_leaves(full)):
                    torch.testing.assert_close(x, z)

    def test_masks_compaction_resume_and_position_offsets(self):
        v = torch.randn(3, 25, 2, dtype=torch.float64)
        mask = torch.rand(3, 25) > .4
        mask[0, :4] = False; mask[2] = False
        v[~mask] = float("nan")
        for config in self.configs():
            model = DirectSpendingModel(2, 3, config).double().eval()
            with torch.no_grad():
                y, full = model(v, mask=mask)
                a, state = model(v[:, :11], mask=mask[:, :11])
                b, state = model(v[:, 11:], state, mask=mask[:, 11:])
                torch.testing.assert_close(torch.cat((a, b), 1), y)
                self.assertEqual(y[~mask].count_nonzero(), 0)
                for row in range(3):
                    compact, _ = model(v[row:row + 1, mask[row]])
                    torch.testing.assert_close(compact[0], y[row, mask[row]])
                if config.core == "transformer":
                    self.assertEqual(full.core.position.tolist(), mask.sum(1).tolist())
                extra = torch.randn(3, 4, 2, dtype=v.dtype)
                resumed, _ = model(extra, full)
                fresh, _ = model(extra[2:3])
                torch.testing.assert_close(resumed[2:3], fresh)

    def test_beta_computed_once_and_declared_features_never_read_inputs(self):
        cfg = SpendingConfig(core="gru", width=8, layers=1, beta_mode="learned", initial_features=(1., 2.))
        model = DirectSpendingModel(2, 2, cfg)
        features = torch.tensor([[3., 4.], [8., 9.]])
        v = torch.randn(2, 19, 2)
        with patch.object(model.initializer, "forward", wraps=model.initializer.forward) as initialize:
            left, state = model(v[:, :5], initial_features=features)
            right, state = model(v[:, 5:], state)
            self.assertEqual(initialize.call_count, 1)
        torch.testing.assert_close(state.initial_beta, model.init_state(2, initial_features=features).initial_beta)
        with self.assertRaisesRegex(ValueError, "once"):
            model(v, state, initial_features=features)

    def test_cross_chunk_gradients_into_spending_and_learned_beta(self):
        for config in self.configs():
            for features in ((), (1., -2.)):
                cfg = replace(config, beta_mode="learned", initial_features=features)
                model = DirectSpendingModel(2, 3, cfg).double()
                streamed = copy.deepcopy(model)
                v = torch.randn(2, 13, 2, dtype=torch.float64, requires_grad=True)
                other_v = v.detach().clone().requires_grad_()
                y, state = model(v)
                a, s = streamed(other_v[:, :5]); b, s = streamed(other_v[:, 5:], s)
                z = torch.cat((a, b), 1)
                first = torch.autograd.grad((y - .3).square().sum() + .2 * state.storage.sum(),
                    (v, *model.parameters()), allow_unused=True)
                second = torch.autograd.grad((z - .3).square().sum() + .2 * s.storage.sum(),
                    (other_v, *streamed.parameters()), allow_unused=True)
                for x, w in zip(first, second):
                    if x is None:
                        self.assertIsNone(w)
                    else:
                        torch.testing.assert_close(x, w, atol=1e-7, rtol=2e-6)
                beta_grad = first[-1]
                self.assertIsNotNone(beta_grad)
                self.assertTrue(torch.isfinite(beta_grad).all())
                self.assertGreater(beta_grad.abs().sum(), 0)
                self.assertTrue(any(g is not None and g.abs().sum() > 1e-8
                                    for g in first[1:1 + len(list(model.core.parameters()))]))

    def test_core_causality_and_explicit_detach_reset_move(self):
        v = torch.randn(2, 23, 2)
        for config in self.configs():
            model = DirectSpendingModel(2, 3, config).eval()
            y, state, info = model(v, diagnostics=True)
            changed = v.clone(); changed[:, 9:] *= -20
            z, _, other = model(changed, diagnostics=True)
            torch.testing.assert_close(y[:, :9], z[:, :9])
            torch.testing.assert_close(info["storage"][:, :10], other["storage"][:, :10])
            detached = detach_state(state)
            self.assertTrue(all(not t.requires_grad for t in tensor_leaves(detached)))
            half = state_to(detached, dtype=torch.float16)
            self.assertIn(half.storage.dtype, (torch.float32, torch.float64))
            reset = model.init_state(2)
            self.assertEqual(reset.storage.count_nonzero(), 0)

    def test_cpu_autocast_zero_input_and_nonfinite_detection(self):
        for config in self.configs():
            model = DirectSpendingModel(2, 3, config)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y, state = model(torch.zeros(2, 9, 2))
                self.assertEqual(y.count_nonzero(), 0)
                loss = y.sum() + state.storage.sum()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertEqual(y.dtype, torch.float32)
            with self.assertRaises(FloatingPointError):
                model(torch.full((2, 9, 2), float("nan")))

    def test_supported_accelerators(self):
        devices = [d for d, ok in (("cuda", torch.cuda.is_available()),
                                    ("mps", torch.backends.mps.is_available())) if ok]
        if not devices:
            self.skipTest("CUDA/MPS unavailable; no implicit backend fallback")
        for device in devices:
            for config in self.configs():
                model = DirectSpendingModel(2, 3, config).to(device)
                v = torch.randn(2, 9, 2, device=device, requires_grad=True)
                y, _ = model(v)
                y.sum().backward()
                self.assertTrue(torch.isfinite(v.grad).all())

    def test_learned_beta_underflow_is_reported_without_energy_floor(self):
        model = DirectSpendingModel(2, 3, SpendingConfig(beta_mode="learned"))
        with torch.no_grad():
            model.initializer.logit.fill_(-1000)
        with self.assertRaisesRegex(FloatingPointError, "underflowed"):
            model.init_state(2)

    def test_learned_context_beta_reset_is_independent_of_ambient_autocast(self):
        cfg = SpendingConfig(beta_mode="learned", initial_features=(1., -2.))
        model = DirectSpendingModel(2, 3, cfg)
        explicit = model.init_state(2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inside = model.init_state(2)
        torch.testing.assert_close(explicit.initial_beta, inside.initial_beta, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
