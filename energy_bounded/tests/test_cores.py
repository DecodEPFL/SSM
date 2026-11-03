import copy
import unittest

import torch
from torch import nn

from energy_bounded import (EnergyBoundedModel, GRUCore, TCNCore, TransformerCore,
                            detach_state, state_bytes, state_to)
from energy_bounded.state import map_state


def cores():
    return (GRUCore(2, 3, width=8, layers=2),
            TCNCore(2, 3, width=8, dilations=(1, 3), kernel_size=3),
            TransformerCore(2, 3, width=8, heads=2, layers=2))


def tensors(state):
    result = []
    map_state(state, lambda t: result.append(t) or t)
    return result


def assert_state_equal(test, first, second, **kwargs):
    left, right = tensors(first), tensors(second)
    test.assertEqual(len(left), len(right))
    for x, y in zip(left, right):
        torch.testing.assert_close(x, y, **kwargs)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_full_chunks_steps_outputs_and_entire_caches(self):
        for core in cores():
            for wrapped in (False, True):
                with self.subTest(core=type(core).__name__, wrapped=wrapped):
                    model = EnergyBoundedModel(core, .9) if wrapped else core
                    model = model.double().eval()
                    u = torch.randn(3, 29, 2, dtype=torch.float64)
                    with torch.no_grad():
                        full, final = model(u)
                        last, parts = None, []
                        cuts = (0, 0, 1, 4, 12, 13, 29, 29)
                        for left, right in zip(cuts[:-1], cuts[1:]):
                            y, last = model(u[:, left:right], last)
                            parts.append(y)
                        torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-8, rtol=2e-8)
                        assert_state_equal(self, last, final, atol=2e-8, rtol=2e-8)
                        state, parts = model.init_state(3, dtype=u.dtype), []
                        for t in range(u.shape[1]):
                            y, state = model.step(state, u[:, t])
                            parts.append(y)
                        torch.testing.assert_close(torch.stack(parts, 1), full, atol=2e-8, rtol=2e-8)
                        assert_state_equal(self, state, final, atol=2e-8, rtol=2e-8)

    def test_masks_pause_state_and_resume_across_chunks_and_steps(self):
        mask = torch.rand(3, 25) > .4
        mask[0, :4] = False
        mask[1, -5:] = False
        mask[2] = False
        for core in cores():
            for wrapped in (False, True):
                model = EnergyBoundedModel(core) if wrapped else core
                model = model.double().eval()
                u = torch.randn(3, 25, 2, dtype=torch.float64)
                u[~mask] = float("nan")
                with torch.no_grad():
                    full, final = model(u, mask=mask)
                    self.assertEqual(full[~mask].count_nonzero(), 0)
                    last, parts = None, []
                    for left, right in ((0, 3), (3, 11), (11, 25)):
                        y, last = model(u[:, left:right], last, mask=mask[:, left:right])
                        parts.append(y)
                    torch.testing.assert_close(torch.cat(parts, 1), full)
                    assert_state_equal(self, last, final)
                    last, parts = model.init_state(3, dtype=u.dtype), []
                    for t in range(25):
                        y, last = model.step(last, u[:, t], mask=mask[:, t])
                        parts.append(y)
                    torch.testing.assert_close(torch.stack(parts, 1), full)
                    assert_state_equal(self, last, final)
                    for b in range(3):
                        compact, state = model(u[b:b + 1, mask[b]])
                        torch.testing.assert_close(compact[0], full[b, mask[b]])
                    # Resume a wholly padded row: zero valid context must be equivalent to reset.
                    extra = torch.randn(3, 4, 2, dtype=u.dtype)
                    resumed, _ = model(extra, final)
                    fresh, _ = model(extra[2:3])
                    torch.testing.assert_close(resumed[2:3], fresh)

    def test_future_perturbation_causality(self):
        u = torch.randn(2, 41, 2)
        for core in cores():
            for wrapped in (False, True):
                model = (EnergyBoundedModel(core) if wrapped else core).eval()
                with torch.no_grad():
                    y, _ = model(u)
                    changed = u.clone(); changed[:, 17:] = 20 * torch.randn_like(changed[:, 17:])
                    other, _ = model(changed)
                torch.testing.assert_close(y[:, :17], other[:, :17])

    def test_cross_chunk_gradients_preserved_including_energy(self):
        for core in cores():
            model = EnergyBoundedModel(core).double()
            other = copy.deepcopy(model)
            u = torch.randn(2, 13, 2, dtype=torch.float64, requires_grad=True)
            x = u.detach().clone().requires_grad_()
            y, state = model(u)
            left, s = other(x[:, :5])
            right, s = other(x[:, 5:], s)
            z = torch.cat((left, right), 1)
            torch.testing.assert_close(y, z)
            grads = torch.autograd.grad(y.square().sum() + state.energy.sum(),
                                        (u, *model.parameters()), allow_unused=True)
            chunks = torch.autograd.grad(z.square().sum() + s.energy.sum(),
                                         (x, *other.parameters()), allow_unused=True)
            for a, b in zip(grads, chunks):
                if a is None:
                    self.assertIsNone(b)
                else:
                    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)

    def test_masked_backward_finite(self):
        for core in cores():
            model = EnergyBoundedModel(core)
            u = torch.randn(3, 17, 2)
            mask = torch.rand(3, 17) > .4
            mask[2] = False
            u[~mask] = float("nan"); u.requires_grad_()
            y, state = model(u, mask=mask)
            (y.square().sum() + state.energy.sum()).backward()
            self.assertTrue(torch.isfinite(u.grad).all())
            self.assertEqual(u.grad[~mask].count_nonzero(), 0)
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())

    def test_transformer_qkv_processes_only_new_tokens_and_offsets(self):
        core = TransformerCore(2, 3, width=8, layers=2).eval()
        seen, hooks = [], []
        for block in core.blocks:
            hooks.append(block.qkv.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[1])))
        with torch.no_grad():
            _, state = core(torch.randn(2, 7, 2))
            first = state
            _, state = core(torch.randn(2, 3, 2), state)
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen, [7, 7, 3, 3])
        self.assertEqual(state.position.tolist(), [10, 10])
        for old, new in zip(first.layers, state.layers):
            torch.testing.assert_close(old[0], new[0][:, :, :7])
            torch.testing.assert_close(old[1], new[1][:, :, :7])
        self.assertGreater(state_bytes(state), state_bytes(first))

    def test_tcn_receptive_field_and_bounded_cache(self):
        core = TCNCore(2, 3, width=8, dilations=(1, 2, 4)).eval()
        self.assertEqual(core.receptive_field, 15)
        u = torch.randn(2, 51, 2)
        changed = u.clone(); changed[:, :10] *= 4
        with torch.no_grad():
            y, state = core(u)
            z, _ = core(changed)
        torch.testing.assert_close(y[:, 10 + core.receptive_field - 1:],
                                   z[:, 10 + core.receptive_field - 1:])
        self.assertEqual(state_bytes(state), state_bytes(core.init_state(2)))

    def test_explicit_detach_move_and_reset(self):
        for core in cores():
            model = EnergyBoundedModel(core)
            _, state = model(torch.randn(2, 8, 2, requires_grad=True))
            detached = detach_state(state)
            self.assertTrue(all(not t.requires_grad for t in tensors(detached)))
            moved = state_to(detached, dtype=torch.float64)
            self.assertTrue(all(t.dtype == torch.float64 for t in tensors(moved) if t.is_floating_point()))
            half = state_to(detached, dtype=torch.float16)
            self.assertIn(half.energy.dtype, (torch.float32, torch.float64))
            reset = model.init_state(2)
            self.assertEqual(reset.energy.count_nonzero(), 0)
            if isinstance(core, TransformerCore):
                self.assertEqual(reset.core.valid.shape[1], 0)

    def test_parameterless_arbitrary_state_core_and_nonfinite_detection(self):
        class Core(nn.Module):
            d_input, d_output = 2, 3

            def init_state(self, batch_size, *, device=None, dtype=None):
                return {"history": [], "steps": 0}

            def forward(self, u, state=None, *, mask=None):
                return u.new_ones(*u.shape[:2], 3), {"history": [], "steps": u.shape[1]}

        model = EnergyBoundedModel(Core())
        y, state = model(torch.zeros(2, 5, 2))
        self.assertEqual(y.count_nonzero(), 0)
        self.assertEqual(state.core["steps"], 5)
        broken = EnergyBoundedModel(GRUCore(2, 3, width=8))
        with torch.no_grad():
            broken.core.out_proj.weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            broken(torch.ones(2, 3, 2))

    def test_cpu_autocast_forward_backward(self):
        for core in cores():
            model = EnergyBoundedModel(core)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y, state = model(torch.randn(2, 9, 2))
                loss = y.square().mean()
            loss.backward()
            self.assertEqual(y.dtype, torch.float32)
            self.assertEqual(state.energy.dtype, torch.float64)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_optional_accelerator_paths(self):
        available = [name for name, yes in (("cuda", torch.cuda.is_available()),
                                            ("mps", torch.backends.mps.is_available())) if yes]
        if not available:
            self.skipTest("No CUDA/MPS backend available on this host")
        for device in available:
            for core in cores():
                model = EnergyBoundedModel(core).to(device).eval()
                u = torch.randn(2, 19, 2, device=device, requires_grad=True)
                y, state = model(u)
                a, s = model(u[:, :7]); b, s = model(u[:, 7:], s)
                torch.testing.assert_close(torch.cat((a, b), 1), y, atol=3e-5, rtol=3e-5)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(u.grad).all())


if __name__ == "__main__":
    unittest.main()
