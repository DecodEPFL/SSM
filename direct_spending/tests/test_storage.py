import unittest

import torch

from direct_spending.storage import StorageLayer, unit_ball_head


def audit(test, v, y, states, beta, gamma=1., mask=None, tolerance=2e-6):
    v, y, beta = v.double(), y.double(), beta.double()
    if mask is not None:
        v = torch.where(mask[..., None], v, 0)
    supplied = beta + (gamma * v).square().sum(-1).cumsum(1)
    spent = y.square().sum(-1).cumsum(1)
    test.assertTrue(((spent - supplied) / (1 + supplied) <= tolerance).all())
    torch.testing.assert_close((states[:, 1:, 0].double() + spent) / (1 + supplied),
                               supplied / (1 + supplied), atol=tolerance, rtol=0)


class StorageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)

    def test_exact_vector_unit_ball_head_zero_boundary_and_extreme(self):
        z = torch.tensor([[0., 0.], [.6, .8], [3., 4.], [1e100, -1e100]], dtype=torch.float64,
                         requires_grad=True)
        p, fraction, saturation = unit_ball_head(z)
        torch.testing.assert_close(p[:3], torch.tensor([[0., 0.], [.6, .8], [.6, .8]], dtype=p.dtype))
        self.assertTrue((p.square().sum(-1) <= 1 + 1e-15).all())
        self.assertEqual(fraction[:, 0].tolist(), [0., 1., 1., 1.])
        self.assertEqual(saturation[:, 0].tolist(), [False, False, True, True])
        (p.sum() + fraction.sum()).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        boundary_z = torch.tensor([[[.6, .8]]], dtype=torch.float64, requires_grad=True)
        output, remaining, _ = StorageLayer()(torch.ones(1, 1, 1, dtype=z.dtype), boundary_z)
        balance_gradient = torch.autograd.grad(output.square().sum() + remaining.sum(), boundary_z)[0]
        torch.testing.assert_close(balance_gradient, torch.zeros_like(balance_gradient), atol=1e-14, rtol=0)

    def test_known_spending_directions_released_energy_and_remaining_storage(self):
        v = torch.tensor([[[0.], [1.], [0.]]], dtype=torch.float64)
        z = torch.tensor([[[0., 0.], [.3, .4], [0., 2.]]], dtype=torch.float64)
        for method in ("reference", "scan"):
            y, last, info = StorageLayer(2.)(v, z, torch.tensor([[9.]], dtype=v.dtype),
                                               implementation=method, diagnostics=True)
            expected = torch.tensor([[[0., 0.], [.3 * 13 ** .5, .4 * 13 ** .5], [0., 9.75 ** .5]]], dtype=v.dtype)
            torch.testing.assert_close(y, expected)
            torch.testing.assert_close(info["storage"].flatten(), torch.tensor([9., 9., 9.75, 0.], dtype=v.dtype))
            self.assertEqual(last.item(), 0.)
            audit(self, v, y, info["storage"], torch.tensor([[9.]]), gamma=2., tolerance=1e-14)

    def test_prefix_identity_independent_samples_with_initial_energy(self):
        for initial in (torch.zeros(3, 1), torch.tensor([[1.], [20.], [.1]])):
            v, z = torch.randn(3, 317, 2), torch.randn(3, 317, 5)
            y, _, info = StorageLayer(.7)(v, z, initial, diagnostics=True)
            audit(self, v, y, info["storage"], initial, gamma=.7)

    def test_zero_input_zero_storage_and_biased_proposals_no_batch_pooling(self):
        v, z = torch.zeros(2, 30, 2), torch.ones(2, 30, 3) * 9
        y, last, _ = StorageLayer()(v, z)
        self.assertEqual(y.count_nonzero(), 0)
        self.assertEqual(last.count_nonzero(), 0)
        v[0, 0] = 30
        y, _, _ = StorageLayer()(v, z)
        self.assertEqual(y[1].count_nonzero(), 0)

    def test_positive_beta_can_fund_zero_input_tail_without_refill(self):
        v = torch.zeros(2, 300, 1, dtype=torch.float64)
        z = torch.full_like(v, .5)
        beta = torch.tensor([[4.], [9.]], dtype=v.dtype)
        y, last, info = StorageLayer(0.)(v, z, beta, diagnostics=True)
        torch.testing.assert_close(info["storage"][:, 1, 0], beta[:, 0] * .75)
        self.assertTrue(y.count_nonzero() > 0)
        audit(self, v, y, info["storage"], beta, gamma=0., tolerance=1e-13)
        self.assertTrue((y.square().sum((1, 2)) <= beta[:, 0] * (1 + 2e-14)).all())

    def test_scan_reference_zeros_ones_lengths_and_dynamic_range(self):
        for length in (0, 1, 2, 3, 17, 1025, 8192):
            v = torch.randn(2, length, 2, dtype=torch.float64)
            z = .3 * torch.randn(2, length, 3, dtype=torch.float64)
            z[:, ::3] = 0
            z[:, 1::3] = 1e100
            if length > 100:
                v[:, 10:100] *= 1e50
            initial = torch.tensor([[.4], [13.]], dtype=v.dtype)
            r = StorageLayer()
            y, last, info = r(v, z, initial, diagnostics=True)
            other, final, _ = r(v, z, initial, implementation="reference")
            torch.testing.assert_close(y, other, atol=1e-11, rtol=1e-12)
            torch.testing.assert_close(last, final, atol=1e-11, rtol=1e-12)
            if length:
                audit(self, v, y, info["storage"], initial, tolerance=1e-12)

    def test_storage_arbitrary_chunks_steps_and_nonzero_incoming_energy(self):
        v, z = torch.randn(3, 79, 2, dtype=torch.float64), torch.randn(3, 79, 4, dtype=torch.float64)
        initial, layer = torch.full((3, 1), 3., dtype=v.dtype), StorageLayer()
        y, last, _ = layer(v, z, initial)
        state, chunks = initial, []
        for a, b in ((0, 1), (1, 1), (1, 19), (19, 61), (61, 79)):
            part, state, _ = layer(v[:, a:b], z[:, a:b], state)
            chunks.append(part)
        torch.testing.assert_close(torch.cat(chunks, 1), y)
        torch.testing.assert_close(state, last)
        state, steps = initial, []
        for t in range(79):
            part, state, _ = layer.step(state, v[:, t], z[:, t])
            steps.append(part)
        torch.testing.assert_close(torch.stack(steps, 1), y)
        torch.testing.assert_close(state, last)

    def test_mask_is_identity_transition_and_nan_padding_has_zero_gradient(self):
        v, z = torch.randn(3, 27, 2), torch.randn(3, 27, 4)
        mask = torch.rand(3, 27) > .4
        mask[2] = False
        v[~mask] = float("nan"); z[~mask] = float("nan")
        v.requires_grad_(); z.requires_grad_()
        initial = torch.ones(3, 1)
        y, last, info = StorageLayer()(v, z, initial, mask=mask, diagnostics=True)
        self.assertEqual(y[~mask].count_nonzero(), 0)
        for b in range(3):
            compact, final, _ = StorageLayer()(v[b:b + 1, mask[b]], z[b:b + 1, mask[b]], initial[b:b + 1])
            torch.testing.assert_close(y[b, mask[b]], compact[0])
            torch.testing.assert_close(last[b], final[0])
        audit(self, v, y, info["storage"], initial, mask=mask)
        (y.sum() + last.sum()).backward()
        self.assertTrue(torch.isfinite(v.grad).all() and torch.isfinite(z.grad).all())
        self.assertEqual(v.grad[~mask].count_nonzero(), 0)

    def test_reference_scan_gradient_agreement_and_finite_differences(self):
        base = (torch.randn(2, 8, 2, dtype=torch.float64),
                .3 * torch.randn(2, 8, 3, dtype=torch.float64),
                torch.full((2, 1), .7, dtype=torch.float64))
        gradients = []
        layer = StorageLayer()
        for method in ("reference", "scan"):
            v, z, initial = (x.clone().requires_grad_() for x in base)
            y, last, _ = layer(v, z, initial, implementation=method)
            gradients.append(torch.autograd.grad(y.square().sum() + .17 * last.sum(), (v, z, initial)))
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
        self.assertTrue(torch.autograd.gradcheck(lambda v, z, s: layer(v, z, s)[0],
            tuple(x.clone().requires_grad_() for x in base)))

    def test_finite_gradients_zero_available_zero_z_and_unit_boundary(self):
        for method in ("reference", "scan"):
            for incoming, input_value, proposal in ((0., 0., 0.), (0., 0., 1.),
                                                    (0., 1., 0.), (0., 1., 1.), (2., 0., .5)):
                v = torch.full((2, 7, 1), input_value, dtype=torch.float64, requires_grad=True)
                z = torch.full_like(v, proposal, requires_grad=True)
                initial = torch.full((2, 1), incoming, dtype=v.dtype, requires_grad=True)
                y, last, info = StorageLayer()(v, z, initial, implementation=method, diagnostics=True)
                (y.sum() + last.sum() + info["spending_fraction"].sum()).backward()
                self.assertTrue(all(torch.isfinite(t.grad).all() for t in (v, z, initial)))

    def test_storage_causality_outputs_and_trace(self):
        v, z = torch.randn(2, 31, 2), torch.randn(2, 31, 3)
        y, _, info = StorageLayer()(v, z, diagnostics=True)
        changed_v, changed_z = v.clone(), z.clone()
        changed_v[:, 13:] *= 100; changed_z[:, 13:] *= -9
        other, _, changed = StorageLayer()(changed_v, changed_z, diagnostics=True)
        torch.testing.assert_close(y[:, :13], other[:, :13])
        torch.testing.assert_close(info["storage"][:, :14], changed["storage"][:, :14])

    def test_cpu_low_precision_inputs_and_invalid_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            v, z = torch.randn(2, 23, 2).to(dtype), torch.randn(2, 23, 3).to(dtype)
            v.requires_grad_(); z.requires_grad_()
            y, s, _ = StorageLayer()(v, z)
            (y.sum() + s.sum()).backward()
            self.assertIn(y.dtype, (torch.float32, torch.float64))
            self.assertTrue(torch.isfinite(v.grad).all() and torch.isfinite(z.grad).all())
        with self.assertRaises(FloatingPointError):
            StorageLayer()(torch.ones(1, 2, 1), torch.full((1, 2, 1), float("nan")))
        with self.assertRaises(ValueError):
            StorageLayer(-1)


if __name__ == "__main__":
    unittest.main()
