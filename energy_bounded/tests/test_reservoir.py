"""Behavioral checks; audits use actual outputs and independent FP64 accounting."""
import copy
import unittest
import warnings

import torch

from energy_bounded import EnergyReservoir


def audit(u, y, energies, gamma=1., initial=None, mask=None, atol=3e-5):
    u, y = u.double(), y.double()
    if mask is not None:
        u = torch.where(mask[..., None], u, 0)
        y = torch.where(mask[..., None], y, 0)
    initial = torch.zeros(u.shape[0], 1, dtype=torch.float64, device=u.device) if initial is None else initial.double()
    supplied = initial + (gamma * u).square().sum(-1).cumsum(1)
    spent = y.square().sum(-1).cumsum(1)
    scale = 1 + supplied
    assert bool(((spent - supplied) / scale <= atol).all())
    torch.testing.assert_close((energies[:, 1:, 0].double() + spent) / scale,
                               supplied / scale, atol=atol, rtol=0)


class ReservoirTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_prefix_identity_per_sample_and_nonzero_energy(self):
        for dtype in (torch.float32, torch.float64):
            for mode in ("scan", "reference"):
                u = torch.randn(4, 511, 3, dtype=dtype)
                v = torch.randn(4, 511, 2, dtype=dtype) * torch.tensor([.1, 2., 0., 8.])[:, None, None]
                initial = torch.tensor([[0.], [3.], [40.], [0.]], dtype=dtype)
                y, e, info = EnergyReservoir(.7)(u, v, initial, mode=mode, diagnostics=True)
                audit(u, y, info["energy"], .7, initial, atol=2e-6)
                torch.testing.assert_close(info["energy"][:, -1], e)

    def test_zero_input_with_bias_and_no_batch_pooling(self):
        u = torch.zeros(2, 20, 1)
        u[0, 0] = 10
        v = torch.ones(2, 20, 4) * 7
        y, _, _ = EnergyReservoir()(u, v)
        self.assertEqual(y[1].count_nonzero(), 0)
        y, e, _ = EnergyReservoir()(u * 0, v)
        self.assertEqual(y.count_nonzero(), 0)
        self.assertEqual(e.count_nonzero(), 0)

    def test_projection_norm_direction_and_under_budget(self):
        u = torch.tensor([[[5.], [0.], [1.]]], dtype=torch.float64)
        v = torch.tensor([[[3., 0.], [0., 10.], [0., 0.]]], dtype=torch.float64)
        y, _, _ = EnergyReservoir()(u, v)
        torch.testing.assert_close(y[0, 0], v[0, 0], rtol=0, atol=0)
        torch.testing.assert_close(y[0, 1], torch.tensor([0., 4.], dtype=y.dtype))
        self.assertEqual(y[0, 2].count_nonzero(), 0)

    def test_reference_scan_long_refill_saturation_and_chunks(self):
        for length in (0, 1, 37, 8192):
            u = torch.randn(3, length, 2, dtype=torch.float64)
            v = torch.randn(3, length, 3, dtype=torch.float64)
            v[:, :length // 3] *= 20
            v[:, length // 3:2 * length // 3] = 0
            initial = torch.tensor([[0.], [2.], [30.]], dtype=torch.float64)
            r = EnergyReservoir(1.2)
            y, e, _ = r(u, v, initial)
            ref, er, _ = r(u, v, initial, mode="reference")
            torch.testing.assert_close(y, ref, rtol=1e-8, atol=1e-8)
            torch.testing.assert_close(e, er, rtol=1e-8, atol=1e-8)
            last, parts = initial, []
            for start in range(0, length, 71):
                part, last, _ = r(u[:, start:start + 71], v[:, start:start + 71], last)
                parts.append(part)
            if parts:
                torch.testing.assert_close(torch.cat(parts, 1), y, rtol=1e-8, atol=1e-8)
                torch.testing.assert_close(last, e)

    def test_single_steps_preserve_nonzero_incoming_energy(self):
        r = EnergyReservoir()
        u, v = torch.randn(3, 83, 2), torch.randn(3, 83, 5)
        initial = torch.full((3, 1), 5.)
        y, e, _ = r(u, v, initial)
        last, parts = initial, []
        for t in range(u.shape[1]):
            part, last, _ = r.step(last, u[:, t], v[:, t])
            parts.append(part)
        torch.testing.assert_close(y, torch.stack(parts, 1))
        torch.testing.assert_close(e, last)

    def test_dynamic_range_fallback_and_strict_mode(self):
        u = torch.ones(1, 1024, 1, dtype=torch.float64)
        v = torch.zeros_like(u)
        v[:, 0] = 1e100
        with warnings.catch_warnings(record=True) as caught:
            y, e, info = EnergyReservoir()(u, v, diagnostics=True)
        self.assertTrue(caught)
        self.assertEqual(info["backend"], "reference_cancellation")
        self.assertAlmostEqual(e.item(), 1023.)
        audit(u, y, info["energy"], atol=1e-12)
        with self.assertRaises(FloatingPointError):
            EnergyReservoir(cancellation="raise")(u, v)

    def test_long_sequence_really_uses_scan_and_fp32_path(self):
        u = torch.ones(2, 32768, 2, dtype=torch.float64)
        v = torch.full((2, 32768, 3), .8, dtype=torch.float64)
        y, e, info = EnergyReservoir()(u, v, diagnostics=True)
        self.assertEqual(info["backend"], "scan")
        audit(u, y, info["energy"], atol=1e-10)
        short_u, short_v = u[:, :8].float(), v[:, :8].float()
        y, e, info = EnergyReservoir(accumulation="float32")(
            short_u, short_v, diagnostics=True)
        self.assertEqual(info["backend"], "scan")
        self.assertEqual(e.dtype, torch.float32)
        audit(short_u, y, info["energy"])

    def test_large_proposals_safe_backward(self):
        u = torch.ones(2, 17, 2, dtype=torch.float64, requires_grad=True)
        v = torch.full((2, 17, 3), 1e100, dtype=torch.float64, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, e, _ = EnergyReservoir()(u, v)
        (y.sum() + e.sum()).backward()
        self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all())

    def test_gradient_agreement_away_from_kinks_and_initial_energy(self):
        base = (torch.randn(2, 9, 2, dtype=torch.float64),
                1.5 * torch.randn(2, 9, 3, dtype=torch.float64),
                torch.full((2, 1), .37, dtype=torch.float64))
        grads = []
        for mode in ("reference", "scan"):
            u, v, e = (x.clone().requires_grad_() for x in base)
            y, last, _ = EnergyReservoir()(u, v, e, mode=mode)
            grads.append(torch.autograd.grad(y.square().sum() + .3 * last.sum(), (u, v, e)))
        for x, y in zip(*grads):
            torch.testing.assert_close(x, y, rtol=1e-8, atol=1e-8)
        r = EnergyReservoir()
        self.assertTrue(torch.autograd.gradcheck(lambda u, v, e: r(u, v, e)[0],
                                                 tuple(x.clone().requires_grad_() for x in base)))

    def test_finite_gradients_at_zero_and_switch(self):
        for mode in ("reference", "scan"):
            for x, z in ((0., 0.), (0., 2.), (1., 1.), (1., 0.)):
                u = torch.full((2, 4, 1), x, dtype=torch.float64, requires_grad=True)
                v = torch.full((2, 4, 1), z, dtype=torch.float64, requires_grad=True)
                y, e, _ = EnergyReservoir()(u, v, mode=mode)
                (y.sum() + e.sum()).backward()
                self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all())

    def test_diagnostic_norm_gradients_are_finite_at_zero(self):
        for mode in ("scan", "reference"):
            u = torch.zeros(2, 4, 2, dtype=torch.float64, requires_grad=True)
            v = torch.zeros(2, 4, 3, dtype=torch.float64, requires_grad=True)
            y, e, info = EnergyReservoir()(u, v, mode=mode, diagnostics=True)
            (y.sum() + e.sum() + info["raw_output_norm"].sum()).backward()
            self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all())

    def test_masked_nan_padding_pauses_energy_and_gradients(self):
        u, v = torch.randn(3, 31, 2), torch.randn(3, 31, 4)
        mask = torch.rand(3, 31) > .4
        u[~mask] = float("nan")
        v[~mask] = float("nan")
        u.requires_grad_(); v.requires_grad_()
        r = EnergyReservoir()
        y, e, info = r(u, v, mask=mask, diagnostics=True)
        for b in range(3):
            compact, last, _ = r(u[b:b + 1, mask[b]], v[b:b + 1, mask[b]])
            torch.testing.assert_close(y[b, mask[b]], compact[0])
            torch.testing.assert_close(e[b], last[0])
        audit(u, y, info["energy"], mask=mask)
        self.assertEqual(y[~mask].count_nonzero(), 0)
        y.sum().backward()
        self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all())

    def test_causality_of_reservoir(self):
        r = EnergyReservoir()
        u, v = torch.randn(2, 31, 2), torch.randn(2, 31, 3)
        y, _, _ = r(u, v)
        u[:, 13:] *= 100; v[:, 13:] *= 300
        other, _, _ = r(u, v)
        torch.testing.assert_close(y[:, :13], other[:, :13])

    def test_supported_cpu_dtypes_and_checkpoint(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            u, v = torch.randn(2, 41, 2).to(dtype), torch.randn(2, 41, 3).to(dtype)
            u.requires_grad_(); v.requires_grad_()
            y, _, _ = EnergyReservoir(.8)(u, v)
            self.assertIn(y.dtype, (torch.float32, torch.float64))
            y.sum().backward()
            self.assertTrue(torch.isfinite(u.grad).all() and torch.isfinite(v.grad).all())
        r = EnergyReservoir(.73)
        other = EnergyReservoir(2.)
        other.load_state_dict(copy.deepcopy(r.state_dict()))
        self.assertEqual(other.gamma, r.gamma)

    def test_validation_and_nonfinite_failures(self):
        for gamma in (0., -1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                EnergyReservoir(gamma)
        r = EnergyReservoir()
        u, v = torch.ones(2, 3, 1), torch.ones(2, 3, 2)
        with self.assertRaises(ValueError):
            r(u, v, torch.full((2, 1), -1.))
        with self.assertRaises(ValueError):
            r(u, v, mask=torch.ones(2, 3))
        with self.assertRaises(FloatingPointError):
            r(u, v * float("nan"))
        with self.assertRaises(FloatingPointError):
            r(u, v * 1e300)


if __name__ == "__main__":
    unittest.main()
