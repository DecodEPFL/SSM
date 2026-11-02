"""Independent certificate, recurrence, differentiation, and integration checks."""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from src.neural_ssm import DeepSSM, DefectL2SSM, SSMConfig
from src.neural_ssm.ssm.cells.lti.defect import (
    apply_thin_defect,
    cwy_orthonormal,
    signed_values_and_defects,
)
from src.neural_ssm.utils.scan import compute_block_lti_states


def symmetric_sqrt(matrix):
    values, vectors = torch.linalg.eigh(matrix)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.mT


class DefectAlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_cwy_matches_explicit_householder_products(self):
        for rows, cols in [(1, 1), (7, 2), (4, 4)]:
            raw = torch.randn(3, rows, cols, dtype=torch.float64)
            raw[0, :, 0] = 0  # zero reflection uses identity, not epsilon normalization
            expected = torch.eye(rows, dtype=raw.dtype).expand(3, rows, rows).clone()
            for i in range(cols):
                v = raw[..., i:i + 1]
                denom = v.square().sum(dim=-2, keepdim=True)
                h = torch.eye(rows, dtype=raw.dtype) - 2 * (v @ v.mT) / denom.clamp_min(1e-300)
                expected = expected @ h
            q = cwy_orthonormal(raw)
            torch.testing.assert_close(q, expected[..., :cols], atol=2e-14, rtol=2e-14)
            torch.testing.assert_close(q.mT @ q, torch.eye(cols).double().expand(3, cols, cols))

    def test_thin_defect_retains_orthogonal_complement(self):
        q = cwy_orthonormal(torch.randn(8, 2, dtype=torch.float64))
        s = torch.tensor([0.4, -0.8], dtype=torch.float64)
        defects = (1 - s.square()).sqrt()
        identity = torch.eye(8, dtype=q.dtype)
        root = apply_thin_defect(q, defects, identity)
        torch.testing.assert_close(root @ root, identity - (q * s.square()) @ q.mT)
        complement = identity - q @ q.mT
        torch.testing.assert_close(root @ complement, complement)

    def test_stable_scalar_defects_and_derivatives(self):
        raw = torch.tensor([-100., -10., 0., 10., 100.], requires_grad=True)
        values, defects = signed_values_and_defects(raw, 1.0)
        self.assertGreater(defects[3].item(), 9e-5)  # tanh(10) rounds to 1 in float32
        self.assertTrue(torch.isfinite(defects).all())
        (values + defects).sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        torch.testing.assert_close(values.square() + defects.square(), torch.ones_like(raw))

    def test_cwy_and_defect_gradcheck(self):
        raw = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(cwy_orthonormal, (raw,)))
        values = torch.tensor([-1.2, 0., 0.7], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: signed_values_and_defects(x, .999), (values,)
        ))

    def test_completion_matches_independent_dense_square_roots(self):
        shapes = [(6, 2, 3, 2), (4, 7, 6, 2), (4, 7, 2, 2), (8, 2, 7, 4), (3, 2, 4, 1)]
        for n, m, p, block in shapes:
            with self.subTest(n=n, m=m, p=p, block=block):
                cell = DefectL2SSM(n, m, p, block_size=block, gamma=1.7, init_scale=1.).double()
                with torch.no_grad():
                    cell.a_raw.normal_()
                a, b, c, d, _ = cell.compute_z_matrices()
                ux, sx, vx, _ = cell.X()
                uy, sy, vy, _ = cell.Y()
                uz, sz, vz, _ = cell.Z()
                x, y, z = (ux * sx) @ vx.T, (uy * sy) @ vy.T, (uz * sz) @ vz.T
                # Undo the absorbed rotations before using the original theorem.
                ua, _, va, _ = cell._a_factors()
                x = torch.block_diag(*ua) @ x
                y = y @ torch.block_diag(*va).T
                def eye(k):
                    return torch.eye(k, dtype=a.dtype)
                expected_b = symmetric_sqrt(eye(n) - a @ a.T) @ x
                expected_c = y @ symmetric_sqrt(eye(n) - a.T @ a)
                expected_d = (
                    -y @ a.T @ x
                    + symmetric_sqrt(eye(p) - y @ y.T)
                    @ z @ symmetric_sqrt(eye(m) - x.T @ x)
                )
                torch.testing.assert_close(b / cell.gamma, expected_b, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(c, expected_c, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(d / cell.gamma, expected_d, atol=1e-12, rtol=1e-12)
                k = torch.cat((torch.cat((a, b / cell.gamma), 1),
                               torch.cat((c, d / cell.gamma), 1)), 0)
                self.assertLess(torch.linalg.matrix_norm(k, 2).item(), 1.)

    def test_no_svd_eigendecomposition_or_inverse_in_parameter_construction(self):
        cell = DefectL2SSM(12, 3, 5)
        with patch.object(torch.linalg, 'svd', side_effect=AssertionError), \
             patch.object(torch.linalg, 'svdvals', side_effect=AssertionError), \
             patch.object(torch.linalg, 'eigh', side_effect=AssertionError), \
             patch.object(torch.linalg, 'inv', side_effect=AssertionError):
            cell.compute_block_matrices()

    def test_uniform_rate_and_certificate_at_saturated_raw_values(self):
        for dtype in (torch.float32, torch.float64):
            cell = DefectL2SSM(8, 3, 5, max_radius=.95).to(dtype=dtype)
            with torch.no_grad():
                cell.a_raw.fill_(30.)
                for factor in (cell.X, cell.Y, cell.Z):
                    factor.values_raw.fill_(-30.)
            a, b, c, d, _ = cell.compute_z_matrices()
            self.assertLessEqual(cell.contraction_rate_bound().item(), .95 + 1e-7)
            k = torch.cat((torch.cat((a, b), 1), torch.cat((c, d), 1)), 0)
            self.assertLessEqual(torch.linalg.matrix_norm(k, 2).item(), 1. + 2e-6)


class DefectRecurrenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_shared_power_scan_matches_dense_reference_including_initial_state(self):
        for size in (1, 2, 4):
            blocks = torch.randn(2, size, size, dtype=torch.float64) * .15
            dense = torch.block_diag(*blocks)
            for length in (0, 1, 2, 3, 8, 17):
                drives = torch.randn(length, 3, 2 * size, dtype=torch.float64)
                initial = torch.randn(3, 2 * size, dtype=torch.float64)
                reference = [initial]
                for v in drives:
                    reference.append(reference[-1] @ dense.T + v)
                actual = compute_block_lti_states(blocks, drives, initial)
                torch.testing.assert_close(actual, torch.stack(reference), atol=1e-12, rtol=1e-12)

    def test_scan_loop_outputs_and_all_gradients(self):
        for size in (2, 4):
            cell = DefectL2SSM(8, 3, 2, block_size=size, train_gamma=True, learn_x0=True).double()
            with torch.no_grad():
                cell.a_raw.normal_(0, .6)  # unequal singular values; complex-scalar scan would fail
            u = torch.randn(2, 13, 3, dtype=torch.float64, requires_grad=True)
            params = (u,) + tuple(cell.parameters())
            y_loop, x_loop = cell(u, mode='loop')
            weights_y, weights_x = torch.randn_like(y_loop), torch.randn_like(x_loop)
            grad_loop = torch.autograd.grad(
                (y_loop * weights_y).sum() + (x_loop * weights_x).sum(), params
            )
            y_scan, x_scan = cell(u, mode='scan')
            grad_scan = torch.autograd.grad(
                (y_scan * weights_y).sum() + (x_scan * weights_x).sum(), params
            )
            torch.testing.assert_close(y_scan, y_loop, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(x_scan, x_loop, atol=1e-12, rtol=1e-12)
            for left, right in zip(grad_loop, grad_scan):
                self.assertTrue(torch.isfinite(left).all())
                torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)

    def test_full_cell_parameter_and_input_gradcheck(self):
        cell = DefectL2SSM(2, 1, 1, train_gamma=True).double()
        parameters = dict(cell.named_parameters())
        u = torch.randn(1, 3, 1, dtype=torch.float64, requires_grad=True)
        def run(inputs, *values):
            return torch.func.functional_call(cell, dict(zip(parameters, values)), (inputs,))[0]
        self.assertTrue(torch.autograd.gradcheck(run, (u, *parameters.values()), fast_mode=True))

    def test_step_outputs_use_pre_update_state_and_satisfy_dissipation(self):
        cell = DefectL2SSM(8, 3, 2, gamma=.7, init_scale=1.).double()
        u = torch.randn(3, 17, 3, dtype=torch.float64)
        x0 = torch.randn(3, 8, dtype=torch.float64)
        outputs, states = cell(u, state=x0)
        a, b, c, d, _ = cell.compute_z_matrices()
        torch.testing.assert_close(outputs, states[:, :-1] @ c.T + u @ d.T)
        torch.testing.assert_close(states[:, 1:], states[:, :-1] @ a.T + u @ b.T)
        residue = (
            states[:, 1:].square().sum(-1) + outputs.square().sum(-1)
            - states[:, :-1].square().sum(-1) - cell.gamma.square() * u.square().sum(-1)
        )
        self.assertLessEqual(residue.max().item(), 1e-12)
        next_state, y = cell.step(x0, u[:, 0])
        torch.testing.assert_close(next_state, states[:, 1])
        torch.testing.assert_close(y, outputs[:, 0])

    def test_finite_horizon_operator_norm_and_incremental_gain(self):
        cell = DefectL2SSM(4, 2, 3, gamma=.4, init_scale=1.).double()
        length = 12
        impulses = torch.eye(length * 2, dtype=torch.float64).reshape(length * 2, length, 2)
        response, _ = cell(impulses)
        operator = response.flatten(1).T
        self.assertLessEqual(torch.linalg.matrix_norm(operator, 2).item(), cell.gamma.item())
        u, v = torch.randn(2, 3, length, 2, dtype=torch.float64)
        x0 = torch.randn(3, 4, dtype=torch.float64)
        yu, _ = cell(u, state=x0)
        yv, _ = cell(v, state=x0)
        yd, _ = cell(u - v)
        torch.testing.assert_close(yu - yv, yd)
        self.assertLessEqual((yu - yv).norm().item(), cell.gamma.item() * (u - v).norm().item())

    def test_chunking_time_first_empty_sequences_and_return_flags(self):
        cell = DefectL2SSM(6, 2, 3).double()
        u = torch.randn(2, 9, 2, dtype=torch.float64)
        whole_y, whole_x = cell(u)
        y1, last = cell(u[:, :4], return_state=False, return_last=True)
        y2, states2 = cell(u[:, 4:], reset_state=False)
        torch.testing.assert_close(torch.cat((y1, y2), 1), whole_y)
        torch.testing.assert_close(last, whole_x[:, 4])
        torch.testing.assert_close(states2[:, -1], whole_x[:, -1])
        yt, xt, last = cell(u.transpose(0, 1), time_first=True, return_last=True)
        torch.testing.assert_close(yt.transpose(0, 1), whole_y)
        torch.testing.assert_close(xt.transpose(0, 1), whole_x)
        for mode in ('loop', 'scan'):
            empty_y, empty_x = cell(u[:, :0], state=last, mode=mode)
            self.assertEqual(empty_y.shape, (2, 0, 3))
            torch.testing.assert_close(empty_x[:, 0], last)
        self.assertEqual(cell(u[0], return_state=False).shape, (1, 9, 3))
        self.assertEqual(cell(u[0, 0], return_state=False).shape, (1, 1, 3))
        cell(u, detach_state=True)
        self.assertFalse(cell.state.requires_grad)
        cell.reset()
        torch.testing.assert_close(cell.state, torch.zeros_like(cell.state))


class DefectIntegrationTests(unittest.TestCase):
    def test_cache_invalidates_for_weights_buffers_dtype_and_grad_mode(self):
        cell = DefectL2SSM(4, 2, 3).eval()
        with torch.no_grad():
            first = cell.compute_block_matrices()
            self.assertIs(first[0], cell.compute_block_matrices()[0])
            cell.a_raw.add_(.1)
            second = cell.compute_block_matrices()
            self.assertIsNot(first[0], second[0])
            cell.log_gamma.add_(math.log(2))
            third = cell.compute_block_matrices()
            torch.testing.assert_close(third[1], 2 * second[1])
            cell.double()
            self.assertEqual(cell.compute_block_matrices()[0].dtype, torch.float64)
        self.assertTrue(cell.compute_block_matrices()[0].requires_grad)

    def test_checkpoint_roundtrip(self):
        original = DefectL2SSM(8, 2, 3, block_size=4).double().eval()
        clone = copy.deepcopy(original)
        with torch.no_grad():
            clone.compute_block_matrices()
            original.a_raw.add_(.2)
            clone.load_state_dict(original.state_dict())
            u = torch.randn(2, 7, 2, dtype=torch.float64)
            torch.testing.assert_close(original(u), clone(u))

    def test_deepssm_registry_gain_and_optimizer_steps(self):
        torch.manual_seed(51)
        config = SSMConfig(param='defect', d_model=4, d_state=8, n_layers=2,
                           ff='LGLU2', gamma=.8, train_gamma=False, defect_block_size=4)
        model = DeepSSM(2, 3, config=config).double()
        u = torch.randn(2, 11, 2, dtype=torch.float64)
        target = .2 * torch.cat((u, u[..., :1]), -1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            y, _ = model(u, mode='scan')
            loss = (y - target).square().mean()
            loss.backward()
            self.assertTrue(all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            ))
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        self.assertLessEqual(model.certified_gain_bound().item(), .8 + 1e-6)
        self.assertLessEqual(y.norm().item(), .8 * u.norm().item() + 1e-10)
        self.assertIsInstance(model.blocks[0].lru, DefectL2SSM)

    def test_invalid_options_and_unsupported_convolution(self):
        for kwargs in [dict(d_state=3), dict(gamma=0), dict(max_radius=1.01),
                       dict(init_rho=1), dict(block_size=3), dict(factor_margin=-.01)]:
            options = dict(d_state=4, d_input=2, d_output=2)
            options.update(kwargs)
            with self.assertRaises(ValueError):
                DefectL2SSM(**options)
        cell = DefectL2SSM(4, 2, 2)
        with self.assertRaisesRegex(ValueError, 'convolution'):
            cell(torch.randn(1, 5, 2), mode='conv')

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile unavailable')
    def test_fullgraph_compiled_scan_matches_eager_outputs_and_gradients(self):
        cell = DefectL2SSM(4, 2, 2).double()
        u = torch.randn(1, 5, 2, dtype=torch.float64, requires_grad=True)
        eager = cell(u)[0]
        expected = torch.autograd.grad(eager.square().sum(), (u, *cell.parameters()))
        compiled = torch.compile(cell, backend='aot_eager', fullgraph=True)
        actual = compiled(u)[0]
        grads = torch.autograd.grad(actual.square().sum(), (u, *cell.parameters()))
        torch.testing.assert_close(actual, eager)
        for a, b in zip(grads, expected):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
