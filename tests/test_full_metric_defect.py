"""Weighted certificates, formerly excluded maps, and full-metric integration."""

import copy
import math
import unittest
from unittest.mock import patch

import torch

from src.neural_ssm import (
    ContextualDeepSSM,
    DeepSSM,
    DefectL2SSM,
    MetricDeepSSM,
    MetricSSMConfig,
    SSMConfig,
)
from src.neural_ssm.ssm.cells.common import bounded_real_matrix
from src.neural_ssm.utils.lyapunov import solve_block_lyapunov


def full_cell(n=6, m=2, p=3, **kwargs):
    return DefectL2SSM(
        n, m, p, state_metric="full", max_radius=1, factor_margin=0, **kwargs
    ).double()


def set_metric_factor(cell, r):
    with torch.no_grad():
        cell.metric_diag_raw.copy_(torch.log(torch.expm1(r.diag())))
        cell.metric_upper_raw.copy_(r[cell.metric_indices[0], cell.metric_indices[1]])


class FullMetricAlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_public_constructor_and_configuration_agree(self):
        options = dict(d_model=2, d_state=5, n_layers=1, param="defect", ff="LGLU2",
                       gamma=1., defect_state_metric="full", defect_max_radius=1.,
                       defect_factor_margin=0.)
        torch.manual_seed(17)
        direct = DeepSSM(1, 1, **options).double()
        torch.manual_seed(17)
        configured = DeepSSM(1, 1, config=SSMConfig(**options, max_phase_b=.5,
                                                 nl_layers=3, train_ff_lip=None)).double()
        u = torch.randn(2, 11, 1, dtype=torch.float64)
        torch.testing.assert_close(direct(u)[0], configured(u)[0])
        self.assertEqual(direct.blocks[0].lru.state_metric, "full")

        metric_options = dict(d_model=2, d_state=5, n_layers=1, state_metric="full",
                              max_radius=1., factor_margin=0.)
        torch.manual_seed(19)
        direct_metric = MetricDeepSSM(1, 1, **metric_options).double()
        torch.manual_seed(19)
        configured_metric = MetricDeepSSM(1, 1, config=MetricSSMConfig(**metric_options)).double()
        torch.testing.assert_close(direct_metric(u)[0], configured_metric(u)[0])

    def test_block_lyapunov_against_independent_convergent_series(self):
        for size in (1, 2, 3, 4):
            blocks = torch.randn(2, size, size, dtype=torch.float64) * .08
            a = torch.block_diag(*blocks)
            r = torch.randn(2 * size, 2 * size, dtype=torch.float64)
            q = r.T @ r + torch.eye(2 * size, dtype=r.dtype)
            expected, term = q.clone(), q
            for _ in range(40):
                term = a.T @ term @ a
                expected = expected + term
            actual = solve_block_lyapunov(blocks, q)
            torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
            torch.testing.assert_close(actual - a.T @ actual @ a, q)

    def test_implicit_lyapunov_backward_and_higher_derivatives(self):
        blocks = (torch.randn(2, 2, 2, dtype=torch.float64) * .1).requires_grad_()
        q = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
        inputs = (blocks, q)
        self.assertTrue(torch.autograd.gradcheck(solve_block_lyapunov, inputs))
        self.assertTrue(torch.autograd.gradgradcheck(solve_block_lyapunov, inputs))

    def test_weighted_bounded_real_certificate_for_rectangular_ports(self):
        for n, m, p, block in ((1, 2, 3, 2), (5, 2, 3, 2), (4, 7, 6, 2),
                                (6, 2, 7, 1), (6, 3, 2, 3), (8, 3, 2, 4)):
            with self.subTest(n=n, m=m, p=p, block=block):
                cell = full_cell(n, m, p, block_size=block, gamma=.7, init_scale=1)
                with torch.no_grad():
                    cell.metric_upper_raw.normal_(0, .3)
                    cell.a_raw.normal_(0, .7)
                a, b, c, d, p_state = cell.compute_z_matrices()
                self.assertEqual(a.shape, (n, n))
                self.assertGreater(torch.linalg.eigvalsh(p_state).min().item(), 0)
                lmi = bounded_real_matrix(a, b, c, d, p_state, cell.gamma)
                self.assertLessEqual(torch.linalg.eigvalsh(lmi).max().item(), 2e-12)
                _, r, _ = cell._compute_state_metric()
                torch.testing.assert_close(p_state - a.T @ p_state @ a, r.T @ r)

    def test_full_metric_initializes_to_identity(self):
        for n, block in ((5, 2), (8, 4)):
            cell = full_cell(n, block_size=block)
            torch.testing.assert_close(cell.storage_matrix(), torch.eye(n).double(),
                                       atol=3e-7, rtol=3e-7)

    def test_construction_uses_no_svd_eigensolver_inverse_or_dense_a(self):
        cell = full_cell(7)
        with patch.object(torch.linalg, "svd", side_effect=AssertionError), \
             patch.object(torch.linalg, "svdvals", side_effect=AssertionError), \
             patch.object(torch.linalg, "eigh", side_effect=AssertionError), \
             patch.object(torch.linalg, "eig", side_effect=AssertionError), \
             patch.object(torch.linalg, "inv", side_effect=AssertionError), \
             patch.object(torch, "block_diag", side_effect=AssertionError):
            cell.compute_block_matrices()

    def test_realizes_cancellation_map_excluded_by_identity_block_storage(self):
        # Two distinct conjugate pole pairs force two separate real 2x2 blocks.
        # Their residues obey C_i B_i=+1,-1. Thus every identity-storage block
        # realization requires gamma >= ||B||*||C|| >= sum_i |C_i B_i| = 2.
        # The actual gain is only .03/(.99*.96), and full P certifies gamma=1.
        cell = full_cell(4, 1, 1)
        j = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
        a = torch.block_diag(.1 * j, .2 * j)
        b = torch.tensor([[1.], [0.], [1.], [0.]], dtype=torch.float64)
        c = torch.tensor([[1., 0., -1., 0.]], dtype=torch.float64)
        p = torch.kron(torch.tensor([[2.125, -1.875], [-1.875, 2.125]]).double(),
                       torch.eye(2).double())
        r = torch.linalg.cholesky(p - a.T @ p @ a).T
        set_metric_factor(cell, r)
        t = torch.linalg.solve(r.T, a.T @ p)
        h = torch.linalg.cholesky(p + t.T @ t).T
        x = h @ b
        y = torch.linalg.solve(r.T, c.T).T
        z = (y @ t @ b) / torch.sqrt((1 - y.square().sum()) * (1 - x.square().sum()))
        with torch.no_grad():
            cell.a_raw.copy_(torch.atanh(torch.tensor([[.1, .1], [.2, .2]], dtype=torch.float64)))
            cell.theta.fill_(math.pi / 2)
            cell.phi.zero_()
            # Rank-one Householder factors: H(e-v)e=v. The scalar factor is -1.
            basis = -x / x.norm()
            e = torch.zeros_like(basis); e[0] = 1
            cell.X.left_raw.copy_(e - basis)
            cell.X.right_raw.fill_(1)
            cell.X.values_raw.fill_(torch.atanh(x.norm()))
            basis = -y.T / y.norm()
            e = torch.zeros_like(basis); e[0] = 1
            cell.Y.left_raw.fill_(1)
            cell.Y.right_raw.copy_(e - basis)
            cell.Y.values_raw.fill_(torch.atanh(y.norm()))
            cell.Z.left_raw.fill_(1)
            cell.Z.right_raw.fill_(1)
            cell.Z.values_raw.fill_(torch.atanh(z.squeeze()))
        actual_a, actual_b, actual_c, actual_d, actual_p = cell.compute_z_matrices()
        for actual, expected in zip((actual_a, actual_b, actual_c, actual_p), (a, b, c, p)):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(actual_d, torch.zeros_like(actual_d), atol=2e-12, rtol=0)
        lmi = bounded_real_matrix(actual_a, actual_b, actual_c, actual_d, actual_p, cell.gamma)
        self.assertLess(torch.linalg.eigvalsh(lmi).max().item(), 0)


class FullMetricRecurrenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_scan_loop_step_and_gradients_with_dense_metric_and_odd_state(self):
        for n, block in ((5, 2), (8, 4)):
            cell = full_cell(n, block_size=block, train_gamma=True, learn_x0=True)
            with torch.no_grad():
                cell.metric_upper_raw.normal_(0, .2)
                cell.a_raw.normal_(0, .6)
                cell.x0_param.normal_()
            u = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
            parameters = (u, *cell.parameters())
            yl, xl = cell(u, mode="loop")
            wy, wx = torch.randn_like(yl), torch.randn_like(xl)
            gl = torch.autograd.grad((yl * wy).sum() + (xl * wx).sum(), parameters)
            ys, xs = cell(u, mode="scan")
            gs = torch.autograd.grad((ys * wy).sum() + (xs * wx).sum(), parameters)
            torch.testing.assert_close(ys, yl)
            torch.testing.assert_close(xs, xl)
            for actual, expected in zip(gs, gl):
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
            next_state, output = cell.step(xs[:, 0], u[:, 0])
            torch.testing.assert_close(next_state, xs[:, 1])
            torch.testing.assert_close(output, ys[:, 0])

    def test_input_and_all_parameter_gradcheck(self):
        cell = full_cell(3, 1, 1, train_gamma=True, learn_x0=True)
        parameters = dict(cell.named_parameters())
        u = torch.randn(1, 3, 1, dtype=torch.float64, requires_grad=True)

        def run(inputs, *values):
            return torch.func.functional_call(cell, dict(zip(parameters, values)), (inputs,))[0]

        self.assertTrue(torch.autograd.gradcheck(run, (u, *parameters.values()), fast_mode=True))

    def test_weighted_dissipation_and_finite_horizon_gain(self):
        cell = full_cell(5, gamma=.6, init_scale=1)
        with torch.no_grad():
            cell.metric_upper_raw.normal_(0, .4)
        u = torch.randn(3, 11, 2, dtype=torch.float64)
        initial = torch.randn(3, 5, dtype=torch.float64)
        outputs, states = cell(u, state=initial)
        energies = cell.state_energy(states)
        residual = (energies[:, 1:] + outputs.square().sum(-1) - energies[:, :-1]
                    - cell.gamma.square() * u.square().sum(-1))
        self.assertLessEqual(residual.max().item(), 2e-12)
        impulses = torch.eye(22, dtype=torch.float64).reshape(22, 11, 2)
        response, _ = cell(impulses)
        self.assertLessEqual(torch.linalg.matrix_norm(response.flatten(1).T, 2).item(), .6)

    def test_streaming_layout_and_empty_sequences(self):
        cell = full_cell(5)
        u = torch.randn(2, 9, 2, dtype=torch.float64)
        y, states = cell(u)
        first, last = cell(u[:, :4], return_state=False, return_last=True)
        second, _ = cell(u[:, 4:], reset_state=False)
        torch.testing.assert_close(torch.cat((first, second), 1), y)
        torch.testing.assert_close(last, states[:, 4])
        yt, xt = cell(u.transpose(0, 1), time_first=True)
        torch.testing.assert_close(yt.transpose(0, 1), y)
        torch.testing.assert_close(xt.transpose(0, 1), states)
        for mode in ("scan", "loop"):
            empty_y, empty_x = cell(u[:, :0], state=last, mode=mode)
            self.assertEqual(empty_y.shape, (2, 0, 3))
            torch.testing.assert_close(empty_x[:, 0], last)


class FullMetricIntegrationTests(unittest.TestCase):
    def test_cache_metric_mutation_checkpoint_and_dtype(self):
        cell = full_cell(5).eval()
        clone = copy.deepcopy(cell)
        with torch.no_grad():
            first = cell.compute_block_matrices()
            p = cell.storage_matrix()
            self.assertIs(first[0], cell.compute_block_matrices()[0])
            self.assertIs(p, cell.storage_matrix())
            cell.metric_upper_raw.add_(.1)
            self.assertIsNot(p, cell.storage_matrix())
            self.assertIsNot(first[0], cell.compute_block_matrices()[0])
            clone.load_state_dict(cell.state_dict())
            u = torch.randn(2, 7, 2, dtype=torch.float64)
            torch.testing.assert_close(clone(u), cell(u))
            cell.float()
            self.assertEqual(cell.storage_matrix().dtype, torch.float32)
        for _ in range(2):
            cell.zero_grad(set_to_none=True)
            cell(u.float())[0].square().sum().backward()
            self.assertTrue(torch.isfinite(cell.metric_diag_raw.grad).all())
            self.assertGreater(cell.metric_upper_raw.grad.abs().max().item(), 0)

    def test_stack_storage_and_zero_input_dissipation(self):
        config = SSMConfig(param="defect", defect_state_metric="full", d_model=2,
                           d_state=5, n_layers=2, gamma=.8, ff="LGLU2",
                           defect_max_radius=1, defect_factor_margin=0)
        model = DeepSSM(1, 1, config=config).double().eval()
        initial = [torch.randn(2, 5).double() for _ in model.blocks]
        with torch.no_grad():
            for block in model.blocks:
                block.lru.metric_upper_raw.normal_(0, .2)
            initial_energy = model.storage_value(initial)
            expected = sum(w * block.lru.state_energy(x)
                           for w, block, x in zip(model.storage_weights(), model.blocks, initial))
            torch.testing.assert_close(initial_energy, expected)
            y, final = model(torch.zeros(2, 7, 1).double(), state=initial, mode="scan")
            residue = y.square().sum((1, 2)) + model.storage_value(final) - initial_energy
            self.assertLessEqual(residue.max().item(), 2e-10)

    def test_metric_stack_weighted_storage_gain_and_gradients(self):
        cfg = MetricSSMConfig(d_model=2, d_state=5, n_layers=2, gamma=.7,
                              state_metric="full", max_radius=1, factor_margin=0)
        model = MetricDeepSSM(1, 1, config=cfg).double()
        with torch.no_grad():
            for block in model.blocks:
                block.core.metric_upper_raw.normal_(0, .15)
        u = torch.randn(2, 9, 1).double()
        ys, states = model(u, mode="scan")
        yl, _ = model(u, mode="loop")
        torch.testing.assert_close(ys, yl)
        ys.square().sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertLessEqual(ys.norm().item(), .7 * u.norm().item())
        energy = model.storage_value(states)
        expected = sum(block.core.state_energy(x) for block, x in zip(model.blocks, states))
        torch.testing.assert_close(energy, expected)
        self.assertLessEqual((energy[:, -1] + ys.square().sum((1, 2))
                              - .7**2 * u.square().sum((1, 2))).max().item(), 2e-10)

    def test_context_gates_reuse_full_metric_core(self):
        cfg = SSMConfig(param="defect", defect_state_metric="full", d_model=2,
                        d_state=5, n_layers=1, gamma=1, ff="LGLU2")
        model = ContextualDeepSSM(1, 1, 1, ssm_config=cfg, context_modes="gate").double().eval()
        u, z = torch.randn(2, 7, 1).double(), torch.randn(2, 7, 1).double()
        y, _ = model(u, z)
        silent, _ = model(torch.zeros_like(u), z)
        torch.testing.assert_close(silent, torch.zeros_like(silent))
        self.assertLessEqual(y.norm().item(), u.norm().item())

    def test_amp_keeps_metric_construction_at_parameter_precision(self):
        cell = full_cell(5).float().eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            matrices = cell.compute_block_matrices()
            self.assertTrue(all(matrix.dtype == torch.float32 for matrix in matrices))
            self.assertEqual(cell.storage_matrix().dtype, torch.float32)

    def test_fullgraph_compiled_outputs_and_gradients(self):
        cell = full_cell(3, 1, 1)
        u = torch.randn(1, 5, 1, dtype=torch.float64, requires_grad=True)
        eager = cell(u)[0]
        expected = torch.autograd.grad(eager.square().sum(), (u, *cell.parameters()))
        compiled = torch.compile(cell, backend="aot_eager", fullgraph=True)
        actual = compiled(u)[0]
        gradients = torch.autograd.grad(actual.square().sum(), (u, *cell.parameters()))
        torch.testing.assert_close(actual, eager)
        for gradient, reference in zip(gradients, expected):
            torch.testing.assert_close(gradient, reference)


if __name__ == "__main__":
    unittest.main()
