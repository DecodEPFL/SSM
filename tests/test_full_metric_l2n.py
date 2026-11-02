"""Weighted normalization: certificates, expressivity, gradients, and block scan."""

import copy
import importlib.util
import unittest
from unittest.mock import patch

import torch

from src.neural_ssm import (
    Block2x2DenseL2SSM, ContextualDeepSSM, DeepSSM, MetricDeepSSM,
    MetricSSMConfig, SSMConfig,
)
from src.neural_ssm.ssm.cells.common import bounded_real_matrix


def full_cell(n=5, m=2, p=3, **kwargs):
    cell = Block2x2DenseL2SSM(n, m, p, state_metric="full", **kwargs).double()
    # Generic, distinct singular values exercise the active normalization branch.
    with torch.no_grad():
        cell.A_blocks_raw.normal_(0, .7)
        cell.K12_raw.normal_(0, .6)
        cell.K21_raw.normal_(0, .6)
        cell.K22_raw.normal_(0, .3)
        cell.metric_upper_raw.normal_(0, .2)
    return cell


class WeightedNormalizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)

    def test_weighted_norm_matches_independent_dense_construction(self):
        cell = full_cell(5, 2, 3, gamma=.7)
        raw_a = torch.block_diag(*cell.A_blocks_raw, cell.A_last_raw.reshape(1, 1))
        l = cell._metric_factor()
        raw = torch.cat((torch.cat((raw_a, cell.K12_raw), 1),
                         torch.cat((cell.K21_raw, cell.K22_raw), 1)), 0)
        left = torch.block_diag(l, torch.eye(3).double())
        right = torch.block_diag(torch.linalg.inv(l), torch.eye(2).double())
        sigma = torch.linalg.matrix_norm(left @ raw @ right, 2)
        alpha = (sigma * (1 + torch.finfo(sigma.dtype).eps)).clamp_min(1)
        a, b, c, d, p = cell.compute_z_matrices()
        for actual, expected in zip((a, b, c, d),
                                   (raw_a / alpha, cell.gamma * cell.K12_raw / alpha,
                                    cell.K21_raw / alpha, cell.gamma * cell.K22_raw / alpha)):
            torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(p, l.T @ l)

    def test_weighted_certificate_for_odd_sizes_and_rectangular_ports(self):
        for n, m, p in ((1, 2, 3), (5, 2, 3), (4, 7, 6), (8, 3, 2)):
            cell = full_cell(n, m, p, gamma=.6)
            a, b, c, d, metric = cell.compute_z_matrices()
            lmi = bounded_real_matrix(a, b, c, d, metric, cell.gamma)
            self.assertLessEqual(torch.linalg.eigvalsh(lmi).max().item(), 2e-11)
            self.assertLess(torch.linalg.eigvals(a).abs().max().item(), 1)
            self.assertIsNone(cell.S)

    def test_admissible_cancellation_map_is_preserved_without_rescaling(self):
        # The two modal residues are +1 and -1. Identity block storage needs
        # gamma >= sum(abs(residues))=2, whereas this full P certifies gamma=1.
        cell = full_cell(4, 1, 1)
        j = torch.tensor([[0., -1.], [1., 0.]]).double()
        a = torch.block_diag(.1 * j, .2 * j)
        b = torch.tensor([[1.], [0.], [1.], [0.]]).double()
        c = torch.tensor([[1., 0., -1., 0.]]).double()
        p = torch.kron(torch.tensor([[2.125, -1.875], [-1.875, 2.125]]).double(),
                       torch.eye(2).double())
        l = torch.linalg.cholesky(p).T
        with torch.no_grad():
            cell.A_blocks_raw.copy_(torch.stack((.1 * j, .2 * j)))
            cell.K12_raw.copy_(b)
            cell.K21_raw.copy_(c)
            cell.K22_raw.zero_()
            cell.metric_diag_raw.copy_(torch.log(torch.expm1(l.diag())))
            cell.metric_upper_raw.copy_(l[cell.metric_indices[0], cell.metric_indices[1]])
        for actual, expected in zip(cell.compute_z_matrices(), (a, b, c, torch.zeros(1, 1).double(), p)):
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        self.assertLess(torch.linalg.eigvalsh(cell.bounded_real_matrix_x()).max().item(), 0)

    def test_distinct_real_poles_and_scalar_mode_are_representable(self):
        cell = full_cell(3, 1, 1)
        with torch.no_grad():
            cell.A_blocks_raw.copy_(torch.diag(torch.tensor([-.4, .7]).double()).unsqueeze(0))
            cell.A_last_raw.fill_(.9)
            cell.K12_raw.fill_(.001)
            cell.K21_raw.fill_(.001)
            cell.K22_raw.zero_()
            cell.metric_upper_raw.zero_()
        a = cell.compute_z_matrices()[0]
        torch.testing.assert_close(a, torch.diag(torch.tensor([-.4, .7, .9]).double()))

    def test_norm_fallback_keeps_certificate_and_avoids_extra_factorizations(self):
        cell = full_cell()
        with patch.object(torch.linalg, "matrix_norm", side_effect=RuntimeError("SVD failure")), \
             patch.object(torch.linalg, "inv", side_effect=AssertionError), \
             patch.object(torch.linalg, "cholesky", side_effect=AssertionError), \
             patch.object(cell, "storage_matrix", side_effect=AssertionError):
            cell.compute_block_matrices()
        with patch.object(torch.linalg, "matrix_norm", side_effect=RuntimeError("SVD failure")):
            self.assertLessEqual(torch.linalg.eigvalsh(cell.bounded_real_matrix_x()).max().item(), 2e-11)

    def test_invalid_modes_and_precision_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "exact_norm"):
            Block2x2DenseL2SSM(5, 2, 2, state_metric="full", exact_norm=False)
        with self.assertRaisesRegex(ValueError, "state_metric"):
            Block2x2DenseL2SSM(4, 2, 2, state_metric="bad")
        cell = full_cell()
        with self.assertRaisesRegex(ValueError, "real blocks"):
            cell(torch.randn(2, 7, 2).double(), mode="conv")
        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            cell.half().compute_block_matrices()


class FullMetricL2NRecurrenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_scan_loop_step_outputs_states_and_all_gradients(self):
        for n in (1, 5, 8):
            cell = full_cell(n, train_gamma=True, learn_x0=True)
            cell.x0_param.data.normal_()
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
                torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
            next_state, output = cell.step(xs[:, 0], u[:, 0])
            torch.testing.assert_close(next_state, xs[:, 1])
            torch.testing.assert_close(output, ys[:, 0])

    def test_all_parameter_and_input_gradcheck(self):
        cell = full_cell(3, 1, 1, train_gamma=True, learn_x0=True)
        parameters = dict(cell.named_parameters())
        u = torch.randn(1, 3, 1, dtype=torch.float64, requires_grad=True)
        def run(inputs, *values):
            return torch.func.functional_call(cell, dict(zip(parameters, values)), (inputs,))[0]
        self.assertTrue(torch.autograd.gradcheck(run, (u, *parameters.values()), fast_mode=True))

    def test_dense_metric_receives_nonzero_training_gradients(self):
        cell = full_cell()
        u = torch.randn(2, 9, 2).double()
        cell(u, return_state=False).square().sum().backward()
        self.assertGreater(cell.metric_upper_raw.grad.abs().max().item(), 1e-9)
        self.assertGreater(cell.metric_diag_raw.grad.abs().max().item(), 1e-9)

    def test_dissipation_streaming_layout_and_empty_sequences(self):
        cell = full_cell(gamma=.6)
        u, initial = torch.randn(2, 9, 2).double(), torch.randn(2, 5).double()
        y, x = cell(u, state=initial)
        energies = cell.state_energy(x)
        residue = (energies[:, 1:] + y.square().sum(-1) - energies[:, :-1]
                   - cell.gamma.square() * u.square().sum(-1))
        self.assertLessEqual(residue.max().item(), 2e-11)
        first, last = cell(u[:, :4], state=initial, return_state=False, return_last=True)
        second, _ = cell(u[:, 4:], reset_state=False)
        torch.testing.assert_close(torch.cat((first, second), 1), y)
        torch.testing.assert_close(last, x[:, 4])
        yt, xt = cell(u.transpose(0, 1), state=initial, time_first=True)
        torch.testing.assert_close(yt.transpose(0, 1), y)
        torch.testing.assert_close(xt.transpose(0, 1), x)
        for mode in ("scan", "loop"):
            empty_y, empty_x = cell(u[:, :0], state=initial, mode=mode)
            self.assertEqual(empty_y.shape, (2, 0, 3))
            torch.testing.assert_close(empty_x[:, 0], initial)


class FullMetricL2NIntegrationTests(unittest.TestCase):
    def test_cache_mutation_checkpoint_and_autocast(self):
        cell = full_cell().float().eval()
        clone = copy.deepcopy(cell)
        with torch.no_grad():
            first = cell.normalized_block_matrices()
            self.assertIs(first[0], cell.normalized_block_matrices()[0])
            cell.metric_upper_raw.add_(.1)
            self.assertIsNot(first[0], cell.normalized_block_matrices()[0])
            clone.load_state_dict(cell.state_dict())
            u = torch.randn(2, 7, 2)
            torch.testing.assert_close(clone(u), cell(u))
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(all(x.dtype == torch.float32 for x in cell.compute_block_matrices()))
            self.assertEqual(cell.storage_matrix().dtype, torch.float32)
            self.assertEqual(cell.state_energy(u.new_ones(2, 5)).dtype, torch.float32)
        cell.double()
        self.assertEqual(cell.storage_matrix().dtype, torch.float64)

    def test_deep_metric_and_context_stacks_use_the_dense_certificate(self):
        config = SSMConfig(param="l2n", l2n_state_metric="full", d_model=2,
                           d_state=5, n_layers=2, gamma=.8, ff="LGLU2")
        deep = DeepSSM(1, 1, config=config).double().eval()
        metric = MetricDeepSSM(1, 1, config=MetricSSMConfig(
            param="l2n", state_metric="full", d_model=2, d_state=5, n_layers=2, gamma=.8,
        )).double().eval()
        for model in (deep, metric):
            cores = [getattr(block, "lru", getattr(block, "core", None)) for block in model.blocks]
            for core in cores:
                core.metric_upper_raw.data.normal_(0, .2)
            initial = [torch.randn(2, 5).double() for _ in cores]
            u = torch.randn(2, 9, 1).double()
            with torch.no_grad():
                ys, xs = model(u, state=initial, mode="scan")
                yl, _ = model(u, state=initial, mode="loop")
                torch.testing.assert_close(ys, yl)
                last = xs if model is deep else tuple(x[:, -1] for x in xs)
                residue = (ys.square().sum((1, 2)) + model.storage_value(last)
                           - model.storage_value(initial) - .8**2 * u.square().sum((1, 2)))
                self.assertLessEqual(residue.max().item(), 2e-10)
            model.train()
            output = model(u)[0]
            output.square().sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        contextual = ContextualDeepSSM(1, 1, 1, ssm_config=config, context_modes="gate").double().eval()
        u, context = torch.randn(2, 7, 1).double(), torch.randn(2, 7, 1).double()
        y, _ = contextual(u, context)
        self.assertLessEqual(y.norm().item(), .8 * u.norm().item())

    @unittest.skipUnless(importlib.util.find_spec("nonlinear_benchmarks"), "optional benchmark dependency unavailable")
    def test_direct_configuration_and_benchmark_factory(self):
        from Test_files import run_benchmarks as runner
        torch.manual_seed(17)
        direct = DeepSSM(1, 1, param="l2n", l2n_state_metric="full", d_model=2,
                         d_state=5, n_layers=1, gamma=1, ff="LGLU2").double()
        torch.manual_seed(17)
        configured = DeepSSM(1, 1, config=SSMConfig(
            param="l2n", l2n_state_metric="full", d_model=2, d_state=5, n_layers=1,
            gamma=1, ff="LGLU2", max_phase_b=.5, nl_layers=3, train_ff_lip=None,
        )).double()
        u = torch.randn(2, 7, 1).double()
        torch.testing.assert_close(direct(u)[0], configured(u)[0])
        cfg = runner.GlobalModelConfig(d_model=2, d_state=5, n_layers=1,
                                       l2n_state_metric="full", gamma_override="auto", ff_override="auto")
        for name in ("l2n", "metric_l2n"):
            model = runner.build_model(name, 1, 1, cfg)
            self.assertEqual(model.mode, "scan")
            self.assertEqual(model.core.config.l2n_state_metric if name == "l2n"
                             else model.core.config.state_metric, "full")
            self.assertEqual(model(u.float()).shape, (2, 7, 1))
        self.assertEqual(runner._initialization_metadata(cfg)["l2n"]["state_metric"], "full")

    def test_fullgraph_compile_matches_outputs_and_gradients(self):
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

    def test_identity_custom_core_storage_uses_scan_coordinates(self):
        # A standalone identity L2N returns S^-1 z, but a metric stack evolves
        # its normalized blocks directly in z. Its energy must therefore be ||z||².
        model = MetricDeepSSM(1, 1, d_model=2, d_state=4, n_layers=1,
                             core_factory=lambda n, m, p: Block2x2DenseL2SSM(n, m, p)).double()
        core = model.blocks[0].core
        core.S.data.mul_(10)
        u = torch.randn(2, 7, 1).double()
        y, states = model(u)
        torch.testing.assert_close(model.storage_value(states), states[0].square().sum(-1))
        self.assertLessEqual((y.square().sum((1, 2)) + model.storage_value(states)[:, -1]
                              - u.square().sum((1, 2))).max().item(), 2e-10)


if __name__ == "__main__":
    unittest.main()
