"""Independent certificates, derivatives, and runtime checks for metric SSMs."""
import copy
import importlib.util
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from neural_ssm import MetricDeepSSM, MetricSSMConfig, Block2x2DenseL2SSM
from neural_ssm.ssm.metric_layers import FeatureMetric, WeightedResidualFeedforward


def dense(a):
    return torch.block_diag(*a.unbind(0))


def make_model(param="defect", **settings):
    cfg = replace(MetricSSMConfig(d_model=3, d_state=4, n_layers=3, d_hidden=5),
                  param=param, **settings)
    return MetricDeepSSM(2, 2, config=cfg).double()


class MetricSSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(19)

    def test_metric_matches_independent_eigenvalue_formula(self):
        metric = FeatureMetric(4).double()
        with torch.no_grad():
            metric.lower_raw.normal_(std=.3)
            metric.diag_raw.normal_(mean=.3, std=.2)
            metric.v_raw.normal_(mean=.4, std=.3)
        q, fq, w, fw = metric()
        v = (torch.nn.functional.softplus(metric.v_raw) + metric.floor).diag()
        lower = (torch.nn.functional.softplus(metric.diag_raw) + metric.floor).diag()
        lower[metric.indices[0], metric.indices[1]] = metric.lower_raw
        eig, vectors = torch.linalg.eigh(lower @ lower.T)
        reference = v.sqrt() @ ((vectors * (2 * eig / (1 + eig))) @ vectors.T) @ v.sqrt()
        torch.testing.assert_close(q, reference)
        torch.testing.assert_close(fq.T @ fq, q)
        torch.testing.assert_close(fw.T @ fw, w)
        torch.testing.assert_close(w, v @ torch.linalg.solve(2 * v - q, v))
        self.assertGreater(torch.linalg.eigvalsh(q).min().item(), 0)
        self.assertGreater(torch.linalg.eigvalsh(2 * v - q).min().item(), 0)

    def test_metric_can_exceed_v_and_terminal_has_no_redundant_parameters(self):
        metric = FeatureMetric(3).double()
        with torch.no_grad():
            value = torch.log(torch.expm1(torch.tensor(3.0**.5 - metric.floor)))
            metric.diag_raw.fill_(value)
        q, _, _, _ = metric()
        v = (torch.nn.functional.softplus(metric.v_raw) + metric.floor).diag()
        torch.testing.assert_close(q, 1.5 * v)
        terminal = FeatureMetric(3, terminal=True).double()
        self.assertEqual(sum(p.numel() for p in terminal.parameters()), 0)
        for matrix in terminal():
            torch.testing.assert_close(matrix, torch.eye(3, dtype=torch.float64))

    def test_residual_certificate_uses_coupled_weights(self):
        metric = FeatureMetric(3).double()
        metric.lower_raw.data.normal_(std=.3)
        q, fq, _, _ = metric()
        ff = WeightedResidualFeedforward(3, 7, 1e-4, "tanh", .001).double()
        h, o = ff.matrices(fq)
        lam = (torch.nn.functional.softplus(ff.lambda_raw) + ff.floor).diag()
        torch.testing.assert_close(q @ o, -h.T @ lam)
        self.assertGreaterEqual(torch.linalg.eigvalsh(2 * lam - o.T @ q @ o).min().item(), -1e-12)
        u, alt = torch.randn(20, 3), torch.randn(20, 3)
        u, alt = u.double(), alt.double()
        y, ya = u + (u @ h.T).tanh() @ o.T, alt + (alt @ h.T).tanh() @ o.T
        self.assertTrue(torch.all(((y-ya) @ q * (y-ya)).sum(-1)
                                 <= ((u-alt) @ q * (u-alt)).sum(-1) + 1e-12))

    def test_each_layer_satisfies_dense_brl_and_nonlinear_energy_inequality(self):
        for param in ("defect", "l2n"):
            model = make_model(param)
            with torch.no_grad():
                for block in model.blocks[:-1]:
                    block.metric.lower_raw.normal_(std=.3)
                    block.metric.v_raw.normal_(std=.3)
            _, prepared, _ = model.prepared_matrices()
            for block, matrices in zip(model.blocks, prepared):
                a, b, c, d = dense(matrices.a), matrices.b, matrices.c, matrices.d
                eye = torch.eye(a.shape[0], dtype=torch.float64)
                brl = torch.cat((
                    torch.cat((eye-a.T@a-c.T@matrices.w@c, -a.T@b-c.T@matrices.w@d), 1),
                    torch.cat((-b.T@a-d.T@matrices.w@c,
                               matrices.q_prev-b.T@b-d.T@matrices.w@d), 1)), 0)
                self.assertGreaterEqual(torch.linalg.eigvalsh(brl).min().item(), -1e-11)
                x, xa = torch.randn(10, 4).double(), torch.randn(10, 4).double()
                u, ua = torch.randn(10, 1, 3).double(), torch.randn(10, 1, 3).double()
                y, states = block(u, x, matrices)
                ya, states_a = block(ua, xa, matrices)
                dy, du = (y-ya)[:, 0], (u-ua)[:, 0]
                dx, dxn = x-xa, states[:, -1]-states_a[:, -1]
                residual = dxn.square().sum(-1)-dx.square().sum(-1)
                residual += (dy @ matrices.q * dy).sum(-1)-(du @ matrices.q_prev * du).sum(-1)
                self.assertLessEqual(residual.max().item(), 1e-11)

    def test_global_incremental_certificate_includes_terminal_storage(self):
        for param in ("defect", "l2n"):
            for activation in ("tanh", "relu", "arctan", "identity"):
                model = make_model(param, gamma=.7, activation=activation)
                u, ua = torch.randn(2, 13, 2).double(), torch.randn(2, 13, 2).double()
                # Equal nonzero initial states still give zero initial difference storage.
                initial = tuple(torch.randn(2, 4).double() for _ in model.blocks)
                y, last = model(u, initial, return_state=False, return_last=True)
                ya, last_a = model(ua, initial, return_state=False, return_last=True)
                terminal = model.storage_value(tuple(x-xa for x, xa in zip(last, last_a)))
                output = (y-ya).square().sum((1, 2))
                limit = model.gamma.square() * (u-ua).square().sum((1, 2))
                self.assertTrue(torch.all(output + terminal <= limit + 1e-11))
                zeros = torch.zeros_like(u)
                torch.testing.assert_close(model(zeros, return_state=False), zeros)

    def test_finite_horizon_jacobian_has_prescribed_norm(self):
        model = make_model(gamma=.6, n_layers=2)
        u = torch.randn(1, 5, 2, dtype=torch.float64)
        jac = torch.autograd.functional.jacobian(lambda x: model(x, return_state=False), u)
        norm = torch.linalg.matrix_norm(jac.reshape(10, 10), 2)
        self.assertLessEqual(norm.item(), model.gamma.item()+1e-11)

    def test_scan_loop_outputs_states_and_parameter_gradients_agree(self):
        for param in ("defect", "l2n"):
            scan, loop = make_model(param), None
            loop = copy.deepcopy(scan)
            u = torch.randn(2, 11, 2).double()
            ys, xs = scan(u, mode="scan")
            yl, xl = loop(u, mode="loop")
            torch.testing.assert_close(ys, yl)
            for a, b in zip(xs, xl):
                torch.testing.assert_close(a, b)
            (ys.square().sum()+sum(x.square().sum() for x in xs)).backward()
            (yl.square().sum()+sum(x.square().sum() for x in xl)).backward()
            for (name, p), (_, pl) in zip(scan.named_parameters(), loop.named_parameters()):
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                torch.testing.assert_close(p.grad, pl.grad)

    def test_metric_and_residual_gradients_match_finite_differences(self):
        model = make_model(n_layers=2)
        u = torch.randn(1, 7, 2).double()
        model(u, return_state=False).square().sum().backward()
        parameters = [model.blocks[0].metric.lower_raw, model.blocks[0].metric.diag_raw,
                      model.blocks[0].metric.v_raw, model.blocks[0].ff.lambda_raw]
        step = 1e-5
        for p in parameters:
            analytical = p.grad[0].item()
            old = p[0].item()
            with torch.no_grad():
                p[0] = old + step
                plus = model(u, return_state=False).square().sum().item()
                p[0] = old - step
                minus = model(u, return_state=False).square().sum().item()
                p[0] = old
            self.assertAlmostEqual(analytical, (plus-minus)/(2*step), delta=2e-7)

    def test_streaming_explicit_and_internal_states_match_full_sequence(self):
        model = make_model()
        u = torch.randn(2, 12, 2).double()
        full, last_full = model(u, return_state=False, return_last=True)
        left, last = model(u[:, :5], return_state=False, return_last=True)
        right, last_explicit = model(u[:, 5:], last, return_state=False, return_last=True)
        torch.testing.assert_close(torch.cat((left, right), 1), full)
        model(u[:, :5], return_state=False)
        right_internal = model(u[:, 5:], return_state=False, reset_state=False)
        torch.testing.assert_close(right_internal, right)
        for a, b in zip(last_explicit, last_full):
            torch.testing.assert_close(a, b)
        model.reset()
        self.assertIsNone(model.state)
        _, detached = model(u, return_state=False, return_last=True, detach_state=True)
        self.assertTrue(all(x.grad_fn is None for x in detached))
        model.eval()
        with torch.no_grad():
            model(u, return_state=False)
        for last in model.state:
            self.assertEqual(last.untyped_storage().nbytes(), last.numel()*last.element_size())

    def test_cache_tracks_shared_metrics_checkpoints_dtype_and_grad_mode(self):
        model = make_model().eval()
        u = torch.randn(1, 7, 2).double()
        with torch.no_grad(), patch.object(model, "_prepare", wraps=model._prepare) as build:
            first = model.prepared_matrices()
            self.assertIs(first, model.prepared_matrices())
            self.assertEqual(build.call_count, 1)
            model.blocks[0].metric.lower_raw.add_(.1)
            second = model.prepared_matrices()
            self.assertIsNot(first, second)
            self.assertFalse(torch.equal(first[1][1].b, second[1][1].b))
            model(u)
            model.load_state_dict(model.state_dict())
            model(u)
            self.assertEqual(build.call_count, 3)
        model(u, return_state=False).sum().backward()
        self.assertIsNotNone(model.blocks[0].metric.lower_raw.grad)
        model.float()
        self.assertEqual(model.prepared_matrices()[0].dtype, torch.float32)

    def test_checkpoints_time_first_empty_sequence_and_single_input(self):
        model = make_model().eval()
        restored = make_model().eval()
        restored.load_state_dict(model.state_dict())
        u = torch.randn(2, 9, 2).double()
        torch.testing.assert_close(model(u)[0], restored(u)[0])
        yt, xt = model(u.transpose(0, 1), time_first=True)
        y, x = model(u)
        torch.testing.assert_close(yt, y.transpose(0, 1))
        for a, b in zip(xt, x):
            torch.testing.assert_close(a, b.transpose(0, 1))
        empty, states = model(u[:, :0])
        self.assertEqual(empty.shape, (2, 0, 2))
        self.assertTrue(all(x.shape == (2, 1, 4) for x in states))
        self.assertEqual(model(u[0, 0])[0].shape, (1, 1, 2))
        integer_gain = MetricDeepSSM(2, 2, d_model=3, d_state=4, n_layers=1, gamma=1)
        self.assertEqual(integer_gain(u.float(), return_state=False).shape, u.shape)

    def test_custom_dense_core_protocol_and_invalid_cores(self):
        class DenseCore(nn.Module):
            def __init__(self, n, inputs, outputs):
                super().__init__()
                self.d_state, self.d_input, self.d_output = n, inputs, outputs
                # A fixed, genuinely dense contraction. requires_grad exercises
                # detachment even when a core returns a view of a leaf tensor.
                basis = torch.linalg.qr(torch.randn(n+inputs, n+inputs))[0]
                self.register_buffer("k", (.5 * basis).requires_grad_())

            def normalized_block_matrices(self):
                n = self.d_state
                return self.k[:n, :n], self.k[:n, n:], self.k[n:, :n], self.k[n:, n:]

        cfg = MetricSSMConfig(d_model=3, d_state=5, n_layers=1, ff="none")
        model = MetricDeepSSM(2, 2, config=cfg, core_factory=DenseCore).double()
        u = torch.randn(2, 9, 2).double()
        torch.testing.assert_close(model(u, mode="scan")[0], model(u, mode="loop")[0])
        model.eval()
        with torch.no_grad():
            _, prepared, _ = model.prepared_matrices()
            self.assertFalse(prepared[0].a.requires_grad)
            k = model.blocks[0].core.k
            self.assertLess(torch.linalg.matrix_norm(k, 2).item(), 1)
        with self.assertRaisesRegex(TypeError, "normalized_block_matrices"):
            MetricDeepSSM(2, 2, config=cfg, core_factory=lambda *args: nn.Identity())
        with self.assertRaisesRegex(ValueError, "Built-in"):
            MetricDeepSSM(2, 2, param="lru")
        with self.assertRaisesRegex(ValueError, "ff must"):
            MetricDeepSSM(2, 2, ff="GLU")
        with self.assertRaisesRegex(ValueError, "even"):
            MetricDeepSSM(2, 2, param="l2n", d_state=5)

    def test_l2n_energy_only_has_no_dense_basis_and_safe_svd_fallback(self):
        core = Block2x2DenseL2SSM(4, 2, 2, energy_only=True).double()
        core.init_on_circle(rho=.8)
        self.assertIsNone(core.S)
        u = torch.randn(2, 7, 2).double()
        torch.testing.assert_close(core(u, mode="scan")[0], core(u, mode="loop")[0])
        with patch("torch.linalg.matrix_norm", side_effect=RuntimeError("SVD failed")):
            a, b, c, d = core.normalized_block_matrices()
        k = torch.cat((torch.cat((dense(a), b), 1), torch.cat((c, d), 1)), 0)
        self.assertLessEqual(torch.linalg.matrix_norm(k, 2).item(), 1)
        core.exact_norm = False
        with self.assertRaisesRegex(ValueError, "exact_norm"):
            core.normalized_block_matrices()

    def test_adam_learning_reduces_loss_and_preserves_certificate(self):
        model = make_model(n_layers=2, gamma=1.0)
        u = .3*torch.randn(4, 24, 2).double()
        target = .35*u
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        initial = (model(u, return_state=False)-target).square().mean().item()
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(u, return_state=False)-target).square().mean()
            loss.backward()
            optimizer.step()
        final = (model(u, return_state=False)-target).square().mean().item()
        self.assertLess(final, initial*.8)
        alt = torch.randn_like(u)
        y, last = model(u, return_state=False, return_last=True)
        ya, last_a = model(alt, return_state=False, return_last=True)
        terminal = model.storage_value(tuple(a-b for a, b in zip(last, last_a)))
        self.assertTrue(torch.all((y-ya).square().sum((1, 2))+terminal
                                 <= (u-alt).square().sum((1, 2))+1e-11))

    def test_fullgraph_compile_matches_outputs_and_all_gradients(self):
        for param in ("defect", "l2n"):
            torch._dynamo.reset()
            eager = make_model(param, n_layers=2)
            compiled_model = copy.deepcopy(eager)
            compiled = torch.compile(compiled_model, backend="aot_eager", fullgraph=True)
            u = torch.randn(2, 8, 2).double().requires_grad_()
            uc = u.detach().clone().requires_grad_()
            y = eager(u, return_state=False)
            yc = compiled(uc, return_state=False)
            torch.testing.assert_close(yc, y)
            y.square().sum().backward()
            yc.square().sum().backward()
            torch.testing.assert_close(uc.grad, u.grad)
            for original, traced in zip(eager.parameters(), compiled_model.parameters()):
                torch.testing.assert_close(traced.grad, original.grad)

    def test_cpu_autocast_keeps_certificate_construction_at_parameter_precision(self):
        for param in ("defect", "l2n"):
            model = make_model(param).float()
            u = torch.randn(2, 9, 2)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                encoder, prepared, decoder = model.prepared_matrices()
                self.assertEqual(encoder.dtype, torch.float32)
                self.assertEqual(decoder.dtype, torch.float32)
                self.assertTrue(all(p.q.dtype == torch.float32 for p in prepared))
                y = model(u, return_state=False)
                self.assertEqual(y.dtype, torch.bfloat16)
                loss = y.float().square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    @unittest.skipUnless(importlib.util.find_spec("nonlinear_benchmarks"), "optional benchmark dependency unavailable")
    def test_benchmark_builders_and_parameter_budget_use_the_new_architecture(self):
        from Test_files import run_benchmarks as runner
        from Test_files import benchmark_ui as ui
        cfg = runner.GlobalModelConfig(d_model=4, d_state=4, n_layers=2,
                                       d_hidden=5, gamma_override=".8")
        for name in ("metric_defect", "metric_l2n"):
            self.assertIn(name, ui.MODELS)
            model = runner.build_model(name, 2, 3, cfg)
            self.assertIsInstance(model.core, runner.MetricDeepSSM)
            y = model(torch.randn(2, 8, 2))
            self.assertEqual(y.shape, (2, 8, 3))
            y.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
            self.assertAlmostEqual(model.diagnostics()["certified_gain_bound"], .8, places=6)
            target = runner._count_params(model)
            matched = runner.build_model(name, 2, 3, cfg, param_budget=target)
            self.assertEqual(runner._count_params(matched), target)
        with self.assertRaisesRegex(ValueError, "--gamma"):
            runner.build_model("metric_defect", 2, 2, replace(cfg, gamma_override="none"))

    @unittest.skipUnless(importlib.util.find_spec("nonlinear_benchmarks"), "optional benchmark dependency unavailable")
    def test_ui_forwards_metric_and_core_settings_without_a_desktop_window(self):
        from Test_files import benchmark_ui as ui
        from Test_files import run_benchmarks as runner
        values = dict(
            epochs="1", batch_size="2", lr=".001", plot_every="1", plot_max_length="32",
            d_model="4", d_state="4", n_layers="2", d_hidden="5", nl_layers="1",
            gamma=".8", report_metric="rmse", param_budget="none", device="cpu",
            ff="auto", mode="auto", metric_activation="arctan", metric_ff="none",
            metric_init_feedthrough=".9", defect_block_size="2", defect_rho=".8",
            defect_max_radius=".999", defect_factor_margin=".001", defect_init_scale=".02",
            defect_state_metric="full", l2n_state_metric="full",
            defect_max_phase=".2", defect_phase_center="0", l2n_rho=".85",
            l2n_max_phase=".3", l2n_phase_center="0", l2n_offdiag_scale=".02",
        )
        variable = lambda value: SimpleNamespace(get=lambda: value)
        panel = ui.BenchmarkUI.__new__(ui.BenchmarkUI)
        panel.vars = {key: variable(value) for key, value in values.items()}
        panel.bench_list = SimpleNamespace(curselection=lambda: (0,))
        panel.model_list = SimpleNamespace(curselection=lambda: tuple(
            ui.MODELS.index(name) for name in ("metric_defect", "metric_l2n")))
        panel.out_var = variable("/tmp/ssm-metric-ui-test")
        for name in ("show_var", "gif_var", "amp_var", "compile_var", "pcg_var",
                     "cudagraph_var", "defect_random_phase_var", "l2n_random_phase_var"):
            setattr(panel, name, variable(False))
        args = runner.build_parser().parse_args(panel._build_command()[3:])
        self.assertEqual(args.models, ["metric_defect", "metric_l2n"])
        self.assertEqual(args.metric_activation, "arctan")
        self.assertEqual(args.metric_ff, "none")
        self.assertEqual(args.metric_init_feedthrough, .9)
        self.assertEqual(args.defect_rho, .8)
        self.assertEqual(args.defect_state_metric, "full")
        self.assertEqual(args.l2n_state_metric, "full")
        self.assertEqual(args.l2n_rho, .85)
        self.assertFalse(args.l2n_random_phase)
        self.assertFalse(args.defect_random_phase)


if __name__ == "__main__":
    unittest.main()
