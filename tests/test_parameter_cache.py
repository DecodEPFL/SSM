"""Cache reuse and lifecycle regressions for certified parameter constructions."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from neural_ssm.rens.ren import REN
from neural_ssm.ssm import DeepSSM, DefectL2SSM
from neural_ssm.utils.runtime import EvalCacheMixin
from neural_ssm.ssm.cells.lti import Block2x2DenseL2SSM
from neural_ssm.ssm.cells.legacy import L2BoundedLTICell, L2RU, lruz
from neural_ssm.utils.scan import GraphedDiagScan, diag_affine_scan
from neural_ssm.static_layers import (
    L2BoundedLinearExact, SandwichFc, SandwichLin,
)


class ParameterCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_all_certified_lti_builders_reuse_and_invalidate(self):
        cases = (
            (L2RU(4, gamma=1.), "set_param", "_compute_ssm"),
            (lruz(2, 2, 4, gamma=1.), "set_param", "_compute_ssm"),
            (L2BoundedLTICell(4, 2, 2), "compute_ssm_matrices", "_compute_ssm_matrices"),
            (Block2x2DenseL2SSM(4, 2, 2), "compute_z_matrices", "_compute_z_matrices"),
            (DefectL2SSM(4, 2, 2), "compute_block_matrices", "_compute_block_matrices"),
        )
        for cell, public, builder in cases:
            with self.subTest(cell=type(cell).__name__):
                cell.eval()
                build = getattr(cell, public)
                with patch.object(cell, builder, wraps=getattr(cell, builder)) as compute:
                    with torch.no_grad():
                        build()
                        build()
                        self.assertEqual(compute.call_count, 1)
                        # Fixed gain buffers are part of the certificate too.
                        gamma = next(b for name, b in cell.named_buffers() if "gamma" in name)
                        gamma.add_(.01)
                        build()
                        self.assertEqual(compute.call_count, 2)
                        cell.train()
                        build()
                        build()
                        self.assertEqual(compute.call_count, 4)
                    cell.eval()
                    # Each grad-enabled evaluation must own a fresh graph.
                    for _ in range(2):
                        cell.zero_grad(set_to_none=True)
                        values = build()
                        sum(v.abs().square().sum() for v in values).backward()
                    self.assertEqual(compute.call_count, 6)
                    self.assertTrue(any(p.grad is not None for p in cell.parameters()))

    def test_projection_refreshes_after_update_and_checkpoint_load(self):
        layer = L2BoundedLinearExact(3, 4).eval()
        x = torch.randn(2, 3)
        checkpoint = copy.deepcopy(layer.state_dict())
        with torch.no_grad():
            original = layer(x)
            layer.W_raw.mul_(.2)
            updated = layer(x)
            torch.testing.assert_close(updated, F.linear(x, layer._effective_weight()))
            self.assertFalse(torch.allclose(updated, original))
            layer.load_state_dict(checkpoint)
            torch.testing.assert_close(layer(x), original)

    def test_projection_refreshes_after_parameter_replacement_and_bound_change(self):
        layer = L2BoundedLinearExact(3, 4).eval()
        x = torch.randn(2, 3)
        with torch.no_grad():
            layer(x)
            # Match the mutation counter deliberately: identity must distinguish it.
            replacement = nn.Parameter(layer.W_raw.detach().clone() * .2)
            while replacement._version < layer.W_raw._version:
                replacement.add_(0)
            layer.W_raw = replacement
            torch.testing.assert_close(layer(x), F.linear(x, layer._effective_weight()))
            layer.bound = .01
            torch.testing.assert_close(layer(x), F.linear(x, layer._effective_weight()))
            self.assertLessEqual(float(torch.linalg.matrix_norm(layer._eval_cache["weight"][1], 2)), .01001)

    def test_projection_eval_gradients_match_uncached_formula(self):
        layer = L2BoundedLinearExact(3, 4).eval()
        x = torch.randn(2, 3)
        with torch.no_grad():
            layer(x)
        cached_grad = torch.autograd.grad(layer(x).square().sum(), layer.W_raw)[0]
        reference_grad = torch.autograd.grad(
            F.linear(x, layer._effective_weight()).square().sum(), layer.W_raw
        )[0]
        torch.testing.assert_close(cached_grad, reference_grad)

    def test_projection_cache_hits_for_exact_and_power_iteration(self):
        for exact in (True, False):
            with self.subTest(exact=exact):
                layer = L2BoundedLinearExact(3, 4, exact_norm=exact).eval()
                x = torch.randn(2, 3)
                with patch.object(layer, "_effective_weight", wraps=layer._effective_weight) as compute:
                    with torch.no_grad():
                        layer(x)
                        layer(x)
                        self.assertEqual(compute.call_count, 1)
                    for _ in range(2):
                        layer.zero_grad(set_to_none=True)
                        layer(x).square().sum().backward()
                    self.assertEqual(compute.call_count, 3)
                    self.assertTrue(torch.isfinite(layer.W_raw.grad).all())

    def test_dtype_conversion_clears_cached_tensors(self):
        cell = Block2x2DenseL2SSM(4, 2, 2).eval()
        cell.init_on_circle(.8)
        layer = L2BoundedLinearExact(2, 2).eval()
        with torch.no_grad():
            cell(torch.randn(2, 5, 2))
            layer(torch.randn(2, 2))
            cell.double()
            layer.double()
            self.assertNotIn("_eval_cache", cell.__dict__)
            self.assertNotIn("_eval_cache", layer.__dict__)
            y, states = cell(torch.randn(2, 5, 2, dtype=torch.float64))
            self.assertEqual(y.dtype, torch.float64)
            self.assertEqual(states.dtype, torch.float64)
            self.assertEqual(layer(torch.randn(2, 2, dtype=torch.float64)).dtype, torch.float64)

    def test_l2n_reuses_lu_across_sequences_and_modes(self):
        cell = Block2x2DenseL2SSM(4, 2, 2).double().eval()
        cell.init_on_circle(.8)
        u = torch.randn(2, 7, 2, dtype=torch.float64)
        with torch.no_grad(), patch("torch.linalg.lu_factor", wraps=torch.linalg.lu_factor) as factor:
            reference = cell(u, mode="scan")
            for mode in ("scan", "conv", "loop"):
                result = cell(u, mode=mode)
                for actual, expected in zip(result, reference):
                    torch.testing.assert_close(actual, expected)
            self.assertEqual(factor.call_count, 1)
            cell.S.add_(.01 * torch.eye(4, dtype=cell.S.dtype))
            result = cell(u, mode="scan")
            self.assertEqual(factor.call_count, 2)
            # A fresh solve of the current basis must recover the returned states.
            z = result[1] @ cell.S.T
            torch.testing.assert_close(result[1], torch.linalg.solve(cell.S, z.flatten(0, 1).T).T.reshape_as(z))

    def test_deepssm_encoder_decoder_reuse_refresh_and_gradients(self):
        model = DeepSSM(2, 2, d_model=4, d_state=4, n_layers=1,
                        param="tv", ff="LGLU2", gamma=1.).eval()
        with patch.object(model, "_spectrally_capped_weight", wraps=model._spectrally_capped_weight) as cap:
            with torch.no_grad():
                model._capped_encoder_decoder()
                model._capped_encoder_decoder()
                self.assertEqual(cap.call_count, 2)
                model.encoder_w.mul_(.2)
                enc, _ = model._capped_encoder_decoder()
                self.assertEqual(cap.call_count, 4)
                torch.testing.assert_close(enc, model._spectrally_capped_weight(model.encoder_w))
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                enc, dec = model._capped_encoder_decoder()
                (enc.square().sum() + dec.square().sum()).backward()
                self.assertIsNotNone(model.encoder_w.grad)
                self.assertIsNotNone(model.decoder_w.grad)

    def test_tv_tvc_cache_projections_but_recompute_selectors(self):
        for param in ("tv", "tvc"):
            with self.subTest(param=param):
                model = DeepSSM(2, 2, d_model=4, d_state=4, n_layers=1,
                                param=param, ff="LGLU2", gamma=1.).eval()
                cell = model.blocks[0].lru
                u = torch.randn(2, 7, 4)
                with patch.object(cell.in_proj, "_effective_weight", wraps=cell.in_proj._effective_weight) as in_cap, \
                     patch.object(cell.out_proj, "_effective_weight", wraps=cell.out_proj._effective_weight) as out_cap, \
                     patch.object(cell.param_net, "forward", wraps=cell.param_net.forward) as selector:
                    with torch.no_grad():
                        cell(u)
                        actual = cell(2 * u)[0]
                        self.assertEqual(in_cap.call_count, 1)
                        self.assertEqual(out_cap.call_count, 1)
                        self.assertEqual(selector.call_count, 2)
                        # Toggling eval clears every cache for an independent recomputation.
                        cell.eval()
                        expected = cell(2 * u)[0]
                        torch.testing.assert_close(actual, expected)

    def test_gain_and_storage_diagnostics_reuse_encoder_decoder_caps(self):
        model = DeepSSM(2, 2, d_model=4, d_state=4, n_layers=1,
                        param="tv", ff="LGLU2", gamma=1.).eval()
        with torch.no_grad(), patch.object(model, "_spectrally_capped_weight", wraps=model._spectrally_capped_weight) as cap:
            model._capped_encoder_decoder()
            model.certified_gain_bound()
            model.storage_weights()
            model.gain_diagnostics()
            self.assertEqual(cap.call_count, 2)
            model.decoder_w.mul_(.2)
            model.storage_weights()
            self.assertEqual(cap.call_count, 4)

    def test_ren_reuses_construction_and_refreshes_after_update(self):
        model = REN(2, 2, 4, 3, gammat=1.).double().eval()
        u = torch.randn(2, 7, 2, dtype=torch.float64)
        with patch.object(model, "_compute_params", wraps=model._compute_params) as compute:
            with torch.no_grad():
                original = model(u)
                torch.testing.assert_close(model(u), original)
                self.assertEqual(compute.call_count, 1)
                model.gamma.mul_(.9)
                refreshed = model(u)
                self.assertEqual(compute.call_count, 2)
                model.eval()
                torch.testing.assert_close(model(u), refreshed)
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                model(u).square().sum().backward()
                self.assertTrue(torch.isfinite(model.X.grad).all())

    def test_sandwich_weights_refresh_and_keep_eval_gradients(self):
        for cls in (SandwichLin, SandwichFc):
            with self.subTest(layer=cls.__name__):
                layer = cls(3, 4, bias=False).eval()
                x = torch.randn(2, 3)
                with torch.no_grad():
                    layer(x)
                    layer.weight.add_(.1 * torch.randn_like(layer.weight))
                    refreshed = layer(x)
                    layer.eval()
                    torch.testing.assert_close(layer(x), refreshed)
                    layer.double()
                    layer(x.double())
                layer(x.double()).square().sum().backward()
                self.assertIsNotNone(layer.weight.grad)

    def test_autocast_does_not_reuse_other_precision(self):
        class Product(EvalCacheMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

            def forward(self):
                return self._eval_cached("product", lambda: self.weight @ self.weight.T)

        model = Product().eval()
        with torch.no_grad():
            original = model()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertEqual(model().dtype, torch.bfloat16)
            torch.testing.assert_close(model(), original)
            self.assertEqual(model().dtype, torch.float32)

    def test_module_created_inside_inference_mode_stays_fresh(self):
        with torch.inference_mode():
            model = L2BoundedLinearExact(2, 2).eval()
            x = torch.randn(2, 2)
            original = model(x)
            model.W_raw.mul_(.2)
            torch.testing.assert_close(model(x), .2 * original)

    def test_compiled_inference_preserves_caches_and_parameter_refresh(self):
        layer = L2BoundedLinearExact(3, 4).eval()
        compiled = torch.compile(layer, backend="aot_eager")
        x = torch.randn(2, 3)
        with torch.inference_mode():
            reference = layer(x)
            weight = layer._eval_cache["weight"][1]
            torch.testing.assert_close(compiled(x), reference)
            self.assertIs(layer._eval_cache["weight"][1], weight)
            layer.W_raw.mul_(.2)
            actual = compiled(x)
            self.assertIsNot(layer._eval_cache["weight"][1], weight)
            torch.testing.assert_close(actual, F.linear(x, layer._effective_weight()))


class ScanGraphCacheTests(unittest.TestCase):
    def test_signature_tracks_both_inputs_and_gradient_requirements(self):
        a = torch.randn(2, 7, 4)
        b = torch.randn_like(a)
        key = GraphedDiagScan._signature(a, b, True)
        changes = (
            (a, b.double()),
            (a, b[:, :, :1]),
            (a, b.transpose(0, 1).contiguous().transpose(0, 1)),
            (a.detach().requires_grad_(True), b),
        )
        for left, right in changes:
            self.assertNotEqual(key, GraphedDiagScan._signature(left, right, True))
        # Flags do not matter for a gradient-free forward capture.
        self.assertEqual(
            GraphedDiagScan._signature(a, b, False),
            GraphedDiagScan._signature(a.detach(), b, False),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_failed_capture_is_not_retried_on_every_call(self):
        scan = GraphedDiagScan()
        a = torch.rand(2, 7, 4, device="cuda")
        b = torch.randn_like(a)
        with torch.no_grad(), patch.object(scan, "_capture_forward", side_effect=RuntimeError("unsupported capture")) as capture:
            for _ in range(2):
                torch.testing.assert_close(scan(a, b), diag_affine_scan(a, b))
            self.assertEqual(capture.call_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_graph_supports_one_or_both_gradient_inputs(self):
        scan = GraphedDiagScan()
        for grad_a, grad_b in ((True, False), (False, True), (True, True)):
            a = torch.rand(2, 7, 4, device="cuda", requires_grad=grad_a)
            b = torch.randn(2, 7, 4, device="cuda", requires_grad=grad_b)
            inputs = tuple(t for t in (a, b) if t.requires_grad)
            expected = diag_affine_scan(a, b)
            expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
            actual = scan(a, b)
            actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
            torch.testing.assert_close(actual, expected)
            for value, ref in zip(actual_grads, expected_grads):
                torch.testing.assert_close(value, ref)
            self.assertIn(scan._signature(a, b, True), scan._graphed)


if __name__ == "__main__":
    unittest.main()
