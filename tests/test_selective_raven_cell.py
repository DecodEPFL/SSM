import math
import unittest

import torch

from src.neural_ssm.ssm import DeepSSM, SSMConfig, L2SelectiveRavenCell
from src.neural_ssm.ssm.layers import _SSM_PARAMETRIZATIONS, _CERTIFIED_PARAMETRIZATIONS
from src.neural_ssm.ssm.selective_cells import _spectral_cap


def _cell(**overrides) -> L2SelectiveRavenCell:
    torch.manual_seed(0)
    kwargs = dict(
        d_model=8,
        num_heads=2,
        num_slots=6,
        key_dim=4,
        value_dim=5,
        top_k=3,
        gamma=1.5,
        alpha=1.0,
        rho_max=0.99,
    )
    kwargs.update(overrides)
    return L2SelectiveRavenCell(**kwargs).eval()


class SelectiveRavenCellTests(unittest.TestCase):
    # 1. Reachable through the existing SSM registry/factory as an LTI-cell option.
    def test_registered_as_certified_lti_cell(self):
        self.assertIn("raven", _SSM_PARAMETRIZATIONS)
        self.assertTrue(_SSM_PARAMETRIZATIONS["raven"].certified)
        self.assertIn("raven", _CERTIFIED_PARAMETRIZATIONS)

        model = DeepSSM(
            d_input=3, d_output=2, d_model=8, n_layers=1,
            param="raven", ff="LGLU2", gamma=1.0,
        ).eval()
        self.assertIsInstance(model.blocks[0].lru, L2SelectiveRavenCell)

    # 2. Output shape is (B, L, d_model).
    def test_output_shape(self):
        cell = _cell()
        z = torch.randn(4, 13, 8)
        y, _ = cell(z, mode="scan")
        self.assertEqual(y.shape, (4, 13, 8))

    # 3. Returned streaming-state shapes are (B,H,M,d_k) and (B,H,M,d_v).
    def test_state_shapes(self):
        cell = _cell()
        z = torch.randn(4, 7, 8)
        _, (s_k, s_v) = cell(z, mode="scan", return_state=False, return_last=True)
        self.assertEqual(s_k.shape, (4, cell.H, cell.M, cell.d_k))
        self.assertEqual(s_v.shape, (4, cell.H, cell.M, cell.d_v))
        # The trajectory's last index is the streaming state (DeepSSM relies on this).
        _, (sk_seq, sv_seq) = cell(z, mode="scan")
        self.assertEqual(sk_seq.shape, (4, 7, cell.H, cell.M, cell.d_k))
        self.assertTrue(torch.allclose(sk_seq[:, -1], s_k))

    # 4. loop and scan match numerically.
    def test_loop_and_scan_match(self):
        cell = _cell()
        z = torch.randn(3, 17, 8)
        y_loop, (sk_loop, sv_loop) = cell(z, mode="loop")
        y_scan, (sk_scan, sv_scan) = cell(z, mode="scan")
        self.assertTrue(torch.allclose(y_loop, y_scan, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(sk_loop, sk_scan, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(sv_loop, sv_scan, atol=1e-5, rtol=1e-4))

    # 5. Streaming one-step inference matches full-sequence loop.
    def test_streaming_matches_full_sequence(self):
        cell = _cell()
        z = torch.randn(2, 11, 8)
        y_full, _ = cell(z, mode="loop", reset_state=True)

        outs, state = [], None
        for t in range(z.shape[1]):
            y_t, state = cell(
                z[:, t:t + 1], state=state, mode="loop",
                reset_state=(t == 0), return_state=False, return_last=True,
            )
            outs.append(y_t)
        y_stream = torch.cat(outs, dim=1)
        self.assertTrue(torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-4))

    # 6. Router forced to zero => exact S_next = rho * S (pure leak, no write).
    def test_router_zero_gives_pure_rho_decay(self):
        cell = _cell()
        cell._router = lambda m: torch.zeros_like(m)  # force r_t == 0
        B = 2
        s_k0 = torch.randn(B, cell.H, cell.M, cell.d_k)
        s_v0 = torch.randn(B, cell.H, cell.M, cell.d_v)
        z = torch.randn(B, 5, 8)
        _, (sk_post, sv_post) = cell(
            z, state=(s_k0, s_v0), mode="loop", reset_state=False,
        )
        rho = cell.rho.item()
        for t in range(z.shape[1]):
            self.assertTrue(torch.allclose(sk_post[:, t], (rho ** (t + 1)) * s_k0, atol=1e-5))
            self.assertTrue(torch.allclose(sv_post[:, t], (rho ** (t + 1)) * s_v0, atol=1e-5))

    # 7. All lambda_t <= rho < 1.
    def test_lambda_bounded_by_rho(self):
        cell = _cell()
        self.assertTrue(torch.all(cell.decay_shape <= 0.0))  # a = -softplus(.) <= 0
        rho = cell.rho
        self.assertLess(rho.item(), 1.0)
        # lambda = rho * exp(a * r) with a <= 0, r >= 0  =>  lambda <= rho.
        r = torch.rand(32, cell.H, cell.M) / cell.alpha  # any non-negative router output
        lam = rho * torch.exp(cell.decay_shape[None] * r)
        self.assertTrue(torch.all(lam <= rho + 1e-6))

    # 8. Effective W_o, W_v, (D) satisfy the conservative gain budget.
    def test_effective_weights_meet_gain_budget(self):
        for use_skip in (False, True):
            with self.subTest(use_skip=use_skip):
                cell = _cell(use_skip=use_skip, gamma=1.3, gamma_skip=0.3 if use_skip else 0.0)
                c, gamma_skip_eff, rho = cell._gain_budget(torch.device("cpu"), torch.float32)
                wv = _spectral_cap(cell.W_v.weight, c)
                wo = _spectral_cap(cell.W_o.weight, c)
                sv = torch.linalg.matrix_norm(wv, ord=2).item()
                so = torch.linalg.matrix_norm(wo, ord=2).item()
                mem = so * sv / (cell.alpha * (1.0 - rho.item()))
                skip = 0.0
                if use_skip:
                    wd = _spectral_cap(cell.D.weight, gamma_skip_eff)
                    skip = torch.linalg.matrix_norm(wd, ord=2).item()
                self.assertLessEqual(mem + skip, cell.gamma.item() + 1e-4)

    # Zero input -> zero output (the property the gain certificate rests on).
    def test_zero_input_gives_zero_output(self):
        for use_skip in (False, True):
            cell = _cell(use_skip=use_skip, gamma_skip=0.3 if use_skip else 0.0)
            z = torch.zeros(2, 9, 8)
            y, _ = cell(z, mode="scan")
            self.assertEqual(torch.count_nonzero(y).item(), 0)

    # Standalone empirical zero-state gain stays below the certified gamma.
    def test_empirical_gain_below_certificate(self):
        torch.manual_seed(1)
        cell = _cell(gamma=0.6, use_skip=True, gamma_skip=0.1)
        z = torch.randn(4, 64, 8)
        y, _ = cell(z, mode="scan", reset_state=True)
        self.assertLessEqual(y.norm().item(), 0.6 * z.norm().item() + 1e-4)

    # Integrates into the certified DeepSSM stack and respects the global gamma.
    def test_deepssm_certificate_with_raven_core(self):
        torch.manual_seed(2)
        model = DeepSSM(
            d_input=3, d_output=2, d_model=8, n_layers=2,
            param="raven", ff="LGLU2", gamma=0.9, train_gamma=False,
        ).eval()
        self.assertLessEqual(model.certified_gain_bound().item(), 0.9 + 1e-5)
        u = torch.randn(2, 40, 3)
        with torch.no_grad():
            y, _ = model(u, mode="scan", reset_state=True)
        self.assertTrue(torch.isfinite(y).all())
        self.assertLessEqual(y.norm().item(), 0.9 * u.norm().item() + 1e-4)

    # learn_x0 must stay incompatible with the certified stack.
    def test_learn_x0_rejected_for_certified_stack(self):
        with self.assertRaisesRegex(ValueError, "zero-state"):
            DeepSSM(
                d_input=2, d_output=2, d_model=8, n_layers=1,
                param="raven", ff="LGLU2", gamma=1.0, learn_x0=True,
            )


if __name__ == "__main__":
    unittest.main()
