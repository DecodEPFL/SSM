"""Storage-function and dissipation tests for the certified stack.

``tests/test_deepssm_gain_certificate.py`` checks that the *reported* bound is
composed correctly and that one random zero-state trajectory respects it. That
is necessary but not sufficient: it passes for a forward map that does not obey
the certificate at all, because a single random input is not a worst case and a
zero-state norm ratio is not the dissipation inequality.

These tests check the inequality the certificate actually asserts,

    V_Q(s_{t+1}) - V_Q(s_t) + ||f_t||^2 <= Gamma^2 ||e_t||^2,

step by step and over a horizon from a nonzero initial state, plus the cases the
theory note singles out: arbitrary context switching, the post-state output path,
and parameter freezing at deployment.
"""

import unittest

import torch

from neural_ssm.ssm.contextual import ContextualDeepSSM
from neural_ssm.ssm.layers import DeepSSM
from neural_ssm.ssm.selective_cells import RobustMambaDiagLTI


def build(**overrides):
    kwargs = dict(
        d_model=8,
        d_state=6,
        n_layers=3,
        gamma=0.9,
        param="tv",
        ff="LGLU2",
        dropout=0.0,
        learn_x0=False,
    )
    kwargs.update(overrides)
    model = DeepSSM(3, 2, **kwargs)
    model.eval()
    return model


def energy(x):
    """Per-batch squared l2 norm of a (B, T, D) signal."""
    return x.reshape(x.shape[0], -1).pow(2).sum(-1)


class StorageFunctionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_storage_weights_match_the_cascade(self):
        """w_i = (||D|| sigma)^2 * prod_{j>i} k_j^2 * l_i^2 * c_i."""
        model = build()
        diag = model.gain_diagnostics()
        blocks = diag["blocks"]
        out = (diag["decoder_norm"] * diag["smooth_scale"]) ** 2

        for i, row in enumerate(blocks):
            suffix = 1.0
            for j in range(i + 1, len(blocks)):
                suffix *= blocks[j]["block_factor"] ** 2
            expected = out * suffix * row["ff_factor"] ** 2 * row["c_ssm"]
            self.assertAlmostEqual(
                row["storage_weight"] / expected, 1.0, places=4,
                msg=f"block {i}: {row['storage_weight']} vs {expected}",
            )

    def test_c_ssm_is_the_young_split_weight(self):
        """c = alpha_eff*(alpha_eff + 1/gamma), with alpha_eff after dropout."""
        model = build()
        for row in model.gain_diagnostics()["blocks"]:
            alpha_eff = row["alpha_ssm"] * row["ssm_drop_factor"]
            expected = alpha_eff * (alpha_eff + 1.0 / row["core_gamma"])
            self.assertAlmostEqual(row["c_ssm"], expected, places=5)
            # and the block factor is l*(1 + alpha_eff*gamma), as the note has it
            self.assertAlmostEqual(
                row["block_factor"],
                row["ff_factor"] * (1.0 + alpha_eff * row["core_gamma"]),
                places=5,
            )

    def test_per_step_dissipation_inequality(self):
        model = build()
        bound = float(model.certified_gain_bound())
        u = torch.randn(4, 30, 3) * 2.0

        state = None
        with torch.no_grad():
            for t in range(u.shape[1]):
                before = model.storage_value(state)
                y, state = model(
                    u[:, t:t + 1], state=state, mode="loop", reset_state=(t == 0)
                )
                after = model.storage_value(state)
                lhs = (after - before) + energy(y)
                rhs = bound ** 2 * energy(u[:, t:t + 1])
                self.assertLessEqual(
                    float((lhs - rhs).max()), 1e-4,
                    msg=f"dissipation inequality violated at step {t}",
                )

    def test_finite_horizon_energy_from_nonzero_initial_state(self):
        model = build()
        bound = float(model.certified_gain_bound())
        u = torch.randn(4, 50, 3)
        s0 = [torch.randn(4, 6) * 3.0 for _ in model.blocks]

        with torch.no_grad():
            v0 = model.storage_value(s0)
            y, sT = model(u, state=s0, mode="loop", reset_state=False)
            vT = model.storage_value(sT)

        lhs = energy(y) + vT
        rhs = bound ** 2 * energy(u) + v0
        self.assertLessEqual(float((lhs - rhs).max()), 1e-4)

    def test_storage_term_is_necessary(self):
        """A large initial state breaks the zero-state bound but not the storage one.

        Without this the horizon test above would prove nothing: it would pass
        with V_Q identically zero.
        """
        model = build()
        bound = float(model.certified_gain_bound())
        u = torch.randn(4, 60, 3)
        s0 = [torch.randn(4, 6) * 200.0 for _ in model.blocks]

        with torch.no_grad():
            v0 = model.storage_value(s0)
            y, sT = model(u, state=s0, mode="loop", reset_state=False)
            vT = model.storage_value(sT)

        naive = energy(y) - bound ** 2 * energy(u)
        self.assertGreater(
            float(naive.max()), 0.0,
            "initial state too small to violate the zero-state bound; the "
            "storage test would be vacuous",
        )
        with_storage = energy(y) + vT - bound ** 2 * energy(u) - v0
        self.assertLessEqual(float(with_storage.max()), 1e-4)

    def test_storage_is_zero_at_zero_state_and_positive_otherwise(self):
        model = build()
        zeros = [torch.zeros(2, 6) for _ in model.blocks]
        with torch.no_grad():
            self.assertEqual(float(model.storage_value(zeros).max()), 0.0)
            self.assertEqual(float(model.storage_value(None)), 0.0)
            nonzero = model.storage_value([torch.ones(2, 6) for _ in model.blocks])
        self.assertGreater(float(nonzero.min()), 0.0)
        self.assertTrue(torch.isfinite(nonzero).all())

    def test_initial_storage_bound_is_the_square_root(self):
        model = build()
        s0 = [torch.randn(3, 6) for _ in model.blocks]
        with torch.no_grad():
            v = model.storage_value(s0)
            b = model.initial_storage_bound(s0)
        torch.testing.assert_close(b, v.sqrt())

    def test_storage_rejects_a_mismatched_state_length(self):
        model = build()
        with self.assertRaises(ValueError):
            model.storage_value([torch.zeros(2, 6)])


class AdversarialContextTests(unittest.TestCase):
    """The certificate must survive context chosen to break it."""

    def setUp(self):
        torch.manual_seed(0)

    def _contextual(self, **overrides):
        kwargs = dict(
            context_modes=("select",),
            d_model=8,
            d_state=6,
            n_layers=2,
            gamma=0.8,
            param="tv",
            ff="LGLU2",
            dropout=0.0,
            learn_x0=False,
        )
        kwargs.update(overrides)
        model = ContextualDeepSSM(3, 4, 2, **kwargs)
        model.eval()
        return model

    def test_bound_holds_under_context_resampled_every_step(self):
        model = self._contextual()
        bound = float(model.certified_gain_bound())
        u = torch.randn(4, 40, 3) * 2.0

        state = None
        with torch.no_grad():
            for t in range(u.shape[1]):
                # a fresh, unrelated, deliberately large context at every step
                ctx = torch.randn(4, 1, 4) * 10.0
                before = model.storage_value(state)
                y, state = model(
                    u[:, t:t + 1], ctx, state=state,
                    mode="loop", reset_state=(t == 0),
                )
                after = model.storage_value(state)
                lhs = (after - before) + energy(y)
                rhs = bound ** 2 * energy(u[:, t:t + 1])
                self.assertLessEqual(float((lhs - rhs).max()), 1e-4, msg=f"step {t}")

    def test_bound_holds_under_adversarial_context_switches(self):
        """Square-wave context switching between extremes at every step."""
        model = self._contextual()
        bound = float(model.certified_gain_bound())
        u = torch.randn(4, 40, 3) * 2.0
        hi = torch.full((4, 1, 4), 50.0)
        lo = torch.full((4, 1, 4), -50.0)

        state = None
        with torch.no_grad():
            for t in range(u.shape[1]):
                y, state = model(
                    u[:, t:t + 1], hi if t % 2 else lo, state=state,
                    mode="loop", reset_state=(t == 0),
                )
                self.assertTrue(torch.isfinite(y).all())
            y_all, _ = model(
                u,
                torch.where(
                    (torch.arange(40) % 2).reshape(1, -1, 1).bool(), hi[:, :1], lo[:, :1]
                ).expand(4, 40, 4),
                mode="loop",
                reset_state=True,
            )
        self.assertLessEqual(
            float(energy(y_all).max()), bound ** 2 * float(energy(u).max()) + 1e-4
        )

    def test_zero_input_and_zero_state_give_zero_output_for_any_context(self):
        """The property the theory note asks for before persistent memory lands."""
        model = self._contextual()
        u = torch.zeros(4, 12, 3)
        with torch.no_grad():
            for scale in (0.0, 1.0, 100.0):
                y, _ = model(u, torch.randn(4, 12, 4) * scale, mode="loop", reset_state=True)
                self.assertEqual(float(y.abs().max()), 0.0)


class DeploymentFreezeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fingerprint_is_stable_and_detects_gain_changes(self):
        model = build()
        first = model.certificate_fingerprint()
        self.assertEqual(first, model.certificate_fingerprint())

        with torch.no_grad():
            model.blocks[0].ssm_res_logit += 0.5
        self.assertNotEqual(first, model.certificate_fingerprint())

    def test_fingerprint_detects_a_decoder_change(self):
        model = build()
        first = model.certificate_fingerprint()
        with torch.no_grad():
            model.decoder_w += 0.01
        self.assertNotEqual(first, model.certificate_fingerprint())

    def test_freeze_stops_gradients_on_certificate_parameters_only(self):
        model = ContextualDeepSSM(
            3, 4, 2, context_modes=("select", "mixer"),
            d_model=8, d_state=6, n_layers=2, gamma=0.8,
            param="tv", ff="LGLU2", dropout=0.0, learn_x0=False,
        )
        model.freeze_certificate()

        for block in model.core.blocks:
            self.assertFalse(block.ssm_res_logit.requires_grad)
            self.assertFalse(block.ff_res_logit.requires_grad)
        self.assertFalse(model.core.encoder_w.requires_grad)
        self.assertFalse(model.core.decoder_w.requires_grad)

        # The context ports stay trainable: the per-step renormalization holds
        # for any value they take, so they may keep adapting online.
        self.assertTrue(any(p.requires_grad for p in model.mixer.parameters()))

    def test_fingerprint_survives_a_context_port_update(self):
        """Updating the context pathway must not move the certificate."""
        model = ContextualDeepSSM(
            3, 4, 2, context_modes=("select",),
            d_model=8, d_state=6, n_layers=2, gamma=0.8,
            param="tv", ff="LGLU2", dropout=0.0, learn_x0=False,
        )
        before = model.freeze_certificate()
        bound_before = float(model.certified_gain_bound())

        with torch.no_grad():
            for block in model.core.blocks:
                for param in block.lru.param_net.parameters():
                    param += torch.randn_like(param) * 0.5

        self.assertEqual(before, model.certificate_fingerprint())
        self.assertAlmostEqual(bound_before, float(model.certified_gain_bound()), places=6)

    def test_bound_still_holds_after_a_context_port_update(self):
        model = ContextualDeepSSM(
            3, 4, 2, context_modes=("select",),
            d_model=8, d_state=6, n_layers=2, gamma=0.8,
            param="tv", ff="LGLU2", dropout=0.0, learn_x0=False,
        )
        model.eval()
        model.freeze_certificate()
        with torch.no_grad():
            for block in model.core.blocks:
                for param in block.lru.param_net.parameters():
                    param += torch.randn_like(param) * 1.0

            u = torch.randn(4, 40, 3) * 2.0
            y, _ = model(u, torch.randn(4, 40, 4) * 5.0, mode="loop", reset_state=True)
        bound = float(model.certified_gain_bound())
        self.assertLessEqual(float(energy(y).max()), bound ** 2 * float(energy(u).max()) + 1e-4)


class PostStateOutputTests(unittest.TestCase):
    def test_post_state_output_withdraws_the_gain_contract(self):
        """Normalizing [[a,b],[c,d]] does not bound [[a,b],[c*a,c*b+d]]."""
        with self.assertWarns(UserWarning):
            cell = RobustMambaDiagLTI(
                d_model=4, d_state=4, gamma=0.5, output_uses_post_state=True
            )
        self.assertTrue(torch.isinf(cell.gain_bound()).all())
        self.assertFalse(torch.isinf(cell.gamma).all())

    def test_pre_state_output_keeps_the_gain_contract(self):
        cell = RobustMambaDiagLTI(d_model=4, d_state=4, gamma=0.5)
        torch.testing.assert_close(cell.gain_bound(), cell.gamma)

    def test_a_stack_never_advertises_a_bound_a_post_state_cell_breaks(self):
        """A cell with no gain contract must not leave a plausible bound standing.

        The withdrawn contract makes the block factor infinite, the decoder
        attenuation collapses to zero, and the stack reports a bound of zero for
        an output that really is identically zero -- a safe degradation rather
        than a number the forward map violates. What matters is the invariant, so
        assert that rather than the mechanism: the reported bound must hold, and
        the uncertified condition must remain visible.
        """
        model = build(n_layers=2)
        with self.assertWarns(UserWarning):
            replacement = RobustMambaDiagLTI(
                d_model=model.config.d_model,
                d_state=model.config.d_state,
                d_out=model.config.d_model,
                gamma=0.5,
                output_uses_post_state=True,
            )
        model.blocks[0].lru = replacement

        u = torch.randn(2, 20, 3) * 2.0
        with torch.no_grad():
            y, _ = model(u, mode="loop", reset_state=True)
        bound = float(model.certified_gain_bound())
        self.assertLessEqual(
            float(energy(y).max()), bound ** 2 * float(energy(u).max()) + 1e-6,
            "the reported bound does not hold for the realized forward map",
        )
        # and the loss of the contract stays visible to a caller
        self.assertFalse(torch.isfinite(model.conservative_gamma_product()).item())


if __name__ == "__main__":
    unittest.main()
