import math
import unittest

import torch

from src.neural_ssm.ssm import DeepSSM, SSMConfig
from src.neural_ssm.ssm.layers import _CERTIFIED_PARAMETRIZATIONS, _SSM_PARAMETRIZATIONS
from src.neural_ssm.static_layers.generic_layers import LayerConfig
from src.neural_ssm.static_layers.lipschitz_mlps import TLIP


def _certified_model(gamma: float = 1.5) -> DeepSSM:
    torch.manual_seed(7)
    model = DeepSSM(
        d_input=2,
        d_output=2,
        d_model=4,
        d_state=4,
        n_layers=2,
        param="tv",
        ff="LGLU2",
        gamma=gamma,
        scale=0.8,
    )
    return model.eval()


class DeepSSMGainCertificateTests(unittest.TestCase):
    def test_ssm_config_does_not_enable_a_certificate_by_default(self):
        self.assertIsNone(SSMConfig().gamma)

    def test_ssm_parametrization_registry_is_the_certification_source(self):
        certified_from_registry = {
            name for name, spec in _SSM_PARAMETRIZATIONS.items() if spec.certified
        }
        self.assertEqual(certified_from_registry, set(_CERTIFIED_PARAMETRIZATIONS))
        self.assertFalse(_SSM_PARAMETRIZATIONS["lru"].certified)

        with self.assertRaisesRegex(ValueError, "Unknown SSM parametrization"):
            DeepSSM(
                d_input=1,
                d_output=1,
                d_model=4,
                d_state=4,
                n_layers=1,
                param="missing",
                gamma=None,
            )

    def test_certified_bound_never_exceeds_prescribed_or_smaller_override(self):
        model = _certified_model(gamma=1.5)

        self.assertLessEqual(model.certified_gain_bound().item(), 1.5 + 1e-6)
        self.assertLessEqual(model.certified_gain_bound(gamma=0.4).item(), 0.4 + 1e-6)
        self.assertLessEqual(model.certified_gain_bound(gamma=20.0).item(), 1.5 + 1e-6)

        model.gamma_t.fill_(20.0)
        self.assertLessEqual(model.certified_gain_bound().item(), 1.5 + 1e-6)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            model.certified_gain_bound(gamma=-0.4)

    def test_encoder_and_decoder_are_capped_without_amplifying_small_weights(self):
        model = _certified_model()

        with torch.no_grad():
            model.encoder_w.mul_(0.1)
            model.decoder_w.mul_(0.2)

        encoder_eff = model._spectrally_capped_weight(model.encoder_w)
        decoder_eff = model._spectrally_capped_weight(model.decoder_w)
        self.assertTrue(torch.allclose(encoder_eff, model.encoder_w))
        self.assertTrue(torch.allclose(decoder_eff, model.decoder_w))

        with torch.no_grad():
            model.encoder_w.mul_(100.0)
            model.decoder_w.mul_(100.0)

        encoder_eff = model._spectrally_capped_weight(model.encoder_w)
        decoder_eff = model._spectrally_capped_weight(model.decoder_w)
        self.assertLessEqual(torch.linalg.matrix_norm(encoder_eff, ord=2).item(), 1.0 + 1e-5)
        self.assertLessEqual(torch.linalg.matrix_norm(decoder_eff, ord=2).item(), 1.0 + 1e-5)

    def test_invalid_certificate_configurations_fail_early(self):
        cases = [
            ({"param": "lru", "ff": "LGLU2", "gamma": 1.0}, "does not provide"),
            ({"param": "tv", "ff": "GLU", "gamma": 1.0}, "cannot be used"),
            ({"param": "tv", "ff": "LGLU", "gamma": 1.0}, "not globally Lipschitz"),
            (
                {"param": "tv", "ff": "LGLU2", "gamma": 1.0, "learn_x0": True},
                "zero-state",
            ),
            ({"param": "tv", "ff": "LGLU2", "gamma": 0.0}, "positive"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    DeepSSM(
                        d_input=1,
                        d_output=1,
                        d_model=4,
                        d_state=4,
                        n_layers=1,
                        **kwargs,
                    )

    def test_smooth_scale_is_stable_and_preserves_the_upper_bound(self):
        cases = [
            (1e-30, -20.0),
            (0.4, 0.0),
            (1.0, 0.0),
            (2.0, 100.0),
        ]
        for gamma, log_product in cases:
            with self.subTest(gamma=gamma, log_product=log_product):
                gamma_t = torch.tensor(gamma, dtype=torch.float64)
                log_gamma_prod = torch.tensor(log_product, dtype=torch.float64)
                scale = DeepSSM._smooth_capped_scale_from_logs(
                    gamma_t=gamma_t,
                    log_gamma_prod=log_gamma_prod,
                    temperature=0.05,
                )
                self.assertTrue(torch.isfinite(scale))
                self.assertGreaterEqual(scale.item(), 0.0)
                self.assertLessEqual(scale.item(), 1.0)
                self.assertLessEqual(
                    (log_gamma_prod + torch.log(scale)).item(),
                    torch.log(gamma_t).item() + 1e-10,
                )

    def test_deep_stack_certified_bound_respects_gamma_in_float32(self):
        # Regression: with a deep stack and large per-block gains the decoder
        # attenuation underflows to a float32 subnormal. The reported certificate
        # must be composed in log space so it never re-inflates above gamma.
        torch.manual_seed(0)
        model = DeepSSM(
            d_input=2,
            d_output=2,
            d_model=4,
            d_state=4,
            n_layers=16,
            param="tv",
            ff="LGLU2",
            gamma=1.0,
        ).eval()
        with torch.no_grad():
            for block in model.blocks:
                block.lru.log_gamma.fill_(4.0)
                block.ssm_res_logit.fill_(20.0)
                block.ff_res_logit.fill_(20.0)
                block.ff.raw_lip.fill_(6.0)

        bound = model.certified_gain_bound().item()
        self.assertTrue(math.isfinite(bound))
        self.assertLessEqual(bound, 1.0 + 1e-5)
        # gain_diagnostics() must agree with certified_gain_bound().
        self.assertAlmostEqual(
            model.gain_diagnostics()["certified_gain_bound"], bound, places=5
        )

        # The realized zero-state gain is also (far) below gamma.
        u = torch.randn(2, 48, 2)
        with torch.no_grad():
            y, _ = model(u, mode="loop", reset_state=True)
        self.assertLessEqual(y.norm().item(), 1.0 * u.norm().item() + 1e-5)

    def test_empirical_zero_state_gain_is_below_the_certificate(self):
        model = _certified_model(gamma=0.7)
        u = torch.randn(3, 80, 2)

        with torch.no_grad():
            y, _ = model(u, mode="loop", reset_state=True)

        self.assertLessEqual(y.norm().item(), 0.7 * u.norm().item() + 1e-5)

    def test_separate_residual_branches_use_the_product_certificate(self):
        model = _certified_model(gamma=1.2)
        block = model.blocks[0]
        terms = block.gain_terms(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )

        expected_ssm = 1.0 + block.ssm_scale * terms["gamma"]
        expected_ff = 1.0 + block.ff_scale * terms["ff_lip"]
        self.assertTrue(torch.allclose(terms["ssm_factor"], expected_ssm))
        self.assertTrue(torch.allclose(terms["ff_factor"], expected_ff))
        self.assertTrue(
            torch.allclose(
                terms["block_factor"],
                expected_ssm * expected_ff,
            )
        )
        self.assertIsNot(block.ssm_res_logit, block.ff_res_logit)

        block.ssm_dropout.p = 0.2
        block.ff_dropout.p = 0.5
        train_terms = block.gain_terms(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
            training=True,
        )
        self.assertAlmostEqual(train_terms["ssm_drop_factor"].item(), 1.25)
        self.assertAlmostEqual(train_terms["ff_drop_factor"].item(), 2.0)

    def test_scalar_gates_are_the_default_and_per_channel_gates_are_vectors(self):
        scalar = _certified_model(gamma=1.5)
        for block in scalar.blocks:
            self.assertEqual(block.ssm_res_logit.shape, torch.Size([]))
            self.assertEqual(block.ff_res_logit.shape, torch.Size([]))

        torch.manual_seed(7)
        model = DeepSSM(
            d_input=2,
            d_output=2,
            d_model=4,
            d_state=4,
            n_layers=2,
            param="tv",
            ff="LGLU2",
            gamma=1.5,
            scale=0.8,
            per_channel_gates=True,
        ).eval()
        for block in model.blocks:
            self.assertEqual(block.ssm_res_logit.shape, torch.Size([4]))
            self.assertEqual(block.ff_res_logit.shape, torch.Size([4]))
            self.assertEqual(block.ssm_scale.shape, torch.Size([4]))

    def test_per_channel_gates_stay_certified_via_worst_channel(self):
        torch.manual_seed(0)
        model = DeepSSM(
            d_input=2,
            d_output=2,
            d_model=4,
            d_state=4,
            n_layers=3,
            param="tv",
            ff="LGLU2",
            gamma=1.3,
            scale=0.8,
            per_channel_gates=True,
        ).eval()
        # Spread the gates widely across channels so the worst-channel reduction in
        # the certificate (max over channels) is actually exercised.
        with torch.no_grad():
            for block in model.blocks:
                block.ssm_res_logit.copy_(torch.tensor([-5.0, 0.0, 3.0, 6.0]))
                block.ff_res_logit.copy_(torch.tensor([6.0, -2.0, 1.0, -4.0]))

        terms = model.blocks[0].gain_terms(
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        # The certificate factor uses the largest gate, not the mean.
        self.assertAlmostEqual(
            terms["alpha_ssm"].item(), torch.sigmoid(torch.tensor(6.0)).item(), places=6
        )
        self.assertAlmostEqual(
            terms["alpha_ff"].item(), torch.sigmoid(torch.tensor(6.0)).item(), places=6
        )

        bound = model.certified_gain_bound().item()
        self.assertTrue(math.isfinite(bound))
        self.assertLessEqual(bound, 1.3 + 1e-5)

        # The realized zero-state gain stays below the certificate.
        u = torch.randn(3, 64, 2)
        with torch.no_grad():
            y, _ = model(u, mode="loop", reset_state=True)
        self.assertLessEqual(y.norm().item(), 1.3 * u.norm().item() + 1e-5)

    def test_scalar_gate_checkpoint_expands_into_per_channel_model(self):
        kwargs = dict(
            d_input=2,
            d_output=2,
            d_model=4,
            d_state=4,
            n_layers=2,
            param="tv",
            ff="LGLU2",
            gamma=1.5,
        )
        scalar = DeepSSM(**kwargs, per_channel_gates=False).eval()
        with torch.no_grad():
            for i, block in enumerate(scalar.blocks):
                block.ssm_res_logit.fill_(-0.7 + i)
                block.ff_res_logit.fill_(0.2 + i)

        vector = DeepSSM(**kwargs, per_channel_gates=True).eval()
        vector.load_state_dict(scalar.state_dict(), strict=True)

        for source, target in zip(scalar.blocks, vector.blocks):
            self.assertTrue(torch.equal(
                target.ssm_res_logit,
                source.ssm_res_logit.expand_as(target.ssm_res_logit),
            ))
            self.assertTrue(torch.equal(
                target.ff_res_logit,
                source.ff_res_logit.expand_as(target.ff_res_logit),
            ))

        u = torch.randn(2, 20, 2)
        with torch.no_grad():
            y_scalar, _ = scalar(u, mode="loop", reset_state=True)
            y_vector, _ = vector(u, mode="loop", reset_state=True)
        self.assertTrue(torch.allclose(y_scalar, y_vector, atol=1e-6, rtol=1e-6))

    def test_chunked_stateful_forward_matches_one_shot_forward(self):
        torch.manual_seed(11)
        model = DeepSSM(
            d_input=2,
            d_output=2,
            d_model=4,
            d_state=4,
            n_layers=2,
            param="lru",
            ff="GLU",
            gamma=None,
        ).eval()
        u = torch.randn(2, 19, 2)

        with torch.no_grad():
            whole, _ = model(u, mode="loop", reset_state=True)
            first, state = model(u[:, :7], mode="loop", reset_state=True)
            second, _ = model(
                u[:, 7:],
                state=state,
                mode="loop",
                reset_state=False,
            )

        self.assertTrue(
            torch.allclose(whole, torch.cat((first, second), dim=1), atol=1e-6, rtol=1e-5)
        )

    def test_legacy_single_residual_gate_checkpoint_is_migrated(self):
        model = _certified_model()
        legacy_state = model.state_dict()
        expected_logits = []
        for index in range(len(model.blocks)):
            prefix = f"blocks.{index}."
            ssm_logit = legacy_state.pop(prefix + "ssm_res_logit")
            legacy_state.pop(prefix + "ff_res_logit")
            legacy_state[prefix + "res_logit"] = ssm_logit
            expected_logits.append(ssm_logit)

        restored = _certified_model()
        restored.load_state_dict(legacy_state, strict=True)
        for block, expected in zip(restored.blocks, expected_logits):
            self.assertTrue(torch.equal(block.ssm_res_logit, expected))
            self.assertTrue(torch.equal(block.ff_res_logit, expected))

    def test_unconstrained_model_reports_no_finite_conservative_bound(self):
        model = DeepSSM(
            d_input=1,
            d_output=1,
            d_model=4,
            d_state=4,
            n_layers=1,
            param="tv",
            ff="GLU",
            gamma=None,
        )
        self.assertTrue(torch.isinf(model.conservative_gamma_product()))

    def test_zero_block_linear_model_can_still_be_certified(self):
        model = DeepSSM(
            d_input=2,
            d_output=2,
            d_model=4,
            d_state=4,
            n_layers=0,
            param="lru",
            ff="GLU",
            gamma=0.5,
            learn_x0=True,
        ).eval()
        self.assertLessEqual(model.certified_gain_bound().item(), 0.5 + 1e-6)

    def test_all_admitted_recurrent_cores_expose_a_finite_certificate(self):
        for param in ("l2ru", "zak", "l2n", "l2nt", "tv", "tvc"):
            with self.subTest(param=param):
                model = DeepSSM(
                    d_input=2,
                    d_output=2,
                    d_model=4,
                    d_state=4,
                    n_layers=1,
                    d_hidden=4,
                    param=param,
                    ff="LGLU2",
                    gamma=1.0,
                    train_gamma=False,
                ).eval()
                with torch.no_grad():
                    y, _ = model(torch.randn(1, 6, 2), mode="loop")

                self.assertTrue(torch.isfinite(y).all())
                self.assertLessEqual(model.certified_gain_bound().item(), 1.0 + 1e-6)

    def test_tlip_accepts_sequence_inputs_and_is_zero_at_zero(self):
        config = LayerConfig(
            d_input=4,
            d_hidden=4,
            d_output=4,
            n_layers=1,
            lip=0.8,
        )
        layer = TLIP(config).eval()
        x = torch.zeros(2, 5, 4)

        with torch.no_grad():
            y = layer(x)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(torch.count_nonzero(y).item(), 0)


if __name__ == "__main__":
    unittest.main()
