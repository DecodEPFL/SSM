import unittest

import torch

from src.neural_ssm import ContextualDeepSSM, timewise_matrix_vector_product


class ContextualSSMTests(unittest.TestCase):
    def test_timewise_matrix_vector_product_matches_manual_matmul(self):
        matrices = torch.randn(2, 5, 3, 4)
        vectors = torch.randn(2, 5, 4)

        y = timewise_matrix_vector_product(matrices, vectors)

        self.assertEqual(y.shape, (2, 5, 3))
        self.assertTrue(
            torch.allclose(y, torch.einsum("btmf,btf->btm", matrices, vectors))
        )

    def test_mixer_mode_is_bounded_and_zero_when_core_features_are_zero(self):
        torch.manual_seed(1)
        model = ContextualDeepSSM(
            d_input=4,
            d_context=3,
            d_output=2,
            context_modes=("mixer",),
            d_features=5,
            mixer_bound=0.8,
            d_model=6,
            d_state=6,
            n_layers=1,
            param="tv",
            ff="LGLU2",
            gamma=1.2,
        ).eval()
        disturbance = torch.zeros(3, 9, 4)
        context = torch.randn(3, 9, 3)

        with torch.no_grad():
            y, state, aux = model(
                disturbance,
                context,
                mode="loop",
                reset_state=True,
                return_aux=True,
            )

        mixer_norms = torch.linalg.vector_norm(aux["mixer"], ord=2, dim=(-2, -1))
        self.assertEqual(y.shape, (3, 9, 2))
        self.assertEqual(aux["features"].shape, (3, 9, 5))
        self.assertEqual(len(state), 1)
        self.assertEqual(torch.count_nonzero(y).item(), 0)
        self.assertLessEqual(mixer_norms.max().item(), 0.8 + 1e-6)
        self.assertLessEqual(model.certified_gain_bound().item(), 0.96 + 1e-6)

    def test_input_mode_finite_horizon_filters_context_inside_one_architecture(self):
        torch.manual_seed(2)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=3,
            d_output=2,
            context_modes=("input",),
            context_filter="finite_horizon",
            horizon=5,
            d_model=6,
            d_state=6,
            n_layers=1,
            param="tv",
            ff="LGLU2",
            gamma=0.9,
        ).eval()
        disturbance = torch.randn(2, 8, 2)
        context = torch.randn(2, 8, 3)

        with torch.no_grad():
            y, _, aux = model(
                disturbance,
                context,
                mode="loop",
                reset_state=True,
                return_aux=True,
            )

        self.assertEqual(y.shape, (2, 8, 2))
        self.assertEqual(aux["core_input"].shape, (2, 8, 5))
        self.assertTrue(torch.equal(aux["filtered_context"][:, :5], context[:, :5]))
        self.assertEqual(torch.count_nonzero(aux["filtered_context"][:, 5:]).item(), 0)
        self.assertLessEqual(model.certified_gain_bound().item(), 0.9 + 1e-6)
        self.assertLessEqual(
            y.norm().item(),
            model.certified_gain_bound().item() * aux["core_input"].norm().item() + 1e-4,
        )

    def test_gate_mode_uses_bounded_residual_gates_without_context_l2_assumption(self):
        torch.manual_seed(3)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=3,
            d_output=2,
            context_modes=("gate",),
            gate_per_channel=True,
            d_model=5,
            d_state=5,
            n_layers=2,
            param="tv",
            ff="LGLU2",
            gamma=1.0,
        ).eval()
        disturbance = torch.zeros(4, 7, 2)
        context = 100.0 * torch.randn(4, 7, 3)

        with torch.no_grad():
            y, _, aux = model(
                disturbance,
                context,
                mode="loop",
                reset_state=True,
                return_aux=True,
            )

        self.assertEqual(torch.count_nonzero(y).item(), 0)
        self.assertEqual(len(aux["context_gates"]), 2)
        for gates in aux["context_gates"]:
            self.assertEqual(gates["ssm"].shape, (4, 7, 5))
            self.assertTrue(torch.all((0.0 <= gates["ssm"]) & (gates["ssm"] <= 1.0)))
            self.assertTrue(torch.all((0.0 <= gates["ff"]) & (gates["ff"] <= 1.0)))
        self.assertLessEqual(model.certified_gain_bound().item(), 1.0 + 1e-6)

    def test_all_context_modes_can_be_enabled_concurrently(self):
        torch.manual_seed(4)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=2,
            d_output=3,
            context_modes=("input", "gate", "mixer"),
            context_filter="exponential",
            context_filter_decay=0.8,
            trainable_context_filter=True,
            d_features=4,
            mixer_bound=0.5,
            d_model=5,
            d_state=5,
            n_layers=2,
            param="tv",
            ff="LGLU2",
            gamma=0.7,
        ).eval()
        disturbance = torch.randn(2, 10, 2)
        context = torch.randn(2, 10, 2)

        with torch.no_grad():
            y, _, aux = model(
                disturbance,
                context,
                mode="loop",
                reset_state=True,
                return_aux=True,
            )

        self.assertEqual(y.shape, (2, 10, 3))
        self.assertEqual(aux["core_input"].shape, (2, 10, 4))
        self.assertEqual(aux["features"].shape, (2, 10, 4))
        self.assertEqual(aux["mixer"].shape, (2, 10, 3, 4))
        self.assertEqual(len(aux["context_gates"]), 2)
        self.assertTrue(any("raw_decay" in name for name, _ in model.named_parameters()))
        self.assertLessEqual(model.certified_gain_bound().item(), 0.35 + 1e-6)

    def test_input_filter_streaming_uses_absolute_time_offset(self):
        torch.manual_seed(5)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=1,
            d_output=2,
            context_modes=("input", "gate", "mixer"),
            context_filter="exponential",
            context_filter_decay=0.8,
            d_features=3,
            mixer_bound=1.3,
            d_model=5,
            d_state=5,
            n_layers=2,
            param="lru",
            ff="GLU",
            gamma=None,
        ).eval()
        disturbance = torch.randn(2, 11, 2)
        context = torch.randn(2, 11, 1)

        with torch.no_grad():
            whole, _ = model(disturbance, context, mode="loop", reset_state=True)
            first, state = model(
                disturbance[:, :6],
                context[:, :6],
                mode="loop",
                reset_state=True,
            )
            second, _ = model(
                disturbance[:, 6:],
                context[:, 6:],
                state=state,
                mode="loop",
                reset_state=False,
                time_offset=6,
            )

        self.assertTrue(
            torch.allclose(whole, torch.cat((first, second), dim=1), atol=1e-6, rtol=1e-5)
        )

    def test_certified_contextual_core_rejects_uncertified_standard_choices(self):
        with self.assertRaisesRegex(ValueError, "does not provide"):
            ContextualDeepSSM(
                d_input=2,
                d_context=1,
                d_output=2,
                context_modes=("input",),
                gamma=1.0,
                param="lru",
                ff="LGLU2",
            )
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            ContextualDeepSSM(
                d_input=2,
                d_context=1,
                d_output=2,
                context_modes=("gate",),
                gamma=1.0,
                param="tv",
                ff="GLU",
            )

    def test_input_mode_difference_is_lossless_and_vanishes_on_constant_context(self):
        torch.manual_seed(6)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=2,
            d_output=2,
            context_modes=("input",),
            context_filter="difference",
            d_model=6,
            d_state=6,
            n_layers=1,
            param="tv",
            ff="LGLU2",
            gamma=0.9,
        ).eval()
        disturbance = torch.randn(2, 8, 2)
        context = torch.zeros(2, 8, 2)
        context[:, 4:] = 1.0  # piecewise-constant, one switch at t=4

        with torch.no_grad():
            y, _, aux = model(
                disturbance, context, mode="loop", reset_state=True, return_aux=True
            )

        filtered = aux["filtered_context"]
        nonzero_times = torch.nonzero(filtered.abs().sum(dim=(0, 2))).flatten().tolist()
        self.assertEqual(y.shape, (2, 8, 2))
        self.assertEqual(nonzero_times, [4])  # increment fires only at the switch
        self.assertLessEqual(model.certified_gain_bound().item(), 0.9 + 1e-6)
        # a difference window is not a fixed l2 window: no finite a-priori offset.
        self.assertEqual(model.context_offset_bound(1.0), float("inf"))

    def test_input_mode_taper_window_is_smooth_and_finite_support(self):
        model = ContextualDeepSSM(
            d_input=2,
            d_context=1,
            d_output=2,
            context_modes=("input",),
            context_filter="taper",
            horizon=8,
            context_filter_ramp=4,
            d_model=6,
            d_state=6,
            n_layers=1,
            param="tv",
            ff="LGLU2",
            gamma=0.9,
        ).eval()
        disturbance = torch.zeros(1, 12, 2)
        context = torch.ones(1, 12, 1)

        with torch.no_grad():
            _, _, aux = model(
                disturbance, context, mode="loop", reset_state=True, return_aux=True
            )

        window = aux["filtered_context"][0, :, 0]
        self.assertTrue(torch.allclose(window[:4], torch.ones(4)))  # flat region
        self.assertEqual(torch.count_nonzero(window[8:]).item(), 0)  # zero past horizon
        roll = window[4:8]
        self.assertTrue(torch.all(roll[1:] <= roll[:-1] + 1e-6))  # monotone roll-off
        self.assertTrue(torch.all((0.0 <= roll) & (roll <= 1.0 + 1e-6)))

    def test_context_offset_bound_dominates_context_only_drive(self):
        torch.manual_seed(7)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=1,
            d_output=2,
            context_modes=("input",),
            context_filter="finite_horizon",
            horizon=6,
            d_model=6,
            d_state=6,
            n_layers=1,
            param="tv",
            ff="LGLU2",
            gamma=0.9,
        ).eval()
        disturbance = torch.zeros(3, 10, 2)  # w = 0 => output is pure context drive
        context = 2.0 * torch.rand(3, 10, 1) - 1.0  # |z_t| <= 1
        amplitude = float(context.abs().amax().item())

        with torch.no_grad():
            y, _ = model(disturbance, context, mode="loop", reset_state=True)

        self.assertLessEqual(y.norm().item(), model.context_offset_bound(amplitude) + 1e-4)

    def test_select_mode_injects_context_into_selective_cells(self):
        torch.manual_seed(8)
        model = ContextualDeepSSM(
            d_input=2,
            d_context=3,
            d_output=2,
            context_modes=("select",),
            d_model=8,
            d_state=8,
            n_layers=2,
            param="tv",
            ff="LGLU2",
            gamma=1.0,
        ).eval()
        u = torch.randn(2, 16, 2)
        z = torch.randn(2, 16, 3)

        with torch.no_grad():
            y, _ = model(u, z, mode="loop")
            y0, _ = model(u, torch.zeros_like(z), mode="loop")

        self.assertEqual(y.shape, (2, 16, 2))
        self.assertTrue(
            all(getattr(b.lru, "supports_select_context", False) for b in model.core.blocks)
        )
        # context reshapes the dynamics ...
        self.assertGreater((y - y0).abs().max().item(), 1e-6)
        # ... but the certified gain is unchanged: ||u_out|| <= gamma * ||u_in||.
        gain = model.certified_gain_bound().item()
        self.assertLessEqual(gain, 1.0 + 1e-6)
        self.assertLessEqual(y.norm().item(), gain * u.norm().item() + 1e-3)

    def test_select_mode_requires_a_selective_core(self):
        with self.assertRaisesRegex(ValueError, "selective core"):
            ContextualDeepSSM(
                d_input=2,
                d_context=3,
                d_output=2,
                context_modes=("select",),
                param="l2ru",
                ff="LGLU2",
                gamma=1.0,
            )


if __name__ == "__main__":
    unittest.main()
