import unittest

import torch

from Test_files.Benchmark import ModelConfig, build_model_from_config
from src.neural_ssm.ssm import MultiHeadRavenRSM


class MultiHeadRavenRSMTests(unittest.TestCase):
    def test_forward_shapes_and_state_shapes(self):
        torch.manual_seed(3)
        layer = MultiHeadRavenRSM(
            d_model=6,
            num_heads=2,
            num_slots=5,
            d_k=3,
            d_v=4,
            top_k=2,
        )
        z = torch.randn(4, 7, 6)

        y, state = layer(z)
        S_k, S_v = state

        self.assertEqual(y.shape, z.shape)
        self.assertEqual(S_k.shape, (4, 2, 5, 3))
        self.assertEqual(S_v.shape, (4, 2, 5, 4))

    def test_inactive_slots_are_preserved_exactly(self):
        torch.manual_seed(5)
        layer = MultiHeadRavenRSM(
            d_model=4,
            num_heads=1,
            num_slots=4,
            d_k=2,
            d_v=3,
            top_k=2,
            residual=False,
            norm=False,
        )
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.bias.copy_(torch.tensor([10.0, 9.0, -10.0, -11.0]))

        S_k0 = torch.randn(2, 1, 4, 2)
        S_v0 = torch.randn(2, 1, 4, 3)
        _, (S_k1, S_v1) = layer(torch.randn(2, 1, 4), state=(S_k0, S_v0))

        self.assertTrue(torch.equal(S_k1[:, :, 2:], S_k0[:, :, 2:]))
        self.assertTrue(torch.equal(S_v1[:, :, 2:], S_v0[:, :, 2:]))

    def test_chunked_streaming_matches_full_sequence(self):
        torch.manual_seed(7)
        layer = MultiHeadRavenRSM(
            d_model=5,
            num_heads=1,
            num_slots=3,
            d_k=2,
            d_v=2,
            top_k=1,
            residual=True,
            norm=True,
        ).eval()
        z = torch.randn(2, 9, 5)

        with torch.no_grad():
            full_y, full_state = layer(z, reset_state=True)
            first_y, state = layer(z[:, :4], reset_state=True)
            second_y, chunk_state = layer(
                z[:, 4:],
                state=state,
                reset_state=False,
            )

        self.assertTrue(torch.allclose(full_y, torch.cat((first_y, second_y), dim=1)))
        self.assertTrue(torch.allclose(full_state[0], chunk_state[0]))
        self.assertTrue(torch.allclose(full_state[1], chunk_state[1]))

    def test_sparse_router_gradients_only_reach_selected_entries(self):
        torch.manual_seed(11)
        layer = MultiHeadRavenRSM(
            d_model=4,
            num_heads=1,
            num_slots=4,
            d_k=2,
            d_v=2,
            top_k=2,
            residual=False,
            norm=False,
        )
        with torch.no_grad():
            layer.router.weight.zero_()
            layer.router.bias.copy_(torch.tensor([8.0, 7.0, -8.0, -9.0]))

        y, _ = layer(torch.randn(1, 1, 4), detach_state=False)
        y.square().sum().backward()

        self.assertEqual(layer.router.bias.grad[2].item(), 0.0)
        self.assertEqual(layer.router.bias.grad[3].item(), 0.0)

    def test_benchmark_builder_can_construct_raven_wrapper(self):
        config = ModelConfig(
            model_type="raven",
            n_u=2,
            n_y=3,
            d_model=6,
            raven_heads=2,
            raven_slots=4,
            raven_top_k=2,
            raven_d_k=3,
            raven_d_v=2,
        )
        model = build_model_from_config(config)
        y, state = model(torch.randn(5, 2))

        self.assertEqual(y.shape, (1, 5, 3))
        self.assertEqual(state[0].shape, (1, 2, 4, 3))
        self.assertEqual(state[1].shape, (1, 2, 4, 2))


if __name__ == "__main__":
    unittest.main()
