"""State and layout contracts shared by the SSM stack and its streaming APIs."""

import copy
import unittest

import torch
from torch import nn

from neural_ssm import ContextualDeepSSM, DeepSSM, SimpleRNN, SSMConfig
from neural_ssm.ssm.layers import SSL


class SSMInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_final_state_path_preserves_outputs_and_gradients(self):
        # This path avoids converting every l2n state back to physical coordinates.
        for param in ("lru", "l2n", "defect", "tv", "tvc"):
            with self.subTest(param=param):
                config = SSMConfig(
                    param=param,
                    d_model=4,
                    d_state=6,
                    ff="LGLU2",
                    gamma=None if param == "lru" else 1.0,
                )
                full = SSL(config).eval()
                final = SSL(config).eval()
                final.load_state_dict(full.state_dict())
                inputs = torch.randn(2, 17, 4, requires_grad=True)
                other_inputs = inputs.detach().clone().requires_grad_()
                output, trajectory = full(inputs, mode="scan")
                other_output, last = final(other_inputs, mode="scan", return_state=False)
                torch.testing.assert_close(output, other_output)
                torch.testing.assert_close(trajectory[:, -1], last)
                loss = output.square().sum() + trajectory[:, -1].abs().square().sum()
                other_loss = other_output.square().sum() + last.abs().square().sum()
                gradients = torch.autograd.grad(
                    loss, (inputs, *full.parameters()), allow_unused=True
                )
                other_gradients = torch.autograd.grad(
                    other_loss, (other_inputs, *final.parameters()), allow_unused=True
                )
                for gradient, other in zip(gradients, other_gradients):
                    if gradient is None:
                        self.assertIsNone(other)
                    else:
                        torch.testing.assert_close(gradient, other, rtol=2e-4, atol=2e-5)

    def test_cell_return_layouts_and_empty_sequences(self):
        for param in ("l2n", "defect", "tv", "tvc"):
            cell = DeepSSM(2, 2, param=param, d_model=4, d_state=6).blocks[0].lru.eval()
            for length in (0, 7):
                with self.subTest(param=param, length=length), torch.no_grad():
                    inputs = torch.randn(2, length, 4)
                    output, trajectory, last = cell(inputs, return_last=True)
                    self.assertEqual(output.shape, (2, length, 4))
                    self.assertEqual(trajectory.shape, (2, length + 1, 6))
                    torch.testing.assert_close(last, trajectory[:, -1])
                    if param in ("l2n", "defect"):
                        time_output, time_states, time_last = cell(
                            inputs.transpose(0, 1), time_first=True, return_last=True
                        )
                        torch.testing.assert_close(time_output, output.transpose(0, 1))
                        torch.testing.assert_close(time_states, trajectory.transpose(0, 1))
                        torch.testing.assert_close(time_last, last)
                    torch.testing.assert_close(cell(inputs, return_state=False), output)
                    final_output, final_state = cell(inputs, return_state=False, return_last=True)
                    torch.testing.assert_close(final_output, output)
                    torch.testing.assert_close(final_state, last)

    def test_lru_accepts_single_samples_and_unbatched_sequences(self):
        cell = DeepSSM(2, 2, param="lru", d_model=4, d_state=6).blocks[0].lru.eval()
        for inputs in (torch.randn(4), torch.randn(7, 4)):
            with self.subTest(shape=inputs.shape), torch.no_grad():
                batched = inputs.reshape(1, -1, 4)
                output, state = cell(inputs)
                reference, reference_state = cell(batched)
                torch.testing.assert_close(output, reference)
                torch.testing.assert_close(state, reference_state)

    def test_unknown_execution_modes_are_rejected(self):
        for param in ("lru", "l2n", "defect", "tv", "tvc"):
            with self.subTest(param=param):
                cell = DeepSSM(2, 2, param=param, d_model=4, d_state=6).blocks[0].lru
                with self.assertRaises(ValueError):
                    cell(torch.randn(2, 7, 4), mode="scna")

    def test_rnn_explicit_state_overrides_reset_and_supports_streaming(self):
        model = SimpleRNN(2, 4, 3, num_layers=2).eval()
        inputs = torch.randn(2, 11, 2)
        initial = torch.randn(2, 2, 4)
        with torch.no_grad():
            output, last = model(inputs, state=initial, reset_state=True)
            hidden, reference_last = model.rnn(inputs, initial)
            torch.testing.assert_close(output, model.out_proj(hidden))
            torch.testing.assert_close(last, reference_last)
            left, state = model(inputs[:, :5], state=initial)
            right, state = model(inputs[:, 5:], state=state)
            torch.testing.assert_close(torch.cat((left, right), dim=1), output)
            torch.testing.assert_close(state, last)

    def test_legacy_dense_interface_retains_pre_step_trajectory(self):
        cell = DeepSSM(2, 2, param="l2nt", d_model=4, d_state=6).blocks[0].lru.eval()
        inputs = torch.randn(2, 7, 4)
        with torch.no_grad():
            output, trajectory = cell(inputs)
            legacy_output, legacy_states = cell.forward_original(inputs)
            torch.testing.assert_close(legacy_output, output)
            torch.testing.assert_close(legacy_states, trajectory[:, :-1])
            time_output, last = cell.forward_original(
                inputs.transpose(0, 1), time_first=True, return_state=False
            )
            torch.testing.assert_close(time_output, output.transpose(0, 1))
            torch.testing.assert_close(last, trajectory[:, -1])

    def test_difference_context_streaming_preserves_encoder_and_input_gradients(self):
        model = ContextualDeepSSM(
            2,
            2,
            2,
            context_modes="input",
            context_filter="difference",
            context_encoder=nn.Linear(3, 2),
            param="l2n",
            ff="LGLU2",
            gamma=1.0,
            d_model=4,
            d_state=6,
            n_layers=1,
        ).eval()
        streamed = copy.deepcopy(model)
        inputs = torch.randn(2, 13, 2, requires_grad=True)
        context = torch.randn(2, 13, 3, requires_grad=True)
        stream_inputs = inputs.detach().clone().requires_grad_()
        stream_context = context.detach().clone().requires_grad_()
        output, _ = model(inputs, context)
        chunks, state = [], None
        for start in (0, 5, 10):
            end = min(start + 5, inputs.shape[1])
            previous = stream_context[:, start - 1 : start] if start else None
            chunk, state = streamed(
                stream_inputs[:, start:end],
                stream_context[:, start:end],
                state=state,
                time_offset=start,
                previous_context=previous,
                detach_state=False,
            )
            chunks.append(chunk)
        stream_output = torch.cat(chunks, dim=1)
        torch.testing.assert_close(stream_output, output, rtol=2e-4, atol=2e-5)
        gradients = torch.autograd.grad(
            output.square().sum(), (inputs, context, *model.parameters()), allow_unused=True
        )
        stream_gradients = torch.autograd.grad(
            stream_output.square().sum(),
            (stream_inputs, stream_context, *streamed.parameters()),
            allow_unused=True,
        )
        for gradient, other in zip(gradients, stream_gradients):
            if gradient is None:
                self.assertIsNone(other)
            else:
                torch.testing.assert_close(gradient, other, rtol=3e-4, atol=2e-5)
        with self.assertRaisesRegex(ValueError, "one sample per batch"):
            model(inputs, context, previous_context=context[:, :2])

    def test_freezing_certificate_includes_legacy_trainable_gain(self):
        model = DeepSSM(
            2,
            2,
            param="zak",
            ff="LGLU2",
            gamma=1.0,
            train_gamma=True,
            d_model=4,
            d_state=6,
            n_layers=1,
        )
        gain = model.blocks[0].lru.gamma_raw
        self.assertTrue(gain.requires_grad)
        model.freeze_certificate()
        self.assertFalse(gain.requires_grad)


if __name__ == "__main__":
    unittest.main()
