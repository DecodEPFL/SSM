"""Mathematical diagnostics, not fitting targets or an additional benchmark corpus."""
import unittest

import torch

from direct_spending import DirectSpendingModel, SpendingConfig
from direct_spending.storage import StorageLayer


class ExpressivityDiagnostics(unittest.TestCase):
    def test_static_zero_input_tail_has_fixed_direction_and_geometric_storage(self):
        v = torch.zeros(1, 9, 1, dtype=torch.float64)
        z = torch.tensor([.3, .4], dtype=v.dtype).reshape(1, 1, 2).expand(1, 9, 2)
        y, _, info = StorageLayer()(v, z, torch.ones(1, 1, dtype=v.dtype), diagnostics=True)
        torch.testing.assert_close(info["storage"][0, :, 0], .75 ** torch.arange(10, dtype=v.dtype))
        torch.testing.assert_close(y[0, :, 0] / y[0, :, 1], torch.full((9,), .75, dtype=v.dtype))
        model = DirectSpendingModel(1, 1, SpendingConfig(core="static_mlp", beta_mode="fixed", fixed_beta=2.))
        with torch.no_grad():
            for parameter in model.core.parameters():
                parameter.zero_()
            output, state = model(torch.zeros(1, 11, 1))
        self.assertEqual(output.count_nonzero(), 0)
        self.assertEqual(state.storage.item(), 2.)

    def test_static_storage_cannot_distinguish_opposite_impulses_after_no_initial_spending(self):
        # An exact signed delay needs y0=0, hence p0=0 and identical storage
        # after +1 and -1. At v1=0 any static f(v1,S0) sees identical arguments.
        impulse = torch.tensor([[[1.]], [[-1.]]], dtype=torch.float64)
        _, storage, _ = StorageLayer()(impulse, torch.zeros_like(impulse))
        torch.testing.assert_close(storage[0], storage[1])
        same_next_spending = torch.full_like(impulse, .4)
        y, _, _ = StorageLayer()(torch.zeros_like(impulse), same_next_spending, storage)
        torch.testing.assert_close(y[0], y[1])
        self.assertFalse(torch.equal(y, impulse))

    def test_absolute_error_can_vanish_while_identity_relative_error_stays_large(self):
        # Continuous example z(v)=v at beta=0/gamma=1: y=v*abs(v) near zero.
        v = torch.tensor([[[1e-9]], [[-1e-9]]], dtype=torch.float64)
        y, _, _ = StorageLayer()(v, v)
        self.assertLess((y - v).abs().max(), 1e-8)
        self.assertGreater(((y - v).abs() / v.abs()).min(), .999999)


if __name__ == "__main__":
    unittest.main()
