import unittest

import torch

from energy_bounded.task import certificate_metrics, teacher, trajectories


class TaskTests(unittest.TestCase):
    def test_zero_initialization_and_analytic_gain(self):
        u, y = trajectories(range(10), 513)
        self.assertEqual(y[:, :3].count_nonzero(), 0)
        self.assertEqual(certificate_metrics(u, y, .9)["worst_normalized_prefix_violation"], 0.)
        self.assertEqual(teacher(torch.zeros(2, 100, 2)).count_nonzero(), 0)

    def test_task_reproducibility_and_independent_trajectory_ids(self):
        x, y = trajectories([3, 7], 64)
        z, w = trajectories([7], 64)
        torch.testing.assert_close(x[1], z[0], rtol=0, atol=0)
        torch.testing.assert_close(y[1], w[0], rtol=0, atol=0)
        val, _ = trajectories([10000], 64)
        self.assertFalse(torch.equal(val, x[:1]))


if __name__ == "__main__":
    unittest.main()
