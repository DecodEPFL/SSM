import tempfile
import unittest
from pathlib import Path

import torch

from energy_bounded import EnergyBoundedModel, GRUCore, TCNCore, TransformerCore
from energy_bounded.benchmark_adapter import BenchmarkAdapter
from energy_bounded.task import certificate_metrics, trajectories


class IntegrationTests(unittest.TestCase):
    def test_adapter_layouts_and_chunk_state(self):
        model = BenchmarkAdapter(EnergyBoundedModel(GRUCore(2, 2, width=8)))
        u, _ = trajectories([0], 32)
        y, _ = model(u[0])
        a, state = model(u[0, :11])
        b, state = model(u[0, 11:], state)
        torch.testing.assert_close(torch.cat((a, b)), y)
        batched, _ = model(u)
        torch.testing.assert_close(y, batched[0])

    def test_existing_system_id_trainer_actual_training_and_validation(self):
        root = Path(__file__).resolve().parents[2]
        if not (root / "Test_files" / "Benchmark.py").exists():
            self.skipTest("Run inside SSM-clean to exercise its optional legacy trainer")
        from Test_files.Benchmark import SystemIDTrainer, TrainingConfig

        train_x, train_y = trajectories([0], 32)
        val_x, val_y = trajectories([10000], 32)
        with tempfile.TemporaryDirectory() as temp:
            for core in (GRUCore(2, 2, width=8), TCNCore(2, 2, width=8),
                         TransformerCore(2, 2, width=8)):
                for wrapped in (False, True):
                    model = BenchmarkAdapter(EnergyBoundedModel(core) if wrapped else core)
                    config = TrainingConfig(num_epochs=1, device="cpu", normalize_data=False,
                                            init_window=0, save_dir=Path(temp))
                    trainer = SystemIDTrainer(model, config, normalizer=None)
                    loss, _ = trainer.train_epoch(train_x[0], train_y[0])
                    val_loss, pred = trainer.validate(val_x[0], val_y[0], n=0)
                    self.assertTrue(torch.isfinite(torch.tensor([loss, val_loss])).all())
                    if wrapped:
                        metrics = certificate_metrics(val_x, pred[None], 1.)
                        self.assertEqual(metrics["worst_normalized_prefix_violation"], 0.)


if __name__ == "__main__":
    unittest.main()
