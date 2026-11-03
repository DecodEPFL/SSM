"""Small CPU diagnostic; writes only the requested JSON, never suite run artifacts.

Run from the repository: python -m direct_spending.validation.learning_rate_check
    --output /tmp/spending-rate-check.json --lrs 0.003 --epochs 40
Uses the failed run's architecture/initialization and native trajectory protocol.
This single-seed diagnostic is not an architecture or hyperparameter ranking.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from Test_files import run_benchmarks as runner
from direct_spending import benchmark as spending
from direct_spending.config import SpendingConfig


class History:
    def __init__(self):
        self.train_losses, self.val_losses = [], []

    def update(self, epoch, train, validation, model):
        self.train_losses, self.val_losses = train.copy(), validation.copy()


def check_reference(config, bench, norm):
    """Compare actual full-trajectory outputs and gradients at initialization."""
    torch.manual_seed(0)
    model = spending.SpendingSim(bench.n_u, bench.n_y, config)
    u = norm.norm_u(torch.as_tensor(bench.train[0].u, dtype=torch.float32))[None]
    target = norm.norm_y(torch.as_tensor(bench.train[0].y, dtype=torch.float32))[None]
    rows, outputs, gradients = {}, {}, {}
    for method in ("scan", "reference"):
        output, _, info = model.model(u, implementation=method, diagnostics=True)
        loss = (output - target).square().mean()
        gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])
        outputs[method], gradients[method] = output.detach(), gradient.detach()
        rows[method] = {"mse": loss.item(), "all_gradients_finite": bool(torch.isfinite(gradient).all()),
                        "gradient_norm": gradient.norm().item(),
                        "head_saturation_rate": info["head_saturation_rate"].item(),
                        "energy_identity_error": info["max_normalized_identity_error"].item(),
                        "max_normalized_energy_violation": info["max_normalized_energy_violation"].item()}
    rows["max_abs_output_difference"] = (outputs["scan"] - outputs["reference"]).abs().max().item()
    rows["relative_gradient_l2_difference"] = ((gradients["scan"] - gradients["reference"]).norm()
                                              / gradients["reference"].norm()).item()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lrs", type=float, nargs="+", default=[.003])
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    bench = runner.load_benchmark("Cascaded_Tanks")
    norm = runner.Standardizer.fit(bench.train)
    config = SpendingConfig(core="transformer", width=16, layers=3, gamma=10.,
                            beta_mode="learned", beta_initial=.1, heads=2, dropout=0.)
    result = {"purpose": "exploratory learning-rate diagnostic, seed 0, no ranking",
              "torch": torch.__version__, "python": platform.python_version(),
              "hardware": platform.machine(), "os": platform.platform(),
              "device": "cpu", "threads": 2, "dtype": "float32 core; float64 auto storage",
              "model_config": config.to_dict(), "coordinates": "standardized",
              "target_compatibility": spending.target_compatibility(bench, norm, config.gamma),
              "scan_reference_initialization_check": check_reference(config, bench, norm),
              "runs": []}
    for lr in args.lrs:
        torch.manual_seed(0)
        model = spending.SpendingSim(bench.n_u, bench.n_y, config)
        history = History()
        row = {"lr": lr, "seed": 0, "epochs_requested": args.epochs,
               "batch_size": 32, "grad_clip": 5., "weight_decay": 0.,
               "scheduler": "CosineAnnealingLR, T_max=epochs_requested",
               "n_params": sum(p.numel() for p in model.parameters())}
        start = time.perf_counter()
        try:
            row["history"] = runner.train_model(model, bench, norm,
                runner.TrainConfig(epochs=args.epochs, lr=lr, grad_clip=5., seed=0),
                torch.device("cpu"), monitor=history)
            row["status"] = "ok"
            row["validation_metrics"], _ = runner.evaluate_validation(model, bench, norm, torch.device("cpu"))
            row["certificate"] = model.certificate()
            row["selected_checkpoint_storage_diagnostics"] = []
            for sequence in bench.validation:
                u = norm.norm_u(torch.as_tensor(sequence.u, dtype=torch.float32))[None]
                diagnostics = model.storage_diagnostics(u, initialization_window=sequence.state_initialization_window_length or 0)
                diagnostics.pop("storage_trace", None)
                row["selected_checkpoint_storage_diagnostics"].append(diagnostics)
        except (FloatingPointError, RuntimeError) as exc:
            row.update(status="error", error=str(exc), history={
                "train_losses": history.train_losses, "val_losses": history.val_losses})
        row["wall_time_s"] = time.perf_counter() - start
        result["runs"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: row[k] for k in ("lr", "status", "wall_time_s")}), flush=True)


if __name__ == "__main__":
    main()
