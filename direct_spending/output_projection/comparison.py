"""Small native-data comparison using the existing trainer, plus isolated CPU timings.

python -m direct_spending.output_projection.comparison --output /tmp/readout-check
Single-seed results are exploratory, not an architecture or tuning ranking.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from direct_spending import benchmark as adapter
from direct_spending.config import CORES, SpendingConfig
from energy_bounded.state import map_state
from Test_files import run_benchmarks as runner


def state_bytes(state):
    unique = {}
    def visit(value):
        storage = value.untyped_storage()
        unique[storage.data_ptr()] = storage.nbytes()
        return value
    map_state(state, visit)
    return sum(unique.values())


def timings(model, lengths=(128, 2048), batch=2, repetitions=3):
    """CPU only: no asynchronous device work, compile, optimizer step or plotting."""
    rows = []
    model.eval()
    for length in lengths:
        inputs = torch.randn(batch, length, model.model.d_input)
        for scope in ("core", "core_plus_readout"):
            def forward():
                return model.model.core(inputs)[0] if scope == "core" else model(inputs)
            for backward in (False, True):
                samples = []
                for trial in range(2 + repetitions):
                    model.zero_grad(set_to_none=True)
                    start = time.perf_counter()
                    if backward:
                        forward().square().mean().backward()
                    else:
                        with torch.inference_mode(): forward()
                    duration = time.perf_counter() - start
                    if trial >= 2: samples.append(duration)
                seconds = statistics.median(samples)
                rows.append({"scope": scope, "batch": batch, "length": length,
                             "operation": "forward_backward" if backward else "inference",
                             "warmup_runs": 2, "repetitions": repetitions,
                             "median_s": seconds, "tokens_s": batch * length / seconds})
    steps = []
    with torch.inference_mode():
        for context in (0, 128, 1024):
            prefix = torch.randn(1, context, model.model.d_input)
            token = torch.randn(1, model.model.d_input)
            for scope in ("core", "core_plus_readout"):
                if scope == "core":
                    _, state = model.model.core(prefix)
                    tick = model.model.core.step
                else:
                    _, state = model.model(prefix)
                    tick = model.model.step
                _, state = tick(state, token)  # One step of warmup outside timing.
                samples = []
                for _ in range(20):
                    start = time.perf_counter()
                    _, state = tick(state, token)
                    samples.append(time.perf_counter() - start)
                steps.append({"scope": scope, "context_start": context,
                              "context_end": context + 21, "batch": 1,
                              "median_step_us": 1e6 * statistics.median(samples),
                              "persistent_state_bytes_end": state_bytes(state)})
    return {"sequences": rows, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cores", nargs="+", choices=CORES, default=["lstm", "transformer"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=10.)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--skip-timings", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = runner.load_benchmark("Cascaded_Tanks")
    norm = runner.Standardizer.fit(data.train)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "exploratory single-seed readout comparison; no architecture ranking",
              "python": platform.python_version(), "torch": str(torch.__version__),
              "hardware": platform.platform(), "device": "cpu", "threads": 2,
              "core_dtype": "float32", "energy_dtype": "float64 (auto)",
              "width": args.width, "layers": args.layers, "seed": args.seed,
              "gamma": args.gamma, "beta": 0., "batch_size": 32, "lr": args.lr,
              "weight_decay": 0., "grad_clip": 1., "dropout": 0., "compile": False,
              "coordinates": "standardized", "epochs": args.epochs,
              "target_compatibility": adapter.target_compatibility(data, norm, args.gamma),
              "runs": []}
    for core in args.cores:
        starting_core = None
        for readout in ("spending", "projected"):
            torch.manual_seed(args.seed)
            cfg = SpendingConfig(core=core, width=args.width, layers=args.layers,
                                 readout=readout, gamma=args.gamma, beta_mode="zero", diagnostics=True)
            model = adapter.SpendingSim(data.n_u, data.n_y, cfg)
            weights = model.model.core.state_dict()
            if starting_core is None:
                starting_core = {k: v.clone() for k, v in weights.items()}
            else:
                assert all(torch.equal(weights[k], v) for k, v in starting_core.items())
            history = runner.train_model(model, data, norm,
                runner.TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed), torch.device("cpu"))
            metrics, _ = runner.evaluate_validation(model, data, norm, torch.device("cpu"))
            for row in metrics:
                row["storage_diagnostics"].pop("storage_trace", None)
                row["storage_diagnostics"].pop("raw_proposal_norm", None)
            result = {"core": core, "readout": readout, "same_initial_core_weights": True,
                      "n_params": runner._count_params(model), "history": history,
                      "validation": metrics, "certificate": model.certificate()}
            checkpoint = args.output / f"{core}-{readout}.pt"
            adapter.save_checkpoint(checkpoint, model, norm, history, {"comparison": str(args.output)})
            restored, _ = adapter.load_checkpoint(checkpoint)
            probe = norm.norm_u(torch.tensor(data.validation[0].u, dtype=torch.float32))[None]
            with torch.no_grad(): torch.testing.assert_close(restored(probe), model(probe))
            result["checkpoint_reload_matches"] = True
            if not args.skip_timings:
                result["timings"] = timings(model)
            report["runs"].append(result)
            (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"core": core, "readout": readout, "n_params": result["n_params"],
                              "rmse": metrics[0]["rmse"], "training_s": history["train_time_s"]}), flush=True)


if __name__ == "__main__":
    main()
