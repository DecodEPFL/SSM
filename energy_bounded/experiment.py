"""Paired training and synchronized, warmed CPU/CUDA/MPS microbenchmarks.

Run from the repository root: python -m energy_bounded.experiment --help
No compile, autotuning, hyperparameter sweep, hidden fallback, or external data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from . import EnergyBoundedModel, EnergyReservoir, GRUCore, TCNCore, TransformerCore, state_bytes
from .task import certificate_metrics, trajectories


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timing(fn, device, *, repeats, warmup):
    begin = time.perf_counter()
    for _ in range(warmup):
        fn()
    synchronize(device)
    warmup_s = time.perf_counter() - begin
    samples = []
    for _ in range(repeats):
        synchronize(device)
        begin = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(1000 * (time.perf_counter() - begin))
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples, "warmup_s": warmup_s}


def make_core(kind, width):
    if kind == "gru":
        return GRUCore(2, 2, width=width)
    if kind == "tcn":
        return TCNCore(2, 2, width=width, dilations=(1, 2, 4))
    if kind == "transformer":
        return TransformerCore(2, 2, width=width, heads=2, layers=2, dropout=0.)
    raise ValueError(kind)


def fit(model, data, batch_indices, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    # Optimizer/gradient buffers are warmed on a clone to preserve paired initialization.
    warm = copy.deepcopy(model)
    warm_optimizer = torch.optim.Adam(warm.parameters(), lr=args.lr)
    x, target = data[0][:args.batch_size], data[1][:args.batch_size]
    begin = time.perf_counter()
    loss = (warm(x)[0] - target).square().mean()
    loss.backward(); warm_optimizer.step()
    synchronize(device)
    warmup_s = time.perf_counter() - begin
    del warm, warm_optimizer
    curve = []
    synchronize(device)
    begin = time.perf_counter()
    for indices in batch_indices:
        x, target = data[0][indices], data[1][indices]
        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(x)
        loss = (pred - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        curve.append(loss.detach().item())
    synchronize(device)
    elapsed = time.perf_counter() - begin
    model.eval()
    return {"training_s": elapsed, "training_warmup_s": warmup_s,
            "training_tokens_per_s": args.steps * args.batch_size * args.train_length / elapsed,
            "loss_curve": curve}


@torch.no_grad()
def evaluate(model, data, gamma):
    x, target = data
    if isinstance(model, EnergyBoundedModel):
        pred, state, diagnostics = model(x, diagnostics=True)
        core = model.core
        clipping = diagnostics["clipping_fraction"].item()
        backend = diagnostics["backend"]
        energy = diagnostics["energy"]
        supplied = (gamma * x.cpu().double()).square().sum(-1).cumsum(1)
        spent = pred.cpu().double().square().sum(-1).cumsum(1)
        identity = ((energy[:, 1:, 0].cpu().double() + spent - supplied) / (1 + supplied)).abs().max().item()
    else:
        pred, state = model(x)
        core, clipping, backend, identity = model, None, "raw", None
    # Also quantify how often the raw-trained model would exceed tank availability.
    raw, _ = core(x)
    _, _, info = EnergyReservoir(gamma)(x, raw, diagnostics=True)
    return {"mse": (pred - target).square().mean().item(),
            "tail_mse": (pred[:, -max(1, x.shape[1] // 8):] - target[:, -max(1, x.shape[1] // 8):]).square().mean().item(),
            "clipping_fraction": clipping,
            "hypothetical_clipping_fraction": info["clipping_fraction"].item(),
            "energy_identity_normalized_abs_error": identity, "reservoir_backend": backend,
            "persistent_state_bytes": state_bytes(state),
            **certificate_metrics(x, pred, gamma)}


def benchmark_core(core, args, device):
    """Isolate wrapper overhead with IDENTICAL trained core weights and inputs."""
    wrapper = EnergyBoundedModel(copy.deepcopy(core), args.gamma).to(device)
    rows = []
    for length in args.lengths:
        x, target = trajectories(range(40_000, 40_000 + args.batch_size), length, device=device)
        for name, model in (("core", core), ("core+wrapper", wrapper)):
            model.eval()
            def inference():
                with torch.no_grad():
                    model(x)
            forward = timing(inference, device, repeats=args.repeats, warmup=args.warmup)
            model.train()
            def forward_backward():
                model.zero_grad(set_to_none=True)
                y, _ = model(x)
                (y - target).square().mean().backward()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            fb = timing(forward_backward, device, repeats=args.repeats, warmup=args.warmup)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            model.eval()
            with torch.no_grad():
                _, state = model(x)
            rows.append({"path": name, "batch": args.batch_size, "length": length,
                         "forward": forward, "forward_backward": fb,
                         "forward_tokens_per_s": args.batch_size * length * 1000 / forward["median_ms"],
                         "forward_backward_tokens_per_s": args.batch_size * length * 1000 / fb["median_ms"],
                         "persistent_state_bytes": state_bytes(state), "cuda_peak_allocated_bytes": peak})
    # One cached step after a fixed context. Setup/prefill excluded from latency.
    for context in args.contexts:
        x, _ = trajectories([45_000], context + 1, device=device)
        for name, model in (("core", core), ("core+wrapper", wrapper)):
            model.eval()
            with torch.no_grad():
                _, state = model(x[:, :context])
                def step():
                    with torch.no_grad():
                        model.step(state, x[:, context])
                latency = timing(step, device, repeats=args.repeats, warmup=args.warmup)
            rows.append({"path": name, "context": context, "batch": 1, "step": latency,
                         "persistent_state_bytes": state_bytes(state)})
    return rows


def benchmark_reservoir(args, device):
    rows = []
    r = EnergyReservoir(args.gamma).to(device)
    for length in args.lengths:
        generator = torch.Generator().manual_seed(82 + length)
        x = torch.randn(args.batch_size, length, 2, generator=generator).to(device)
        raw = (2 * torch.randn(args.batch_size, length, 2, generator=generator)).to(device)
        for mode in ("scan", "reference"):
            proposal = raw.clone().requires_grad_()
            def forward():
                with torch.no_grad():
                    r(x, proposal, mode=mode)
            def backward():
                proposal.grad = None
                y, energy, _ = r(x, proposal, mode=mode)
                (y.square().mean() + energy.mean()).backward()
            ft = timing(forward, device, repeats=args.repeats, warmup=args.warmup)
            bt = timing(backward, device, repeats=args.repeats, warmup=args.warmup)
            _, _, info = r(x, proposal, mode=mode, diagnostics=True)
            rows.append({"length": length, "batch": args.batch_size, "mode": mode,
                         "backend_used": info["backend"], "forward": ft, "forward_backward": bt})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "pilot.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-length", type=int, default=64)
    parser.add_argument("--long-length", type=int, default=256)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--contexts", type=int, nargs="+", default=[16, 128, 512])
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.train_length, args.long_length, args.repeats,
           args.threads, args.width, *args.lengths, *args.contexts) < 1 or args.batch_size > 96:
        parser.error("Positive sizes required; batch size cannot exceed 96")
    if args.gamma < .9:
        parser.error("This task is analytically compatible with gamma>=.9; choose it before fitting")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; no implicit CPU fallback")
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; no implicit CPU fallback")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    datasets = {"train": trajectories(range(96), args.train_length, device=device),
                "validation": trajectories(range(10_000, 10_016), args.train_length, device=device),
                "test": trajectories(range(20_000, 20_016), args.train_length, device=device),
                "long_test": trajectories(range(30_000, 30_008), args.long_length, device=device)}
    target_audits = {name: certificate_metrics(x, y, args.gamma) for name, (x, y) in datasets.items()}
    if any(a["worst_normalized_prefix_violation"] > 1e-6 for a in target_audits.values()):
        raise RuntimeError("Target initialization is incompatible with the chosen gain")
    try:
        info = json.loads(subprocess.check_output(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], text=True,
            stderr=subprocess.DEVNULL))["SPHardwareDataType"][0]
        hardware = {k: info.get(k) for k in ("machine_model", "chip_type", "physical_memory",
                                            "number_processors")}
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError):
        hardware = {"processor": platform.processor(), "machine": platform.machine()}
    result = {"metadata": {"torch": torch.__version__, "python": platform.python_version(),
                           "platform": platform.platform(), "hardware": hardware, "device": str(device),
                           "dtype": "float32", "energy_dtype": str(EnergyReservoir().energy_dtype(device)),
                           "cuda_available": torch.cuda.is_available(),
                           "mps_available": torch.backends.mps.is_available(),
                           "compilation": "none; eager PyTorch", "initial_energy": 0.,
                           "ranking": "exploratory; fixed settings, no tuning sweep",
                           "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                             for p in Path(__file__).parent.glob("*.py")},
                           "memory": "logical persistent state bytes; CPU peak activation memory unmeasured",
                           "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
              "task": {"teacher_gain_upper_bound": .9, "target_audits": target_audits,
                       "split_ids": {"train": [0, 95], "validation": [10000, 10015],
                                     "test": [20000, 20015], "long_test": [30000, 30007]},
                       "rollout": "forced response on fresh longer trajectories; no autoregressive target feedback"},
              "training": [], "benchmarks": {}, "reservoir_benchmarks": []}
    for seed in args.seeds:
        generator = torch.Generator().manual_seed(9000 + seed)
        indices = [torch.randperm(96, generator=generator)[:args.batch_size].to(device)
                   for _ in range(args.steps)]
        for kind in ("gru", "tcn", "transformer"):
            torch.manual_seed(seed)
            base = make_core(kind, args.width).to(device)
            initial_digest = hashlib.sha256(repr([p.detach().cpu().tolist()
                                                  for p in base.parameters()]).encode()).hexdigest()
            raw = copy.deepcopy(base)
            wrapped = EnergyBoundedModel(copy.deepcopy(base), args.gamma).to(device)
            for arm, model in (("raw", raw), ("wrapped", wrapped)):
                print(f"seed={seed} {kind}/{arm}: {args.steps} training steps", flush=True)
                row = {"seed": seed, "core": kind, "arm": arm,
                       "initial_parameter_sha256": initial_digest,
                       "parameters": sum(p.numel() for p in model.parameters()),
                       **fit(model, datasets["train"], indices, args, device)}
                row["evaluation"] = {name: evaluate(model, datasets[name], args.gamma)
                                     for name in ("validation", "test", "long_test")}
                result["training"].append(row)
                print(f"  test MSE={row['evaluation']['test']['mse']:.6g}; training={row['training_s']:.3f}s", flush=True)
            if seed == args.seeds[0]:
                print(f"benchmark {kind}: identical trained core weights with/without wrapper", flush=True)
                result["benchmarks"][kind] = benchmark_core(wrapped.core, args, device)
    result["reservoir_benchmarks"] = benchmark_reservoir(args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
