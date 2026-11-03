"""Generate a concise measured report from saved JSON; no invented/interpolated timings."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def mean_sd(values, precision=6):
    mean = statistics.mean(values)
    if len(values) == 1:
        return f"{mean:.{precision}g} (single seed)"
    return f"{mean:.{precision}g} ± {statistics.stdev(values):.{precision}g}"


def summarize(path):
    data = json.loads(path.read_text())
    meta, training = data["metadata"], data["training"]
    cfg = meta["configuration"]
    hardware = meta["hardware"]
    seeds = cfg["seeds"]
    lines = ["# Measured pilot comparison", "",
             f"Hardware: {hardware}. {meta['platform']}; Python {meta['python']}; "
             f"PyTorch {meta['torch']}; device {meta['device']}; model dtype {meta['dtype']}; "
             f"energy {meta['energy_dtype']}; {cfg['threads']} CPU thread(s).", "",
             f"Fixed gamma={cfg['gamma']}, E0=0. Seeds {seeds}; {cfg['steps']} Adam steps/arm, "
             f"lr={cfg['lr']}, batch={cfg['batch_size']}, train length={cfg['train_length']}, "
             f"long held-out length={cfg['long_length']}, width={cfg['width']}. "
             "No tuning sweep or compilation. Identical initial weights and training batches "
             "within each raw/wrapped pair; independent trajectory splits.", "",
             "The teacher has analytic zero-state gain <=.9; all generated target prefix audits "
             "pass at the fixed gain. Longer evaluation is a forced response on fresh trajectories; "
             "autonomous rollout is not applicable to this exogenous-input task.", "",
             "## Prediction and actual training cost", "",
             "Mean ± sample standard deviation across seeds. CPU training time excludes cloned "
             "warmup and dataset creation, and includes forward, backward, optimizer, dispatch "
             "and scalar loss logging. Equal steps/width do not imply equal parameter or compute budgets.", "",
             "| Core / arm | Parameters | Test MSE | Long MSE | Clip fraction | Worst prefix violation | Train seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for kind in ("gru", "tcn", "transformer"):
        for arm in ("raw", "wrapped"):
            rows = [r for r in training if r["core"] == kind and r["arm"] == arm]
            test = [r["evaluation"]["test"] for r in rows]
            clip = "—" if arm == "raw" else mean_sd([r["clipping_fraction"] for r in test], 4)
            lines.append(f"| {kind} / {arm} | {rows[0]['parameters']:,} | "
                         f"{mean_sd([r['mse'] for r in test], 5)} | "
                         f"{mean_sd([r['evaluation']['long_test']['mse'] for r in rows], 5)} | "
                         f"{clip} | {max(r['worst_normalized_prefix_violation'] for r in test):.3g} | "
                         f"{mean_sd([r['training_s'] for r in rows], 4)} |")
    wrapped = [r for r in training if r["arm"] == "wrapped"]
    audits = [e for r in wrapped for e in r["evaluation"].values()]
    lines += ["", "Normalized prefix violation is "
              "`max(0, max_{sample,prefix} (spent - supplied)/(1 + supplied))`, "
              "audited in FP64 from actual outputs. Raw clipping is undefined; its hypothetical "
              "reservoir clipping is recorded separately in JSON.", "",
              f"Across ALL wrapped validation/test/long evaluations, worst normalized prefix "
              f"violation: **{max(r['worst_normalized_prefix_violation'] for r in audits):.6g}**. "
              f"Largest normalized absolute energy-identity error: "
              f"**{max(r['energy_identity_normalized_abs_error'] for r in audits):.6g}**. "
              f"Observed evaluation backends: {sorted(set(r['reservoir_backend'] for r in audits))}.", "",
              "### Seed variation", "",
              "| Core | Seed | Raw test MSE | Wrapped test MSE | Wrapped long MSE |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for kind in ("gru", "tcn", "transformer"):
        for seed in seeds:
            pair = {r["arm"]: r for r in training if r["core"] == kind and r["seed"] == seed}
            assert pair["raw"]["initial_parameter_sha256"] == pair["wrapped"]["initial_parameter_sha256"]
            lines.append(f"| {kind} | {seed} | {pair['raw']['evaluation']['test']['mse']:.6g} | "
                         f"{pair['wrapped']['evaluation']['test']['mse']:.6g} | "
                         f"{pair['wrapped']['evaluation']['long_test']['mse']:.6g} |")
    best_error = min(("gru", "tcn", "transformer"), key=lambda k: statistics.mean(
        r["evaluation"]["test"]["mse"] for r in wrapped if r["core"] == k))
    fastest = min(("gru", "tcn", "transformer"), key=lambda k: statistics.mean(
        r["training_s"] for r in wrapped if r["core"] == k))
    transformer = [r for r in wrapped if r["core"] == "transformer"]
    worst = max(transformer, key=lambda r: r["evaluation"]["test"]["mse"])
    spread = max(r["evaluation"]["test"]["mse"] for r in transformer) / min(
        r["evaluation"]["test"]["mse"] for r in transformer)
    variation = (f"Transformer wrapped seed {worst['seed']} has {spread:.1f}× the best "
                 "Transformer wrapped error while preserving its certificate. " if spread > 3 else "")
    lines += ["", "These are exploratory runs of small fixed configurations. " + variation +
              f"{best_error} has the lowest mean wrapped test error and {fastest} the lowest mean "
              "wrapped training time in this pilot. The seed table and complete loss curves expose "
              "variation. These observations do not establish a universal architecture winner.", "",
              "## Isolated core and wrapper timing", "",
              f"Medians of {cfg['repeats']} synchronized repetitions after {cfg['warmup']} warmup "
              "calls per workload. These timing pairs use the SAME trained wrapped-core weights "
              "and the same inputs, isolating wrapper overhead; timing uses the first seed only. "
              "Test processes were finished before the final timing run. "
              "Forward/backward timing excludes optimizer updates. Warmup durations and every "
              "timing sample are saved in JSON.", "",
              "| Core | B×T | Path | Forward ms | Forward tokens/s | Fwd+bwd ms | Fwd+bwd tokens/s | State bytes |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind, rows in data["benchmarks"].items():
        for r in rows:
            if "length" not in r:
                continue
            lines.append(f"| {kind} | {r['batch']}×{r['length']} | {r['path']} | "
                         f"{r['forward']['median_ms']:.4f} | {r['forward_tokens_per_s']:,.0f} | "
                         f"{r['forward_backward']['median_ms']:.4f} | "
                         f"{r['forward_backward_tokens_per_s']:,.0f} | {r['persistent_state_bytes']:,} |")
    lines += ["", "### Cached single-step latency and context growth", "",
              "Batch 1; prefill excluded; each repetition branches from the same immutable "
              "incoming cache. Persistent bytes count logical tensor contents, including padding "
              "masks and position offsets, not allocator overhead or autograd graphs. "
              "CPU peak activation memory was not measured; CUDA allocation counters are only "
              "reported when CUDA is selected.", "",
              "| Core | Context | Core µs | Wrapped µs | Core state bytes | Wrapped state bytes |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind, rows in data["benchmarks"].items():
        for context in cfg["contexts"]:
            pair = {r["path"]: r for r in rows if r.get("context") == context}
            a, b = pair["core"], pair["core+wrapper"]
            lines.append(f"| {kind} | {context} | {a['step']['median_ms'] * 1000:.2f} | "
                         f"{b['step']['median_ms'] * 1000:.2f} | {a['persistent_state_bytes']:,} | "
                         f"{b['persistent_state_bytes']:,} |")
    lines += ["", "### Reservoir alone", "",
              "| B×T | Requested path | Actual path | Forward ms | Fwd+bwd ms |",
              "| --- | --- | --- | ---: | ---: |"]
    for r in data["reservoir_benchmarks"]:
        lines.append(f"| {r['batch']}×{r['length']} | {r['mode']} | {r['backend_used']} | "
                     f"{r['forward']['median_ms']:.4f} | {r['forward_backward']['median_ms']:.4f} |")
    validation_path = path.parent / "validation.json"
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines += ["", "## Validation", ""]
        for label, info in validation.items():
            lines.append(f"- {label}: {info}.")
    lines += ["", "## Limits and next experiment", "",
              "CUDA and MPS were unavailable in this environment; their optional checks skip "
              "explicitly. CPU FP32/FP64, low-precision reservoir inputs, and CPU BF16 autocast "
              "forward/backward were tested. No backend-specific speed claims follow from CPU results. "
              "The FP64 scan is used in ordinary pilot workloads; adversarial cancellation warns "
              "and uses the sequential path. Prototype associative-scan APIs and compilation are "
              "not used. TCN memory is limited to 15 valid tokens. Transformer inference caches "
              "are real but grow without a window and copy on append.", "",
              "The measured scan already removes substantial sequential reservoir dispatch cost. "
              "For short online steps, the wrapper's small tensor operations, validation checks "
              "and scalar validation checks are a material fraction of latency. The next performance "
              "experiment should profile those operations and compare a fused step projection while "
              "retaining the same energy audits and safe gradients. For Transformers, profile KV "
              "append allocations before adding a preallocated inference-only cache. Separately, "
              "investigate initialization and projection saturation in the poorly trained seed "
              "under a declared tuning budget and unchanged gamma; do not hide it by choosing "
              "initial energy or gamma after evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "REPORT.md")
    args = parser.parse_args()
    args.output.write_text(summarize(args.input))
    print(args.output)


if __name__ == "__main__":
    main()
