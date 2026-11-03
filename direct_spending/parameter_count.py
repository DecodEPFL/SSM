"""Count the runner's actual models, dimensions and parameter-budget policy; no training."""
from __future__ import annotations
from contextlib import redirect_stdout
import io


def parameter_counts(args):
    import torch
    from Test_files import run_benchmarks as runner

    config = runner.model_config_from_args(args)
    runner._validate_initialization(args.models, config)
    rows = []
    # Reuse the real factory, including parameter matching. CPU construction
    # does not require the selected training accelerator or allocate any caches.
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        for name in runner.resolve_benchmark_names(args.benchmarks):
            data = runner.load_benchmark(name)
            budget, policy = runner.resolve_param_budget(args.param_budget, data.n_u, data.n_y, config)
            for model_name in args.models:
                model = runner.build_model(model_name, data.n_u, data.n_y, config, param_budget=budget)
                rows.append({"benchmark": name, "model": model_name,
                             "label": model.spending_config.label if hasattr(model, "spending_config") else model_name,
                             "input_features": data.n_u, "output_features": data.n_y,
                             "trainable": runner._count_params(model),
                             "total": sum(p.numel() for p in model.parameters()),
                             "effective_width": runner._effective_width(model),
                             "budget": budget or None, "budget_policy": policy})
    return rows


def main():
    from Test_files import run_benchmarks as runner
    args = runner.build_parser().parse_args()
    for row in parameter_counts(args):
        print(f"{row['benchmark']} / {row['label']}\n"
              f"  {row['trainable']:,} trainable parameters; {row['total']:,} total\n"
              f"  Inputs: {row['input_features']}  Outputs: {row['output_features']}  "
              f"Effective width: {row['effective_width']}\n"
              f"  Parameter budget: {row['budget_policy']} ({row['budget'] or 'off'})\n")


if __name__ == "__main__":
    main()
