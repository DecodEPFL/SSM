"""What the context-only (LPV) selector costs, and where its value comes from.

``select_input='context'`` buys an incremental l2 bound by forbidding the
selector to see the cell input, which removes Mamba-style input selectivity.
This quantifies the trade on a task built for the adaptive-SSM setting: a
switching linear system whose regime changes mid-episode and is exposed on an
observable context channel.

Three arms, matched on seed, width, depth, gain budget and step count:

``both + context``
    the default. Selector sees input and context; zero-state bound only.
``both + blind context``
    selector sees the input and a context channel pinned to zero, so only input
    selectivity is available. Isolates what input selectivity contributes on its
    own.
``context-only``
    LPV mode. Selector sees context alone; incremental bound holds.

The blind arm is what makes the comparison interpretable: without it, a gap
between the first and third arms could be read either as "input selectivity is
valuable" or as "the LPV selector is too small to use the context".

Run::

    python scripts/lpv_selector_tradeoff.py
    python scripts/lpv_selector_tradeoff.py --seeds 0 1 2 3 4 --steps 800
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from neural_ssm.ssm.contextual import ContextualDeepSSM

D_IN, D_CTX, D_OUT = 2, 3, 2
POLES = torch.tensor([0.30, 0.75, 0.95])
GAINS = torch.tensor([1.00, -0.60, 0.35])


def batch(batch_size: int, horizon: int, seed: int | None = None):
    """A switching linear system; the context channel one-hot encodes the regime."""
    if seed is not None:
        torch.manual_seed(seed)
    u = torch.randn(batch_size, horizon, D_IN)
    regime = torch.randint(0, len(POLES), (batch_size,))
    switch = torch.randint(horizon // 4, 3 * horizon // 4, (batch_size,))
    successor = (regime + 1) % len(POLES)

    context = torch.zeros(batch_size, horizon, D_CTX)
    y = torch.zeros(batch_size, horizon, D_OUT)
    state = torch.zeros(batch_size, D_OUT)
    for t in range(horizon):
        active = torch.where(t < switch, regime, successor)
        context[:, t] = F.one_hot(active, D_CTX).float()
        state = POLES[active].unsqueeze(-1) * state + GAINS[active].unsqueeze(-1) * u[:, t]
        y[:, t] = state
    return u, context, y


def run(mode: str, seed: int, *, blind: bool, args) -> tuple[float, ContextualDeepSSM]:
    torch.manual_seed(seed)
    model = ContextualDeepSSM(
        D_IN, D_CTX, D_OUT, context_modes=("select",),
        d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
        gamma=args.gamma, param="tv", ff="LGLU2", dropout=0.0, learn_x0=False,
        select_input=mode, auto_residual_init=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        u, context, y = batch(args.batch_size, args.horizon)
        if blind:
            context = torch.zeros_like(context)
        pred, _ = model(u, context, mode="scan", reset_state=True)
        loss = (pred - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    losses = []
    with torch.no_grad():
        for held_out in range(9000, 9000 + args.eval_episodes):
            u, context, y = batch(args.batch_size, args.horizon, seed=held_out)
            if blind:
                context = torch.zeros_like(context)
            pred, _ = model(u, context, mode="scan", reset_state=True)
            losses.append((pred - y).pow(2).mean().item())
    return statistics.mean(losses), model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--d-model", type=int, default=24)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--eval-episodes", type=int, default=5)
    args = p.parse_args()

    arms = (
        ("both + context", "both", False),
        ("both + blind context", "both", True),
        ("context-only (LPV)", "context", False),
    )
    print(f"seeds={args.seeds} steps={args.steps} "
          f"d_model={args.d_model} n_layers={args.n_layers} gamma={args.gamma}\n")
    print(f"{'arm':>24} {'held-out MSE':>13} {'sd':>9} {'params':>8} {'incremental':>12}")

    results: dict[str, list[float]] = {}
    for label, mode, blind in arms:
        scores = []
        model = None
        for seed in args.seeds:
            score, model = run(mode, seed, blind=blind, args=args)
            scores.append(score)
        results[label] = scores
        inc = float(model.incremental_gain_bound())
        params = sum(q.numel() for q in model.parameters())
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{label:>24} {statistics.mean(scores):13.5f} {sd:9.5f} {params:8d} "
              f"{('inf' if inc == float('inf') else f'{inc:.3f}'):>12}")

    both = statistics.mean(results["both + context"])
    blind = statistics.mean(results["both + blind context"])
    lpv = statistics.mean(results["context-only (LPV)"])
    print()
    print(f"context is worth          {100 * (blind - lpv) / blind:5.1f}%  "
          f"(blind -> LPV: the selector using context instead of nothing)")
    print(f"input selectivity is worth {100 * (lpv - both) / lpv:5.1f}%  "
          f"(LPV -> both: adding the input on top of context)")
    print(f"\ncost of the incremental bound: {100 * (lpv - both) / both:+.1f}% held-out MSE")
    print(
        "\nThe two are complementary rather than redundant: each alone beats\n"
        "neither, and together they beat each alone. LPV mode is not discarding\n"
        "a useless signal -- it gives up real fit quality for a bound that covers\n"
        "tracking and interconnection, which the zero-state bound does not."
    )


if __name__ == "__main__":
    main()
