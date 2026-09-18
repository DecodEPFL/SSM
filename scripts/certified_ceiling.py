"""Certified performance ceiling for a scalar plant family.

Answers, with no learning at all, the questions ``docs/adaptive_ssm_theory.md``
defers to experiment:

1. What does the ``gamma_Q`` budget cost in closed-loop performance versus an
   uncertified controller?  ("cost of the certificate")
2. How much can *any* context signal possibly buy under that budget -- the gap
   between one fixed certified controller and a per-plant oracle-scheduled
   certified controller?  ("headroom for context")
3. How do both scale with ``gamma_Q``?

Run this before building a learning pipeline against a plant family: if the
headroom is small, no learned context can demonstrate value on that family at
that budget, and a null result will be uninterpretable.

Plant family (theory doc eq. 17-18), with ``r = 0`` and baseline ``u = -0.5 x + v``::

    x_{t+1} = alpha x_t + b v_t + d_t,   alpha = a - 0.5 b,   e_t = x_t

The controller ``Q : e -> v`` is LTI of order ``n_q``, certified by
``||Q||_inf <= gamma_Q`` (the l2-induced gain of an LTI map is its H-infinity
norm), giving::

    T_{d->x}(z) = 1 / (z - alpha - b Q(z)),   Q(z) = D + sum_k c_k / (z - p_k)

Performance is ``||T_{d->x}||_inf``, reported worst-case and mean over the box.

Subcommands
-----------
``ceiling``        sweep gamma_Q; fixed vs oracle-scheduled vs uncertified
``bound-tightness``  compare the switching-safe g_R bound against the coupled one
``family-design``  headroom for several candidate families at one budget

Examples
--------
    python scripts/certified_ceiling.py ceiling
    python scripts/certified_ceiling.py ceiling --a -0.05 1.05 --gammas 0.2 0.4375
    python scripts/certified_ceiling.py bound-tightness
    python scripts/certified_ceiling.py family-design --gamma 0.2
"""

from __future__ import annotations

import argparse
import itertools
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

# Real coefficients => the frequency response is conjugate-symmetric, so the
# upper half of the unit circle determines the H-infinity norm.
GRID = np.exp(1j * np.linspace(0.0, np.pi, 241))

Plant = tuple[float, float]  # (alpha, b)


# ---------------------------------------------------------------------------
# Controller parameterisation
#
# Q(z) = D + sum_k c_k / (z - p_k), with poles p_k = tanh(theta_k) in (-1, 1).
# Q is *linear* in (c, D), so the gain cap is an exact rescaling rather than a
# constrained solve -- the same trick the certified layers use.
# ---------------------------------------------------------------------------
def unpack(theta: np.ndarray, n_q: int) -> tuple[np.ndarray, np.ndarray, float]:
    if n_q == 0:
        return np.zeros(0), np.zeros(0), float(theta[0])
    return np.tanh(theta[:n_q]), theta[n_q : 2 * n_q], float(theta[2 * n_q])


def q_response(p: np.ndarray, c: np.ndarray, d: float) -> np.ndarray:
    """``Q(e^{jw})`` on the frequency grid."""
    out = np.full(GRID.shape, complex(d))
    for pk, ck in zip(p, c):
        out += ck / (GRID - pk)
    return out


def apply_gain_cap(
    p: np.ndarray, c: np.ndarray, d: float, gamma_q: float
) -> tuple[np.ndarray, float]:
    """Rescale ``(c, d)`` so ``||Q||_inf <= gamma_q``.  Only ever shrinks."""
    if not np.isfinite(gamma_q):
        return c, d
    norm = float(np.max(np.abs(q_response(p, c, d))))
    if norm <= gamma_q or norm < 1e-12:
        return c, d
    s = gamma_q / norm
    return c * s, d * s


def closed_loop_matrix(alpha: float, b: float, p: np.ndarray, c: np.ndarray, d: float):
    n_q = len(p)
    A = np.zeros((1 + n_q, 1 + n_q))
    A[0, 0] = alpha + b * d
    if n_q:
        A[0, 1:] = b * c
        A[1:, 0] = 1.0
        A[1:, 1:] = np.diag(p)
    return A


def closed_loop_gain(
    alpha: float,
    b: float,
    p: np.ndarray,
    c: np.ndarray,
    d: float,
    q_w: np.ndarray | None = None,
) -> float:
    """``||T_{d->x}||_inf``, or ``inf`` when the closed loop is unstable.

    ``q_w`` lets a caller reuse one ``Q(e^{jw})`` evaluation across every plant
    in a family, which dominates the runtime of the sweeps below.
    """
    if np.max(np.abs(np.linalg.eigvals(closed_loop_matrix(alpha, b, p, c, d)))) >= 1.0 - 1e-9:
        return np.inf
    if q_w is None:
        q_w = q_response(p, c, d)
    return float(np.max(1.0 / np.abs(GRID - alpha - b * q_w)))


# ---------------------------------------------------------------------------
# Plant families
# ---------------------------------------------------------------------------
def make_family(
    a_range: tuple[float, float] = (0.5, 1.05),
    b_range: tuple[float, float] = (0.8, 1.2),
    n: int = 7,
) -> list[Plant]:
    """Grid the ``(a, b)`` box and return the induced ``(alpha, b)`` pairs."""
    return [
        (a - 0.5 * b, b)
        for a, b in itertools.product(
            np.linspace(*a_range, n), np.linspace(*b_range, n)
        )
    ]


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def _objective(n_q: int, gamma_q: float, reduce_: Callable, plants: Sequence[Plant]):
    def cost(theta: np.ndarray) -> float:
        p, c, d = unpack(theta, n_q)
        c, d = apply_gain_cap(p, c, d, gamma_q)
        q_w = q_response(p, c, d)
        gains = [closed_loop_gain(al, b, p, c, d, q_w) for al, b in plants]
        if not np.all(np.isfinite(gains)):
            return 1e6  # destabilising: push the search away
        return float(reduce_(gains))

    return cost


def optimise(
    n_q: int,
    gamma_q: float,
    reduce_: Callable,
    plants: Sequence[Plant],
    restarts: int = 12,
    seed: int = 0,
) -> tuple[float, np.ndarray | None]:
    """Minimise ``reduce_(closed-loop gain)`` over certified controllers.

    Nelder-Mead from random restarts.  The objective is non-smooth (a max over
    frequency, and a max over plants when ``reduce_`` is ``np.max``), so a
    derivative-free method with restarts is the pragmatic choice; the parameter
    space is at most five-dimensional for ``n_q = 2``.
    """
    rng = np.random.default_rng(seed)
    cost = _objective(n_q, gamma_q, reduce_, plants)
    dim = 1 if n_q == 0 else 2 * n_q + 1
    best, best_x = np.inf, None
    for _ in range(restarts):
        res = minimize(
            cost,
            rng.normal(scale=0.5, size=dim),
            method="Nelder-Mead",
            options={"maxiter": 1500, "xatol": 1e-7, "fatol": 1e-9},
        )
        if res.fun < best:
            best, best_x = float(res.fun), res.x
    return best, best_x


def evaluate(
    theta: np.ndarray, n_q: int, gamma_q: float, plants: Sequence[Plant]
) -> np.ndarray:
    p, c, d = unpack(theta, n_q)
    c, d = apply_gain_cap(p, c, d, gamma_q)
    q_w = q_response(p, c, d)
    return np.array([closed_loop_gain(al, b, p, c, d, q_w) for al, b in plants])


def oracle_scheduled(
    n_q: int, gamma_q: float, plants: Sequence[Plant], restarts: int = 6, seed: int = 0
) -> np.ndarray:
    """Per-plant optimal certified controller: the ceiling for any context."""
    return np.array(
        [optimise(n_q, gamma_q, np.max, [pl], restarts=restarts, seed=seed)[0]
         for pl in plants]
    )


def headroom(
    plants: Sequence[Plant], gamma_q: float, n_q: int = 2, seed: int = 0
) -> dict[str, float]:
    """Fixed vs oracle-scheduled certified controller on one family."""
    _, x_worst = optimise(n_q, gamma_q, np.max, plants, seed=seed)
    worst_fixed = float(evaluate(x_worst, n_q, gamma_q, plants).max())
    _, x_mean = optimise(n_q, gamma_q, np.mean, plants, seed=seed)
    mean_fixed = float(evaluate(x_mean, n_q, gamma_q, plants).mean())
    oracle = oracle_scheduled(n_q, gamma_q, plants, seed=seed)
    return {
        "worst_fixed": worst_fixed,
        "mean_fixed": mean_fixed,
        "worst_oracle": float(oracle.max()),
        "mean_oracle": float(oracle.mean()),
        "worst_headroom": worst_fixed - float(oracle.max()),
        "mean_headroom": mean_fixed - float(oracle.mean()),
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_ceiling(args) -> None:
    plants = make_family(tuple(args.a), tuple(args.b), args.n)
    alphas = [al for al, _ in plants]
    print(
        f"family: {len(plants)} grid points, a in [{args.a[0]}, {args.a[1]}], "
        f"b in [{args.b[0]}, {args.b[1]}], alpha in "
        f"[{min(alphas):.3f}, {max(alphas):.3f}]"
    )
    base = np.array(
        [closed_loop_gain(al, b, np.zeros(0), np.zeros(0), 0.0) for al, b in plants]
    )
    print(f"baseline (Q = 0):  worst {base.max():.3f}   mean {base.mean():.3f}")
    print(f"controller order n_q = {args.order}\n")

    header = (
        f"{'gamma_Q':>8} | {'fixed Q':>18} | {'oracle-scheduled Q':>18} | "
        f"{'context headroom':>18}"
    )
    print(header)
    print(
        f"{'':>8} | {'worst':>9}{'mean':>9} | {'worst':>9}{'mean':>9} | "
        f"{'worst':>9}{'mean':>9}"
    )
    print("-" * len(header))

    for gq in args.gammas:
        h = headroom(plants, gq, n_q=args.order, seed=args.seed)
        tag = "inf" if not np.isfinite(gq) else f"{gq:.4f}"
        rel = 100.0 * h["mean_headroom"] / h["mean_fixed"] if h["mean_fixed"] else 0.0
        print(
            f"{tag:>8} | {h['worst_fixed']:9.3f}{h['mean_fixed']:9.3f} | "
            f"{h['worst_oracle']:9.3f}{h['mean_oracle']:9.3f} | "
            f"{h['worst_headroom']:9.3f}{h['mean_headroom']:9.3f}   ({rel:.1f}%)"
        )

    print(
        "\n'context headroom' = best FIXED certified Q minus PER-PLANT oracle "
        "certified Q.\nThat is the most a perfect context estimator could win. "
        "Near zero means no\nlearned context can demonstrate value on this family "
        "at that budget."
    )


def cmd_bound_tightness(args) -> None:
    """The g_R bound maxes b and |alpha| independently, but they are coupled."""
    grid = list(
        itertools.product(np.linspace(*args.a, 111), np.linspace(*args.b, 111))
    )
    coupled = max(b / (1.0 - abs(a - 0.5 * b)) for a, b in grid)
    decoupled = args.b[1] / (1.0 - max(abs(a - 0.5 * b) for a, b in grid))
    arg = max(grid, key=lambda ab: ab[1] / (1.0 - abs(ab[0] - 0.5 * ab[1])))

    print("g_R: switching-safe (decoupled) vs frozen-plant (coupled)\n")
    print(
        f"  max_b / (1 - max|alpha|)  : {decoupled:.4f}   -> gamma_Q < {1/decoupled:.4f}"
    )
    print(
        f"  coupled max over the box  : {coupled:.4f}   -> gamma_Q < {1/coupled:.4f}"
        f"   (at a={arg[0]:.3f}, b={arg[1]:.3f})"
    )
    print(f"\n  budget gained by a dwell-time assumption: {(decoupled/coupled - 1)*100:.0f}%")
    print(
        "\nThe decoupled bound is CORRECT for arbitrary switching: the loop can\n"
        "sit at high |alpha| and then meet high b.  The coupled bound needs the\n"
        "plant held fixed, or switching slow enough to re-establish the envelope."
    )


def cmd_family_design(args) -> None:
    """Headroom for candidate families at one budget.

    The section-8 family shows near-zero headroom because the required
    correction has the same sign across the whole box and saturates the budget.
    A family whose required correction changes sign leaves a fixed controller
    nothing safe to do, so context earns its keep at a permitted budget.
    """
    candidates = [
        ("doc sec.8:  a[0.50, 1.05]", (0.50, 1.05), (0.8, 1.2)),
        ("sign-symmetric alpha:  a[-0.05, 1.05]", (-0.05, 1.05), (0.8, 1.2)),
        ("wider sign-symmetric:  a[-0.40, 1.05]", (-0.40, 1.05), (0.8, 1.2)),
        ("actuator reversal:  b[-1.2, 1.2]", (0.50, 1.05), (-1.2, 1.2)),
    ]
    print(f"gamma_Q = {args.gamma}, controller order {args.order}")
    print("headroom = fixed minus oracle-scheduled (worst / mean)\n")
    for label, a_range, b_range in candidates:
        plants = make_family(a_range, b_range, args.n)
        base = np.array(
            [closed_loop_gain(al, b, np.zeros(0), np.zeros(0), 0.0) for al, b in plants]
        )
        h = headroom(plants, args.gamma, n_q=args.order, seed=args.seed)
        rel = 100.0 * h["mean_headroom"] / h["mean_fixed"] if h["mean_fixed"] else 0.0
        print(
            f"{label:<40} baseline worst {base.max():7.3f} | "
            f"fixed {h['worst_fixed']:7.3f}/{h['mean_fixed']:6.3f} | "
            f"oracle {h['worst_oracle']:7.3f}/{h['mean_oracle']:6.3f} | "
            f"headroom {h['worst_headroom']:6.3f}/{h['mean_headroom']:6.3f} ({rel:.1f}%)"
        )
    print(
        "\nActuator reversal makes the fixed baseline u = -0.5x destabilising, so\n"
        "it violates the assumption that the baseline interconnection is\n"
        "internally stable uniformly over the family.  Widening alpha does not."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--a", nargs=2, type=float, default=[0.5, 1.05],
                        metavar=("LO", "HI"), help="range of the plant pole a")
    common.add_argument("--b", nargs=2, type=float, default=[0.8, 1.2],
                        metavar=("LO", "HI"), help="range of the input gain b")
    common.add_argument("--n", type=int, default=7,
                        help="grid points per axis (default: 7)")
    common.add_argument("--order", type=int, default=2,
                        help="controller order n_q (default: 2)")
    common.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("ceiling", parents=[common],
                       help="sweep gamma_Q: fixed vs oracle-scheduled")
    c.add_argument("--gammas", nargs="+", type=float,
                   default=[0.1, 0.2, 7 / 24, 0.5, 1.0, 2.0, np.inf])
    c.set_defaults(func=cmd_ceiling)

    b = sub.add_parser("bound-tightness", parents=[common],
                       help="switching-safe vs frozen-plant g_R")
    b.set_defaults(func=cmd_bound_tightness)

    f = sub.add_parser("family-design", parents=[common],
                       help="headroom for candidate plant families")
    f.add_argument("--gamma", type=float, default=0.2)
    f.set_defaults(func=cmd_family_design)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
