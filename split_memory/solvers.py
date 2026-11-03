"""Explicit finite-sweep autograd and causally stopped adaptive Jacobi."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


class ConvergenceError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"Jacobi did not meet tolerance: bound={diagnostics['state_error_bound']:.6g}, "
                         f"tolerance={diagnostics['tolerance']:.6g}, sweeps={diagnostics['sweeps']}")


def recurrence(c, u, weight, initial):
    predecessor = torch.cat((initial[:, None], c[:, :-1]), 1)
    return torch.tanh(F.linear(predecessor, weight) + u)


def residual(c, u, weight, initial, valid):
    proposal = recurrence(c, u, weight, initial)
    return torch.where(valid[..., None], (c - proposal).abs(), 0).amax()


def fixed_jacobi(u, weight, initial, valid, sweeps):
    """Pure, compilable tensor kernel; no device synchronization inside the loop."""
    c = torch.zeros_like(u)
    for _ in range(sweeps):
        c = torch.where(valid[..., None], recurrence(c, u, weight, initial), 0)
    return c


def sequential_reader(u, weight, initial, valid):
    current, values = initial, []
    for index in range(u.shape[1]):
        proposal = torch.tanh(F.linear(current, weight) + u[:, index])
        current = torch.where(valid[:, index, None], proposal, current)
        values.append(current)
    return torch.stack(values, 1) if values else u


@dataclass
class SolveResult:
    states: torch.Tensor
    sweeps: int
    verification_passes: int
    measured_residual: torch.Tensor
    state_error_bound: torch.Tensor
    converged: bool


def adaptive_jacobi(u, weight, initial, valid, *, rho, tolerance, max_sweeps):
    """Freeze certified *causal prefixes*, so future inputs cannot select past iterates.

    A residual evaluation certifies the OLD current iterate. Its proposal is reused
    as the next sweep for positions not frozen. All predecessors of a frozen token
    freeze with it; their final residuals cannot change in later sweeps. Stopping
    decisions are discrete and synchronize once per iteration.
    """
    c = torch.zeros_like(u)
    frozen = ~valid
    threshold = tolerance * (1 - rho)
    for sweeps in range(max_sweeps + 1):
        proposal = recurrence(c, u, weight, initial)
        errors = torch.where(valid[..., None], (c - proposal).abs(), 0).amax(-1)
        measured = errors.amax()
        bound = measured / (1 - rho)
        if sweeps == 0:
            bound = bound.clamp_max(1)  # zero guesses: E0 <= 1
        # Inclusive prefix condition is per sample, and cannot depend on future tokens.
        frozen = frozen | ((errors.cummax(1).values <= threshold) & valid)
        if (sweeps == 0 and tolerance >= 1) or bool(frozen.all()):
            return SolveResult(c, sweeps, 1, measured, bound, True)
        if sweeps == max_sweeps:
            return SolveResult(c, sweeps, 1, measured, bound, bool(bound <= tolerance))
        c = torch.where(frozen[..., None], c, proposal)
    raise AssertionError("Unreachable")
