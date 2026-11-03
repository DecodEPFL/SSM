"""Serializable architecture/solver settings; safe to import in the native UI."""
from dataclasses import asdict, dataclass
import math

MODEL_SPECS = {"split_memory": {"kind": "split_memory"}}
READERS = ("recurrent", "pointwise", "linear")
SOLVERS = ("fixed", "adaptive", "sequential")


@dataclass(frozen=True)
class SplitMemoryConfig:
    heads: int = 8
    computation_width: int = 16
    reader: str = "recurrent"
    rho: float = 0.7
    memory_radius: float = 1.0
    initial_retention: float = 0.95
    row_delta: float = 1e-4
    solver: str = "fixed"
    memory_backend: str = "scan"
    tolerance: float = 1e-4
    fixed_sweeps: int = 32
    max_sweeps: int = 64
    require_convergence: bool = True
    diagnostics: bool = False
    projection_scale: float = 0.1
    mixing_scale: float = 0.02

    def __post_init__(self):
        for key in ("heads", "computation_width", "fixed_sweeps", "max_sweeps"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("rho", "memory_radius", "initial_retention", "row_delta", "tolerance",
                    "projection_scale", "mixing_scale"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.rho < 1:
            raise ValueError("rho must satisfy 0 <= rho < 1")
        if not 0 < self.initial_retention < 1:
            raise ValueError("initial retention must be strictly between zero and one")
        if min(self.memory_radius, self.row_delta, self.tolerance, self.projection_scale) <= 0:
            raise ValueError("memory radius, row delta, tolerance and projection scale must be positive")
        if self.mixing_scale < 0:
            raise ValueError("mixing scale must be nonnegative")
        if self.reader not in READERS or self.solver not in SOLVERS:
            raise ValueError("Unknown reader or solver")
        if self.memory_backend not in ("scan", "sequential"):
            raise ValueError("Memory backend must be scan or sequential")
        if not isinstance(self.require_convergence, bool) or not isinstance(self.diagnostics, bool):
            raise ValueError("require_convergence and diagnostics must be boolean")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


def sufficient_sweeps(rho, tolerance, diameter=1.0):
    """Zero initial guesses and tanh targets have E0 <= 1, regardless of c_init."""
    if not 0 <= rho < 1 or not math.isfinite(tolerance) or tolerance <= 0 or not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Invalid contraction or error scale")
    if tolerance >= diameter:
        return 0
    if rho == 0:
        return 1
    return math.ceil((math.log(diameter) - math.log(tolerance)) / -math.log(rho))
