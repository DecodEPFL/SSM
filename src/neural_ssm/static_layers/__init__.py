"""Ordinary layers in generic_layers; bounded constructions in lipschitz_mlps."""
import sys as _sys

from . import generic_layers, lipschitz_mlps
from .generic_layers import LayerConfig, GLU, MLP
from .lipschitz_mlps import (
    L2BoundedLinearExact,
    TLIP,
    LMLP,
    L2BoundedGLU,
    L2BoundedGLUv2,
    BudgetedL2BoundedGLUv2,
    GroupSort,
    LipGroupSortBranch,
    MultiBranchLipMixer,
    cayley,
    FirstChannel,
    SandwichLin,
    SandwichFc,
)

__all__ = [
    "LayerConfig", "L2BoundedLinearExact", "MLP", "TLIP", "LMLP", "GLU",
    "L2BoundedGLU", "L2BoundedGLUv2", "BudgetedL2BoundedGLUv2",
    "GroupSort", "LipGroupSortBranch", "MultiBranchLipMixer",
    "cayley", "FirstChannel", "SandwichLin", "SandwichFc",
]

# Compatibility for the previous split, kept here rather than in forwarding
# files. Mixed old modules (glu/mlp) use this package's combined public exports.
for _name, _module in {
    "config": generic_layers,
    "linear": lipschitz_mlps,
    "sandwich": lipschitz_mlps,
    "mixers": lipschitz_mlps,
    "glu": _sys.modules[__name__],
    "mlp": _sys.modules[__name__],
}.items():
    _sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
del _name, _module, _sys
