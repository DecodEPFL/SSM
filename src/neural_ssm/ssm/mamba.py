# python
# file: src/neural_ssm/ssm/mamba.py
# Compatibility shim — this file has been merged into selective_cells.py
from .selective_cells import (
    RobustMambaDiagSSM,
    _normalize_to_3d,
    _diag_scan,
    spectral_norm_2x2_a_b_c,
)
from ..static_layers.lipschitz_mlps import L2BoundedLinearExact

__all__ = [
    "RobustMambaDiagSSM",
    "L2BoundedLinearExact",
    "_normalize_to_3d",
    "_diag_scan",
    "spectral_norm_2x2_a_b_c",
]
