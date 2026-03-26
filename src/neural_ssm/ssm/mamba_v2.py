# python
# file: src/neural_ssm/ssm/mamba_v2.py
# Compatibility shim — this file has been merged into selective_cells.py
from .selective_cells import RobustMambaDiagLTI, spectral_norm_2x2_abcd

__all__ = ["RobustMambaDiagLTI", "spectral_norm_2x2_abcd"]
