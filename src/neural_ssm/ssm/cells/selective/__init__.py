"""Current input/context-selective cells with a common quadratic storage."""

from ....utils.runtime import (
    normalize_to_3d as _normalize_to_3d,  # noqa: F401 -- compatibility export
)
from ..common import (
    _diag_scan as _diag_scan,
    _selector_input as _selector_input,
    spectral_norm_2x2_a_b_c as spectral_norm_2x2_a_b_c,
    spectral_norm_2x2_abcd as spectral_norm_2x2_abcd,
)
from .tv import RobustMambaDiagSSM
from .tvc import RobustMambaDiagLTI

__all__ = ["RobustMambaDiagSSM", "RobustMambaDiagLTI"]
