"""Cell families: fixed LTI, selective, and explicitly retained legacy models."""

from ...utils.runtime import (
    normalize_to_3d as _normalize_to_3d,  # noqa: F401 -- compatibility export
)
from .common import (
    _CONTRACTION_EPS as _CONTRACTION_EPS,
    _complex_real_transform_blocks as _complex_real_transform_blocks,
    _conv_diag_linear as _conv_diag_linear,
    _scan_diag_linear as _scan_diag_linear,
    lru_forward_loop as lru_forward_loop,
)
from .legacy import L2RU, L2BoundedLTICell, lruz
from .lti import LRU, Block2x2DenseL2SSM, DefectL2SSM

__all__ = ["LRU", "Block2x2DenseL2SSM", "DefectL2SSM", "L2RU", "lruz", "L2BoundedLTICell"]
