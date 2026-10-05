"""SSM stack, context wrapper, and current/legacy cell families."""
import sys as _sys

from .config import SSMConfig, SSMConfigDict
from . import layers, contextual, cells
from .layers import SSL, DeepSSM, PureLRUR, SimpleRNN
from .cells.lti import LRU, Block2x2DenseL2SSM, DefectL2SSM, defect as _defect
from .cells.selective import RobustMambaDiagSSM, RobustMambaDiagLTI
from .cells.legacy import L2RU, lruz, L2BoundedLTICell
from .contextual import ContextualDeepSSM, timewise_matrix_vector_product
from .cells import common as _common, selective as _selective
from ..utils import runtime as _runtime, scan as _scan

__all__ = [
    "SSMConfig", "SSMConfigDict", "SSL", "DeepSSM", "PureLRUR", "SimpleRNN",
    "LRU", "Block2x2DenseL2SSM", "DefectL2SSM",
    "RobustMambaDiagSSM", "RobustMambaDiagLTI",
    "L2RU", "lruz", "L2BoundedLTICell",
    "ContextualDeepSSM", "timewise_matrix_vector_product",
]

# Old imports resolve directly to real modules. Keeping aliases in one place
# avoids forwarding files and introduces no wrappers around model execution.
for _name, _module in {
    "deep": layers,
    "block": layers,
    "baselines": layers,
    "certification": layers,
    "feedforward": layers,
    "lti_cells": cells,
    "selective_cells": _selective,
    "defect": _defect,
    "scan_utils": _scan,
    "cache_utils": _runtime,
    "state_utils": _runtime,
    "contextual.model": contextual,
    "contextual.filters": contextual,
    "contextual.ports": contextual,
    "cells._linear": _common,
    "cells.selective._common": _common,
}.items():
    _sys.modules[f"{__name__}.{_name}"] = _module
    if "." not in _name:
        globals()[_name] = _module
del _name, _module, _common, _defect, _selective, _runtime, _scan, _sys
