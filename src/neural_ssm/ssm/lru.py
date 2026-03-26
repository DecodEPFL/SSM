# python
# file: src/neural_ssm/ssm/lru.py
# Compatibility shim — this file has been split into:
#   lti_cells.py  (LTI cell implementations)
#   selective_cells.py  (time-varying/selective cells)
#   layers.py     (SSMConfig, SSL, DeepSSM, PureLRUR, SimpleRNN)
from .lti_cells import (
    LRU,
    L2RU,
    lruz,
    L2BoundedLTICell,
    Block2x2DenseL2SSM,
    _normalize_to_3d,
    _scan_diag_linear,
    lru_forward_loop,
    _complex_real_transform_blocks,
    _CONTRACTION_EPS,
)
from .selective_cells import RobustMambaDiagSSM, RobustMambaDiagLTI
from .experimental import ExpertSelectiveTimeVaryingSSM, Block2x2SelectiveBCDExpertsL2SSM
from .layers import SSMConfig, SSL, DeepSSM, PureLRUR, SimpleRNN

__all__ = [
    "LRU", "L2RU", "lruz", "L2BoundedLTICell", "Block2x2DenseL2SSM",
    "RobustMambaDiagSSM", "RobustMambaDiagLTI",
    "ExpertSelectiveTimeVaryingSSM", "Block2x2SelectiveBCDExpertsL2SSM",
    "SSMConfig", "SSL", "DeepSSM", "PureLRUR", "SimpleRNN",
]
