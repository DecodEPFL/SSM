"""Supported state-space-model API.

Research prototypes, including Raven-style memory and transformer experiments,
live under :mod:`neural_ssm.experimental` and are intentionally excluded from
this stable namespace.
"""

from .lti_cells import LRU, L2RU, lruz, L2BoundedLTICell, Block2x2DenseL2SSM
from .selective_cells import RobustMambaDiagSSM, RobustMambaDiagLTI
from .layers import SSMConfig, SSL, DeepSSM, PureLRUR, SimpleRNN
from .contextual import (
    ContextualDeepSSM,
    timewise_matrix_vector_product,
)

__all__ = [
    "LRU", "L2RU", "lruz", "L2BoundedLTICell", "Block2x2DenseL2SSM",
    "RobustMambaDiagSSM", "RobustMambaDiagLTI",
    "SSMConfig", "SSL", "DeepSSM", "PureLRUR", "SimpleRNN",
    "ContextualDeepSSM", "timewise_matrix_vector_product",
]
