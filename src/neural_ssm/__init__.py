"""Public neural-ssm API; individual implementations live in focused subpackages."""
from . import ssm, rens, static_layers as layers
from .ssm import (
    SSMConfig,
    SSMConfigDict,
    SSL,
    DeepSSM,
    PureLRUR,
    SimpleRNN,
    LRU,
    Block2x2DenseL2SSM,
    DefectL2SSM,
    RobustMambaDiagSSM,
    RobustMambaDiagLTI,
    ContextualDeepSSM,
    timewise_matrix_vector_product,
    L2RU,
    lruz,
    L2BoundedLTICell,
)
from .rens import REN
from .static_layers import LayerConfig, GLU, MLP, LMLP, TLIP

__all__ = [
    "SSMConfig", "SSMConfigDict", "SSL", "DeepSSM", "PureLRUR", "SimpleRNN",
    "LRU", "Block2x2DenseL2SSM", "DefectL2SSM",
    "RobustMambaDiagSSM", "RobustMambaDiagLTI",
    "ContextualDeepSSM", "timewise_matrix_vector_product",
    "L2RU", "lruz", "L2BoundedLTICell", "REN",
    "LayerConfig", "GLU", "MLP", "LMLP", "TLIP", "ssm", "rens", "layers",
]

__version__ = "0.43.0"
