# python
# file: src/neural_ssm/__init__.py
from importlib import import_module as _imp

# Re-export subpackages for discoverability
from . import ssm as ssm
from . import rens as rens
from . import static_layers as layers  # public alias

# Top-level classes and configs
from .ssm.lru import LRU, L2RU, lruz, SSMConfig, SSL, DeepSSM, PureLRUR
from .rens.ren import REN

# Common layers exposed at top-level for convenience
try:
    # TLIP is not in generic_layers; don't import it here
    from .static_layers.generic_layers import LayerConfig, GLU, MLP
except ImportError:
    pass

try:
    # If TLIP exists, it's typically in lipschitz_mlps
    from .static_layers.lipschitz_mlps import LMLP, TLIP
except ImportError:
    pass

# Export only names that are available
__all__ = [n for n in (
    "LRU", "L2RU", "lruz", "SSMConfig", "SSL", "DeepSSM", "PureLRUR",
    "REN",
    "layers", "ssm", "rens",
    "LayerConfig", "GLU", "MLP", "LMLP", "TLIP",
) if n in globals()]

__version__ = "0.1.0"

def __getattr__(name):
    # Optional lazy/compat shims; keep internals movable
    redirects = {
        "layers": "neural_ssm.static_layers",
    }
    if name in redirects:
        return _imp(redirects[name])
    raise AttributeError(f"module neural_ssm has no attribute {name!r}")

del _imp
