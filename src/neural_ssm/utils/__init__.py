"""Two shared utility modules: scan kernels and runtime state/caching."""
import sys as _sys

from . import runtime

# Earlier small modules resolve to the same implementations, without extra files
# or forwarding calls on the model's runtime path.
for _name in ("cache", "state", "tensors"):
    _sys.modules[f"{__name__}.{_name}"] = runtime
    globals()[_name] = runtime
del _name, _sys
