"""Scanned bounded memory and a contractive nonlinear reader (lazy torch imports)."""
from .config import SplitMemoryConfig

__all__ = ["SplitMemoryConfig", "SplitMemoryContractiveRNN", "SplitMemoryState", "detach_state", "state_to"]


def __getattr__(name):
    if name == "SplitMemoryContractiveRNN":
        from .model import SplitMemoryContractiveRNN
        return SplitMemoryContractiveRNN
    if name in ("SplitMemoryState", "detach_state", "state_to"):
        from . import state
        return getattr(state, name)
    raise AttributeError(name)
