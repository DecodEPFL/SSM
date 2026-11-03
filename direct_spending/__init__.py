"""Exact direct-spending models. Lazy exports keep the existing Tk UI lightweight."""


def __getattr__(name):
    if name in ("detach_state", "state_to"):
        from .state import detach_state, state_to
        return {"detach_state": detach_state, "state_to": state_to}[name]
    if name == "SpendingConfig":
        from .config import SpendingConfig
        return SpendingConfig
    if name in ("DirectSpendingModel", "SpendingState"):
        from .model import DirectSpendingModel, SpendingState
        return {"DirectSpendingModel": DirectSpendingModel, "SpendingState": SpendingState}[name]
    if name in ("StorageLayer", "unit_ball_head"):
        from .storage import StorageLayer, unit_ball_head
        return {"StorageLayer": StorageLayer, "unit_ball_head": unit_ball_head}[name]
    raise AttributeError(name)


__all__ = ["SpendingConfig", "DirectSpendingModel", "SpendingState", "StorageLayer", "unit_ball_head",
           "detach_state", "state_to"]
