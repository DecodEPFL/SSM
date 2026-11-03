"""Serializable configuration shared by models, benchmark parser, and Tk controls."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

CORES = ("static_mlp", "gru", "lstm", "tcn", "transformer")
MODEL_SPECS = {"direct_spending": {"kind": "direct_spending"},
               **{"spending_" + core: {"kind": "direct_spending", "spending_core": core}
                  for core in CORES}}


@dataclass(frozen=True)
class SpendingConfig:
    core: str = "gru"
    width: int = 16
    layers: int = 2
    gamma: float = 1.0
    gamma_mode: str = "fixed"
    gamma_min: float = 1e-4
    gamma_max: float = 100.0
    beta_mode: str = "zero"
    fixed_beta: float = 0.0
    beta_initial: float = 0.1
    beta_width: int = 8
    initial_features: tuple = ()
    tcn_kernel: int = 3
    tcn_dilations: tuple = (1, 2, 4)
    heads: int = 2
    dropout: float = 0.0
    storage_impl: str = "scan"
    energy_precision: str = "auto"
    diagnostics: bool = False
    context_features: int = 0
    readout: str = "spending"
    linear_init: str = "zero"

    def __post_init__(self):
        object.__setattr__(self, "initial_features", tuple(self.initial_features))
        object.__setattr__(self, "tcn_dilations", tuple(self.tcn_dilations))
        if self.core not in CORES:
            raise ValueError(f"Unknown spending core {self.core!r}")
        if self.readout not in ("spending", "projected"):
            raise ValueError("Readout must be spending or projected")
        if self.linear_init not in ("zero", "identity"):
            raise ValueError("Linear path initialization must be zero or identity")
        if not isinstance(self.context_features, int) or isinstance(self.context_features, bool) or self.context_features < 0:
            raise ValueError("context_features must be a nonnegative integer")
        if self.beta_mode not in ("zero", "fixed", "learned"):
            raise ValueError("beta mode must be zero, fixed, or learned")
        if self.gamma_mode not in ("fixed", "learned"):
            raise ValueError("gamma mode must be fixed or learned")
        if self.gamma_mode == "learned":
            if not (math.isfinite(self.gamma_min) and math.isfinite(self.gamma_max)
                    and 0 < self.gamma_min < self.gamma < self.gamma_max):
                raise ValueError("Learned gamma needs finite 0 < gamma_min < initial gamma < gamma_max")
        if self.storage_impl not in ("scan", "reference"):
            raise ValueError("storage implementation must be scan or reference")
        if self.energy_precision not in ("auto", "float32", "float64"):
            raise ValueError("energy precision must be auto, float32, or float64")
        for name in ("width", "layers", "beta_width", "tcn_kernel", "heads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("gamma", "fixed_beta", "beta_initial", "dropout"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.gamma < 0 or self.fixed_beta < 0 or self.beta_initial <= 0:
            raise ValueError("gamma/fixed beta must be nonnegative; learned initial beta must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not self.tcn_dilations or any(not isinstance(d, int) or d < 1 for d in self.tcn_dilations):
            raise ValueError("TCN dilations must be nonempty positive integers")
        if any(not math.isfinite(float(x)) for x in self.initial_features):
            raise ValueError("Initial context features must be finite declared constants")
        if self.core == "transformer" and (self.width % 2 or self.width % self.heads):
            raise ValueError("Transformer width must be even and divisible by heads")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    @property
    def label(self):
        certificate = "zero-bias" if self.beta_mode == "zero" else f"{self.beta_mode} initial energy"
        gain = " / learned gamma" if self.gamma_mode == "learned" else ""
        family = "Output projection + linear path" if self.readout == "projected" else "Direct spending"
        return f"{family} / {self.core} / {certificate}{gain}"
