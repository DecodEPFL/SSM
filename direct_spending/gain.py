"""Optional learned, bounded global gain; fixed mode adds no checkpoint tensors."""
from functools import lru_cache
import math

import torch
from torch import nn


@lru_cache(maxsize=32)
def _endpoints(lower, upper, dtype):
    # Round inward on CPU once per dtype/configuration. The device computation
    # then clamps to exactly representable constants within the declared bounds.
    lo, hi = torch.tensor(lower, dtype=dtype), torch.tensor(upper, dtype=dtype)
    if lo.item() < lower:
        lo = torch.nextafter(lo, torch.tensor(float("inf"), dtype=dtype))
    if hi.item() > upper:
        hi = torch.nextafter(hi, torch.tensor(0., dtype=dtype))
    if not (torch.isfinite(lo) and torch.isfinite(hi) and 0 < lo < hi):
        raise ValueError("Gamma bounds must be positive, finite and distinct at the gain computation dtype")
    return lo.item(), hi.item()


class GainBound(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.gamma_mode == "learned":
            lo, hi = _endpoints(config.gamma_min, config.gamma_max, torch.float32)
            ratio = (config.gamma - lo) / (hi - lo)
            if not 0 < ratio < 1:
                raise ValueError("Initial gamma must lie strictly inside the representable gain bounds")
            self.logit = nn.Parameter(torch.tensor(math.log(ratio) - math.log1p(-ratio)))

    def forward(self):
        cfg = self.config
        if cfg.gamma_mode == "fixed":
            return cfg.gamma
        dtype = torch.promote_types(self.logit.dtype, torch.float32)
        lo, hi = _endpoints(cfg.gamma_min, cfg.gamma_max, dtype)
        weight = self.logit.to(dtype).sigmoid()
        # Clamp only the gain parameter's roundoff, never the reservoir budget.
        return (lo + (hi - lo) * weight).clamp(min=lo, max=hi)
