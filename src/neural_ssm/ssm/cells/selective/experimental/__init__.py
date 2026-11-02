"""Compatibility namespace; retained expert implementations live in ssm.experimental."""

import sys as _sys

from .... import experimental as _experts
from ....experimental import Block2x2SelectiveBCDExpertsL2SSM, ExpertSelectiveTimeVaryingSSM

__all__ = ["ExpertSelectiveTimeVaryingSSM", "Block2x2SelectiveBCDExpertsL2SSM"]

for _name in ("dense_experts", "block_experts"):
    _sys.modules[f"{__name__}.{_name}"] = _experts
del _name, _sys, _experts
