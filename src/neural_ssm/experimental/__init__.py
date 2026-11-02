"""Compatibility namespace for the remaining selective expert prototypes.

Canonical implementations live in ``ssm.experimental``.
This namespace is not imported by the supported package API.
"""
from ..ssm.experimental import ExpertSelectiveTimeVaryingSSM, Block2x2SelectiveBCDExpertsL2SSM

__all__ = ["ExpertSelectiveTimeVaryingSSM", "Block2x2SelectiveBCDExpertsL2SSM"]
