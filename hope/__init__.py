"""hope-tensorflow: a TF/Keras implementation of HOPE (Behrouz et al., NeurIPS 2025)."""

from hope.layers import HopeAttention, SelfModifyingLayer
from hope.memory import AssociativeMemory, CMSBank, ContinuumMemorySystem
from hope.model import HOPE

__version__ = "0.0.1"
__all__ = [
    "AssociativeMemory",
    "CMSBank",
    "ContinuumMemorySystem",
    "HOPE",
    "HopeAttention",
    "SelfModifyingLayer",
    "__version__",
]
