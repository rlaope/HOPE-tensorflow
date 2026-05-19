"""hope-tensorflow: a TF/Keras implementation of HOPE (Behrouz et al., NeurIPS 2025)."""

from hope.layers import SelfModifyingLayer
from hope.memory import AssociativeMemory, CMSBank, ContinuumMemorySystem

__version__ = "0.0.1"
__all__ = [
    "AssociativeMemory",
    "CMSBank",
    "ContinuumMemorySystem",
    "SelfModifyingLayer",
    "__version__",
]
