"""hope-tensorflow: a TF/Keras implementation of HOPE (Behrouz et al., NeurIPS 2025)."""

from hope.baseline import MiniTransformer, TransformerBlock
from hope.layers import HopeAttention, SelfModifyingLayer
from hope.memory import AssociativeMemory, CMSBank, ContinuumMemorySystem
from hope.model import HOPE
from hope.optimizers import DGD, DeepOptimizer

__version__ = "0.0.1"
__all__ = [
    "AssociativeMemory",
    "CMSBank",
    "ContinuumMemorySystem",
    "DGD",
    "DeepOptimizer",
    "HOPE",
    "HopeAttention",
    "MiniTransformer",
    "SelfModifyingLayer",
    "TransformerBlock",
    "__version__",
]
