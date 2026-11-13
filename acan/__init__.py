"""
ACAN: Algebraic Concept Attention Network (Algebraic Clustering Model v8)
=========================================================================

A White-Box Differentiable Concept Discovery and Universal Tabular Autocompletion
framework bridging Relational Generalization Trees and Competitive Slot Attention.
"""

from .core.network import ModularClusterNet, HAS_TORCH
from .core.model import AlgebraicClusterModel, GraphClusterModel
from .core.learner import AlgebraicClusterLearner, ACANPredictor

__version__ = "8.5.0"
__all__ = [
    "ModularClusterNet",
    "AlgebraicClusterModel",
    "GraphClusterModel",
    "AlgebraicClusterLearner",
    "ACANPredictor",
    "HAS_TORCH",
]
