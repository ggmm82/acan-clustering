"""
ACAN-LM (`acan_lm`) — Algebraic Concept Attention Language Model (v8.0.0)
Hugging Face compatible Small Language Model package.
"""

from .configuration_acan_lm import ACANLMConfig
from .modeling_acan_lm import ACANForCausalLM, ACANConceptAttentionLayer

__version__ = "8.0.0"
__all__ = ["ACANLMConfig", "ACANForCausalLM", "ACANConceptAttentionLayer"]
