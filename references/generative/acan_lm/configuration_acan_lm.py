"""
ACAN-LM Configuration module compatible with Hugging Face Transformers (`PretrainedConfig`).
Enables seamless loading via `AutoConfig.from_pretrained(..., trust_remote_code=True)`.
"""

import json
import os
from typing import Dict, Any, Optional

try:
    from transformers import PretrainedConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    class PretrainedConfig:
        model_type = "acan_lm"
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self) -> Dict[str, Any]:
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        def save_pretrained(self, save_directory: str):
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
            cfg_path = os.path.join(pretrained_model_name_or_path, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.update(kwargs)
            return cls(**data)


class ACANLMConfig(PretrainedConfig):
    """
    Configuration class for ACAN-LM (Algebraic Concept Attention Language Model).
    Combines MiniCPM-style WSD/RMSNorm/SwiGLU efficiencies with ACAN v8
    Competitive Concept Slot Attention and Feature-Channel Relevance Gating (R).
    """
    model_type = "acan_lm"

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 192,
        intermediate_size: int = 512,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        num_concept_slots: int = 32,
        local_window_size: int = 32,
        relevance_alpha: float = 2.0,
        mass_gamma: float = 1.15,
        attention_beta: float = 8.0,
        slot_pruning_threshold: float = 0.02,
        use_group_behavior_coupling: bool = True,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        mask_token_id: int = 3,
        architectures: Optional[list] = None,
        auto_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_concept_slots = num_concept_slots
        self.local_window_size = local_window_size
        self.relevance_alpha = relevance_alpha
        self.mass_gamma = mass_gamma
        self.attention_beta = attention_beta
        self.slot_pruning_threshold = slot_pruning_threshold
        self.use_group_behavior_coupling = use_group_behavior_coupling
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.architectures = architectures or ["ACANForCausalLM"]
        self.auto_map = auto_map or {
            "AutoConfig": "configuration_acan_lm.ACANLMConfig",
            "AutoModelForCausalLM": "modeling_acan_lm.ACANForCausalLM",
        }
        super().__init__(**kwargs)
