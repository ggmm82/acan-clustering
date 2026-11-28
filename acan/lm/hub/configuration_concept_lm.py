from transformers import PretrainedConfig


class ConceptLMConfig(PretrainedConfig):
    model_type = "concept-lm"

    def __init__(self, vocab_size=32000, eos_token_id=0, pad_token_id=None, network=None, stop_token_ids=None, **kwargs):
        self.vocab_size = vocab_size
        self.network = network or {}
        self.stop_token_ids = list(stop_token_ids or [])
        super().__init__(eos_token_id=eos_token_id, pad_token_id=eos_token_id if pad_token_id is None else pad_token_id, **kwargs)
