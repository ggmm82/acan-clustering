"""All-layer ACAN profile: unchanged attention equations, explicit cache position.

Hybrid models infer sequence length from their remaining dense cache. When every
attention layer is ACAN, this adapter supplies that count from the causal concept
state instead. No dense KV tensors are created as a substitute.
"""
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache
from .attention import ACANConceptAttentionLayer


class ConceptCache(DynamicCache):
    def __init__(self,config):
        super().__init__(config=config)
        self.concept_seen=0

    def get_seq_length(self,layer_idx=0):
        return self.concept_seen


class FullConceptAttention(ACANConceptAttentionLayer):
    def forward(self,*args,**kwargs):
        cache=kwargs.get('past_key_values')
        if cache is not None and not isinstance(cache,ConceptCache):
            raise ValueError('Il profilo interamente ACAN richiede ConceptCache.')
        result=super().forward(*args,**kwargs)
        if cache is not None and self.cache is not None:
            cache.concept_seen=self.cache['seen']
        return result


class FullConceptLlama(LlamaForCausalLM):
    def forward(self,input_ids=None,attention_mask=None,past_key_values=None,use_cache=None,**kwargs):
        cache_enabled=self.config.use_cache if use_cache is None else use_cache
        if cache_enabled and past_key_values is None:
            past_key_values=ConceptCache(self.config)
        return super().forward(input_ids=input_ids,attention_mask=attention_mask,
            past_key_values=past_key_values,use_cache=use_cache,**kwargs)

    def _prepare_cache_for_generation(self,generation_config,model_kwargs,generation_mode,batch_size,max_cache_length):
        if generation_config.num_beams!=1:
            raise ValueError('Il profilo ACAN supporta un solo beam.')
        if generation_config.use_cache and model_kwargs.get('past_key_values') is None:
            model_kwargs['past_key_values']=ConceptCache(self.config)
        return super()._prepare_cache_for_generation(generation_config,model_kwargs,generation_mode,batch_size,max_cache_length)
