"""Explicit hybrid and dense-LoRA control arms for matched experiments."""
import torch
from .grafting import load_base,add_lora


def load_model(path,options,device='cpu',evaluate_base=None):
    if options.get('architecture','hybrid')=='hybrid' and options.get('graft_fraction',.5)==.5:
        model,report=load_base(path,options,device,evaluate_base)
        return shared_adapter(model,report,options)
    from transformers import AutoModelForCausalLM
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
    all_concept=options.get('architecture','hybrid')=='hybrid'
    if all_concept:
        from .full_attention import FullConceptLlama
        model_class=FullConceptLlama
    else:
        model_class=AutoModelForCausalLM
    model=model_class.from_pretrained(path,local_files_only=True,trust_remote_code=False,
        use_safetensors=True,torch_dtype=dtype,attn_implementation='eager').to(device)
    if options['sequence_length']+1>model.config.max_position_embeddings:
        raise ValueError('Blocco oltre il contesto del modello.')
    baseline=evaluate_base(model) if evaluate_base else None
    if all_concept:
        from .full_attention import FullConceptAttention
        heads,kv_heads=model.config.num_attention_heads,model.config.num_key_value_heads
        for index,layer in enumerate(model.model.layers):
            old=layer.self_attn
            new=FullConceptAttention(model.config,index,slots=options['slots'],window=options['window']).to(device=device,dtype=old.q_proj.weight.dtype)
            with torch.no_grad():
                for name in ('q_proj','k_proj','v_proj','o_proj'):
                    if getattr(old,name).bias is not None:raise ValueError('Proiezioni con bias non supportate.')
                    weight=getattr(old,name).weight
                    if name in ('k_proj','v_proj') and kv_heads!=heads:
                        weight=weight.view(kv_heads,-1,weight.shape[-1]).repeat_interleave(heads//kv_heads,0).reshape(model.config.hidden_size,-1)
                    getattr(new,name).weight.copy_(weight)
            layer.self_attn=new
        report=dict(profile='acan-all-layers-causal-v1',replaced_layers=list(range(len(model.model.layers))),
                    total_layers=len(model.model.layers),slots=options['slots'],window=options['window'],fraction=1.,baseline_nll=baseline)
        return shared_adapter(add_lora(model,options['lora_rank']).to(device),report,options)
    report=dict(profile='dense-lora-control-v1',replaced_layers=[],total_layers=len(model.model.layers),
                slots=0,window=0,fraction=0.,baseline_nll=baseline)
    return add_lora(model,options['lora_rank']).to(device),report


def inference_contract():
    import hashlib
    from pathlib import Path
    root=Path(__file__).parent
    return hashlib.sha256(b''.join(path.read_bytes() for path in (root/'research_model.py',root/'full_attention.py',root/'shared_attention.py',root.parent/'algebra.py',root/'cp_attention.py',root.parent/'core'/'variational.py',root/'concept_attention.py',root.parent/'concept_algebra.py'))).hexdigest()


def check_inference_contract(options,state):
    recorded=state.get('model_contract')
    from .compatibility import LEGACY_INFERENCE, PRE_JOINT_INFERENCE, PRE_V9_INFERENCE, PRE_GRAPH_INFERENCE
    legacy = ('routing_profile' not in options and recorded == LEGACY_INFERENCE) or (options.get('routing_profile') in ('shared-algebra-v1','legacy-causal-v1') and recorded == 'c823eeaae94de3b650709710cc2f245328a9a693d7b640af3096d6c8ddf6ba8b')
    legacy = legacy or (options.get('routing_profile') in ('shared-algebra-v1','legacy-causal-v1','cp-routing-v1') and recorded == PRE_JOINT_INFERENCE)
    # Earlier profiles are numerically unchanged by the addition of cp-v9.
    legacy = legacy or (options.get('routing_profile') not in ('cp-v9','concept-graph-v1') and recorded == PRE_V9_INFERENCE)
    legacy = legacy or (options.get('routing_profile')!='concept-graph-v1' and recorded == PRE_GRAPH_INFERENCE)
    if recorded is not None and recorded!=inference_contract() and not legacy:
        raise ValueError('Modello di un’altra versione: usare l’immagine worker che lo ha creato.')
    if options.get('graft_fraction',.5)==1. and recorded is None:
        raise ValueError('Checkpoint interamente ACAN privo del contratto del modello.')


def shared_adapter(model,report,options):
    if options.get('routing_profile') in ('shared-algebra-v1','cp-routing-v1','cp-joint-v1','cp-v9','concept-graph-v1'):
        from .shared_attention import SharedAlgebraAttention
        if options.get("routing_profile") in ("cp-v9", "concept-graph-v1"):
            from .concept_attention import ConceptGraphAttention, ConceptSlotAttention
            SharedAlgebraAttention = ConceptGraphAttention if options["routing_profile"] == "concept-graph-v1" else ConceptSlotAttention
        elif options.get("routing_profile") == "cp-routing-v1":
            from .cp_attention import CPUnifiedAttention
            SharedAlgebraAttention = CPUnifiedAttention
        elif options.get("routing_profile") == "cp-joint-v1":
            from .cp_attention import CPJointAttention
            SharedAlgebraAttention = CPJointAttention
        from .full_attention import FullConceptAttention
        from .attention import ACANConceptAttentionLayer
        class SharedFullAttention(FullConceptAttention,SharedAlgebraAttention):
            pass
        for module in list(model.modules()):
            if isinstance(module,FullConceptAttention):
                module.__class__=SharedFullAttention
            elif isinstance(module,ACANConceptAttentionLayer):
                module.__class__=SharedAlgebraAttention
            if options.get('routing_profile')=='cp-joint-v1' and isinstance(module,ACANConceptAttentionLayer):
                module.initialize_concepts()
        if options['routing_profile'] in ('cp-v9','concept-graph-v1'):
            # Concepts are statistics of the prefix: the layer has no prototype parameters to learn.
            core=model.get_base_model() if hasattr(model,'get_base_model') else model
            frequency=core.model.rotary_emb.inv_freq.detach().clone()
            for module in model.modules():
                if isinstance(module,ACANConceptAttentionLayer):
                    module.rope_frequency=frequency
                    module.G.requires_grad_(False);module.omega.requires_grad_(False)
        report=dict(report,algebra=options['routing_profile'])
        if options['routing_profile'] in ('cp-v9','concept-graph-v1'):
            report['cp_definition']='explained-variance-v1'
        if options['routing_profile']=='cp-joint-v1':
            report['cp_definition']='pair-supported-cp-v1'
    return model,report
