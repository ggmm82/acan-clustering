"""Reviewed Llama/SmolLM2 graft boundary; no arbitrary Hub Python execution."""
import torch
from .attention import ACANConceptAttentionLayer, PROFILE

BASE_MODELS = ('HuggingFaceTB/SmolLM2-135M', 'HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-1.7B')


def graft(model, slots=32, window=32):
    if model.config.model_type != 'llama' or not hasattr(model, 'model'):
        raise ValueError('Reviewed graft supports SmolLM2/Llama architecture only; MiniCPM remote-code adapters are not reviewed.')
    layers = model.model.layers
    start = len(layers) // 2
    if start < 1 or len(layers) % 2:
        raise ValueError('A 50% graft requires a positive even number of layers.')
    heads, kv_heads = model.config.num_attention_heads, model.config.num_key_value_heads
    for index in range(start, len(layers)):
        old = layers[index].self_attn
        if any(getattr(old, name).bias is not None for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')):
            raise ValueError('This graft profile requires bias-free attention projections.')
        new = ACANConceptAttentionLayer(model.config, index, slots=slots, window=window).to(
            device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        with torch.no_grad():
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                weight = getattr(old, name).weight
                if name in ('k_proj', 'v_proj') and kv_heads != heads:
                    weight = weight.view(kv_heads, -1, weight.shape[-1]).repeat_interleave(heads // kv_heads, 0).reshape(model.config.hidden_size, -1)
                getattr(new, name).weight.copy_(weight)
        layers[index].self_attn = new
    return dict(profile=PROFILE, replaced_layers=list(range(start, len(layers))), total_layers=len(layers),
                slots=slots, window=window, fraction=.5)


def add_lora(model, rank=8):
    from peft import LoraConfig, TaskType, get_peft_model
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=.05,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type=TaskType.CAUSAL_LM, bias='none'))
    for module in model.modules():
        if isinstance(module, ACANConceptAttentionLayer):
            module.G.data = module.G.data.float()
            module.omega.data = module.omega.data.float()
            module.G.requires_grad_(True)
            module.omega.requires_grad_(True)
    return model


def build_from_config(config, options):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(config, attn_implementation='eager')
    report = graft(model, options['slots'], options['window'])
    return add_lora(model, options['lora_rank']), report


def load_base(path, options, device='cpu', evaluate_base=None):
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, torch_dtype=dtype, attn_implementation='eager')
    if options['sequence_length'] + 1 > model.config.max_position_embeddings:
        raise ValueError('Il blocco supera il contesto del modello base (incluso il token target finale).')
    model = model.to(device)
    baseline = evaluate_base(model) if evaluate_base is not None else None
    report = graft(model, options['slots'], options['window'])
    report['baseline_nll'] = baseline
    model = add_lora(model, options['lora_rank']).to(device)
    return model, report


def reset_streams(model):
    for module in model.modules():
        if isinstance(module, ACANConceptAttentionLayer):
            module.reset_stream()
