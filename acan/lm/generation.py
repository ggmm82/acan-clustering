"""Generate from immutable trained snapshots with measured whole-model cache bytes."""
import json
import math
from pathlib import Path
import time
from contextlib import nullcontext
import torch

from . import checkpoints
from .research_model import load_model as load_base,check_inference_contract
from .grafting import reset_streams
from .attention import ACANConceptAttentionLayer
from .store import Store
from .trainer import resolve_base


def generate(root, request_id):
    torch.set_num_threads(1)
    store = Store(root)
    with store.db() as db:
        row = dict(db.execute('SELECT * FROM generations WHERE id=?', (request_id,)).fetchone())
    request = json.loads(row['request'])
    job = store.job(row['job'], row['owner'])
    try:
        directory = store.directory('jobs', job['id']) / 'checkpoints' / request['checkpoint']
        manifest = checkpoints.verify(directory)
        options = job['config']
        path, base = resolve_base(store, job, options)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        state = json.loads((directory / 'state.json').read_text())
        check_inference_contract(options,state)
        device = state['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Questo checkpoint richiede il worker CUDA originale.')
        model, graft = load_base(path, options, device)
        checkpoints.restore(directory, model, expected=manifest['identity'], restore_rng=False)
        model.eval(); reset_streams(model)
        from .probes import chat_prompt, stop_tokens
        chat = request.get('chat', False)                     # a question to answer, as in the conversation data
        inputs = tokenizer(chat_prompt(request['prompt']) if chat else request['prompt'], return_tensors='pt')['input_ids'].to(device)
        if inputs.shape[1] < 1 or inputs.shape[1] + request['max_new_tokens'] > min(model.config.max_position_embeddings, 4096):
            raise ValueError('Prompt vuoto o contesto oltre il limite supportato.')
        from .inspection import Capture
        inspect = request.get('inspect_layers', False)
        if inspect and inputs.shape[1] > 256:
            raise ValueError('Ispezione limitata a prompt di 256 token.')
        capture = Capture(model, 'inference', tokenizer, request['max_new_tokens']) if inspect else None
        torch.manual_seed(request['seed'])
        if device == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        sampling = dict(do_sample=request['temperature'] > 0)
        if sampling['do_sample']:
            sampling['temperature'] = request['temperature']
        with torch.inference_mode(), capture if capture is not None else nullcontext():
            # On request the end-of-text token is ignored until the requested length: models trained
            # on short documents otherwise stop after a line.
            length = dict(min_new_tokens=request['max_new_tokens']) if request.get('full_length') and not chat else {}
            if chat:
                length['eos_token_id'] = stop_tokens(tokenizer)
            output = model.generate(inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=request['max_new_tokens'], **length, **sampling, use_cache=True, num_beams=1,
                                    pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        produced = output.sequences.shape[1] - inputs.shape[1]
        dense_bytes = 0
        for layer in output.past_key_values.layers:
            for tensor in (getattr(layer, 'keys', None), getattr(layer, 'values', None)):
                if tensor is not None:
                    dense_bytes += tensor.numel() * tensor.element_size()
        slots_bytes, explanations = 0, []
        for module in model.modules():
            if isinstance(module, ACANConceptAttentionLayer):
                slots_bytes += module.cache_bytes()
                info = module.last_info
                relevance = info['relevance'][0].mean(0)
                top = torch.topk(relevance, min(8, relevance.numel()))
                explanations.append(dict(layer=module.layer_idx, active_slots=int(info['active_slots'][0]),
                                         allocated_slots=module.k, masses=info['masses'][0].cpu().tolist(),
                                         top_latent_channels=top.indices.cpu().tolist(), relevance=top.values.cpu().tolist()))
        length = output.past_key_values.get_seq_length()
        element_size = next(model.parameters()).element_size()
        dense_reference = 2 * model.config.num_hidden_layers * length * model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads) * element_size
        result = dict(text=tokenizer.decode(output.sequences[0, inputs.shape[1]:], skip_special_tokens=True),
                      checkpoint=request['checkpoint'], step=state['step'], generated_tokens=produced,
                      seconds=elapsed, tokens_per_second=produced/max(elapsed, 1e-6),
                      cache_dense_layers_bytes=dense_bytes, cache_acan_layers_bytes=slots_bytes,
                      cache_total_bytes=dense_bytes+slots_bytes, dense_gqa_reference_bytes=dense_reference,
                      cache_reduction_fraction=1-(dense_bytes+slots_bytes)/dense_reference if dense_reference else None,
                      cache_tokens=length, explanations=explanations,
                      layer_inspection=capture.report(state['step']) if capture else None,
                      gpu_peak_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None)
        with store.db() as db:
            db.execute("UPDATE generations SET status='completed',result=? WHERE id=?", (json.dumps(result), request_id))
    except Exception as exc:
        with store.db() as db:
            db.execute("UPDATE generations SET status='failed',error=? WHERE id=?", (str(exc), request_id))
        raise
