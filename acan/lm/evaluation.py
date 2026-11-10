"""Explicit same-test-set comparison with the original model, on actual cache bytes."""
import gc
import json
import math
from pathlib import Path
import time
import torch
from . import checkpoints
from .attention import ACANConceptAttentionLayer
from .grafting import reset_streams
from .research_data import Blocks
from .research_model import load_model,check_inference_contract
from .store import Store,digest_file
from .trainer import resolve_base


class Cancelled(Exception):
    pass


def quality(model,blocks,device,limit,check=lambda:None):
    model.eval();loss_sum=correct=targets=batches=0
    with torch.no_grad():
        for batch in blocks.evaluation('test',limit):
            check()
            batch=batch.to(device);result=model(batch,labels=batch,use_cache=False)
            count=batch.shape[0]*(batch.shape[1]-1)
            loss_sum+=float(result.loss)*count
            correct+=int((result.logits[:,:-1].argmax(-1)==batch[:,1:]).sum())
            targets+=count;batches+=1
    value=loss_sum/targets
    return dict(nll=value,perplexity=math.exp(value) if value<700 else None,
                token_accuracy=correct/targets,tokens=targets,batches=batches,block_indices=blocks.evaluation_indices('test',limit))


def cache_bytes(model,cache):
    storages={}
    for layer in cache.layers:
        for tensor in (getattr(layer,'keys',None),getattr(layer,'values',None)):
            if tensor is not None:storages[tensor.untyped_storage().data_ptr()]=tensor.untyped_storage().nbytes()
    dense=sum(storages.values())
    concept=sum(module.cache_bytes() for module in model.modules() if isinstance(module,ACANConceptAttentionLayer))
    return dict(dense_layers=dense,acan_layers=concept,total=dense+concept)


def throughput(model,ids,device,new_tokens):
    def sync():
        if device=='cuda':torch.cuda.synchronize()
    model.eval();reset_streams(model)
    with torch.no_grad():
        model(ids[:,:min(16,ids.shape[1])],use_cache=False)
        reset_streams(model);sync();started=time.perf_counter()
        result=model(ids,use_cache=True);sync();prefill=time.perf_counter()-started
        cache=result.past_key_values;next_token=result.logits[:,-1].argmax(-1)[:,None]
        sync();started=time.perf_counter()
        for _ in range(new_tokens):
            result=model(next_token,past_key_values=cache,use_cache=True)
            cache=result.past_key_values;next_token=result.logits[:,-1].argmax(-1)[:,None]
        sync();decode=time.perf_counter()-started
        memory=cache_bytes(model,cache)
    reset_streams(model)
    return dict(prompt_tokens=ids.shape[1],new_tokens=new_tokens,prefill_seconds=prefill,
                prefill_tokens_per_second=ids.shape[1]/max(prefill,1e-9),
                decode_seconds=decode,decode_tokens_per_second=new_tokens/max(decode,1e-9),cache=memory)


def evaluate_checkpoint(root,identifier,stop_event=None):
    store=Store(root)
    with store.db() as db:row=dict(db.execute('SELECT * FROM evaluations WHERE id=?',(identifier,)).fetchone())
    options=json.loads(row['request']);job=store.job(row['job'],row['owner']);directory=store.directory('jobs',job['id'])
    def check():
        with store.db() as db:status=db.execute('SELECT status FROM evaluations WHERE id=?',(identifier,)).fetchone()[0]
        if status=='cancelling' or (stop_event and stop_event.is_set()):raise Cancelled()
    try:
        check()
        torch.set_num_threads(1)
        snapshot=directory/'checkpoints'/options['checkpoint'];manifest=checkpoints.verify(snapshot)
        state=json.loads((snapshot/'state.json').read_text());device=state['device']
        check_inference_contract(job['config'],state)
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('Checkpoint CUDA: usare il worker originale.')
        metadata=json.loads((directory/'document-tokens.json').read_text())
        for role,expected in metadata['sha256'].items():
            if role not in ('train','validation','test') or digest_file(directory/(role+'.bin'))!=expected:
                raise ValueError('Split token modificato.')
        blocks=Blocks(directory,job['config']['sequence_length'],job['config']['seed'])
        path,base=resolve_base(store,job,job['config'])
        dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported() else torch.float32
        from transformers import AutoModelForCausalLM
        records={}
        for arm in ('original','selected_checkpoint'):
            check()
            if arm=='original':
                model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                    use_safetensors=True,torch_dtype=dtype,attn_implementation='eager').to(device)
            else:
                model,_=load_model(path,job['config'],device)
                checkpoints.restore(snapshot,model,expected=manifest['identity'],restore_rng=False)
            scores=quality(model,blocks,device,options['max_batches'],check);performance=[]
            for length in options['context_lengths']:
                check()
                if length+options['new_tokens']>model.config.max_position_embeddings or length>len(blocks.tokens['test']):
                    raise ValueError('Contesto oltre modello o campione di test.')
                ids=torch.tensor(blocks.tokens['test'][:length].copy(),dtype=torch.long,device=device)[None]
                performance.append(throughput(model,ids,device,options['new_tokens']))
            records[arm]=dict(quality=scores,performance=performance)
            del model;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        check()
        comparison=[]
        for original,current in zip(records['original']['performance'],records['selected_checkpoint']['performance']):
            comparison.append(dict(context=current['prompt_tokens'],
                prefill_speedup=current['prefill_tokens_per_second']/original['prefill_tokens_per_second'],
                decode_speedup=current['decode_tokens_per_second']/original['decode_tokens_per_second'],
                total_cache_reduction=1-current['cache']['total']/original['cache']['total']))
        result=dict(job=job['id'],checkpoint=options['checkpoint'],step=state['step'],
                    architecture=job['config'].get('architecture','hybrid'),device=device,base=base['repo'],revision=base['revision'],
                    records=records,comparison=comparison,dataset={key:metadata[key] for key in ('documents','duplicates','tokens')},
                    protocol=dict(test='deterministically spread sample of disjoint document test split, never used for checkpoint selection',
                        max_batches=options['max_batches'],seed=job['config']['seed'],tuned=False,
                        attention_backend='eager for both arms',gpu_verified=device=='cuda',evaluation_sha256=digest_file(Path(__file__)),
                        scope='Bounded diagnostic; not a general-language or MMLU certification. Public text may overlap original pretraining.'))
        with store.db() as db:db.execute("UPDATE evaluations SET status='completed',result=? WHERE id=?",(json.dumps(result,allow_nan=False),identifier))
    except Cancelled:
        with store.db() as db:db.execute("UPDATE evaluations SET status='cancelled',error=NULL WHERE id=?",(identifier,))
    except Exception as exc:
        with store.db() as db:db.execute("UPDATE evaluations SET status='failed',error=? WHERE id=?",(str(exc),identifier))
        raise
