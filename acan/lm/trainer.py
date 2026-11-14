"""Single-device LoRA CPT with WSD, resumable cursors and durable step boundaries."""
import json
import math
import os
from pathlib import Path
import random
import time

from . import checkpoints
from .data import Blocks, PreparationPaused, prepare
from .store import Store, atomic_json
from .grafting import BASE_MODELS, load_base


def validate_options(options):
    defaults = dict(base_model=BASE_MODELS[0], total_steps=1000, sequence_length=128, batch_size=1,
                    accumulation=8, learning_rate=1e-4, slots=16, window=16, lora_rank=8,
                    save_steps=100, save_seconds=300, seed=42, device='auto', gradient_checkpointing=True)
    if set(options) - set(defaults) - {'corpus_id', 'name'}:
        raise ValueError('Opzioni generative non riconosciute.')
    result = dict(defaults, **options)
    if result['base_model'] not in BASE_MODELS or result['device'] not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Modello o dispositivo non supportato.')
    for key, lo, hi in [('total_steps',1,10_000_000),('sequence_length',16,2048),('batch_size',1,8),
                       ('accumulation',1,256),('slots',2,64),('window',1,128),('lora_rank',2,64),
                       ('save_steps',1,10000),('save_seconds',10,3600),('seed',0,2**31-1)]:
        if isinstance(result[key], bool) or not isinstance(result[key], int) or not lo <= result[key] <= hi:
            raise ValueError(f'{key}: atteso intero tra {lo} e {hi}.')
    if not 1e-7 <= result['learning_rate'] <= .01 or not math.isfinite(result['learning_rate']):
        raise ValueError('Learning rate non valido.')
    if not isinstance(result['gradient_checkpointing'], bool):
        raise ValueError('Gradient checkpointing non valido.')
    result['name'] = str(result.get('name') or result['base_model'].split('/')[-1])[:120]
    return result


def resolve_base(store, job, options):
    if options.get('architecture') == 'native':                 # born from zero: its own tokenizer, no pretrained weights
        from .native import resolve
        return resolve(store, job, options)
    from huggingface_hub import HfApi, snapshot_download
    directory = store.directory('jobs', job['id'])
    recorded = directory / 'base.json'
    if recorded.exists():
        base = json.loads(recorded.read_text())
    else:
        revision = HfApi().model_info(options['base_model']).sha
        base = dict(repo=options['base_model'], revision=revision)
        atomic_json(recorded, base)
    path = snapshot_download(base['repo'], revision=base['revision'], cache_dir=str(store.root / 'hub'),
                             allow_patterns=['*.json','*.safetensors','*.model','tokenizer.*','vocab.*','merges.txt'])
    files = sorted(item for item in Path(path).rglob('*') if item.is_file()
                   and item.suffix in ('.json','.safetensors','.model'))
    hashes = {str(item.relative_to(path)): checkpoints.digest_file(item) for item in files}
    if 'files_sha256' in base and base['files_sha256'] != hashes:
        raise ValueError('Cache dei pesi/tokenizer modificata: ripristinare i file della revisione registrata.')
    if 'files_sha256' not in base:
        base['files_sha256'] = hashes
        atomic_json(recorded, base)
    return path, base


def wsd_lr(step, total, maximum):
    warmup = max(1, int(total * .05))
    decay = max(warmup, int(total * .8))
    if step < warmup:
        return maximum * (step + 1) / warmup
    if step < decay:
        return maximum
    fraction = (step - decay) / max(1, total - decay - 1)
    return maximum * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))


def validation_loss(model, blocks, device):
    import torch
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in blocks.validation():
            batch = batch.to(device)
            losses.append(float(model(batch, labels=batch, use_cache=False).loss))
    model.train()
    return sum(losses) / len(losses)


def run_training(root, identifier, stop_event=None):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from threadpoolctl import threadpool_limits
    store = Store(root)
    job = store.job(identifier)
    config = job['config']
    directory = store.directory('jobs', identifier)
    should_pause = lambda: bool((stop_event and stop_event.is_set()) or store.job(identifier)['pause'])
    try:
        torch.set_num_threads(1)
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA non disponibile: collegare un worker GPU oppure scegliere esplicitamente CPU.')
        device = 'cuda' if config['device'] != 'cpu' and torch.cuda.is_available() else 'cpu'
        if device == 'cpu' and config['base_model'] != BASE_MODELS[0]:
            raise ValueError('Il profilo CPU di verifica supporta SmolLM2-135M; per i modelli più grandi usare CUDA.')
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        if device == 'cuda':
            torch.cuda.manual_seed_all(config['seed']); torch.cuda.reset_peak_memory_stats()
        if should_pause():
            raise PreparationPaused()
        store.update(identifier, status='preparing', error=None)
        path, base = resolve_base(store, job, config)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
        corpus = store.corpus(config['corpus_id'], job['owner'], ready=True)
        source = store.directory('corpora', corpus['id']) / 'corpus'
        if checkpoints.digest_file(source) != corpus['sha256']:
            raise ValueError('Il corpus è stato modificato dopo il caricamento.')
        tokens = prepare(source, corpus['name'], tokenizer, directory,
                         dict(corpus=corpus['sha256'], base=base), should_pause)
        if should_pause():
            raise PreparationPaused()
        blocks = Blocks(directory / 'tokens.bin', config['sequence_length'], config['seed'])
        identity = checkpoints.fingerprint(dict(config=config, base=base, corpus=corpus['sha256'], tokens=tokens['sha256']))
        checkpoint_root = directory / 'checkpoints'
        checkpoint_root.mkdir(exist_ok=True)
        previous = checkpoints.for_resume(checkpoint_root, identity, job['checkpoint'], job['progress'].get('step',0))
        evaluate_base = None if previous else lambda model: validation_loss(model, blocks, device)
        with threadpool_limits(limits=1):
            model, graft = load_base(path, config, device, evaluate_base)
            if config['gradient_checkpointing']:
                model.enable_input_require_grads()
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
            state = dict(step=0, cursor=dict(epoch=0, offset=0), tokens_trained=0, baseline_nll=graft['baseline_nll'],
                         device=device, graft=graft, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         total_parameters=sum(p.numel() for p in model.parameters()), identity=identity)
            if previous:
                state = checkpoints.restore(previous, model, optimizer, identity)
                state.pop('numpy_rng', None)
                with store.db() as db:
                    db.execute('DELETE FROM metrics WHERE job=? AND step>?', (identifier, state['step']))
            else:
                state['validation_nll'] = validation_loss(model, blocks, device)
                previous = checkpoints.save(checkpoint_root, model, optimizer, state, identity)
            store.update(identifier, checkpoint=previous.name, progress=state, status='running')
            last_save = time.monotonic()
            started, initial_tokens = time.monotonic(), state['tokens_trained']
            model.train()
            while state['step'] < config['total_steps'] and not should_pause():
                lr = wsd_lr(state['step'], config['total_steps'], config['learning_rate'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_sum, actual_tokens = 0., 0
                for _ in range(config['accumulation']):
                    batch = blocks.batch(config['batch_size'], state['cursor']).to(device)
                    loss = model(batch, labels=batch, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise ValueError('Loss non finita: conservato l’ultimo checkpoint valido.')
                    (loss / config['accumulation']).backward()
                    loss_sum += float(loss.detach()) / config['accumulation']
                    actual_tokens += batch.shape[0] * (batch.shape[1]-1)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                if not torch.isfinite(norm):
                    raise ValueError('Gradienti non finiti: conservato l’ultimo checkpoint valido.')
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                state.update(step=state['step']+1, tokens_trained=state['tokens_trained']+actual_tokens,
                             training_nll=loss_sum, learning_rate=lr)
                state['tokens_per_second'] = (state['tokens_trained'] - initial_tokens) / max(time.monotonic()-started, 1e-6)
                state['gpu_peak_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else None
                save_due = state['step'] % config['save_steps'] == 0 or time.monotonic()-last_save >= config['save_seconds']
                if save_due or should_pause() or state['step'] == config['total_steps']:
                    state['validation_nll'] = validation_loss(model, blocks, device)
                    state['validation_perplexity'] = math.exp(state['validation_nll']) if state['validation_nll'] < 700 else None
                    previous = checkpoints.save(checkpoint_root, model, optimizer, state, identity)
                    store.update(identifier, checkpoint=previous.name)
                    checkpoints.retain(checkpoint_root)
                    last_save = time.monotonic()
                store.record(identifier, state['step'], state)
            # A pause before the next step still commits the exact current state.
            if store.job(identifier)['checkpoint'] != previous.name or should_pause():
                previous = checkpoints.save(checkpoint_root, model, optimizer, state, identity)
                store.update(identifier, checkpoint=previous.name)
                checkpoints.retain(checkpoint_root)
            store.update(identifier, status='completed' if state['step'] >= config['total_steps'] else 'paused',
                         pause=0, progress=state, error=None)
    except PreparationPaused:
        store.update(identifier, status='paused', pause=0, error=None)
    except Exception as exc:
        store.update(identifier, status='failed', error=str(exc), pause=0)
        raise
