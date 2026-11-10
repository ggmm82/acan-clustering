"""Opt-in controlled CPT/distillation. Legacy trainer and core contract are unchanged."""
import hashlib
import json
import math
from pathlib import Path
import random
import time
from . import checkpoints
from .data import PreparationPaused
from .research_data import prepare, Blocks
from .distillation import objective
from .store import Store
from .research_model import load_model as load_base,inference_contract
from .trainer import BASE_MODELS, resolve_base, validation_loss, wsd_lr


def contract():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in
        ('research_trainer.py','research_data.py','distillation.py','research_options.py','research_model.py','full_attention.py'))).hexdigest()


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
        blocks = Blocks(directory, config['sequence_length'], config['seed'], config['validation_batches'])
        identity = checkpoints.fingerprint(dict(config=config, base=base, corpus=corpus['sha256'], tokens=tokens['sha256'], research_contract=contract()))
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
                         total_parameters=sum(p.numel() for p in model.parameters()), identity=identity,
                         model_contract=inference_contract(),dataset={key:tokens[key] for key in ('tokens','documents','duplicates','sha256')},
                         validation_blocks=blocks.evaluation_indices('validation',config['validation_batches']))
            if previous:
                state = checkpoints.restore(previous, model, optimizer, identity)
                state.pop('numpy_rng', None)
                with store.db() as db:
                    db.execute('DELETE FROM metrics WHERE job=? AND step>?', (identifier, state['step']))
            else:
                state['validation_nll'] = validation_loss(model, blocks, device)
                previous = checkpoints.save(checkpoint_root, model, optimizer, state, identity)
            if state.get('early_stop_reason'):
                state['early_stop_bad_checks']=0
                state.pop('early_stop_reason',None)
            state.setdefault('best_validation_nll',state['validation_nll'])
            state.setdefault('early_stop_bad_checks',0)
            teacher = None
            if config['distillation_weight']>0:
                from transformers import AutoModelForCausalLM
                with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device=='cuda' else []):
                    teacher=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                        use_safetensors=True,torch_dtype=next(model.parameters()).dtype,attn_implementation='eager').to(device)
                teacher.eval();teacher.requires_grad_(False)
            store.update(identifier, checkpoint=previous.name, progress=state, status='running')
            last_save = time.monotonic()
            started, initial_tokens = time.monotonic(), state['tokens_trained']
            model.train()
            while state['step'] < config['total_steps'] and not should_pause():
                lr = wsd_lr(state['step'], config['total_steps'], config['learning_rate'])
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_sum, actual_tokens, objective_sum, kl_sum = 0., 0, 0., 0.
                for _ in range(config['accumulation']):
                    batch = blocks.batch(config['batch_size'], state['cursor']).to(device)
                    output = model(batch, labels=batch, use_cache=False)
                    loss = output.loss
                    training_objective = loss
                    divergence = None
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits=teacher(batch,use_cache=False).logits
                        training_objective,divergence=objective(output.logits,teacher_logits,loss,
                            config['distillation_weight'],config['distillation_temperature'])
                    if not torch.isfinite(loss):
                        raise ValueError('Loss non finita: conservato l’ultimo checkpoint valido.')
                    if not torch.isfinite(training_objective):
                        raise ValueError('Obiettivo di distillazione non finito.')
                    (training_objective / config['accumulation']).backward()
                    objective_sum+=float(training_objective.detach())/config['accumulation']
                    kl_sum+=float(divergence.detach())/config['accumulation'] if divergence is not None else 0.
                    loss_sum += float(loss.detach()) / config['accumulation']
                    actual_tokens += batch.shape[0] * (batch.shape[1]-1)
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                if not torch.isfinite(norm):
                    raise ValueError('Gradienti non finiti: conservato l’ultimo checkpoint valido.')
                groups={'acan_gradient_norm':[], 'lora_gradient_norm':[]}
                for name,parameter in model.named_parameters():
                    if parameter.grad is not None:
                        group='lora_gradient_norm' if 'lora_' in name else 'acan_gradient_norm'
                        groups[group].append(parameter.grad.detach().float().square().sum())
                for name,values in groups.items():
                    state[name]=float(torch.stack(values).sum().sqrt()) if values else 0.
                state['gradient_norm']=float(norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                state.update(step=state['step']+1, tokens_trained=state['tokens_trained']+actual_tokens,
                             training_nll=loss_sum, training_objective=objective_sum, teacher_kl=kl_sum, learning_rate=lr)
                state['tokens_per_second'] = (state['tokens_trained'] - initial_tokens) / max(time.monotonic()-started, 1e-6)
                state['gpu_peak_bytes'] = torch.cuda.max_memory_allocated() if device == 'cuda' else None
                save_due = state['step'] % config['save_steps'] == 0 or time.monotonic()-last_save >= config['save_seconds']
                if save_due or should_pause() or state['step'] == config['total_steps']:
                    state['validation_nll'] = validation_loss(model, blocks, device)
                    state['validation_perplexity'] = math.exp(state['validation_nll']) if state['validation_nll'] < 700 else None
                    if state['validation_nll'] < state['best_validation_nll']-config['early_stop_min_delta']:
                        state['best_validation_nll']=state['validation_nll'];state['early_stop_bad_checks']=0
                    else:
                        state['early_stop_bad_checks']+=1
                    if config['early_stop_patience'] and state['early_stop_bad_checks']>=config['early_stop_patience']:
                        state['early_stop_reason']='Validazione senza miglioramenti sufficienti: conservato il checkpoint migliore.'
                        store.update(identifier,pause=1,status='pausing')
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
