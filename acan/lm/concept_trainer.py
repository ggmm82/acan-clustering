"""Long learning of the concept language model (acan/lm/concept_lm.py): reading the text is counting it.

A step reads one piece of the training text and adds its sums to the model. Every ``save_steps`` steps
(or ``save_seconds`` seconds) the model is settled, its weights are chosen by the judgment on half of the
validation pieces, its error is measured on the other half, and a checkpoint is saved in the format every
project uses: pause, resume after a restart, best checkpoint, milestones and the queue work as for any
project. A queue item starts from the sums of the previous text (they are added, nothing is forgotten).

The tokenizer is learned from the texts as for the native model (acan/lm/native.py), the text is split in
training, validation and test documents as for every controlled project (acan/lm/research_data.py).
"""
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from . import checkpoints
from .data import PreparationPaused
from .store import Store

PROFILE = 'concept-lm-v1'


def contract():
    """The sources that define what a checkpoint of this model means."""
    root = Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in ('concept_lm.py', 'concept_trainer.py'))).hexdigest()


def piece_tokens(config):
    """Tokens read by one step."""
    return int(config.get('concept_piece', 1_000_000))


def previous_checkpoint(store, job):
    """The latest checkpoint of the item before this one in its queue (its sums are the start)."""
    series = job['config'].get('series') or {}
    before = [other for other in store.jobs(job['owner']) if (other['config'].get('series') or {}).get('id') == series.get('id')
              and other['config']['series']['order'] < series.get('order', 0)]
    if not before:
        raise ValueError('Primo file della coda: non c\'è un apprendimento precedente da cui ripartire.')
    previous = max(before, key=lambda other: other['config']['series']['order'])
    if previous['status'] in ('queued', 'preparing', 'running', 'pausing') or not previous['checkpoint']:
        raise ValueError('Il file precedente della coda non ha ancora un checkpoint salvato.')
    path = store.directory('jobs', previous['id'])/'checkpoints'/previous['checkpoint']
    if not (path/'manifest.json').exists():
        raise ValueError('Checkpoint del file precedente non trovato: ripristina il backup.')
    return path


def judged_pieces(blocks, limit):
    """Validation pieces: the even ones choose the weights, the odd ones measure the error (never the same)."""
    indices = blocks.evaluation_indices('validation', 2*limit)
    pieces = [np.asarray(blocks._block('validation', index), np.int64) for index in indices]
    return pieces[0::2], pieces[1::2] or pieces[0::2]


def measure(model, pieces):
    """Mean error (nats per token) on pieces of text the model never counted, every position read causally."""
    values = np.concatenate([model.logp(piece, np.arange(1, len(piece))) for piece in pieces])
    return float(-values.mean())


def record_answers(directory, model, tokenizer, state, questions=None, new_tokens=64):
    """The answers to the same questions at every save, as for every project (acan/lm/probes.py).

    Drawn with temperature 0.7 from a fixed seed: deterministic, so two saves differ only because the
    model changed. With temperature 0 a counting model repeats its most frequent phrase.
    """
    from .media import readable
    from .probes import QUESTIONS, chat_prompt, stop_tokens
    started = time.monotonic()
    stop = set(stop_tokens(tokenizer))
    answers = []
    for question in (questions or QUESTIONS):
        ids = tokenizer(chat_prompt(question))['input_ids']
        produced = model.generate(ids, new_tokens, stop=stop, temperature=.7, seed=0)
        answers.append(dict(question=question, answer=readable(tokenizer, produced).strip()))
    entry = dict(step=state['step'], tokens_trained=state.get('tokens_trained'), validation_nll=state.get('validation_nll'),
                 seconds=time.monotonic()-started, answers=answers)
    with (Path(directory)/'answers.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False)+'\n')
    return entry


def quantities(model, config):
    """What the page shows about the model while it learns."""
    levels = model.concepts() if not model.counter else None
    return dict(concepts_per_level=levels, concepts_total=sum(levels) if levels else None, statistics_bytes=model.nbytes(),
                budget_bytes=model.budget_bytes, generalized_concepts=model.generalized, context_depth=model.depth,
                document_depth=model.document, weights_global=[float(x) for x in model.a],
                weights_document=[float(x) for x in model.b], memory_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def run_training(root, identifier, stop_event=None):
    from transformers import AutoTokenizer
    from .concept_lm import ConceptLM, default_budget, load, save
    from .native import resolve
    from .research_data import Blocks, prepare
    from .retention import retain_with_milestones
    store = Store(root)
    job = store.job(identifier)
    config = job['config']
    directory = store.directory('jobs', identifier)
    should_pause = lambda: bool((stop_event and stop_event.is_set()) or store.job(identifier)['pause'])
    try:
        if should_pause():
            raise PreparationPaused()
        store.update(identifier, status='preparing', error=None)
        path, base = resolve(store, job, config)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
        corpus = store.corpus(config['corpus_id'], job['owner'], ready=True)
        source = store.directory('corpora', corpus['id'])/'corpus'
        if checkpoints.digest_file(source) != corpus['sha256']:
            raise ValueError('Il corpus è stato modificato dopo il caricamento.')
        tokens = prepare(source, corpus['name'], tokenizer, directory, dict(corpus=corpus['sha256'], base=base), should_pause)
        if should_pause():
            raise PreparationPaused()
        blocks = Blocks(directory, config['sequence_length'], config['seed'], config['validation_batches'])
        train = blocks.tokens['train']
        piece = piece_tokens(config)
        share = min(config.get('passes') or 1., 1.)              # counting a text twice teaches nothing new
        steps = max(1, math.ceil(len(train)*share/piece))
        if config.get('total_steps') != steps and not job['checkpoint']:
            config['total_steps'] = steps
            store.set_config(identifier, config)
        identity = checkpoints.fingerprint(dict(config={key: value for key, value in config.items() if key != 'total_steps'},
                                                base=base, corpus=corpus['sha256'], tokens=tokens['sha256'], contract=contract()))
        root_checkpoints = directory/'checkpoints'
        root_checkpoints.mkdir(exist_ok=True)
        previous = checkpoints.for_resume(root_checkpoints, identity, job['checkpoint'], job['progress'].get('step', 0))
        budget = int(config.get('concept_memory_mb') or 0)*(1 << 20) or default_budget()
        tune, judge = judged_pieces(blocks, config['validation_batches'])
        if previous:
            model, state = load(previous, identity, budget)
            with store.db() as db:
                db.execute('DELETE FROM metrics WHERE job=? AND step>?', (identifier, state['step']))
        else:
            model = ConceptLM(len(tokenizer), config.get('concept_depth', 6), config.get('concept_document', 4), budget)
            state = dict(step=0, tokens_trained=0, device='cpu', identity=identity, model_contract=contract(), profile=PROFILE,
                         dataset={key: tokens[key] for key in ('tokens', 'documents', 'duplicates', 'sha256')},
                         graft=dict(profile=PROFILE, vocabulary=len(tokenizer), replaced_layers=[], total_layers=0,
                                    slots=0, window=0, fraction=1.))
            if config.get('warm_start'):
                source = previous_checkpoint(store, job)
                start, _ = load(source, budget_bytes=budget)
                model.merge(start)
                model.a, model.b = start.a.copy(), start.b.copy()
                state['warm_start'] = dict(checkpoint=source.name, job=source.parent.parent.name,
                                           tokens=start.tokens)
            state['baseline_nll'] = measure(model, judge)       # before reading this text
            state['validation_nll'] = state['baseline_nll']
            state['best_validation_nll'] = state['validation_nll']
            state.update(quantities(model, config))
            previous = save(root_checkpoints, model, state, identity)
        store.update(identifier, checkpoint=previous.name, progress=state, status='running')
        started, initial = time.monotonic(), state['tokens_trained']
        last_save = time.monotonic()
        while state['step'] < config['total_steps'] and not should_pause():
            begin = state['step']*piece
            ids = np.asarray(train[begin:begin+piece], np.int64)
            model.add(ids)
            state.update(step=state['step']+1, tokens_trained=state['tokens_trained']+len(ids))
            state['tokens_per_second'] = (state['tokens_trained']-initial)/max(time.monotonic()-started, 1e-6)
            due = state['step'] % config['save_steps'] == 0 or time.monotonic()-last_save >= config['save_seconds']
            if due or should_pause() or state['step'] == config['total_steps']:
                model.settle()
                model.judge(tune)
                state['validation_nll'] = measure(model, judge)
                state['validation_perplexity'] = math.exp(state['validation_nll']) if state['validation_nll'] < 700 else None
                following = np.asarray(train[state['step']*piece:state['step']*piece+4096], np.int64)
                if len(following) > 1:                           # the text it is about to read: unseen, as validation
                    state['training_nll'] = float(-model.logp(following, np.arange(1, len(following))).mean())
                state['best_validation_nll'] = min(state.get('best_validation_nll', state['validation_nll']), state['validation_nll'])
                state.update(quantities(model, config))
                previous = save(root_checkpoints, model, state, identity)
                store.update(identifier, checkpoint=previous.name)
                retain_with_milestones(root_checkpoints, milestone=config.get('milestone_steps', 0))
                if config.get('chat_probes'):
                    record_answers(directory, model, tokenizer, state, config.get('probe_prompts') or None)
                last_save = time.monotonic()
            store.record(identifier, state['step'], state)
        store.update(identifier, status='completed' if state['step'] >= config['total_steps'] else 'paused',
                     pause=0, progress=state, error=None)
    except PreparationPaused:
        store.update(identifier, status='paused', pause=0, error=None)
    except Exception as exc:
        store.update(identifier, status='failed', error=str(exc), pause=0)
        raise


def generate(store, job, request, directory):
    """Continue a prompt with a checkpoint: the result in the shape the generation page reads."""
    from transformers import AutoTokenizer
    from .concept_lm import load
    from .media import readable
    from .probes import chat_prompt, stop_tokens
    model, state = load(directory)
    tokenizer = AutoTokenizer.from_pretrained(str(store.directory('jobs', job['id'])/'tokenizer'), local_files_only=True,
                                              trust_remote_code=False)
    chat = request.get('chat', False)
    ids = tokenizer(chat_prompt(request['prompt']) if chat else request['prompt'])['input_ids']
    if not ids:
        raise ValueError('Prompt vuoto.')
    stop = set(stop_tokens(tokenizer)) if chat else (set() if request.get('full_length') else {tokenizer.eos_token_id})
    started = time.perf_counter()
    produced = model.generate(ids, request['max_new_tokens'], stop=stop, temperature=request['temperature'], seed=request['seed'])
    elapsed = time.perf_counter()-started
    statistics = model.nbytes()
    return dict(text=readable(tokenizer, produced), checkpoint=request['checkpoint'], step=state['step'],
                generated_tokens=len(produced), seconds=elapsed, tokens_per_second=len(produced)/max(elapsed, 1e-6),
                cache_dense_layers_bytes=0, cache_acan_layers_bytes=statistics, cache_total_bytes=statistics,
                dense_gqa_reference_bytes=None, cache_reduction_fraction=None, cache_tokens=len(ids)+len(produced),
                explanations=[], layer_inspection=None, gpu_peak_bytes=None, concept_model=True,
                concepts_per_level=model.concepts())


def evaluate(store, job, request, directory, check):
    """The error on the test documents (never counted, never used to choose the weights)."""
    from .concept_lm import load
    from .research_data import Blocks
    model, state = load(directory)
    folder = store.directory('jobs', job['id'])
    blocks = Blocks(folder, job['config']['sequence_length'], job['config']['seed'], job['config']['validation_batches'])
    limit = int(request.get('test_batches', 32))
    values, correct = [], 0
    for index in blocks.evaluation_indices('test', limit):
        check()
        piece = np.asarray(blocks._block('test', index), np.int64)
        values.append(model.logp(piece, np.arange(1, len(piece))))
    every = np.concatenate(values)
    nll = float(-every.mean())
    return dict(nll=nll, perplexity=math.exp(nll) if nll < 700 else None, tokens=len(every), batches=len(values),
                block_indices=blocks.evaluation_indices('test', limit), step=state['step'], concept_model=True)
