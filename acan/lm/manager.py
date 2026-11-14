"""Detached workers survive API/browser restarts; one exclusive accelerator lease."""
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid

from . import checkpoints
from .store import ACTIVE


def launch(store, job, kind='train', generation_id=None):
    resource = store.resource()
    queued = False
    try:
        try:
            fcntl.flock(resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Worker occupato: mettere in pausa il training prima di generare o avviare un altro progetto.') from None
        if kind == 'train':
            store.update(job['id'], status='queued', pause=0, error=None)
            queued = True
        args = [sys.executable, '-m', 'acan.lm.worker', '--root', str(store.root), '--job', job['id'],
                '--lease-fd', str(resource.fileno()), '--kind', kind]
        if generation_id:
            args += ['--generation', generation_id]
        log = store.directory('jobs', job['id']) / 'worker.log'
        with log.open('ab', buffering=0) as stream:
            subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                             pass_fds=(resource.fileno(),))
    except Exception:
        if kind == 'train' and queued:
            store.update(job['id'], status='failed', error='Impossibile avviare il worker.')
        raise
    finally:
        # Do not LOCK_UN: the inherited descriptor keeps the same flock alive.
        resource.close()


def snapshots(store, job):
    root = store.directory('jobs', job['id']) / 'checkpoints'
    from .retention import protect_available
    protect_available(root)
    result = []
    paths = list(root.glob('step-*'))
    pointer = root/'best.json'
    best = json.loads(pointer.read_text()) if pointer.exists() else None
    if best:
        paths.append(root/best['name'])
    for path in sorted(paths, reverse=True):
        if (path / 'manifest.json').exists():
            try:
                manifest=json.loads((path/'manifest.json').read_text())
                state_file=path/'state.json'
                state=json.loads(state_file.read_text()) if state_file.exists() else {}
            except (FileNotFoundError,ValueError):
                continue
            result.append(dict(name=path.name, step=manifest['step'],
                               validation_nll=state.get('validation_nll'),compatible=manifest.get('contract')==checkpoints.contract(),
                               best=bool(best and path.name==best['name'])))
    return sorted(result, key=lambda item:(not item['best'], -item['step']))



def request_generation(store, job, owner, options):
    if job['status'] in ACTIVE:
        raise ValueError('Premi Pausa e attendi il salvataggio prima di provare il checkpoint sulla stessa GPU.')
    allowed = {'prompt','checkpoint','max_new_tokens','temperature','seed','inspect_layers','full_length'}
    if set(options) - allowed:
        raise ValueError('Opzioni generazione non valide.')
    request = dict(dict(max_new_tokens=64, temperature=.7, seed=42, inspect_layers=False, full_length=False), **options)
    if not isinstance(request['inspect_layers'], bool) or not isinstance(request['full_length'], bool):
        raise ValueError('Ispezione layer non valida.')
    if not isinstance(request.get('prompt'), str) or not 1 <= len(request['prompt']) <= 32000:
        raise ValueError('Prompt non valido.')
    if not isinstance(request['max_new_tokens'], int) or not 1 <= request['max_new_tokens'] <= 256:
        raise ValueError('Scegli da 1 a 256 nuovi token.')
    if request['inspect_layers'] and request['max_new_tokens'] > 32:
        raise ValueError('Per ispezionare i layer scegli al massimo 32 nuovi token.')
    if not isinstance(request['seed'], int) or not 0 <= request['seed'] < 2**31 or not 0 <= request['temperature'] <= 2:
        raise ValueError('Seed o temperatura non validi.')
    if request.get('checkpoint') not in {item['name'] for item in snapshots(store, job)}:
        raise ValueError('Checkpoint non più disponibile: aggiorna la lista.')
    identifier = uuid.uuid4().hex
    with store.db() as db:
        db.execute('INSERT INTO generations(id,job,owner,status,request) VALUES(?,?,?,?,?)',
                   (identifier, job['id'], owner, 'running', json.dumps(request)))
    try:
        launch(store, job, 'generate', identifier)
    except Exception:
        with store.db() as db:
            db.execute("UPDATE generations SET status='failed',error='Worker occupato.' WHERE id=?", (identifier,))
        raise
    return identifier


SERIES_VARIANTS = {
    'chosen': ('impostazioni scelte', {}),
    'fewer_concepts': ('metà dei concetti', None),
    'all_layers': ('tutti i layer a concetti', None),
    'dense': ("attenzione normale", {'architecture': 'dense-control'}),
}


def series_configs(base, variants):
    """The projects of a series: the same text, steps and seed, one setting changed in each."""
    if not isinstance(variants, list) or not 2 <= len(variants) <= 4 or len(set(variants)) != len(variants) \
            or any(item not in SERIES_VARIANTS for item in variants):
        raise ValueError('Scegli da 2 a 4 varianti diverse.')
    configs = []
    for item in variants:
        label, change = SERIES_VARIANTS[item]
        if item == 'fewer_concepts':                           # stronger compression, possible on the CPU too
            change = {'slots': max(int(base.get('slots', 16))//2, 2)}
        elif item == 'all_layers':
            change = {'graft_fraction': 1., 'sequence_length': min(int(base.get('sequence_length', 128)), 64)}
        configs.append((item, label, dict(base, **change, name=f"{base.get('name', 'Serie')} · {label}")))
    return configs


def continue_series(store, identifier):
    """When a project of a series ends on its own, the next one starts; a pause by the user stops the series."""
    job = store.job(identifier)
    series = job['config'].get('series')
    if not series:
        return None
    ended = job['status'] == 'completed' or (job['status'] == 'paused' and job['progress'].get('early_stop_reason'))
    if not ended:
        return None
    waiting = [other for other in store.jobs(job['owner']) if (other['config'].get('series') or {}).get('id') == series['id']
               and other['status'] == 'created']
    if not waiting:
        return None
    following = min(waiting, key=lambda other: other['config']['series']['order'])
    launch(store, following)
    return following['id']


def request_evaluation(store,job,owner,options):
    if job['config'].get('protocol')!='controlled-v1':
        raise ValueError('Il progetto legacy non ha un test indipendente: crea un nuovo esperimento controllato.')
    if job['status'] in ACTIVE:
        raise ValueError('Metti in pausa il training prima del confronto.')
    if not isinstance(options,dict) or set(options)-{'checkpoint','max_batches','new_tokens','context_lengths'}:
        raise ValueError('Opzioni del confronto non valide.')
    request=dict(max_batches=4,new_tokens=8,context_lengths=[32,128])
    request.update(options)
    if request.get('checkpoint') not in {item['name'] for item in snapshots(store,job)}:
        raise ValueError('Checkpoint non disponibile.')
    import torch
    max_batches=64 if torch.cuda.is_available() else 16
    for key,lo,hi in [('max_batches',1,max_batches),('new_tokens',1,32)]:
        value=request[key]
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
            raise ValueError('Budget non valido: '+key)
    contexts=request['context_lengths']
    if not isinstance(contexts,list) or not 1<=len(contexts)<=4 or any(isinstance(x,bool) or not isinstance(x,int) or not 16<=x<=8192 for x in contexts):
        raise ValueError('Scegli al massimo quattro contesti tra 16 e 8192 token.')
    identifier=uuid.uuid4().hex
    with store.db() as db:db.execute('INSERT INTO evaluations(id,job,owner,status,request) VALUES(?,?,?,?,?)',
        (identifier,job['id'],owner,'running',json.dumps(request)))
    try:
        launch(store,job,'evaluate',identifier)
    except Exception:
        with store.db() as db:db.execute("UPDATE evaluations SET status='failed',error='Worker occupato o non disponibile.' WHERE id=?",(identifier,))
        raise
    return identifier
