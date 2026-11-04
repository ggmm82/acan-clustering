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
    result = []
    for path in sorted(root.glob('step-*'), reverse=True):
        if (path / 'manifest.json').exists():
            manifest = json.loads((path / 'manifest.json').read_text())
            result.append(dict(name=path.name, step=manifest['step']))
    return result


def request_generation(store, job, owner, options):
    if job['status'] in ACTIVE:
        raise ValueError('Premi Pausa e attendi il salvataggio prima di provare il checkpoint sulla stessa GPU.')
    allowed = {'prompt','checkpoint','max_new_tokens','temperature','seed'}
    if set(options) - allowed:
        raise ValueError('Opzioni generazione non valide.')
    request = dict(dict(max_new_tokens=64, temperature=.7, seed=42), **options)
    if not isinstance(request.get('prompt'), str) or not 1 <= len(request['prompt']) <= 32000:
        raise ValueError('Prompt non valido.')
    if not isinstance(request['max_new_tokens'], int) or not 1 <= request['max_new_tokens'] <= 256:
        raise ValueError('Scegli da 1 a 256 nuovi token.')
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
