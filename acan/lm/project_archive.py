"""Portable backups of one resumable Generativo project."""
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid
import zipfile

from . import checkpoints
from .concept_trainer import PROFILE, training_contract
from .native import _hashes
from .research_data import ROLES
from .retention import protect
from .store import ACTIVE, atomic_json, digest_file

SNAPSHOT = re.compile(r'(?:step|best|milestone)-[a-z0-9-]{1,100}')
CHECKPOINT_FILES = ('manifest.json', 'state.json', 'trainables.safetensors', 'optimizer.pt', 'rng.pt')
JOB_FILES = ('base.json', 'document-tokens.json', *(role + '.bin' for role in ROLES))
OPTIONAL = ('answers.jsonl', 'public-source.json')


def identity(config, base, corpus, tokens):
    return checkpoints.fingerprint(dict(
        config={key: value for key, value in config.items() if key != 'total_steps'},
        base=base, corpus=corpus, tokens=tokens, contract=training_contract(config)))


def available_disk(store, size):
    disk = shutil.disk_usage(store.root)
    if size > disk.free - min(1024**3, disk.total // 4):
        raise ValueError('Spazio insufficiente per il trasferimento del progetto.')


def export_project(store, owner, identifier):
    from .manager import snapshots

    job = store.job(identifier, owner)
    if job['status'] in ACTIVE or not job['checkpoint']:
        raise ValueError('Metti il progetto in pausa e attendi un checkpoint prima di esportare.')
    config = job['config']
    if config.get('architecture') != 'concepts' or config.get('routing_profile') != PROFILE:
        raise ValueError('L’esportazione supporta i progetti della rete a concetti.')
    corpus = store.corpus(config['corpus_id'], owner, ready=True)
    root = store.directory('jobs', identifier)
    archive = tempfile.TemporaryFile(dir=store.root)
    try:
        with store.resource() as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Il worker è occupato. Riprova quando è libero.') from None
            job = store.job(identifier, owner)
            if job['status'] in ACTIVE or job['config'] != config:
                raise ValueError('Il progetto è cambiato. Aggiorna la pagina e riprova.')
            latest = job['checkpoint']
            saved = checkpoints.verify(root/'checkpoints'/latest)
            state = json.loads((root/'checkpoints'/latest/'state.json').read_text())
            base = json.loads((root/'base.json').read_text())
            tokens = json.loads((root/'document-tokens.json').read_text())
            if saved['identity'] != identity(config, base, corpus['sha256'], tokens['sha256']):
                raise ValueError('Il checkpoint non corrisponde alla configurazione attuale del progetto.')
            best = next((s['name'] for s in snapshots(store, job) if s['best']), None)
            files = {f'job/{name}': root/name for name in JOB_FILES}
            files.update({f'job/tokenizer/{name}': root/'tokenizer'/name for name in _hashes(root/'tokenizer')})
            files.update({f'job/{name}': root/name for name in OPTIONAL if (root/name).is_file()})
            for name in dict.fromkeys([latest, best]):
                if name:
                    checkpoints.verify(root/'checkpoints'/name)
                    files.update({f'job/checkpoints/{name}/{part}': root/'checkpoints'/name/part
                                  for part in CHECKPOINT_FILES})
            files['corpus'] = store.directory('corpora', corpus['id'])/'corpus'
            available_disk(store, sum(path.stat().st_size for path in files.values()))
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as out:
                out.writestr('project.json', json.dumps(dict(version=1, config=config, latest=latest, best=best,
                                                            corpus=dict(name=corpus['name'], sha256=corpus['sha256'])),
                                                        allow_nan=False))
                with store.db() as db, out.open('history.jsonl', 'w', force_zip64=True) as history:
                    for row in db.execute('SELECT record FROM metrics WHERE job=? AND step<=? ORDER BY step',
                                          (identifier, state['step'])):
                        history.write(row[0].encode() + b'\n')
                for name, path in files.items():
                    out.write(path, name)
        archive.seek(0)
        return archive
    except BaseException:
        archive.close()
        raise


def _stage(store, identifier):
    if not re.fullmatch(r'[0-9a-f]{32}', identifier):
        raise ValueError('Identificatore trasferimento non valido.')
    return store.root/'project-imports'/identifier


def begin_import(store, owner, size):
    if type(size) is not int or size <= 0:
        raise ValueError('Dimensione del file non valida.')
    available_disk(store, size)
    identifier = uuid.uuid4().hex
    folder = _stage(store, identifier)
    folder.mkdir(parents=True)
    atomic_json(folder/'upload.json', dict(owner=owner, size=size, received=0))
    return {'id': identifier}


def _upload(folder, owner):
    try:
        data = json.loads((folder/'upload.json').read_text())
    except FileNotFoundError:
        raise KeyError('Trasferimento non trovato.') from None
    if data['owner'] != owner:
        raise KeyError('Trasferimento non trovato.')
    return data


def append_import(store, owner, identifier, offset, chunk):
    folder = _stage(store, identifier)
    _upload(folder, owner)
    with (folder/'upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = _upload(folder, owner)
        if not chunk or len(chunk) > 2*1024**2 or offset != data['received'] or offset+len(chunk) > data['size']:
            raise ValueError('Chunk o posizione del trasferimento non validi.')
        available_disk(store, len(chunk))
        archive = folder/'project.zip'
        with archive.open('r+b' if archive.exists() else 'wb') as stream:
            stream.seek(offset)
            stream.write(chunk)
            stream.truncate()
        data['received'] += len(chunk)
        atomic_json(folder/'upload.json', data)
    return {'received': data['received']}


def cancel_import(store, owner, identifier):
    folder = _stage(store, identifier)
    _upload(folder, owner)
    with (folder/'upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _upload(folder, owner)
        shutil.rmtree(folder)


def _extract(store, archive, folder):
    with zipfile.ZipFile(archive) as source:
        listing = source.infolist()
        names = {item.filename for item in listing}
        if len(names) != len(listing) or 'project.json' not in names:
            raise ValueError('Archivio progetto non valido.')
        if source.getinfo('project.json').file_size > 16*1024**2:
            raise ValueError('Indice del progetto troppo grande.')
        data = json.loads(source.read('project.json'))
        if data.get('version') != 1:
            raise ValueError('Versione archivio progetto non supportata.')
        latest, best = data['latest'], data['best']
        if not latest.startswith('step-') or any(not SNAPSHOT.fullmatch(n) for n in [latest] + ([best] if best else [])):
            raise ValueError('Nome checkpoint non valido.')
        required = {'project.json', 'history.jsonl', 'corpus', *(f'job/{name}' for name in JOB_FILES)}
        required.update(f'job/checkpoints/{name}/{part}' for name in {latest,best} if name for part in CHECKPOINT_FILES)
        if not required.issubset(names):
            raise ValueError('Archivio progetto incompleto.')
        for name in names - required:
            if name in {f'job/{part}' for part in OPTIONAL}:
                continue
            if not re.fullmatch(r'job/tokenizer/[a-zA-Z0-9_.-]+', name) or Path(name).name in ('.', '..'):
                raise ValueError('File inatteso nell’archivio progetto.')
        available_disk(store, sum(item.file_size for item in listing))
        for name in names - {'project.json'}:
            path = folder/name
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(name) as stream, path.open('xb') as output:
                shutil.copyfileobj(stream, output, 1024**2)
        return data


def _register(store, owner, folder, data):
    from safetensors import safe_open
    from .research_options import validate_options

    source_config = data['config']
    if source_config.get('routing_profile') != PROFILE or source_config.get('architecture') != 'concepts':
        raise ValueError('Il progetto deve usare la rete a concetti.')
    config = validate_options({k: v for k, v in source_config.items() if k not in ('series', 'control_of')})
    root = folder/'job'
    base = json.loads((root/'base.json').read_text())
    tokens = json.loads((root/'document-tokens.json').read_text())
    corpus_path = folder/'corpus'
    corpus_name = data['corpus']['name']
    corpus_hash = digest_file(corpus_path)
    if (not corpus_name.lower().endswith(('.txt', '.jsonl')) or Path(corpus_name).name != corpus_name
            or corpus_hash != data['corpus']['sha256'] or not tokens['complete']
            or base['files_sha256'] != _hashes(root/'tokenizer') or not (root/'tokenizer'/'tokenizer.json').is_file()):
        raise ValueError('Corpus, tokenizer o token del progetto non validi.')
    if tokens['identity'] != checkpoints.fingerprint(dict(
            identity=dict(corpus=corpus_hash, base=base), protocol='document-splits-v1')):
        raise ValueError('La preparazione dei token non corrisponde al corpus e al tokenizer.')
    for role in ROLES:
        path = root/(role+'.bin')
        if path.stat().st_size != 4*tokens['tokens'][role] or digest_file(path) != tokens['sha256'][role]:
            raise ValueError('Archivio token danneggiato.')
    old_identity = identity(source_config, base, corpus_hash, tokens['sha256'])
    corpus_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    config.update(corpus_id=corpus_id, warm_start=False,
                  series=dict(id=uuid.uuid4().hex, order=0, variant='queue', label=corpus_name))
    new_identity = identity(config, base, corpus_hash, tokens['sha256'])
    latest, best = data['latest'], data['best']
    cp = root/'checkpoints'
    states = {}
    for name in dict.fromkeys([latest, best]):
        if not name:
            continue
        directory = cp/name
        manifest = checkpoints.verify(directory, old_identity)
        state = json.loads((directory/'state.json').read_text())
        if state['step'] != manifest['step'] or not 0 <= state['step'] <= config['total_steps']:
            raise ValueError('Passo del checkpoint non valido.')
        with safe_open(str(directory/'trainables.safetensors'), framework='np') as model:
            if 'net_text' not in model.keys():
                raise ValueError('Checkpoint della rete a concetti non valido.')
        state['identity'] = new_identity
        state.pop('structure', None)
        atomic_json(directory/'state.json', state)
        manifest['identity'] = new_identity
        manifest['sha256']['state.json'] = digest_file(directory/'state.json')
        atomic_json(directory/'manifest.json', manifest)
        states[name] = state
    progress = states[latest]
    if best:
        state = states[best]
        if state['step'] > progress['step'] or state['validation_nll'] > progress['validation_nll']:
            raise ValueError('Miglior checkpoint non valido.')
        atomic_json(cp/'best.json', dict(name=best, source=latest, step=state['step'],
                                         validation_nll=state['validation_nll']))
    else:
        protect(cp, latest)
    def metrics():
        with (folder/'history.jsonl').open(encoding='utf-8') as history:
            for line in history:
                row = json.loads(line)
                step = row['step']
                if type(step) is not int or not 0 <= step <= progress['step']:
                    raise ValueError('Passo nello storico non valido.')
                yield job_id, step, json.dumps(row, allow_nan=False)
    job_dest = store.root/'jobs'/job_id
    corpus_dest = store.root/'corpora'/corpus_id
    job_dest.parent.mkdir(exist_ok=True)
    corpus_dest.mkdir(parents=True)
    try:
        size = corpus_path.stat().st_size
        os.replace(corpus_path, corpus_dest/'corpus')
        os.replace(root, job_dest)
        with store.db() as db:
            db.execute('INSERT INTO corpora VALUES(?,?,?,?,?,?,?)',
                       (corpus_id, owner, corpus_name, size, size, 'ready', corpus_hash))
            db.execute('INSERT INTO jobs(id,owner,status,config,progress,checkpoint,updated) VALUES(?,?,?,?,?,?,?)',
                       (job_id, owner, 'completed' if progress['step'] == config['total_steps'] else 'paused',
                        json.dumps(config), json.dumps(progress, allow_nan=False), latest, time.time()))
            db.executemany('INSERT INTO metrics VALUES(?,?,?)', metrics())
    except BaseException:
        shutil.rmtree(corpus_dest)
        if job_dest.exists():
            shutil.rmtree(job_dest)
        raise
    return store.job(job_id, owner)


def finish_import(store, owner, identifier):
    folder = _stage(store, identifier)
    _upload(folder, owner)
    with (folder/'upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        upload = _upload(folder, owner)
        if upload['received'] != upload['size']:
            raise ValueError('Trasferimento incompleto.')
        try:
            with tempfile.TemporaryDirectory(dir=folder) as staging:
                directory = Path(staging)
                data = _extract(store, folder/'project.zip', directory)
                job = _register(store, owner, directory, data)
        except (AttributeError, KeyError, TypeError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
            raise ValueError('Archivio di progetto non valido o incompleto.') from exc
        shutil.rmtree(folder)
        return job
