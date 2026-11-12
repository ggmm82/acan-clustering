"""Durable, anonymous-workspace-isolated LM jobs and streaming corpora."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid

ACTIVE = ('queued', 'preparing', 'running', 'pausing')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def owner_key(key):
    if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key):
        raise ValueError('Identificatore workspace non valido.')
    return hashlib.sha256(key.encode()).hexdigest()


class Store:
    def __init__(self, root):
        self._history_cache = {}
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    status TEXT NOT NULL, config TEXT NOT NULL, progress TEXT NOT NULL DEFAULT '{}',
                    checkpoint TEXT, pause INTEGER NOT NULL DEFAULT 0, error TEXT, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS corpora (id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    name TEXT NOT NULL, expected INTEGER NOT NULL, received INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL, sha256 TEXT);
                CREATE TABLE IF NOT EXISTS metrics (job TEXT, step INTEGER, record TEXT, PRIMARY KEY(job,step));
                CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, job TEXT, owner TEXT,
                    status TEXT, request TEXT, result TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS layer_inspections (job TEXT PRIMARY KEY, pending INTEGER NOT NULL DEFAULT 0, result TEXT);
                CREATE TABLE IF NOT EXISTS generations (id TEXT PRIMARY KEY, job TEXT, owner TEXT,
                    status TEXT, request TEXT, result TEXT, error TEXT);
            ''')

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.root / 'state.sqlite', timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA synchronous=FULL')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def directory(self, kind, identifier):
        if kind not in ('jobs', 'corpora') or not re.fullmatch('[0-9a-f]{32}', identifier):
            raise ValueError('Identificatore non valido.')
        directory = self.root / kind / identifier
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def job(self, identifier, owner=None):
        with self.db() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
        if row is None or (owner is not None and row['owner'] != owner):
            raise KeyError('Progetto non trovato.')
        item = dict(row)
        item['config'], item['progress'] = json.loads(item['config']), json.loads(item['progress'])
        return item

    def jobs(self, owner):
        with self.db() as db:
            ids = [row[0] for row in db.execute('SELECT id FROM jobs WHERE owner=? ORDER BY updated DESC', (owner,))]
        return [self.job(identifier, owner) for identifier in ids]

    def create_job(self, owner, config):
        self.corpus(config['corpus_id'], owner, ready=True)
        identifier = uuid.uuid4().hex
        self.directory('jobs', identifier)
        with self.db() as db:
            db.execute('INSERT INTO jobs(id,owner,status,config,updated) VALUES(?,?,?,?,?)',
                       (identifier, owner, 'created', json.dumps(config), time.time()))
        return self.job(identifier, owner)

    def update(self, identifier, **changes):
        allowed = {'status', 'progress', 'checkpoint', 'pause', 'error'}
        if not changes or not set(changes).issubset(allowed):
            raise ValueError('Aggiornamento stato non valido.')
        if 'progress' in changes:
            changes['progress'] = json.dumps(changes['progress'], allow_nan=False)
        changes['updated'] = time.time()
        with self.db() as db:
            db.execute('UPDATE jobs SET ' + ','.join(key+'=?' for key in changes) + ' WHERE id=?',
                       (*changes.values(), identifier))
        if changes.get('checkpoint'):
            from .retention import protect
            protect(self.directory('jobs', identifier)/'checkpoints', changes['checkpoint'])

    def pause(self, identifier, owner):
        job = self.job(identifier, owner)
        if job['status'] in ACTIVE:
            self.update(identifier, pause=1, status='pausing')
        return self.job(identifier, owner)

    def record(self, identifier, step, record):
        import resource
        record['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO metrics VALUES(?,?,?)', (identifier, step, json.dumps(record, allow_nan=False)))
        self.update(identifier, progress=record)

    def history(self, identifier, owner, start=None, limit=500, overview=False, end=None):
        job=self.job(identifier, owner)
        if end is not None and (not isinstance(end,int) or start is None or end<start):
            raise ValueError('Intervallo dello storico non valido.')
        if not isinstance(limit,int) or not 1<=limit<=2000 or (start is not None and (not isinstance(start,int) or start<0)):
            raise ValueError('Intervallo dello storico non valido.')
        with self.db() as db:
            if overview:
                last=db.execute('SELECT step FROM metrics WHERE job=? ORDER BY step DESC LIMIT 1',(identifier,)).fetchone()
                signature=(job['updated'],last[0] if last else None)
                cached=self._history_cache.get(identifier)
                if cached and cached[0]==signature:
                    rows=cached[1]
                else:
                    count=db.execute('SELECT COUNT(*) FROM metrics WHERE job=?',(identifier,)).fetchone()[0]
                    if not count:return []
                    stride=max(1,(count+999)//1000)
                    cursor=db.execute('WITH ordered AS (SELECT record, ROW_NUMBER() OVER (ORDER BY step) AS idx '
                                      'FROM metrics WHERE job=?) SELECT record FROM ordered '
                                      'WHERE (idx-1)%?=0 OR idx=? ORDER BY idx',(identifier,stride,count))
                    rows=[]
                    for row in cursor:
                        metric=json.loads(row[0])
                        rows.append((json.dumps({key:metric[key] for key in ('step','training_nll','validation_nll') if key in metric}),))
                    if len(self._history_cache)>=16:
                        self._history_cache.pop(next(iter(self._history_cache)),None)
                    self._history_cache[identifier]=(signature,rows)
            elif start is not None:
                rows=db.execute('SELECT record FROM metrics WHERE job=? AND step>=? AND (? IS NULL OR step<=?) ORDER BY step LIMIT ?',
                                (identifier,start,end,end,limit)).fetchall()
            else:
                rows=db.execute('SELECT record FROM metrics WHERE job=? ORDER BY step DESC LIMIT ?', (identifier,limit)).fetchall()
                rows.reverse()
        return [json.loads(row[0]) for row in rows]

    def create_corpus(self, owner, name, size):
        if not isinstance(size, int) or not 1 <= size <= 8*1024**3:
            raise ValueError('Dimensione corpus non valida (massimo 8 GiB).')
        space = shutil.disk_usage(self.root)
        reserve = min(1024**3, space.total // 4)
        if space.free < size + reserve:
            raise ValueError('Spazio insufficiente per il corpus e la riserva checkpoint.')
        name = Path(name).name
        if not name.lower().endswith(('.txt', '.jsonl')):
            raise ValueError('Carica un corpus UTF-8 .txt o .jsonl (campo text).')
        identifier = uuid.uuid4().hex
        self.directory('corpora', identifier)
        with self.db() as db:
            db.execute('INSERT INTO corpora(id,owner,name,expected,status) VALUES(?,?,?,?,?)',
                       (identifier, owner, name, size, 'uploading'))
        return self.corpus(identifier, owner)

    def corpus(self, identifier, owner, ready=False):
        with self.db() as db:
            row = db.execute('SELECT * FROM corpora WHERE id=? AND owner=?', (identifier, owner)).fetchone()
        if row is None or (ready and row['status'] != 'ready'):
            raise KeyError('Corpus non trovato o caricamento incompleto.')
        return dict(row)

    def corpora(self, owner):
        with self.db() as db:
            return [dict(row) for row in db.execute('SELECT * FROM corpora WHERE owner=?', (owner,))]

    def append_corpus(self, identifier, owner, offset, chunk):
        if not chunk or len(chunk) > 2*1024*1024:
            raise ValueError('Chunk non valido (massimo 2 MiB).')
        space = shutil.disk_usage(self.root)
        if space.free < len(chunk) + min(1024**3, space.total // 4):
            raise ValueError('Upload fermato: riserva di spazio per i checkpoint raggiunta.')
        directory = self.directory('corpora', identifier)
        with (directory / 'upload.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            corpus = self.corpus(identifier, owner)
            if corpus['status'] != 'uploading' or offset != corpus['received'] or offset+len(chunk) > corpus['expected']:
                raise ValueError('Offset upload non valido.')
            with (directory / 'corpus').open('r+b' if (directory / 'corpus').exists() else 'w+b') as stream:
                stream.truncate(offset); stream.seek(offset); stream.write(chunk)
                stream.flush(); os.fsync(stream.fileno())
            with self.db() as db:
                db.execute('UPDATE corpora SET received=? WHERE id=?', (offset+len(chunk), identifier))
        return self.corpus(identifier, owner)

    def finish_corpus(self, identifier, owner):
        corpus = self.corpus(identifier, owner)
        if corpus['received'] != corpus['expected']:
            raise ValueError('Upload incompleto.')
        path = self.directory('corpora', identifier) / 'corpus'
        # Strict incremental UTF-8 validation does not load a multi-GB file.
        with path.open('r', encoding='utf-8') as stream:
            for _ in iter(lambda: stream.read(1024*1024), ''):
                pass
        with self.db() as db:
            db.execute("UPDATE corpora SET status='ready',sha256=? WHERE id=?", (digest_file(path), identifier))
        return self.corpus(identifier, owner, ready=True)

    def resource(self):
        return (self.root / 'accelerator.lock').open('a')

    def busy(self):
        with self.resource() as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return False
            except BlockingIOError:
                return True

    def reconcile(self):
        if self.busy():
            return
        with self.db() as db:
            db.execute("UPDATE jobs SET status='interrupted',error='Worker interrotto: riprendi dall’ultimo checkpoint verificato.' WHERE status IN ('queued','preparing','running','pausing')")
            db.execute("UPDATE generations SET status='failed',error='Worker interrotto.' WHERE status='running'")
            db.execute("UPDATE evaluations SET status='failed',error='Worker interrotto.' WHERE status IN ('running','cancelling')")
