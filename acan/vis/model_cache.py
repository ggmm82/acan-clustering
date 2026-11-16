"""Trained models kept on the server: the page holds only a short reference to them.

A model of a large table is several megabytes of JSON. Kept in a ``dcc.Store`` it would travel to the browser
and back to the server with every callback that reads it (the graph, the evaluation, the list of concepts...),
and the page would seem stuck after training. ``keep`` writes the JSON once to a cache folder, named by its
SHA-256, and returns ``acan-model:<sha>``; ``resolve`` gives the JSON back for a reference and leaves any other
value as it is (a model uploaded from a file before this change, or by a test). Files are shared by every
process of the server; the oldest are removed beyond a size limit. On the server the cache is a volume
(``ACAN_MODEL_CACHE``), so a model survives a restart or an update of the server.
"""
import hashlib
import os
import re
import tempfile
from pathlib import Path

PREFIX = 'acan-model:'
PERSISTENT_BYTES = 512 << 20                                   # on the server's volume (survives restarts)
TEMPORARY_BYTES = 96 << 20                                     # in /tmp, which is small and lost at a restart


def folder():
    """The cache: ACAN_MODEL_CACHE (a volume of the server) when it can be written, else the temporary folder."""
    preferred = os.environ.get('ACAN_MODEL_CACHE')
    if preferred:
        path = Path(preferred)
        try:
            path.mkdir(parents=True, exist_ok=True)
            if os.access(path, os.W_OK):
                return path, PERSISTENT_BYTES
        except OSError:
            pass
    path = Path(tempfile.gettempdir())/'acan-models'
    path.mkdir(parents=True, exist_ok=True)
    return path, TEMPORARY_BYTES


def _prune(root, limit):
    files = sorted(root.glob('*.json'), key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in files)
    for old in files[:-1]:                                      # the newest model always stays
        if total <= limit:
            break
        total -= old.stat().st_size
        old.unlink(missing_ok=True)


def keep(model_json):
    """The reference of a model's JSON, written to the cache if it is not there yet.

    If the cache cannot be written (a full or read-only disk) the JSON itself is kept in the page, as before.
    """
    if not isinstance(model_json, str) or model_json.startswith(PREFIX):
        return model_json
    digest = hashlib.sha256(model_json.encode()).hexdigest()
    try:
        root, limit = folder()
        target = root/f'{digest}.json'
        if target.exists():
            target.touch()
        else:
            temporary = root/f'.{digest}.{os.getpid()}.tmp'
            try:
                temporary.write_text(model_json, encoding='utf-8')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            _prune(root, limit)
    except OSError:
        return model_json
    return PREFIX+digest


def resolve(data):
    """The JSON of a reference; any other value unchanged."""
    if not isinstance(data, str) or not data.startswith(PREFIX):
        return data
    digest = data[len(PREFIX):]
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Riferimento al modello non valido.')
    path = folder()[0]/f'{digest}.json'
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise ValueError('Il modello non è più sul server (riavvio o pulizia): ricaricalo dal file scaricato o addestralo di nuovo.') from None


def load_graph(data):
    from acan import GraphClusterModel
    return GraphClusterModel.loadModel(resolve(data))
