"""Trained models kept on the server: the page holds only a short reference to them.

A model of a large table is several megabytes of JSON. Kept in a ``dcc.Store`` it would travel to the browser
and back to the server with every callback that reads it (the graph, the evaluation, the list of concepts...),
and the page would seem stuck after training. ``keep`` writes the JSON once to a cache folder, named by its
SHA-256, and returns ``acan-model:<sha>``; ``resolve`` gives the JSON back for a reference and leaves any other
value as it is (a model uploaded from a file before this change, or by a test). Files are shared by every
process of the server and the oldest are removed beyond ``LIMIT``.
"""
import hashlib
import os
import re
import tempfile
from pathlib import Path

PREFIX = 'acan-model:'
LIMIT = 200


def folder():
    path = Path(os.environ.get('ACAN_MODEL_CACHE') or Path(tempfile.gettempdir())/'acan-models')
    path.mkdir(parents=True, exist_ok=True)
    return path


def keep(model_json):
    """The reference of a model's JSON, written to the cache if it is not there yet."""
    if not isinstance(model_json, str) or model_json.startswith(PREFIX):
        return model_json
    digest = hashlib.sha256(model_json.encode()).hexdigest()
    root = folder()
    target = root/f'{digest}.json'
    if not target.exists():
        temporary = root/f'.{digest}.{os.getpid()}.tmp'
        temporary.write_text(model_json, encoding='utf-8')
        os.replace(temporary, target)
        files = sorted(root.glob('*.json'), key=lambda path: path.stat().st_mtime)
        for old in files[:max(len(files)-LIMIT, 0)]:
            old.unlink(missing_ok=True)
    else:
        target.touch()
    return PREFIX+digest


def resolve(data):
    """The JSON of a reference; any other value unchanged."""
    if not isinstance(data, str) or not data.startswith(PREFIX):
        return data
    digest = data[len(PREFIX):]
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Riferimento al modello non valido.')
    path = folder()/f'{digest}.json'
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        raise ValueError('Il modello non è più sul server (riavvio o pulizia): ricaricalo dal file scaricato o addestralo di nuovo.') from None


def load_graph(data):
    from acan import GraphClusterModel
    return GraphClusterModel.loadModel(resolve(data))
