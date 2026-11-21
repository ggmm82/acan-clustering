"""Checkpoints of the generative concept model and the counts of the document being read.

The generative model is the concept network (acan/lm/net_text.py, explained in acan/MODEL.md); checkpoints of the
earlier model of counts (acan/lm/pair_text.py) are still read. This module saves and loads the checkpoints in the
format every project uses, and counts what followed each context earlier in the same text for that earlier model.
"""
import json
import time
from pathlib import Path

import numpy as np

OLD_CHECKPOINT = ('Questo checkpoint è stato salvato con un modello generativo precedente (contatore di suffissi '
                  'o tabella causale), che non fa più parte del prodotto. Riaddestra il progetto con il modello a concetti.')


def document_counts(keys, valid, ids):
    """For every position: C, N(own token), D of its context among the EARLIER positions of the same text only."""
    n = len(ids)
    position = np.arange(n)

    def earlier(group):
        order = np.lexsort((position, group))
        g = group[order]
        start = np.r_[True, g[1:] != g[:-1]]
        first = np.flatnonzero(start)[np.cumsum(start)-1]
        out = np.empty(n, np.int64)
        out[order] = np.arange(n)-first
        return out
    with np.errstate(over='ignore'):
        pair = keys*np.uint64(1000003)+(ids+1).astype(np.uint64)  # (context, token) as one key
    C = earlier(keys)
    N = earlier(pair)
    novel = (N == 0).astype(np.int64)
    order = np.lexsort((position, keys))
    g, f = keys[order], novel[order]
    start = np.r_[True, g[1:] != g[:-1]]
    running = np.cumsum(f)
    before = np.r_[0, running][np.flatnonzero(start)][np.cumsum(start)-1]
    D = np.empty(n, np.int64)
    D[order] = running-f-before
    out = np.stack((C, N, D)).astype(float)
    out[:, ~valid] = 0
    return out


def memory_limit():
    """Bytes the process may use: the memory of the machine, or the limit of its container if lower."""
    import os
    try:
        total = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        total = 4 << 30
    for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = Path(name).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            total = min(total, int(value))
    return total


def default_budget():
    """Bytes for the statistics: a quarter of the memory the process may use, at most 2 GB.

    Adding two tables holds both, their sum and the sorting of their keys: measured, the process peaks at
    2.5 to 3 times the statistics (2.3 GB with 794 MB, 5.1 GB with 2 GB), plus what it holds anyway, so a
    quarter stays inside the limit.
    """
    return int(min(memory_limit()//4, 2 << 30))


def save(directory, model, state, identity):
    """A checkpoint in the format every project uses (manifest, sums, state), so pause, resume, best and
    milestones work as for any other project. The sums go in ``trainables.safetensors``; there is no optimizer."""
    import os
    import shutil
    import uuid
    from safetensors.numpy import save_file
    from . import checkpoints
    from .store import atomic_json, digest_file, sync_directory
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['step']:012d}-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    temporary, final = root/('.'+name+'.tmp'), root/name
    temporary.mkdir()
    try:
        save_file({key: np.ascontiguousarray(value) for key, value in model.arrays().items()}, str(temporary/'trainables.safetensors'))
        (temporary/'optimizer.pt').write_bytes(b'concept-lm: no optimizer\n')
        atomic_json(temporary/'state.json', state)
        (temporary/'rng.pt').write_bytes(b'concept-lm: no random state\n')
        files = ['trainables.safetensors', 'optimizer.pt', 'state.json', 'rng.pt']
        for filename in files:
            with (temporary/filename).open('rb') as stream:
                os.fsync(stream.fileno())
        manifest = dict(format=1, contract=checkpoints.contract(), identity=identity, step=state['step'],
                        sha256={filename: digest_file(temporary/filename) for filename in files})
        atomic_json(temporary/'manifest.json', manifest)
        sync_directory(temporary)
        os.replace(temporary, final)
        sync_directory(root)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load(directory, expected=None, budget_bytes=None):
    """The model and the state of a checkpoint (verified as every checkpoint)."""
    from safetensors.numpy import load_file
    from . import checkpoints
    checkpoints.verify(directory, expected)
    arrays = load_file(str(Path(directory)/'trainables.safetensors'))
    state = json.loads((Path(directory)/'state.json').read_text())
    return from_checkpoint_arrays(arrays, budget_bytes), state


def from_checkpoint_arrays(arrays, budget_bytes=None):
    """The model of a checkpoint. Only the concept model of text (concept-pairs-v1) is loaded."""
    if 'net_text' in arrays:                                     # the concept network alone
        from .net_text import NetText
        return NetText.from_arrays(arrays, budget_bytes)
    if 'pair_text' not in arrays:
        raise ValueError(OLD_CHECKPOINT)
    from .pair_text import PairText
    return PairText.from_arrays(arrays, budget_bytes)
