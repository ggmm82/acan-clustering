"""Keep the best immutable checkpoint outside the rolling latest-three window.

This changes retention only; the attention/trainer/checkpoint format and their
existing compatibility contract remain untouched.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import uuid
from .store import atomic_json, sync_directory


def _link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def protect(root, name):
    root = Path(root)
    if not name or Path(name).name != name:
        raise ValueError('Checkpoint non valido.')
    source = root / name
    if not (source / 'manifest.json').is_file() or not (source / 'state.json').is_file():
        return None
    try:
        state = json.loads((source / 'state.json').read_text())
    except FileNotFoundError:
        return None  # rolling retention removed a stale scan entry
    loss = state.get('validation_nll')
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        return None
    with (root / '.best.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pointer = root / 'best.json'
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
        if previous and previous['validation_nll'] <= loss and (root/previous['name']/'manifest.json').exists():
            return previous
        from .checkpoints import verify
        try:
            verify(source)
        except FileNotFoundError:
            return previous
        temporary = root / ('.best-'+uuid.uuid4().hex+'.tmp')
        destination = root / ('best-'+uuid.uuid4().hex)
        try:
            shutil.copytree(source, temporary, copy_function=_link_or_copy)
            sync_directory(temporary)
            os.replace(temporary, destination)
            sync_directory(root)
            value = dict(name=destination.name, source=name, step=state['step'], validation_nll=float(loss))
            atomic_json(pointer, value)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        for old in root.glob('best-*'):
            if old.is_dir() and old != destination:
                shutil.rmtree(old)
        sync_directory(root)
        return value


def protect_available(root):
    root = Path(root)
    # For older projects, recover the best of the checkpoints that still exist.
    # A checkpoint already deleted by a previous release cannot be recovered.
    for directory in sorted(root.glob('step-*')):
        if (directory/'manifest.json').exists():
            try:
                protect(root,directory.name)
            except (FileNotFoundError,ValueError):
                # An incompatible historical checkpoint remains listed for
                # inspection; it must not break the entire workspace listing.
                continue
