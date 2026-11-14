"""Atomic verified checkpoints, including optimizer/cursor/RNG at step boundaries."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

from .store import atomic_json, digest_file, sync_directory


def contract():
    root = Path(__file__).parent
    names = ['attention.py','grafting.py','trainer.py','data.py','checkpoints.py']
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in names)).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(directory, expected=None):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('format') != 1 or manifest.get('contract') != contract():
        raise ValueError('Checkpoint di un’altra versione: usare l’immagine worker che lo ha creato.')
    if expected is not None and manifest['identity'] != expected:
        raise ValueError('Checkpoint incompatibile con base, corpus, tokenizer o configurazione del training.')
    for name, checksum in manifest['sha256'].items():
        if Path(name).name != name or digest_file(directory / name) != checksum:
            raise ValueError('Checkpoint corrotto o incompleto: ' + name)
    if set(manifest['sha256']) != {'trainables.safetensors', 'optimizer.pt', 'state.json', 'rng.pt'}:
        raise ValueError('Elenco file checkpoint non valido.')
    return manifest


def save(directory, model, optimizer, state, identity, scaler=None):
    import numpy as np
    import torch
    from safetensors.torch import save_file
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['step']:012d}-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    temporary = root / ('.' + name + '.tmp')
    final = root / name
    temporary.mkdir()
    try:
        weights = {key: value.detach().cpu().contiguous().clone() for key, value in model.named_parameters() if value.requires_grad}
        save_file(weights, str(temporary / 'trainables.safetensors'))
        torch.save(dict(optimizer=optimizer.state_dict(), scaler=scaler.state_dict() if scaler else None), temporary / 'optimizer.pt')
        numpy_state = np.random.get_state()
        state = dict(state, numpy_rng=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])
        atomic_json(temporary / 'state.json', state)
        torch.save(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        python=random.getstate()), temporary / 'rng.pt')
        files = ['trainables.safetensors', 'optimizer.pt', 'state.json', 'rng.pt']
        for filename in files:
            with (temporary / filename).open('rb') as stream:
                os.fsync(stream.fileno())
        manifest = dict(format=1, contract=contract(), identity=identity, step=state['step'],
                        sha256={filename: digest_file(temporary / filename) for filename in files})
        atomic_json(temporary / 'manifest.json', manifest)
        sync_directory(temporary)
        os.replace(temporary, final)
        sync_directory(root)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def restore(directory, model, optimizer=None, expected=None, scaler=None, restore_rng=True):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    directory = Path(directory)
    verify(directory, expected)
    weights = load_file(str(directory / 'trainables.safetensors'))
    trainable = {key: value for key, value in model.named_parameters() if value.requires_grad}
    if set(weights) != set(trainable):
        raise ValueError('Parametri trainabili incompatibili con il checkpoint.')
    with torch.no_grad():
        for key, parameter in trainable.items():
            if weights[key].shape != parameter.shape:
                raise ValueError('Dimensione parametro incompatibile: ' + key)
            parameter.copy_(weights[key].to(parameter.device, dtype=parameter.dtype))
    saved = torch.load(directory / 'optimizer.pt', map_location='cpu', weights_only=True)
    if optimizer is not None:
        optimizer.load_state_dict(saved['optimizer'])
    if scaler is not None and saved['scaler']:
        scaler.load_state_dict(saved['scaler'])
    state = json.loads((directory / 'state.json').read_text())
    if restore_rng:
        rng = torch.load(directory / 'rng.pt', map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['cpu'])
        if rng['cuda']:
            if len(rng['cuda']) != torch.cuda.device_count():
                raise ValueError('Numero GPU differente dal checkpoint: usare la stessa configurazione per ripresa riproducibile.')
            torch.cuda.set_rng_state_all(rng['cuda'])
        random.setstate(rng['python'])
        random_np = state['numpy_rng']
        np.random.set_state((random_np[0], np.array(random_np[1], dtype=np.uint32), *random_np[2:]))
    return state


def latest(root, identity):
    candidates = sorted(Path(root).glob('step-*'), reverse=True)
    if not candidates:
        return None
    # Never silently fall back from a corrupt latest complete checkpoint.
    verify(candidates[0], identity)
    return candidates[0]


def retain(root, keep=3, milestone=0):
    """Keep the latest ``keep`` checkpoints; with ``milestone``, one every that many steps is kept for good."""
    directories = sorted(Path(root).glob('step-*'), reverse=True)
    for directory in directories[keep:]:
        step = int(directory.name.split('-')[1])
        if milestone and step and step % milestone == 0:
            os.replace(directory, directory.with_name('milestone-'+directory.name[len('step-'):]))
        else:
            shutil.rmtree(directory)
    sync_directory(root)


def for_resume(root, identity, saved_name=None, completed_steps=0):
    root = Path(root)
    if saved_name and (Path(saved_name).name != saved_name or not (root / saved_name).is_dir()):
        raise ValueError('Checkpoint registrato assente: ripristinare il backup. Non avvio un nuovo training da zero.')
    checkpoint = latest(root, identity)
    if checkpoint is None and (saved_name or completed_steps):
        raise ValueError('Stato appreso senza checkpoint disponibile: ripristinare il backup prima di continuare.')
    return checkpoint
