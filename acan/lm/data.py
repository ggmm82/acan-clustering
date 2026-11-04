"""Resumable streaming tokenization, disjoint validation and deterministic cursor."""
import json
import os
from pathlib import Path
import numpy as np
from .checkpoints import fingerprint
from .store import atomic_json, digest_file


class PreparationPaused(Exception):
    pass


def prepare(source, name, tokenizer, directory, identity, should_pause=lambda: False):
    directory = Path(directory)
    metadata = directory / 'tokens.json'
    token_path = directory / 'tokens.bin'
    marker = fingerprint(identity)
    progress = json.loads(metadata.read_text()) if metadata.exists() else dict(identity=marker, source_offset=0, tokens=0, complete=False)
    if progress['identity'] != marker:
        raise ValueError('Corpus o tokenizer diversi dalla preparazione salvata.')
    if progress['complete']:
        if token_path.stat().st_size != progress['tokens'] * 4 or digest_file(token_path) != progress.get('sha256'):
            raise ValueError('Archivio token incompleto.')
        return progress
    with open(source, 'rb') as text, token_path.open('r+b' if token_path.exists() else 'w+b') as tokens:
        text.seek(progress['source_offset'])
        tokens.truncate(progress['tokens'] * 4); tokens.seek(0, 2)
        records = 0
        while True:
            if should_pause():
                tokens.flush(); os.fsync(tokens.fileno()); atomic_json(metadata, progress)
                raise PreparationPaused()
            line = text.readline(16*1024*1024 + 1)
            if not line:
                break
            if len(line) > 16*1024*1024:
                raise ValueError('Documento oltre 16 MiB: suddividere il corpus in documenti/righe più piccoli.')
            document = line.decode('utf-8')
            if name.lower().endswith('.jsonl'):
                if not document.strip():
                    progress['source_offset'] = text.tell()
                    continue
                parsed = json.loads(document)
                document = parsed.get('text') if isinstance(parsed, dict) else None
                if not isinstance(document, str):
                    raise ValueError('Ogni record JSONL deve avere un campo text stringa.')
            ids = tokenizer.encode(document, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            tokens.write(np.asarray(ids, dtype='<u4').tobytes())
            progress.update(source_offset=text.tell(), tokens=progress['tokens'] + len(ids))
            records += 1
            if records % 100 == 0:
                tokens.flush(); os.fsync(tokens.fileno()); atomic_json(metadata, progress)
        tokens.flush(); os.fsync(tokens.fileno())
    progress['complete'] = True
    progress['sha256'] = digest_file(token_path)
    atomic_json(metadata, progress)
    return progress


class Blocks:
    def __init__(self, path, length, seed=42):
        self.tokens = np.memmap(path, mode='r', dtype='<u4')
        self.length, self.seed = length, seed
        total = len(self.tokens) // (length + 1)
        if total < 3:
            raise ValueError('Corpus troppo piccolo: servono almeno tre blocchi completi (ridurre la lunghezza avanzata).')
        self.train_count = max(1, int(total * .95))
        self.validation_ids = list(range(self.train_count, total))[:8]
        self.cached_epoch = None

    def batch(self, batch_size, cursor):
        import torch
        items = []
        for _ in range(batch_size):
            if cursor['offset'] >= self.train_count:
                cursor['epoch'] += 1; cursor['offset'] = 0
            if self.cached_epoch != cursor['epoch']:
                generator = torch.Generator().manual_seed(self.seed + cursor['epoch'])
                self.order = torch.randperm(self.train_count, generator=generator).tolist()
                self.cached_epoch = cursor['epoch']
            index = self.order[cursor['offset']]
            cursor['offset'] += 1
            start = index * (self.length + 1)
            items.append(np.array(self.tokens[start:start+self.length+1], dtype=np.int64))
        return torch.from_numpy(np.stack(items))

    def validation(self):
        import torch
        for index in self.validation_ids:
            start = index * (self.length + 1)
            yield torch.tensor(np.array(self.tokens[start:start+self.length+1], dtype=np.int64))[None]
