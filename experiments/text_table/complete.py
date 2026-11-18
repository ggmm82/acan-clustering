"""Completa un testo con i concetti del modello tabulare addestrato su WikiText-2."""
import argparse
import hashlib
from pathlib import Path
import pickle
import numpy as np
from tokenizers import Tokenizer

from .reading import LevelReading, TextCursor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True, help='Checkpoint sperimentale locale fidato')
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=1, help='Numero massimo di nuovi token')
    parser.add_argument('prompt')
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error('Il numero di token deve essere positivo.')
    with args.table.open('rb') as handle:
        table = pickle.load(handle)
    if getattr(table, 'tokenizer_sha256', None) != hashlib.sha256(args.tokenizer.read_bytes()).hexdigest():
        parser.error('Serve il tokenizer associato al checkpoint salvato da layers --save-table.')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    # Fixed on validation for the published experiment.
    reader = LevelReading(table.counts, [level['parents'] for level in table.model.levels[1:]], kappa=getattr(table, 'kappa_reading', .5))
    cursor = TextCursor(table, reader, observed_context=getattr(table, 'observed_context', 1))
    ids = tokenizer.encode(args.prompt).ids
    cursor.prefill(ids)
    for _ in range(args.tokens):
        token = int(np.argmax(cursor.distribution()))
        if token == tokenizer.token_to_id('[EOS]'):
            break
        ids.append(token)
        cursor.observe(token)
    print(tokenizer.decode(ids))


if __name__ == '__main__':
    main()
