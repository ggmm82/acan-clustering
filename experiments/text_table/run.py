"""Prepara il tokenizer BPE 8.000 del protocollo, imparato solo sul train di WikiText-2.

python -m experiments.text_table.run --data DIR --output preparazione.json

Scrive DIR/bpe-8000.json e DIR/bpe-source.sha256 (l'impronta del train da cui è stato imparato).
Un tokenizer già presente si riusa solo se è stato imparato dallo stesso train.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def tokenizer_for(data):
    texts = {name: [str(t) for t in pd.read_parquet(data/(name+'.parquet'))['text'] if str(t).strip()]
             for name in ('train', 'validation', 'test')}
    source = hashlib.sha256((data/'train.parquet').read_bytes()).hexdigest()
    path, identity = data/'bpe-8000.json', data/'bpe-source.sha256'
    if path.exists() and identity.exists() and identity.read_text() == source:
        return Tokenizer.from_file(str(path)), texts, source
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts['train'], trainers.BpeTrainer(
        vocab_size=8000, special_tokens=['[UNK]', '[EOS]'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.save(str(path))
    identity.write_text(source)
    return tokenizer, texts, source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tokenizer, texts, source = tokenizer_for(a.data)
    eos = tokenizer.token_to_id('[EOS]')
    tokens = {name: int(sum(len(piece.ids)+1 for piece in tokenizer.encode_batch(lines))) for name, lines in texts.items()}
    report = dict(train_sha256=source, tokenizer_sha256=hashlib.sha256((a.data/'bpe-8000.json').read_bytes()).hexdigest(),
                  vocabulary=tokenizer.get_vocab_size(), eos=eos, tokens=tokens)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    np.seterr(all='ignore')
    main()
