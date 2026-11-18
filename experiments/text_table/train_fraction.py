"""Ricostruisce il train ridotto da articoli interi e apprende il BPE soltanto dai testi scelti."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def subset(source, destination, fraction=0.1):
    if not 0 < fraction < 1:
        raise ValueError("Quota fuori intervallo.")
    groups = []
    for line in pq.read_table(source / "train.parquet", columns=["text"])["text"].to_pylist():
        if not line or not line.strip():
            continue
        title = line.strip()
        if not groups or (
            title.startswith("= ") and title.endswith(" =") and not title.startswith("= =")
        ):
            groups.append([])
        groups[-1].append(line)
    number = round(fraction * len(groups))
    chosen = np.sort(np.random.default_rng(42).choice(len(groups), number, replace=False))
    lines = [line for i in chosen for line in groups[i]]
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": lines}), destination / "train.parquet")
    for split in ("validation", "test"):
        shutil.copyfile(source / (split + ".parquet"), destination / (split + ".parquet"))
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        lines,
        trainer=trainers.BpeTrainer(
            vocab_size=8000,
            special_tokens=["[UNK]", "[EOS]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.save(str(destination / "bpe-8000.json"))
    return dict(
        seed=42,
        articles_selected=number,
        articles_total=len(groups),
        lines=len(lines),
        article_ids=chosen.tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    (args.destination / "subset.json").write_text(
        json.dumps(subset(args.data, args.destination), indent=2) + "\n"
    )
