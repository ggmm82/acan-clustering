"""Confronto con e senza memoria, con pesi scelti indipendentemente prima del test."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from acan.lm.concept_lm import ConceptLM
from .conditional_cursor import ConditionalText
from .conditional_eval import losses
from .conditional_memory import judge
from .conditional_run import heldout
from .dyadic_control import sections
from .dyadic_run import corpus
from .run import blocks


def run(data, checkpoint):
    tokenizer = Tokenizer.from_file(str(data / "bpe-8000.json"))
    train, starts = corpus(data, "train", tokenizer)
    valid, vs = corpus(data, "validation", tokenizer)
    test, ts = corpus(data, "test", tokenizer)
    vo = blocks(valid)[1][:32]
    offsets = np.array(
        json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())[
            "fresh_offsets"
        ]
    )
    counter = ConceptLM(tokenizer.get_vocab_size(), 6, 4, budget_bytes=1 << 30)
    for article in np.split(train, starts[1:]):
        counter.add(article)
    counter.settle()
    pieces = [piece for piece, pos in sections(valid, vs, vo) if len(piece) > 1]
    results = {}
    counter_losses = {}
    for name, depth in (("with_document", 4), ("without_document", 0)):
        counter.document = depth
        counter.a.fill(1.0)
        counter.b.fill(1.0)
        counter.judge(pieces)
        counter_losses[name] = np.concatenate(
            [-counter.logp(piece, pos) for piece, pos in sections(test, ts, offsets)]
        )
    del counter
    if checkpoint is None:
        return {
            key: dict(counter=float(loss.mean()), tokens=len(loss))
            for key, loss in counter_losses.items()
        }
    with checkpoint.open("rb") as stream:
        model = ConditionalText(*pickle.load(stream))
    tuning = heldout(model.geometry, valid, vs, vo).set_base(model.base)
    testing = heldout(model.geometry, test, ts, offsets)
    for name in counter_losses:
        if name == "without_document":
            model.words = None
            model.memory = []
            judge(model.reading, tuning, [])
        own = losses(model, testing)()
        other = counter_losses[name]
        delta = (own - other).reshape(-1, 511).mean(1)
        results[name] = dict(
            concepts=float(own.mean()),
            counter=float(other.mean()),
            tokens=len(own),
            difference=float(delta.mean()),
            paired_se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.data, args.checkpoint), indent=2) + "\n")
