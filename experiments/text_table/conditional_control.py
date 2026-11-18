"""Confronto appaiato del checkpoint selezionato con il contatore, articolo per articolo."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from acan.lm.concept_lm import ConceptLM
from .conditional_cursor import ConditionalText
from .conditional_eval import losses
from .conditional_run import heldout
from .dyadic_control import sections
from .dyadic_run import corpus
from .run import blocks


def measure(path, checkpoint, offsets):
    tokenizer = Tokenizer.from_file(str(path / "bpe-8000.json"))
    train, starts = corpus(path, "train", tokenizer)
    counter = ConceptLM(tokenizer.get_vocab_size(), 6, 4, budget_bytes=1 << 30)
    for article in np.split(train, starts[1:]):
        counter.add(article)
    counter.settle()
    valid, vs = corpus(path, "validation", tokenizer)
    pieces = [p for p, w in sections(valid, vs, blocks(valid)[1][:32]) if len(p) > 1]
    counter.judge(pieces)
    test, ts = corpus(path, "test", tokenizer)
    reference = np.concatenate([-counter.logp(x, where) for x, where in sections(test, ts, offsets)])
    del counter
    with checkpoint.open("rb") as f:
        model = ConditionalText(*pickle.load(f))
    candidate = losses(model, heldout(model.geometry, test, ts, offsets))()
    delta = (candidate - reference).reshape(-1, 511).mean(1)
    return dict(
        counter_nll=float(reference.mean()),
        candidate_nll=float(candidate.mean()),
        difference=float(delta.mean()),
        paired_block_se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
        improved_blocks=int((delta < 0).sum()),
        blocks=len(delta),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    offsets = json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())[
        "fresh_offsets"
    ]
    result = measure(args.data, args.checkpoint, np.asarray(offsets))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
