"""Contatore di riferimento con gli stessi confini di articolo e blocco della tabella diadica."""

import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer
from acan.lm.concept_lm import ConceptLM
from .dyadic_run import corpus
from .run import blocks


def sections(ids, starts, offsets):
    for offset in offsets:
        boundaries = np.r_[offset, starts[(starts > offset) & (starts < offset + 512)], offset + 512]
        for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
            yield ids[a:b], np.arange(int(i == 0), b - a)


def run(path):
    tokenizer = Tokenizer.from_file(str(path / "bpe-8000.json"))
    ids, starts = corpus(path, "train", tokenizer)
    model = ConceptLM(8000, 6, 4, budget_bytes=1 << 30)
    # The legacy control counts contiguous pieces, here split additionally at article boundaries.
    for article in np.split(ids, starts[1:]):
        model.add(article)
    model.settle()
    valid, vs = corpus(path, "validation", tokenizer)
    _, vo = blocks(valid)
    pieces = [p for p, w in sections(valid, vs, vo[:32]) if len(p) > 1]
    model.judge(pieces)
    test, ts = corpus(path, "test", tokenizer)
    reference = json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())
    offsets = np.asarray(reference["fresh_offsets"])

    def score():
        all_losses = np.concatenate(
            [-model.logp(piece, where) for piece, where in sections(test, ts, offsets)]
        )
        assert len(all_losses) == 64 * 511
        return float(all_losses.mean())

    result = dict(tokens=len(ids), budget_bytes=1 << 30, exact=score(), test_offsets=offsets.tolist())
    model.judge_contexts(pieces, final=True)
    result["suffix_cp"] = score()
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(1):
        args.output.write_text(json.dumps(run(args.data), indent=2) + "\n")
