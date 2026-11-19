"""Matched context-depth counters, retrained and calibrated with each document setting."""

import argparse
import json
from pathlib import Path
import numpy as np
from .counter import ConceptLM
from .pair_layers import datasets


def score(data):
    splits = datasets(data)
    train, starts = splits["train"]

    def pieces(split, limit, begin=0):
        tokens, starts = splits[split]
        starts = starts[(starts >= begin) & (starts < limit)]
        for a, b in zip(starts, np.r_[starts[1:], limit]):
            yield tokens[a:b], np.arange(int(a % 512 == 0), b - a)

    tune = [p for p, ix in pieces("validation", 32 * 512) if len(p) > 1]
    test = list(pieces("test", 64 * 512))
    control = list(pieces("validation", 64 * 512, 32 * 512))
    results = {}
    for width in (1, 2, 4, 6):
        model = ConceptLM(8000, width, 0, budget_bytes=1 << 30)
        for article in np.split(train, starts[1:]):
            model.add(article)
        model.settle()
        result = {}
        for depth in (0, min(width, 4)):
            model.document = depth
            model.b = np.ones(depth + 1)
            model.a.fill(1)
            model.judge(tune)
            loss = np.concatenate([-model.logp(p, where) for p, where in test])
            mode = "with_document" if depth else "without_document"
            result[mode] = float(loss.mean())
            result[mode + "_control"] = float(
                np.concatenate([-model.logp(p, ix) for p, ix in control]).mean()
            )
            result[mode + "_block_nll"] = loss.reshape(-1, 511).mean(1).tolist()
        results[str(width)] = result
        print(width, result["without_document"], result["with_document"], flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(score(args.data), indent=2) + "\n")
