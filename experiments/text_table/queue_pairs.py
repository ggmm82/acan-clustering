"""Train the website adapter from a fresh tokenizer vocabulary, then score the reference article prefixes."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from tokenizers import Tokenizer

from acan.lm.pair_text import PairText
from .document_pairs import evaluate
from .common import corpus
from .common import blocks


def run(args):
    tokenizer = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    if args.ids:
        ids = np.memmap(args.ids, mode="r", dtype="<u2")
        starts = np.load(args.starts)
    else:
        ids, starts = corpus(args.data, "train", tokenizer)
    val, val_starts = corpus(args.data, "validation", tokenizer)
    tune = []
    for a in blocks(val)[1][:32]:
        boundaries = val_starts[(val_starts > a) & (val_starts < a + 512)] - a
        tune.extend(np.split(val[a : a + 512], boundaries))
    model = PairText(tokenizer.get_vocab_size(), None, args.budget_mib << 20)
    begin = time.monotonic()
    report = dict(tokens=len(ids), articles=len(starts), budget_mib=args.budget_mib, steps=[])
    for a in range(0, len(ids), 1000000):
        b = min(len(ids), a + 1000000)
        l, r = np.searchsorted(starts, [a, b])
        model.add(ids[a:b], ids, tune, lambda: None, documents=starts, starts_in_piece=starts[l:r] - a)
        if a == 0 or b == len(ids) or b % 10000000 == 0:
            report["steps"].append(
                dict(
                    tokens=b,
                    seconds=time.monotonic() - begin,
                    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    nbytes=model.nbytes(),
                )
            )
            print(report["steps"][-1], flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["concepts"] = model.concepts()
    report["scores"] = evaluate(model.model, args.data, retune=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ids", type=Path)
    p.add_argument("--starts", type=Path)
    p.add_argument("--budget-mib", type=int, default=768)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
