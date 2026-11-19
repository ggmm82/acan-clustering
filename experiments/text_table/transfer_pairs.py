"""Stream WikiText-103 into the selected pair graph and the control with equal count-table budgets."""

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

from acan.lm.pair_training import add_chunk
from .counter import ConceptLM
from .long_counter import evaluation
from .document_pairs import evaluate
from tokenizers import Tokenizer


def run(args):
    with args.model.open("rb") as f:
        model = pickle.load(f)
    model.tables = {}  # Relearn the global histograms on the new corpus.
    ids = np.memmap(args.ids, dtype=np.uint16, mode="r")
    starts = np.load(args.starts)
    tokenizer = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    control = ConceptLM(tokenizer.get_vocab_size(), 6, 4, budget_bytes=args.budget_mib << 20)
    tables = {}
    tail = np.zeros(0, np.int32)
    # Per-article calls preserve the counter's suffix at piece boundaries.
    tick = time.monotonic()
    report = dict(tokens=len(ids), articles=len(starts), budget_mib=args.budget_mib, steps=[])
    for a in range(0, len(ids), 500000):
        b = min(a + 500000, len(ids))
        piece = np.asarray(ids[a:b], dtype=np.int32)
        boundaries = starts[np.searchsorted(starts, a) : np.searchsorted(starts, b)] - a
        tail = add_chunk(model, piece, boundaries, tail, tables, args.budget_mib << 20)
        for lo, hi in zip(
            np.unique(np.r_[0, boundaries]), np.r_[boundaries[boundaries > 0], len(piece)]
        ):
            if lo or (len(boundaries) and boundaries[0] == 0):
                control.context_tail = None
            control.add_context_piece(piece[lo:hi])
        if not a or b == len(ids) or b // 500000 % 20 == 0:
            report["steps"].append(
                dict(
                    tokens=b,
                    seconds=time.monotonic() - tick,
                    table_bytes=sum(t.nbytes() for t in tables.values()),
                    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
            )
            print(report["steps"][-1], flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    model.tables = tables
    report["native"] = evaluate(model, args.data, retune=True)
    t, c = evaluation(args.data, "validation", tokenizer)
    (e,) = evaluation(args.data, "test", tokenizer)
    control.settle()
    for depth in (0, 4):
        control.document = depth
        control.a.fill(1)
        control.b.fill(1)
        control.judge(*t)
        report["counter_" + str(depth)] = {
            name: float(np.concatenate([-control.logp(p, i) for p, i in zip(*group)]).mean())
            for name, group in [("control", c), ("test", e)]
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("model", "ids", "starts", "data", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--budget-mib", type=int, default=256)
    run(p.parse_args())
