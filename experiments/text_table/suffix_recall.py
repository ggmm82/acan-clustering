"""Test the same one-shot facts, branching before revealing any query answer."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from acan.lm.concept_lm import ConceptLM
from acan.lm.context_refinement import SuffixCursor
from .run import encoded_splits
from .recall import recall_inputs
from .run import blocks
from .refinement import sources


def probe(model, pool, filler):
    gaps = (1000, 10000, 100000, 300000)
    cues, answers, facts, prefix = recall_inputs(pool, filler, 8, 42, 16, gaps)
    report = dict(facts_sha256=hashlib.sha256(facts.astype("<i8").tobytes()).hexdigest(), runs=[])
    chosen = model.refinement
    for gap in gaps:
        rows = {}
        for i, cue in enumerate(cues):
            start = time.perf_counter()
            cursor = SuffixCursor(model).prefill(prefix[: len(facts) + gap]).prefill(cue)
            prefill = time.perf_counter() - start
            task = "association" if i < 16 else "copy"
            candidates = answers[:16] if i < 16 else answers[16:]
            for name, refinement in [("exact", None), ("cp", chosen)]:
                model.refinement = refinement
                start = time.perf_counter()
                p = cursor.distribution()
                decode = time.perf_counter() - start
                np.testing.assert_allclose(p.sum(), 1.0, atol=1e-10)
                y = answers[i]
                rank = 1 + (p > p[y]).sum() + 0.5 * ((p == p[y]).sum() - 1)
                rows.setdefault((task, name), []).append(
                    [p.argmax() == y, candidates[p[candidates].argmax()] == y, rank, prefill, decode]
                )
        for (task, name), results in rows.items():
            hit, closed, rank, prefill, decode = np.asarray(results).T
            report["runs"].append(
                dict(
                    task=task,
                    reading=name,
                    gap=gap,
                    trials=len(hit),
                    full_top1=int(hit.sum()),
                    candidate_top1=int(closed.sum()),
                    median_rank=float(np.median(rank)),
                    median_prefill_s=float(np.median(prefill)),
                    median_distribution_ms=float(np.median(decode) * 1000),
                )
            )
    model.refinement = chosen
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = encoded_splits(args.data, ("train", "validation", "test"))
    train = data["train"]
    model = ConceptLM(8000, 6, 4, budget_bytes=1024 << 20)
    with threadpool_limits(1):
        for i in range(0, len(train), 32768):
            model.add_context_piece(train[i : i + 32768])
        model.settle()
        model.judge_contexts(blocks(data["validation"])[0][:32], final=True)
        pool = np.unique(train[8:200008])
        report = probe(model, pool, data["test"])
    report["source_sha256"] = sources(args.data)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
