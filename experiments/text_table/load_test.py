"""Bounded-memory stress run, using repeated WikiText with a rotated vocabulary (not a quality test)."""

import argparse
import json
import resource
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits

from acan.lm.concept_lm import ConceptLM
from .run import blocks
from .refinement import sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-mib", type=int, default=1024)
    args = parser.parse_args()
    from .layers import encoded_splits

    data = encoded_splits(args.data, ("train", "validation"))
    source = data["train"]
    valid, _ = blocks(data["validation"])
    budget = 256 << 20
    model = ConceptLM(8000, 6, 4, budget_bytes=budget)
    targets = np.unique([min(m << 18, args.input_mib << 18) for m in (16, 64, 256, args.input_mib)])
    report = dict(
        protocol="Rotated replay of WikiText training ids, 4 bytes/token. Operational load only.",
        budget_bytes=budget,
        source_sha256=sources(args.data),
        steps=[],
    )
    start = time.monotonic()
    with threadpool_limits(1):
        for end in targets:
            while model.tokens < end:
                offset = model.tokens % len(source)
                length = min(200008, len(source) - offset, end - model.tokens)
                ids = (source[offset : offset + length] + (model.tokens // len(source)) * 29) % model.V
                model.add_context_piece(ids)
            model.settle()
            model.judge_contexts(valid[:32], final=True)
            report["steps"].append(
                dict(
                    input_bytes=int(model.tokens * 4),
                    seconds=time.monotonic() - start,
                    statistic_bytes=model.nbytes(),
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    exact_contexts=model.concepts(),
                    selected_concepts=len(np.unique(model.refinement.mapping)) if model.refinement else 0,
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
