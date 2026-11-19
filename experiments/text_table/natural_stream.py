"""One pass over the two original WikiText-103 train shards, same BPE and held-out WikiText blocks."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer

from acan.lm.concept_lm import ConceptLM
from acan.lm.store import digest_file
from .run import encoded_splits
from .refinement import losses, sources
from .run import blocks


def pieces(shard, tokenizer, size=200008):
    eos = tokenizer.token_to_id("[EOS]")
    carry = np.empty(0, np.int32)
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=512, columns=["text"]):
        texts = [text for text in batch.column(0).to_pylist() if text and text.strip()]
        if not texts:
            continue
        rows = tokenizer.encode_batch(texts)
        ids = np.concatenate([carry] + [np.asarray(row.ids + [eos], np.int32) for row in rows])
        end = len(ids) // size * size
        for i in range(0, end, size):
            yield ids[i : i + size]
        carry = ids[end:].copy()
    if len(carry):
        yield carry


def paired(model, texts):
    chosen = model.refinement
    model.refinement = None
    exact = losses(model, texts)
    model.refinement = chosen
    cp = losses(model, texts)
    return dict(
        exact=float(exact.mean()),
        cp=float(cp.mean()),
        paired_se=float((cp - exact).std(ddof=1) / np.sqrt(len(texts))),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="WikiText-2 BPE and held-out splits")
    parser.add_argument(
        "--shards", type=Path, nargs=2, required=True, help="Original WikiText-103 train shards in order"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    valid, test = encoded_splits(args.data, ("validation", "test")).values()
    tune, _ = blocks(valid)
    offsets = json.loads((Path(__file__).parent / "results/wikitext2-refinement-stream.json").read_text())[
        "fresh_offsets"
    ]
    test_pieces = [test[i : i + 512] for i in offsets]
    budget = 256 << 20
    report = dict(
        protocol="WikiText-103 raw train, one pass in file order, no rotation or sampling. Same WikiText held-out blocks and WikiText-2 training tokenizer.",
        source_sha256=sources(args.data),
        shards_sha256={p.name: digest_file(p) for p in args.shards},
        test_offsets=offsets,
        budget_bytes=budget,
        steps=[],
    )
    model = ConceptLM(8000, 6, 4, budget_bytes=budget)
    start = time.monotonic()
    with threadpool_limits(1):
        for shard in args.shards:
            for ids in pieces(shard, tokenizer):
                model.add_context_piece(ids)
            model.judge_contexts(tune[:32], final=True)
            report["steps"].append(
                dict(
                    shard=shard.name,
                    train_tokens=model.tokens,
                    selected_classes=model.refinement.classes if model.refinement else 0,
                    statistics_bytes=model.nbytes(),
                    seconds=time.monotonic() - start,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    validation_control=paired(model, tune[32:]),
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["test"] = paired(model, test_pieces)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
