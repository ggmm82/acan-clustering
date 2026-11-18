"""Train and reload the standard web queue on WikiText-2 with the optional CP reading."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import pyarrow.parquet as pq
from acan.lm.concept_lm import load
from acan.lm.concept_trainer import run_training
from acan.lm.research_data import Blocks
from acan.lm.research_options import validate_options
from acan.lm.store import Store, owner_key
from acan.vis.long_training import profile
from .natural_stream import paired
from .refinement import sources


def run(data_dir, root):
    start = time.perf_counter()
    store = Store(root)
    owner = owner_key("c" * 64)
    records = []
    for split in ("train", "validation", "test"):
        texts = pq.read_table(data_dir / (split + ".parquet"), columns=["text"]).column(0)
        for text in texts.to_pylist():
            if text.strip():
                records.append(json.dumps(dict(text=text, split=split), ensure_ascii=False))
    raw = ("\n".join(records) + "\n").encode()
    corpus = store.create_corpus(owner, "wiki.jsonl", len(raw))
    for i in range(0, len(raw), 2 << 20):
        store.append_corpus(corpus["id"], owner, i, raw[i : i + (2 << 20)])
    store.finish_corpus(corpus["id"], owner)
    config = dict(
        profile(False),
        corpus_id=corpus["id"],
        context_refinement=True,
        concept_memory_mb=256,
        name="WikiText-2",
    )
    job = store.create_job(owner, validate_options(config))
    run_training(root, job["id"])
    result = store.job(job["id"])
    if result["status"] != "completed":
        raise ValueError(result["error"])
    folder = store.directory("jobs", job["id"])
    model, state = load(folder / "checkpoints" / result["checkpoint"])
    blocks = Blocks(folder, config["sequence_length"])
    indices = blocks.evaluation_indices("test", 64)
    return dict(
        protocol="Local long-training worker with web profile, 32k BPE, 256 MiB, WikiText-2 train/validation/test JSONL with explicit splits and duplicate removal.",
        corpus_sha256=hashlib.sha256(raw).hexdigest(),
        source_sha256=sources(data_dir),
        step=state["step"],
        train_tokens=model.tokens,
        vocabulary=model.V,
        seconds=time.perf_counter() - start,
        document_tokens=state["dataset"]["tokens"],
        selected_classes=state["context_classes"],
        pooled_contexts=state["pooled_contexts"],
        validation_nll=state["validation_nll"],
        baseline_before_file=state["baseline_nll"],
        bytes=model.nbytes(),
        test=paired(model, [blocks._block("test", i) for i in indices]),
        indices=indices,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        receipt = run(args.data, directory)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
