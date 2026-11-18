"""Compare paired next-token losses under one shared storage limit."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits
from acan.lm.concept_lm import ConceptLM
from acan.lm.context_refinement import ContextRefinement, partitions
from .run import blocks


def sources(data):
    return {
        name: hashlib.sha256((data / name).read_bytes()).hexdigest()
        for name in ("bpe-8000.json", "train.parquet", "validation.parquet", "test.parquet")
    }


def losses(model, texts):
    return np.array([-model.logp(text, np.arange(1, len(text))).mean() for text in texts])


def truncate_context(model, texts):
    """Ablate just the available suffix lengths in the fitted reader, keeping weights fixed."""
    nll = {depth: [] for depth in (0, 1, 2, 3, model.depth)}
    for text in texts:
        counts, document, prior, *extra = model._prepared(0, text, np.arange(1, len(text)))
        for depth in nll:
            cache = (counts[: depth + 1], document[: depth + 1], prior, *extra)
            nll[depth].append(-model._factor(0, cache).mean())
    return {depth: float(np.mean(scores)) for depth, scores in nll.items()}


def run(train, validation, test, budget, continuous=False):
    start = time.monotonic()
    model = ConceptLM(8000, depth=6, document=4, budget_bytes=budget)
    for i in range(0, len(train), 32768):
        if continuous:
            model.add_context_piece(train[i : i + 32768])
        else:
            model.add(train[i : i + 32768])
    model.settle()
    valid, _ = blocks(validation)
    standard, offsets = blocks(test)
    # Exclude every test block used by the previous ablations.
    previous = json.loads((Path(__file__).parent / "results/wikitext2-context.json").read_text())
    excluded = np.r_[offsets, previous["fresh_test_offsets"], previous["conditional_counts"]["fresh_offsets"]]
    pool = np.setdiff1d(np.arange(len(test) // 512) * 512, excluded)
    fresh = np.sort(np.random.default_rng(20261120).choice(pool, 64, replace=False))
    groups = {
        "historical": standard,
        "fresh": [test[i : i + 512] for i in fresh],
        "validation_control": valid[32:],
    }
    model.judge(valid[:32])
    baseline = {key: losses(model, texts) for key, texts in groups.items()}
    report = dict(
        continuous=continuous,
        tokens=len(train),
        budget_bytes=budget,
        baseline_bytes=model.nbytes(),
        fresh_offsets=fresh.tolist(),
        exact_contexts=model.concepts(),
        baseline_seconds=time.monotonic() - start,
    )
    start = time.monotonic()
    maps = partitions(model)
    chosen = ContextRefinement.select(model, valid[:32], maps, model.tokens)
    report.update(
        cp_cuts=[len(np.unique(m)) for m in maps],
        refinement_seconds=time.monotonic() - start,
        selected_classes=len(np.unique(chosen.mapping)) if chosen else None,
        strengths=chosen.weights.tolist() if chosen else [],
        comparisons={},
    )
    model.refinement = chosen
    model.context_trained_at = model.tokens
    report["candidate_bytes"] = model.nbytes()
    report["pooled_contexts"] = {d: len(t.keys) for d, t in chosen.tables.items()} if chosen else {}
    for key, texts in groups.items():
        measured = losses(model, texts)
        delta = measured - baseline[key]
        report["comparisons"][key] = dict(
            baseline=float(baseline[key].mean()),
            candidate=float(measured.mean()),
            paired_se=float(delta.std(ddof=1) / np.sqrt(len(delta))),
        )
    if continuous:
        report["fresh_nll_by_visible_suffix"] = truncate_context(model, groups["fresh"])
    if chosen:
        # Identical class sizes and tuning protocol, random token identities.
        model.refinement = None
        shuffled = np.random.default_rng(18).permutation(chosen.mapping)
        model.refinement = ContextRefinement.select(model, valid[:32], [shuffled], model.tokens)
        report["shuffled_fresh"] = float(losses(model, groups["fresh"]).mean())
        model.refinement = chosen
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--budget-mib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--continuous", action="store_true", help="Carry the suffix across I/O boundaries in both readings"
    )
    args = parser.parse_args()
    from .layers import encoded_splits

    arrays = encoded_splits(args.data, ("train", "validation", "test"))
    with threadpool_limits(1):
        _, report = run(**arrays, budget=args.budget_mib << 20, continuous=args.continuous)
    report["source_sha256"] = sources(args.data)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
