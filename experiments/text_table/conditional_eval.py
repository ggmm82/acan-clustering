"""Un solo test del checkpoint scelto in validazione, con controlli di causalità e richiamo."""

import argparse
import json
import pickle
from pathlib import Path
import resource

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from acan.concept_algebra import categorical_reading
from .conditional_cursor import ConditionalText
from .conditional_memory import document_stats
from .conditional_run import Rows, heldout
from .dyadic import TARGET
from .dyadic_run import corpus, recall


def extent(parts):
    return max((lag + (1 << j) for lag, j, b in parts), default=0)


def losses(model, rows):
    rows.set_base(model.base)
    global_stats = [(f, k, t.lookup(rows.keys(f), rows.y)) for f, t, k in model.reading.levels]
    local_stats = [(f, k, document_stats(rows, f)) for f, k in model.memory]

    from .word_regions import observed_stats

    words = model.words
    word_stats = (
        observed_stats(words, model.reading.prior, rows, [s for s, k, b in words.weights]) if words else {}
    )

    def read(global_filter=lambda f: True, local_filter=lambda f: True, with_words=True):
        p = model.reading.prior[rows.y].copy() if rows.initial is None else rows.initial.copy()
        for levels, include in ((global_stats, global_filter), (local_stats, local_filter)):
            for f, k, (m, n, d) in levels:
                if include(f):
                    p = categorical_reading(n, m, d, p, k, np)
        if words and with_words:
            for slot, k, beta in words.weights:
                m, n, d, ratio = word_stats[slot]
                p = categorical_reading(n, m, d, p, 1 + k * ratio**beta, np)
        return -np.log(p)

    return read


def native_parity(model, tokens):
    rows = Rows(model.geometry, np.r_[tokens, 0], np.array([0]), np.arange(len(tokens) + 1))
    features = [f for f, t, k in model.reading.levels]
    frame = pd.DataFrame(
        np.column_stack([rows.keys(f) for f in features]), columns=pd.Index(features, tupleize_cols=False)
    )
    if model.base is not None:
        from .dyadic import frame as base_frame

        levels, age = model.base.levels(rows.tokens, rows.starts)
        x, c = model.base.context(levels, rows.pos, age)
        frame = pd.concat([base_frame(x, c, model.base.columns), frame], axis=1)
    native, _ = model.graph.predict_proba(frame, TARGET)
    cursor = model.cursor()
    p = np.exp(-losses(model, rows)())
    diff = 0.0
    for i, y in enumerate(rows.y):
        np.testing.assert_allclose(cursor.distribution()[y], p[i], rtol=1e-9, atol=1e-12)
        q = model.reading.distribution(
            cursor.current.__getitem__, cursor.base.distribution() if cursor.base else None
        )
        diff = max(diff, float(np.max(np.abs(q - native[i]))))
        if i < len(tokens):
            cursor.observe(y)
    np.testing.assert_allclose(
        model.cursor().prefill(tokens).distribution(), cursor.distribution(), atol=1e-12
    )
    return diff


def main():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--data", type=Path, required=True)
    command.add_argument(
        "--candidates",
        nargs="+",
        type=Path,
        required=True,
        help="Ciascun JSON di conditional_run ha il checkpoint .pkl accanto.",
    )
    command.add_argument("--output", type=Path, required=True)
    args = command.parse_args()
    reports = [json.loads(p.read_text()) for p in args.candidates]
    selected = min(range(len(reports)), key=lambda i: reports[i]["tuning"])
    with args.candidates[selected].with_suffix(".pkl").open("rb") as stream:
        model = ConditionalText(*pickle.load(stream))
    report = reports[selected].copy()
    report["candidates"] = [
        dict(
            embedding=r["embedding"],
            word_coordinate="word_coordinate" in r,
            dimensions=r["dimensions"],
            neighbors=r["neighbors"],
            tuning=r["tuning"],
            control=r["validation_control"],
        )
        for r in reports
    ]
    tok = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    train, starts = corpus(args.data, "train", tok)
    test, ts = corpus(args.data, "test", tok)
    offsets = np.asarray(
        json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())[
            "fresh_offsets"
        ]
    )
    rows = heldout(model.geometry, test, ts, offsets)
    read = losses(model, rows)
    full = read()
    local = read(lambda f: extent(f) <= 1, lambda f: extent(f) <= 1, with_words=False)
    static = read(local_filter=lambda f: False, with_words=False)
    report["test"] = dict(
        offsets=offsets.tolist(),
        tokens=len(full),
        nll=float(full.mean()),
        without_document=float(static.mean()),
        without_word_coordinate=float(read(with_words=False).mean()),
        last_token_and_document_unigram=float(local.mean()),
        bands={
            f"{a}-{b-1}": float(full.reshape(-1, 511)[:, a - 1 : b - 1].mean())
            for a, b in ((1, 32), (32, 128), (128, 256), (256, 512))
        },
    )
    report["test"]["omit_context_extent"] = {
        str(2**j): float(read(lambda f: extent(f) != 2**j, lambda f: extent(f) != 2**j).mean())
        for j in range(model.geometry.scales)
    }
    # Only this diagnostic uses literal token pairs. Their membership never enters the predictor.
    v = len(model.geometry.vectors)
    ok = np.ones(len(train) - 1, bool)
    ok[starts[1:] - 1] = False
    seen = np.unique((train[:-1].astype(np.int64) * v + train[1:])[ok])
    positions = rows.pos
    unseen = (rows.age[positions] > 2) & ~np.isin(
        rows.tokens[np.maximum(positions - 2, 0)].astype(np.int64) * v + rows.tokens[positions - 1], seen
    )
    report["unseen_literal_pair"] = dict(
        tokens=int(unseen.sum()), nll=float(full[unseen].mean()), only_latest=float(local[unseen].mean())
    )
    # Scramble the semantic identities of the known attributes. The true token stays in the target.
    rng = np.random.default_rng(617)
    for j, col in enumerate(rows.columns):
        codes = np.unique(col[col != 0])
        permutation = np.arange(2 ** (model.geometry.depths[j] + 1), dtype=np.uint32)
        permutation[codes] = rng.permutation(codes)
        rows.columns[j] = permutation[col]
    if model.words:
        original = model.words.codes
        model.words.codes = original[rng.permutation(len(original))]
    report["shuffled_concepts_nll"] = float(losses(model, rows)().mean())
    if model.words:
        model.words.codes = original
    unused = np.setdiff1d(np.arange(len(test) // 512) * 512, offsets)
    fresh = np.sort(np.random.default_rng(8071).choice(unused, 64, replace=False))
    confirmation = losses(model, heldout(model.geometry, test, ts, fresh))
    report["additional_test"] = dict(
        offsets=fresh.tolist(),
        nll=float(confirmation().mean()),
        without_word_coordinate=float(confirmation(with_words=False).mean()),
    )
    report["max_difference_native_static"] = native_parity(model, test[:48])
    np.testing.assert_allclose(report["max_difference_native_static"], 0, atol=1e-10)
    report["checkpoint_bytes"] = args.candidates[selected].with_suffix(".pkl").stat().st_size
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["recall"] = recall(model, train, test, tok)
    report["evaluation_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
