"""Nested native CP scales. Keep complete evaluation articles and score only the original offsets."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table
from acan.lm.pair_training import geometry, finer_spans
from .common import corpus
from .pair_layers import age, route_pairs
from .pair_reading import Columns, compose_exact, root_probabilities, stages, read
from acan.core.conditional import calibrated_weights
from .common import blocks


def full_articles(data, folder, split):
    tok = Tokenizer.from_file(str(data / "bpe-8000.json"))
    tokens, starts = corpus(data, split, tok)
    offsets = (
        json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())[
            "fresh_offsets"
        ]
        if split == "test"
        else blocks(tokens)[1]
    )
    row = Columns.__new__(Columns)
    row.y, row.starts, row.age = tokens, starts, age(tokens, starts)
    row.pos = np.concatenate([np.arange(a + 1, a + 512) for a in offsets])
    row.spans = {"e1": tokens.astype(np.uint32) + 1}
    with (folder / "root.pkl").open("rb") as f:
        graph, vectors = pickle.load(f)
    ids = graph.assignments(pd.DataFrame(vectors, columns=graph.encoder.numeric)).argmax(1) + 1
    row.spans["c1"] = ids[tokens]
    child = tokens
    for width in (2, 4):
        with (folder / f"pair-{width}.pkl").open("rb") as f:
            graph, centres = pickle.load(f)
        child = route_pairs(graph, child, vectors, width // 2, np.flatnonzero(row.age >= width - 1))
        row.spans[f"c{width}"] = child
        vectors = centres
    row.target = row.y[row.pos]
    return row


def restore_local(splits, folder):
    for width in (2, 4):
        with (folder / f"local-{width}.pkl").open("rb") as f:
            route = pickle.load(f)
        for name, row in splits.items():
            if name == "train":
                continue
            result = row.spans[f"c{width}"].copy()
            for recent, older, count, keys in route:
                parent = result if recent == "current" else row.spans[recent]
                left = np.zeros(len(result), dtype=np.uint32)
                h = width // 2
                left[h:] = row.spans[older][:-h]
                left[row.age < width - 1] = 0
                encoder = CategoryPairs()
                encoder.observed = keys
                code = encoder.transform(parent, left)
                seen = code > 0
                result[seen] = count + code[seen]
            row.spans[f"s{width}"] = result


def run(data, folder, output):
    splits = {
        "train": Columns(folder, "train", local=True),
        **{s: full_articles(data, folder, s) for s in ("validation", "test")},
    }
    from copy import copy

    ctrl = copy(splits["validation"])
    ctrl.pos = ctrl.pos[32 * 511 :]
    ctrl.target = ctrl.y[ctrl.pos]
    splits["validation"].pos = splits["validation"].pos[: 32 * 511]
    splits["validation"].target = splits["validation"].y[splits["validation"].pos]
    compose_exact(splits)
    restore_local(splits, folder)
    splits["control"] = ctrl
    graph, base, _ = root_probabilities(folder, splits)
    # The second half of validation shares the article representation with tuning.
    stats = {}
    train = splits["train"]

    def propose(f):
        if f in stats:
            return
        t = Table.count_regions([(train.feature(f), train.y)], 8000)
        stats[f] = {s: t.lookup(splits[s].feature(f)[splits[s].pos], splits[s].target) for s in base}

    for _, features in stages(True):
        for f in features:
            propose(f)
    with (folder / "pair-4.pkl").open("rb") as f:
        _, vectors = pickle.load(f)
    results = {}
    best = np.inf
    misses = 0
    for w in (6, 8, 16, 32, 64, 128, 256, 512):
        tick = time.monotonic()
        info = {}
        if w > 6:
            distinct = {k: v for k, v in splits.items() if k != "control"}
            cache = folder / f"long-{w}.pkl"
            if cache.exists():
                with cache.open("rb") as f:
                    g, new_vectors = pickle.load(f)
                for row in distinct.values():
                    child = row.spans[f"c{w//2}"]
                    ok = np.flatnonzero(row.age >= w - 1)
                    packed = CategoryPairs.keys(child[ok - w // 2], child[ok])
                    typ, inv = np.unique(packed, return_inverse=True)
                    raw = np.column_stack(
                        (vectors[(typ >> 32).astype(int)], vectors[(typ & 0xFFFFFFFF).astype(int)])
                    )
                    frame = pd.DataFrame(raw, columns=g.encoder.numeric)
                    codes = np.concatenate(
                        [
                            g.assignments(frame.iloc[i : i + 1024]).argmax(1) + 1
                            for i in range(0, len(frame), 1024)
                        ]
                    )
                    col = np.zeros(len(child), np.uint32)
                    col[ok] = codes[inv]
                    row.spans[f"c{w}"] = col
                vectors = new_vectors
            else:
                g, vectors, info = geometry(distinct, vectors, w)
                with cache.open("wb") as f:
                    pickle.dump((g, vectors), f)
            info["concepts"] = g.concepts
            info["local"] = finer_spans(distinct, graph.refine_category("token"), w, g.concepts)
            half = w // 2
            features = (
                [f"c{w}", f"s{w}"]
                + [f"{l}{half}|{r}{half}" for l in "cse" for r in "cse" if (l, r) != ("e", "e")]
                + [f"e{w}"]
            )
            for f in features:
                propose(f)
        else:
            features = list(stats)
        # Freeze all shorter readings, so that a gain is attributable only to new evidence.
        weights = calibrated_weights(
            [stats[f]["validation"] for f in features], [1.0] * len(features), base["validation"]
        )
        pred = {s: p.copy() for s, p in base.items()}
        chosen = {}
        for f, weight in zip(features, weights):
            if np.isfinite(weight):
                chosen[f] = float(weight)
                for s in pred:
                    pred[s] = read(stats[f][s], pred[s], weight)
        losses = {s: float(-np.log(p).mean()) for s, p in pred.items()}
        improved = losses["control"] < best - 1e-12
        if improved:
            best = losses["control"]
            base = pred
        misses = 0 if improved else misses + 1
        info["test_block_nll"] = (-np.log(pred["test"])).reshape(-1, 511).mean(1).tolist()
        info["accepted"] = bool(improved)
        results[w] = dict(
            **losses,
            selected=chosen,
            eligible_test=int((splits["test"].age[splits["test"].pos] >= w).sum()),
            **info,
            seconds=time.monotonic() - tick,
            consecutive_without_gain=misses,
        )
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(w, losses, info, misses, flush=True)
        if misses == 2:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "workdir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.workdir, a.output)
