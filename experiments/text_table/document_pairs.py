"""Compile the selected native graph, calibrate causal concept memory, and test lexical recall."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table, SharedProfiles
from acan.lm.pair_model import PairEncoder, PairLanguageModel, shift
from .pair_reading import Columns, local_stats, read
from acan.lm.pair_training import finer_spans
from acan.core.conditional import calibrated_weights
from .common import recall, corpus


def get(file):
    with file.open("rb") as stream:
        return pickle.load(stream)


def compile_graph(folder, scores):
    tr = Columns(folder, "train", local=True)
    root = get(folder / "trained-root.pkl")
    base, leaf = get(folder / "root.pkl")
    token_classes = base.assignments(pd.DataFrame(leaf, columns=base.encoder.numeric)).argmax(1) + 1
    nodes = {}
    exact = {}
    local = {}
    largest = max(int(w) for w, s in scores.items() if s["accepted"])
    widths = [(2, 1, 1), (3, 1, 2), (4, 2, 2), (5, 1, 4), (6, 2, 4)]
    widths += [(2**j, 2 ** (j - 1), 2 ** (j - 1)) for j in range(3, largest.bit_length())]
    for w, l, r in widths:
        left = shift(tr.spans[f"e{l}"], r, tr.age)
        d = CategoryPairs().fit(left, tr.spans[f"e{r}"])
        exact[w] = l, r, d
        tr.spans[f"e{w}"] = d.transform(left, tr.spans[f"e{r}"])
    for w in (2, 4):
        nodes[w] = get(folder / f"pair-{w}.pkl")
        steps = []
        for recent, older, offset, keys in get(folder / f"local-{w}.pkl"):
            d = CategoryPairs()
            d.observed = keys
            steps.append((older, recent, offset, d))
        local[w] = steps
    for j in range(3, largest.bit_length()):
        w = 2**j
        nodes[w] = get(folder / f"long-{w}.pkl")
        g, centres = nodes[w]
        vectors = nodes[w // 2][1]
        child = tr.spans[f"c{w//2}"]
        ok = np.flatnonzero(tr.age >= w - 1)
        packed = CategoryPairs.keys(child[ok - w // 2], child[ok])
        typ, inv = np.unique(packed, return_inverse=True)
        frame = pd.DataFrame(
            np.column_stack((vectors[(typ >> 32).astype(int)], vectors[(typ & 0xFFFFFFFF).astype(int)])),
            columns=g.encoder.numeric,
        )
        codes = np.concatenate(
            [g.assignments(frame.iloc[i : i + 1024]).argmax(1) + 1 for i in range(0, len(frame), 1024)]
        )
        col = np.zeros(len(child), np.uint32)
        col[ok] = codes[inv]
        tr.spans[f"c{w}"] = col
        local[w] = []
        finer_spans({"train": tr}, root.refine_category("token"), w, g.concepts, local[w])
    weights = {}
    for w, s in scores.items():
        if s["accepted"]:
            weights.update(s["selected"])
    enc = PairEncoder(leaf, token_classes, nodes, exact, local)
    tables = {f: SharedProfiles(Table.count_regions([(tr.feature(f), tr.y)], 8000)) for f in weights}
    return PairLanguageModel(enc, root, tables, weights, {})


def evaluate(model, data, retune=False):
    from .common import blocks

    tok = Tokenizer.from_file(str(data / "bpe-8000.json"))
    ps = {}
    stats = {}
    glob = {}
    for split in ("validation", "test"):
        y, starts = corpus(data, split, tok)
        row = model.encoder.encode(y, starts)
        offsets = (
            blocks(y)[1]
            if split == "validation"
            else json.loads(
                (Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text()
            )["fresh_offsets"]
        )
        for name, offs in (
            [("validation", offsets[:32]), ("control", offsets[32:])]
            if split == "validation"
            else [("test", offsets)]
        ):
            row.pos = np.concatenate([np.arange(a + 1, a + 512) for a in offs])
            targets = y[row.pos]
            p = model.root_probability(row, row.pos)
            glob[name] = {
                f: model.tables[f].lookup(row.feature(f)[row.pos], targets) for f in model.weights
            }
            ps[name] = p
            fields = ["root", "e1", "c1", "e2", "c2", "s2", "e4", "c4", "s4"] + [
                f"s{w}" for w in model.encoder.nodes if w >= 8
            ]
            stats[name] = {f: local_stats(row, row.feature(f)) for f in fields}
    if retune:
        keys = list(model.weights)
        kappa = calibrated_weights(
            [glob["validation"][f] for f in keys], list(model.weights.values()), ps["validation"]
        )
        model.weights = {f: k for f, k in zip(keys, kappa) if np.isfinite(k)}
    for name in ps:
        for f, k in model.weights.items():
            ps[name] = read(glob[name][f], ps[name], k)
    result = {}
    for mode, names in [
        ("exact", [f for f in fields if f == "root" or f.startswith("e")]),
        ("concepts", [f for f in fields if not f.startswith("e")]),
        ("combined", fields),
    ]:
        weights = calibrated_weights(
            [stats["validation"][f] for f in names], [1.0] * len(names), ps["validation"]
        )
        chosen = {f: k for f, k in zip(names, weights) if np.isfinite(k)}
        result[mode] = {"weights": chosen}
        for name, p in ps.items():
            for f, k in chosen.items():
                p = read(stats[name][f], p, k)
            result[mode][name] = float(-np.log(p).mean())
    model.memory = result["combined"]["weights"]
    result["static"] = {s: float(-np.log(p).mean()) for s, p in ps.items()}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", type=Path, help="Optional trusted local scratch file for transfer experiments."
    )
    for k in ("data", "workdir", "scores", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    model = compile_graph(a.workdir, json.loads(a.scores.read_text()))
    result = evaluate(model, a.data)
    model.encoder._memo.clear()
    if a.model:
        with a.model.open("wb") as out:
            pickle.dump(model, out, pickle.HIGHEST_PROTOCOL)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)
    tok = Tokenizer.from_file(str(a.data / "bpe-8000.json"))
    train, _ = corpus(a.data, "train", tok)
    test, _ = corpus(a.data, "test", tok)
    for m in ("combined", "exact", "concepts"):
        model.memory = result[m]["weights"]
        result[m]["recall"] = recall(model, train, test, tok)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
