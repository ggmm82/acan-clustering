"""Read native span concepts and exact categorical children, adding one horizon at a time."""

import argparse
import json
import pickle
import resource
from pathlib import Path
import time

import numpy as np
import pandas as pd

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table
from acan.concept_algebra import categorical_reading
from acan.lm.concept_lm import document_counts
from .pair_layers import age, batches
from .conditional_memory import calibrated_weights


def lag(values, n, ages):
    """For a target row, return the span ending immediately before its n newer tokens."""
    out = np.zeros_like(values)
    out[n + 1 :] = values[: -(n + 1)]
    out[ages <= n] = 0
    return out


class Columns:
    def __init__(self, folder, split, local=False, widths=(1, 2, 4), native_local=False):
        source = "validation" if split == "control" else split
        self.y = np.load(folder / (source + "-y.npy"))
        self.starts = np.load(folder / (source + "-starts.npy"))
        self.age = age(self.y, self.starts)
        self.spans = {"e1": self.y.astype(np.uint32) + 1}
        for w in widths:
            self.spans["c" + str(w)] = np.load(folder / (source + f"-{w}.npy"), mmap_mode="r")
        if local:
            for w in (2, 4):
                self.spans["s" + str(w)] = np.load(folder / (source + f"-local-{w}.npy"), mmap_mode="r")
        if native_local:
            self.spans["g2"] = np.load(folder / (source + "-native-local-2.npy"), mmap_mode="r")
        self.pos = (
            np.arange(len(self.y)) if split == "train" else np.flatnonzero(np.arange(len(self.y)) % 512)
        )
        if split == "validation":
            self.pos = self.pos[: 32 * 511]
        elif split == "control":
            self.pos = self.pos[32 * 511 :]
        self.target = self.y[self.pos]

    def feature(self, feature):
        if "|" not in feature:
            return lag(self.spans[feature], 0, self.age)
        left, right = feature.split("|")
        width = int(right[1:])
        return CategoryPairs.keys(
            lag(self.spans[left], width, self.age), lag(self.spans[right], 0, self.age)
        )


def compose_exact(splits):
    for width, left, right in ((2, 1, 1), (3, 1, 2), (4, 2, 2), (5, 1, 4), (6, 2, 4)):
        older, recent = "e" + str(left), "e" + str(right)

        def halves(s):
            x = s.spans[older]
            out = np.zeros_like(x)
            out[right:] = x[:-right]
            out[s.age < right] = 0
            return out, s.spans[recent]

        encoder = CategoryPairs().fit(*halves(splits["train"]))
        for s in splits.values():
            s.spans["e" + str(width)] = encoder.transform(*halves(s))


def stages(local=False, native_local=False):
    # The odd suffixes use a leaf and the dyadic suffix. Six uses two followed by four.
    for w, left, right in ((1, 0, 0), (2, 1, 1), (3, 1, 2), (4, 2, 2), (5, 1, 4), (6, 2, 4)):
        if w == 1:
            yield w, ["c1", "e1"]
            continue
        a, b = str(left), str(right)
        features = ["c" + str(w)] if w in (2, 4) else []
        features += ["c" + a + "|c" + b, "e" + a + "|c" + b, "c" + a + "|e" + b]
        if local:
            if w in (2, 4):
                features += ["s" + str(w)]
            for l in ("s" + a, "c" + a, "e" + a):
                for r in ("s" + b, "c" + b, "e" + b):
                    if "s" not in l + r or ("s1" in (l, r)):
                        continue
                    features.append(l + "|" + r)
        if native_local:
            if w == 2:
                features.append("g2")
            if left == 2 or right == 2:
                left_views = ("g2", "c2", "e2") if left == 2 else ("c" + a, "e" + a)
                right_views = ("g2", "c2", "e2") if right == 2 else ("c" + b, "e" + b)
                features.extend(l + "|" + r for l in left_views for r in right_views if "g2" in (l, r))
        yield w, features + ["e" + str(w)]


def root_probabilities(folder, splits):
    cache = folder / "trained-root.pkl"
    with (folder / "root.pkl").open("rb") as stream:
        graph, vectors = pickle.load(stream)
    cols = graph.encoder.numeric
    vectors = np.vstack([np.full(vectors.shape[1], np.nan), vectors])
    indices = {k: s.feature("e1") for k, s in splits.items()}

    def frames(k, selected):
        for ix in batches(selected, 2048):
            yield pd.DataFrame(vectors[indices[k][ix]], columns=cols), splits[k].y[ix]

    if cache.exists():
        with cache.open("rb") as stream:
            graph = pickle.load(stream)
    else:
        graph.fit_observed_target(
            "token",
            range(8000),
            frames("train", splits["train"].pos),
            frames("validation", splits["validation"].pos),
        )
        with cache.open("wb") as stream:
            pickle.dump(graph, stream)
    head = graph.observed_targets["token"]["probability"]
    weights = graph.assignments(pd.DataFrame(vectors, columns=cols))

    def probs(k):
        selected = splits[k].pos
        return np.concatenate(
            [
                np.einsum("ij,ji->i", weights[indices[k][ix]], head[:, splits[k].y[ix]])
                for ix in batches(selected, 512)
            ]
        )

    return graph, {k: probs(k) for k in splits if k != "train"}, vectors


def read(stats, prior, k):
    m, n, d = stats
    return categorical_reading(n, m, d, prior, k, np)


def local_stats(s, keys):
    # Compress (article/block, observed key) without arithmetic hash collisions.
    doc = s.age == 0
    groups = np.unique(np.rec.fromarrays([np.cumsum(doc), keys]), return_inverse=True)[1]
    return document_counts(groups.astype(np.uint64), keys != 0, s.y)[:, s.pos]


def measure(folder, output, local=False, native_local=False):
    start = time.monotonic()
    splits = {
        k: Columns(folder, k, local, native_local=native_local)
        for k in ("train", "validation", "control", "test")
    }
    compose_exact(splits)
    graph, base, vectors = root_probabilities(folder, splits)
    reading = graph.refine_category("token")
    validation, test = splits["validation"], splits["test"]
    static, document = {}, {}
    tables = {}
    document_names = ["root"]
    results = {}

    def fitted(names):
        weights = calibrated_weights(
            [v["validation"] for _, v in names], [1.0] * len(names), base["validation"]
        )
        ps = {k: v.copy() for k, v in base.items()}
        kept = []
        for (f, stats), kappa in zip(names, weights):
            if np.isfinite(kappa):
                for split in ps:
                    ps[split] = read(stats[split], ps[split], kappa)
                kept.append((f, kappa))
        return ps, kept

    for width, features in stages(local, native_local):
        for f in features:
            train = splits["train"]
            table = Table.count_regions([(train.feature(f), train.y)], 8000)
            tables[f] = table
            static[f] = {
                s: table.lookup(splits[s].feature(f)[splits[s].pos], splits[s].target) for s in base
            }
            if "|" not in f and width <= 4:
                document_names.append(f)
        if width not in (1, 2, 4, 6):
            continue
        for f in document_names:
            if f not in document:
                document[f] = {}
                for name in base:
                    s = splits[name]
                    keys = np.ones(len(s.y), dtype=np.uint32) if f == "root" else s.feature(f)
                    document[f][name] = local_stats(s, keys)

        comparisons = {
            "full": list(static.items()),
            "exact_only": [(f, v) for f, v in static.items() if f.startswith("e") and "|" not in f],
            "no_exact_pairs": [
                (f, v)
                for f, v in static.items()
                if f == "e1" or not (f.startswith("e") and "|" not in f)
            ],
            "with_document": [*static.items(), *((f"doc:{f}", v) for f, v in document.items())],
        }
        result, kept = {}, {}
        for name, items in comparisons.items():
            ps, kept[name] = fitted(items)
            loss = -np.log(ps["test"])
            result[name] = float(loss.mean())
            result[name + "_control"] = float(-np.log(ps["control"]).mean())
            if name == "full":
                result["validation"] = float(-np.log(ps["validation"]).mean())
                result["block_nll"] = loss.reshape(-1, 511).mean(1).tolist()
                reading.levels = [(f, tables[f], k) for f, k in kept[name]]
                ix = test.pos[::1021]
                frame = pd.DataFrame(vectors[test.feature("e1")[ix]], columns=graph.encoder.numeric)
                for f, _, _ in reading.levels:
                    frame[f] = test.feature(f)[ix]
                q, _ = graph.predict_proba(frame, "token")
                np.testing.assert_allclose(
                    q[np.arange(len(ix)), test.y[ix]], ps["test"][::1021], rtol=1e-10
                )
            elif name == "with_document":
                result["document_block_nll"] = loss.reshape(-1, 511).mean(1).tolist()
        result["selected"] = kept["full"]
        result["document_selected"] = kept["with_document"]
        results[str(width)] = result
        print(width, {k: v for k, v in result.items() if isinstance(v, float)}, flush=True)
        stop = width == 4 and min(results["2"]["full"], result["full"]) >= results["1"]["full"]
        receipt = dict(
            test_rows=len(test.target),
            tuning_rows=len(validation.pos),
            geometry=json.loads((folder / "geometry.json").read_text()),
            scores=results,
            decision="stop_no_gain" if stop else "continue",
            seconds=time.monotonic() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        if stop:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--local", action="store_true")
    p.add_argument("--native-local", action="store_true")
    args = p.parse_args()
    measure(args.workdir, args.output, args.local, args.native_local)
