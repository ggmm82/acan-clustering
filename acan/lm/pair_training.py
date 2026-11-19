"""Fit native numeric pair concepts and judge categorical descendants on separate documents."""

import numpy as np
import pandas as pd

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table
from acan.core.concepts import ConceptModel
from acan.core.conditional import calibrated_weights
from acan.concept_algebra import categorical_reading
from .pair_model import EncodedPrefix, PairEncoder, PairLanguageModel, shift


def semantic_vectors(ids, vocabulary, dimensions=32, top=512, seed=0, starts=None):
    """Token coordinates: the neighbours on the left and right of each token, compressed by SVD."""
    from sklearn.utils.extmath import randomized_svd
    ids=np.asarray(ids,dtype=np.int64)
    frequency=np.bincount(ids,minlength=vocabulary)
    top=min(top,vocabulary)
    common=np.argsort(-frequency,kind='stable')[:top]
    where=np.full(vocabulary,top,dtype=int);where[common]=np.arange(top)
    left,right=ids[:-1],ids[1:]
    if starts is not None:
        valid=np.ones(len(left),bool)
        valid[np.asarray(starts)[1:]-1]=False
        left,right=left[valid],right[valid]
    before=np.bincount(right*(top+1)+where[left],minlength=vocabulary*(top+1)).reshape(vocabulary,top+1)
    after=np.bincount(left*(top+1)+where[right],minlength=vocabulary*(top+1)).reshape(vocabulary,top+1)
    profile=np.hstack([np.sqrt(a/np.maximum(a.sum(1,keepdims=True),1)) for a in (before,after)])
    center=profile[frequency>0].mean(0)
    _,singular,directions=randomized_svd((profile if starts is None else profile[frequency>0])-center,n_components=min(dimensions,profile.shape[1]),n_iter=5,random_state=seed)
    return ((profile-center)@directions.T).astype(np.float32),singular


def geometry(splits, vectors, width):
    half = width // 2

    def keys(row):
        child = row.spans["e1" if width == 2 else f"c{half}"].astype(np.uint32, copy=False)
        older = np.zeros_like(child)
        older[half:] = child[:-half]
        older[row.age < width - 1] = 0
        return CategoryPairs.keys(older, child)

    def frame(key):
        x = np.column_stack((vectors[(key >> 32).astype(int)], vectors[(key & 0xFFFFFFFF).astype(int)]))
        return pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])

    packed = keys(splits["train"])
    types, counts = np.unique(packed[packed > 0], return_counts=True)
    x = frame(types)
    graph = ConceptModel(
        max_concepts=256, bins=24, judged_rows=8192, refine_rows=32768, covariance=False, seed=42
    )
    graph.fit_counted_numeric(x, counts)
    for row in splits.values():
        packed = keys(row)
        typ, inv = np.unique(packed, return_inverse=True)
        valid = np.flatnonzero(typ > 0)
        codes = np.zeros(len(typ), np.uint32)
        for idx in np.array_split(valid, max(1, (len(valid) + 1023) // 1024)):
            if len(idx):
                codes[idx] = graph.assignments(frame(typ[idx])).argmax(1) + 1
        row.spans[f"c{width}"] = codes[inv]
        if row is splits["train"]:
            train_codes = codes[valid]
    mass = np.bincount(train_codes, weights=counts, minlength=graph.concepts + 1)
    means = np.zeros((len(mass), x.shape[1]))
    for j, col in enumerate(x.to_numpy().T):
        means[:, j] = np.bincount(train_codes, weights=counts * col, minlength=len(mass)) / np.maximum(
            mass, 1
        )
    if means.shape[1] > 32:
        centre = np.average(means, axis=0, weights=mass)
        _, basis = np.linalg.eigh(((means - centre) * mass[:, None]).T @ (means - centre))
        means = (means - centre) @ basis[:, -32:]
    return graph, means, dict(types=len(types), occurrences=int(counts.sum()), concepts=graph.concepts)


def finer_spans(splits, reading, width, size, routes=None):
    half = width // 2

    def halves(row, older, recent):
        left = np.zeros(len(row.y), np.uint32)
        left[half:] = row.spans[older][:-half]
        left[row.age < width - 1] = 0
        return left, row.spans[recent]

    tr = splits["train"]
    exact = f"e{half}"
    dictionary = CategoryPairs().fit(*halves(tr, exact, exact))
    for row in splits.values():
        row.spans[f"e{width}"] = dictionary.transform(*halves(row, exact, exact))
        row.spans[f"s{width}"] = row.spans[f"c{width}"].copy()
    judged = (np.cumsum(tr.age == 0) - 1) % 5 == 4
    reports = []
    pairs = [(f"c{half}", exact), (exact, f"s{width}")]
    if width >= 4:
        pairs.insert(0, (f"s{half}", f"s{half}"))
    for older, recent in pairs:
        l, r = halves(tr, older, recent)
        full = shift(r, 1, tr.age), shift(l, 1, tr.age), tr.y
        if not judged.any() or judged.all():
            break
        t, report = reading.subconcepts(
            tuple(a[~judged] for a in full), tuple(a[judged] for a in full), full
        )
        for row in splits.values():
            l, r = halves(row, older, recent)
            ix = t.find(CategoryPairs.keys(r, l))
            ok = ix >= 0
            row.spans[f"s{width}"][ok] = size + 1 + ix[ok]
        if routes is not None:
            enc = CategoryPairs()
            enc.observed = t.keys
            routes.append((older, "current" if recent == f"s{width}" else recent, size, enc))
        size += len(t.keys)
        reports.append(report)
    return reports


def stages(local=False):
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
        yield w, features + ["e" + str(w)]


def add_chunk(model, ids, starts, tail, tables, budget):
    row = model.encoder.encode(np.r_[tail, ids], np.unique(np.r_[0, len(tail) + starts]).astype(int))
    begin = len(tail)
    strengths = getattr(model, "training_weights", model.weights)
    total_weight = sum(k**-0.5 for k in strengths.values())
    for feature in strengths:
        key = row.feature(feature)[begin:]
        part = Table.count_regions([(key, ids.astype(np.int64, copy=False))], model.vocabulary)
        total = part if feature not in tables else tables[feature] + part
        # Each table inherits through its preceding readout when a rare key is removed.
        limit = int(budget * strengths[feature] ** -0.5 / total_weight)
        if total.nbytes() > limit:
            sizes = 24 + 16 * total.D
            at = np.argsort(-total.C / (sizes + 0.0), kind="stable")
            keep = np.zeros(len(at), bool)
            keep[at[np.cumsum(sizes[at]) <= limit]] = True
            total = total.keep(keep)
        tables[feature] = total
    age = int(row.age[-1]) + 1
    return row.y[-min(age, model.encoder.width) :].copy()


def sample_documents(tokens, eos, limit=262144, block=4096, documents=None):
    """Sample contiguous training windows across the file, preserving all internal boundaries."""
    size = min(len(tokens), limit)
    chunks = min((size + block - 1) // block, max(1, len(tokens) // block))
    widths = np.diff(np.linspace(0, size, chunks + 1, dtype=int))
    starts = (
        np.r_[0, np.cumsum(widths)[:-1]]
        if size == len(tokens)
        else np.linspace(0, len(tokens) - widths[-1], chunks, dtype=int)
    )
    pieces = [np.asarray(tokens[i : i + w], np.int32) for i, w in zip(starts, widths)]
    ids = np.concatenate(pieces)
    boundaries = np.r_[0, np.cumsum(widths)[:-1], np.flatnonzero(ids[:-1] == eos) + 1]
    if documents is not None:
        for start, width, offset in zip(starts, widths, np.r_[0, np.cumsum(widths)[:-1]]):
            a, b = np.searchsorted(documents, [start, start + width])
            boundaries = np.r_[boundaries, documents[a:b] - start + offset]
    return ids, np.unique(boundaries)


def validation_rows(pieces, eos):
    ids = np.concatenate(pieces).astype(np.int32)
    return EncodedPrefix(
        ids,
        np.unique(
            np.r_[0, np.cumsum([len(p) for p in pieces])[:-1], np.flatnonzero(ids[:-1] == eos) + 1]
        ).astype(int),
    )


def learn(tokens, vocabulary, eos, tuning, check, documents=None):
    """Discover the geometry on train and select predictors on held-out next-token likelihood."""
    ids, starts = sample_documents(tokens, eos, documents=documents)
    tr = EncodedPrefix(ids, starts)
    va = validation_rows(tuning, eos)
    rows = {"train": tr, "tune": va}
    vectors, _ = semantic_vectors(ids, vocabulary, dimensions=8, top=96, seed=42, starts=starts)
    names = [f"x{i}" for i in range(vectors.shape[1])]
    root = ConceptModel(
        max_concepts=256, bins=24, judged_rows=8192, refine_rows=32768, covariance=False, seed=42
    )
    root.fit(pd.DataFrame(vectors[np.unique(ids)], columns=names))
    classes = root.assignments(pd.DataFrame(vectors, columns=names)).argmax(1) + 1
    padded = np.vstack([np.full(vectors.shape[1], np.nan), vectors])

    def batches(row):
        previous = row.feature("e1")
        for i in range(0, len(previous), 2048):
            check()
            yield pd.DataFrame(padded[previous[i : i + 2048]], columns=names), row.y[i : i + 2048]

    root.fit_observed_target("token", range(vocabulary), batches(tr), batches(va))
    enc = PairEncoder(vectors, classes, {}, {}, {})
    enc.open_exact = True
    model = PairLanguageModel(enc, root, {}, {}, {})
    for row in rows.values():
        row.spans["c1"] = classes[row.y]
    base = model.root_probability(va, np.arange(len(va.y)))
    reading = root.refine_category("token")
    old_vectors = np.nan_to_num(padded)
    misses = 0
    for width in (1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128, 256, 512):
        check()
        if width > 1 and tr.age.max() < width - 1:
            break
        if width > 1:
            right = 2 ** ((width - 1).bit_length() - 1)
            left = width - right
            # At 3,5,6 the more recent half is the longer one.
            if width & (width - 1):
                left, right = min(left, right), max(left, right)
            l, r = f"e{left}", f"e{right}"
            dictionary = CategoryPairs().fit(shift(tr.spans[l], right, tr.age), tr.spans[r])
            enc.exact[width] = left, right, dictionary
            for row in rows.values():
                row.exact_columns[f"e{width}"] = (l, r, right)
                row.spans[f"e{width}"] = dictionary.transform(
                    shift(row.spans[l], right, row.age), row.spans[r]
                )
            if width & (width - 1) == 0:
                graph, old_vectors, _ = geometry(rows, old_vectors, width)
                enc.nodes[width] = graph, old_vectors
                enc.local[width] = []
                finer_spans(rows, reading, width, graph.concepts, enc.local[width])
        if width <= 6:
            features = dict(stages(True))[width]
        else:
            h = width // 2
            features = (
                [f"c{width}", f"s{width}"]
                + [f"{a}{h}|{b}{h}" for a in "cse" for b in "cse" if (a, b) != ("e", "e")]
                + [f"e{width}"]
            )
        stats = []
        for f in features:
            t = Table.count_regions([(tr.feature(f), tr.y)], vocabulary)
            stats.append(t.lookup(va.feature(f), va.y))
        weights = calibrated_weights(stats, [1.0] * len(stats), base)
        gain = False
        for f, k, (m, n, d) in zip(features, weights, stats):
            if np.isfinite(k):
                base = categorical_reading(n, m, d, base, k, np)
                model.weights[f] = k
                gain = True
        if width >= 8:
            misses = 0 if gain else misses + 1
            if misses == 2:
                break
    enc.width = max([6, *enc.nodes])
    # Every training target is subsequently counted exactly once by the streaming worker.
    model.tables = {f: Table.empty(vocabulary) for f in model.weights}
    return model
