"""Tabelle causali di concetti su finestre di lunghezza 1, 2, 4, ... ."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.extmath import randomized_svd
from acan.core.concepts import ConceptModel

TARGET = "token"
OPTIONS = dict(max_concepts=64, judged_rows=64, bins=16, covariance=False, seed=42)


def names(prefix, n):
    return [f"{prefix}{i:05}" for i in range(n)]


def frame(x, codes, columns):
    out = pd.DataFrame(x, columns=names("x", x.shape[1]))
    for j, name in enumerate(columns):
        ids = codes[:, j]
        out[name] = pd.Categorical.from_codes(ids, categories=names("c", int(ids.max(initial=-1)) + 1))
    return out


class Reader:
    """Precompute the fitted diagonal likelihood; categorical evidence is indexed, never one-hot."""

    def __init__(self, graph):
        self.graph = graph
        self.enc = graph.encoder
        self.columns = [c for c in self.enc.categorical if c != TARGET]
        self.indices = {}
        for name in self.columns:
            cats = self.enc.categories[name]
            ids = np.array([int(c[1:]) for c in cats], dtype=int)
            mapping = np.full(ids.max(initial=-1) + 1, -1, int)
            mapping[ids] = np.arange(len(ids)) + self.enc.offsets[name]
            self.indices[name] = mapping
        self.work = self.parameters(graph.statistics)
        self.above = self.parameters(graph.above) if graph.above is not None else None

    def parameters(self, s):
        g, v, logp = self.graph._parameters(s)
        return (
            g,
            1 / v,
            g / v,
            g * g / v + np.log(2 * np.pi * v),
            logp,
            np.log(np.maximum(s["C"], 1e-300) / s["C"].sum()),
        )

    def encoded(self, x, codes):
        if x.shape[1] < len(self.enc.numeric):
            x = np.pad(
                x.astype(float), ((0, 0), (0, len(self.enc.numeric) - x.shape[1])), constant_values=np.nan
            )
        x = (x - self.enc.mean) / self.enc.std
        mask = np.isfinite(x)
        ids = []
        for j, name in enumerate(self.columns):
            source = codes[:, j]
            mapping = self.indices[name]
            valid = (source >= 0) & (source < len(mapping))
            places = np.full(len(source), -1, int)
            places[valid] = mapping[source[valid]]
            ids.append(places)
        return np.where(mask, x, 0.0), mask, ids

    @staticmethod
    def route(params, x, mask, ids, target=None, weight=0):
        g, a, ga, b, logp, prior = params
        logits = prior - 0.5 * ((x * x) @ a.T - 2 * x @ ga.T + mask @ b.T)
        for col in ids:
            ok = col >= 0
            logits[ok] += logp[:, col[ok]].T
        if target is not None:
            logits += weight * logp[:, target].T
        logits -= logits.max(1, keepdims=True)
        np.exp(logits, out=logits)
        return logits / logits.sum(1, keepdims=True)

    def seen(self, x, codes):
        x, mask, ids = self.encoded(x, codes)
        above = None
        if self.above is not None:
            above = self.route(self.above, x, mask, ids)
            x = x - (above @ self.above[0]) * mask
        return x, mask, ids, above

    def assignments(self, x, codes):
        x, mask, ids, _ = self.seen(x, codes)
        return self.route(self.work, x, mask, ids)

    def layer_sizes(self):
        return [level["concepts"] for level in self.graph.levels]


def frequencies(ids, vocabulary):
    counts = np.zeros(vocabulary, np.int64)
    for begin in range(0, len(ids), 65536):
        counts += np.bincount(ids[begin : begin + 65536], minlength=vocabulary)
    return counts


def embeddings(ids, starts, vocabulary, method, top=24, dimensions=12):
    frequency = frequencies(ids, vocabulary)
    common = np.argsort(-frequency, kind="stable")[:top]
    bucket = np.full(vocabulary, len(common))
    bucket[common] = np.arange(len(common))
    width = len(common) + 1
    before = np.zeros((vocabulary, width), np.int64)
    after = np.zeros_like(before)
    for begin in range(1, len(ids), 65536):
        at = np.arange(begin, min(begin + 65536, len(ids)))
        boundaries = starts[np.searchsorted(starts, begin) : np.searchsorted(starts, at[-1] + 1)]
        at = at[~np.isin(at, boundaries)]
        left = ids[at - 1]
        right = ids[at]
        np.add.at(before, (right, bucket[left]), 1)
        np.add.at(after, (left, bucket[right]), 1)
    values = [np.sqrt(a / np.maximum(a.sum(1, keepdims=True), 1)) for a in (before, after)]
    profile = np.column_stack(values)
    seen = frequency > 0
    if method == "svd":
        center = profile[seen].mean(0)
        _, _, axes = randomized_svd(
            profile[seen] - center, n_components=min(dimensions, *profile[seen].shape), random_state=42
        )
        return (profile - center) @ axes.T, []
    if method != "cp":
        raise ValueError(method)
    # Same train-only profiles. The embedding coordinates are probabilities of the learned concepts.
    graph = ConceptModel(**OPTIONS).fit(pd.DataFrame(profile[seen], columns=names("x", profile.shape[1])))
    reader = Reader(graph)
    return reader.assignments(profile, np.empty((len(profile), 0), int)), reader.layer_sizes()


class Pair:
    def __init__(self, left, right, vectors, classes, following=None):
        self.vectors = vectors
        x, c = self.inputs(left, right)
        self.cols = [] if vectors is not None else ["older", "recent"]
        training = frame(x, c, self.cols)
        targets = []
        if following is not None:
            targets = names("next", following.shape[1])
            training = pd.concat([training, pd.DataFrame(following, columns=targets)], axis=1)
        self.graph = ConceptModel(**OPTIONS).fit(training, labels=targets)
        self.reader = Reader(self.graph)
        self.lookup = None
        if vectors is None:
            a, b = np.indices((classes, classes))
            self.lookup = self.recognize(a.ravel(), b.ravel()).reshape(classes, classes)

    def inputs(self, left, right):
        if self.vectors is None:
            return np.empty((len(left), 0)), np.column_stack([left, right])
        return np.column_stack([self.vectors[left], self.vectors[right]]), np.empty((len(left), 0), int)

    def recognize(self, left, right):
        if self.lookup is not None:
            return self.lookup[left, right]
        return self.reader.assignments(*self.inputs(left, right)).argmax(1).astype(np.int32)


def endpoint_level(child, age, half, pair, out=None):
    values = np.empty(len(child), np.int32) if out is None else out
    for begin in range(0, len(child), 2048):
        t = np.arange(begin, min(begin + 2048, len(child)))
        values[t] = -1
        t = t[age[t] >= 2 * half]
        if len(t):
            values[t] = pair.recognize(child[t - half], child[t])
    return values


def ages(n, starts):
    first = np.zeros(n, int)
    first[starts] = starts
    return np.arange(n) - np.maximum.accumulate(first) + 1


class Dyadic:
    def __init__(self, vectors, scales=20):
        self.vectors = vectors
        self.scales = scales
        self.pairs = []

    def fit(self, ids, starts, pair_rows=2048, target_rows=12000):
        with tempfile.TemporaryDirectory() as directory:
            return self.fit_columns(ids, np.asarray(starts), pair_rows, target_rows, Path(directory))

    def fit_columns(self, ids, starts, pair_rows, target_rows, directory):
        n = len(ids)
        if n < 2 or not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0) or starts[-1] >= n:
            raise ValueError("Servono token e inizi di documento ordinati a partire da zero.")
        self.pairs = []
        ends = np.r_[starts[1:], n]

        def disk(name):
            return np.lib.format.open_memmap(
                directory / (name + ".npy"), mode="w+", dtype=np.int32, shape=(n,)
            )

        age = disk("age")
        for begin in range(0, n, 65536):
            at = np.arange(begin, min(begin + 65536, n))
            age[at] = at - starts[np.searchsorted(starts, at, side="right") - 1] + 1
        levels = [np.asarray(ids)]
        rng = np.random.default_rng(42)

        def sample(width, count):
            cumulative = np.cumsum(np.maximum(ends - starts - width, 0))
            ranks = rng.choice(cumulative[-1], min(count, cumulative[-1]), replace=False)
            docs = np.searchsorted(cumulative, ranks, side="right")
            offsets = ranks - np.r_[0, cumulative[:-1]][docs]
            return starts[docs] + width - 1 + offsets, int(cumulative[-1])

        self.support = []
        for j in range(1, self.scales):
            width = 2**j
            end, count = sample(width, pair_rows)
            self.support.append(count)
            if not count:
                break
            child = levels[-1]
            vectors = self.vectors if j == 1 else None
            pair = Pair(
                child[end - width // 2],
                child[end],
                vectors,
                None if j == 1 else self.pairs[-1].graph.concepts,
                self.vectors[ids[end + 1]],
            )
            self.pairs.append(pair)
            levels.append(endpoint_level(child, age, width // 2, pair, disk(str(width))))
        self.columns = [f"s{j}" for j in range(1, len(levels))]
        end, _ = sample(1, target_rows)
        if not len(end):
            raise ValueError("Nessun prefisso con un token successivo nel train.")
        t = np.sort(end + 1)
        x, c = self.context(levels, t, age)
        training = frame(x, c, self.columns)
        vocabulary = len(self.vectors)
        training[TARGET] = pd.Categorical.from_codes(ids[t], categories=names("t", vocabulary))
        self.graph = ConceptModel(**dict(OPTIONS, bins=32, judged_rows=256)).fit(training, labels=[TARGET])
        self.finish(ids, age, levels)
        return self

    def context(self, levels, positions, age):
        previous = np.maximum(positions - 1, 0)
        x = self.vectors[levels[0][previous]].copy()
        x[age[positions] <= 1] = np.nan
        c = (
            np.column_stack([col[previous] for col in levels[1:]])
            if len(levels) > 1
            else np.empty((len(x), 0), int)
        )
        c[age[positions] <= 1] = -1
        return x, c

    def finish(self, ids, age, levels):
        """Accumulate the ordinary supervised working-level sums over every row, including rare targets."""
        graph = self.graph
        enc = graph.encoder
        start = enc.offsets[TARGET]
        self.complete_dictionary(graph, ids, len(self.vectors))
        reader = Reader(graph)
        total = {k: np.zeros_like(v) for k, v in graph.statistics.items()}
        parents = graph.above
        if parents is not None:
            # Recover each parent's contributions before their mixture, as in ConceptModel._above.
            from acan.concept_algebra import unbiased

            cm = np.maximum(parents["Cm"], 1e-12)
            var = unbiased(np.maximum(parents["Q"] / cm - (parents["U"] / cm) ** 2, 0), parents["Cm"], np)
            parent_var = np.where(parents["Cm"] > 1, np.maximum(var, 1e-6), graph._prior(parents)[0])
            parent_freq = np.empty_like(parents["N"])
            parent_imp = parents["PG"] / np.maximum(parents["C"][:, None], 1e-12)
            for j, block in enumerate(graph._blocks()):
                n = parents["N"][:, block]
                m = n.sum(1, keepdims=True)
                parent_freq[:, block] = (n + graph.kappa * graph._prior(parents)[1][:, block]) / (
                    m + graph.kappa
                )
                g = unbiased(1 - ((n / np.maximum(m, 1e-12)) ** 2).sum(1), m[:, 0], np)
                parent_imp[:, j] = np.where(m[:, 0] > 1, g, parent_imp[:, j])
            links = np.zeros((graph.concepts, len(parents["C"])))
        weight = max(len(enc.numeric) + len(self.columns), 1)
        for begin in range(0, len(ids), 2048):
            t = np.arange(begin, min(begin + 2048, len(ids)))
            x, c = self.context(levels, t, age)
            x, mask, cats, above = reader.seen(x, c)
            y = start + ids[t]
            w = reader.route(reader.work, x, mask, cats, y, weight)
            total["C"] += w.sum(0)
            total["Cm"] += w.T @ mask
            total["U"] += w.T @ x
            total["Q"] += w.T @ (x * x)
            for col in [*cats, y]:
                ok = col >= 0
                categories, inverse = np.unique(col[ok], return_inverse=True)
                one = sparse.csr_matrix(
                    (np.ones(ok.sum()), (inverse, np.flatnonzero(ok))), shape=(len(categories), len(t))
                )
                total["N"][:, categories] += (one @ w).T
            if above is not None:
                links += w.T @ above
                total["PV"] += w.T @ (mask * (above @ parent_var))
        if parents is None:
            total.update(
                PV=total["Cm"].copy(),
                PP=np.outer(total["C"], graph.root_p),
                PG=np.outer(total["C"], graph.root_g),
            )
        else:
            total.update(PP=links @ parent_freq, PG=links @ parent_imp)
        graph.statistics = total
        graph.levels[-1]["statistics"] = total
        self.reader = Reader(graph)
        self.head = np.exp(self.reader.work[4][:, start:])
        self.head /= self.head.sum(1, keepdims=True)

    @staticmethod
    def complete_dictionary(graph, ids, vocab):
        enc = graph.encoder
        old = enc.categories[TARGET]
        start = enc.offsets[TARGET]
        cols = np.r_[np.arange(start), start + np.array([int(s[1:]) for s in old])]
        enc.categories[TARGET] = names("t", vocab)
        enc.width = start + vocab

        def extend(value):
            out = np.zeros((*value.shape[:-1], enc.width))
            out[..., cols] = value
            return out

        graph.root_p = extend(graph.root_p)
        # The complete dictionary has the same empirical training prior as a tabular categorical column.
        root = frequencies(ids, vocab) / len(ids)
        graph.root_p[start:] = np.maximum(root, 1e-12)
        graph.root_g[-1] = 1 - root @ root
        expanded = set()
        for stats in [graph.statistics, graph.above] + [
            s[key] for s in graph.levels for key in ("statistics", "table")
        ]:
            if stats is None or id(stats) in expanded:
                continue
            expanded.add(id(stats))
            stats["N"] = extend(stats["N"])
            stats["PP"] = extend(stats["PP"])
            stats["PP"][:, start:] = stats["C"][:, None] * graph.root_p[start:]
            stats["PG"][:, -1] = stats["C"] * graph.root_g[-1]

    def levels(self, ids, starts=(0,)):
        age = ages(len(ids), np.asarray(starts))
        result = [np.asarray(ids)]
        for j, pair in enumerate(self.pairs, 1):
            result.append(endpoint_level(result[-1], age, 1 << (j - 1), pair))
        return result, age

    def cursor(self):
        return Cursor(self)

    def generate(self, tokens, count):
        """Completamento deterministico della colonna target, una posizione alla volta."""
        cursor = self.cursor().prefill(tokens)
        result = []
        for _ in range(count):
            token = int(cursor.distribution().argmax())
            result.append(token)
            cursor.observe(token)
        return result


class Cursor:
    def __init__(self, model):
        self.model = model
        self.n = 0
        self.buffers = [np.full(1 << j, -1, np.int32) for j in range(len(model.pairs))]
        self.last = [-1] * (1 + len(model.pairs))

    def prefill(self, ids):
        for begin in range(0, len(ids), 2048):
            child = np.asarray(ids[begin : begin + 2048], dtype=np.int32)
            size = len(child)
            times = self.n + np.arange(size)
            self.last[0] = int(child[-1])
            for j, (ring, pair) in enumerate(zip(self.buffers, self.model.pairs)):
                half = len(ring)
                older = np.empty_like(child)
                overlap = min(half, size)
                older[:overlap] = ring[times[:overlap] % half]
                older[overlap:] = child[: size - overlap]
                ring[times[-overlap:] % half] = child[-overlap:]
                valid = times >= 2 * half - 1
                result = np.full(size, -1, np.int32)
                if valid.any():
                    result[valid] = pair.recognize(older[valid], child[valid])
                child = result
                self.last[j + 1] = int(child[-1])
            self.n += size
        return self

    def observe(self, token):
        value = int(token)
        self.last[0] = value
        for j, (ring, pair) in enumerate(zip(self.buffers, self.model.pairs)):
            half = len(ring)
            slot = self.n % half
            older = ring[slot]
            ring[slot] = value
            if self.n < 2 * half - 1:
                value = -1
            elif pair.lookup is None:
                value = int(pair.recognize(np.array([older]), np.array([value]))[0])
            else:
                value = int(pair.lookup[older, value])
            self.last[j + 1] = value
        self.n += 1

    def distribution(self):
        v = self.model.vectors
        x = v[self.last[0] : self.last[0] + 1] if self.n else np.full((1, v.shape[1]), np.nan)
        w = self.model.reader.assignments(x, np.array([self.last[1:]], dtype=int))
        return (w @ self.model.head)[0]
