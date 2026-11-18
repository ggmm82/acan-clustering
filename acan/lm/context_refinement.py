"""Categorical backoff through a CP partition of the oldest token in a suffix.

Each intermediate table is a projection of the *same* retained exact counts.
It can be rebuilt after more training without replaying the corpus or losing
the meanings of counts when the learned partition changes.
"""

import numpy as np
from scipy import sparse
from acan.concept_algebra import categorical_reading
from .concept_lm import GRID, Table


def partitions(model):
    """Fit the tabular graph to left/right neighbour profiles from the retained bigrams."""
    import pandas as pd
    from sklearn.utils.extmath import randomized_svd
    from acan.core.concepts import ConceptModel

    vocab = model.V
    keys, target, count = model.levels[1].pairs()
    if not len(keys):
        return []
    source = keys.astype(np.int64) - 1
    frequency = np.bincount(source, weights=count, minlength=vocab)
    seen = np.flatnonzero(frequency >= 3)
    if len(seen) < 4:
        return []
    order = seen[np.argsort(-frequency[seen], kind="stable")]
    sample = order[:8000]
    top = order[:512]
    bucket = np.full(vocab, len(top), dtype=int)
    bucket[top] = np.arange(len(top))
    profiles = []
    for row, neighbour in ((target, source), (source, target)):
        matrix = sparse.csr_matrix(
            (count.astype(float), (row, bucket[neighbour])), shape=(vocab, len(top) + 1)
        )
        sums = np.asarray(matrix.sum(1)).ravel()
        matrix.data = np.sqrt(matrix.data / np.repeat(np.maximum(sums, 1), np.diff(matrix.indptr)))
        profiles.append(matrix)
    profile = sparse.hstack(profiles, format="csr")
    sample_values = profile[sample].toarray()
    center = sample_values.mean(0)
    sample_values -= center
    _, _, directions = randomized_svd(sample_values, min(32, *sample_values.shape), random_state=0)

    def frame(rows):
        return pd.DataFrame(
            profile[rows] @ directions.T - center @ directions.T,
            columns=[f"axis{i}" for i in range(len(directions))],
        )

    graph = ConceptModel(max_concepts=256, judged_rows=512, covariance=False).fit(frame(sample))
    members = np.concatenate(
        [graph.assignments(frame(seen[i : i + 512])).argmax(1) for i in range(0, len(seen), 512)]
    )
    maps = []
    for level in reversed(graph.levels):
        _, compact = np.unique(members, return_inverse=True)
        mapping = np.full(vocab, compact.max() + 1, dtype=np.int64)
        mapping[seen] = compact
        maps.append(mapping)
        if level["parents"] is not None:
            members = level["parents"].argmax(0)[members]
    return maps


def group_keys(keys, mapping, vocabulary):
    """Replace just the oldest token; nearer tokens retain identity and order."""
    base = np.uint64(vocabulary + 1)
    return (keys // base) * np.uint64(mapping.max() + 1) + mapping[(keys % base).astype(int) - 1].astype(
        np.uint64
    )


def pool(table, mapping):
    keys, targets, counts = table.pairs()
    groups, inverse = np.unique(group_keys(keys, mapping, table.V), return_inverse=True)
    cells, assignments = np.unique(inverse * (table.V + 1) + targets, return_inverse=True)
    # Integer reductions also work for counts larger than the exact range of float64.
    totals = np.zeros(len(groups), dtype=np.int64)
    values = np.zeros(len(cells), dtype=np.int64)
    np.add.at(totals, inverse, counts)
    np.add.at(values, assignments, counts)
    return Table(groups, totals, cells, values, table.V)


class ContextRefinement:
    def __init__(self, mapping, weights, trained_at):
        self.mapping = np.asarray(mapping, dtype=np.int64)
        self.classes = int(self.mapping.max()) + 1
        self.weights = np.asarray(weights, dtype=float)
        self.trained_at = int(trained_at)
        self.dirty = False
        self.tables = {}

    def build(self, model):
        self.dirty = False
        self.tables = {}
        remaining = (
            model.budget_bytes - sum(t.nbytes() for t in model.levels) if model.budget_bytes else np.inf
        )
        remaining -= self.mapping.nbytes + self.weights.nbytes
        for depth, weight in enumerate(self.weights, 1):
            if not weight:
                continue
            table = pool(model.levels[depth], self.mapping)
            if table.nbytes() <= remaining:
                self.tables[depth] = table
                remaining -= table.nbytes()
        return self

    def nbytes(self):
        return self.mapping.nbytes + self.weights.nbytes + sum(t.nbytes() for t in self.tables.values())

    def query(self, depth, keys, targets, valid):
        table = self.tables[depth]
        key = group_keys(np.where(valid, keys, np.uint64(1)), self.mapping, table.V)
        return tuple(value * valid for value in table.lookup(key, targets))

    @classmethod
    def select(cls, model, pieces, maps, trained_at):
        """Choose cut and smoothing on true held-out tokens; skipping every cut is a candidate."""
        if not maps:
            return None
        caches = [model._prepared(0, ids, np.arange(1, len(ids))) for ids in pieces]

        def joined(index, level):
            return [np.concatenate([c[index][level][i] for c in caches]) for i in range(3)]

        global_counts = [joined(0, j) for j in range(model.depth + 1)]
        local_counts = [joined(1, j) for j in range(len(caches[0][1]))]
        base = np.concatenate([c[2] for c in caches])

        def loss(extra, weights):
            p = base
            for j, (c, n, d) in enumerate(global_counts):
                if j in extra and weights[j - 1]:
                    cc, nn, dd = extra[j]
                    p = categorical_reading(nn, cc, dd, p, weights[j - 1], np)
                p = categorical_reading(n, c, d, p, model.a[j], np)
            for j, (c, n, d) in enumerate(local_counts):
                p = categorical_reading(n, c, d, p, model.b[j], np)
            return -np.log(p).mean()

        best_loss, winner = loss({}, []), None
        # Three intermediate cuts keep uint64 keys exact for the supported vocabularies.
        depth = min(3, model.depth)
        for mapping in maps:
            candidate = cls(mapping, np.ones(depth), trained_at).build(model)
            columns = {j: [] for j in candidate.tables}
            for ids in pieces:
                for j, (key, valid) in enumerate(model.readings[0].keys(ids, [], model.V)):
                    if j in columns:
                        columns[j].append(candidate.query(j, key[1:], ids[1:], valid[1:]))
            extra = {
                j: [np.concatenate([v[i] for v in parts]) for i in range(3)] for j, parts in columns.items()
            }
            weights = np.zeros(depth)
            for _ in range(2):
                for j in extra:
                    trials = []
                    for weight in (0.0, *GRID):
                        weights[j - 1] = weight
                        trials.append((loss(extra, weights), weight))
                    weights[j - 1] = min(trials)[1]
            score = loss(extra, weights)
            if score < best_loss:
                best_loss = score
                winner = cls(mapping, weights, trained_at)
        return None if winner is None else winner.build(model)


class SuffixCursor:
    """Causal document counts indexed by ordered suffix; only the tail retains positions.

    Each depth caps distinct (suffix, continuation) cells. When full, the oldest
    suffix at that depth is forgotten as a unit, leaving the shorter levels intact.
    """

    def __init__(self, model):
        from collections import OrderedDict, deque

        self.model = model
        self.tail = deque(maxlen=max(model.depth, model.document))
        self.levels = [OrderedDict() for _ in range(model.document + 1)] if model.document else []
        self.cells = np.zeros(len(self.levels), dtype=int)
        self.limit = model.document_limit()
        self.base = model.V + 1

    def keys(self, depth):
        yield 0
        key = 0
        for token in list(reversed(self.tail))[:depth]:
            key = (key * self.base + token + 1) & ((1 << 64) - 1)
            yield key

    def observe(self, token):
        y = int(token)
        for j, (key, level) in enumerate(zip(self.keys(self.model.document), self.levels)):
            entry = level.get(key)
            if entry is None:
                entry = level[key] = [0, {}]
            entry[0] += 1
            counts = entry[1]
            if y not in counts:
                self.cells[j] += 1
            counts[y] = counts.get(y, 0) + 1
            if j and self.cells[j] > self.limit:
                _, evicted = level.popitem(last=False)
                self.cells[j] -= len(evicted[1])
        self.tail.append(y)

    def _predict(self, token=None, a=None, b=None):
        model = self.model
        a = model.a if a is None else a
        b = model.b if b is None else b
        p = np.full(model.V, 1 / model.V) if token is None else 1 / model.V
        for j, key in enumerate(self.keys(model.depth)):
            tables = [(model.levels[j], key, a[j])]
            ref = model.refinement
            if ref is not None and j in ref.tables:
                coarse = (key // self.base) * ref.classes + int(ref.mapping[key % self.base - 1])
                tables.insert(0, (ref.tables[j], coarse, ref.weights[j - 1]))
            for table, k, kappa in tables:
                ix = table.find(np.array([k], np.uint64))[0]
                if ix < 0:
                    continue
                w = kappa * max(table.D[ix], 1)
                den = table.C[ix] + w
                if token is None:
                    ids, n = table.followers(ix)
                    p *= w / den
                    p[ids] += n / den
                else:
                    query = ix * (model.V + 1) + token
                    index = np.searchsorted(table.cells, query)
                    n = table.counts[index] if index < len(table.cells) and table.cells[index] == query else 0
                    p = (n + w * p) / den
        for j, (key, level) in enumerate(zip(self.keys(model.document), self.levels)):
            entry = level.get(key)
            if entry is None:
                continue
            total, counts = entry
            w = b[j] * max(len(counts), 1)
            if token is None:
                p *= w / (total + w)
                ids = np.fromiter(counts, dtype=int)
                p[ids] += np.fromiter(counts.values(), dtype=float) / (total + w)
            else:
                p = (counts.get(token, 0) + w * p) / (total + w)
        return p

    def distribution(self):
        return self._predict()

    def probability(self, token, a=None, b=None):
        return self._predict(int(token), a, b)

    def prefill(self, ids):
        for token in ids:
            self.observe(token)
        return self
