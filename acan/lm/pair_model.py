"""Causal encoding and categorical reading of the native tabular concept layers."""

from collections import Counter
from copy import copy

import numpy as np
import pandas as pd

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table


def shift(values, distance, ages):
    out = np.zeros_like(values)
    if distance == 0:
        return values
    out[distance:] = values[:-distance]
    out[ages < distance] = 0
    return out


def stride(width):
    """How often the topic of a scale is recognized again: what W tokens talk about changes slowly."""
    return max(1, width // 8)


def span_means(vectors, row, width, at=None):
    """The mean vector of the tokens (i-width, i] of the document of each position i (or of ``at``)."""
    n = len(row.y)
    at = np.arange(n) if at is None else np.asarray(at)
    total = np.cumsum(np.vstack([np.zeros(vectors.shape[1]), vectors[row.y]]), 0)
    end = at + 1
    begin = np.maximum(end - width, end - 1 - row.age[at])
    return (total[end] - total[begin]) / (end - begin)[:, None]


def recognize(graph, x):
    """The most probable concept (1-based) of numeric rows: the steps of ``graph.assignments(...).argmax(1)``
    without the table, the encoding of a frame and the last normalization, which do not change the maximum."""
    x = (np.asarray(x, float) - graph.encoder.mean) / graph.encoder.std
    data = (x, np.ones_like(x), np.zeros((len(x), graph.encoder.width)))
    seen = graph._seen(data)[0]
    s = graph.statistics
    logits = graph._log_density(s, seen, graph._weights(0.)) + np.log(np.maximum(s['C'], 1e-300) / s['C'].sum())
    return logits.argmax(1) + 1


def topic_codes(graph, vectors, row, width):
    """The concept of what the last ``width`` tokens talk about, recognized every ``stride(width)`` tokens of a
    document and kept in between: position i reads the topic of the span ending at its last anchor."""
    step = stride(width)
    anchor = np.arange(len(row.y)) - row.age % step
    points, back = np.unique(anchor, return_inverse=True)
    means = span_means(vectors, row, width, points)
    codes = np.zeros(len(points), np.uint32)
    for i in range(0, len(codes), 65536):
        codes[i:i + 65536] = recognize(graph, means[i:i + 65536])
    return codes[back]


def topic_of(graph, vectors, recent, age, width):
    """The topic column of the last token read, as ``topic_codes`` gives it in bulk.

    ``recent`` are the last tokens of the document (at least ``width + stride(width)`` of them, or all
    since it began) and ``age`` is the position of the last one in its document.
    """
    if age < 0 or not len(recent):
        return 0
    first = max(0, len(recent) - 1 - age)                       # where the document begins inside ``recent``
    end = len(recent) - age % stride(width)
    span = vectors[np.asarray(recent[max(end - width, first):end])]
    return int(recognize(graph, span.mean(0)[None])[0])


def topic_chain(model, keys):
    """q(topic) / q(general) for every token, or None when no topic concept has been seen.

    The topics of all scales are read as a chain, from the shortest span to the longest: each scale is the
    categorical reading of its concept with the scale before as level above, the general frequency first,
    so every scale adds only what the shorter ones did not say. The concept's evidence sets how far the
    reading moves (kappa pseudo-observations per distinct token, as in every reading of the model).
    """
    topics = getattr(model, "topics", {})
    q, seen = model.unigram, False
    for f in sorted(topics, key=lambda name: int(name[1:])):
        kappa, active = topics[f]
        table = model.tables.get(f)
        if not active or table is None or not keys[f]:
            continue
        ix = table.find(np.array([keys[f]], np.uint64))[0]
        if ix < 0:
            continue
        targets, counts = table.followers(ix)
        a = kappa * max(table.D[ix], 1)
        q = q * (a / (table.C[ix] + a))
        q[targets] += counts / (table.C[ix] + a)
        seen = True
    return q / model.unigram if seen else None


def reach(f):
    """How many tokens before the next one a field looks at: the sum of the spans it joins."""
    return 0 if f == "root" else sum(int(part[1:]) for part in f.split("|"))


def merged_chain(corpus, memory):
    """One chain from the general to the particular: corpus readings and memory of the document, each
    in its own order, merged by how far back they look. A reading of the document on a short context is
    followed by the readings of the longer contexts, which correct it where they know better."""
    out, i = [], 0
    corpus = list(corpus)
    for f in memory:
        while i < len(corpus) and reach(corpus[i]) <= reach(f):
            out.append(("corpus", corpus[i])); i += 1
        out.append(("memory", f))
    return out + [("corpus", f) for f in corpus[i:]]


def chain(model):
    order = getattr(model, "order", None)
    if order is None:  # models saved before the merged chain: corpus first, then the document
        order = [("corpus", f) for f in model.weights] + [("memory", f) for f in model.memory]
    return order


def read_distribution(model, previous, keys, memory):
    """The probability of every token: one chain from the general to the particular, where every step is
    a reading of the corpus or of the memory of the document (chain), then what the topics say: Bayes'
    rule, the reading times q(topic) / q(general) (topic_chain), normalized."""
    p = model.leaf_weights[previous] @ model.head
    for kind, f in chain(model):
        if kind == "corpus":
            k, table = model.weights[f], model.tables[f]
            ix = table.find(np.array([keys[f]], np.uint64))[0]
            if ix < 0:
                continue
            targets, counts = table.followers(ix)
            weight = k * max(table.D[ix], 1)
            denom = table.C[ix] + weight
            p = p * (weight / denom)
            p[targets] += counts / denom
        else:
            counts = memory(f, keys[f])
            if counts:
                weight = model.memory[f] * len(counts)
                denom = sum(counts.values()) + weight
                p = p * (weight / denom)
                p[list(counts)] += np.fromiter(counts.values(), float) / denom
    ratio = topic_chain(model, keys) if getattr(model, "topics", None) else None
    if ratio is not None:
        p = p * ratio
        p /= p.sum()
    return p


def walk(model, row):
    """The distribution at every position of a text read in bulk, exactly as a cursor reads it token by token."""
    names = list(dict.fromkeys([*model.weights, *model.memory, *getattr(model, "topics", {})]))
    keys = {f: row.feature(f) for f in names}
    previous = np.r_[0, row.y[:-1].astype(np.int64) + 1]
    previous[row.age == 0] = 0
    seen = {f: {} for f in model.memory}
    for i in range(len(row.y)):
        if row.age[i] == 0:
            seen = {f: {} for f in model.memory}
        yield read_distribution(model, previous[i], {f: int(keys[f][i]) for f in names},
                                lambda f, key: seen[f].get(key) if key else None)
        for f in model.memory:
            key = int(keys[f][i])
            if key:
                seen[f].setdefault(key, Counter())[int(row.y[i])] += 1


class EncodedPrefix:
    def __init__(self, ids, starts):
        self.y = np.asarray(ids, dtype=np.int32)
        positions = np.arange(len(ids))
        self.age = positions - starts[np.searchsorted(starts, positions, side="right") - 1]
        self.exact_columns = {}
        self.spans = {"e1": self.y.astype(np.uint32) + 1}

    def feature(self, f):
        if f == "root":
            return np.ones(len(self.y), np.uint32)
        parts = f.split("|")
        recent = parts[-1]
        if len(parts) == 1 and recent in self.exact_columns:
            left, right, width = self.exact_columns[recent]
            return CategoryPairs.keys(
                shift(self.spans[left], width + 1, self.age), shift(self.spans[right], 1, self.age)
            )
        key = shift(self.spans[recent], 1, self.age)
        if len(parts) == 2:
            old = shift(self.spans[parts[0]], 1 + int(recent[1:]), self.age)
            key = CategoryPairs.keys(old, key)
        return key


class PairEncoder:
    def __init__(self, vectors, token_classes, nodes, exact, local):
        self.vectors = vectors
        self.token_classes = token_classes
        self.nodes = nodes
        self.exact = exact
        self.local = local
        self.width = max([6, *nodes])
        self._memo = {}
        self.open_exact = False
        self.topics = {}                                        # W -> concept model of the mean of W topic vectors
        self.topic_vectors = None                               # tokens by the documents they appear in (LSA)

    def encode(self, ids, starts=None, topics=True):
        starts = np.asarray([0] if starts is None else starts, dtype=int)
        row = EncodedPrefix(ids, starts)
        row.spans["c1"] = self.token_classes[row.y]
        child = row.y.astype(np.uint32) + 1
        vectors = self.vectors
        for w, (graph, next_vectors) in self.nodes.items():
            half = w // 2
            k = CategoryPairs.keys(shift(child, half, row.age), child)
            typ, inv = np.unique(k, return_inverse=True)
            active = np.flatnonzero(typ != 0)
            codes = np.zeros(len(typ), np.uint32)
            # A bounded direct-mapped cache saves repeated recognition of identical ordered pairs.
            if w not in self._memo:
                slots = 1 << (20 if w == 2 else 16)
                self._memo[w] = (np.zeros(slots, np.uint64), np.zeros(slots, np.uint32))
            keys, values = self._memo[w]
            mixed = typ ^ (typ >> np.uint64(33))
            slot = (
                (mixed * np.uint64(11400714819323198485)) >> np.uint64(64 - (len(keys).bit_length() - 1))
            ).astype(int)
            hit = keys[slot[active]] == typ[active]
            codes[active[hit]] = values[slot[active[hit]]]
            missing = active[~hit]
            for i in range(0, len(missing), 1024):
                ix = missing[i : i + 1024]
                left = (typ[ix] >> 32).astype(int) - int(w == 2)
                right = (typ[ix] & 0xFFFFFFFF).astype(int) - int(w == 2)
                x = np.column_stack((vectors[left], vectors[right]))
                codes[ix] = (
                    graph.assignments(pd.DataFrame(x, columns=graph.encoder.numeric)).argmax(1) + 1
                )
            keys[slot[missing]] = typ[missing]
            values[slot[missing]] = codes[missing]
            child = codes[inv]
            row.spans[f"c{w}"] = child
            vectors = next_vectors
        for w, (left, right, enc) in self.exact.items():
            if getattr(self, "open_exact", False):
                row.exact_columns[f"e{w}"] = (f"e{left}", f"e{right}", right)
            row.spans[f"e{w}"] = enc.transform(
                shift(row.spans[f"e{left}"], right, row.age), row.spans[f"e{right}"]
            )
        for w, steps in self.local.items():
            result = row.spans[f"c{w}"].copy()
            for older, recent, offset, encoder in steps:
                parent = result if recent == "current" else row.spans[recent]
                left = shift(row.spans[older], w // 2, row.age)
                left[row.age < w - 1] = 0
                code = encoder.transform(parent, left)
                seen = code > 0
                result[seen] = offset + code[seen]
            row.spans[f"s{w}"] = result
        if topics:
            for w, graph in getattr(self, "topics", {}).items():
                row.spans[f"t{w}"] = topic_codes(graph, self.topic_vectors, row, w)
        return row


class PairLanguageModel:
    def __init__(self, encoder, root, tables, weights, memory):
        self.encoder, self.root = encoder, root
        self.tables, self.weights, self.memory = tables, weights, memory
        self.topics = {}                                        # f -> [kappa, beta]: topics read by multiplying
        self.unigram = None
        self.vocabulary = root.observed_targets["token"]["probability"].shape[1]
        leaf = np.vstack([np.full(encoder.vectors.shape[1], np.nan), encoder.vectors])
        self.leaf_weights = root.assignments(pd.DataFrame(leaf, columns=root.encoder.numeric))
        self.head = root.observed_targets["token"]["probability"]

    def root_probability(self, row, pos):
        prev = row.feature("e1")
        return np.concatenate(
            [
                np.einsum("ij,ji->i", self.leaf_weights[prev[ix]], self.head[:, row.y[ix]])
                for ix in np.array_split(pos, max(1, (len(pos) + 511) // 512))
            ]
        )

    def cursor(self, ids=()):
        return PairCursor(self).prefill(ids)


class PairCursor:
    def __init__(self, model):
        self.model = model
        self.tail = np.empty(0, np.int32)
        self.tables = {}
        self.next_keys = None
        self.overlay = {f: {} for f in model.memory}
        # What the document talks about: its last tokens and the topic of every scale, kept as bulk does.
        self.age, self.recent, self.topic = -1, np.empty(0, np.int32), {}

    def prefill(self, ids):
        if not len(ids):
            return self
        if len(self.tail) or self.tables or any(self.overlay.values()):
            for token in ids:
                self.observe(token)
            return self
        ids = np.asarray(ids, np.int32)
        pending = []
        for start in range(0, len(ids), 131072):
            left = max(0, start - self.model.encoder.width)
            end = min(len(ids), start + 131072)
            row = self.model.encoder.encode(ids[left:end])
            select = slice(start - left, None)
            layer = {
                f: Table.count_regions([(row.feature(f)[select], row.y[select])], self.model.vocabulary)
                for f in self.model.memory
            }
            level = 0
            while level < len(pending) and pending[level] is not None:
                layer = {f: t + pending[level][f] for f, t in layer.items()}
                pending[level] = None
                level += 1
            if level == len(pending):
                pending.append(layer)
            else:
                pending[level] = layer
        for layer in pending:
            if layer is not None:
                self.tables = (
                    layer if not self.tables else {f: t + self.tables[f] for f, t in layer.items()}
                )
        self.tail = ids[-self.model.encoder.width :].copy()
        self.next_keys = None
        self.age, self.recent = len(ids) - 1, ids[-self._keep():].copy()
        self._topics()
        return self

    def _keep(self):
        topics = getattr(self.model.encoder, "topics", {})
        return max([1, *(w + stride(w) for w in topics)])

    def _topics(self):
        enc = self.model.encoder
        for w, graph in getattr(enc, "topics", {}).items():
            if self.age >= 0 and self.age % stride(w) == 0 or w not in self.topic:
                self.topic[w] = topic_of(graph, enc.topic_vectors, self.recent, self.age, w)

    def fork(self):
        other = copy(self)
        other.topic = dict(self.topic)
        other.overlay = {f: {k: c.copy() for k, c in vals.items()} for f, vals in self.overlay.items()}
        return other

    def _keys(self):
        if self.next_keys is None:
            row = self.model.encoder.encode(np.r_[self.tail, np.int32(0)], topics=False)
            if any(w not in self.topic for w in getattr(self.model.encoder, "topics", {})):
                self._topics()
            for w in getattr(self.model.encoder, "topics", {}):  # the topics come from the state, as in bulk
                column = np.zeros(len(row.y), np.uint32)
                if len(row.y) > 1:
                    column[-2] = self.topic[w]
                row.spans[f"t{w}"] = column
            self.next_keys = {
                f: int(row.feature(f)[-1])
                for f in dict.fromkeys([*self.model.weights, *self.model.memory, *getattr(self.model, "topics", {})])
            }
        return self.next_keys

    def observe(self, token, remember=True):
        """Read a token. ``remember=False`` does not add it to the memory of the document: the text the
        model writes itself is not evidence of what the document says."""
        keys = self._keys()
        if remember:
            for f, values in self.overlay.items():
                key = keys[f]
                if key:
                    values.setdefault(key, Counter())[int(token)] += 1
        self.tail = np.r_[self.tail, np.int32(token)][-self.model.encoder.width :]
        self.age += 1
        self.recent = np.r_[self.recent, np.int32(token)][-self._keep():]
        self._topics()
        self.next_keys = None

    def distribution(self):
        keys = self._keys()

        def memory(f, key):
            counts = Counter()
            table = self.tables.get(f)
            if table is not None and key:
                ix = table.find(np.array([key], np.uint64))[0]
                if ix >= 0:
                    ys, ns = table.followers(ix)
                    counts.update(dict(zip(ys, ns)))
            counts.update(self.overlay[f].get(key, {}))
            return counts

        previous = int(self.tail[-1]) + 1 if len(self.tail) else 0
        return read_distribution(self.model, previous, keys, memory)
