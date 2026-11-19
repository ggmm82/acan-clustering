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

    def encode(self, ids, starts=None):
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
        return row


class PairLanguageModel:
    def __init__(self, encoder, root, tables, weights, memory):
        self.encoder, self.root = encoder, root
        self.tables, self.weights, self.memory = tables, weights, memory
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
        return self

    def fork(self):
        other = copy(self)
        other.overlay = {f: {k: c.copy() for k, c in vals.items()} for f, vals in self.overlay.items()}
        return other

    def _keys(self):
        if self.next_keys is None:
            row = self.model.encoder.encode(np.r_[self.tail, np.int32(0)])
            self.next_keys = {
                f: int(row.feature(f)[-1])
                for f in dict.fromkeys([*self.model.weights, *self.model.memory])
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
        self.next_keys = None

    def distribution(self):
        model = self.model
        p = self.model.leaf_weights[int(self.tail[-1]) + 1 if len(self.tail) else 0] @ model.head
        keys = self._keys()
        for f, k in model.weights.items():
            table = model.tables[f]
            ix = table.find(np.array([keys[f]], np.uint64))[0]
            if ix >= 0:
                targets, counts = table.followers(ix)
                weight = k * max(table.D[ix], 1)
                denom = table.C[ix] + weight
                p = p * (weight / denom)
                p[targets] += counts / denom
        for f, k in model.memory.items():
            key = keys[f]
            counts = Counter()
            table = self.tables.get(f)
            if table is not None and key:
                ix = table.find(np.array([key], np.uint64))[0]
                if ix >= 0:
                    ys, ns = table.followers(ix)
                    counts.update(dict(zip(ys, ns)))
            counts.update(self.overlay[f].get(key, {}))
            if counts:
                weight = k * len(counts)
                denom = sum(counts.values()) + weight
                p *= weight / denom
                p[list(counts)] += np.fromiter(counts.values(), float) / denom
        return p
