"""Lettura incrementale delle stesse regioni diadiche usate dal modello tabulare."""

from collections import Counter
from copy import copy

import numpy as np

from acan.core.category_table import Table
from .dyadic import TARGET


class ConditionalText:
    def __init__(self, geometry, graph, memory, base=None):
        self.geometry, self.graph, self.memory = geometry, graph, memory
        self.base = base
        self.reading = graph.category_regions[TARGET]
        self.features = list(dict.fromkeys([f for f, t, k in self.reading.levels] + [f for f, k in memory]))

    def cursor(self):
        return Cursor(self)

    def generate(self, prompt, count):
        cursor = self.cursor().prefill(prompt)
        result = []
        for _ in range(count):
            result.append(int(cursor.distribution().argmax()))
            cursor.observe(result[-1])
        return result


class Cursor:
    def __init__(self, model):
        self.model = model
        self.base = model.base.cursor() if model.base is not None else None
        self.n = 0
        geometry = model.geometry
        # At scale j an ordered composition needs the child from 2**j positions earlier.
        self.rings = [np.zeros(1 << j, np.uint32) for j in range(len(geometry.pairs))]
        self.tails = [
            np.zeros(
                1 + max((lag for f in model.features for lag, s, b in f if s == j), default=0),
                dtype=np.uint32,
            )
            for j in range(len(geometry.depths))
        ]
        self.window = 1 << (geometry.scales - 1)
        self.tables = {f: Table.empty(len(geometry.vectors)) for f, k in model.memory}
        self.previous = self.tables.copy()
        self.delta = {f: {} for f, k in model.memory}
        self.current = {f: int(not f) for f in model.features}

    def fork(self):
        other = copy(self)
        other.base = self.base.fork() if self.base is not None else None
        other.rings = [r.copy() for r in self.rings]
        other.tails = [r.copy() for r in self.tails]
        other.current = self.current.copy()
        other.tables = self.tables.copy()
        other.previous = self.previous.copy()
        other.delta = {f: {k: c.copy() for k, c in m.items()} for f, m in self.delta.items()}
        return other

    def advance(self, ids):
        if self.base is not None:
            self.base.prefill(ids)
        g = self.model.geometry
        times = self.n + np.arange(len(ids))
        child = ids
        cols = [g.token_codes[ids]]
        for j, ((tree, vectors), ring) in enumerate(zip(g.pairs, self.rings)):
            half = len(ring)
            overlap = min(half, len(ids))
            older = np.empty_like(child)
            older[:overlap] = ring[times[:overlap] % half]
            older[overlap:] = child[: len(ids) - overlap]
            ring[times[-overlap:] % half] = child[-overlap:]
            valid = times >= 2 * half - 1
            new = np.zeros(len(ids), np.uint32)
            if valid.any():
                x = np.column_stack((vectors[older[valid]], vectors[child[valid]]))
                new[valid] = tree.encode(x)
            cols.append(new)
            child = new
        windows = [np.concatenate((tail, col)) for tail, col in zip(self.tails, cols)]
        target_times = np.r_[times, self.n + len(ids)]
        keys = {}
        for f in self.model.features:
            a = np.ones(len(ids) + 1, np.uint64)
            for lag, j, b in f:
                start = len(self.tails[j]) - 1 - lag
                code = windows[j][start : start + len(a)] >> (g.depths[j] - b)
                a = (a << np.uint64(b)) | (code & ((1 << b) - 1)).astype(np.uint64)
            if f:
                available = max(lag + (1 << j) for lag, j, b in f)
                a[target_times < available] = 0
            keys[f] = a[:-1]
            self.current[f] = int(a[-1])
        self.tails = [w[-len(t) :].copy() for w, t in zip(windows, self.tails)]
        self.n += len(ids)
        return keys

    def rotate(self):
        if self.n % self.window:
            return
        # Two adjacent epochs keep at least the requested horizon after warmup.
        # Merge only completed observations. Tables shared by a fork are immutable.
        for f, table in self.tables.items():
            extra = self.delta[f]
            if extra:
                keys = sorted(extra)
                targets, counts, masses = [], [], []
                for i, key in enumerate(keys):
                    counts_for_key = sorted(extra[key].items())
                    targets.extend(i * (table.V + 1) + y for y, c in counts_for_key)
                    counts.extend(c for y, c in counts_for_key)
                    masses.append(sum(c for y, c in counts_for_key))
                table = table + Table(
                    np.asarray(keys, np.uint64),
                    np.asarray(masses),
                    np.asarray(targets),
                    np.asarray(counts),
                    table.V,
                )
            self.previous[f] = table
            self.tables[f] = Table.empty(table.V)
            self.delta[f] = {}

    def prefill(self, ids):
        ids = np.asarray(ids, dtype=np.int32)
        begin = 0
        while begin < len(ids):
            size = min(32768, self.window - self.n % self.window, len(ids) - begin)
            part = ids[begin : begin + size]
            keys = self.advance(part)
            for f in self.tables:
                valid = keys[f] != 0
                if valid.any():
                    table = Table.count(keys[f][valid], part[valid], self.tables[f].V)
                    self.tables[f] = self.tables[f] + table
            self.rotate()
            begin += size
        return self

    def observe(self, token):
        for f, m in self.delta.items():
            key = self.current[f]
            if key:
                if key not in m:
                    m[key] = Counter()
                m[key][token] += 1
        self.advance(np.asarray([token], dtype=np.int32))
        self.rotate()

    def distribution(self):
        p = self.model.reading.distribution(
            self.current.__getitem__, self.base.distribution() if self.base else None
        )
        for f, kappa in self.model.memory:
            key = self.current[f]
            observed = []
            for tab in (self.previous[f], self.tables[f]):
                index = tab.find([key])[0]
                if index >= 0:
                    observed.append(tab.followers(index))
            extra = self.delta[f].get(key)
            if extra:
                observed.append(
                    (np.fromiter(extra.keys(), dtype=int), np.fromiter(extra.values(), dtype=int))
                )
            if not observed:
                continue
            c = sum(int(counts.sum()) for _, counts in observed)
            d = (
                len(observed[0][0])
                if len(observed) == 1
                else len(np.unique(np.concatenate([y for y, _ in observed])))
            )
            w = kappa * d
            p *= w / (c + w)
            for ys, counts in observed:
                p[ys] += counts / (c + w)
        return p
