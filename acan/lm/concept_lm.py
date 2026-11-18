"""The language model that is the concept model: no neural network, no gradient.

The training text is a table. Every position t is a row: the tokens before it are its attributes,
the token at t is its target. The concepts are those of the concept graph (MODEL.md): nodes that keep
only sums of their members, read with mass times similarity, generalized where the reader loses least,
judged on rows they never saw.

Concepts by context. At level j a concept holds the positions whose last j tokens are equal (level 0
holds every position). It keeps

    C = members,   N(y) = members followed by y,   D = distinct tokens that follow.

Reading. A concept is read with its level above (the context one token shorter) as prior, as a concept
of a table is read with the scale of its level above:

    p_j(y) = (N_j(y) + w_j p_{j-1}(y)) / (C_j + w_j),      w_j = a_j * D_j,

so a concept followed by many different tokens trusts its parent more. The level above of level 0 is
the uniform distribution. A context never seen leaves the reading as it is (the concept was generalized
into its parent). The weights a_j are chosen by the judgment on unseen text.

Concepts of the text being read. The same concepts built only from the earlier positions of the text in
hand, read after the global ones with the global reading as prior (weights b_j): what the text said
before can be said again, at any distance.

Learning is counting: a piece of text is a table of sums, and two tables are added together. Pieces are
added like the dyadic blocks of the other concept layers: equal tables are merged two by two, so every
position is merged about log2(pieces) times. Above the memory budget the concepts the reader loses least
on are generalized into their parent (their sums are already in it): the loss is the log-likelihood of
the members read through the parent instead of the concept.

Contexts are written as integers: key_j = key_{j-1} * (V + 1) + token + 1 in 64 bits. Up to
(V + 1)^j < 2^64 (four tokens with 32,000 tokens of vocabulary) the key is exact; beyond, it wraps
around as a hash, and two contexts may share a key with probability about n^2 / 2^64.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from acan.concept_algebra import KAPPA_GRID, categorical_reading

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
MARGIN = .6                                                    # pruning leaves the sums at this fraction of the budget
GRID = KAPPA_GRID                                               # weights tried by the judgment, as for a table


def context_keys(ids, depth, vocabulary):
    """Keys of the contexts of every position for levels 0..depth (uint64); positions with fewer tokens get 0 and are invalid."""
    ids = np.asarray(ids, np.int64)
    base = np.uint64(vocabulary+1)
    keys = [np.zeros(len(ids), np.uint64)]
    valid = [np.ones(len(ids), bool)]
    key = np.zeros(len(ids), np.uint64)
    with np.errstate(over='ignore'):
        for j in range(1, depth+1):
            shifted = np.zeros(len(ids), np.uint64)
            shifted[j:] = (ids[:-j]+1).astype(np.uint64) if j < len(ids) else shifted[j:]
            key = key*base+shifted
            keys.append(key.copy())
            ok = np.zeros(len(ids), bool); ok[j:] = True
            valid.append(ok)
    return keys, valid


def _union(a, b):
    """The values of two sorted arrays without repeats, sorted."""
    both = np.concatenate((a, b))
    both.sort(kind='stable')                                    # two sorted runs: a merge
    keep = np.empty(len(both), bool)
    keep[0] = True
    np.not_equal(both[1:], both[:-1], out=keep[1:])
    return both[keep]


def _add_into(tables, others):
    """tables[j] += others[j], a level at a time, so the old sums of a level are freed before the next."""
    for j in range(len(tables)):
        tables[j] = tables[j]+others[j]
        others[j] = None


def _threshold(ratios, sizes, need):
    """The smallest loss per byte t such that the concepts with ratio <= t hold at least ``need`` bytes.

    Found by halving the interval (60 times), so no ordering of all concepts is held in memory.
    """
    low = min((float(r.min()) for r in ratios if len(r)), default=0.)
    high = max((float(r.max()) for r in ratios if len(r)), default=0.)
    if sum(int(s.sum(dtype=np.int64)) for s in sizes) <= need:
        return high

    def freed(t):
        return sum(int(s[r <= t].sum(dtype=np.int64)) for r, s in zip(ratios, sizes))
    if freed(low) >= need:
        return low
    for _ in range(60):
        middle = (low+high)/2
        if middle in (low, high):
            break
        low, high = (low, middle) if freed(middle) >= need else (middle, high)
    return high


def _scatter_add(total, where, values):
    """total[where] += values for indices that do not repeat (each list holds every key once)."""
    total[where] += values


class Table:
    """The concepts of one level: sorted keys with C, D and the cells (concept, token) with their counts."""

    def __init__(self, keys, C, cells, counts, vocabulary):
        self.V = vocabulary
        self.keys, self.C, self.cells, self.counts = keys, C, cells, counts
        self.D = np.bincount((cells//(vocabulary+1)).astype(np.int64), minlength=len(keys)) if len(keys) else np.zeros(0, np.int64)

    @classmethod
    def count(cls, keys, targets, vocabulary):
        """The sums of one level from the context keys and the targets of some positions."""
        concepts, which, C = np.unique(keys, return_inverse=True, return_counts=True)
        cells, counts = np.unique(which.astype(np.int64)*(vocabulary+1)+targets, return_counts=True)
        return cls(concepts, C.astype(np.int64), cells, counts.astype(np.int64), vocabulary)

    @classmethod
    def empty(cls, vocabulary):
        return cls(np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), vocabulary)

    def pairs(self):
        """Every cell as (context key, token, count)."""
        concept = (self.cells//(self.V+1)).astype(np.int64)
        return self.keys[concept], self.cells % (self.V+1), self.counts

    def __add__(self, other):
        """Two tables of the same level added together: sums are additive.

        Both lists are sorted, so the sum is a merge of sorted lists: the keys of both, each once, and the
        cells (concept, token) of both renumbered on the merged keys, each once. Every array is built in
        place where possible: the process holds the two tables, their sum and about one more copy of the
        cells.
        """
        if not len(other.keys):
            return self
        if not len(self.keys):
            return other
        base = self.V+1
        keys = _union(self.keys, other.keys)
        renumbered = []
        for table in (self, other):
            at = np.searchsorted(keys, table.keys)
            C_part = table.C
            cells = table.cells//base
            np.take(at, cells, out=cells)
            cells *= base
            cells += table.cells % base
            renumbered.append((at, C_part, cells, table.counts))
            del at
        C = np.zeros(len(keys), np.int64)
        for at, C_part, _, _ in renumbered:
            _scatter_add(C, at, C_part)
        cells = _union(renumbered[0][2], renumbered[1][2])
        counts = np.zeros(len(cells), np.int64)
        for _, _, part, n in renumbered:
            _scatter_add(counts, np.searchsorted(cells, part), n)
        return Table(keys, C, cells, counts, self.V)

    def keep(self, alive):
        """Only the concepts marked alive, with their cells."""
        concept = (self.cells//(self.V+1)).astype(np.int64)
        kept = alive[concept]
        renumber = np.cumsum(alive)-1
        cells = renumber[concept[kept]]*(self.V+1)+self.cells[kept] % (self.V+1)
        return Table(self.keys[alive], self.C[alive], cells, self.counts[kept], self.V)

    def find(self, keys):
        """Index of the concept of every key, -1 where there is none."""
        if not len(self.keys):
            return np.full(len(keys), -1, np.int64)
        place = np.minimum(np.searchsorted(self.keys, keys), len(self.keys)-1)
        return np.where(self.keys[place] == keys, place, -1)

    def lookup(self, keys, targets):
        """C, N(target) and D of the concept of every key (zeros where there is none)."""
        index = self.find(keys)
        seen = index >= 0
        safe = np.where(seen, index, 0)
        cell = safe*(self.V+1)+targets
        N = np.zeros(len(keys))
        if len(self.cells):
            at = np.minimum(np.searchsorted(self.cells, cell), len(self.cells)-1)
            N = np.where(seen & (self.cells[at] == cell), self.counts[at], 0).astype(float)
        if not len(self.keys):
            z = np.zeros(len(keys)); return z, z, z
        return np.where(seen, self.C[safe], 0).astype(float), N, np.where(seen, self.D[safe], 0).astype(float)

    def followers(self, index):
        """The tokens that follow concept ``index`` and their counts."""
        start = np.searchsorted(self.cells, index*(self.V+1))
        end = np.searchsorted(self.cells, (index+1)*(self.V+1))
        return self.cells[start:end] % (self.V+1), self.counts[start:end]

    @property
    def size(self):
        return len(self.keys)+len(self.cells)

    def nbytes(self):
        return sum(array.nbytes for array in (self.keys, self.C, self.D, self.cells, self.counts))


EXACT = -1                                                      # a context symbol that is the token itself
PATH_DEPTH = 2                                                  # tokens of context read by the paths through concepts


class Reading:
    """One factor of a path: the concepts that read one step from the context, for every node of the step.

    ``node[v]`` is the concept of token v reached by the steps before (0 for the first step), ``value[v]``
    is what this step reads of v (a concept of the next level, or the token itself), ``size`` the values.
    Its levels are keys: level 0 is the node, every next level adds one symbol of the context, the token at
    a distance or its concept (``symbols``: (map, distance), map EXACT for the token itself).
    """

    def __init__(self, node, value, size, symbols):
        self.node, self.value, self.size, self.symbols = np.asarray(node, np.int64), np.asarray(value, np.int64), int(size), list(symbols)
        pairs = self.node*(self.size+1)+self.value
        self.codes, self.token_pair = np.unique(pairs, return_inverse=True)      # (node, value) pairs that exist
        self.pair_node = self.codes//(self.size+1)
        self.pair_value = self.codes % (self.size+1)
        self.children = np.bincount(self.pair_node, minlength=int(self.node.max())+1)  # values each node can take

    def keys(self, ids, maps, vocabulary):
        """Keys and validity of every level for every position, as a generator (one level held at a time)."""
        n = len(ids)
        base = np.uint64(vocabulary+1)
        key = self.node[ids].astype(np.uint64)
        valid = np.ones(n, bool)
        yield key, valid
        with np.errstate(over='ignore'):
            for kind, distance in self.symbols:
                source = ids if kind == EXACT else maps[kind][ids]
                shifted = np.zeros(n, np.uint64)
                if distance < n:
                    shifted[distance:] = (source[:-distance]+1).astype(np.uint64)
                key = key*base+shifted
                valid = valid.copy(); valid[:min(distance, n)] = False
                yield key, valid


def standard_paths(levels, depth):
    """The paths read by the model from levels of token concepts (each a list of maps, general to particular).

    Always the direct path (the token from the exact contexts). For every hierarchy of concepts, the path
    through its last two levels (general, then particular, then the token) with the particular concepts in
    the context. More paths read a little better when the memory is free, worse when it is not: the sums
    of every path share the same budget. Returns (steps, context) with indices into the list of maps.
    """
    maps, paths = [], [((), EXACT)]
    for hierarchy in levels:
        start = len(maps)
        maps.extend(hierarchy)
        n = len(hierarchy)
        if n == 0:
            continue
        index = lambda i: start+min(max(i, 0), n-1)
        chosen = tuple(dict.fromkeys(index(s) for s in (n-2, n-1)))
        paths.append((chosen, index(n-1)))
    return maps, list(dict.fromkeys(paths))


class ConceptLM:
    """Concepts of a text by context, global (the training text) and of the text being read.

    A path reads the next token through concepts of tokens: p(y) = prod_s p(step_s(y) | steps before, context),
    every step with the reading of a categorical attribute; the direct path reads the token at once. The
    probability of a token is the mixture of the paths, with weights from the judgment.
    """

    def __init__(self, vocabulary, depth=6, document=4, budget_bytes=None, maps=(), paths=None, path_depth=None):
        self.V, self.depth, self.document = int(vocabulary), int(depth), int(document)
        self.path_depth = self.depth if path_depth is None else min(int(path_depth), self.depth)
        self.budget_bytes = budget_bytes
        self.maps = [np.asarray(m, np.int64) for m in maps]
        self.paths = list(paths) if paths is not None else [((), EXACT)]
        self.readings, self.of_path = [], []
        identity = np.arange(self.V)
        for steps, context in self.paths:
            reach = self.depth if not steps else self.path_depth     # contexts of the paths through concepts: up to path_depth
            symbols = [(k, d) for d in range(1, reach+1) for k in ((context, EXACT) if context != EXACT else (EXACT,))]
            chain = [self.maps[s] for s in steps]+[identity]
            node = np.zeros(self.V, np.int64)
            own = []
            for value in chain:
                size = self.V if value is identity else int(value.max())+1
                own.append(len(self.readings))
                self.readings.append(Reading(node, value, size, symbols))
                node = np.unique(node*(self.V+1)+value, return_inverse=True)[1]      # the concept reached so far
            self.of_path.append(own)
        self.path_of = [k for k, own in enumerate(self.of_path) for _ in own]
        # The tables of every reading in one list: level j of reading r is levels[first[r]+j].
        self.first, self.parent, self.level_of, self.reading_of = [], [], [], []
        for r, reading in enumerate(self.readings):
            self.first.append(len(self.parent))
            for j in range(len(reading.symbols)+1):
                self.parent.append(None if j == 0 else len(self.parent)-1)
                self.level_of.append(j); self.reading_of.append(r)
        self.levels = [Table.empty(self.readings[r].size) for r in self.reading_of]
        self.a = np.ones(len(self.levels))
        self.b = np.ones(len(self.readings)*(self.document+1))
        self.w = np.full(len(self.paths), 1./len(self.paths))
        self.counter = []                                       # tables of 2^i pieces each, not yet merged into ``levels``
        self.tokens = 0
        self.generalized = 0                                    # concepts generalized into their parent to stay in the budget

    def _tables(self, r):
        return range(self.first[r], self.first[r]+len(self.readings[r].symbols)+1)

    # ---- learning: counting and adding ------------------------------------------------
    def piece(self, ids):
        """The tables of a piece of text (positions after its first token, contexts inside the piece)."""
        ids = np.asarray(ids, np.int64)
        tables = []
        for reading in self.readings:
            targets = reading.value[ids]
            for key, valid in reading.keys(ids, self.maps, self.V):
                tables.append(Table.count(key[valid], targets[valid], reading.size))
        return tables

    def add(self, ids):
        """Learn a piece of text: count it and add it to the binary counter of pieces."""
        tables = self.piece(ids)
        self.tokens += len(ids)
        level = 0
        while level < len(self.counter) and self.counter[level] is not None:
            _add_into(tables, self.counter[level])
            self.counter[level] = None
            level += 1
            tables = self._within_budget(tables)
        if level == len(self.counter):
            self.counter.append(None)
        self.counter[level] = tables
        if self.budget_bytes and self.nbytes() > self.budget_bytes:
            self.settle()                                       # the budget holds for all the sums, not for one table

    def settle(self):
        """Add every table of the counter into the levels (before reading or saving)."""
        pending, self.counter = [tables for tables in self.counter if tables is not None], []
        for tables in pending:
            _add_into(self.levels, tables)
        del pending
        self.levels = self._within_budget(self.levels)
        return self

    def merge(self, other):
        """Add another model learned on other text with the same concepts (a learning queue): the sums of both."""
        self.settle(); other.settle()
        _add_into(self.levels, other.levels)
        self.levels = self._within_budget(self.levels)
        self.tokens += other.tokens
        return self

    def nbytes(self):
        tables = list(self.levels)+[t for tables in self.counter if tables is not None for t in tables]
        return sum(table.nbytes() for table in tables)

    def concepts(self):
        return [len(table.keys) for table in self.levels]

    def _within_budget(self, tables):
        """Generalize into their parent the concepts the reader loses least on, until the tables fit the budget.

        The loss of a concept is the log-likelihood its members lose when they are read through the
        parent instead of the concept (in nats), every member read without itself:
        sum_y N(y) log(p_concept(y) / p_parent(y)), with p_concept = (N - 1 + w p_parent) / (C - 1 + w),
        w = a D (the reading), and p_parent the parent's frequencies (uniform for a value it never saw). A
        concept seen once loses nothing. Its sums are already in the parent, so nothing is lost but the
        distinction. Level 0 is never generalized.
        """
        if not self.budget_bytes:
            return tables
        total = sum(table.nbytes() for table in tables)
        if total <= self.budget_bytes:
            return tables
        # Loss per byte of every concept, a table at a time (single precision: only their order matters).
        ratios, sizes = [None]*len(tables), [None]*len(tables)
        for t in range(len(tables)):
            if self.parent[t] is None:
                ratios[t], sizes[t] = np.zeros(0, np.float32), np.zeros(0, np.int32)
                continue
            size = self._bytes_per_concept(tables[t]).astype(np.int32)
            ratios[t] = (self._losses(tables, t)/np.maximum(size, 1)).astype(np.float32)
            sizes[t] = size
        need = total-self.budget_bytes*MARGIN                   # a margin, so the next pieces fit before pruning again
        threshold = _threshold(ratios, sizes, need)             # generalize from the cheapest until under budget
        # Concepts below the threshold go; of those exactly at it (ties), only as many as still needed,
        # the longest contexts first.
        missing = need-sum(int(size[ratio < threshold].sum(dtype=np.int64)) for ratio, size in zip(ratios, sizes))
        alives = [None]*len(tables)
        for t in sorted(range(len(tables)), key=lambda t: -self.level_of[t]):
            if self.parent[t] is None:
                continue
            alive = ratios[t] > threshold
            tied = np.flatnonzero(ratios[t] == threshold)
            if len(tied):
                taken = int(np.searchsorted(np.cumsum(sizes[t][tied], dtype=np.int64), max(missing, 0)))+1 if missing > 0 else 0
                alive[tied[taken:]] = True
                missing -= int(sizes[t][tied[:taken]].sum(dtype=np.int64))
            alives[t] = alive
        for t in range(len(tables)):
            if alives[t] is None:
                continue
            self.generalized += int(len(alives[t])-alives[t].sum())
            tables[t] = tables[t].keep(alives[t])
            ratios[t] = sizes[t] = None
        return tables

    def _bytes_per_concept(self, table):
        cells = np.bincount((table.cells//(table.V+1)).astype(np.int64), minlength=len(table.keys))
        return 8*3+cells*16

    def _losses(self, tables, t):
        """Loss of reading every concept of table t through its parent (the level above in its reading)."""
        table, parent = tables[t], tables[self.parent[t]]
        if not len(table.keys):
            return np.zeros(0)
        concept = (table.cells//(table.V+1)).astype(np.int64)
        token = table.cells % (table.V+1)
        # The parent key of a concept is not written in it: it is the key of the same positions one symbol shorter.
        # It is recovered from the key where the key is exact, and otherwise the parent is read as the whole level above.
        base = np.uint64(self.V+1)
        reading, j = self.readings[self.reading_of[t]], self.level_of[t]
        exact = (self.V+1)**(j+(1 if reading.node.max() > 0 else 0)) < 2**64
        if exact:
            parent_keys = table.keys//base
            C_p, N_p, _ = parent.lookup(parent_keys[concept], token)
        else:
            C_p, N_p = np.full(len(concept), parent.C.sum(), float), np.zeros(len(concept))
        p_parent = np.where(C_p > 0, (N_p+1.)/(C_p+table.V), 1./table.V)
        # Every member read without itself (as the judgment reads rows it never saw): a concept seen once
        # tells nothing about its member, and a concept is worth what it adds to the parent for the others.
        weight = self.a[t]*np.maximum(table.D[concept], 1)
        p_concept = (table.counts-1+weight*p_parent)/(table.C[concept]-1+weight)
        per_cell = table.counts*np.log(np.maximum(p_concept, 1e-300)/p_parent)
        # A path reads the token with its weight in the mixture: what one of its concepts adds counts as much.
        share = self.w[self.path_of[self.reading_of[t]]]*len(self.paths)
        return share*np.bincount(concept, weights=per_cell, minlength=len(table.keys))

    # ---- reading ----------------------------------------------------------------------------
    def _prepared(self, r, ids, positions):
        """The sums every level of reading r gives the positions: global (C, N, D) and of the text in hand."""
        reading = self.readings[r]
        y, node = reading.value[ids], reading.node[ids]
        out = []
        for t, (key, valid) in zip(self._tables(r), reading.keys(ids, self.maps, self.V)):
            C, N, D = self.levels[t].lookup(key[positions], y[positions])
            ok = valid[positions]
            out.append((C*ok, N*ok, D*ok))
        document = []
        if self.document:
            keys, valid = context_keys(ids, self.document, self.V)
            with np.errstate(over='ignore'):
                for j in range(0, self.document+1):             # level 0: every earlier token of the text in hand
                    key = keys[j]*np.uint64(self.V+1)+node.astype(np.uint64)      # the context and the node
                    C, N, D = document_counts(key, valid[j], y)
                    document.append((C[positions], N[positions], D[positions]))
        base = 1./reading.children[node[positions]]
        return out, document, base

    def _factor(self, r, cache, a=None, b=None):
        """log p of reading r at the prepared positions."""
        a = self.a if a is None else a
        b = self.b if b is None else b
        out, document, p = cache
        start = self.first[r]
        for j, (C, N, D) in enumerate(out):
            p = categorical_reading(N, C, D, p, a[start+j], np)
        for j, (C, N, D) in enumerate(document):
            p = categorical_reading(N, C, D, p, b[r*(self.document+1)+j], np)
        return np.log(p)

    def path_logp(self, ids, positions=None):
        """log p of every path at every position (paths x positions)."""
        ids = np.asarray(ids, np.int64)
        positions = np.arange(len(ids)) if positions is None else np.asarray(positions)
        return np.stack([sum(self._factor(r, self._prepared(r, ids, positions)) for r in own) for own in self.of_path])

    def logp(self, ids, positions=None, a=None, b=None):
        """log p (nats) of the token at every position of a text, read causally.

        The global levels read the context of each position; the levels of the text read only the earlier
        positions of ``ids`` itself. The paths are mixed with their weights.
        """
        ids = np.asarray(ids, np.int64)
        positions = np.arange(len(ids)) if positions is None else np.asarray(positions)
        each = np.stack([sum(self._factor(r, self._prepared(r, ids, positions), a, b) for r in own) for own in self.of_path])
        return _mix(each, self.w)

    def judge(self, texts, positions=None):
        """Choose the weights: a_j, b_j of every reading (two rounds per weight), then the mixture of the paths.

        Each reading is a factor of its path, so its weights are chosen on its own log-likelihood; the
        mixture weights are those that read best the same texts (expectation-maximization).
        """
        positions = positions or [np.arange(1, len(text)) for text in texts]
        texts = [np.asarray(text, np.int64) for text in texts]
        factors = []
        for r in range(len(self.readings)):
            caches = [self._prepared(r, text, where) for text, where in zip(texts, positions)]

            def score(a, b):
                return np.mean(np.concatenate([self._factor(r, cache, a, b) for cache in caches]))
            for t in list(self._tables(r))*2:
                values = [score(np.r_[self.a[:t], value, self.a[t+1:]], self.b) for value in GRID]
                self.a[t] = GRID[int(np.argmax(values))]
            for j in list(range(0, self.document+1))*2:
                i = r*(self.document+1)+j
                values = [score(self.a, np.r_[self.b[:i], value, self.b[i+1:]]) for value in GRID]
                self.b[i] = GRID[int(np.argmax(values))]
            factors.append(np.concatenate([self._factor(r, cache) for cache in caches]))
        each = np.stack([sum(factors[r] for r in own) for own in self.of_path])
        w = np.full(len(self.paths), 1./len(self.paths))
        for _ in range(200 if len(self.paths) > 1 else 0):
            share = np.log(w)[:, None]+each
            share = np.exp(share-share.max(0)); share /= share.sum(0)
            w = np.maximum(share.mean(1), 1e-12); w /= w.sum()
        self.w = w
        return -float(_mix(each, w).mean())

    def _step(self, r, ids, tail_keys):
        """p(value | node, context) of reading r for the next position, for every (node, value) pair."""
        reading = self.readings[r]
        q = 1./reading.children[reading.pair_node]
        nodes = np.arange(len(reading.children))
        base = np.uint64(self.V+1)
        key = nodes.astype(np.uint64)
        n = len(ids)
        start = self.first[r]
        with np.errstate(over='ignore'):
            for j in range(len(reading.symbols)+1):
                if j:
                    kind, distance = reading.symbols[j-1]
                    if distance > n:
                        break
                    token = ids[n-distance]
                    symbol = token if kind == EXACT else self.maps[kind][token]
                    key = key*base+np.uint64(symbol+1)
                table = self.levels[start+j]
                index = table.find(key)
                found = np.flatnonzero(index >= 0)
                if not len(found):
                    continue
                q = _update(q, reading, found, index[found], table, self.a[start+j])
        if self.document:
            for j in range(0, self.document+1):
                if j > n:
                    break
                earlier = np.flatnonzero(tail_keys[j][:n] == tail_keys[j][n])
                earlier = earlier[earlier >= j]
                if not len(earlier):
                    continue
                y = ids[earlier]
                pair = np.searchsorted(reading.codes, reading.node[y]*(reading.size+1)+reading.value[y])
                cells, counts = np.unique(pair, return_counts=True)
                node = reading.pair_node[cells]
                C = np.bincount(node, weights=counts, minlength=len(reading.children))
                D = np.bincount(node, minlength=len(reading.children))
                weight = self.b[r*(self.document+1)+j]*np.maximum(D, 1)
                scale = np.where(C > 0, weight/(C+weight), 1.)
                q = q*scale[reading.pair_node]
                q[cells] += counts/(C+weight)[node]
        return q

    def distribution(self, ids):
        """Probabilities of the next token after ``ids`` (the whole vocabulary)."""
        ids = np.asarray(ids, np.int64)
        tail = np.r_[ids, 0]                                     # a position after the text
        keys, _ = context_keys(tail, self.document, self.V) if self.document else ([], None)
        p = np.zeros(self.V)
        for weight, own in zip(self.w, self.of_path):
            path = np.ones(self.V)
            for r in own:
                path *= self._step(r, ids, keys)[self.readings[r].token_pair]
            p += weight*path
        return p

    def generate(self, ids, new_tokens, stop=(), temperature=0., seed=0):
        """Continue a text: the most probable token (temperature 0) or one drawn from the reading."""
        rng = np.random.default_rng(seed)
        ids = list(np.asarray(ids, np.int64))
        produced = []
        for _ in range(new_tokens):
            p = self.distribution(np.asarray(ids))
            if temperature > 0:
                q = np.exp(np.log(np.maximum(p, 1e-300))/temperature)
                token = int(rng.choice(len(q), p=q/q.sum()))
            else:
                token = int(np.argmax(p))
            if token in stop:
                break
            ids.append(token); produced.append(token)
        return produced

    # ---- saving ---------------------------------------------------------------------------
    def arrays(self):
        self.settle()
        out = dict(a=self.a, b=self.b, w=self.w, document_zero=np.array([1], np.int64), path_depth=np.array([self.path_depth], np.int64), shape=np.array([self.V, self.depth, self.document, self.tokens, self.generalized], np.int64))
        width = max([len(steps) for steps, _ in self.paths]+[0])
        out['paths'] = np.array([[context]+list(steps)+[-1]*(width-len(steps)) for steps, context in self.paths], np.int64)
        if self.maps:
            out['maps'] = np.stack(self.maps).astype(np.int64)
        for j, table in enumerate(self.levels):
            out.update({f'level{j}.keys': table.keys, f'level{j}.C': table.C, f'level{j}.cells': table.cells,
                        f'level{j}.counts': table.counts})
        return out

    @classmethod
    def from_arrays(cls, arrays, budget_bytes=None):
        V, depth, document, tokens, generalized = (int(x) for x in arrays['shape'])
        maps = list(arrays['maps']) if 'maps' in arrays else []
        paths = None
        if 'paths' in arrays:                                    # written before the paths: only the direct one
            paths = [(tuple(int(s) for s in row[1:] if s >= 0), int(row[0])) for row in arrays['paths']]
        path_depth = int(arrays['path_depth'][0]) if 'path_depth' in arrays else None
        model = cls(V, depth, document, budget_bytes, maps, paths, path_depth)
        model.a, model.b, model.tokens, model.generalized = np.array(arrays['a']), np.array(arrays['b']), tokens, generalized
        if 'w' in arrays:
            model.w = np.array(arrays['w'])
        if 'document_zero' not in arrays:                       # written before level 0 of the text in hand: it reads nothing
            model.b[::model.document+1] = 1e12
        model.levels = [Table(arrays[f'level{j}.keys'], arrays[f'level{j}.C'], arrays[f'level{j}.cells'],
                              arrays[f'level{j}.counts'], model.levels[j].V) for j in range(len(model.levels))]
        return model


def _mix(each, weights):
    """log sum_k w_k exp(each_k): the mixture of the paths."""
    top = each.max(0)
    return np.log((weights[:, None]*np.exp(each-top)).sum(0))+top


def _update(q, reading, nodes, index, table, kappa):
    """The reading of one level for the given nodes: q(pair) <- (N + w q(pair)) / (C + w) on their pairs."""
    S = reading.size+1
    C = table.C[index].astype(float)
    weight = kappa*np.maximum(table.D[index], 1)
    scale = np.ones(len(reading.children))
    scale[nodes] = weight/(C+weight)
    q = q*scale[reading.pair_node]
    starts = np.searchsorted(table.cells, index*S)
    ends = np.searchsorted(table.cells, (index+1)*S)
    lengths = ends-starts
    if lengths.sum():
        owner = np.repeat(np.arange(len(index)), lengths)
        at = np.arange(lengths.sum())-np.repeat(np.cumsum(lengths)-lengths, lengths)+np.repeat(starts, lengths)
        value = table.cells[at] % S
        pair = np.searchsorted(reading.codes, nodes[owner]*S+value)
        q[pair] += table.counts[at]/(C+weight)[owner]
    return q


def token_concepts(ids, vocabulary, resolutions=(16, 48), top=256, rows=8000, seed=0, least=3):
    """Levels of concepts of the tokens, found by the concept model itself (acan/core/concepts.py).

    A token is a row: how often each of the ``top`` most frequent tokens comes just before it and just
    after it (square roots of the frequencies, so two rows are compared as two distributions), reduced to
    its main directions (singular value decomposition) at a few resolutions. The concept model finds the
    concepts of these rows and their levels, as for any table. The ``rows`` most frequent tokens make the
    table; every other token seen at least ``least`` times is read by the model (its most likely concept);
    a token never seen enough has a concept of its own. Returns, for every resolution, the maps from token
    to concept, from the most general level to the most particular.
    """
    import pandas as pd
    from acan.core.concepts import ConceptModel
    ids = np.asarray(ids, np.int64)
    V = int(vocabulary)
    frequency = np.bincount(ids, minlength=V)
    common = np.argsort(-frequency, kind='stable')[:top]
    where = np.full(V, top, np.int64); where[common] = np.arange(len(common))
    before = np.bincount(ids[1:]*(top+1)+where[ids[:-1]], minlength=V*(top+1)).reshape(V, top+1).astype(float)
    after = np.bincount(ids[:-1]*(top+1)+where[ids[1:]], minlength=V*(top+1)).reshape(V, top+1).astype(float)
    profile = np.hstack([np.sqrt(before/np.maximum(before.sum(1, keepdims=True), 1)),
                         np.sqrt(after/np.maximum(after.sum(1, keepdims=True), 1))])
    seen = np.flatnonzero(frequency >= least)
    table = seen[np.argsort(-frequency[seen], kind='stable')[:rows]]
    center = profile[table].mean(0)
    _, _, directions = np.linalg.svd(profile[table]-center, full_matrices=False)
    out = []
    for d in resolutions:
        d = min(d, len(directions))
        project = lambda which: pd.DataFrame((profile[which]-center)@directions[:d].T, columns=[f'd{i}' for i in range(d)])
        model = ConceptModel(seed=seed, covariance=False).fit(project(table))
        member = model.assignments(project(seen)).argmax(1)
        chain = [member]
        for level in reversed(model.levels[1:]):
            chain.append(level['parents'].argmax(0)[chain[-1]])
        maps = []
        for concept in reversed(chain):
            full = np.full(V, 0, np.int64)
            _, full[seen] = np.unique(concept, return_inverse=True)
            full[np.setdiff1d(np.arange(V), seen)] = full[seen].max()+1 if len(seen) else 0
            maps.append(full)
        out.append(maps)
    return out


def document_counts(keys, valid, ids):
    """For every position: C, N(own token), D of its context among the EARLIER positions of the same text only."""
    n = len(ids)
    position = np.arange(n)

    def earlier(group):
        order = np.lexsort((position, group))
        g = group[order]
        start = np.r_[True, g[1:] != g[:-1]]
        first = np.flatnonzero(start)[np.cumsum(start)-1]
        out = np.empty(n, np.int64)
        out[order] = np.arange(n)-first
        return out
    with np.errstate(over='ignore'):
        pair = keys*np.uint64(1000003)+(ids+1).astype(np.uint64)  # (context, token) as one key
    C = earlier(keys)
    N = earlier(pair)
    novel = (N == 0).astype(np.int64)
    order = np.lexsort((position, keys))
    g, f = keys[order], novel[order]
    start = np.r_[True, g[1:] != g[:-1]]
    running = np.cumsum(f)
    before = np.r_[0, running][np.flatnonzero(start)][np.cumsum(start)-1]
    D = np.empty(n, np.int64)
    D[order] = running-f-before
    out = np.stack((C, N, D)).astype(float)
    out[:, ~valid] = 0
    return out


def memory_limit():
    """Bytes the process may use: the memory of the machine, or the limit of its container if lower."""
    import os
    try:
        total = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        total = 4 << 30
    for name in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            value = Path(name).read_text().strip()
        except OSError:
            continue
        if value.isdigit():
            total = min(total, int(value))
    return total


def default_budget():
    """Bytes for the statistics: a quarter of the memory the process may use, at most 2 GB.

    Adding two tables holds both, their sum and the sorting of their keys: measured, the process peaks at
    2.5 to 3 times the statistics (2.3 GB with 794 MB, 5.1 GB with 2 GB), plus what it holds anyway, so a
    quarter stays inside the limit.
    """
    return int(min(memory_limit()//4, 2 << 30))


def save(directory, model, state, identity):
    """A checkpoint in the format every project uses (manifest, sums, state), so pause, resume, best and
    milestones work as for any other project. The sums go in ``trainables.safetensors``; there is no optimizer."""
    import os
    import shutil
    import uuid
    from safetensors.numpy import save_file
    from . import checkpoints
    from .store import atomic_json, digest_file, sync_directory
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['step']:012d}-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    temporary, final = root/('.'+name+'.tmp'), root/name
    temporary.mkdir()
    try:
        save_file({key: np.ascontiguousarray(value) for key, value in model.arrays().items()}, str(temporary/'trainables.safetensors'))
        (temporary/'optimizer.pt').write_bytes(b'concept-lm: no optimizer\n')
        atomic_json(temporary/'state.json', state)
        (temporary/'rng.pt').write_bytes(b'concept-lm: no random state\n')
        files = ['trainables.safetensors', 'optimizer.pt', 'state.json', 'rng.pt']
        for filename in files:
            with (temporary/filename).open('rb') as stream:
                os.fsync(stream.fileno())
        manifest = dict(format=1, contract=checkpoints.contract(), identity=identity, step=state['step'],
                        sha256={filename: digest_file(temporary/filename) for filename in files})
        atomic_json(temporary/'manifest.json', manifest)
        sync_directory(temporary)
        os.replace(temporary, final)
        sync_directory(root)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load(directory, expected=None, budget_bytes=None):
    """The model and the state of a checkpoint (verified as every checkpoint)."""
    from safetensors.numpy import load_file
    from . import checkpoints
    checkpoints.verify(directory, expected)
    arrays = load_file(str(Path(directory)/'trainables.safetensors'))
    state = json.loads((Path(directory)/'state.json').read_text())
    return from_checkpoint_arrays(arrays, budget_bytes), state


def from_checkpoint_arrays(arrays, budget_bytes=None):
    if 'text_graph' in arrays:
        from .text_graph import TextGraph
        return TextGraph.from_arrays(arrays)
    return ConceptLM.from_arrays(arrays, budget_bytes)
