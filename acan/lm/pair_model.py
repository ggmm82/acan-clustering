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


QUESTION_SCALES = (8, 32, 128)                                  # concepts of the question read by the answer
QUESTION_EXACT = (2, 4)                                         # exact last tokens of the question


def question_columns(row, turns, scales):
    """Columns that carry the last completed user turn to every later position of its document.

    A chat is written ``<|im_start|>user ... <|im_end|> <|im_start|>assistant ...``. At the token that
    closes a user turn, the spans ending at its last word describe the question: its concepts at the
    given scales (``q{w}``) and its exact last tokens (``qe{w}``). From there they stay, unchanged, until
    the next user turn closes or the document ends, so the row that predicts any token of the answer sees
    the question however far back it is. Value 0: no question yet.
    """
    if turns is None:
        return
    start, user, end = turns
    y = row.y
    n = len(y)
    # The turn each <|im_end|> closes: the latest <|im_start|> before it, if its role is the user.
    opens = np.flatnonzero(y == start)
    closes = np.flatnonzero(y == end)
    before = np.searchsorted(opens, closes) - 1
    ok = before >= 0
    closes, opener = closes[ok], opens[before[ok]]
    doc = np.cumsum(row.age == 0)
    user_turn = (opener + 1 < closes) & (y[np.minimum(opener + 1, n - 1)] == user) & (doc[opener] == doc[closes])
    closes = closes[user_turn & (closes >= 1)]
    last = np.full(n, -1)
    last[closes] = closes - 1                                   # the last token of the question
    last = np.maximum.accumulate(last)
    # Not across documents: a question belongs to the document it was asked in.
    first = np.flatnonzero(row.age == 0)
    begin = first[np.searchsorted(first, np.arange(n), side="right") - 1]
    last[last < begin] = -1
    have = last >= 0
    for name, source in [*((f"q{w}", f"c{w}") for w in scales), *((f"qe{w}", f"e{w}") for w in QUESTION_EXACT)]:
        if source not in row.spans:
            continue
        column = np.zeros(n, np.uint32)
        column[have] = row.spans[source][last[have]]
        row.spans[name] = column


def is_question(name):
    return name.startswith("qe") or (name.startswith("q") and name[1:].isdigit())


def question_sources(row):
    """(question column, the span it copies) for the columns this row can have."""
    pairs = [*((f"q{w}", f"c{w}") for w in QUESTION_SCALES), *((f"qe{w}", f"e{w}") for w in QUESTION_EXACT)]
    return [(name, source) for name, source in pairs if source in row.spans]


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
        if "+" in f:                                            # two columns read at the same position
            first, second = (shift(self.spans[name], 1, self.age) for name in f.split("+"))
            return CategoryPairs.keys(first, second)
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
        self.turns = None                                       # (<|im_start|>, user, <|im_end|>) ids, if a chat

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
        question_columns(row, getattr(self, "turns", None), [w for w in QUESTION_SCALES if w in self.nodes])
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
        # The last completed user turn, carried as the bulk columns carry it (question_columns).
        self.question = {}
        self.role, self.asking, self.words = False, False, 0

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
            for name, _ in question_sources(row):              # the question the prompt has asked so far
                if name in row.spans and row.spans[name][-1]:
                    self.question[name] = int(row.spans[name][-1])
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
        if getattr(self.model.encoder, "turns", None) is not None:
            for token in ids[-4096:]:                          # where the prompt ends: inside which turn
                self._follow(int(token))
        return self

    def _follow(self, token):
        start, user, end = self.model.encoder.turns
        if token == start:
            self.role, self.asking, self.words = True, False, 0
        elif self.role:
            self.role, self.asking, self.words = False, token == user, 0
        elif token == end:
            self.asking = False
        elif self.asking:
            self.words += 1

    def fork(self):
        other = copy(self)
        other.overlay = {f: {k: c.copy() for k, c in vals.items()} for f, vals in self.overlay.items()}
        other.question = dict(self.question)
        return other

    def _turn(self, token):
        """Follow the chat turns; when a user turn closes, keep the columns of its last word."""
        turns = getattr(self.model.encoder, "turns", None)
        if turns is None:
            return
        start, user, end = turns
        if token == start:
            self.role, self.asking, self.words = True, False, 0
        elif self.role:
            self.role, self.asking, self.words = False, token == user, 0
        elif token == end:
            if self.asking and self.words and len(self.tail):
                row = self.model.encoder.encode(self.tail)    # the tail ends with the last word of the question
                self.question = {name: int(row.spans[source][-1]) for name, source in question_sources(row)}
            self.asking = False
        elif self.asking:
            self.words += 1

    def _keys(self):
        if self.next_keys is None:
            row = self.model.encoder.encode(np.r_[self.tail, np.int32(0)])
            for name in list(row.spans):                         # the question comes from the state, not the tail
                if is_question(name):
                    row.spans[name][-2 if len(row.y) > 1 else -1:] = self.question.get(name, 0)
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
        self._turn(int(token))
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
