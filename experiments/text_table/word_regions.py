"""Una coordinata semantica per le parole composte da più token BPE nel documento."""

from collections import Counter
from copy import copy
import re

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import randomized_svd

from acan.concept_algebra import KAPPA_GRID, categorical_reading
from acan.core.conditional import SemanticCuts

WORD = re.compile(r"[^\W\d_]+")


class WordProfiles:
    """Fit word profiles and CP cuts only on training lines; predict the original BPE target."""

    @classmethod
    def fit(cls, read_lines, tokenizer):
        frequency = Counter()
        for line in read_lines():
            frequency.update(WORD.findall(line.lower()))
        words = [w for w, c in frequency.most_common() if c >= 2]
        index = {w: i for i, w in enumerate(words)}
        size = len(words)
        if size < 2:
            raise ValueError("Servono almeno due parole ricorrenti nel training.")
        neighbors = min(4096, size)
        counts = sparse.csr_matrix((size, 2 * neighbors), dtype=np.float64)

        # Bound the number of word positions in a batch. A separator separates successive lines.
        def accumulate(batch):
            ids = np.asarray(batch, dtype=np.int32)
            lines = np.cumsum(ids == size + 1)
            matrix = sparse.csr_matrix(counts.shape, dtype=np.float64)
            for k in range(1, min(6, len(ids))):
                x, y = ids[:-k], ids[k:]
                same = lines[:-k] == lines[k:]
                for a, b, offset in ((x, y, 0), (y, x, neighbors)):
                    keep = same & (a < size) & (b < neighbors)
                    matrix += sparse.csr_matrix(
                        (np.full(np.count_nonzero(keep), 1 / k), (a[keep], b[keep] + offset)),
                        shape=counts.shape,
                    )
            return matrix

        batch = []
        for line in read_lines():
            batch.extend(index.get(w, size) for w in WORD.findall(line.lower()))
            batch.append(size + 1)
            if len(batch) >= 131072:
                counts += accumulate(batch)
                batch = []
        if batch:
            counts += accumulate(batch)
        counts = counts.tocoo()
        totals = np.bincount(counts.row, weights=counts.data, minlength=size)
        context = np.bincount(counts.col, weights=counts.data, minlength=2 * neighbors) ** 0.75
        counts.data = np.maximum(
            np.log(counts.data * context.sum() / (totals[counts.row] * context[counts.col])), 0.0
        )
        u, s, _ = randomized_svd(
            counts.astype(np.float32).tocsr(), min(64, size, 2 * neighbors), n_iter=4, random_state=42
        )
        vectors = u * np.sqrt(s)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        codes = SemanticCuts().fit(vectors, np.ones(size), 12).encode(vectors)
        return cls(index, vectors, codes, tokenizer)

    def __init__(self, index, vectors, codes, tokenizer):
        self.index, self.vectors, self.codes = index, vectors, codes
        self.pieces = [tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())]
        self.max_length = max(map(len, index))
        self.weights = []


class Cell:
    def __init__(self):
        self.members = {}
        self.counts = Counter()
        self.log_evidence = 0.0

    def clone(self):
        other = copy(self)
        other.members = {w: c.copy() for w, c in self.members.items()}
        other.counts = self.counts.copy()
        return other

    def add(self, w, y, prior):
        if w not in self.members and len(self.members) >= 128:
            del self.members[next(iter(self.members))]
        self.members.setdefault(w, Counter())[y] += 1
        c = self.counts[y]
        self.log_evidence += np.log((c + 1 / len(prior)) / (c + prior[y]))
        self.counts[y] += 1

    def statistics(self, query, vectors, vocabulary, y):
        members = list(self.members)
        scores = vectors[members] @ vectors[query]
        weights = np.exp(16 * (scores - scores.max()))
        mass = distinct = 0.0
        counts = 0.0 if y is not None else np.zeros(vocabulary)
        for z, histogram in zip(weights, self.members.values()):
            mass += z * sum(histogram.values())
            distinct += z * len(histogram)
            if y is None:
                for target, count in histogram.items():
                    counts[target] += z * count
            else:
                counts += z * histogram.get(y, 0)
        # A single landmark gives no evidence for a dependence on the changing word.
        log_ratio = self.log_evidence * (1 - 1 / len(members))
        return mass, counts, distinct, np.exp(-np.clip(log_ratio, -40, 40))


class Banks:
    def __init__(self):
        self.cells = {}
        self.owned = None

    def fork(self):
        other = Banks()
        other.cells = self.cells.copy()
        self.owned = set()
        other.owned = set()
        return other

    def add(self, key, word, target, prior):
        cell = self.cells.get(key)
        if cell is None:
            self.cells[key] = cell = Cell()
        elif self.owned is not None and key not in self.owned:
            self.cells[key] = cell = cell.clone()
        if self.owned is not None:
            self.owned.add(key)
        cell.add(word, target, prior)


class WordDocument:
    def __init__(self, profiles, prior, window, slots=None):
        self.profiles, self.prior, self.window = profiles, prior, window
        self.slots = [s for s, k, b in profiles.weights] if slots is None else slots
        self.done = (-1, -1, -1)
        self.fragment = ""
        self.latest = -1
        self.n = 0
        self.active = Banks()
        self.next_epoch = None

    def fork(self):
        other = copy(self)
        other.active = self.active.fork()
        other.next_epoch = None if self.next_epoch is None else self.next_epoch.fork()
        return other

    def regions(self):
        row = (*self.done, self.latest)
        if -1 in row:
            return
        cuts = tuple(int(self.profiles.codes[i]) for i in row)
        for slot in self.slots:
            yield slot, (slot, *cuts[:slot], *cuts[slot + 1 :]), row[slot]

    def observe(self, y):
        for _, key, w in self.regions():
            self.active.add(key, w, y, self.prior)
            if self.next_epoch is not None:
                self.next_epoch.add(key, w, y, self.prior)
        piece = self.profiles.pieces[y]
        if piece in ("[EOS]", "[BOS]", "[PAD]"):
            self.done = (-1, -1, -1)
            self.fragment = ""
        elif piece.startswith("Ġ"):
            if self.fragment != "":
                self.done = (*self.done[1:], self.latest)
            self.fragment = piece[1:].lower()
        elif self.fragment is not None:
            self.fragment += piece.lower()
        if self.fragment is not None and len(self.fragment) > self.profiles.max_length:
            self.fragment = None
        self.latest = self.profiles.index.get(self.fragment, -1)
        self.n += 1
        if self.n % self.window == 0:
            if self.next_epoch is not None:
                self.active = self.next_epoch
            self.next_epoch = Banks()

    def statistics(self, y):
        for slot, key, word in self.regions():
            cell = self.active.cells.get(key)
            if cell is not None:
                yield slot, cell.statistics(word, self.profiles.vectors, len(self.prior), y)

    def distribution(self, p):
        stats = dict(self.statistics(None))
        for slot, kappa, beta in self.profiles.weights:
            if slot in stats:
                m, n, d, ratio = stats[slot]
                p = categorical_reading(n, m, d, p, 1 + kappa * ratio**beta, np)
        return p


def observed_stats(profiles, prior, rows, slots):
    stats = {s: np.zeros((4, len(rows.tokens))) for s in slots}
    for a, b in zip(rows.starts, np.r_[rows.starts[1:], len(rows.tokens)]):
        doc = WordDocument(profiles, prior, rows.window, slots)
        for i in range(a, b):
            y = int(rows.tokens[i])
            for slot, values in doc.statistics(y):
                stats[slot][:, i] = values
            doc.observe(y)
    return {s: x[:, rows.pos] for s, x in stats.items()}


def judge_words(profiles, prior, rows, p):
    slots = [3, 2, 1, 0]
    stats = observed_stats(profiles, prior, rows, slots)
    profiles.weights = []
    for slot in slots:
        mass, n, d, ratio = stats[slot]
        best = -np.log(p).mean()
        choice = None
        for beta in (0.0, 0.125, 0.25, 0.5, 1.0):
            for k in KAPPA_GRID:
                q = categorical_reading(n, mass, d, p, 1 + k * ratio**beta, np)
                loss = -np.log(q).mean()
                if loss < best:
                    best, choice = loss, (k, beta, q)
        if choice:
            k, beta, p = choice
            profiles.weights.append((slot, k, beta))
    return p


def read_words(profiles, prior, rows, p):
    stats = observed_stats(profiles, prior, rows, [s for s, k, b in profiles.weights])
    for slot, k, beta in profiles.weights:
        m, n, d, ratio = stats[slot]
        p = categorical_reading(n, m, d, p, 1 + k * ratio**beta, np)
    return p
