"""Sparse additive masses of categorical attributes in observed concepts."""

import numpy as np


def _union(a, b):
    """The values of two sorted arrays without repeats, sorted."""
    both = np.concatenate((a, b))
    both.sort(kind='stable')                                    # two sorted runs: a merge
    keep = np.empty(len(both), bool)
    keep[0] = True
    np.not_equal(both[1:], both[:-1], out=keep[1:])
    return both[keep]


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
    def count_regions(cls, chunks, vocabulary):
        """Merge counts from disjoint row batches. Region zero is a missing observation."""
        partial = []
        for keys, targets in chunks:
            seen = keys != 0
            table = cls.count(keys[seen], targets[seen], vocabulary)
            level = 0
            while level < len(partial) and partial[level] is not None:
                table = partial[level] + table
                partial[level] = None
                level += 1
            if level == len(partial):
                partial.append(table)
            else:
                partial[level] = table
        total = cls.empty(vocabulary)
        for table in partial:
            if table is not None:
                total = total + table
        return total

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
        return self.lookup_indices(self.find(keys), targets)

    def lookup_indices(self, index, targets):
        seen = index >= 0
        safe = np.where(seen, index, 0)
        cell = safe*(self.V+1)+targets
        N = np.zeros(len(index))
        if len(self.cells):
            at = np.minimum(np.searchsorted(self.cells, cell), len(self.cells)-1)
            N = np.where(seen & (self.cells[at] == cell), self.counts[at], 0).astype(float)
        if not len(self.keys):
            z = np.zeros(len(index)); return z, z, z
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

def compact(values):
    return values.astype(np.min_scalar_type(int(values.max(initial=0))), copy=False)


class SharedProfiles:
    """One categorical prototype per identical single-outcome histogram within a level.

    Contexts keep distinct keys. If y and its mass coincide, the complete readout is the same
    for every parent distribution. Mixed histograms are preserved without approximation.
    """

    def __init__(self, table):
        self.keys = compact(table.keys)
        self.V = table.V
        pure = table.D == 1
        first = np.cumsum(table.D) - table.D
        targets = table.cells[first[pure]] % (self.V+1)
        groups, inverse = np.unique(np.column_stack((table.C[pure],targets)),axis=0,return_inverse=True)
        owner = table.cells//(self.V+1)
        mixed = ~pure[owner]
        mapping = np.empty(len(pure),np.int64)
        mapping[pure] = inverse
        mapping[~pure] = np.arange(np.count_nonzero(~pure))+len(groups)
        self.mapping = compact(mapping)
        indices = np.r_[np.arange(len(groups))*(self.V+1)+groups[:,1],
                        mapping[owner[mixed]]*(self.V+1)+table.cells[mixed]%(self.V+1)]
        mass = np.r_[groups[:,0],table.C[~pure]]
        counts = np.r_[groups[:,0],table.counts[mixed]]
        self.profiles = Table(np.arange(len(mass)),compact(mass),indices,compact(counts),self.V)
        self.profiles.keys = compact(self.profiles.keys)
        self.profiles.D = compact(self.profiles.D)
        self.C,self.D = self.profiles.C,self.profiles.D

    def find(self, keys):
        if not len(self.keys):
            return np.full(len(keys),-1,dtype=np.int64)
        at = np.minimum(np.searchsorted(self.keys,keys),len(self.keys)-1)
        return np.where(self.keys[at]==keys,self.mapping[at].astype(np.int64),-1)

    def lookup(self, keys, targets):
        return self.profiles.lookup_indices(self.find(keys),targets)

    def followers(self, index):
        return self.profiles.followers(index)

    def nbytes(self):
        return self.keys.nbytes+self.mapping.nbytes+self.profiles.nbytes()
