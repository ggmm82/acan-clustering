"""The concept language model: concepts of the training text by context, and concepts of the text being read.

Global level j: the training positions whose last j symbols are equal. Sums C, N(y), D (distinct
followers). Prior of a concept: its level above (context one shorter), weight a_j * D (Witten-Bell
from the sums), a_j chosen by the judgment (positions read by sums that do not contain them).
Document level j: the same concepts built only from the earlier positions of the text being read,
with the global prediction as their prior: the model can copy what this text said before.
Pruning: a concept with fewer than ``least`` members cannot support itself and is left to its parent.
"""
import numpy as np

def contexts(ids, j, V):
    key = np.zeros(len(ids), np.int64)
    for k in range(1, j+1):
        shifted = np.full(len(ids), -1, np.int64); shifted[k:] = ids[:-k]
        key = np.where((key < 0) | (shifted < 0), -1, key*V+shifted)
    return key

class ConceptLM:
    def __init__(self, V, depth=8, doc_depth=4, least=1):
        self.V, self.depth, self.doc_depth, self.least = V, depth, doc_depth, least
        self.a = np.ones(depth+1); self.b = np.ones(doc_depth+1)

    def fit(self, ids, judged=50000, seed=0):
        V = self.V
        self.levels = []
        for j in range(self.depth+1):
            key = contexts(ids, j, V); ok = key >= 0
            concepts, which, C = np.unique(key[ok], return_inverse=True, return_counts=True)
            keep = C >= self.least if j else np.ones(len(C), bool)
            cells, counts = np.unique(which*V+ids[ok], return_counts=True)
            D = np.bincount(cells//V, minlength=len(concepts))
            # pruned concepts are dropped with their cells: the reader falls back to the level above
            alive = keep[cells//V]
            remap = np.cumsum(keep)-1
            self.levels.append(dict(keys=concepts[keep], C=C[keep], D=D[keep], cells=remap[cells[alive]//V]*V+cells[alive] % V, counts=counts[alive]))
        rng = np.random.default_rng(seed)
        self.judged = rng.choice(np.arange(self.depth, len(ids)), min(judged, len(ids)-self.depth), replace=False)
        self.train_ids = ids
        grid = tuple(2.**k for k in range(-5, 9))
        for j in list(range(self.depth+1))*2:
            scores = []
            for value in grid:
                trial = self.a.copy(); trial[j] = value
                scores.append(self._global(ids, self.judged, trial, leave_out=True, upto=j).mean())
            self.a[j] = grid[int(np.argmax(scores))]
        return self

    def concepts(self):
        return int(sum(len(level['keys']) for level in self.levels))

    def _lookup(self, level, key, y):
        V = self.V
        if not len(level['keys']):
            z = np.zeros(len(key)); return z, z, z
        place = np.minimum(np.searchsorted(level['keys'], key), len(level['keys'])-1)
        seen = (level['keys'][place] == key) & (key >= 0)
        cell = place*V+y
        at = np.minimum(np.searchsorted(level['cells'], cell), max(len(level['cells'])-1, 0))
        N = np.where(seen & (level['cells'][at] == cell), level['counts'][at], 0) if len(level['cells']) else np.zeros(len(key))
        return np.where(seen, level['C'][place], 0.), N.astype(float), np.where(seen, level['D'][place], 0.)

    def _global(self, ids, positions, a, leave_out=False, upto=None):
        p = np.full(len(positions), 1./self.V)
        for j, level in enumerate(self.levels[:None if upto is None else upto+1]):
            C, N, D = self._lookup(level, contexts(ids, j, self.V)[positions], ids[positions])
            if leave_out:
                D = D-(N == 1); C, N = np.maximum(C-1, 0), np.maximum(N-1, 0)
            weight = a[j]*np.maximum(D, 1)
            p = (N+weight*p)/(C+weight)
        return np.log(p) if upto is not None else p

    def _document(self, ids, j):
        """For every position: C, N(own symbol), D of its context of length j among the EARLIER positions only."""
        V, n = self.V, len(ids)
        key = contexts(ids, j, V)
        position = np.arange(n)

        def earlier(group):
            """How many earlier positions share the group of each position (groups as integers)."""
            order = np.lexsort((position, group))
            g = group[order]
            start = np.r_[True, g[1:] != g[:-1]]
            first = np.flatnonzero(start)[np.cumsum(start)-1]
            out = np.empty(n, np.int64)
            out[order] = np.arange(n)-first
            return out
        C = earlier(key)
        pair = key*V+ids
        N = earlier(pair)
        # distinct earlier followers: earlier positions that are the first of their (context, symbol)
        novel = (N == 0).astype(np.int64)
        order = np.lexsort((position, key))
        g, f = key[order], novel[order]
        start = np.r_[True, g[1:] != g[:-1]]
        running = np.cumsum(f)
        before_group = np.r_[0, running][np.flatnonzero(start)][np.cumsum(start)-1]
        D = np.empty(n, np.int64)
        D[order] = running-f-before_group
        out = np.stack((C, N, D)).astype(float)
        out[:, key < 0] = 0
        return out

    def logp(self, ids, positions=None, b=None):
        """log p of every symbol of a text, read causally: global concepts, then the text's own concepts."""
        positions = np.arange(len(ids)) if positions is None else positions
        p = self._global(ids, positions, self.a)
        b = self.b if b is None else b
        for j in range(1, self.doc_depth+1):
            C, N, D = self._document(ids, j)[:, positions]
            weight = b[j]*np.maximum(D, 1)
            p = (N+weight*p)/(C+weight)
        return np.log(p)

    def fit_document(self, ids, positions):
        """The weights of the text's own concepts, by judgment on a held-out stretch of training text."""
        grid = tuple(2.**k for k in range(-5, 9))
        for j in list(range(1, self.doc_depth+1))*2:
            scores = []
            for value in grid:
                trial = self.b.copy(); trial[j] = value
                scores.append(self.logp(ids, positions, trial).mean())
            self.b[j] = grid[int(np.argmax(scores))]
        return self
