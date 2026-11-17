"""Lettura categorica dei livelli esistenti, senza una seconda ricerca di concetti."""
import numpy as np
from scipy import sparse
from acan.concept_algebra import categorical_reading


class LevelReading:
    """Somme per concetto; il livello generale dà il prior al particolare.

    `parents[i]` maps level i+1 to level i (each column sums to one).
    Counts are propagated from leaf memberships, so every row has total mass one
    at every level. The root has a uniform proper prior for unseen categories.
    """
    def __init__(self, counts, parents, kappa=1.):
        self.kappa = float(kappa)
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError('Il peso del prior deve essere positivo e finito.')
        self.parents = [np.array(p, dtype=float, copy=True) for p in parents]
        self.counts = [sparse.csr_matrix(counts, dtype=float, copy=True)]
        self.counts[0].eliminate_zeros()
        if min(self.counts[0].shape) < 1 or np.any(self.counts[0].data < 0) or not np.isfinite(self.counts[0].data).all():
            raise ValueError('Conteggi non validi.')
        children = self.counts[0].shape[0]
        for parent in reversed(self.parents):
            if parent.ndim != 2 or parent.shape[1] != children or parent.shape[0] < 1:
                raise ValueError('Dimensioni dei livelli incoerenti.')
            if not np.isfinite(parent).all() or np.any(parent < 0) or not np.allclose(parent.sum(0), 1.):
                raise ValueError('Ogni concetto deve distribuire massa uno tra i genitori.')
            children = parent.shape[0]
        for parent in reversed(self.parents):
            self.counts.insert(0, sparse.csr_matrix(parent) @ self.counts[0])
        self.columns = [c.tocsc() for c in self.counts]  # querying one token must not scan every sparse pair
        self.mass = [c.sum(1).A1 for c in self.counts]
        self.distinct = [np.diff(c.indptr) for c in self.counts]
        self.root = self.counts[0].sum(0).A1
        self.vocabulary = len(self.root)

    def probability(self, tokens):
        tokens = np.asarray(tokens, dtype=int)
        prior = categorical_reading(self.root[tokens], self.root.sum(),
            np.count_nonzero(self.root), np.full(len(tokens), 1/self.vocabulary), self.kappa, np)[None]
        for i, counts in enumerate(self.counts):
            if i:
                prior = self.parents[i-1].T @ prior
            prior = categorical_reading(self.columns[i][:, tokens].toarray(), self.mass[i][:, None],
                self.distinct[i][:, None], prior, self.kappa, np)
        return prior

    def document(self):
        return DocumentReading(self)


class DocumentReading:
    """The revealed prefix, accumulated in the same learned levels.

    At every level, the global counts read with the document's parent as prior;
    the document counts then read with that global prediction as prior. Both
    steps use categorical_reading. No new concepts or positional keys are made.
    """
    def __init__(self, reading):
        self.reading = reading
        self.level_counts = [{} for c in reading.counts]
        self.level_mass = [np.zeros(c.shape[0]) for c in reading.counts]
        self.level_distinct = [np.zeros(c.shape[0]) for c in reading.counts]
        self.counts = self.level_counts[-1]
        self.mass = self.level_mass[-1]
        self.distinct = self.level_distinct[-1]
        self.observations = 0

    def probability(self, membership, tokens):
        tokens = np.asarray(tokens, dtype=int)
        r = self.reading
        leaf = np.stack([self.counts.get(int(t), np.zeros_like(self.mass)) for t in tokens], axis=1)
        root = categorical_reading(r.root[tokens], r.root.sum(), np.count_nonzero(r.root),
            np.full(len(tokens), 1/r.vocabulary), r.kappa, np)
        prior = categorical_reading(leaf.sum(0), self.observations, len(self.counts), root, r.kappa, np)[None]
        for i, counts in enumerate(r.counts):
            if i:
                prior = r.parents[i-1].T @ prior
            global_head = categorical_reading(r.columns[i][:, tokens].toarray(), r.mass[i][:, None],
                r.distinct[i][:, None], prior, r.kappa, np)
            doc = np.stack([self.level_counts[i].get(int(t), np.zeros_like(self.level_mass[i])) for t in tokens], axis=1)
            prior = categorical_reading(doc, self.level_mass[i][:, None],
                self.level_distinct[i][:, None], global_head, r.kappa, np)
        return np.asarray(membership) @ prior

    def observe(self, membership, token):
        token = int(token)
        if not 0 <= token < self.reading.vocabulary:
            raise ValueError('Token fuori dal vocabolario.')
        membership = np.asarray(membership, dtype=float)
        if membership.shape != self.mass.shape or not np.isfinite(membership).all() or np.any(membership < 0) or not np.isclose(membership.sum(), 1.):
            raise ValueError('Appartenenze non valide.')
        memberships = [membership]
        for parent in reversed(self.reading.parents):
            memberships.insert(0, parent @ memberships[0])
        for counts, mass, distinct, weights in zip(self.level_counts,self.level_mass,self.level_distinct,memberships):
            previous = counts.setdefault(token,np.zeros_like(mass))
            distinct += (previous == 0) & (weights > 0)
            previous += weights
            mass += weights
        self.observations += 1


class TextCursor:
    """One running prefix: a short ordered window and the learned concepts' sums."""
    def __init__(self, table, reading):
        from collections import deque
        self.table = table
        self.reading = reading
        self.state = reading.document()
        self.window = deque(maxlen=table.rows.context)
        self._membership = None

    def membership(self):
        if self._membership is None:
            frame = self.table.rows.frame(np.asarray(self.window, dtype=int), [len(self.window)])
            self._membership = self.table.model.assignments(frame)[0]
        return self._membership

    def distribution(self):
        return self.probability(np.arange(self.reading.vocabulary))

    def probability(self, tokens):
        tokens = np.asarray(tokens, dtype=int)
        return self.state.probability(self.membership(), tokens)

    def observe(self, token):
        self.state.observe(self.membership(), token)
        self.window.append(int(token))
        self._membership = None
        return self
