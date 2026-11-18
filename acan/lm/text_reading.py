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

    def prefix_statistics(self, membership, tokens):
        """Batch sufficient statistics, written before each revealed token is added."""
        tokens = np.asarray(tokens, dtype=int)
        membership = np.asarray(membership, dtype=float)
        if tokens.ndim != 1 or np.any(tokens < 0) or np.any(tokens >= self.vocabulary):
            raise ValueError('Token fuori dal vocabolario.')
        if membership.shape != (len(tokens),len(self.mass[-1])) or not np.isfinite(membership).all() or np.any(membership < 0) or not np.allclose(membership.sum(1),1.):
            raise ValueError('Appartenenze non valide.')
        levels = [membership]
        for parent in reversed(self.parents):
            levels.insert(0, levels[0] @ parent.T)
        statistics = []
        root_distinct = np.zeros(len(tokens))
        for level, weights in enumerate(levels):
            counts = np.zeros_like(weights)
            distinct = np.zeros_like(weights)
            mass = np.zeros_like(weights)
            seen = {}
            live = np.zeros(weights.shape[1])
            total = np.zeros_like(live)
            for pos, token in enumerate(tokens):
                previous = seen.get(int(token))
                if previous is None:
                    previous = np.zeros_like(live)
                counts[pos], distinct[pos], mass[pos] = previous, live, total
                if level == len(levels)-1:
                    root_distinct[pos] = len(seen)
                live += (previous == 0) & (weights[pos] > 0)
                total += weights[pos]
                seen[int(token)] = previous + weights[pos]
            statistics.append((counts.T,mass.T,distinct.T))
        return statistics, root_distinct

    def sequence_probability(self, membership, tokens, prefix=None):
        """Exactly the sequential document reading over a fixed batch; a cached prefix must describe these same inputs."""
        tokens = np.asarray(tokens, dtype=int)
        statistics, root_distinct = self.prefix_statistics(membership,tokens) if prefix is None else prefix
        root = categorical_reading(self.root[tokens],self.root.sum(),np.count_nonzero(self.root),
            np.full(len(tokens),1/self.vocabulary),self.kappa,np)
        prior = categorical_reading(statistics[-1][0].sum(0),np.arange(len(tokens)),root_distinct,root,self.kappa,np)[None]
        for i, (counts,mass,distinct) in enumerate(statistics):
            if i:
                prior = self.parents[i-1].T @ prior
            global_head = categorical_reading(self.columns[i][:,tokens].toarray(),self.mass[i][:,None],
                self.distinct[i][:,None],prior,self.kappa,np)
            prior = categorical_reading(counts,mass,distinct,global_head,self.kappa,np)
        return (np.asarray(membership)*prior.T).sum(1)

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
        full = len(tokens) == r.vocabulary and np.array_equal(tokens, np.arange(r.vocabulary))
        def columns(level):
            values = self.level_counts[level]
            size = len(self.level_mass[level])
            if full:
                result = np.zeros((size, len(tokens)))
                for token, weights in values.items():
                    result[:, token] = weights
                return result
            zero = np.zeros(size)
            return np.stack([values.get(int(t), zero) for t in tokens], axis=1)
        leaf = columns(-1)
        root = categorical_reading(r.root[tokens], r.root.sum(), np.count_nonzero(r.root),
            np.full(len(tokens), 1/r.vocabulary), r.kappa, np)
        prior = categorical_reading(leaf.sum(0), self.observations, len(self.counts), root, r.kappa, np)[None]
        for i, counts in enumerate(r.counts):
            if i:
                prior = r.parents[i-1].T @ prior
            global_head = categorical_reading(r.columns[i][:, tokens].toarray(), r.mass[i][:, None],
                r.distinct[i][:, None], prior, r.kappa, np)
            doc = leaf if i == len(r.counts)-1 else columns(i)
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


    def observe_many(self, memberships, tokens):
        """Accumulate an already revealed block before the next query.

        Recognition uses only local prefix features. Once memberships are known,
        summing by token gives exactly the sufficient statistics of repeated observe.
        """
        tokens = np.asarray(tokens)
        memberships = np.asarray(memberships, dtype=float)
        if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError('Il blocco deve contenere identificatori interi.')
        if np.any(tokens < 0) or np.any(tokens >= self.reading.vocabulary):
            raise ValueError('Token fuori dal vocabolario.')
        if (memberships.shape != (len(tokens), len(self.mass)) or
                not np.isfinite(memberships).all() or np.any(memberships < 0) or
                not np.allclose(memberships.sum(axis=1), 1.)):
            raise ValueError('Appartenenze non valide.')
        unique, group = np.unique(tokens, return_inverse=True)
        counts = np.zeros((len(unique), len(self.mass)))
        np.add.at(counts, group, memberships)
        for level in reversed(range(len(self.level_counts))):
            stored = self.level_counts[level]
            self.level_mass[level] += counts.sum(axis=0)
            for token, addition in zip(unique, counts):
                previous = stored.get(int(token))
                if previous is None:
                    self.level_distinct[level] += addition > 0
                    stored[int(token)] = addition.copy()
                else:
                    self.level_distinct[level] += (previous == 0) & (addition > 0)
                    previous += addition
            if level:
                counts = counts @ self.reading.parents[level - 1].T
        self.observations += len(tokens)


class TextCursor:
    """One running prefix: a short ordered window and the learned concepts' sums."""
    def __init__(self, table, reading, observed_context=None):
        from collections import deque
        self.table = table
        self.reading = reading
        self.state = reading.document()
        self.observed_context = observed_context
        self.window = deque(maxlen=table.rows.context if observed_context is None else observed_context)
        self._membership = None

    def membership(self):
        if self._membership is None:
            frame = self.table.rows.frame(np.asarray(self.window, dtype=int), [len(self.window)],
                                          observed_context=self.observed_context)
            self._membership = self.table.model.assignments(frame)[0]
        return self._membership

    def prefill(self, tokens, chunk=512):
        """Read known tokens in bounded batches, including across separate calls."""
        if chunk < 1:
            raise ValueError('La dimensione del blocco deve essere positiva.')
        tokens = np.asarray(tokens)
        if tokens.ndim != 1 or (tokens.size and (not np.issubdtype(tokens.dtype, np.integer)
                or np.any(tokens < 0) or np.any(tokens >= self.reading.vocabulary))):
            raise ValueError('Token fuori dal vocabolario.')
        for start in range(0, len(tokens), chunk):
            block = tokens[start:start+chunk]
            previous = len(self.window)
            ids = np.concatenate((np.asarray(self.window, dtype=int), block))
            frame = self.table.rows.frame(ids, np.arange(previous, len(ids)),
                                          observed_context=self.observed_context)
            self.state.observe_many(self.table.model.assignments(frame), block)
            self.window.extend(block)
            self._membership = None
        return self

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


def posterior_counts(table, reading, train, width, chunk=512):
    """One pass over the original rows with the observed training token."""
    # Store K x V learned counts, independent of training sequence length.
    totals = np.zeros((reading.vocabulary, table.model.concepts))
    head = reading.probability(np.arange(reading.vocabulary))
    stop = table.rows.context + table.model.rows
    if len(train) < stop:
        raise ValueError("Il train non copre le righe del modello.")
    for start in range(table.rows.context, stop, chunk):
        pos = np.arange(start, min(start+chunk, stop))
        weights = table.model.assignments(table.rows.frame(train, pos, observed_context=width))
        targets = train[pos]
        posterior = weights * head[:, targets].T
        posterior /= posterior.sum(axis=1, keepdims=True)
        np.add.at(totals, targets, posterior)
    return sparse.csr_matrix(totals.T)
