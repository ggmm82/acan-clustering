"""Raffinamenti per CP su profili numerici e lettura di un target categorico per regioni."""

import numpy as np

from acan.concept_algebra import KAPPA_GRID, categorical_reading, pair_mass, unbiased


class SemanticCuts:
    """Nested cuts of numeric profiles. A split needs positive pair CP against its own parent."""

    def fit(self, vectors, masses, depth=16):
        vectors = np.asarray(vectors, dtype=float)
        masses = np.asarray(masses, dtype=float)
        if vectors.ndim != 2 or masses.shape != (len(vectors),) or not np.any(masses > 0):
            raise ValueError("Servono profili e masse di training.")
        self.depth = depth
        # A uint32 stores a prefix with a leading sentinel bit, including the missing code 0.
        if not 1 <= depth <= 30:
            raise ValueError("Profondità CP non rappresentabile.")
        self.rules = {}
        active = {1: np.flatnonzero(masses > 0)}
        for _ in range(depth):
            children = {}
            for node, ix in active.items():
                if len(ix) < 2:
                    continue
                x, w = vectors[ix], masses[ix]
                mean = np.average(x, axis=0, weights=w)
                centered = x - mean
                variance = np.average(centered**2, axis=0, weights=w)
                varying = variance > 1e-15
                if not varying.any():
                    continue
                # Propose a bisection along the two widest coordinates; judge all coordinates.
                axes = np.argsort(-variance, kind="stable")[:2]
                parent = unbiased(variance[varying], w.sum(), np)
                normalized = centered[:, varying] / np.sqrt(parent)
                best = 0.0
                for axis in axes:
                    if not varying[axis]:
                        continue
                    right = x[:, axis] > mean[axis]
                    score = 0.0
                    for group in (right, ~right):
                        ww = w[group]
                        total = ww.sum()
                        if total <= 1:
                            continue
                        xx = normalized[group]
                        mu = np.average(xx, axis=0, weights=ww)
                        var = np.average((xx - mu) ** 2, axis=0, weights=ww)
                        score += (
                            pair_mass(total, np) * (1 - np.minimum(2.0, unbiased(var, total, np))).sum()
                        )
                    if score > best:
                        best, split = score, (axis, mean[axis], right)
                if best > 0:
                    axis, threshold, right = split
                    self.rules[node] = (axis, threshold)
                    for bit, subset in enumerate((ix[~right], ix[right])):
                        children[2 * node + bit] = subset
            active = children
        self.axis = np.zeros(2**depth, dtype=np.int32)
        self.threshold = np.full(2**depth, np.inf)
        for node, (axis, threshold) in self.rules.items():
            self.axis[node], self.threshold[node] = axis, threshold
        del self.rules
        return self

    def encode(self, vectors):
        """All points, including unseen ones, traverse numeric thresholds. Leaf codes are padded."""
        vectors = np.asarray(vectors)
        node = np.ones(len(vectors), np.uint32)
        rows = np.arange(len(node))
        for _ in range(self.depth):
            node = 2 * node + (vectors[rows, self.axis[node]] > self.threshold[node])
        return node


class ConditionalReading:
    """Read observed hard memberships with the shared categorical operator and a judged prior."""

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.levels = []

    def propose(self, table, query_keys, query_y, prior):
        """Judge complete training sums against the preceding readout on unseen rows."""
        mass, counts, distinct = table.lookup(query_keys, query_y)
        best = float(-np.log(prior).mean())
        choice = None
        for kappa in KAPPA_GRID:
            p = categorical_reading(counts, mass, distinct, prior, kappa, np)
            loss = float(-np.log(p).mean())
            if loss < best:
                best, choice = loss, (table, kappa, p)
        return best, choice

    def probability(self, keys, targets, omit=(), prior=None):
        p = self.prior[targets].copy() if prior is None else np.array(prior, copy=True)
        for feature, table, kappa in self.levels:
            if feature in omit:
                continue
            m, n, d = table.lookup(keys(feature), targets)
            p = categorical_reading(n, m, d, p, kappa, np)
        return p

    def distribution(self, key, prior=None):
        p = self.prior.copy() if prior is None else np.array(prior, dtype=float, copy=True)
        for feature, table, kappa in self.levels:
            index = table.find([key(feature)])[0]
            if index >= 0:
                targets, counts = table.followers(index)
                weight = kappa * max(table.D[index], 1)
                p *= weight / (table.C[index] + weight)
                p[targets] += counts / (table.C[index] + weight)
        return p

    def apply(self, frame, prior):
        """Native predict_proba on a matrix of rows, including rows with unavailable regions."""
        p = prior.copy()
        for feature, table, kappa in self.levels:
            if feature not in frame:
                continue
            values = frame[feature].fillna(0).to_numpy(dtype=np.uint64)
            indices = table.find(values)
            for index in np.unique(indices[indices >= 0]):
                rows = np.flatnonzero(indices == index)
                targets, counts = table.followers(index)
                weight = kappa * max(table.D[index], 1)
                denominator = table.C[index] + weight
                p[rows] *= weight / denominator
                p[np.ix_(rows, targets)] += counts[None, :] / denominator
        return p
