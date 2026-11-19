"""Raffinamenti per CP su profili numerici e lettura di un target categorico per regioni."""

import numpy as np

from .category_pairs import PairColumns

from acan.concept_algebra import KAPPA_GRID, categorical_reading


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

    def subconcepts(self, build, judgment, complete):
        """Discover supported refinements inside each observed categorical parent of a tabular target."""
        from .category_pairs import CategoryPairs, category_rows, judged_subconcepts
        from .category_table import Table

        accepted, report = judged_subconcepts(build, judgment, self.vocabulary)
        parents, children, target = category_rows(complete, self.vocabulary)
        keys = CategoryPairs.keys(parents, children)
        table = Table.count_regions([(keys, target)], self.vocabulary)
        return table.keep(np.isin(table.keys, accepted, assume_unique=True)), report

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
            if isinstance(feature, PairColumns):
                values = feature.keys(frame)
            elif feature in frame:
                values = frame[feature].fillna(0).to_numpy(dtype=np.uint64)
            else:
                continue
            indices = table.find(values)
            for index in np.unique(indices[indices >= 0]):
                rows = np.flatnonzero(indices == index)
                targets, counts = table.followers(index)
                weight = kappa * max(table.D[index], 1)
                denominator = table.C[index] + weight
                p[rows] *= weight / denominator
                p[np.ix_(rows, targets)] += counts[None, :] / denominator
        return p


def calibrated_weights(stats, initial, mass_prior):
    """Choose categorical smoothing strengths jointly on held-out target probabilities."""
    weights = list(initial)
    for _ in range(4):
        # Each more specific reading is affine in the preceding probability.
        after = []
        slope, intercept = np.ones(len(mass_prior)), np.zeros(len(mass_prior))
        for (m, n, d), k in reversed(list(zip(stats, weights))):
            after.append((slope, intercept))
            if np.isfinite(k):
                w = k * np.maximum(d, 1)
                intercept = intercept + slope * n / (m + w)
                slope = slope * w / (m + w)
        prior = mass_prior
        changed = False
        for i, ((m, n, d), (a, b)) in enumerate(zip(stats, reversed(after))):
            winner, best, result = np.inf, -np.log(a * prior + b).mean(), prior
            for k in KAPPA_GRID:
                candidate = categorical_reading(n, m, d, prior, k, np)
                loss = -np.log(a * candidate + b).mean()
                if loss < best - 1e-12:
                    winner, best, result = k, loss, candidate
            changed |= weights[i] != winner
            weights[i], prior = winner, result
        if not changed:
            break
    return weights
