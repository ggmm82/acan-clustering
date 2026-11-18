"""Somme temporanee delle sole righe del documento già completate, negli stessi concetti CP."""

import numpy as np

from acan.concept_algebra import KAPPA_GRID, categorical_reading
from acan.lm.concept_lm import document_counts
from .conditional import proposals, region


def document_stats(rows, feature):
    at = np.arange(len(rows.tokens))
    key = region(rows.columns, rows.age, at, rows.depths, feature)
    docs = np.searchsorted(rows.starts, at, side="right") - 1
    # Dense local numbering makes the (document, region) map collision-free.
    _, group = np.unique(np.rec.fromarrays([docs, key]), return_inverse=True)
    valid = key != 0
    group = group.astype(np.uint64)
    if rows.age.max(initial=0) <= rows.window:
        return document_counts(group, valid, rows.tokens)[:, rows.pos]
    stats = np.zeros((3, len(at)))
    for a, b in zip(rows.starts, np.r_[rows.starts[1:], len(at)]):
        for i in range(a, b, rows.window):
            left, stop = max(a, i - rows.window), min(b, i + rows.window)
            stats[:, i:stop] = document_counts(group[left:stop], valid[left:stop], rows.tokens[left:stop])[
                :, i - left :
            ]
    return stats[:, rows.pos]


def candidates(depths):
    yield ()
    for stage in proposals(depths):
        yield from stage
    # A three/four-token cue is a pair of semantic spans; both are learned and ordered.
    if len(depths) >= 2:
        for b in (6, 10, depths[1]):
            yield ((0, 1, min(b, depths[1])), (2, 1, min(b, depths[1])))


def calibrate(model, rows):
    p = model.probability(rows.keys, rows.y, prior=rows.initial)
    memory = []
    for f in dict.fromkeys(candidates(rows.depths)):
        mass, counts, distinct = document_stats(rows, f)
        best = -np.log(p).mean()
        selected = None
        for kappa in KAPPA_GRID:
            q = categorical_reading(counts, mass, distinct, p, kappa, np)
            loss = -np.log(q).mean()
            if loss < best:
                best, selected = loss, (kappa, q)
        if selected:
            kappa, p = selected
            memory.append((f, kappa))
    return memory


def probabilities(model, memory, rows, omit=()):
    p = model.probability(rows.keys, rows.y, omit, prior=rows.initial)
    for f, kappa in memory:
        if f not in omit:
            m, n, d = document_stats(rows, f)
            p = categorical_reading(n, m, d, p, kappa, np)
    return p


def judge(model, rows, memory):
    """Coordinate minimization of actual next-category NLL, including dropping redundant levels."""
    global_levels = [(f, t.lookup(rows.keys(f), rows.y)) for f, t, k in model.levels]
    local_levels = [(f, document_stats(rows, f)) for f, k in memory]
    stats = [s for f, s in global_levels + local_levels]
    weights = [k for f, t, k in model.levels] + [k for f, k in memory]
    mass_prior = model.prior[rows.y] if rows.initial is None else rows.initial
    for _ in range(4):
        # Each more specific reading is affine in the preceding probability.
        after = []
        slope, intercept = np.ones(len(rows.y)), np.zeros(len(rows.y))
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
    n = len(model.levels)
    model.levels = [(f, t, k) for (f, t, _), k in zip(model.levels, weights[:n]) if np.isfinite(k)]
    return [(f, k) for (f, _), k in zip(memory, weights[n:]) if np.isfinite(k)]
