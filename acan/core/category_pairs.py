"""Lossless dictionary for two ordered categorical observations in one tabular row."""

import numpy as np


class CategoryPairs:
    """Zero denotes an unavailable child or a previously unseen pair."""

    @staticmethod
    def keys(older, recent):
        older, recent = np.asarray(older), np.asarray(recent)
        if older.shape != recent.shape or older.ndim != 1:
            raise ValueError("Servono due colonne della stessa lunghezza.")
        for col in (older, recent):
            if col.dtype.kind not in "iu" or np.any(col < 0) or col.max(initial=0) > 0xFFFFFFFF:
                raise ValueError("Gli indici delle categorie devono essere interi a 32 bit senza segno.")
        codes = (older.astype(np.uint64) << 32) | recent.astype(np.uint64)
        codes[(older == 0) | (recent == 0)] = 0
        return codes

    def fit(self, older, recent):
        keys = self.keys(older, recent)
        self.observed = np.unique(keys[keys != 0])
        return self

    def transform(self, older, recent):
        keys = self.keys(older, recent)
        if not len(self.observed):
            return np.zeros(len(keys), dtype=np.uint32)
        idx = np.minimum(np.searchsorted(self.observed, keys), len(self.observed) - 1)
        return np.where(self.observed[idx] == keys, idx + 1, 0).astype(np.uint32)


def category_rows(columns, vocabulary):
    parent, child, y = map(np.asarray, columns)
    if vocabulary < 1 or y.ndim != 1 or not len(y) or parent.shape != y.shape or child.shape != y.shape:
        raise ValueError("Servono due colonne osservate e un target per ogni riga.")
    if y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= vocabulary):
        raise ValueError("Categoria target fuori dal dizionario.")
    return parent, child, y.astype(np.int64, copy=False)


def judged_subconcepts(build, judge, vocabulary):
    """Select local categorical splits by paired out-of-sample log score within each parent.

    Both triples are (parent key, child key, target category index). Zero is a
    missing predictor. A proposed child refines only its parent, irrespective of
    the number of regions elsewhere. A child without judgment evidence inherits
    its parent. Score each occurrence before aggregating within the branch.
    """
    from acan.concept_algebra import KAPPA_GRID, categorical_reading
    from .category_table import Table

    p, c, y = category_rows(build, vocabulary)
    vp, vc, vy = category_rows(judge, vocabulary)
    child = CategoryPairs.keys(p, c)
    vchild = CategoryPairs.keys(vp, vc)
    parent = Table.count_regions([(p, y)], vocabulary)
    fine = Table.count_regions([(child, y)], vocabulary)
    cnt = np.bincount(y, minlength=vocabulary) + 1
    base = (cnt / cnt.sum())[vy]

    def calibrated(table, keys, prior):
        m, n, d = table.lookup(keys, vy)
        best, prediction = np.inf, prior
        for weight in KAPPA_GRID:
            estimate = categorical_reading(n, m, d, prior, weight, np)
            score = -np.log(estimate).mean()
            if score < best:
                best, prediction = score, estimate
        return prediction

    broad = calibrated(parent, vp, base)
    narrow = calibrated(fine, vchild, broad)
    gain = np.log(narrow) - np.log(broad)
    index = fine.find(vchild)
    seen = index >= 0
    k = len(fine.keys)

    def aggregate(weights=None):
        return np.bincount(index[seen], weights=weights, minlength=k)

    reads = aggregate()
    total = aggregate(gain[seen])
    square = aggregate(gain[seen] ** 2)
    # One standard error of the paired gain on the sum of the held-out reads.
    error = np.sqrt(np.maximum(0, reads * square - total**2) / np.maximum(reads - 1, 1))
    selected = (reads >= 2) & (total > error)
    return fine.keys[selected], dict(
        proposed=k,
        judged=int((reads >= 2).sum()),
        accepted=int(selected.sum()),
        accepted_judgment_rows=int(reads[selected].sum()),
        judge_rows=len(vy),
    )


class PairColumns:
    """Read two fitted categorical predictor dictionaries from a tabular input frame."""

    def __init__(self, parent, child, encoder):
        self.names = (parent, child)
        self.codes = [{v: i + 1 for i, v in enumerate(encoder.categories[name])} for name in self.names]

    def parts(self, frame):
        parts = []
        for name, mapping in zip(self.names, self.codes):
            if name not in frame:
                parts.append(np.zeros(len(frame), dtype=np.uint32))
            else:
                parts.append(
                    frame[name].astype("string").map(mapping).fillna(0).to_numpy(dtype=np.uint32)
                )
        return tuple(parts)

    def keys(self, frame):
        return CategoryPairs.keys(*self.parts(frame))
