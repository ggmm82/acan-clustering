"""Parent-local CP fits with sparse routing to exactly one child model per row."""

import numpy as np


class LocalConcepts:
    def __init__(self, parent):
        self.parent = parent
        self.branches = {}
        self.concepts = 0
        self.judged = 0

    @staticmethod
    def grouped(ids):
        order = np.argsort(ids, kind="stable")
        for rows in np.split(order, np.flatnonzero(np.diff(ids[order])) + 1):
            if len(rows):
                yield int(ids[rows[0]]), rows

    def _parents(self, frame):
        parts = []
        for begin in range(0, len(frame), 2048):
            parts.append(self.parent.assignments(frame.iloc[begin : begin + 2048]).argmax(1))
        return np.concatenate(parts) if parts else np.empty(0, int)

    def fit(self, frame):
        from .concepts import ConceptModel

        self.branches = {}
        self.concepts = self.parent.concepts
        self.judged = 0
        for node, rows in self.grouped(self._parents(frame)):
            part = frame.iloc[rows]
            # Fewer than two disjoint pairs cannot support a split by the pair criterion.
            if len(part) < 4 or not (part.nunique(dropna=True) > 1).any():
                continue
            child = ConceptModel(
                max_concepts=len(part),
                rows_per_concept=self.parent.rows_per_concept,
                bins=12,
                judged_rows=len(part),
                refine_rows=8192,
                covariance=False,
                seed=self.parent.seed,
                merge_fraction=0.35,
                em_steps=2,
            )
            child.fit(part)
            self.judged += min(len(part), child.refine_rows, child.judged_rows)
            if child.concepts > 1:
                self.branches[node] = self.concepts, child
                self.concepts += child.concepts
        return self

    def transform(self, frame):
        parent = self._parents(frame)
        codes = parent.copy()
        for node, rows in self.grouped(parent):
            if node in self.branches:
                offset, child = self.branches[node]
                for i in range(0, len(rows), 2048):
                    batch = rows[i : i + 2048]
                    codes[batch] = offset + child.assignments(frame.iloc[batch]).argmax(1)
        return codes
