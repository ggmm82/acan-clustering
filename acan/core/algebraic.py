"""The concept graph with concepts given by a formula instead of a search (prototype).

Everything of ``ConceptModel`` stays: the encoding, the additive statistics of a concept, the
reading as mass times expected similarity on the observed attributes, the judgment on rows the
concepts did not see, the predictions. What changes is how rows are assigned to concepts. The
present model searches: it merges, pair by pair, the concepts whose union costs the reader least.
Here a row's concept is a code computed in closed form:

* directions: with a categorical target, Fisher's discriminant directions (between-class over
  within-class scatter, one eigen-decomposition); with a numeric target, the least-squares
  direction; then the principal directions (one SVD). Without a target, the principal directions;
* levels: bit j of a row is its projection on direction j against the mean projection of the
  concept it belongs to at level j - 1. Each level removes the centre of its concept, as the levels
  of the present model do, and splits what is left in two;
* depth: the judgment of the present model (log-probability of what unseen rows must read)
  chooses how many levels, the fewest within one standard error of the best.

No merge is searched; the only repetitions are over levels (at most ``max_depth``), each a pass of
grouped sums over all rows. The same rule assigns tokens in acan/lm/algebraic_attention.py, where
the code is the age of a token and the signs of its key.
"""
import numpy as np
import pandas as pd

from acan.core.concepts import ConceptModel, Encoder


def fisher_directions(features, classes):
    """Directions that separate the classes most against their spread inside, best first."""
    labels, codes = np.unique(classes, return_inverse=True)
    if len(labels) < 2:
        return np.zeros((features.shape[1], 0))
    overall = features.mean(0)
    within = np.zeros((features.shape[1],)*2)
    between = np.zeros_like(within)
    for k in range(len(labels)):
        part = features[codes == k]
        weight = len(part)/len(features)
        centred = part-part.mean(0)
        within += weight*(centred.T@centred)/max(len(part), 1)
        gap = part.mean(0)-overall
        between += weight*np.outer(gap, gap)
    within += 1e-3*np.trace(within)/max(len(within), 1)*np.eye(len(within)) + 1e-9*np.eye(len(within))
    values, vectors = np.linalg.eig(np.linalg.solve(within, between))
    order = np.argsort(-values.real)[:len(labels)-1]
    directions = vectors.real[:, order]
    return directions/np.maximum(np.linalg.norm(directions, axis=0), 1e-12)


def centred_codes(features, directions):
    """Cell of every row at every level: bit j against the mean projection of its cell at level j - 1."""
    n = len(features)
    cell = np.zeros(n, int)
    cells = [cell.copy()]
    for j in range(directions.shape[1]):
        projection = features@directions[:, j]
        total = np.bincount(cell, weights=projection)
        count = np.bincount(cell)
        centre = (total/np.maximum(count, 1))[cell]
        cell = 2*cell+(projection > centre)
        cells.append(np.unique(cell, return_inverse=True)[1])     # renumber: cells present only
        cell = cells[-1]
    return cells


def own_direction(part, classes=None):
    """The direction along which a concept splits: Fisher's between its classes, else its principal one."""
    centred = part-part.mean(0)
    if classes is not None and len(np.unique(classes)) > 1:
        fisher = fisher_directions(part, classes)
        if fisher.shape[1]:
            return fisher[:, 0]
    if len(part) < 2:
        return None
    vector = np.linalg.svd(centred, full_matrices=False)[2][0]
    return vector


def divisive_codes(features, depth, classes=None, smallest=2):
    """Cell of every row at every level; each cell splits along its own direction at its own centre.

    The split of a concept is closed-form (one eigen-decomposition or one SVD of its members): what
    separates its classes when it has a target, else its principal direction. Cells too small to
    split stay whole.
    """
    n = len(features)
    cell = np.zeros(n, int)
    cells = [cell.copy()]
    for _ in range(depth):
        bit = np.zeros(n, int)
        for c in range(cell.max()+1):
            members = np.flatnonzero(cell == c)
            if len(members) < 2*smallest:
                continue
            direction = own_direction(features[members], None if classes is None else classes[members])
            if direction is None:
                continue
            projection = features[members]@direction
            bit[members] = projection > projection.mean()
        cell = np.unique(2*cell+bit, return_inverse=True)[1]
        if cell.max() == cells[-1].max():
            break                                                # nothing split any more
        cells.append(cell)
    return cells


class AlgebraicConceptModel(ConceptModel):
    def __init__(self, max_depth=10, **options):
        super().__init__(**options)
        self.max_depth = max_depth

    def _features(self, data):
        """The attributes that recognize a row (targets excluded), as numbers."""
        x, mask, onehot = data[:3]
        keep_numeric = [j for j, name in enumerate(self.encoder.numeric) if name not in self.labels]
        keep_onehot = np.ones(self.encoder.width, bool)
        for label in self.labels:
            if label in self.encoder.categorical:
                keep_onehot[self.encoder.block(label)] = False
        # Every attribute weighs as one, as in the reading: a categorical block is scaled so that its
        # variance adds up to one, like a standardized numeric attribute.
        scaled = onehot.copy()
        for block in self._blocks():
            spread = scaled[:, block].var(0).sum()
            scaled[:, block] /= np.sqrt(max(spread, 1e-12))
        features = np.concatenate((x[:, keep_numeric], scaled[:, keep_onehot]), 1)
        return features-features.mean(0)

    def _directions(self, frame, features):
        principal = np.linalg.svd(features, full_matrices=False)[2].T if features.size else np.zeros((features.shape[1], 0))
        supervised = np.zeros((features.shape[1], 0))
        for label in self.labels[:1]:
            target = frame[label]
            if label in self.encoder.categorical:
                known = target.notna().to_numpy()
                supervised = fisher_directions(features[known], target[known].astype(str).to_numpy())
            else:
                known = target.notna().to_numpy()
                y = target[known].to_numpy(float)
                weights = np.linalg.lstsq(features[known], y-y.mean(), rcond=None)[0]
                supervised = (weights/max(np.linalg.norm(weights), 1e-12))[:, None]
        return np.concatenate((supervised, principal), 1)

    def fit(self, frame, labels=(), encoder=None, progress=None):
        self.labels = list(labels)
        self.encoder = encoder or Encoder().fit(frame)
        base = self.encoder.transform(frame)
        n = len(frame)
        observed = np.zeros(self.encoder.width)
        for block in self._blocks():
            observed[block] = max(base[2][:, block].sum(), 1)
        self.root_p = np.maximum(base[2].sum(0)/observed, 1e-12)
        self.root_g = np.array([1-(self.root_p[block]**2).sum() for block in self._blocks()])
        self.resolution = self._resolution(base)
        features = len(self.encoder.numeric)+len(self.encoder.categorical)-len(self.labels)
        weights = self.train_weights = self._weights(float(max(features, 1)))
        rng = np.random.default_rng(self.seed)
        judged = np.sort(rng.choice(n, min(n, self.judged_rows), replace=False))
        x = self._features(base)
        depth = int(min(self.max_depth, max(np.log2(max(n/self.rows_per_concept, 1)), 1)))
        classes = None
        if self.labels and self.labels[0] in self.encoder.categorical:
            classes = frame[self.labels[0]].astype(str).where(frame[self.labels[0]].notna(), None).to_numpy()
            classes = np.where(pd.isna(classes), '\u2205', classes).astype(str)
        elif self.labels:                                        # a numeric target: its halves, split at the median
            target = frame[self.labels[0]].to_numpy(float)
            classes = np.where(np.isnan(target), 'unknown', np.where(target > np.nanmedian(target), 'high', 'low'))
        cells = divisive_codes(x, depth, classes, smallest=self.rows_per_concept)
        steps = []
        for level, cell in enumerate(cells):
            membership = np.eye(cell.max()+1)[cell]
            statistics = self._statistics(membership, base)
            score = self._score(statistics, base, weights, judged)
            steps.append(dict(depth=level, k=membership.shape[1], statistics=statistics, membership=membership,
                              rows=score, score=float(score.mean()), error=float(score.std()/np.sqrt(len(score)))))
        best = max(steps, key=lambda step: step['score'])
        # The simplest depth that reads unseen rows as well as the best one, judged row by row on the
        # same rows (the paired difference, as the levels of the present model are judged).
        def as_good(step):
            gap = best['rows']-step['rows']
            return gap.mean() <= gap.std()/np.sqrt(len(gap))+1e-9
        working = min((step for step in steps if as_good(step)), key=lambda step: step['k'])
        self.statistics = working['statistics']
        self.cross, self.cov, self.above = None, np.zeros(0, int), None
        self._together(base, weights, judged)                   # attributes that vary together, adopted by judgment
        self.path = [dict(k=step['k'], score=step['score'], restart=0) for step in steps]
        self.selection = dict(best_k=best['k'], best_score=best['score'], standard_error=best['error'],
                              chosen_k=working['k'], chosen_score=working['score'], depth=working['depth'])
        # The graph: one level per depth up to the working one, joined by the parent of each cell.
        self.levels = []
        for place, step in enumerate(steps[1:working['depth']+1], start=1):
            membership = step['membership']
            parents = None
            if place > 1:
                above = steps[place-1]['membership']
                parents = (above.T@membership)/np.maximum(membership.sum(0), 1e-12)
            self.levels.append(dict(depth=place, concepts=int(membership.shape[1]), pair_criterion=None, score=step['score'],
                                    mass=membership.sum(0)/n, parents=parents, statistics=step['statistics'],
                                    table=step['statistics']))
        self.edges = []
        self.final_score = working['score']
        return self
