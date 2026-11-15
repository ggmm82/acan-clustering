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


# ---- concepts as the maximum of the thesis CP (Category Utility), corrected for chance -------------------

class Attributes:
    """The attributes of a concept's rows as columns, grouped by attribute, with their weights.

    A numeric attribute is one column; a categorical one is its indicator columns. CP^2 is the
    weighted mean over attributes of eta^2 (explained variance; for indicators, explained Gini
    impurity), which is K times the Category Utility of Gluck and Corter.
    """

    def __init__(self, columns, groups, weights):
        self.columns, self.groups, self.weights = columns, groups, np.asarray(weights, float)
        self.group_of = np.concatenate([np.full(len(group), a) for a, group in enumerate(groups)]) if groups else np.zeros(0, int)

    def cu(self, codes, rows):
        """Category Utility of a partition of ``rows``, corrected for chance: mean over attributes of
        (eta^2 - E[eta^2 | independence]) / (1 - E[...]), divided by the number of concepts.

        E[eta^2] = (K - 1)/(n - 1) for an attribute independent of the partition: a split that only
        explains noise is worth about nothing, so concepts do not multiply on noise.
        """
        x = self.columns[rows]
        n = len(x)
        cells, codes = np.unique(codes, return_inverse=True)
        k = len(cells)
        if k < 2 or n <= k:
            return 0.
        count = np.bincount(codes, minlength=k).astype(float)
        sums = np.zeros((k, x.shape[1]))
        np.add.at(sums, codes, x)
        mean = x.mean(0)
        between = (sums**2/count[:, None]).sum(0) - n*mean**2                       # per column
        total = ((x-mean)**2).sum(0)
        b = np.bincount(self.group_of, weights=between, minlength=len(self.groups))
        t = np.bincount(self.group_of, weights=total, minlength=len(self.groups))
        valid = t > 1e-12
        eta = np.where(valid, b/np.maximum(t, 1e-12), 0.)
        chance = (k-1)/(n-1)
        corrected = (eta-chance)/(1-chance)
        weights = self.weights*valid
        if weights.sum() <= 0:
            return 0.
        return float((weights*corrected).sum()/weights.sum()/k)

    def best_threshold(self, projection, rows):
        """The two-way split of ``rows`` along ``projection`` with the largest corrected CU (prefix sums)."""
        order = np.argsort(projection, kind='stable')
        x = self.columns[rows][order]
        n = len(x)
        if n < 4:
            return None, 0.
        prefix = np.cumsum(x, 0)
        total_sum = prefix[-1]
        left = np.arange(1, n)[:, None].astype(float)
        right = n-left
        mean = total_sum/n
        between = (prefix[:-1]**2/left + (total_sum-prefix[:-1])**2/right) - n*mean**2         # (n-1, columns)
        total = ((x-mean)**2).sum(0)
        b = np.zeros((n-1, len(self.groups)))
        np.add.at(b.T, self.group_of, between.T)
        t = np.bincount(self.group_of, weights=total, minlength=len(self.groups))
        valid = t > 1e-12
        eta = np.where(valid, b/np.maximum(t, 1e-12), 0.)
        chance = 1/(n-1)
        weights = self.weights*valid
        cu = ((eta-chance)/(1-chance)*weights).sum(1)/max(weights.sum(), 1e-12)/2
        values = projection[order]
        cu[values[1:] <= values[:-1]] = -np.inf                  # equal values stay together
        cu[:1] = cu[-1:] = -np.inf                               # no concept of a single row
        best = int(np.argmax(cu))
        if not np.isfinite(cu[best]):
            return None, 0.
        threshold = (values[best]+values[best+1])/2
        return threshold, float(cu[best])


def cu_children(attributes, features, rows, categorical_codes, most=3, directions=2, rng=None, permutations=0):
    """Best partition and its CU, kept only if it beats what the same search finds with independent attributes.

    The reference: every attribute's values shuffled over the rows separately (links between
    attributes destroyed, each attribute's distribution kept), and the same search run again.
    A division that only explains the attribute it is made of is found as well on shuffled data,
    so it does not pass: concepts do not multiply on noise.
    """
    codes, value = _cu_search(attributes, features, rows, categorical_codes, most, directions)
    if codes is None or not permutations:
        return codes, value
    null = []
    for _ in range(permutations):
        orders = [rng.permutation(len(rows)) for _ in attributes.groups]
        columns = attributes.columns[rows].copy()
        for order, group in zip(orders, attributes.groups):
            columns[:, group] = columns[order][:, group]
        shuffled = Attributes(columns, attributes.groups, attributes.weights)
        shuffled.assign_weights = getattr(attributes, 'assign_weights', attributes.weights)
        local = np.arange(len(rows))
        # The features and categorical codes follow the same shuffles as their attributes.
        feature_part = np.zeros_like(features[rows])
        for j in range(features.shape[1]):
            feature_part[:, j] = features[rows][orders[attributes.feature_group[j]], j]
        codes_part = [c[rows][orders[g]] for c, g in zip(categorical_codes, attributes.categorical_group)]
        null.append(_cu_search(shuffled, feature_part, local, codes_part, most, directions, full=True)[1])
    if value <= max(null):
        return None, value
    return codes, value


def _cu_search(attributes, features, rows, categorical_codes, most=3, directions=2, full=False):
    """``full``: ``features`` and codes are already the rows of the concept."""
    """The partition of a concept that maximizes the corrected Category Utility, or None.

    Candidate splits, each in closed form: the values of a categorical attribute; the best
    threshold of a numeric attribute or of a principal direction of the concept. Partitions are
    the cells of the best one, two or three candidates taken together: the four combinations of
    two attributes are one partition, as Category Utility wants when they explain more than
    either attribute alone. The partition kept is the maximum; none if nothing is above zero.
    """
    candidates = []
    part = features if full else features[rows]
    local_rows = rows if not full else np.arange(len(part))
    for codes in categorical_codes:
        values = codes if full else codes[rows]
        if len(np.unique(values)) > 1:
            candidates.append(values)
    for j in range(part.shape[1]):
        threshold, _ = attributes.best_threshold(part[:, j], local_rows)
        if threshold is not None:
            candidates.append((part[:, j] > threshold).astype(int))
    centred = part-part.mean(0)
    if len(part) > 2 and np.abs(centred).sum() > 0:
        for direction in np.linalg.svd(centred, full_matrices=False)[2][:directions]:
            projection = centred@direction
            threshold, _ = attributes.best_threshold(projection, local_rows)
            if threshold is not None:
                candidates.append((projection > threshold).astype(int))
    if not candidates:
        return None, 0.
    local = rows if not full else np.arange(len(part))
    columns = attributes.columns[local] if not full else attributes.columns
    scored = sorted(((attributes.cu(codes, local), index) for index, codes in enumerate(candidates)), reverse=True)
    best_codes, best_cu = None, 0.
    combined = np.zeros(len(part), int)
    for taken, (value, index) in enumerate(scored[:most]):
        combined = combined*(candidates[index].max()+1)+candidates[index]
        codes = lloyd(attributes, columns, np.unique(combined, return_inverse=True)[1])
        value = attributes.cu(codes, local)
        if value > best_cu+1e-12:
            best_codes, best_cu = codes, value
    return best_codes, best_cu


def lloyd(attributes, columns, codes, steps=3):
    """Steps that can only raise the quadratic CP for the cells it has: every row to the nearest
    cell centre in the metric of the CP (each attribute divided by its total spread, times its weight).
    Cells left empty disappear, so the stragglers of a combination of thresholds are absorbed."""
    mean = columns.mean(0)
    spread = np.bincount(attributes.group_of, weights=((columns-mean)**2).sum(0), minlength=len(attributes.groups))
    # Rows are assigned from what is observed when the model is used: a target weighs nothing here,
    # as in the recognition of ConceptModel; it counts only in the CP that judges the partition.
    scale = (getattr(attributes, 'assign_weights', attributes.weights)/np.maximum(spread, 1e-12))[attributes.group_of]
    for _ in range(steps):
        k = codes.max()+1
        count = np.bincount(codes, minlength=k).astype(float)
        centres = np.zeros((k, columns.shape[1]))
        np.add.at(centres, codes, columns)
        centres /= np.maximum(count, 1)[:, None]
        distance = ((columns**2)*scale).sum(1)[:, None]-2*(columns*scale)@centres.T+((centres**2)*scale).sum(1)[None]
        distance[:, count == 0] = np.inf
        updated = np.unique(distance.argmin(1), return_inverse=True)[1]
        if np.array_equal(updated, codes):
            break
        codes = updated
    return codes


class CUConceptModel(ConceptModel):
    """Concepts that maximize the thesis CP (Category Utility, corrected for chance), level by level.

    Each concept is divided into the children that maximize the corrected CU on its own rows (its
    rows are the context, as in the levels of ConceptModel); a concept whose best division is worth
    nothing stays whole. Levels are the generations of this division. The judgment of ConceptModel
    chooses which level makes the predictions; the covariance of the working level is adopted by
    judgment as in ConceptModel. The CP is unsupervised, as in the thesis: a target is one attribute
    like the others, and predicting it is completing it.
    """

    def __init__(self, max_levels=12, permutations=19, **options):
        super().__init__(**options)
        self.max_levels_cu, self.permutations = max_levels, permutations

    def _attributes(self, base, target_weight):
        x, mask, onehot = base[:3]
        columns, groups, weights, start = [], [], [], 0
        for j, name in enumerate(self.encoder.numeric):
            columns.append(x[:, j:j+1]); groups.append([start]); start += 1
            weights.append(target_weight if name in self.labels else 1.)
        for name in self.encoder.categorical:
            block = onehot[:, self.encoder.block(name)]
            columns.append(block); groups.append(list(range(start, start+block.shape[1]))); start += block.shape[1]
            weights.append(target_weight if name in self.labels else 1.)
        attributes = Attributes(np.concatenate(columns, 1) if columns else np.zeros((len(x), 0)), groups, weights)
        attributes.assign_weights = np.ones(len(groups))           # unsupervised: all attributes, as in the CP
        # Which attribute each feature column and each categorical code belongs to (for the shuffles).
        attributes.feature_group = [j for j, name in enumerate(self.encoder.numeric) if name not in self.labels]
        offset = len(self.encoder.numeric)
        attributes.categorical_group = [offset+i for i, name in enumerate(self.encoder.categorical) if name not in self.labels]
        return attributes

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
        features_count = len(self.encoder.numeric)+len(self.encoder.categorical)-len(self.labels)
        weights = self.train_weights = self._weights(float(max(features_count, 1)))
        rng = np.random.default_rng(self.seed)
        judged = np.sort(rng.choice(n, min(n, self.judged_rows), replace=False))
        # The CP of the thesis is unsupervised: every attribute weighs one, a target included.
        attributes = self._attributes(base, 1.)
        x, mask, onehot = base[:3]
        numeric = [j for j, name in enumerate(self.encoder.numeric) if name not in self.labels]
        features = x[:, numeric]
        categorical_codes = [onehot[:, self.encoder.block(name)].argmax(1)*(onehot[:, self.encoder.block(name)].sum(1) > 0)
                             + (onehot[:, self.encoder.block(name)].sum(1) == 0)*onehot[:, self.encoder.block(name)].shape[1]
                             for name in self.encoder.categorical if name not in self.labels]
        cell = np.zeros(n, int)
        generations = [cell.copy()]
        for _ in range(self.max_levels_cu):
            following = cell.copy()*0
            next_id, changed = 0, False
            for c in range(cell.max()+1):
                rows = np.flatnonzero(cell == c)
                children, value = ((None, 0.) if len(rows) < 2*self.rows_per_concept else
                                   cu_children(attributes, features, rows, categorical_codes, rng=rng, permutations=self.permutations))
                if children is None or value <= 0:
                    following[rows] = next_id; next_id += 1
                else:
                    following[rows] = next_id+children; next_id += children.max()+1; changed = True
            if not changed:
                break
            cell = following
            generations.append(cell.copy())
        steps = []
        for level, codes in enumerate(generations):
            membership = np.eye(codes.max()+1)[codes]
            statistics = self._statistics(membership, base)
            score = self._score(statistics, base, weights, judged)
            steps.append(dict(depth=level, k=membership.shape[1], statistics=statistics, membership=membership,
                              rows=score, score=float(score.mean()), error=float(score.std()/np.sqrt(len(score)))))
        best = max(steps, key=lambda step: step['score'])
        def as_good(step):
            gap = best['rows']-step['rows']
            return gap.mean() <= gap.std()/np.sqrt(len(gap))+1e-9
        working = min((step for step in steps if as_good(step)), key=lambda step: step['k'])
        self.statistics = working['statistics']
        self.cross, self.cov, self.above = None, np.zeros(0, int), None
        self._together(base, weights, judged)
        self.path = [dict(k=step['k'], score=step['score'], restart=0) for step in steps]
        self.selection = dict(best_k=best['k'], best_score=best['score'], standard_error=best['error'],
                              chosen_k=working['k'], chosen_score=working['score'], depth=working['depth'])
        self.levels = []
        for place, step in enumerate(steps[1:working['depth']+1], start=1):
            membership = step['membership']
            parents = None if place == 1 else (steps[place-1]['membership'].T@membership)/np.maximum(membership.sum(0), 1e-12)
            self.levels.append(dict(depth=place, concepts=int(membership.shape[1]), pair_criterion=None, score=step['score'],
                                    mass=membership.sum(0)/n, parents=parents, statistics=step['statistics'], table=step['statistics']))
        self.edges, self.final_score = [], working['score']
        return self
