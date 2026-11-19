"""The concept graph on tables (profile ``concept-graph-v1``).

One model, defined in acan/concept_algebra.py and docs/MODEL.md, shared with the
attention layer (acan/lm/concept_attention.py):

* a concept is a node holding additive statistics of its members;
* membership is mass times expected similarity, on the observed attributes only;
* generalization adds together the pair whose merge costs the reader least;
* concepts form levels. Level 1 holds the general concepts of the whole table;
  each further level holds the particular behaviours that are distinct only at
  the scale of the level above. A level is the cut of the generalization that
  maximizes the pair criterion against the level above, and it is kept only if
  it reads unseen observations better;
* consecutive levels are joined by a dense matrix: a concept may descend from
  several concepts of the level above, so what is one concept is one node even
  when it is reached from different general concepts;
* inside a level two concepts are joined by an edge when observations hesitate
  between them.

Every step is linear in the number of attributes.
"""
import numpy as np
import pandas as pd

from acan.concept_algebra import KAPPA_GRID, LOG_2PI, categorical_reading, centre_variance, pair_mass, reader_cost, reader_radius, unbiased
from acan.core.scan import Scan, scan, scan_part

TIE = .05                                                       # relative difference of pair criteria that counts as a tie
CONSTANT = 1e-6                                                 # standardized dispersion below which an attribute is constant


class Encoder:
    """Standardized numeric columns and categorical codes; statistics from training only."""

    def fit(self, frame):
        self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in frame if c not in self.numeric]
        values = frame[self.numeric].to_numpy(float) if self.numeric else np.zeros((len(frame), 0))
        self.mean = np.nan_to_num(np.nanmean(values, axis=0)) if values.size else np.zeros(0)
        std = np.nan_to_num(np.nanstd(values, axis=0)) if values.size else np.zeros(0)
        self.std = np.where(std > 1e-12, std, 1.)
        self.categories = {c: sorted(frame[c].dropna().astype(str).unique()) for c in self.categorical}
        self.offsets, start = {}, 0
        for c in self.categorical:
            self.offsets[c] = start
            start += len(self.categories[c])
        self.width = start
        return self

    def fit_chunks(self, chunks):
        """The same statistics from a table read a chunk at a time: sums are enough."""
        count = total = square = None
        for frame in chunks:
            if count is None:
                self.numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
                self.categorical = [c for c in frame if c not in self.numeric]
                count, total, square = (np.zeros(len(self.numeric)) for _ in range(3))
                seen = {c: set() for c in self.categorical}
            values = frame[self.numeric].apply(pd.to_numeric, errors='coerce').to_numpy(float) if self.numeric else np.zeros((len(frame), 0))
            present = ~np.isnan(values)
            values = np.where(present, values, 0.)
            count, total, square = count+present.sum(0), total+values.sum(0), square+(values*values).sum(0)
            for c in self.categorical:
                seen[c].update(frame[c].dropna().astype(str).unique())
        if count is None:
            raise ValueError('Tabella vuota.')
        self.mean = np.where(count > 0, total/np.maximum(count, 1), 0.)
        std = np.sqrt(np.maximum(square/np.maximum(count, 1)-self.mean**2, 0.))
        self.std = np.where(std > 1e-12, std, 1.)
        self.categories = {c: sorted(seen[c]) for c in self.categorical}
        self.offsets, start = {}, 0
        for c in self.categorical:
            self.offsets[c] = start
            start += len(self.categories[c])
        self.width = start
        return self

    def transform(self, frame):
        n = len(frame)
        x = np.full((n, len(self.numeric)), np.nan)
        for j, c in enumerate(self.numeric):
            if c in frame:
                x[:, j] = (pd.to_numeric(frame[c], errors='coerce').to_numpy(float)-self.mean[j])/self.std[j]
        mask = ~np.isnan(x)
        onehot = np.zeros((n, self.width))
        for c in self.categorical:
            if c not in frame:
                continue
            index = {value: i for i, value in enumerate(self.categories[c])}
            codes = frame[c].map(lambda v: index.get(str(v), -1) if pd.notna(v) else -1).to_numpy()
            rows = np.flatnonzero(codes >= 0)
            onehot[rows, self.offsets[c]+codes[rows]] = 1
        return np.where(mask, x, 0.), mask.astype(float), onehot

    def block(self, column):
        return slice(self.offsets[column], self.offsets[column]+len(self.categories[column]))


def _logsumexp(values, axis):
    top = values.max(axis=axis, keepdims=True)
    return (np.log(np.exp(values-top).sum(axis=axis, keepdims=True))+top).squeeze(axis)


def _softmax(logits):
    top = logits.max(axis=1, keepdims=True)
    e = np.exp(logits-top)
    total = e.sum(axis=1, keepdims=True)
    return e/total, (top+np.log(total))[:, 0]


class ConceptModel:
    max_levels = 8                                              # capacity limit, not a requested depth
    resolution = 0.

    def __init__(self, max_concepts=1024, rows_per_concept=4, kappa=1., seed=0,
                 judged_rows=2000, merge_fraction=.05, em_steps=3, restarts=1, restart_rows=5000,
                 covariance=True, covariance_attributes=128, shared_start=128, bins=None, refine_rows=20000, chunk_rows=100000, workers=1,
                 distinct=False):
        self.shared_start = shared_start
        self.bins, self.refine_rows = bins, refine_rows          # concepts of a block of the scan; rows the levels are refined on
        self.chunk_rows = chunk_rows                             # rows read at a time from a large table
        self.workers = workers                                   # processes that read the chunks of a large table
        self.report = None                                       # called with what is happening while learning, if set
        self.distinct = distinct                                 # the reading weighs the level above by the values a concept has
        self.covariance, self.covariance_attributes = covariance, covariance_attributes
        self.cross, self.cov = None, np.zeros(0, int)            # cross products of the working level, if adopted
        self.max_concepts, self.rows_per_concept = max_concepts, rows_per_concept
        self.kappa, self.seed, self.judged_rows = kappa, seed, judged_rows
        self.merge_fraction, self.em_steps = merge_fraction, em_steps
        self.restarts, self.restart_rows = restarts, restart_rows

    # ---- nodes: additive statistics --------------------------------------------
    def _blocks(self):
        return [self.encoder.block(name) for name in self.encoder.categorical]

    def _statistics(self, t, data):
        """Sums over the members. ``PV`` and ``PP`` add up the scale each member has in the level above."""
        x, mask, onehot = data[:3]
        s = dict(C=t.sum(0), Cm=t.T@mask, U=t.T@x, Q=t.T@(x*x), N=t.T@onehot)
        if len(data) > 3:
            scale, frequency, impurity = data[3:]
            s.update(PV=t.T@(mask*scale), PP=t.T@frequency, PG=t.T@impurity)
        else:                                                    # the root gives every row the same scale
            s.update(PV=s['Cm'].copy(), PP=np.outer(s['C'], self.root_p), PG=np.outer(s['C'], self.root_g))
        return s

    def _root_scale(self, x, onehot):
        """Scale of the root for every row: unit variance, the frequencies and the impurity of the table."""
        return (np.ones_like(x), np.broadcast_to(self.root_p, onehot.shape),
                np.broadcast_to(self.root_g, (len(x), len(self.root_g))))

    def _prior(self, s):
        """Scale of the level above as seen by each concept: the mean over its members."""
        variance = np.where(s['Cm'] > 1e-9, s['PV']/np.maximum(s['Cm'], 1e-12), 1.)
        frequency = s['PP']/np.maximum(s['C'], 1e-12)[..., None] if s['PP'].shape[-1] else s['PP']
        return np.maximum(variance, 1e-9), np.maximum(frequency, 1e-12)

    def _parameters(self, s, prior=None):
        """Centres; variances and frequencies shrink to the level above by kappa pseudo-members."""
        prior_v, prior_p = prior or self._prior(s)
        g, v = centre_variance(s['Cm'], s['U'], s['Q'], prior_v, self.kappa, np)
        v = np.maximum(v, self.resolution)                       # no finer than the attribute is measured
        size = np.zeros_like(s['N'])
        distinct = np.ones_like(s['N'])
        for block in self._blocks():
            size[..., block] = s['N'][..., block].sum(-1, keepdims=True)
            if self.distinct:
                distinct[..., block] = (s['N'][..., block] > 1e-9).sum(-1, keepdims=True)
        log_p = np.log(categorical_reading(s['N'], size, distinct, prior_p, self.kappa, np))
        return g, v, log_p

    @staticmethod
    def _resolution(data):
        """Variance of one step of each numeric attribute: a value on a grid is known to within a step.

        A count or a grade takes few values; a Gaussian narrower than their spacing is a point
        mass, and its density says nothing. The spacing is the smallest gap between two values.
        """
        x, mask = data[:2]
        floor = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            values = np.unique(x[mask[:, j] > 0, j])
            if len(values) > 1:
                floor[j] = np.diff(values).min()**2/12
        return floor

    # ---- membership ------------------------------------------------------------
    def _log_density(self, s, data, weights):
        x, mask, onehot = data[:3]
        wn, wc = weights
        g, v, log_p = self._parameters(s)
        a = 1/v
        together = self.cross is not None and s is self.statistics
        if together:                                             # these attributes are recognized together
            wn = wn.copy()
            wn[self.cov] = 0
        numeric = -.5*(((x*x)*wn)@a.T-2*(x*wn)@(g*a).T+(mask*wn)@(g*g*a+np.log(v)+LOG_2PI).T)
        if together:
            numeric = numeric+self._block_density(x, mask)
        return numeric+(onehot*wc)@log_p.T

    # ---- attributes that vary together, in the working level only ---------------
    def _cross(self, t, data):
        """Sum over the members of the products between features: still additive."""
        xc = data[0][:, self.cov]
        p = len(self.cov)
        return (t.T@(xc[:, :, None]*xc[:, None, :]).reshape(len(xc), p*p)).reshape(-1, p, p)

    def _covariance(self):
        """Centre and covariance of the features of each concept, shrunk to the scale of the root."""
        c, u = self.statistics['C'], self.statistics['U'][:, self.cov]
        g = u/np.maximum(c, 1e-12)[:, None]
        scatter = self.cross-c[:, None, None]*g[:, :, None]*g[:, None, :]
        return g, (scatter+self.kappa*np.eye(len(self.cov)))/(c+self.kappa)[:, None, None]

    def _block_density(self, x, mask):
        """Log-density of the features taken together; a missing one is dropped from the Gaussian exactly."""
        g, sigma = self._covariance()
        xc, mc = x[:, self.cov], mask[:, self.cov] > 0
        out = np.zeros((len(xc), len(g)))
        patterns, which = np.unique(mc, axis=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            rows = np.flatnonzero(which.reshape(-1) == index)
            if not pattern.any():
                continue
            factor = np.linalg.cholesky(sigma[:, pattern][:, :, pattern])
            volume = 2*np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(1)
            for i in range(len(g)):
                d = np.linalg.solve(factor[i], (xc[rows][:, pattern]-g[i, pattern]).T)
                out[rows, i] = -.5*((d*d).sum(0)+volume[i]+pattern.sum()*LOG_2PI)
        return out

    def _block_unseen(self, x, mask, t):
        """The same log-density from concepts that do not contain the row: a rank-one change, in closed form."""
        p = len(self.cov)
        _, sigma = self._covariance()
        volume = np.linalg.slogdet(sigma)[1]
        distance = -2*self._block_density(x, mask)-volume-p*LOG_2PI
        c = self.statistics['C'][None]
        rest = np.maximum(c-t, 0)
        safe = np.maximum(rest, 1e-12)
        q = distance/(c+self.kappa)
        left = np.maximum(1-t*c/safe*q, 1e-12)
        away = (c/safe)**2*(rest+self.kappa)*q/left
        unseen = volume[None]+p*np.log(c+self.kappa)+np.log(left)-p*np.log(rest+self.kappa)
        alone = rest < 1e-9                                      # nobody else: the scale of the root
        away = np.where(alone, (x[:, self.cov]**2).sum(1)[:, None], away)
        unseen = np.where(alone, 0., unseen)
        return -.5*(away+unseen+p*LOG_2PI)

    def _together(self, data, weights, judged):
        """Let the features of a concept vary together, if unseen rows are then read better.

        Only the working level, after the search: the search stays linear in the attributes.
        """
        block = np.array([j for j, name in enumerate(self.encoder.numeric) if name not in self.labels], int)
        if not (self.covariance and self.labels and 1 < len(block) <= self.covariance_attributes
                and bool((data[1][:, block] > 0).all())):
            return
        plain, before = self.statistics, self._score(self.statistics, data, weights, judged)
        self.cov = block
        t = self._route(plain, data, weights)[0]
        for _ in range(4):
            self.statistics = self._statistics(t, data)
            self.cross = self._cross(t, data)
            t = self._route(self.statistics, data, weights)[0]
        self.statistics, self.cross = self._statistics(t, data), self._cross(t, data)
        gain = self._score(self.statistics, data, weights, judged)-before
        if gain.mean() <= gain.std()/np.sqrt(len(gain))+1e-9:
            self.statistics, self.cross, self.cov = plain, None, np.zeros(0, int)

    def _route(self, s, data, weights):
        prior = np.log(np.maximum(s['C'], 1e-300)/s['C'].sum())
        return _softmax(self._log_density(s, data, weights)+prior)

    # ---- edges inside a level --------------------------------------------------
    def _edges(self, t):
        """Connect the two concepts each observation belongs to most, by the product of its memberships."""
        k = t.shape[1]
        if k < 2:
            return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
        top = np.argpartition(-t, 1, axis=1)[:, :2]
        rows = np.arange(len(t))
        weight = t[rows, top[:, 0]]*t[rows, top[:, 1]]
        low, high = top.min(1), top.max(1)
        total = np.zeros(k*k)
        np.add.at(total, low*k+high, weight)
        index = np.flatnonzero(total > 0)
        return index//k, index % k, total[index]

    # ---- dispersion against the level above ------------------------------------
    def _dispersion(self, s, pairs=False):
        """Per attribute, the dispersion of each concept divided by the dispersion of the level above.

        ``pairs`` uses pairs of distinct members (a single member has no dispersion to compare).
        Returns (ratios, is_read): one column per attribute, and which attributes are read.
        """
        prior_v, prior_p = self._prior(s)
        cm = np.maximum(s['Cm'], 1e-12)
        variance = np.maximum(s['Q']/cm-(s['U']/cm)**2, 0)
        if pairs:
            variance = unbiased(variance, s['Cm'], np)
        # An attribute that is already constant in the level above has nothing left to share: ratio one.
        columns = [np.where(prior_v > CONSTANT, variance/np.maximum(prior_v, CONSTANT), 1.)]
        read = [name in self.labels for name in self.encoder.numeric]
        base = s['PG']/np.maximum(s['C'], 1e-12)[..., None]
        for place, (name, block) in enumerate(zip(self.encoder.categorical, self._blocks())):
            size = np.maximum(s['N'][..., block].sum(-1), 1e-12)
            gini = 1-((s['N'][..., block]/size[..., None])**2).sum(-1)
            if pairs:
                gini = unbiased(gini, size, np)
            columns.append(np.where(base[..., place] > CONSTANT, gini/np.maximum(base[..., place], CONSTANT), 1.)[..., None])
            read.append(name in self.labels)
        return np.concatenate(columns, -1), np.asarray(read, bool)

    def _radius(self, s):
        """Spread of a concept in the part that is recognized, on the scale of the level above."""
        ratio, read = self._dispersion(s)
        recognized = ~read if (~read).any() else np.ones_like(read)
        return reader_radius(ratio[..., recognized].mean(-1), np)

    def pair_criterion(self, s, within=False):
        """Quadratic CP: mean over pairs of distinct observations of [same concept] x similarity.

        The similarity of two observations is centred on the average pair of the level above,
        so the value is zero for one concept and for single observations, and it grows only
        with concepts that are both large and distinct at that scale.
        """
        ratio, _ = self._dispersion(s, pairs=True)
        n = s['C'].sum()
        similarity = 1-np.minimum(ratio, 2.)                     # bounded: as unlike as it can be alike
        pairs = pair_mass(s['C'], np)
        # ``within`` averages over the pairs that share a concept: how much more alike they are,
        # whatever the size of the concepts.
        return float((pairs*similarity.mean(-1)).sum()/max(pairs.sum() if within else n*(n-1), 1e-12))

    # ---- generalization: where the reader loses least --------------------------
    def _pair_costs(self, s, first, second):
        """Reader's loss for each listed pair. A row reads the concepts it belongs to: reads received = mass."""
        own = self._radius(s)
        together = self._radius({key: value[first]+value[second] for key, value in s.items()})
        return reader_cost(s['C'][first], own[first], s['C'][second], own[second], together)

    def _merge_costs(self, s, weights=None, rows=None):
        """All pairs for the given concepts (rows x k)."""
        k = len(s['C'])
        index = np.arange(k) if rows is None else np.asarray(rows)
        cost = np.empty((len(index), k))
        everyone = np.arange(k)
        for place, concept in enumerate(index):
            cost[place] = self._pair_costs(s, np.full(k, concept), everyone)
        cost[np.arange(len(index)), index] = np.inf
        return cost

    def _generalize(self, s, data, weights, count):
        """Add together the disjoint pairs that cost the reader least. Every pair is a candidate."""
        k = len(s['C'])
        first, second = np.triu_indices(k, 1)
        # Adding a pair materializes its sufficient statistics, including categorical counts.
        batch = min(100000, max(1, 1000000//max(self.encoder.width+len(self.encoder.numeric), 1)))
        cost = np.concatenate([self._pair_costs(s, first[start:start+batch], second[start:start+batch])
                               for start in range(0, len(first), batch)])
        used, pairs = np.zeros(k, bool), []
        for place in np.argsort(cost):
            i, j = first[place], second[place]
            if not used[i] and not used[j]:
                used[i] = used[j] = True
                pairs.append((i, j))
                if len(pairs) == count:
                    break
        keep = np.ones(k, bool)
        merged = {key: value.copy() for key, value in s.items()}
        for i, j in pairs:
            for key in merged:
                merged[key][i] += merged[key][j]
            keep[j] = False
        return {key: value[keep] for key, value in merged.items()}

    def _em(self, s, data, weights, steps):
        for _ in range(steps):
            t, _ = self._route(s, data, weights)
            s = self._statistics(t, data)
            alive = s['C'] > 1e-8
            if not alive.all():
                s = {key: value[alive] for key, value in s.items()}
        return s

    # ---- judgment --------------------------------------------------------------
    def _weights(self, target_weight):
        wn = np.ones(len(self.encoder.numeric))
        wc = np.ones(self.encoder.width)
        for label in self.labels:
            if label in self.encoder.numeric:
                wn[self.encoder.numeric.index(label)] = target_weight
            else:
                wc[self.encoder.block(label)] = target_weight
        return wn, wc

    def _score(self, s, data, weights, rows, structure=False):
        """Log-probability of what each row must read, from concepts that never saw that row.

        The row's own contribution is subtracted from the additive statistics before it is
        recognized and read. With a target, the target is read from the features. Without
        one, every attribute is read in turn from the others, as when a table is completed:
        an attribute that the others do not predict gains nothing from more concepts.
        ``structure`` asks for the second reading also when there is a target: it is how a
        level is judged as a description of the table.
        """
        x, mask, onehot = (part[rows] for part in data[:3])
        scale, frequency = (part[rows] for part in data[3:5]) if len(data) > 3 else self._root_scale(x, onehot)[:2]
        t = self._route(s, (x, mask, onehot), weights)[0]
        k, encoder = len(s['C']), self.encoder
        above = self._prior(s)
        scores = np.empty(len(x))
        # Each categorical observation contributes to exactly one bin in its attribute.
        # Leave it out of that bin, keeping totals and distinct counts for the entire block.
        # This avoids materializing a rows x concepts x categories tensor for a wide target.
        blocks = self._blocks()
        categorical = []
        for block in blocks:
            observed = onehot[:, block]
            present = observed.sum(1) > 0
            binary = ((observed == 0) | (observed == 1)).all() and np.all(observed.sum(1) <= 1)
            codes = observed.argmax(1) if observed.shape[1] else np.zeros(len(x), int)
            counts = s['N'][:, block]
            categorical.append((present, codes, counts, counts.sum(1), (counts > 1e-9).sum(1), binary))
        size = max(int(4e6//(k*max(x.shape[1]+len(blocks), 1))), 1)
        for start in range(0, len(x), size):
            part = slice(start, start+size)
            r = t[part][:, :, None]
            xs, ms = x[part][:, None], mask[part][:, None]
            cm = np.maximum(s['Cm'][None]-r*ms, 0)
            u = s['U'][None]-r*xs
            q = np.maximum(s['Q'][None]-r*(xs*xs), 0)
            mass = np.maximum(s['C'][None]-t[part], 0)
            alone = mass < 1e-9
            g, v = centre_variance(cm, u, q, above[0], self.kappa, np)
            v = np.where(cm > 1e-9, np.maximum(v, self.resolution), scale[part][:, None])
            terms, read = [], []
            for j, name in enumerate(encoder.numeric):
                terms.append(-.5*((xs[..., j]-g[..., j])**2/v[..., j]+np.log(v[..., j])+LOG_2PI)*ms[..., j])
                read.append(name in self.labels)
            for name, block, info in zip(encoder.categorical, blocks, categorical):
                present, codes, counts, totals, distinct, binary = info
                if not binary:
                    # Custom matrix encoders may supply several values or weights for an attribute.
                    # Preserve the original weighted log-density, one observation at a time.
                    terms_block = np.empty_like(mass)
                    for i, obs in enumerate(onehot[part, block]):
                        left = np.maximum(counts-r[i]*obs, 0)
                        d = (left > 1e-9).sum(1, keepdims=True) if self.distinct else 1.
                        prob = categorical_reading(left, left.sum(1, keepdims=True), d,
                                                   above[1][:, block], self.kappa, np)
                        prior = np.maximum(frequency[start+i, block], 1e-12)
                        terms_block[i] = np.log(np.where(alone[i, :, None], prior, prob))@obs
                    terms.append(terms_block)
                elif counts.shape[1]:
                    code = codes[part]
                    before = counts[:, code].T
                    after = np.maximum(before-t[part], 0.)
                    total = np.maximum(totals-before, 0.)+after
                    d = distinct-(before > 1e-9).astype(int)+(after > 1e-9) if self.distinct else 1.
                    prior = above[1][:, block.start+code].T
                    probability = categorical_reading(after, total, d, prior, self.kappa, np)
                    fallback = frequency[np.arange(start, start+len(code)), block.start+code][:, None]
                    term = np.where(alone, np.log(np.maximum(fallback, 1e-12)), np.log(probability))
                    terms.append(term*present[part,None])
                else:
                    terms.append(np.zeros_like(mass))
                read.append(name in self.labels)
            terms = np.stack(terms, -1)
            if self.cross is not None and s is self.statistics:  # the features are read together
                terms[..., self.cov] = 0
                terms[..., self.cov[0]] = self._block_unseen(x[part], mask[part], t[part])
            observed = np.column_stack((mask[part], *[item[0][part] for item in categorical])) > 0
            read = np.asarray(read, bool)
            prior = np.log(np.maximum(mass, 1e-300))
            if self.labels and not structure:
                routed, _ = _softmax(prior+terms[..., ~read].sum(-1))
                scores[part] = np.log(np.maximum((routed*np.exp(terms[..., read].sum(-1))).sum(1), 1e-300))
            else:
                # Attribute j read from all the others: sum_k softmax(total - terms_j)_k exp(terms_jk)
                # = exp(LSE_k total - LSE_k(total - terms_j)), for every attribute at once.
                total = prior+terms.sum(-1)
                value = (_logsumexp(total, 1)[:, None]-_logsumexp(total[..., None]-terms, 1))*observed
                scores[part] = value.sum(1)/np.maximum(observed.sum(1), 1)
        return scores

    # ---- training --------------------------------------------------------------
    def _block(self, k, n):
        """Concepts of a block of the scan: a quarter of the specific concepts allowed, between 32 and 128.

        A table of at most 256 rows is not generalized in blocks: every row is a specific concept.
        """
        if self.bins:
            return int(self.bins)
        if n <= 256:
            return int(n)
        return int(min(128, max(32, 1 << int(np.log2(max(k//4, 1))))))

    def _path(self, data, weights, judged, rng, progress=None, restarts=None, start=None, specific=None):
        """From specific concepts to one concept; every step is judged and measured against the level above.

        The specific concepts come from one pass over the rows (acan/core/scan.py): blocks of rows
        generalized two by two where the reader loses least. ``specific`` gives them when the pass
        was made on more rows than ``data`` holds.
        """
        n = len(data[0])
        k = int(min(start or self.max_concepts, max(n//self.rows_per_concept, 1)))
        path = []
        # The blocks depend on the order of the rows: a small table is read in a few orders.
        for restart in range(1 if specific is not None else restarts or (self.restarts if n <= self.restart_rows else 1)):
            s = specific if specific is not None else scan(self, data, self._block(k, n), rng.permutation(n))
            s = self._em(s, data, weights, 10)
            judged_at, first_k = np.inf, len(s['C'])
            while True:
                k_now = len(s['C'])
                step = dict(k=k_now, restart=restart, data=data, statistics=s, pair_criterion=self.pair_criterion(s),
                            score=-np.inf, error=0.)
                # Judging is the costly part: among many concepts, a step is judged when one in ten is gone.
                if k_now <= 64 or k_now <= .9*judged_at:
                    scores, judged_at = self._score(s, data, weights, judged), k_now
                    step.update(score=float(scores.mean()), scores=scores, error=float(scores.std()/np.sqrt(len(scores))))
                path.append(step)
                self._tell('levels', concepts=k_now, start=first_k, search=1 if start is None else 2,
                           score=step['score'] if np.isfinite(step['score']) else None)
                if np.isfinite(step['score']) and progress is not None and progress(step) is False:
                    return path
                if len(s['C']) == 1:
                    break
                s = self._generalize(s, data, weights, max(int(len(s['C'])*self.merge_fraction), 1))
                s = self._em(s, data, weights, self.em_steps)
        return path

    @staticmethod
    def _hard(t):
        result = np.zeros_like(t)
        result[np.arange(len(t)), t.argmax(1)] = 1
        return result

    def _above(self, s, data, weights, hard=False):
        """What a level gives to every row: membership, centre, dispersion, frequencies and impurity."""
        t = self._route(s, data, weights)[0]
        if hard:                                                 # as a partition: every row in its concept
            t = self._hard(t)
            s = self._statistics(t, data)
        cm = np.maximum(s['Cm'], 1e-12)
        centre = s['U']/cm
        variance = unbiased(np.maximum(s['Q']/cm-centre**2, 0), s['Cm'], np)
        variance = np.where(s['Cm'] > 1, np.maximum(variance, CONSTANT), self._prior(s)[0])
        size = np.zeros_like(s['N'])
        for block in self._blocks():
            size[:, block] = s['N'][:, block].sum(1, keepdims=True)
        frequency = (s['N']+self.kappa*self._prior(s)[1])/(size+self.kappa)
        above = s['PG']/np.maximum(s['C'], 1e-12)[:, None]
        impurity = np.empty_like(above)
        for place, block in enumerate(self._blocks()):
            members = s['N'][:, block].sum(1)
            own = unbiased(1-((s['N'][:, block]/np.maximum(members, 1e-12)[:, None])**2).sum(1), members, np)
            impurity[:, place] = np.where(members > 1, own, above[:, place])
        return t, t@centre, t@variance, t@frequency, t@impurity

    def _context(self, s, data, weights, hard=False):
        """The table as the next level sees it: every row minus the centre of its concept, with its scale.

        The root does the same with the mean and the deviation of the table. A behaviour that
        shows up in several concepts of this level is, seen from below, one and the same.
        """
        _, centre, variance, frequency, impurity = self._above(s, data, weights, hard)
        return (data[0]-centre*data[1], data[1], data[2], variance, frequency, impurity)

    @staticmethod
    def _most_general(steps):
        """The cut with the largest pair criterion; among near ties (TIE), the one with fewest concepts."""
        top = max(step['pair_criterion'] for step in steps)
        return min((step for step in steps if step['pair_criterion'] >= top-TIE*abs(top)), key=lambda step: step['k'])

    def _tell(self, phase, **facts):
        """What is happening now, for whoever follows the learning: the phase and its quantities."""
        if self.report is not None:
            self.report(dict(phase=phase, **facts))

    def _roots(self, base):
        """Frequencies and impurities of the table, and the grid of its numbers: the scale of the root."""
        observed = np.zeros(self.encoder.width)
        for block in self._blocks():
            observed[block] = max(base[2][:, block].sum(), 1)
        self.root_p = np.maximum(base[2].sum(0)/observed, 1e-12)
        self.root_g = np.array([1-(self.root_p[block]**2).sum() for block in self._blocks()])
        self.resolution = self._resolution(base)

    def fit(self, frame, labels=(), encoder=None, progress=None):
        self.category_regions = {}
        self.observed_targets = {}
        if len(frame) > self.refine_rows:                        # a large table is read a chunk at a time
            step = self.chunk_rows
            self._total = len(frame)
            return self.fit_chunks(lambda: (frame[start:start+step] for start in range(0, len(frame), step)),
                                   labels, encoder, progress)
        self.labels = list(labels)
        self.encoder = encoder or Encoder().fit(frame)
        self.rows = len(frame)
        return self._learn(self.encoder.transform(frame), progress)

    def fit_chunks(self, read, labels=(), encoder=None, progress=None):
        """Learn from a table too large to hold: ``read()`` gives its rows as an iterator of frames.

        Three passes, each linear in the rows and with one chunk in memory:
        1. the scale of every attribute, and ``refine_rows`` rows drawn at random;
        2. the scan: every row enters a block once, and the blocks are generalized (acan/core/scan.py);
        3. the levels are found on the rows drawn, starting from the concepts of the scan, and in a
           last pass every row is added to the concepts of the working level: their sums are over
           the whole table.
        """
        self.category_regions = {}
        self.observed_targets = {}
        self.labels = list(labels)
        rng = np.random.default_rng(self.seed)
        self.encoder = encoder or Encoder().fit_chunks(read())
        drawn, keys, self.rows = None, np.zeros(0), 0
        total = getattr(self, '_total', None)                    # known when the table is held in memory
        join = lambda first, second: pd.concat((first, second)) if hasattr(first, 'iloc') else np.concatenate((first, second))
        take = lambda rows, index: rows.iloc[index] if hasattr(rows, 'iloc') else rows[index]
        for frame in read():                                     # the rows with the smallest random keys: a uniform draw
            self.rows += len(frame)
            key = rng.random(len(frame))
            drawn = frame if drawn is None else join(drawn, frame)
            keys = np.concatenate((keys, key))
            if len(keys) > self.refine_rows:
                keep = np.sort(np.argpartition(keys, self.refine_rows)[:self.refine_rows])
                drawn, keys = take(drawn, keep), keys[keep]
            self._tell('scale', rows=self.rows, total=total)
        sample = self.encoder.transform(drawn)
        self._roots(sample)
        k = int(min(self.max_concepts, max(self.rows//self.rows_per_concept, 1)))
        blocks = Scan(self, self._block(k, self.rows))
        # Chunks are independent: their blocks are added afterwards. The processes are started afresh
        # (spawn), so a script that asks for them must protect its entry point with ``if __name__ == '__main__'``.
        if self.workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                waiting = []
                for frame in read():                             # a few chunks in flight, in the order of the table
                    waiting.append(pool.submit(scan_part, (self, blocks.bins, frame)))
                    if len(waiting) >= 2*self.workers:
                        blocks.absorb(waiting.pop(0).result())
                        self._tell('scan', rows=blocks.rows, total=self.rows, concepts=blocks.concepts())
                for part in waiting:
                    blocks.absorb(part.result())
                    self._tell('scan', rows=blocks.rows, total=self.rows, concepts=blocks.concepts())
        else:
            for frame in read():
                blocks.push(self.encoder.transform(frame))
                self._tell('scan', rows=blocks.rows, total=self.rows, concepts=blocks.concepts())
        self._learn(sample, progress, specific=blocks.result())
        if self.rows > len(drawn):
            self._finish(read)
        return self

    def _finish(self, read):
        """Every row of the table is added to the concepts of the working level, as it recognizes them."""
        total, cross, done = None, None, 0
        for frame in read():
            done += len(frame)
            self._tell('finish', rows=done, total=self.rows, concepts=len(self.statistics['C']))
            data = self._seen(self.encoder.transform(frame))[0]
            t = self._route(self.statistics, data, self.train_weights)[0]
            part = self._statistics(t, data)
            total = part if total is None else {key: total[key]+part[key] for key in part}
            if self.cross is not None:
                products = self._cross(t, data)
                cross = products if cross is None else cross+products
        self.statistics = total
        if self.cross is not None:
            self.cross = cross
        self.levels[-1]['statistics'] = self.statistics

    def _learn(self, base, progress=None, specific=None):
        n = len(base[0])
        self._roots(base)
        features = len(self.encoder.numeric)+len(self.encoder.categorical)-len(self.labels)
        weights = self.train_weights = self._weights(float(max(features, 1)))   # targets weigh as much as all features
        rng = np.random.default_rng(self.seed)
        judged = np.sort(rng.choice(n, min(n, self.judged_rows), replace=False))
        root = base

        # Level 1: the general concepts of the whole table, on the scale of the root.
        general = self._path(root, weights, judged, rng, progress, specific=specific)
        single = min(general, key=lambda step: step['k'])
        first = self._most_general(general)
        if first['pair_criterion'] <= 1e-9:
            first = single                                       # nothing is more alike than the average pair
        particular, local = [], root
        if first['k'] > 1:
            # The same search again on the table as the general concepts give it: centred and scaled
            # by the concept of each row, recognized from what is observed when the model is used.
            # What is distinct only inside a general concept becomes visible, and what is the same
            # in several general concepts becomes one concept.
            local = self._context(first['statistics'], root, self._weights(0.))
            # Behaviours shared by several general concepts are few by nature: the search starts from few.
            particular = self._path(local, weights, judged, rng, restarts=1, start=self.shared_start)
        # Working level: the most general step that reads unseen rows as well as the best one.
        every = general+particular
        best = max(every, key=lambda step: step['score'])
        # Ties are ties: rounding must not make a finer level look better than a general one.
        # Most general: fewest nodes in the graph. A level seen through the general concepts needs them too.
        nodes = lambda step: step['k']+(first['k'] if step['data'] is not root else 0)
        working = min((step for step in every if step['score'] >= best['score']-best['error']-1e-9),
                      key=lambda step: (nodes(step), -step['score']))
        local_work = any(step is working for step in particular)
        data, pool = (local, particular) if local_work else (root, general)
        # Levels between the general concepts and the working level: each is the cut that
        # maximizes the pair criterion against the level above.
        # Seen through the general concepts a level may well have fewer concepts than they are:
        # the behaviours that several general concepts share are counted once.
        levels, current = [], None
        if local_work or first['k'] < working['k']:
            levels, current = [first], first
        while current is not None and len(levels) < self.max_levels-1:
            # Candidates are the steps of the generalization that produced the working level:
            # along one generalization the pair criterion has a single interior maximum.
            finer = current['k'] if current['data'] is data else 1
            between = [step for step in pool if step['restart'] == working['restart']
                       and finer < step['k'] < working['k']]
            if not between:
                break
            # Measured on partitions: every row counts in its concept, and takes its scale from its
            # concept of the current level. A candidate equal to the current level measures zero.
            against = self._context(current['statistics'], current['data'], weights, hard=True)
            for step in between:
                membership = self._hard(self._route(step['statistics'], data, weights)[0])
                step['pair_criterion'] = self.pair_criterion(self._statistics(membership, against))
            following = self._most_general(between)
            # What a level adds is measured in units of what the general concepts add:
            # within the tolerance of ties, there is no further level.
            if following['pair_criterion'] <= TIE*max(first['pair_criterion'], 0)+1e-9:
                break
            # The pair criterion cannot stop by itself: a level is kept only if, as a description
            # of the table, it reads unseen rows better than the level above.
            for step in (current, following):
                if 'structure' not in step:
                    step['structure'] = self._score(step['statistics'], step['data'], weights, judged, structure=True)
            gain = following['structure']-current['structure']
            if gain.mean() <= gain.std()/np.sqrt(len(gain))+1e-9:
                break
            levels.append(following)
            current = following
        self.statistics = self._em(working['statistics'], data, weights, 20)
        levels = [level for level in levels if level['data'] is not data or level['k'] < len(self.statistics['C'])]
        # The working level is chosen by judgment, so its size says nothing against the level above:
        # what counts is whether the pairs that share one of its concepts are more alike than they
        # were in the level above. If they are not (a tie), the level above is not a separate level;
        # without a target nothing else justifies the working level, and the level above takes its place.
        while levels:
            final = self._hard(self._route(self.statistics, data, weights)[0])
            against = self._context(levels[-1]['statistics'], levels[-1]['data'], weights, hard=True)
            if self.pair_criterion(self._statistics(final, against), within=True) > TIE:
                break
            if not self.labels:
                working = levels.pop()
                data = working['data']
                self.statistics = self._em(working['statistics'], data, weights, 20)
            elif len(levels) > 1:
                levels.pop()
            elif data is not root:                               # level 1 again, seen through itself
                working, data = first, root
                self.statistics = self._em(working['statistics'], data, weights, 20)
                levels.pop()
            else:
                break
        levels.append(working)
        self.cross, self.cov = None, np.zeros(0, int)
        if data is root:
            self._together(data, weights, judged)
        chosen = levels[-1]
        self.path = [dict(k=step['k'], score=step['score'], restart=step['restart']) for step in general if np.isfinite(step['score'])]
        self.selection = dict(best_k=best['k'], best_score=best['score'], standard_error=best['error'],
                              chosen_k=chosen['k'], chosen_score=chosen['score'])
        # The graph: levels, the dense matrices between consecutive levels, edges inside the last level.
        self.above = first['statistics'] if data is not root else None      # the working level reads rows through level 1
        routed = [self._route(level['statistics'], level['data'], weights)[0] for level in levels[:-1]]
        last = self._route(self.statistics, data, weights)[0]
        routed.append(last)
        self.levels, above = [], root
        for place, (level, membership) in enumerate(zip(levels, routed)):
            mass = membership.sum(0)
            statistics = self.statistics if place == len(levels)-1 else level['statistics']
            parents = None
            if place:                                            # P(concept of the level above | concept): columns sum to one
                parents = (routed[place-1].T@membership)/np.maximum(mass, 1e-12)
            criterion = self.pair_criterion(self._statistics(self._hard(membership), above))
            above = self._context(statistics, level['data'], weights, hard=True)
            # ``table``: the same concepts summed over the rows as they are in the table, whatever
            # level they are seen through: what a person reads when inspecting the concept.
            self.levels.append(dict(depth=place+1, concepts=int(membership.shape[1]), pair_criterion=criterion,
                                    score=level['score'], mass=mass/n, parents=parents, statistics=statistics,
                                    table=self._statistics(membership, base)))
        first_edge, second_edge, weight = self._edges(last)
        self.edges = [(int(i), int(j), float(w/n)) for i, j, w in zip(first_edge, second_edge, weight)]
        self._judge_reading(data, weights, judged)
        self.final_score = float(self._score(self.statistics, data, weights, judged).mean())
        return self

    def _judge_reading(self, data, weights, judged):
        """The weight of the level above in the reading, chosen by the judgment once the concepts are found.

        The concepts are searched with kappa pseudo-members; then the reading weighs the level above by
        kappa times the values a concept has (categorical_reading), with the kappa that reads the judged
        rows best. The same rule, chosen the same way, reads the text (acan/lm/concept_lm.py).
        """
        self.distinct = True
        scores = []
        for value in KAPPA_GRID:
            self.kappa = value
            scores.append(float(self._score(self.statistics, data, weights, judged).mean()))
        self.kappa = KAPPA_GRID[int(np.argmax(scores))]

    # ---- use -------------------------------------------------------------------
    @property
    def concepts(self):
        return len(self.statistics['C'])

    def _seen(self, data):
        """Rows as the working level sees them, and the centre the level above gives to each."""
        if self.above is None:
            return data, np.zeros_like(data[0])
        centre = self._above(self.above, data, self._weights(0.))[1]
        return self._context(self.above, data, self._weights(0.)), centre

    def assignments(self, frame, observed_only=True):
        data = self._seen(self.encoder.transform(frame))[0]
        weights = self._weights(0.) if observed_only else self.train_weights
        return self._route(self.statistics, data, weights)[0]

    def fit_subconcepts(self, observations):
        """Fit the same CP procedure inside each parent and return a sparse hard router.

        Only predictor columns enter the local geometry. The fitted parent supplies the
        coordinate system and routing for later rows. Each branch is judged on its own rows.
        """
        from .subconcepts import LocalConcepts

        features = [c for c in (*self.encoder.numeric, *self.encoder.categorical) if c not in self.labels]
        return LocalConcepts(self).fit(observations[features])

    def fit_observed_target(self, label, categories, chunks, validation):
        """Fit a new categorical column after fitting the geometry on its observed predictors.

        Batches give predictor rows and integer indices into categories. The same memberships read
        and update the target. Validation chooses its smoothing without changing the geometry.
        """
        if label in self.encoder.numeric or label in self.encoder.categorical:
            raise ValueError("La colonna da aggiungere deve essere esterna ai predittori addestrati.")
        categories = list(categories)
        if not categories or len(set(categories)) != len(categories):
            raise ValueError("Serve un dizionario di categorie distinte.")
        counts = np.zeros((self.concepts, len(categories)))

        def encode(rows, targets):
            targets = np.asarray(targets)
            if targets.ndim != 1 or len(targets) != len(rows) or not np.issubdtype(targets.dtype, np.integer):
                raise ValueError("Serve una categoria codificata per ogni riga.")
            if np.any((targets < 0) | (targets >= len(categories))):
                raise ValueError("Categoria fuori dal dizionario.")
            return self.assignments(rows), targets

        for rows, targets in chunks:
            if not len(rows):
                continue
            shares, targets = encode(rows, targets)
            order = np.argsort(targets, kind='stable')
            values, starts = np.unique(targets[order], return_index=True)
            counts[:, values] += np.add.reduceat(shares[order], starts, axis=0).T
        if counts.sum() == 0:
            raise ValueError("Servono righe di training per il target.")
        mass = counts.sum(1, keepdims=True)
        distinct = (counts > 1e-9).sum(1, keepdims=True)
        frequency = counts.sum(0) + 1
        prior = frequency / frequency.sum()
        scores = np.zeros(len(KAPPA_GRID))
        n = 0
        for rows, targets in validation:
            shares, targets = encode(rows, targets)
            n += len(targets)
            for j, kappa in enumerate(KAPPA_GRID):
                q = categorical_reading(counts[:, targets], mass, distinct, prior[targets], kappa, np)
                p = np.einsum('ij,ji->i', shares, q)
                scores[j] -= np.log(np.maximum(p, 1e-300)).sum()
        if not n:
            raise ValueError("Servono righe di validazione per il giudizio.")
        kappa = KAPPA_GRID[int(scores.argmin())]
        if not hasattr(self, 'observed_targets'):
            self.observed_targets = {}
        self.observed_targets[label] = dict(
            categories=categories,
            kappa=kappa,
            probability=categorical_reading(counts, mass, distinct, prior, kappa, np),
        )
        getattr(self, 'category_regions', {}).pop(label, None)
        return self

    def refine_category(self, label):
        """Extend a categorical target with CP regions observed in additional table columns.

        A missing region inherits the ordinary prediction. The label never participates in routing.
        The caller proposes cuts and judges their categorical sums on separate rows.
        """
        from acan.core.conditional import ConditionalReading

        fitted_head = label in getattr(self, 'observed_targets', {})
        if not fitted_head and (label not in self.labels or label not in self.encoder.categorical):
            raise ValueError("Il raffinamento richiede un target categorico addestrato.")
        if not hasattr(self, 'category_regions'):
            self.category_regions = {}
        self.category_regions.pop(label, None)
        base, categories = self.predict_proba(pd.DataFrame(index=[0]), label)
        reading = ConditionalReading(len(categories))
        reading.prior = base[0] / base[0].sum()
        self.category_regions[label] = reading
        return reading

    def fit_category_pair(self, label, parent, child, build, judgment):
        """Judge a categorical child inside each parent on separate labeled tabular rows."""
        from copy import copy
        from .category_pairs import PairColumns

        for name in (parent, child):
            if name not in self.encoder.categorical or name in self.labels or name == label:
                raise ValueError("Servono due predittori categorici già appresi.")
        if parent == child:
            raise ValueError("I due predittori devono essere distinti.")
        feature = PairColumns(parent, child, self.encoder)
        base = copy(self)
        base.category_regions = dict(getattr(self, 'category_regions', {}))
        if label in base.category_regions:
            reading = copy(base.category_regions[label])
            reading.levels = [(f, table, k) for f, table, k in reading.levels
                              if not isinstance(f, PairColumns) or f.names != feature.names]
            base.category_regions[label] = reading
        else:
            reading = base.refine_category(label)
        _, categories = base.predict_proba(build.iloc[:1], label)
        dictionary = {value:i for i,value in enumerate(categories)}
        def parts(frame):
            values = frame[label] if label in getattr(self, 'observed_targets', {}) else frame[label].astype('string')
            ids = values.map(dictionary)
            eligible = ids.notna()
            return (*feature.parts(frame.loc[eligible]), ids[eligible].to_numpy(dtype=int)), eligible
        train, _ = parts(build)
        valid, keep = parts(judgment)
        if not len(train[-1]) or not len(valid[-1]):
            raise ValueError("Servono esempi di training e giudizio con target noto.")
        table, report = reading.subconcepts(train, valid, train)
        prob, _ = base.predict_proba(judgment.loc[keep], label)
        keys = feature.keys(judgment.loc[keep])
        _, choice = reading.propose(table, keys, valid[-1], prob[np.arange(len(keys)), valid[-1]])
        if choice:
            reading.levels.append((feature, table, choice[1]))
        self.category_regions = {**getattr(self, 'category_regions', {}), label: reading}
        return {**report, 'applied': choice is not None}

    def predict_proba(self, frame, label):
        head = getattr(self, 'observed_targets', {}).get(label)
        observed = frame.drop(columns=[*self.labels, label], errors='ignore')
        t = self.assignments(observed)
        if head is None:
            _, _, log_p = self._parameters(self.statistics)
            probability = t @ np.exp(log_p[:, self.encoder.block(label)])
            categories = self.encoder.categories[label]
        else:
            probability = t @ head['probability']
            categories = head['categories']
        reading = getattr(self, 'category_regions', {}).get(label)
        if reading is not None:
            probability = reading.apply(observed, probability)
        return probability, categories

    def predict(self, frame, label):
        if label in self.encoder.numeric:
            frame = frame.drop(columns=[c for c in self.labels if c in frame])
            data, centre = self._seen(self.encoder.transform(frame))
            t = self._route(self.statistics, data, self._weights(0.))[0]
            j = self.encoder.numeric.index(label)
            g, _, _ = self._parameters(self.statistics)
            return (centre[:, j]+t@g[:, j])*self.encoder.std[j]+self.encoder.mean[j]
        probability, classes = self.predict_proba(frame, label)
        return np.asarray(classes)[probability.argmax(1)]

    def explained_variance(self, frame):
        """eta^2 per attribute (numeric) or explained Gini impurity (categorical); CP = sqrt(mean)."""
        data = self.encoder.transform(frame)
        s = self._statistics(self._route(self.statistics, self._seen(data)[0], self.train_weights)[0], data)
        f = s['C']/s['C'].sum()
        cm = np.maximum(s['Cm'], 1e-12)
        g = s['U']/cm
        fm = s['Cm']/np.maximum(s['Cm'].sum(0), 1e-12)
        mu = (fm*g).sum(0)
        total = np.maximum(s['Q'].sum(0)/np.maximum(s['Cm'].sum(0), 1e-12)-mu*mu, 1e-12)
        result = {c: float((fm[:, j]*(g[:, j]-mu[j])**2).sum()/total[j]) for j, c in enumerate(self.encoder.numeric)}
        for c in self.encoder.categorical:
            block = self.encoder.block(c)
            counts = s['N'][:, block]
            size = np.maximum(counts.sum(1, keepdims=True), 1e-12)
            p, root = counts/size, counts.sum(0)/max(counts.sum(), 1e-12)
            impurity = 1-(root*root).sum()
            weight = size[:, 0]/size.sum()
            result[c] = float((weight[:, None]*(p-root)**2).sum()/impurity) if impurity > 1e-12 else 0.
        result['CP'] = float(np.sqrt(np.mean([result[c] for c in self.encoder.numeric+self.encoder.categorical])))
        return result


ConceptMixture = ConceptModel                                   # earlier name
