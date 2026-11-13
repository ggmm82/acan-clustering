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

from acan.concept_algebra import LOG_2PI, centre_variance, pair_mass, reader_cost, reader_radius, unbiased

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


def _softmax(logits):
    top = logits.max(axis=1, keepdims=True)
    e = np.exp(logits-top)
    total = e.sum(axis=1, keepdims=True)
    return e/total, (top+np.log(total))[:, 0]


class ConceptModel:
    max_levels = 8                                              # capacity limit, not a requested depth

    def __init__(self, max_concepts=1024, rows_per_concept=4, kappa=1., seed=0,
                 judged_rows=2000, merge_fraction=.05, em_steps=3, restarts=5, restart_rows=5000,
                 covariance=True, covariance_attributes=128):
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
        size = np.zeros_like(s['N'])
        for block in self._blocks():
            size[..., block] = s['N'][..., block].sum(-1, keepdims=True)
        log_p = np.log((s['N']+self.kappa*prior_p)/(size+self.kappa))
        return g, v, log_p

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
        cost = np.concatenate([self._pair_costs(s, first[start:start+100000], second[start:start+100000])
                               for start in range(0, len(first), 100000)])
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
        size = max(int(4e6//(k*max(x.shape[1]+onehot.shape[1], 1))), 1)
        for start in range(0, len(x), size):
            part = slice(start, start+size)
            r = t[part][:, :, None]
            xs, ms, hs = x[part][:, None], mask[part][:, None], onehot[part][:, None]
            rest = {key: s[key][None]-r*value for key, value in (('Cm', ms), ('U', xs), ('Q', xs*xs), ('N', hs))}
            rest = {key: np.maximum(value, 0) if key != 'U' else value for key, value in rest.items()}
            rest['C'] = np.maximum(s['C'][None]-t[part], 0)
            alone = rest['C'] < 1e-9                             # nobody else: the row's own level above
            # The level above is context, not something the row says about itself: it stays as it is.
            g, v, log_p = self._parameters(rest, above)
            v = np.where(rest['Cm'] > 1e-9, v, scale[part][:, None])
            terms, read = [], []                                 # one log-density per attribute: rows x k
            for j, name in enumerate(encoder.numeric):
                terms.append(-.5*((xs[..., j]-g[..., j])**2/v[..., j]+np.log(v[..., j])+LOG_2PI)*ms[..., j])
                read.append(name in self.labels)
            for name, block in zip(encoder.categorical, self._blocks()):
                fallback = np.log(np.maximum(frequency[part][:, None, block], 1e-12))
                value = np.where(alone[..., None], fallback, log_p[..., block])
                terms.append((value*hs[..., block]).sum(-1))
                read.append(name in self.labels)
            terms = np.stack(terms, -1)
            if self.cross is not None and s is self.statistics:  # the features are read together
                terms[..., self.cov] = 0
                terms[..., self.cov[0]] = self._block_unseen(x[part], mask[part], t[part])
            observed = np.concatenate((ms[:, 0], *[hs[:, 0, block].sum(-1, keepdims=True) for block in self._blocks()]), -1) > 0
            read = np.asarray(read, bool)
            prior = np.log(np.maximum(rest['C'], 1e-300))
            if self.labels and not structure:
                routed, _ = _softmax(prior+terms[..., ~read].sum(-1))
                scores[part] = np.log(np.maximum((routed*np.exp(terms[..., read].sum(-1))).sum(1), 1e-300))
            else:
                total = prior+terms.sum(-1)
                value = np.zeros(len(total))
                for j in range(terms.shape[-1]):                 # read attribute j from all the others
                    routed, _ = _softmax(total-terms[..., j])
                    value += observed[:, j]*np.log(np.maximum((routed*np.exp(terms[..., j])).sum(1), 1e-300))
                scores[part] = value/np.maximum(observed.sum(1), 1)
        return scores

    # ---- training --------------------------------------------------------------
    def _path(self, data, weights, judged, rng, progress=None, restarts=None):
        """From specific concepts to one concept; every step is judged and measured against the level above."""
        n = len(data[0])
        k = int(min(self.max_concepts, max(n//self.rows_per_concept, 1)))
        path = []
        # Generalization is a greedy search: repeat it from other specific concepts.
        for restart in range(restarts or (self.restarts if n <= self.restart_rows else 1)):
            seeds = rng.choice(n, k, replace=False)
            seed_t = np.zeros((n, k))
            seed_t[seeds, np.arange(k)] = 1
            s = self._em(self._statistics(seed_t, data), data, weights, 10)
            judged_at = np.inf
            while True:
                k_now = len(s['C'])
                step = dict(k=k_now, restart=restart, data=data, statistics=s, pair_criterion=self.pair_criterion(s),
                            score=-np.inf, error=0.)
                # Judging is the costly part: among many concepts, a step is judged when one in ten is gone.
                if k_now <= 64 or k_now <= .9*judged_at:
                    scores, judged_at = self._score(s, data, weights, judged), k_now
                    step.update(score=float(scores.mean()), scores=scores, error=float(scores.std()/np.sqrt(len(scores))))
                path.append(step)
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
        features = len(self.encoder.numeric)+len(self.encoder.categorical)-len(self.labels)
        weights = self.train_weights = self._weights(float(max(features, 1)))   # targets weigh as much as all features
        rng = np.random.default_rng(self.seed)
        judged = np.sort(rng.choice(n, min(n, self.judged_rows), replace=False))
        root = base

        # Level 1: the general concepts of the whole table, on the scale of the root.
        general = self._path(root, weights, judged, rng, progress)
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
            particular = self._path(local, weights, judged, rng, restarts=1)
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
            self.levels.append(dict(depth=place+1, concepts=int(membership.shape[1]), pair_criterion=criterion,
                                    score=level['score'], mass=mass/n, parents=parents, statistics=statistics))
        first_edge, second_edge, weight = self._edges(last)
        self.edges = [(int(i), int(j), float(w/n)) for i, j, w in zip(first_edge, second_edge, weight)]
        self.final_score = float(self._score(self.statistics, data, weights, judged).mean())
        return self

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

    def predict_proba(self, frame, label):
        t = self.assignments(frame.drop(columns=[c for c in self.labels if c in frame]))
        _, _, log_p = self._parameters(self.statistics)
        return t@np.exp(log_p[:, self.encoder.block(label)]), self.encoder.categories[label]

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
