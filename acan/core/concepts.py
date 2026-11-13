"""Concept mixture with one predictive criterion (research profile ``cp-v9``).

See docs/MATHEMATICAL-FOUNDATION.md, part B. A concept is a product density over
attributes, stored as additive sufficient statistics. Three operations exist:

* routing      posterior of the concepts given the observed attributes;
* generalizing summing the statistics of two concepts (the thesis operator);
* scoring      log-probability of what must be predicted (the target when
               there is one, every attribute otherwise), computed for each
               row by concepts from which that row has been subtracted.

Training starts from many specific concepts and generalizes bottom-up, always
merging the pair whose merge costs least description length. The retained
level is the most general one whose score is within one standard
error of the best. No concept count, purity or similarity threshold is given.
"""
import numpy as np
import pandas as pd

from acan.concept_algebra import centre_variance, description_length

LOG_2PI = float(np.log(2*np.pi))


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


class ConceptMixture:
    def __init__(self, max_concepts=1024, rows_per_concept=4, kappa=1., seed=0,
                 judged_rows=2000, merge_fraction=.05, em_steps=3,
                 restarts=5, restart_rows=5000, judged_fraction=1., covariance=False):
        self.max_concepts, self.rows_per_concept = max_concepts, rows_per_concept
        self.kappa, self.seed, self.judged_rows = kappa, seed, judged_rows
        self.merge_fraction, self.em_steps = merge_fraction, em_steps
        self.restarts, self.restart_rows = restarts, restart_rows
        self.judged_fraction = judged_fraction                  # share of the rows that choose the level
        self.covariance = covariance                            # numeric features of a concept vary together
        self.cov = np.zeros(0, int)                             # their positions among the numeric attributes

    # ---- sufficient statistics -------------------------------------------------
    def _statistics(self, t, data):
        x, mask, onehot = data
        s = dict(C=t.sum(0), Cm=t.T@mask, U=t.T@x, Q=t.T@(x*x), N=t.T@onehot)
        if len(self.cov):                                       # cross products: still a sum over the members
            xc = x[:, self.cov]
            p = len(self.cov)
            s['S'] = (t.T@(xc[:, :, None]*xc[:, None, :]).reshape(len(xc), p*p)).reshape(-1, p, p)
        return s

    def _covariance(self, c, u, cross):
        """Centre and covariance of the correlated block, shrunk to the unit diagonal of its level."""
        g = u/np.maximum(c, 1e-12)[..., None]
        scatter = cross-c[..., None, None]*g[..., :, None]*g[..., None, :]
        return g, (scatter+self.kappa*np.eye(u.shape[-1]))/(c+self.kappa)[..., None, None]

    def _block_density(self, s, x, mask):
        """Log-density of the correlated block; a missing attribute is dropped from the Gaussian exactly."""
        g, sigma = self._covariance(s['C'], s['U'][:, self.cov], s['S'])
        xc, mc = x[:, self.cov], mask[:, self.cov] > 0
        out = np.zeros((len(xc), len(g)))
        patterns, which = np.unique(mc, axis=0, return_inverse=True)
        for index, pattern in enumerate(patterns):
            rows = np.flatnonzero(which.reshape(-1) == index)
            if not pattern.any():
                continue
            sub = sigma[:, pattern][:, :, pattern]
            inverse, logdet = np.linalg.inv(sub), np.linalg.slogdet(sub)[1]
            for start in range(0, len(g), 32):
                part = slice(start, start+32)
                d = xc[rows][:, None, pattern]-g[None, part][:, :, pattern]
                distance = np.einsum('nkp,kpq,nkq->nk', d, inverse[part], d)
                out[rows, part] = -.5*(distance+logdet[part]+pattern.sum()*LOG_2PI)
        return out

    def _parameters(self, s):
        """Posterior means; variances and frequencies shrink to the root by kappa."""
        g, v = centre_variance(s['Cm'], s['U'], s['Q'], 1., self.kappa, np)   # root variance is one
        ca = np.zeros_like(s['N'])
        for c in self.encoder.categorical:
            block = self.encoder.block(c)
            ca[:, block] = s['N'][:, block].sum(1, keepdims=True)
        log_p = np.log((s['N']+self.kappa*self.root_p)/(ca+self.kappa))
        return g, v, log_p

    def _log_density(self, s, data, weights):
        x, mask, onehot = data
        wn, wc = weights
        wn = self._diagonal(wn)
        g, v, log_p = self._parameters(s)
        a = 1/v
        numeric = -.5*(((x*x)*wn)@a.T-2*(x*wn)@(g*a).T+(mask*wn)@(g*g*a+np.log(v)+LOG_2PI).T)
        if len(self.cov):
            numeric = numeric+self._block_density(s, x, mask)
        return numeric+(onehot*wc)@log_p.T

    def _diagonal(self, wn):
        """Weights of the attributes still treated one by one."""
        if not len(self.cov):
            return wn
        wn = wn.copy()
        wn[self.cov] = 0
        return wn

    def _block_length(self, c, u, cross):
        """Code length of the correlated block: half the mass times the log-volume."""
        return .5*c*np.linalg.slogdet(self._covariance(c, u, cross)[1])[1]

    def _route(self, s, data, weights):
        prior = np.log(np.maximum(s['C'], 1e-300)/s['C'].sum())
        return _softmax(self._log_density(s, data, weights)+prior)

    # ---- generalization --------------------------------------------------------
    def _merge_costs(self, s, weights, rows=None):
        """Increase of description length when two concepts are summed (k x k, or the given rows x k)."""
        wn, wc = weights
        wn = self._diagonal(wn)
        g, v, log_p = self._parameters(s)
        k = len(s['C'])
        own = description_length(s['Cm'], v, s['C'], np, wn)-(s['N']*log_p*wc).sum(1)
        if len(self.cov):
            own = own+self._block_length(s['C'], s['U'][:, self.cov], s['S'])
        index = np.arange(k) if rows is None else np.asarray(rows)
        cost = np.empty((len(index), k))
        for start in range(0, len(index), 64):
            part = index[start:start+64]
            merged = {key: s[key][part, None]+s[key][None] for key in ('Cm', 'U', 'Q', 'N')}
            cm = merged['Cm']
            _, vm = centre_variance(cm, merged['U'], merged['Q'], 1., self.kappa, np)
            ca = np.zeros_like(merged['N'])
            for c in self.encoder.categorical:
                block = self.encoder.block(c)
                ca[..., block] = merged['N'][..., block].sum(-1, keepdims=True)
            lp = np.log((merged['N']+self.kappa*self.root_p)/(ca+self.kappa))
            c = s['C'][part, None]+s['C'][None]
            together = description_length(cm, vm, c, np, wn)-(merged['N']*lp*wc).sum(-1)
            if len(self.cov):
                together = together+self._block_length(c, merged['U'][..., self.cov], s['S'][part, None]+s['S'][None])
            cost[start:start+64] = together-own[part, None]-own[None]
        cost[np.arange(len(index)), index] = np.inf
        return cost

    def _generalize(self, s, weights, count):
        cost = self._merge_costs(s, weights)
        k = len(cost)
        upper = np.triu_indices(k, 1)
        order = np.argsort(cost[upper])
        used, pairs = np.zeros(k, bool), []
        for position in order:
            i, j = upper[0][position], upper[1][position]
            if not used[i] and not used[j]:
                used[i] = used[j] = True
                pairs.append((i, j, cost[i, j]))
                if len(pairs) == count:
                    break
        keep = np.ones(k, bool)
        merged = {key: value.copy() for key, value in s.items()}
        for i, j, _ in pairs:
            for key in merged:
                merged[key][i] += merged[key][j]
            keep[j] = False
        return {key: value[keep] for key, value in merged.items()}, [float(c) for _, _, c in pairs]

    def _em(self, s, data, weights, steps):
        for _ in range(steps):
            t, _ = self._route(s, data, weights)
            s = self._statistics(t, data)
            alive = s['C'] > 1e-8
            if not alive.all():
                s = {key: value[alive] for key, value in s.items()}
        return s

    # ---- criterion -------------------------------------------------------------
    def _weights(self, target_weight):
        wn = np.ones(len(self.encoder.numeric))
        wc = np.ones(self.encoder.width)
        for label in self.labels:
            if label in self.encoder.numeric:
                wn[self.encoder.numeric.index(label)] = target_weight
            else:
                wc[self.encoder.block(label)] = target_weight
        return wn, wc

    def _score(self, s, data, weights, rows):
        """Log-probability of what each row must predict, from concepts that never saw that row.

        The row's own contribution is subtracted from the additive statistics
        before it is routed and read: every observation is judged by evidence
        from the others, as a causal prefix judges the next token.
        """
        x, mask, onehot = (part[rows] for part in data)
        t = self._route(s, (x, mask, onehot), weights)[0]
        k, encoder = len(s['C']), self.encoder
        numeric_targets = np.array([c in self.labels for c in encoder.numeric], bool)
        one_by_one = ~numeric_targets
        one_by_one[self.cov] = False
        if len(self.cov):
            p = len(self.cov)
            block = -2*self._block_density(s, x, mask)             # distance + log-volume + p log 2 pi, all members included
            logdet = np.linalg.slogdet(self._covariance(s['C'], s['U'][:, self.cov], s['S'])[1])[1]
            distance = block-logdet-p*LOG_2PI
            norm = (x[:, self.cov]**2).sum(1)
        scores = np.empty(len(x))
        size = max(int(4e6//(k*max(x.shape[1]+onehot.shape[1], 1))), 1)
        for start in range(0, len(x), size):
            part = slice(start, start+size)
            r = t[part][:, :, None]
            xs, ms, hs = x[part][:, None], mask[part][:, None], onehot[part][:, None]
            cm = np.maximum(s['Cm'][None]-r*ms, 0)
            u, q = s['U'][None]-r*xs, s['Q'][None]-r*xs*xs
            g, v = centre_variance(cm, u, q, 1., self.kappa, np)
            numeric = -.5*((xs-g)**2/v+np.log(v)+LOG_2PI)*ms
            counts = np.maximum(s['N'][None]-r*hs, 0)
            prior = np.log(np.maximum(s['C'][None]-t[part], 1e-300))
            feature = prior+numeric[:, :, one_by_one].sum(-1)
            if len(self.cov):
                # Removing a member is a rank-one change of the scatter: closed form in the distance.
                c, own = s['C'][None], t[part]
                rest = np.maximum(c-own, 0)
                safe = np.maximum(rest, 1e-12)
                q = distance[part]/(c+self.kappa)
                left = np.maximum(1-own*c/safe*q, 1e-12)
                away = (c/safe)**2*(rest+self.kappa)*q/left
                volume = logdet[None]+p*np.log(c+self.kappa)+np.log(left)-p*np.log(rest+self.kappa)
                alone = rest < 1e-9                              # nobody else: the level's own prior
                away = np.where(alone, norm[part, None], away)
                volume = np.where(alone, 0., volume)
                feature = feature-.5*(away+volume+p*LOG_2PI)
            target = numeric[:, :, numeric_targets].sum(-1)
            for c in encoder.categorical:
                block = encoder.block(c)
                total = counts[:, :, block].sum(-1, keepdims=True)
                log_p = np.log((counts[:, :, block]+self.kappa*self.root_p[block])/(total+self.kappa))
                value = (log_p*hs[:, :, block]).sum(-1)
                if c in self.labels:
                    target = target+value
                else:
                    feature = feature+value
            if self.labels:
                routed, _ = _softmax(feature)
                scores[part] = np.log(np.maximum((routed*np.exp(target)).sum(1), 1e-300))
            else:
                scores[part] = _softmax(feature+target)[1]-np.log(max(len(data[0])-1, 1))
        return scores

    # ---- training --------------------------------------------------------------
    def fit(self, frame, labels=(), encoder=None, progress=None):
        self.labels = list(labels)
        self.encoder = encoder or Encoder().fit(frame)
        data = self.encoder.transform(frame)
        n = len(frame)
        observed = np.zeros(self.encoder.width)
        for c in self.encoder.categorical:
            block = self.encoder.block(c)
            observed[block] = max(data[2][:, block].sum(), 1)
        self.root_p = np.maximum(data[2].sum(0)/observed, 1e-12)
        features = len(self.encoder.numeric)+len(self.encoder.categorical)-len(self.labels)
        block = np.array([j for j, name in enumerate(self.encoder.numeric) if name not in self.labels], int)
        complete = len(block) and bool((data[1][:, block] > 0).all())
        self.cov = block if self.covariance and len(block) > 1 and complete else np.zeros(0, int)
        weights = self._weights(float(max(features, 1)))        # targets weigh as much as all features
        rng = np.random.default_rng(self.seed)
        judged = np.sort(rng.choice(n, max(min(int(round(n*self.judged_fraction)), self.judged_rows), 1), replace=False))
        self.judged = judged
        k = int(min(self.max_concepts, max(n//self.rows_per_concept, 1)))
        path = []
        # Generalization is a greedy search: repeat it from other specific
        # concepts and let the same criterion choose among everything visited.
        for restart in range(self.restarts if n <= self.restart_rows else 1):
            seeds = rng.choice(n, k, replace=False)
            seed_t = np.zeros((n, k))
            seed_t[seeds, np.arange(k)] = 1
            s = self._em(self._statistics(seed_t, data), data, weights, 10)
            while True:
                scores = self._score(s, data, weights, judged)
                path.append(dict(k=len(s['C']), score=float(scores.mean()), restart=restart,
                                 error=float(scores.std()/np.sqrt(len(scores))), statistics=s))
                if progress is not None and progress(path[-1]) is False:
                    break
                if len(s['C']) == 1:
                    break
                count = max(int(len(s['C'])*self.merge_fraction), 1)
                s, _ = self._generalize(s, weights, count)
                s = self._em(s, data, weights, self.em_steps)
        best = max(path, key=lambda step: step['score'])
        # Ties are ties: rounding must not make a finer level look better than a general one.
        chosen = min((step for step in path if step['score'] >= best['score']-best['error']-1e-9),
                     key=lambda step: (step['k'], -step['score']))
        self.path = [dict(k=step['k'], score=step['score'], restart=step['restart']) for step in path]
        self.selection = dict(best_k=best['k'], best_score=best['score'], standard_error=best['error'],
                              chosen_k=chosen['k'], chosen_score=chosen['score'])
        self.statistics = self._em(chosen['statistics'], data, weights, 20)
        self.final_score = float(self._score(self.statistics, data, weights, judged).mean())
        self.train_weights = weights
        return self

    # ---- use -------------------------------------------------------------------
    @property
    def concepts(self):
        return len(self.statistics['C'])

    def assignments(self, frame, observed_only=True):
        data = self.encoder.transform(frame)
        weights = self._weights(0.) if observed_only else self.train_weights
        return self._route(self.statistics, data, weights)[0]

    def predict_proba(self, frame, label):
        t = self.assignments(frame.drop(columns=[c for c in self.labels if c in frame]))
        _, _, log_p = self._parameters(self.statistics)
        return t@np.exp(log_p[:, self.encoder.block(label)]), self.encoder.categories[label]

    def predict(self, frame, label):
        if label in self.encoder.numeric:
            t = self.assignments(frame.drop(columns=[c for c in self.labels if c in frame]))
            j = self.encoder.numeric.index(label)
            g, _, _ = self._parameters(self.statistics)
            return (t@g[:, j])*self.encoder.std[j]+self.encoder.mean[j]
        probability, classes = self.predict_proba(frame, label)
        return np.asarray(classes)[probability.argmax(1)]

    def explained_variance(self, frame):
        """eta^2 per attribute (numeric) or explained Gini impurity (categorical); CP = sqrt(mean)."""
        data = self.encoder.transform(frame)
        s = self._statistics(self._route(self.statistics, data, self.train_weights)[0], data)
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


class NormalScores:
    """Monotone map of each numeric feature to a Gaussian scale, from training ranks only.

    Optional, off by default. On this scale an attribute without structure is
    exactly what one Gaussian concept expects, and an extreme value cannot
    stretch the scale; but ranks also erase the gaps between well-separated
    groups, which is what resolves fine structure (docs/CONCEPT-GRAPH.md).
    At most ``knots`` quantiles are stored; values in between are interpolated.
    """

    def __init__(self, knots=256):
        self.knots, self.maps = knots, {}

    def fit(self, frame, skip=()):
        from scipy.special import ndtri
        for column in frame:
            if column in skip or not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            values = np.sort(frame[column].dropna().to_numpy(float))
            if len(values) < 2 or values[0] == values[-1]:
                continue
            unique, first, counts = np.unique(values, return_index=True, return_counts=True)
            score = ndtri((first+(counts-1)/2+.5)/len(values))       # mid-rank of tied values
            if len(unique) > self.knots:
                keep = np.unique(np.round(np.linspace(0, len(unique)-1, self.knots)).astype(int))
                unique, score = unique[keep], score[keep]
            self.maps[column] = (unique, score)
        return self

    def transform(self, frame):
        frame = frame.copy()
        for column, (values, score) in self.maps.items():
            if column in frame:
                x = pd.to_numeric(frame[column], errors='coerce').to_numpy(float)
                frame[column] = np.where(np.isnan(x), np.nan, np.interp(x, values, score))
        return frame


class ConceptTree:
    """Levels of concepts built top-down; each level is the prior of the next.

    A concept is refined by fitting the same model on the rows routed to it, on
    their own scale: the child's variances and frequencies shrink to the parent
    concept instead of the root. A child chooses its level with half of its rows;
    it is kept only if the other half, each row judged by concepts that never saw
    it, is predicted better than by the parent by more than one standard error.
    No purity or size threshold.
    """
    max_depth = 8                                               # capacity limit, not a requested depth

    def __init__(self, factory, data, depth=0, budget=None):
        self.model = factory(data, depth > 0)
        self.children, self.depth = {}, depth
        budget = [float('inf')] if budget is None else budget   # nodes still allowed in the whole tree
        budget[0] -= 1
        model = self.model
        encoded = model.encoder.transform(data)
        self.scores = model._score(model.statistics, encoded, model.train_weights, np.arange(len(encoded[0])))
        # The same rows read by ONE concept at this level: what a split must improve on.
        one = {key: value.sum(0, keepdims=True) for key, value in model.statistics.items()}
        self.single = model._score(one, encoded, model.train_weights, np.arange(len(encoded[0])))
        if depth+1 >= self.max_depth or model.concepts < 2:
            return
        routed = model.assignments(data).argmax(1)              # the branch prediction would take
        minimum = 2*model.rows_per_concept                      # below this a child cannot hold two concepts
        for concept in range(model.concepts):
            rows = np.flatnonzero(routed == concept)
            if len(rows) < minimum or budget[0] < 1:
                continue
            subset = data.iloc[rows].reset_index(drop=True) if hasattr(data, 'iloc') else data[rows]
            spent = budget[0]
            child = ConceptTree(factory, subset, depth+1, budget)
            if child.model.concepts < 2 and not child.children:
                budget[0] = spent
                continue
            gain = child.scores-self.scores[rows]
            if not model.labels:                                # joint densities: same units and the mass of the branch
                gain = gain+(encoded[1][rows]*np.log(model.encoder.std/child.model.encoder.std)).sum(1)
                gain = gain+np.log(len(rows)/len(routed))
            # The child chose its level with half of its rows; the other half decides whether it is kept.
            fresh = np.setdiff1d(np.arange(len(rows)), child.model.judged)
            better = lambda values: len(values) > 1 and values.mean() > values.std()/np.sqrt(len(values))+1e-9
            # Kept only if the split itself helps (against one concept on the same scale) and the
            # refined branch predicts better than the parent did.
            if better((child.scores-child.single)[fresh]) and better(gain[fresh]):
                self.children[concept] = child
                self.rows = getattr(self, 'rows', {})
                self.rows[concept] = rows
                self.scores[rows] = self.scores[rows]+gain
            else:
                budget[0] = spent

    @property
    def concepts(self):
        """Terminal concepts of the whole tree."""
        return self.model.concepts-len(self.children)+sum(child.concepts for child in self.children.values())

    @property
    def levels(self):
        return 1+max((child.levels for child in self.children.values()), default=0)

    def nodes(self):
        return 1+sum(child.nodes() for child in self.children.values())

    def predict_proba(self, frame, label, classes):
        """Mixture along the levels: a refined concept is read through its children."""
        model = self.model
        t = model.assignments(frame)
        _, _, log_p = model._parameters(model.statistics)
        local = model.encoder.categories[label]
        leaf = np.zeros((model.concepts, len(classes)))
        leaf[:, [classes.index(c) for c in local]] = np.exp(log_p[:, model.encoder.block(label)])
        out = np.zeros((len(frame), len(classes)))
        for concept in range(model.concepts):
            child = self.children.get(concept)
            if child is None:
                out += t[:, concept:concept+1]*leaf[concept]
                continue
            active = np.flatnonzero(t[:, concept] > 1e-6)
            if len(active):
                rows = frame.iloc[active] if hasattr(frame, 'iloc') else frame[active]
                out[active] += t[active, concept:concept+1]*child.predict_proba(rows, label, classes)
        return out


def judged_covariance(plain, correlated, data):
    """True when concepts with covariance predict the judged rows better by more than one standard error."""
    if not len(correlated.cov):
        return False
    scores = [model._score(model.statistics, model.encoder.transform(data), model.train_weights, plain.judged)
              for model in (plain, correlated)]
    gain = scores[1]-scores[0]
    return bool(gain.mean() > gain.std()/np.sqrt(len(gain))+1e-9)


def fit_tree(frame, labels=(), seed=0, normal_scores=False, covariance='judged', **options):
    """Convenience for tables: the judged tree; covariance inside the concepts when judged better."""
    scores = NormalScores().fit(frame, skip=labels) if normal_scores else None
    data = scores.transform(frame) if scores else frame
    make = lambda rows, child, cov: ConceptMixture(seed=seed, judged_fraction=.5 if child else 1., covariance=cov,
                                                   **options).fit(rows, labels)
    use = covariance is True
    if covariance == 'judged':
        use = judged_covariance(make(data, False, False), make(data, False, True), data)
    tree = ConceptTree(lambda rows, child: make(rows, child, use), data)
    tree.covariance = use
    tree.scores_map = scores
    return tree
