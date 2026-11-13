"""The concept graph on tables (profile ``concept-graph-v1``).

One model, defined in acan/concept_algebra.py and docs/MODEL.md, shared with the
attention layer (acan/lm/concept_attention.py):

* a concept is a node holding additive statistics of its members;
* membership is mass times expected similarity, on the observed attributes only;
* generalization adds together the pair whose merge costs the reader least:
  reads received times the radius of the merged concept;
* two concepts are connected by an edge when some observation hesitates between
  them: the edges are the inspectable structure of what was learned;
* a level is judged by observations subtracted from the concepts that read them,
  and the most general level within one standard error of the best is kept.

Every step is linear in the number of attributes. No concept count, purity,
similarity, depth or covariance is involved.
"""
import numpy as np
import pandas as pd

from acan.concept_algebra import LOG_2PI, centre_variance, reader_cost, reader_radius


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
    def __init__(self, max_concepts=1024, rows_per_concept=4, kappa=1., seed=0,
                 judged_rows=2000, merge_fraction=.05, em_steps=3, restarts=5, restart_rows=5000):
        self.max_concepts, self.rows_per_concept = max_concepts, rows_per_concept
        self.kappa, self.seed, self.judged_rows = kappa, seed, judged_rows
        self.merge_fraction, self.em_steps = merge_fraction, em_steps
        self.restarts, self.restart_rows = restarts, restart_rows

    # ---- nodes: additive statistics --------------------------------------------
    def _statistics(self, t, data):
        x, mask, onehot = data
        return dict(C=t.sum(0), Cm=t.T@mask, U=t.T@x, Q=t.T@(x*x), N=t.T@onehot)

    def _parameters(self, s):
        """Centres; variances and frequencies shrink to the root by kappa pseudo-members."""
        g, v = centre_variance(s['Cm'], s['U'], s['Q'], 1., self.kappa, np)   # root variance is one
        ca = np.zeros_like(s['N'])
        for c in self.encoder.categorical:
            block = self.encoder.block(c)
            ca[:, block] = s['N'][:, block].sum(1, keepdims=True)
        log_p = np.log((s['N']+self.kappa*self.root_p)/(ca+self.kappa))
        return g, v, log_p

    # ---- membership ------------------------------------------------------------
    def _log_density(self, s, data, weights):
        x, mask, onehot = data
        wn, wc = weights
        g, v, log_p = self._parameters(s)
        a = 1/v
        numeric = -.5*(((x*x)*wn)@a.T-2*(x*wn)@(g*a).T+(mask*wn)@(g*g*a+np.log(v)+LOG_2PI).T)
        return numeric+(onehot*wc)@log_p.T

    def _route(self, s, data, weights):
        prior = np.log(np.maximum(s['C'], 1e-300)/s['C'].sum())
        return _softmax(self._log_density(s, data, weights)+prior)

    # ---- edges: who hesitates between whom -------------------------------------
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

    # ---- generalization: where the reader loses least --------------------------
    def _radius(self, cm, u, q, counts):
        """Spread of a concept around its centre in the part that is recognized (root scale one).

        With a target the recognized part is the features; without one it is every attribute.
        """
        variance = np.maximum(q/np.maximum(cm, 1e-12)-(u/np.maximum(cm, 1e-12))**2, 0)
        parts = [variance[..., j] for j, name in enumerate(self.encoder.numeric) if name not in self.labels]
        for name in self.encoder.categorical:
            if name in self.labels:
                continue
            block = self.encoder.block(name)
            size = np.maximum(counts[..., block].sum(-1, keepdims=True), 1e-12)
            root = 1-(self.root_p[block]**2).sum()
            gini = 1-((counts[..., block]/size)**2).sum(-1)
            parts.append(gini/root if root > 1e-12 else gini*0)
        return reader_radius(np.mean(parts, axis=0), np)

    def _pair_costs(self, s, first, second):
        """Reader's loss for each listed pair. A row reads the concepts it belongs to: reads received = mass."""
        own = self._radius(s['Cm'], s['U'], s['Q'], s['N'])
        merged = {key: s[key][first]+s[key][second] for key in ('Cm', 'U', 'Q', 'N')}
        together = self._radius(merged['Cm'], merged['U'], merged['Q'], merged['N'])
        return reader_cost(s['C'][first], own[first], s['C'][second], own[second], together)

    def _merge_costs(self, s, weights=None, rows=None):
        """All pairs for the given concepts (rows x k); used to read the levels of a fitted graph."""
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
            feature = prior+numeric[:, :, ~numeric_targets].sum(-1)
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
        weights = self._weights(float(max(features, 1)))        # targets weigh as much as all features
        rng = np.random.default_rng(self.seed)
        judged = np.sort(rng.choice(n, min(n, self.judged_rows), replace=False))
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
                s = self._generalize(s, data, weights, count)
                s = self._em(s, data, weights, self.em_steps)
        best = max(path, key=lambda step: step['score'])
        # Ties are ties: rounding must not make a finer level look better than a general one.
        chosen = min((step for step in path if step['score'] >= best['score']-best['error']-1e-9),
                     key=lambda step: (step['k'], -step['score']))
        self.path = [dict(k=step['k'], score=step['score'], restart=step['restart']) for step in path]
        self.selection = dict(best_k=best['k'], best_score=best['score'], standard_error=best['error'],
                              chosen_k=chosen['k'], chosen_score=chosen['score'])
        self.statistics = self._em(chosen['statistics'], data, weights, 20)
        first, second, weight = self._edges(self._route(self.statistics, data, weights)[0])
        self.edges = [(int(i), int(j), float(w/n)) for i, j, w in zip(first, second, weight)]
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


ConceptMixture = ConceptModel                                   # earlier name
