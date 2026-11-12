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
                 restarts=5, restart_rows=5000):
        self.max_concepts, self.rows_per_concept = max_concepts, rows_per_concept
        self.kappa, self.seed, self.judged_rows = kappa, seed, judged_rows
        self.merge_fraction, self.em_steps = merge_fraction, em_steps
        self.restarts, self.restart_rows = restarts, restart_rows

    # ---- sufficient statistics -------------------------------------------------
    def _statistics(self, t, data):
        x, mask, onehot = data
        return dict(C=t.sum(0), Cm=t.T@mask, U=t.T@x, Q=t.T@(x*x), N=t.T@onehot)

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
        g, v, log_p = self._parameters(s)
        a = 1/v
        numeric = -.5*(((x*x)*wn)@a.T-2*(x*wn)@(g*a).T+(mask*wn)@(g*g*a+np.log(v)+LOG_2PI).T)
        return numeric+(onehot*wc)@log_p.T

    def _route(self, s, data, weights):
        prior = np.log(np.maximum(s['C'], 1e-300)/s['C'].sum())
        return _softmax(self._log_density(s, data, weights)+prior)

    # ---- generalization --------------------------------------------------------
    def _merge_costs(self, s, weights):
        """Increase of description length when two concepts are summed (k x k)."""
        wn, wc = weights
        g, v, log_p = self._parameters(s)
        k = len(s['C'])
        own = description_length(s['Cm'], v, s['C'], np, wn)-(s['N']*log_p*wc).sum(1)
        cost = np.empty((k, k))
        for start in range(0, k, 64):
            rows = slice(start, start+64)
            merged = {key: s[key][rows, None]+s[key][None] for key in ('Cm', 'U', 'Q', 'N')}
            cm = merged['Cm']
            _, vm = centre_variance(cm, merged['U'], merged['Q'], 1., self.kappa, np)
            ca = np.zeros_like(merged['N'])
            for c in self.encoder.categorical:
                block = self.encoder.block(c)
                ca[..., block] = merged['N'][..., block].sum(-1, keepdims=True)
            lp = np.log((merged['N']+self.kappa*self.root_p)/(ca+self.kappa))
            c = s['C'][rows, None]+s['C'][None]
            together = description_length(cm, vm, c, np, wn)-(merged['N']*lp*wc).sum(-1)
            cost[rows] = together-own[rows, None]-own[None]
        cost[np.arange(k), np.arange(k)] = np.inf
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
                s, _ = self._generalize(s, weights, count)
                s = self._em(s, data, weights, self.em_steps)
        best = max(path, key=lambda step: step['score'])
        chosen = min((step for step in path if step['score'] >= best['score']-best['error']),
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
