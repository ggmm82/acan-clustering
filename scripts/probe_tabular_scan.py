"""Tabular concepts from one pass over the rows: the dyadic blocks of the language layer, with the tabular cost."""
import os, sys, time, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, 'tests')
from acan.core.concepts import ConceptModel
from acan.concept_algebra import reader_cost

KEYS = ('C', 'Cm', 'U', 'Q', 'N', 'PV', 'PP', 'PG')


def rows_as_concepts(model, data):
    x, mask, onehot = data[:3]
    n = len(x)
    s = dict(C=np.ones(n), Cm=mask.copy(), U=x.copy(), Q=x*x, N=onehot.copy())
    if len(data) > 3:
        scale, frequency, impurity = data[3:]
        s.update(PV=mask*scale, PP=np.array(frequency), PG=np.array(impurity))
    else:
        s.update(PV=mask.copy(), PP=np.broadcast_to(model.root_p, onehot.shape).copy(), PG=np.broadcast_to(model.root_g, (n, len(model.root_g))).copy())
    return s


def aggregate(model, s, bins):
    """s: statistics (blocks, m, ...). Adds together, in every block, the pair the reader loses least on, until ``bins`` remain."""
    blocks, m = s['C'].shape
    s = {key: value.copy() for key, value in s.items()}
    rows = np.arange(blocks)
    own = model._radius(s)                                       # (blocks, m)
    cost = np.empty((blocks, m, m))
    for a in range(m):                                           # the first costs: every pair once
        together = model._radius({key: value[:, a:a+1]+value for key, value in s.items()})
        cost[:, a] = reader_cost(s['C'][:, a:a+1], own[:, a:a+1], s['C'], own, together)
    cost[:, np.arange(m), np.arange(m)] = np.inf
    dead = np.zeros((blocks, m), bool)
    for _ in range(m-bins):
        flat = cost.reshape(blocks, -1).argmin(1)
        first, second = np.minimum(flat//m, flat % m), np.maximum(flat//m, flat % m)
        for key, value in s.items():
            value[rows, first] += value[rows, second]
            value[rows, second] = 0
        dead[rows, second] = True
        one = {key: value[rows, first][:, None] for key, value in s.items()}
        own[rows, first] = model._radius(one)[:, 0]
        fresh = reader_cost(one['C'], own[rows, first][:, None], s['C'], own, model._radius({key: one[key]+value for key, value in s.items()}))
        fresh[dead] = np.inf
        fresh[rows, first] = np.inf
        cost[rows, first] = fresh
        cost[rows, :, first] = fresh
        cost[rows, second] = np.inf
        cost[rows, :, second] = np.inf
    keep = np.argsort(dead, axis=1, kind='stable')[:, :bins]      # the concepts that remain, in order
    return {key: np.take_along_axis(value, keep.reshape(blocks, bins, *([1]*(value.ndim-2))), 1) for key, value in s.items()}


def scan(model, data, bins, order, chunk=4096):
    """One pass: rows in blocks of ``bins``, equal blocks generalized two by two. Returns the concepts left."""
    s = rows_as_concepts(model, data)
    s = {key: value[order] for key, value in s.items()}
    n = len(order)
    blocks = n//bins
    node = {key: value[:blocks*bins].reshape(blocks, bins, *value.shape[1:]) for key, value in s.items()}
    left = [{key: value[blocks*bins:] for key, value in s.items()}]          # rows that do not fill a block
    while node['C'].shape[0] > 1:
        count = node['C'].shape[0]
        if count % 2:                                            # the odd block waits, as a digit of the binary counter
            left.append({key: value[-1] for key, value in node.items()})
            node = {key: value[:-1] for key, value in node.items()}
        paired = {key: value.reshape(value.shape[0]//2, 2*value.shape[1], *value.shape[2:]) for key, value in node.items()}
        parts = [aggregate(model, {key: value[start:start+chunk] for key, value in paired.items()}, bins)
                 for start in range(0, paired['C'].shape[0], chunk)]
        node = {key: np.concatenate([part[key] for part in parts]) for key in KEYS}
    left.append({key: value[0] for key, value in node.items()} if node['C'].shape[0] else {key: value[:0, 0] for key, value in node.items()})
    out = {key: np.concatenate([part[key] for part in left]) for key in KEYS}
    alive = out['C'] > 0
    return {key: value[alive] for key, value in out.items()}


class ScanConceptModel(ConceptModel):
    """ConceptModel whose specific concepts come from the scan instead of random rows and their refinement."""
    refine = True                                               # reassign the rows while generalizing (as today)

    def _em(self, s, data, weights, steps):
        return super()._em(s, data, weights, steps) if self.refine else s

    def _path(self, data, weights, judged, rng, progress=None, restarts=None, start=None):
        n = len(data[0])
        k = int(min(start or self.max_concepts, max(n//self.rows_per_concept, 1)))
        bins = int(os.environ.get('BINS', 16))
        s = scan(self, data, bins, rng.permutation(n))
        s = self._em(s, data, weights, self.em_steps)
        path, judged_at = [], np.inf
        while True:
            k_now = len(s['C'])
            step = dict(k=k_now, restart=0, data=data, statistics=s, pair_criterion=self.pair_criterion(s), score=-np.inf, error=0.)
            if k_now <= 64 or k_now <= .9*judged_at:
                scores, judged_at = self._score(s, data, weights, judged), k_now
                step.update(score=float(scores.mean()), scores=scores, error=float(scores.std()/np.sqrt(len(scores))))
            path.append(step)
            if len(s['C']) == 1:
                break
            s = self._generalize(s, data, weights, max(int(len(s['C'])*self.merge_fraction), 1))
            s = self._em(s, data, weights, self.em_steps)
        return path


if __name__ == '__main__':
    from test_quality_regression import bundled, local
    from acan.vis.datasets import clean_frame, read_arff, split_for_training, target_and_filters
    def table(path):
        path = str(path)
        raw = pd.read_csv(path) if path.endswith('.csv') else read_arff(Path(path).read_bytes())
        frame = clean_frame(raw); labels, filters = target_and_filters(frame); return frame, labels, filters
    cases = [('iris', lambda: (bundled('iris'), ['kind'], []), range(10), .95), ('wine', lambda: (bundled('wine'), ['kind'], []), range(10), .95),
             ('breast_cancer', lambda: (bundled('breast_cancer'), ['kind'], []), range(2), .92), ('sonar', lambda: table('tests/data/sonar.csv'), range(1, 6), .83),
             ('vehicle', lambda: table('tests/data/vehicle.arff'), range(1, 4), .75), ('vote', lambda: table('tests/data/vote.arff'), range(1, 6), .94),
             ('titanic', lambda: table(local('titanic.csv')), range(1, 11), .80), ('credit-g', lambda: table(local('credit-g.arff')), range(1, 6), .63),
             ('letter', lambda: table('.data/letter.arff'), range(1, 2), .0)]
    variants = sys.argv[1].split(','); only = sys.argv[2:]
    for name, load, seeds, threshold in cases:
        if only and name not in only: continue
        frame, labels, filters = load(); line = f"{name:13s} soglia {threshold:.2f}"
        for variant in variants:
            scores, sizes, seconds = [], [], 0.
            for seed in seeds:
                train, test, _ = split_for_training(frame, labels, filters, seed=seed)
                model = ConceptModel(seed=seed) if variant == 'attuale' else ScanConceptModel(seed=seed)
                model.refine = variant != 'somme'
                t = time.time(); model.fit(train, labels); seconds += time.time()-t
                scores.append((model.predict(test, labels[0]) == test[labels[0]].astype(str)).mean()); sizes.append(model.concepts)
            line += f" | {variant} {np.mean(scores):.3f} ({int(np.median(sizes))} concetti, {seconds/len(seeds):.1f}s)"
        print(line, flush=True)
