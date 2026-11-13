"""Study of multilevel concept trees on tables (not the application path).

At every node the cp-v9 generalization path is computed and one level is kept:
by the pair criterion CP* (default) or by the leave-one-out judgment
(MODE=loo). A branch that prediction would reach, with at least 12 rows and
purity below 0.92 (the application's thresholds), is refined recursively.
The prediction is the mixture along the tree.

Example: python scripts/probe_hierarchy.py table.csv other.xlsx
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from acan.core.concepts import ConceptMixture
from acan.vis.datasets import clean_frame, read_arff, split_for_training, target_and_filters

MIN_ROWS, PURITY = 12, .92


def cp_star(model, s):
    """Probability that two distinct rows share a concept, weighted by the concept's relevance."""
    g, v, log_p = model._parameters(s)
    c = s['C']; n = c.sum()
    w = c*np.maximum(c-1, 0)/(n*(n-1))
    rel = [np.clip(1-v, 0, 1)]                              # numeric: tightness (root variance one)
    for name in model.encoder.categorical:
        b = model.encoder.block(name); p = np.exp(log_p[:, b]); root = model.root_p[b]
        rel.append((((p**2).sum(1)-(root**2).sum())/(1-(root**2).sum()+1e-12)).clip(0, 1)[:, None])
    r = np.concatenate(rel, 1)
    return float((w[:, None]*r).sum()/r.shape[1])


class Node:
    def __init__(self, frame, label, depth, max_depth, seed, counter):
        counter['nodes'] += 1
        steps = []
        self.model = ConceptMixture(seed=seed, restarts=1)
        self.model.fit(frame, [label], progress=lambda step: steps.append(step) or True)
        best = max(steps, key=lambda st: cp_star(self.model, st['statistics']))
        data = self.model.encoder.transform(frame)
        if os.environ.get('MODE') != 'loo':
            self.model.statistics = self.model._em(best['statistics'], data, self.model.train_weights, 20)
        self.label, self.children = label, {}
        # A branch receives the rows that prediction would send to it: routed by features only.
        hard = self.model.assignments(frame.drop(columns=[label])).argmax(1)
        counter['concepts'] += self.model.concepts
        for i in range(self.model.concepts):
            rows = frame[hard == i]
            if depth+1 >= max_depth or len(rows) < MIN_ROWS or len(rows) == len(frame):
                continue
            if rows[label].value_counts(normalize=True).iloc[0] >= PURITY:
                continue
            self.children[i] = Node(rows.reset_index(drop=True), label, depth+1, max_depth, seed, counter)
            counter['depth'] = max(counter['depth'], depth+2)

    def proba(self, frame, classes):
        t = self.model.assignments(frame.drop(columns=[self.label], errors='ignore'))
        _, _, log_p = self.model._parameters(self.model.statistics)
        local = self.model.encoder.categories[self.label]
        leaf = np.zeros((self.model.concepts, len(classes)))
        leaf[:, [classes.index(c) for c in local]] = np.exp(log_p[:, self.model.encoder.block(self.label)])
        out = np.zeros((len(frame), len(classes)))
        for i in range(self.model.concepts):
            if i in self.children:
                active = t[:, i] > 1e-4
                if active.any():
                    out[active] += t[active, i:i+1]*self.children[i].proba(frame[active], classes)
            else:
                out += t[:, i:i+1]*leaf[i]
        return out

def load(p):
    raw = pd.read_csv(p) if p.endswith('.csv') else pd.read_excel(p) if p.endswith('.xlsx') else read_arff(open(p, 'rb').read())
    return clean_frame(raw)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        frame = load(name); labels, filters = target_and_filters(frame); label = labels[0]
        for seed in (42, 7, 21) if 'letter' not in name else (42,):
            train, test, _ = split_for_training(frame, labels, filters, seed=seed)
            classes = sorted(train[label].astype(str).unique()); truth = test[label].astype(str)
            row = dict(file=name, seed=seed)
            for depth in ((1, 2, 3, 6) if "letter" not in name else (3,)):
                counter = dict(nodes=0, concepts=0, depth=1); started = time.monotonic()
                tree = Node(train, label, 0, depth, seed, counter)
                predicted = np.asarray(classes)[tree.proba(test, classes).argmax(1)]
                row[f'depth<={depth}'] = (round(100*accuracy_score(truth, predicted), 2), counter['concepts'], counter['nodes'], counter['depth'], round(time.monotonic()-started))
                print(name, seed, 'max', depth, row[f'depth<={depth}'], flush=True)
