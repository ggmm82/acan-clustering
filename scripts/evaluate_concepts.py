"""Held-out comparison of the concept graph with the previous model and baselines.

Example: python scripts/evaluate_concepts.py data/*.xlsx data/*.csv --output reports/concepts.json
Splits, targets and column filters are the application's own. Only aggregate
metrics are written; never input rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from acan.core.concepts import ConceptModel
from acan.vis.datasets import clean_frame, read_arff, split_for_training, target_and_filters
from acan.vis.evaluation import encode

warnings.filterwarnings('ignore')


def load(path):
    suffix = path.suffix.lower()
    raw = (pd.read_csv(path) if suffix == '.csv' else pd.read_excel(path) if suffix == '.xlsx'
           else read_arff(path.read_bytes()))
    return clean_frame(raw)


def metrics(truth, predicted):
    return dict(accuracy=float(accuracy_score(truth, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(truth, predicted)))


def app_model(train, test, labels, profile):
    from acan.vis.inference import infer
    from acan.vis.training import graph_stats, train_native
    options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
                   min_samples_split=12, similarity_threshold=.92, routing_profile=profile)
    graph = train_native(train, labels, options)
    predicted = infer(graph.to_json(), test)[labels[0]].astype(str)
    stats = graph_stats(graph)
    return predicted, sum(int((node.Counts > 0).sum()) for node in graph.clusterModel.values()), stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 21])
    parser.add_argument('--current', action='store_true', help='Also train the previous cp-joint-v1 hierarchy.')
    parser.add_argument('--app', action='store_true', help='Also train cp-v9 through the application path.')
    parser.add_argument('--no-baselines', action='store_true')
    args = parser.parse_args()
    runs = []
    with threadpool_limits(limits=1):
        for path in args.files:
            frame = load(path)
            labels, filters = target_and_filters(frame)
            for seed in args.seeds:
                train, test, info = split_for_training(frame, labels, filters, seed=seed)
                target = labels[0]
                truth = test[target].astype(str)
                run = dict(dataset=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seed=seed,
                           target=target, train_rows=len(train), test_rows=len(test),
                           excluded_columns=info['excluded_columns'], methods={})
                started = time.monotonic()
                model = ConceptModel(seed=seed).fit(train, labels)
                predicted = model.predict(test, target)
                run['methods']['concept-graph'] = dict(metrics(truth, predicted), concepts=model.concepts, edges=len(model.edges),
                    levels=[dict(concepts=level['concepts'], pair_criterion=level['pair_criterion'],
                                 several_parents=None if level['parents'] is None else
                                 float(((level['parents'] > .1).sum(0) > 1).mean())) for level in model.levels],
                    seconds=time.monotonic()-started, selection=model.selection,
                    train_accuracy=float(accuracy_score(train[target].astype(str), model.predict(train, target))),
                    cp=model.explained_variance(train)['CP'])
                for flag, profile in ((args.app, 'concept-graph-v1 (app)'), (args.current, 'cp-joint-v1')):
                    if flag:
                        started = time.monotonic()
                        predicted, prototypes, stats = app_model(train, test, labels, profile.split()[0])
                        run['methods'][profile] = dict(metrics(truth, predicted), concepts=prototypes,
                                                       seconds=time.monotonic()-started)
                if not args.no_baselines:
                    x, y = encode(train.drop(columns=labels), test.drop(columns=labels))
                    for name, estimator in (('knn5', KNeighborsClassifier(5)),
                                            ('random_forest', RandomForestClassifier(200, random_state=seed, n_jobs=1)),
                                            ('hist_gradient_boosting', HistGradientBoostingClassifier(random_state=seed))):
                        started = time.monotonic()
                        predicted = estimator.fit(x, train[target].astype(str)).predict(y)
                        run['methods'][name] = dict(metrics(truth, predicted), seconds=time.monotonic()-started)
                runs.append(run)
                print(path.name, seed, {name: (round(m['accuracy']*100, 2), m.get('concepts')) for name, m in run['methods'].items()}, flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(runs=runs), indent=1)+'\n')


if __name__ == '__main__':
    main()
