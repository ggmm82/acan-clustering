"""Audit clustering without targets, retaining missing rows unless explicitly filtered.

Example: python scripts/evaluate_cp_clustering.py GiocoSet.xlsx --compare-legacy \
    --output reports/clustering.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from acan.vis.datasets import clean_frame, read_arff, split_for_training
from acan.vis.training import train_native, graph_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare-legacy', action='store_true')
    parser.add_argument('--complete-columns', nargs='*', default=[], help='Explicitly exclude rows missing these columns.')
    parser.add_argument('--drop-columns', nargs='*', default=[], help='Explicitly omit these observed attributes.')
    parser.add_argument('--gui-split', action='store_true', help='Reproduce the current GUI training subset instead of the full file.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--depth', type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.depth <= 6:
        parser.error('epochs must be positive and depth must be between 1 and 6')
    suffix = args.input.suffix.lower()
    if suffix not in ('.csv', '.xlsx', '.arff'):
        parser.error('expected CSV, XLSX or ARFF')
    raw = (pd.read_csv(args.input) if suffix == '.csv' else pd.read_excel(args.input)
           if suffix == '.xlsx' else read_arff(args.input.read_bytes()))
    data = clean_frame(raw)
    if set(args.complete_columns+args.drop_columns)-set(data.columns):
        parser.error('selected column absent from input')
    original_rows = len(data)
    data = data.dropna(subset=args.complete_columns).drop(columns=args.drop_columns)
    retained_rows = len(data)
    if data.empty or not len(data.columns):
        parser.error('no observations or attributes remain')
    if args.gui_split:
        data, _, _ = split_for_training(data, [])
    options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300,
                   max_depth=args.depth, max_epochs=args.epochs,
                   min_samples_split=12, similarity_threshold=.92)
    report = dict(source=args.input.name,
                  source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  labels=[], original_rows=original_rows, retained_rows=retained_rows,
                  training_rows=len(data), observed_columns=list(data.columns),
                  complete_columns=args.complete_columns, dropped_columns=args.drop_columns,
                  gui_split=args.gui_split, options=options, runs=[])
    profiles = ['legacy-v8', 'cp-joint-v1'] if args.compare_legacy else ['cp-joint-v1']
    with threadpool_limits(limits=1):
        for profile in profiles:
            started = time.monotonic()
            graph = train_native(data, [], dict(options, routing_profile=profile))
            root = graph.clusterModel[graph.rootId]
            state = getattr(root, 'routing_parameters', None) or {}
            report['runs'].append(dict(profile=profile, stats=graph_stats(graph),
                hard_nonempty=sum(int((node.Counts > 0).sum()) for node in graph.clusterModel.values()),
                training_nll=state.get('training_nll'),
                root_cp=root.cp/np.sqrt(len(root.cols)),
                seconds=time.monotonic()-started))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
