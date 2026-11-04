"""Explicit local benchmark of user tables; exports aggregate metrics only.

Example: python scripts/evaluate_datasets.py table.xlsx other.csv --output results.json
The target and feature exclusions use the GUI suggestions; inspect the output to
confirm they match your intended task. This command runs all seven baselines.
"""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from threadpoolctl import threadpool_limits
from acan.vis.datasets import clean_frame, target_and_filters, split_for_training, check_budget
from acan.vis.training import train_native, graph_stats
from acan.vis.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', help='Override the suggested target for every file')
    args = parser.parse_args()
    options = dict(max_depth=1, max_epochs=100, min_samples_split=12,
                   similarity_threshold=.92, lr=.08, tau=.2, alpha=.5,
                   device='numpy', batch_size=300)
    results = []
    with threadpool_limits(limits=1):
        for path in args.files:
            if path.suffix.lower() == '.csv':
                raw = pd.read_csv(path)
            elif path.suffix.lower() == '.xlsx':
                raw = pd.read_excel(path, engine='openpyxl')
            else:
                parser.error(f'Unsupported format: {path.suffix}; use CSV or XLSX')
            frame = clean_frame(raw)
            labels, filters = target_and_filters(frame, [args.target] if args.target else None)
            if args.target and labels != [args.target]:
                parser.error(f'Target {args.target} absent from {path.name}')
            train, test, info = split_for_training(frame, labels, filters)
            size = check_budget(train)
            started = time.perf_counter()
            graph = train_native(train, labels, options)
            elapsed = time.perf_counter() - started
            report = evaluate(graph.to_json(), test, train, baselines=True)
            results.append(dict(dataset=path.name, info=info, encoded=size,
                                training_seconds=elapsed, graph=graph_stats(graph),
                                scores=report['scores'], timings=report['timings'],
                                protocol=report['protocol']))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(options=options, results=results), indent=2)+'\n')
            print(f'{path.name}: {len(train)} training / {len(test)} validation; saved aggregate metrics.', flush=True)


if __name__ == '__main__':
    main()
