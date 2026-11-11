"""Compare one proposed CP model against prior implementations on held-out rows.

Example: python scripts/verify_cp_generalization.py --baseline ../PR8 \
    --output reports/comparison.json table.csv other.xlsx
Only aggregate metrics and hashes are exported; never input records.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import pandas as pd
    import acan
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from threadpoolctl import threadpool_limits
    from acan import GraphClusterModel
    from acan.vis.datasets import clean_frame, target_and_filters, split_for_training, read_arff
    from acan.vis.training import train_native, graph_stats
    from acan.vis.inference import infer as gui_infer
    source = args.files[0]
    raw = (pd.read_csv(source) if source.suffix.lower() == '.csv' else
           pd.read_excel(source) if source.suffix.lower() == '.xlsx' else read_arff(source.read_bytes()))
    frame = clean_frame(raw)
    labels, filters = target_and_filters(frame)
    train, test, info = split_for_training(frame, labels, filters, seed=args.seed)
    package = Path(acan.__file__).parent
    fingerprint = hashlib.sha256(b''.join(p.relative_to(package).as_posix().encode()+p.read_bytes()
        for p in sorted(package.rglob('*.py')))).hexdigest()
    issues = []

    def predict(payload, table):
        try:
            return gui_infer(payload, table)
        except ValueError as exc:
            # The historical GUI guard rejects Vote's completely missing row.
            # Its native train-fitted model can predict it; do not drop the row.
            if not table.drop(columns=labels).isna().all(axis=1).any():
                raise
            if 'No objects to concatenate' not in str(exc) and 'attributo noto' not in str(exc):
                raise
            issues.append(str(exc))
            from acan.preprocessing import transform_features
            model = GraphClusterModel.loadModel(payload)
            features = transform_features(table.drop(columns=labels), model.preprocessing)
            probabilities, _, _ = model.predictDataset(features, labels)
            root = model.clusterModel[model.rootId]
            result = table.copy()
            for label in labels:
                result[label] = probabilities[root.fullLabels([label])].idxmax(axis=1).str[len(str(label))+1:].to_numpy()
            return result

    options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300,
        max_depth=3, max_epochs=100, min_samples_split=12, similarity_threshold=.92,
        routing_profile='legacy-v8' if args.variant == 'legacy' else 'cp-joint-v1')
    with threadpool_limits(limits=1):
        started = time.monotonic()
        graph = train_native(train, labels, options)
        seconds = time.monotonic()-started
        payload = graph.to_json()
        predictions, fitted = predict(payload, test), predict(payload, train)
        restored = GraphClusterModel.loadModel(payload)
        before = restored.to_json()
        identical = bool(np.array_equal(predict(payload, test.head(20))[labels[0]],
            predict(before, test.head(20))[labels[0]]))
    result = dict(dataset=source.name, variant=args.variant, seed=args.seed,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), code_sha256=fingerprint,
        labels=labels, filters=filters, info=info, options=options, seconds=seconds,
        stats=graph_stats(graph), hard_nonempty_groups=sum(int((np.asarray(node.Counts)>0).sum())
            for node in graph.clusterModel.values()),
        accuracy=accuracy_score(test[labels[0]], predictions[labels[0]]),
        balanced_accuracy=balanced_accuracy_score(test[labels[0]], predictions[labels[0]]),
        macro_f1=f1_score(test[labels[0]], predictions[labels[0]], average='macro'),
        train_accuracy=accuracy_score(train[labels[0]], fitted[labels[0]]),
        majority_baseline=float((test[labels[0]] == train[labels[0]].mode()[0]).mean()),
        persistence_identical=identical, inference_does_not_mutate_model=before == restored.to_json(),
        gui_inference_issues=sorted(set(issues)))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 21])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source-root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument('--variant', default='exclusive', help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, default=42, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.workers < 1:
        parser.error('--workers must be positive')
    variants = [('exclusive', ROOT)]
    if args.baseline:
        if not (args.baseline/'acan').is_dir():
            parser.error('--baseline must contain a previous checkout with acan/')
        variants = [('legacy', args.baseline.resolve()), ('joint', args.baseline.resolve())]+variants
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix='acan-cp-checks-') as directory:
        jobs = [(p.resolve(), seed, variant, root) for p in args.files
                for seed in args.seeds for variant, root in variants]

        def run(index, job):
            source, seed, variant, root = job
            output = Path(directory)/f'{index}.json'
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                '--source-root', str(root), '--variant', variant, '--seed', str(seed),
                '--output', str(output), str(source)], env=env, capture_output=True, text=True)
            if completed.returncode:
                return dict(dataset=source.name, variant=variant, seed=seed,
                    error=(completed.stdout+completed.stderr)[-3500:])
            return json.loads(output.read_text())

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, index, job) for index, job in enumerate(jobs)]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                args.output.write_text(json.dumps(records, indent=2)+'\n')
                print(json.dumps({key:record[key] for key in
                    ('dataset','variant','seed','accuracy','balanced_accuracy','stats','error') if key in record}), flush=True)
    if any('error' in record for record in records):
        raise SystemExit('Some comparisons failed; inspect the explicit error records.')


if __name__ == '__main__':
    main()
