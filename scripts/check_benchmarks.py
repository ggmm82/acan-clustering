"""Validate newly executed synthetic experiments and attach receipts to the audit."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def check(directory, audit_file):
    root = Path(directory) / 'results'
    names = ('exp1_irrelevant_attributes.csv', 'exp2_gioco_interaction.csv',
             'exp3_universal_autocompletion.csv', 'exp4_multidataset_benchmark.csv',
             'exp4_noise_robustness.csv', 'exp5_convergence_dynamics.csv')
    tables = {name: pd.read_csv(root / name) for name in names}
    required = [
        {'n_irrelevant', 'ACAN_v8_mean'}, {'n_irrelevant', 'ACAN_v8_mean'},
        {'masked_attributes_per_row', 'ACAN_v8_Acc'}, {'Dataset', 'ACAN_v8_Acc', 'ACAN_Active_Slots'},
        {'noise_level_pct', 'ACAN_v8_mean'}, {'epoch', 'cp', 'loss', 'k_active'},
    ]
    if any(not columns.issubset(tables[name].columns) for name, columns in zip(names, required)):
        raise ValueError('Missing benchmark measurement columns')
    if tables[names[0]].shape[0] != 7 or tables[names[1]].shape[0] != 4:
        raise ValueError('Incomplete irrelevant-attribute or XOR experiment')
    for name in names[:2]:
        if not (tables[name]['ACAN_v8_mean'] >= 99).all():
            raise ValueError('Regression on selected synthetic CP fixtures: ' + name)
    completion = tables[names[2]]['ACAN_v8_Acc'].to_numpy()
    if completion.shape != (4,) or np.max(np.abs(completion - [94.29, 92.86, 90., 85.71])) > 2:
        raise ValueError('Masked completion differs from the audited synthetic fixture')
    if len(tables[names[3]]) != 6 or len(tables[names[4]]) != 6 or len(tables[names[5]]) < 20:
        raise ValueError('Incomplete synthetic benchmark suite')
    if tables[names[5]].iloc[-1]['k_active'] != 4:
        raise ValueError('Concept compression regression on the audited four-group fixture')
    for name, table in tables.items():
        if not np.isfinite(table.select_dtypes(include='number').to_numpy()).all():
            raise ValueError('Non-finite result: ' + name)
    summary = dict(status='passed', provenance='synthetic; not real UCI datasets', backend='numpy',
                   preprocessing='fit encoding and scale on training only',
                   masked_completion=completion.tolist(),
                   irrelevant=tables[names[0]][['n_irrelevant', 'ACAN_v8_mean']].to_dict('records'),
                   xor=tables[names[1]][['n_irrelevant', 'ACAN_v8_mean']].to_dict('records'),
                   synthetic_multidataset=tables[names[3]].to_dict('records'),
                   csv_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})
    archive = Path(audit_file).parent / 'paper-benchmarks.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for name in names:
            output.write(root / name, name)
        output.writestr('provenance.json', json.dumps(summary, indent=2))
    summary['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = json.loads(Path(audit_file).read_text())
    report['benchmarks'] = summary
    Path(audit_file).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='reports/benchmark')
    parser.add_argument('--audit', default='reports/paper-audit.json')
    args = parser.parse_args()
    check(args.directory, args.audit)
