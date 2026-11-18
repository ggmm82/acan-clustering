"""Verifica quali posizioni del contesto aiutano il riconoscimento dei concetti appresi."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .reading import LevelReading
from .layers import encoded_splits
from .run import blocks, evaluate
from acan.lm.concept_lm import ConceptLM


def losses(table, reading, pieces, width):
    result = []
    for ids in pieces:
        frame = table.rows.frame(ids, np.arange(len(ids)), observed_context=width)
        membership = table.model.assignments(frame)
        probability = reading.sequence_probability(membership, ids)[1:]
        result.append(float(-np.log(probability).mean()))
    return np.asarray(result)


def measure(table, validation, test):
    reading = LevelReading(table.counts, [level['parents'] for level in table.model.levels[1:]], .5)
    valid, _ = blocks(validation)
    widths = [1, 2, 4, 8]
    tuning = {width: float(losses(table, reading, valid[:32], width).mean()) for width in widths}
    chosen = min(tuning, key=tuning.get)
    _, previous = blocks(test)
    available = np.setdiff1d(np.arange(len(test)//512)*512, previous)
    starts = np.sort(np.random.default_rng(20261118).choice(available, 64, replace=False))
    def compare(pieces):
        base = losses(table, reading, pieces, table.rows.context)
        selected = losses(table, reading, pieces, chosen)
        delta = selected-base
        return dict(original_nll=float(base.mean()), selected_nll=float(selected.mean()),
                    difference=float(delta.mean()), paired_block_se=float(delta.std(ddof=1)/np.sqrt(len(delta))),
                    blocks=len(delta), predictions=511*len(delta))
    return dict(tuning=tuning, selected_width=chosen, kappa=reading.kappa,
                validation_control=compare(valid[32:]), fresh_test=compare([test[s:s+512] for s in starts]),
                fresh_test_offsets=starts.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True, help='Checkpoint sperimentale locale fidato')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.table.open('rb') as handle:
        table = pickle.load(handle)
    with threadpool_limits(limits=1):
        arrays = encoded_splits(args.data, ('train', 'validation', 'test'))
        result = measure(table, arrays['validation'], arrays['test'])
        history = arrays['train'][:table.model.rows + table.rows.context]
        control = ConceptLM(len(table.rows.vectors), depth=6, document=4)
        for offset in range(0, len(history), 32768):
            control.add(history[offset:offset+32768])
        control.settle()
        valid, _ = blocks(arrays['validation'])
        control.judge(valid[:32])
        pieces = [arrays['test'][s:s+512] for s in result['fresh_test_offsets']]
        result['counter_first_200008_tokens'] = evaluate(control, pieces)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
