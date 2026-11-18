"""Recupero di associazioni casuali e completamento di sequenze su un prefisso lungo."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .reading import LevelReading


def reveal(table, state, prefix, start, stop, chunk=512):
    for offset in range(start, stop, chunk):
        positions = np.arange(offset, min(offset + chunk, stop))
        membership = table.model.assignments(table.rows.frame(prefix, positions))
        state.observe_many(membership, prefix[positions])


def probe(table, filler, seed=42, trials=16, gaps=(1000, 10000, 100000, 300000), kappa=.5):
    """Each query sees the shared prefix and its cue. Query branches never share answers."""
    rng = np.random.default_rng(seed)
    reading = LevelReading(table.counts, [level['parents'] for level in table.model.levels[1:]], kappa)
    state = reading.document()
    width = table.rows.context
    # Draw ordinary observed tokens; reserve disjoint identifiers for cues and answers.
    pool = np.flatnonzero(reading.root > 0)
    pool = pool[pool >= 2]
    draw = rng.choice(pool, size=trials*(width+3)+width-1, replace=False)
    answers = draw[:2*trials]
    keys = draw[2*trials:3*trials]
    carrier = draw[3*trials:3*trials+width-1]
    random_cues = draw[3*trials+width-1:].reshape(trials, width)
    cues = np.vstack(([np.r_[carrier, key] for key in keys], random_cues))
    facts = np.column_stack((cues, answers)).ravel()
    filler = np.asarray(filler)
    filler = filler[~np.isin(filler, np.r_[answers, keys, random_cues[:, -1]])]
    if len(filler) < max(gaps):
        raise ValueError('Testo insufficiente dopo aver escluso chiavi e risposte.')
    prefix = np.r_[facts, filler[:max(gaps)]]
    start = 0
    rows = []
    for gap in gaps:
        stop = len(facts)+gap
        reveal(table, state, prefix, start, stop)
        start = stop
        for task, indices in [('associazione',range(trials)), ('copia',range(trials,2*trials))]:
            top1 = closed = 0
            ranks = []
            for i in indices:
                branch = deepcopy(state, {id(reading): reading})
                query = np.r_[prefix[stop-width:stop], cues[i]]
                reveal(table, branch, query, width, len(query))
                membership = table.model.assignments(table.rows.frame(query, [len(query)]))[0]
                probabilities = branch.probability(membership, np.arange(reading.vocabulary))
                np.testing.assert_allclose(probabilities.sum(), 1., atol=1e-10)
                y = answers[i]
                top1 += int(probabilities.argmax() == y)
                candidates = answers[indices.start:indices.stop]
                closed += int(candidates[probabilities[candidates].argmax()] == y)
                ranks.append(float(1 + (probabilities > probabilities[y]).sum()
                                   + .5*((probabilities == probabilities[y]).sum()-1)))
            rows.append(dict(task=task, filler_tokens=gap, trials=trials, full_top1=top1,
                             candidate_top1=closed, median_rank=float(np.median(ranks))))
    return dict(seed=seed, trials_per_task=trials, vocabulary=reading.vocabulary,
                kappa=kappa, concepts=table.model.concepts,
                levels=[level['concepts'] for level in table.model.levels],
                facts_sha256=hashlib.sha256(facts.astype('<i8').tobytes()).hexdigest(),
                runs=rows)


def main():
    import pickle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', required=True, type=Path, help='Checkpoint sperimentale locale fidato')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with args.table.open('rb') as handle:
        table = pickle.load(handle)
    with threadpool_limits(limits=1):
        from .layers import encoded_splits
        result = probe(table, encoded_splits(args.data, ('test',))['test'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
