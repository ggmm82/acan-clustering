"""Verifica dei livelli del medesimo ConceptModel e della memoria causale.

I parquet e il tokenizer train-only sono quelli della prima prova.
Nessuna variante del nucleo tabulare viene installata o registrata nell'app.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .model import VectorTable, semantic_vectors
from .reading import LevelReading
from .run import blocks


def prepare(table, tokens):
    pieces, offsets = blocks(tokens)
    return [(ids, table.model.assignments(table.rows.frame(ids, np.arange(len(ids)))))
            for ids in pieces], offsets


def evaluate(reading, pieces, document):
    losses = []
    for ids, membership in pieces:
        if document:
            probability = reading.sequence_probability(membership, ids)[1:]
        else:
            heads = reading.probability(ids)
            probability = (membership[1:]*heads[:, 1:].T).sum(1)
        losses.append(-np.log(np.maximum(probability, 1e-300)))
    loss = np.stack(losses)
    return dict(nll=float(loss.mean()), positions=loss.size,
        bands={f'{a}-{b}':float(loss[:, max(1,a)-1:b-1].mean())
               for a,b in [(0,32),(32,128),(128,256),(256,512)]})


def measure(table, validation, test):
    reading = LevelReading(table.counts, [level['parents'] for level in table.model.levels[1:]])
    valid, validation_offsets = prepare(table, validation)
    heldout, test_offsets = prepare(table, test)
    runs = []
    # Fixed before measurement; validation alone selects smoothing strength.
    for document in (False, True):
        scores = []
        for kappa in (.03125,.125,.5,2.,8.,32.):
            reading.kappa = kappa
            scores.append((evaluate(reading, valid[:32], document)['nll'], kappa))
        reading.kappa = min(scores)[1]
        runs.append(dict(document=document, kappa=reading.kappa, tuning=scores,
            validation=evaluate(reading, valid[32:], document), test=evaluate(reading, heldout, document)))
    # An ablation of the same reading: pool all concepts into the root.
    # This checks whether context-specific concepts add more than document frequencies.
    root = LevelReading(table.counts.sum(0), [])
    root_valid = [(ids,np.ones((len(ids),1))) for ids,_ in valid]
    root_test = [(ids,np.ones((len(ids),1))) for ids,_ in heldout]
    control = []
    for document in (False, True):
        scores = []
        for kappa in (.03125,.125,.5,2.,8.,32.):
            root.kappa = kappa
            scores.append((evaluate(root,root_valid[:32],document)['nll'],kappa))
        root.kappa = min(scores)[1]
        control.append(dict(document=document,kappa=root.kappa,tuning=scores,
            validation=evaluate(root,root_valid[32:],document),test=evaluate(root,root_test,document)))
    return dict(runs=runs, root_control=control, concepts=table.model.concepts,
        levels=[level['concepts'] for level in table.model.levels],
        validation_offsets=validation_offsets.tolist(), test_offsets=test_offsets.tolist())


def encoded_splits(data, splits):
    import pandas as pd
    from tokenizers import Tokenizer
    source = hashlib.sha256((data/'train.parquet').read_bytes()).hexdigest()
    if (data/'bpe-source.sha256').read_text() != source:
        raise ValueError('Il tokenizer non corrisponde al train: eseguire prima text_table.run.')
    tokenizer = Tokenizer.from_file(str(data/'bpe-8000.json'))
    arrays = {}
    for name in splits:
        lines = [str(t) for t in pd.read_parquet(data/(name+'.parquet'))['text'] if str(t).strip()]
        arrays[name] = np.concatenate([np.asarray(piece.ids+[tokenizer.token_to_id('[EOS]')],np.int32)
            for piece in tokenizer.encode_batch(lines)])
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measure-cost', action='store_true')
    parser.add_argument('--save-table', type=Path, help='Salva il modello locale per le prove successive')
    args = parser.parse_args()
    arrays = encoded_splits(args.data, ('train','validation','test'))
    with threadpool_limits(limits=2):
        start = time.monotonic()
        vectors, _ = semantic_vectors(arrays['train'],8000)
        table = VectorTable(vectors,8,max_concepts=256,judged_rows=512,refine_rows=20000,
            chunk_rows=2048,covariance=False,workers=1).fit(arrays['train'],rows=200000)
        seconds = time.monotonic()-start
        if args.save_table:
            import pickle
            table.tokenizer_sha256 = hashlib.sha256((args.data/'bpe-8000.json').read_bytes()).hexdigest()
            with args.save_table.open('wb') as handle:
                pickle.dump(table, handle)
        result = measure(table, arrays['validation'], arrays['test'])
        if args.measure_cost:
            result['cost'] = measure_prefix_cost(table, arrays['test'], result['runs'][1]['kappa'])
    result['training_seconds'] = seconds
    result['input_sha256'] = {name:hashlib.sha256(tokens.tobytes()).hexdigest() for name,tokens in arrays.items()}
    result['core'] = 'Unchanged ConceptModel; no target-weight patch.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def measure_prefix_cost(table, tokens, kappa):
    """Recognition + true-token reading + update, fixed-size batches, no quality claim."""
    reading = LevelReading(table.counts, [level['parents'] for level in table.model.levels[1:]],kappa)
    state = reading.document()
    reports = []
    for end in (1000,10000,100000,300000):
        if len(tokens) < end:
            raise ValueError('Servono almeno300000 token per la misura del costo.')
        begin = state.observations
        start = time.perf_counter()
        for offset in range(begin,end,1024):
            positions = np.arange(offset,min(offset+1024,end))
            membership = table.model.assignments(table.rows.frame(tokens,positions))
            for pos, weights in zip(positions,membership):
                token = tokens[pos]
                state.probability(weights,[token])
                state.observe(weights,token)
        seconds = time.perf_counter()-start
        reports.append(dict(prefix=end,interval_tokens=end-begin,seconds=seconds,
            milliseconds_per_token=1000*seconds/(end-begin),
            stored_count_values=sum(value.size for counts in state.level_counts for value in counts.values()),
            count_bytes=sum(value.nbytes for counts in state.level_counts for value in counts.values())))
    return reports


if __name__ == '__main__':
    main()
