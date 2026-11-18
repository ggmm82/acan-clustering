"""Evaluate the installed worker's graph over the complete WikiText-2 training split."""
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
from threadpoolctl import threadpool_limits

from acan.lm.text_graph import TextGraph
from .layers import encoded_splits
from .run import blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = encoded_splits(args.data, ('train', 'test'))
    train = data['train']
    pieces, offsets = blocks(data['test'])
    model = TextGraph(8000)
    report = dict(piece_tokens=200008, test_offsets=offsets.tolist(), measurements=[])
    with threadpool_limits(1):
        start = time.monotonic()
        for offset in range(0, len(train), report['piece_tokens']):
            model.add(train[offset:offset+report['piece_tokens']])
            if offset == 0 or model.tokens == len(train):
                loss = np.mean([-model.logp(piece)[1:].mean() for piece in pieces])
                report['measurements'].append(dict(tokens=model.tokens, levels=model.concepts(),
                    nll=float(loss), elapsed_seconds=time.monotonic()-start,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
