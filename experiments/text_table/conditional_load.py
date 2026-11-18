"""Carico di un GiB sulla somma sparsa di un livello CP. La qualità si misura su dati separati."""

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from acan.core.category_table import Table
from .conditional import region
from .dyadic import TARGET, ages
from .dyadic_run import corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bytes", type=int, default=1 << 30)
    args = p.parse_args()
    with args.checkpoint.open("rb") as f:
        g, graph, *_ = pickle.load(f)
    tok = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    ids, starts = corpus(args.data, "train", tok)
    # The finest recent-token region actually kept by the judgement.
    feature = max(
        (f for f, t, k in graph.category_regions[TARGET].levels if len(f) == 1 and f[0][:2] == (0, 0)),
        key=lambda f: f[0][2],
    )
    col = g.token_codes[ids]
    age = ages(len(ids), starts)
    where = np.flatnonzero(age > 1)
    chunk = 131072
    rows = (args.bytes + 11) // 12
    expected = np.zeros(len(g.vectors), np.int64)

    def batches():
        for begin in range(0, rows, chunk):
            at = where[np.arange(begin, min(begin + chunk, rows)) % len(where)]
            keys = region([col], age, at, g.depths, feature)
            y = ids[at]
            expected[:] += np.bincount(y, minlength=len(expected))
            yield keys, y

    begin = time.perf_counter()
    table = Table.count_regions(batches(), len(g.vectors))
    seconds = time.perf_counter() - begin
    np.testing.assert_array_equal(table.C.sum(), rows)
    np.testing.assert_array_equal(
        np.bincount(table.cells % (table.V + 1), weights=table.counts, minlength=table.V), expected
    )
    args.output.write_text(
        json.dumps(
            dict(
                rows=rows,
                input_region_and_target_bytes=rows * 12,
                repeated_wikitext_training_rows=len(where),
                feature=feature,
                concepts=len(table.keys),
                seconds=seconds,
                rows_per_second=rows / seconds,
                table_bytes=table.nbytes(),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
