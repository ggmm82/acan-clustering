"""Measure native parent-local CP geometry over every distinct training pair."""

import argparse
import json
import pickle
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
from acan.core.category_pairs import CategoryPairs


def run(folder):
    start = time.monotonic()
    with (folder / "pair-2.pkl").open("rb") as stream:
        graph, _ = pickle.load(stream)
    with (folder / "root.pkl").open("rb") as stream:
        _, vectors = pickle.load(stream)

    def keys(split):
        y = np.load(folder / (split + "-y.npy"))
        starts = np.load(folder / (split + "-starts.npy"))
        older = np.roll(y + 1, 1)
        older[starts] = 0
        return CategoryPairs.keys(older, y + 1)

    def frame(pairs):
        left = (pairs >> 32).astype(int) - 1
        right = (pairs & 0xFFFFFFFF).astype(int) - 1
        return pd.DataFrame(
            np.column_stack((vectors[left], vectors[right])), columns=graph.encoder.numeric
        )

    pair = keys("train")
    unique = np.unique(pair[pair > 0])
    local = graph.fit_subconcepts(frame(unique))
    with (folder / "native-local-router.pkl").open("wb") as stream:
        pickle.dump(local, stream, protocol=pickle.HIGHEST_PROTOCOL)
    for split in ("train", "validation", "test"):
        pair = keys(split)
        values, ix = np.unique(pair, return_inverse=True)
        keep = values != 0
        codes = np.zeros(len(values), np.uint32)
        codes[keep] = local.transform(frame(values[keep])) + 1
        col = codes[ix]
        np.save(folder / (split + "-native-local-2.npy"), col)
        if split == "train":
            observed = int(np.unique(col[col > 0]).size)
    report = dict(
        pair_types=len(unique),
        native_parents=graph.concepts,
        split_parents=len(local.branches),
        occupied_codes=observed,
        allocated_codes=local.concepts,
        judged_rows=local.judged,
        largest_branch_concepts=max([0] + [g.concepts for _, g in local.branches.values()]),
        seconds=time.monotonic() - start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (folder / "native-local-geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True, type=Path)
    run(p.parse_args().workdir)
