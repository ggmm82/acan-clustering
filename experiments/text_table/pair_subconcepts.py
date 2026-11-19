"""Discover high-resolution categorical children of native span concepts on disjoint training articles."""

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np

from acan.core.category_pairs import CategoryPairs
from .pair_reading import Columns, compose_exact, lag


def discover(folder):
    with (folder / "trained-root.pkl").open("rb") as stream:
        reading = pickle.load(stream).refine_category("token")
    rows = {s: Columns(folder, s) for s in ("train", "validation", "test")}
    compose_exact(rows)
    tr = rows["train"]
    judgment = (np.cumsum(tr.age == 0) - 1) % 5 == 4
    report = {}
    for w in (2, 4):
        now = time.monotonic()
        h = w // 2
        # A fallback CP code exists even for a pair with no lexical evidence.
        local = {s: r.spans["c" + str(w)].copy() for s, r in rows.items()}
        count = json.loads((folder / "geometry.json").read_text())[str(w)]["concepts"]
        route = []
        passes = [("e" + str(h), "c" + str(h))]
        if w == 4:
            passes.insert(0, ("s2", "s2"))
        passes.append(("current", "e" + str(h)))
        layers = []
        for recent, older in passes:

            def children(s):
                r = rows[s]
                parent = local[s] if recent == "current" else r.spans[recent]
                left = np.zeros(len(parent), dtype=np.uint32)
                left[h:] = r.spans[older][:-h]
                left[r.age < w - 1] = 0
                return parent, left

            parent, left = children("train")
            parent = lag(parent, 0, tr.age)
            left = lag(left, 0, tr.age)
            full = parent, left, tr.y
            table, stats = reading.subconcepts(
                tuple(a[~judgment] for a in full), tuple(a[judgment] for a in full), full
            )
            # Local indices are unique among accepted children. A missing branch keeps its inherited code.
            for s in rows:
                p, c = children(s)
                ix = table.find(CategoryPairs.keys(p, c))
                valid = ix >= 0
                local[s][valid] = ix[valid] + 1 + count
            route.append((recent, older, count, table.keys))
            count += len(table.keys)
            layers.append(dict(parent=recent, child=older, **stats))
        for s, r in rows.items():
            r.spans["s" + str(w)] = local[s]
            np.save(folder / (s + f"-local-{w}.npy"), local[s])
        with (folder / f"local-{w}.pkl").open("wb") as stream:
            pickle.dump(route, stream)
        report[str(w)] = dict(
            layers=layers,
            allocated=count,
            observed=int(np.unique(local["train"][local["train"] > 0]).size),
            seconds=time.monotonic() - now,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        print(w, report[str(w)], flush=True)
    (folder / "local.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True, type=Path)
    discover(p.parse_args().workdir)
