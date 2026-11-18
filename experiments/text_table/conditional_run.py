"""Misura su blocchi WikiText della lettura per celle CP affinate, selezionata sulla validazione."""

import argparse
import json
from pathlib import Path
import pickle
import resource
import time
import tempfile

import numpy as np
from tokenizers import Tokenizer

from acan.core.category_table import SharedProfiles, Table
from .conditional import SpanGeometry, proposals, region
from .conditional_memory import calibrate, judge, probabilities
from .dyadic import Dyadic, TARGET, ages, embeddings
from .dyadic_run import corpus
from .refinement import sources
from .run import blocks


class Rows:
    def __init__(self, geometry, ids, starts, positions, columns=None):
        self.columns = geometry.transform(ids, starts) if columns is None else columns
        self.age = ages(len(ids), starts)
        self.pos = positions
        self.y = ids[positions]
        self.initial = None
        self.window = 1 << (geometry.scales - 1)
        self.depths = geometry.depths
        self.tokens, self.starts = ids, np.asarray(starts)

    def keys(self, parts):
        return region(self.columns, self.age, self.pos, self.depths, parts)

    def set_base(self, base):
        if base is None:
            return self
        levels, age = base.levels(self.tokens, self.starts)
        self.initial = np.empty(len(self.pos))
        for begin in range(0, len(self.pos), 512):
            at = self.pos[begin : begin + 512]
            x, c = base.context(levels, at, age)
            w = base.reader.assignments(x, c)
            self.initial[begin : begin + len(at)] = (w * base.head[:, self.tokens[at]].T).sum(1)
        return self


def heldout(geometry, ids, starts, offsets):
    blocks = []
    boundaries = []
    for i, start in enumerate(offsets):
        block = ids[start : start + 512]
        if len(block) != 512:
            raise ValueError("Blocco incompleto.")
        blocks.append(block)
        boundaries.extend([i * 512, *(starts[(starts > start) & (starts < start + 512)] - start + i * 512)])
    positions = np.arange(512 * len(blocks)).reshape(-1, 512)[:, 1:].ravel()
    return Rows(geometry, np.concatenate(blocks), np.asarray(boundaries), positions)


def evaluate(model, rows):
    losses = -np.log(model.probability(rows.keys, rows.y, prior=rows.initial))
    return dict(
        nll=float(losses.mean()),
        tokens=len(losses),
        bands={
            f"{a}-{b-1}": float(losses.reshape(-1, 511)[:, a - 1 : b - 1].mean())
            for a, b in [(1, 32), (32, 128), (128, 256), (256, 512)]
        },
    )


def count_observed(column, depth):
    seen = np.zeros(2 ** (depth + 1), bool)
    for begin in range(0, len(column), 65536):
        seen[column[begin : begin + 65536]] = True
    return int(seen[1:].sum())


class TrainingRows:
    def __init__(self, geometry, columns, ids, starts):
        self.geometry, self.columns, self.ids, self.starts = geometry, columns, ids, starts

    def chunks(self, feature):
        for begin in range(0, len(self.ids), 262144):
            at = np.arange(begin, min(begin + 262144, len(self.ids)))
            age = at - self.starts[np.searchsorted(self.starts, at, side="right") - 1] + 1
            key = region(self.columns, age, at, self.geometry.depths, feature, age_at_positions=True)
            yield key, self.ids[begin : begin + len(at)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("cp", "cp_log", "svd"), required=True)
    p.add_argument("--precision", type=int, default=16)
    p.add_argument(
        "--base", type=Path, help="Checkpoint di dyadic_run; in assenza si riaddestra la base CP"
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--neighbors", type=int, default=24)
    args = p.parse_args()
    start = time.monotonic()
    tok = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    train, starts = corpus(args.data, "train", tok)
    valid, vs = corpus(args.data, "validation", tok)
    offsets = blocks(valid)[1]
    v, levels = embeddings(train, starts, tok.get_vocab_size(), args.mode, top=args.neighbors, dimensions=8)
    print("embedding", args.mode, v.shape, flush=True)
    geometry = SpanGeometry(v)
    scratch = tempfile.TemporaryDirectory()
    columns = geometry.fit(train, starts, precision=args.precision, directory=Path(scratch.name))
    print("geometria", geometry.depths, flush=True)
    full = TrainingRows(geometry, columns, train, starts)
    tuning = heldout(geometry, valid, vs, offsets[:32])
    if args.base:
        with args.base.open("rb") as stream:
            base = pickle.load(stream)
    else:
        base_vectors, _ = embeddings(train, starts, len(v), "cp")
        base = Dyadic(base_vectors).fit(train, starts)
    graph = base.graph
    if len(graph.encoder.categories[TARGET]) != len(v):
        raise ValueError("Vocabolario del checkpoint incompatibile.")
    model = graph.refine_category(TARGET)
    base = base if graph.concepts > 1 else None
    tuning.set_base(base)
    p = model.prior[tuning.y] if tuning.initial is None else tuning.initial
    report = dict(
        embedding=args.mode,
        dimensions=v.shape[1],
        embedding_levels=levels,
        source_sha256=sources(args.data),
        train_tokens=len(train),
        tuning_offsets=offsets[:32].tolist(),
        control_offsets=offsets[32:].tolist(),
        scale_concepts={
            str(1 << j): count_observed(col, geometry.depths[j]) for j, col in enumerate(columns)
        },
        refinements=[],
    )
    for stage in proposals(geometry.depths):
        best = -np.log(p).mean()
        winner = None
        for feature in dict.fromkeys(stage):
            loss, choice = model.propose(
                Table.count_regions(full.chunks(feature), len(v)), tuning.keys(feature), tuning.y, p
            )
            if choice and loss < best:
                best, winner = loss, (feature, *choice)
        if winner:
            feature, table, kappa, p = winner
            model.levels.append((feature, table, kappa))
            report["refinements"].append(
                dict(parts=feature, concepts=len(table.keys), kappa=kappa, tuning_nll=float(best))
            )
            print("lettura", feature, round(best, 5), flush=True)
    scratch.cleanup()
    memory = judge(model, tuning, calibrate(model, tuning))
    report["bytes_before_sharing"] = sum(t.nbytes() for _, t, _ in model.levels)
    model.levels = [(f, SharedProfiles(t), k) for f, t, k in model.levels]
    report["bytes_after_sharing"] = sum(t.nbytes() for _, t, _ in model.levels)
    report["fit_seconds"] = time.monotonic() - start
    report["neighbors"] = args.neighbors
    report["kept"] = [
        dict(parts=f, regions=len(t.keys), prototypes=len(t.C), kappa=k) for f, t, k in model.levels
    ]
    report["memory"] = memory
    report["tuning"] = float(-np.log(probabilities(model, memory, tuning)).mean())
    control = heldout(geometry, valid, vs, offsets[32:]).set_base(base)
    report["validation_control"] = float(-np.log(probabilities(model, memory, control)).mean())
    report["control_without_document"] = evaluate(model, control)
    if args.checkpoint:
        with args.checkpoint.open("wb") as stream:
            pickle.dump((geometry, graph, memory, base), stream, protocol=5)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
