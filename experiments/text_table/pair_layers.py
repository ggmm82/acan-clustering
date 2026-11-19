"""Learn ordered compositions of adjacent spans with the tabular ConceptModel."""

import argparse
import json
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from acan.core.concepts import ConceptModel
from .common import embeddings
from .common import corpus
from .common import blocks


def datasets(data):
    tokenizer = Tokenizer.from_file(str(data / "bpe-8000.json"))
    train, starts = corpus(data, "train", tokenizer)
    splits = {"train": (train, starts)}
    for split in ("validation", "test"):
        tokens, starts = corpus(data, split, tokenizer)
        if split == "test":
            offsets = json.loads(
                (Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text()
            )["fresh_offsets"]
        else:
            offsets = blocks(tokens)[1]
        boundaries = np.concatenate(
            [
                np.r_[i * 512, i * 512 + starts[(starts > a) & (starts < a + 512)] - a]
                for i, a in enumerate(offsets)
            ]
        )
        splits[split] = np.concatenate([tokens[a : a + 512] for a in offsets]), boundaries
    return splits


def batches(indices, size=8192):
    for i in range(0, len(indices), size):
        yield indices[i : i + size]


def age(tokens, starts):
    i = np.arange(len(tokens))
    return i - starts[np.searchsorted(starts, i, side="right") - 1]


def pair_frame(children, vectors, half, endpoints):
    values = np.column_stack((vectors[children[endpoints - half]], vectors[children[endpoints]]))
    return pd.DataFrame(values, columns=[f"x{i}" for i in range(values.shape[1])])


def route_pairs(graph, children, vectors, half, endpoints):
    codes = np.zeros(len(children), dtype=np.int32)
    for ix in batches(endpoints, 2048):
        codes[ix] = graph.assignments(pair_frame(children, vectors, half, ix)).argmax(1) + 1
    return codes


def check_prefixes(folder):
    """Re-encode isolated prefixes and compare with the complete held-out blocks, including resets."""
    tokens = np.load(folder / "test-y.npy")
    starts = np.load(folder / "test-starts.npy")
    with (folder / "root.pkl").open("rb") as stream:
        _, leaf_vectors = pickle.load(stream)
    levels = []
    for width in (2, 4):
        with (folder / f"pair-{width}.pkl").open("rb") as stream:
            graph, centres = pickle.load(stream)
        levels.append((width, graph, centres, np.load(folder / f"test-{width}.npy")))
    checked = 0
    for a in starts:
        for length in (1, 2, 4, 6, 17):
            end = min(len(tokens), a + length)
            children = tokens[a:end]
            vectors = leaf_vectors
            local_starts = starts[(starts >= a) & (starts < end)] - a
            ages = age(children, local_starts)
            for width, graph, centres, expected in levels:
                children = route_pairs(
                    graph, children, vectors, width // 2, np.flatnonzero(ages >= width - 1)
                )
                np.testing.assert_array_equal(children, expected[a:end])
                vectors = centres
            checked += 1
    return checked


def train_pairs(data, folder):
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "trained-root.pkl").unlink(missing_ok=True)
    splits = datasets(data)
    for s, (tokens, starts) in splits.items():
        np.save(folder / (s + "-y.npy"), tokens)
        np.save(folder / (s + "-starts.npy"), starts)
    train, starts = splits["train"]
    vectors, _ = embeddings(train, starts, 8000, "svd", top=96, dimensions=8)
    options = dict(
        max_concepts=256, judged_rows=8192, refine_rows=32768, bins=24, covariance=False, seed=42
    )
    cols = [f"x{i}" for i in range(vectors.shape[1])]
    root = ConceptModel(**options).fit(pd.DataFrame(vectors[np.unique(train)], columns=cols))
    # Dense token IDs are vocabulary entries; missing spans have the separate code zero.
    token_classes = root.assignments(pd.DataFrame(vectors, columns=cols)).argmax(1) + 1
    previous = {s: tokens for s, (tokens, starts) in splits.items()}
    for s, tokens in previous.items():
        np.save(folder / (s + "-1.npy"), token_classes[tokens])
    with (folder / "root.pkl").open("wb") as stream:
        pickle.dump((root, vectors), stream)
    stats = {"1": dict(concepts=int(root.concepts), observed=int(len(np.unique(token_classes[train]))))}
    for width in (2, 4):
        begin = time.monotonic()
        half = width // 2
        valid = {
            s: np.flatnonzero(age(tokens, start) >= width - 1) for s, (tokens, start) in splits.items()
        }
        cols = [f"x{i}" for i in range(2 * vectors.shape[1])]

        def rows(s, ix):
            return pair_frame(previous[s], vectors, half, ix)

        graph = ConceptModel(**options)
        graph.fit_chunks(lambda: (rows("train", ix) for ix in batches(valid["train"])))
        next_columns = {}
        for s, indices in valid.items():
            col = route_pairs(graph, previous[s], vectors, half, indices)
            np.save(folder / (s + f"-{width}.npy"), col)
            next_columns[s] = col
        # The descriptor transmitted upward is a concept's mean ordered child descriptor.
        # No target or test observation participates in its estimation.
        child = previous["train"]
        codes = next_columns["train"]
        size = graph.concepts + 1
        moments = np.zeros((size, len(cols)))
        counts = np.bincount(codes[valid["train"]], minlength=size)
        for ix in batches(valid["train"]):
            pair = rows("train", ix).to_numpy()
            for j in range(len(cols)):
                moments[:, j] += np.bincount(codes[ix], weights=pair[:, j], minlength=size)
        vectors = moments / np.maximum(counts[:, None], 1)
        with (folder / f"pair-{width}.pkl").open("wb") as stream:
            pickle.dump((graph, vectors), stream)
        previous = next_columns
        stats[str(width)] = dict(
            concepts=int(graph.concepts),
            observed=int((counts > 0).sum()),
            rows=len(valid["train"]),
            seconds=time.monotonic() - begin,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        (folder / "geometry.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(width, stats[str(width)], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--workdir", required=True, type=Path)
    args = p.parse_args()
    train_pairs(args.data, args.workdir)
    print("Verified held-out prefixes:", check_prefixes(args.workdir))
