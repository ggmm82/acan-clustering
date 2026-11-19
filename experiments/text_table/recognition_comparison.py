"""Confronto appaiato del solo riconoscimento su una tabella causale comune, senza memoria."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy import sparse
from tokenizers import Tokenizer

from acan.core.concepts import ConceptModel
from acan.core.conditional import SemanticCuts
from acan.concept_algebra import KAPPA_GRID, categorical_reading
from .dyadic import embeddings
from .dyadic_run import corpus
from .run import blocks


def contexts(tokens, starts, vectors, positions, scales):
    """Ordered, nonoverlapping rings. Each scale adds the older half of its prefix."""
    sums = np.empty((len(tokens) + 1, vectors.shape[1]), dtype=np.float64)
    sums[0] = 0
    for i in range(0, len(tokens), 32768):
        n = min(i + 32768, len(tokens))
        sums[i + 1 : n + 1] = np.cumsum(vectors[tokens[i:n]], axis=0, dtype=float) + sums[i]
    for ix in positions:
        first = starts[np.searchsorted(starts, ix, side="right") - 1]
        fields = []
        for level in range(scales):
            width = 1 << level
            end = ix - width // 2
            begin = ix - width
            present = begin >= first
            mean = (sums[np.maximum(end, 0)] - sums[np.maximum(begin, 0)]) / (width - width // 2)
            fields.extend([mean * present[:, None], present[:, None]])
        yield np.column_stack(fields)


def prepare(path, folder):
    tok = Tokenizer.from_file(str(path / "bpe-8000.json"))
    train, st = corpus(path, "train", tok)
    vectors, _ = embeddings(train, st, tok.get_vocab_size(), "svd", top=96, dimensions=8)
    scales = int(np.log2(np.diff(np.r_[st, len(train)]).max())) + 1
    sample = np.sort(np.random.default_rng(42).choice(len(train), 32768, replace=False))
    raw = next(contexts(train, st, vectors, [sample], scales))
    # Fit one common compression on training contexts; also preserve the immediate token vector.
    mean = raw.mean(0)
    _, axes = np.linalg.eigh(np.cov(raw, rowvar=False))
    axes = axes[:, -16:]

    def store(split, ids, starts):
        pos = np.arange(len(ids))
        mat = np.lib.format.open_memmap(
            folder / (split + ".npy"), mode="w+", dtype=np.float32, shape=(len(pos), 24)
        )
        for i, x in enumerate(
            contexts(ids, starts, vectors, (pos[j : j + 8192] for j in range(0, len(pos), 8192)), scales)
        ):
            mat[i * 8192 : i * 8192 + len(x)] = np.column_stack((x[:, :8], (x - mean) @ axes))
        np.save(folder / (split + "-y.npy"), np.asarray(ids, dtype=np.int32))
        if split != "train":
            np.save(folder / (split + "-starts.npy"), starts)

    store("train", train, st)
    for split in ("validation", "test"):
        ids, starts = corpus(path, split, tok)
        if split == "test":
            offsets = json.loads(
                (Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text()
            )["fresh_offsets"]
        else:
            offsets = blocks(ids)[1]
        seq = np.concatenate([ids[a : a + 512] for a in offsets])
        boundaries = np.concatenate(
            [
                np.r_[i * 512, i * 512 + starts[(starts > a) & (starts < a + 512)] - a]
                for i, a in enumerate(offsets)
            ]
        )
        store(split, seq, boundaries)
    digest = hashlib.sha256()
    with (folder / "train.npy").open("rb") as stream:
        for buffer in iter(lambda: stream.read(1 << 20), b""):
            digest.update(buffer)
    return dict(
        scales=scales,
        dimensions=24,
        rows=len(train),
        pca_sample=len(sample),
        train_matrix_sha256=digest.hexdigest(),
    )


def judge(weights, features, labels, root, head):
    """Categorical counts from all train rows; no row-by-vocabulary output on validation."""
    n = head.sum(1)
    d = np.count_nonzero(head > 1e-9, axis=1)

    def score(split, k):
        q = categorical_reading(head, n[:, None], d[:, None], root, k, np)
        x, y = features[split], labels[split]
        ps = []
        for begin in range(0, len(y), 1024):
            at = slice(begin, begin + 1024)
            ps.extend((weights(x[at]) * q[:, y[at]].T).sum(1))
        return -np.log(np.maximum(ps, 1e-300))

    # Cache the responsibilites instead of rerouting the same validation rows for every kappa.
    w = np.concatenate([weights(chunk) for chunk in np.array_split(features["validation"], 32)])
    y = labels["validation"]
    best = np.inf
    for k in KAPPA_GRID:
        q = categorical_reading(head, n[:, None], d[:, None], root, k, np)
        loss = -np.log(np.maximum(np.einsum("ij,ij->i", w, q[:, y].T), 1e-300))
        tune = loss[: 32 * 511].mean()
        if tune < best:
            best, choice, control = tune, k, loss[32 * 511 :].mean()
    return dict(
        tuning=float(best),
        control=float(control),
        kappa=choice,
        test=float(score("test", choice).mean()),
    )


def compare(folder, method):
    x = {s: np.load(folder / (s + ".npy"), mmap_mode="r") for s in ("train", "validation", "test")}
    y = {s: np.load(folder / (s + "-y.npy"), mmap_mode="r") for s in x}
    for split in ("validation", "test"):
        selected = np.arange(len(y[split])) % 512 != 0
        x[split], y[split] = x[split][selected], y[split][selected]
    train = x["train"]
    vocabulary = 8000
    prior = (np.bincount(y["train"], minlength=vocabulary) + 1) / (len(train) + vocabulary)
    begin = time.monotonic()
    names = [f"x{j}" for j in range(train.shape[1])]
    if method == "native":

        def frames():
            for i in range(0, len(train), 8192):
                yield pd.DataFrame(train[i : i + 8192], columns=names)

        model = ConceptModel(
            max_concepts=256, bins=24, judged_rows=4096, refine_rows=32768, seed=42, covariance=False
        )
        model.fit_chunks(frames)

        def weights(a):
            return model.assignments(pd.DataFrame(a, columns=names))

        k = model.concepts
        selection = [s["concepts"] for s in model.levels]
    else:
        model = SemanticCuts().fit(train, np.ones(len(train)), depth=8)

        def weights(a):
            return np.eye(256)[model.encode(a) - 256]

        k = 256
        selection = []
    head = np.zeros((k, vocabulary))
    for i in range(0, len(train), 2048):
        targets = y["train"][i : i + 2048]
        w = weights(train[i : i + len(targets)])
        one = sparse.csr_matrix(
            (np.ones(len(targets)), (targets, np.arange(len(targets)))), shape=(vocabulary, len(targets))
        )
        head += (one @ w).T
    result = judge(weights, x, y, prior, head)
    result.update(
        concepts=int((head.sum(1) > 0).sum()),
        native_levels=selection,
        seconds=time.monotonic() - begin,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "native", "tree"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    result = (
        prepare(args.data, args.workdir)
        if args.stage == "prepare"
        else compare(args.workdir, args.stage)
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
