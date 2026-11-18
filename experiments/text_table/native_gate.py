"""Giudizio sul lettore del solo ConceptModel, anche con aggiornamenti causali nel documento."""

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd

from acan.concept_algebra import KAPPA_GRID, categorical_reading
from acan.core.concepts import ConceptModel


def score_document(w, y, starts, global_q):
    """Sufficient statistics just before each target, reset at the physical article/block boundary."""
    c = np.zeros_like(w)
    n = np.zeros_like(w)
    d = np.zeros_like(w)
    root = np.zeros((len(y), 3))
    for a, b in zip(starts, np.r_[starts[1:], len(y)]):
        hist, root_hist = {}, {}
        mass = np.zeros(w.shape[1])
        distinct = np.zeros_like(mass)
        for i in range(a, b):
            token = int(y[i])
            old = hist.setdefault(token, np.zeros_like(mass))
            n[i], c[i], d[i] = old, mass, distinct
            root[i] = i - a, root_hist.get(token, 0), len(root_hist)
            root_hist[token] = root_hist.get(token, 0) + 1
            updated = old + w[i]
            distinct += (updated > 1e-9) & (old <= 1e-9)
            hist[token] = updated
            mass += w[i]
    global_p = (w * global_q[:, y].T).sum(1)
    return global_p, root, w, c, n, d


def losses(cache, k_root, k_local):
    p, stats, w, c, n, d = cache
    if np.isfinite(k_root):
        p = categorical_reading(stats[:, 1], stats[:, 0], stats[:, 2], p, k_root, np)
    if np.isfinite(k_local):
        q = categorical_reading(n, c, d, p[:, None], k_local, np)
        p = (w * q).sum(1)
    return -np.log(np.maximum(p, 1e-300))


def run(folder):
    start = time.monotonic()
    x = {s: np.load(folder / (s + ".npy"), mmap_mode="r") for s in ("train", "validation", "test")}
    y = {s: np.load(folder / (s + "-y.npy"), mmap_mode="r") for s in x}
    cols = [f"x{i}" for i in range(x["train"].shape[1])]

    def batches(split, indices, step):
        for a in range(0, len(indices), step):
            i = indices[a : a + step]
            yield pd.DataFrame(x[split][i], columns=cols), y[split][i]

    train = np.arange(len(x["train"]))
    tune = np.arange(32 * 512).reshape(-1, 512)[:, 1:].ravel()
    model = ConceptModel(
        max_concepts=256, bins=24, judged_rows=4096, refine_rows=32768, seed=42, covariance=False
    )
    model.fit_chunks(lambda: (df for df, target in batches("train", train, 8192)))
    model.fit_observed_target(
        "token", range(8000), batches("train", train, 2048), batches("validation", tune, 512)
    )
    fit_seconds = time.monotonic() - start
    table = model.observed_targets["token"]["probability"]

    def cache(split):
        weights = np.concatenate(
            [model.assignments(chunk) for chunk, _ in batches(split, np.arange(len(y[split])), 1024)]
        )
        starts = np.load(folder / (split + "-starts.npy"))
        return score_document(weights, y[split], starts, table)

    valid = cache("validation")
    # The static head has already been calibrated using the same tuning positions.
    small = tuple(a[tune] for a in valid)
    grid = (np.inf,) + tuple(KAPPA_GRID)
    # Cache the affine coefficients of the local readout for each smoothing strength.
    best = np.inf
    p, stats, w, c, n, d = small
    for kl in grid:
        if np.isinf(kl):
            slope, intercept = 1.0, 0.0
        else:
            strength = kl * np.maximum(d, 1)
            slope = (w * strength / (c + strength)).sum(1)
            intercept = (w * n / (c + strength)).sum(1)
        for kr in grid:
            root = (
                p
                if np.isinf(kr)
                else categorical_reading(stats[:, 1], stats[:, 0], stats[:, 2], p, kr, np)
            )
            score = -np.log(np.maximum(intercept + slope * root, 1e-300)).mean()
            if score < best:
                best, chosen = score, (kr, kl)
    test = cache("test")
    mask = np.arange(len(y["test"])) % 512 != 0
    static = losses(test, np.inf, np.inf)[mask]
    local = losses(test, *chosen)[mask]
    native, _ = model.predict_proba(pd.DataFrame(x["test"][:32], columns=cols), "token")
    np.testing.assert_allclose(native[np.arange(32), y["test"][:32]], test[0][:32], atol=1e-12)
    return dict(
        train_rows=len(train),
        tuning_rows=len(tune),
        test_rows=len(static),
        native_levels=[s["concepts"] for s in model.levels],
        static_kappa=model.observed_targets["token"]["kappa"],
        with_document=float(local.mean()),
        without_document=float(static.mean()),
        document_kappa=[float(k) if np.isfinite(k) else None for k in chosen],
        fit_seconds=fit_seconds,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(run(args.workdir), indent=2) + "\n")
