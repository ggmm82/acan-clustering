"""Measure the native token prediction with a lossless previous-token column and no document memory."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from acan.concept_algebra import categorical_reading
from acan.core.category_table import Table
from acan.core.concepts import ConceptModel
from acan.lm.concept_lm import ConceptLM
from .dyadic_control import sections
from .dyadic_run import corpus
from .run import blocks


def previous(tokens, starts):
    """Key zero means the preceding token is unavailable within this article or block."""
    keys = np.empty(len(tokens), dtype=np.uint32)
    keys[1:] = tokens[:-1] + 1
    keys[starts] = 0
    return keys


def measure(data, folder):
    started = time.monotonic()
    tokenizer = Tokenizer.from_file(str(data / "bpe-8000.json"))
    vocab = tokenizer.get_vocab_size()
    train, starts = corpus(data, "train", tokenizer)
    y = {s: np.load(folder / (s + "-y.npy"), mmap_mode="r") for s in ("train", "validation", "test")}
    x = {s: np.load(folder / (s + ".npy"), mmap_mode="r") for s in y}
    np.testing.assert_array_equal(train, y["train"])
    keys = {"train": previous(train, starts)}
    for s in ("validation", "test"):
        keys[s] = previous(y[s], np.load(folder / (s + "-starts.npy")))
    # The binary representation is injective for every categorical ID, including missing.
    bits = np.arange(vocab.bit_length(), dtype=np.uint32)
    cols = [f"x{i}" for i in range(x["train"].shape[1])] + [f"prev_bit_{i}" for i in bits]

    def frame(s, ix):
        binary = (keys[s][ix, None] >> bits) & 1
        return pd.DataFrame(np.column_stack((x[s][ix], binary)), columns=cols)

    def batches(s, indices, step=2048):
        for i in range(0, len(indices), step):
            ix = indices[i : i + step]
            yield frame(s, ix), y[s][ix]

    all_train = np.arange(len(train))
    tune = np.arange(32 * 512).reshape(-1, 512)[:, 1:].ravel()
    test = np.arange(len(y["test"])).reshape(-1, 512)[:, 1:].ravel()
    model = ConceptModel(
        max_concepts=256, bins=24, judged_rows=4096, refine_rows=32768, seed=42, covariance=False
    )
    model.fit_chunks(lambda: (f for f, _ in batches("train", all_train, 8192)))
    model.fit_observed_target(
        "token", range(vocab), batches("train", all_train), batches("validation", tune)
    )
    head = model.observed_targets["token"]["probability"]

    def native(s, indices):
        return np.concatenate(
            [(model.assignments(f) * head[:, target].T).sum(1) for f, target in batches(s, indices, 512)]
        )

    table = Table.count_regions(((keys["train"], train),), vocab)
    training_frequency = np.bincount(train, minlength=vocab) + 1
    unigram = training_frequency / training_frequency.sum()
    reading = model.refine_category("token")

    def condition(base_valid, base_test):
        _, choice = reading.propose(table, keys["validation"][tune], y["validation"][tune], base_valid)
        if choice is None:
            return base_test, None
        _, kappa, _ = choice
        m, n, d = table.lookup(keys["test"][test], y["test"][test])
        return categorical_reading(n, m, d, base_test, kappa, np), kappa

    p_native = native("test", test)
    p_exact, k_exact = condition(unigram[y["validation"][tune]], unigram[y["test"][test]])
    p_both, k_both = condition(native("validation", tune), p_native)
    if k_both is not None:
        reading.levels.append(("previous_token", table, k_both))
    check_ix = test[::251]
    check = frame("test", check_ix).assign(previous_token=keys["test"][check_ix])
    output, _ = model.predict_proba(check, "token")
    np.testing.assert_allclose(
        output[np.arange(len(check)), y["test"][check_ix]], p_both[::251], rtol=1e-11
    )

    # Evaluate the deployed counter with precisely one previous token, no adaptive document counts.
    counter = ConceptLM(vocab, depth=1, document=0, budget_bytes=1 << 30)
    for article in np.split(train, starts[1:]):
        counter.add(article)
    counter.settle()
    ids, boundaries = corpus(data, "validation", tokenizer)
    valid_offsets = blocks(ids)[1][:32]
    valid_pieces = list(sections(ids, boundaries, valid_offsets))
    np.testing.assert_array_equal(np.concatenate([p[ix] for p, ix in valid_pieces]), y["validation"][tune])
    counter.judge([p for p, _ in valid_pieces if len(p) > 1])
    ids, boundaries = corpus(data, "test", tokenizer)
    offsets = json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())[
        "fresh_offsets"
    ]
    records = list(sections(ids, boundaries, offsets))
    np.testing.assert_array_equal(np.concatenate([p[ix] for p, ix in records]), y["test"][test])
    counter_loss = np.concatenate([-counter.logp(p, ix) for p, ix in records])

    def nll(p):
        return float(-np.log(p).mean())

    results = dict(
        native_lossless_bits=nll(p_native),
        exact_identity_only=nll(p_exact),
        native_with_exact_category=nll(p_both),
        one_token_counter=float(counter_loss.mean()),
    )
    _, seen_counts, _ = table.lookup(keys["test"][test], y["test"][test])
    partition = {}
    for name, mask in (("seen", seen_counts > 0), ("unseen_or_start", seen_counts == 0)):
        partition[name] = dict(tokens=int(mask.sum()), exact_identity_only=nll(p_exact[mask]))
    return dict(
        train_rows=len(train),
        validation_rows=len(tune),
        test_rows=len(test),
        document=False,
        baseline_geometry_dimensions=x["train"].shape[1],
        identity_bits=len(bits),
        vocabulary=vocab,
        observed_preceding_types=len(table.keys),
        observed_bigrams=len(table.cells),
        levels=[s["concepts"] for s in model.levels],
        static_kappa=model.observed_targets["token"]["kappa"],
        exact_only_kappa=k_exact,
        native_with_exact_kappa=k_both,
        test_nll=results,
        exact_bigram_diagnostics=partition,
        seconds=time.monotonic() - started,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.data, args.workdir), indent=2) + "\n")
