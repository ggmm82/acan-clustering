"""Funzioni comuni alle prove sulle coppie di concetti: blocchi, corpus per articolo, vettori, richiamo."""

import hashlib
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
from sklearn.utils.extmath import randomized_svd


def blocks(ids,count=64,length=512,seed=42):
    # Non-overlapping blocks, selected once and saved as exact token offsets.
    available=np.arange(len(ids)//length)
    if len(available)<count:raise ValueError('Testo insufficiente per il protocollo.')
    starts=np.sort(np.random.default_rng(seed).choice(available,count,replace=False))*length
    return [ids[start:start+length] for start in starts],starts


def corpus(path, split, tokenizer):
    starts = []
    n = 0
    eos = tokenizer.token_to_id("[EOS]")
    with tempfile.TemporaryFile() as stream:
        for batch in pq.ParquetFile(path / (split + ".parquet")).iter_batches(
            batch_size=512, columns=["text"]
        ):
            lines = [s for s in batch.column(0).to_pylist() if s and s.strip()]
            for text, tokens in zip(lines, tokenizer.encode_batch(lines)):
                title = text.strip()
                if n == 0 or (
                    title.startswith("= ") and title.endswith(" =") and not title.startswith("= =")
                ):
                    starts.append(n)
                ids = np.asarray(tokens.ids + [eos], dtype=np.int32)
                stream.write(ids.tobytes())
                n += len(ids)
        stream.flush()
        mapped = np.memmap(stream, mode="r", dtype=np.int32, shape=(n,))
    return mapped, np.asarray(starts)


def frequencies(ids, vocabulary):
    counts = np.zeros(vocabulary, np.int64)
    for begin in range(0, len(ids), 65536):
        counts += np.bincount(ids[begin : begin + 65536], minlength=vocabulary)
    return counts


def embeddings(ids, starts, vocabulary, method, top=24, dimensions=12):
    frequency = frequencies(ids, vocabulary)
    common = np.argsort(-frequency, kind="stable")[:top]
    bucket = np.full(vocabulary, len(common))
    bucket[common] = np.arange(len(common))
    width = len(common) + 1
    before = np.zeros((vocabulary, width), np.int64)
    after = np.zeros_like(before)
    for begin in range(1, len(ids), 65536):
        at = np.arange(begin, min(begin + 65536, len(ids)))
        boundaries = starts[np.searchsorted(starts, begin) : np.searchsorted(starts, at[-1] + 1)]
        at = at[~np.isin(at, boundaries)]
        left = ids[at - 1]
        right = ids[at]
        np.add.at(before, (right, bucket[left]), 1)
        np.add.at(after, (left, bucket[right]), 1)
    values = [np.sqrt(a / np.maximum(a.sum(1, keepdims=True), 1)) for a in (before, after)]
    profile = np.column_stack(values)
    seen = frequency > 0
    if method == "svd":
        center = profile[seen].mean(0)
        _, _, axes = randomized_svd(
            profile[seen] - center, n_components=min(dimensions, *profile[seen].shape), random_state=42
        )
        return (profile - center) @ axes.T, []
    raise ValueError(method)


def recall_inputs(pool, filler, width, seed, trials, gaps):
    rng = np.random.default_rng(seed)
    # Draw ordinary observed tokens; reserve disjoint identifiers for cues and answers.
    pool = pool[pool >= 2]
    draw = rng.choice(pool, size=trials*(width+3)+width-1, replace=False)
    answers = draw[:2*trials]
    keys = draw[2*trials:3*trials]
    carrier = draw[3*trials:3*trials+width-1]
    random_cues = draw[3*trials+width-1:].reshape(trials, width)
    cues = np.vstack(([np.r_[carrier, key] for key in keys], random_cues))
    facts = np.column_stack((cues, answers)).ravel()
    filler = np.asarray(filler)
    filler = filler[~np.isin(filler, np.r_[answers, keys, random_cues[:, -1]])]
    if len(filler) < max(gaps):
        raise ValueError('Testo insufficiente dopo aver escluso chiavi e risposte.')
    prefix = np.r_[facts, filler[:max(gaps)]]
    return cues, answers, facts, prefix


def recall(model, train, test, tokenizer):
    gaps = (1000, 10000, 100000, 300000)
    cues, answers, facts, prefix = recall_inputs(np.unique(train[8:200008]), test, 8, 42, 16, gaps)
    rows = []
    # Paired lexical reformulations of a key, with a random held-out answer per key.
    keys = [
        ("doctor", "physician"),
        ("child", "kid"),
        ("sofa", "couch"),
        ("car", "automobile"),
        ("rabbit", "bunny"),
        ("forest", "woods"),
        ("shop", "store"),
        ("movie", "film"),
        ("street", "road"),
        ("gift", "present"),
        ("mistake", "error"),
        ("job", "occupation"),
        ("ship", "vessel"),
        ("stone", "rock"),
        ("picture", "image"),
        ("home", "house"),
    ]

    def cue(key):
        return tokenizer.encode(" The secret code for the " + key + " is").ids

    paraphrases = [cue(b) for _, b in keys]
    assoc = np.concatenate([np.r_[cue(a), y] for (a, _), y in zip(keys, answers)])
    for gap in gaps:
        for task, cases, targets, history in [
            ("association", cues[:16], answers[:16], prefix[: len(facts) + gap]),
            ("copy", cues[16:], answers[16:], prefix[: len(facts) + gap]),
            (
                "lexical_literal",
                [cue(a) for a, _ in keys],
                answers[:16],
                np.r_[assoc, prefix[len(facts) : len(facts) + gap]],
            ),
            (
                "lexical_paraphrase",
                paraphrases,
                answers[:16],
                np.r_[assoc, prefix[len(facts) : len(facts) + gap]],
            ),
        ]:
            cursor = model.cursor()
            start = time.perf_counter()
            cursor.prefill(history)
            prefill = time.perf_counter() - start
            hits = closed = 0
            ranks = []
            timings = []
            for words, y in zip(cases, targets):
                branch = cursor.fork()
                for word in words:
                    branch.observe(word)
                start = time.perf_counter()
                p = branch.distribution()
                branch.observe(int(p.argmax()))
                timings.append(time.perf_counter() - start)
                hits += int(p.argmax() == y)
                closed += int(targets[p[targets].argmax()] == y)
                ranks.append(1 + np.sum(p > p[y]) + 0.5 * (np.sum(p == p[y]) - 1))
            rows.append(
                dict(
                    task=task,
                    gap=gap,
                    trials=len(cases),
                    full_top1=hits,
                    candidate_top1=closed,
                    median_rank=float(np.median(ranks)),
                    prefill_seconds=prefill,
                    median_generated_ms=1000 * float(np.median(timings)),
                )
            )
    return dict(facts_sha256=hashlib.sha256(facts.astype("<i8").tobytes()).hexdigest(), tasks=rows)
