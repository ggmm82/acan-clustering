"""Confronto causale della tabella di scale su WikiText-2."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import tempfile
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from tokenizers import Tokenizer
from .dyadic import Dyadic, embeddings
from .refinement import sources
from .run import blocks
from .recall import recall_inputs


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


def evaluate(model, ids, starts, offsets, ablations=False):
    omit = [None] + (list(range(model.scales)) if ablations else [])
    losses = {s: [] for s in omit}
    shuffled = []
    shuffled_all = []
    rng = np.random.default_rng(613)
    perms = [rng.permutation(pair.graph.concepts) for pair in model.pairs]
    token_permutation = rng.permutation(len(model.vectors))
    for offset in offsets:
        block = ids[offset : offset + 512]
        boundaries = np.r_[0, starts[(starts > offset) & (starts < offset + len(block))] - offset]
        levels, age = model.levels(block, boundaries)
        pos = np.arange(1, len(block))
        x, c = model.context(levels, pos, age)

        def loss(x, c):
            w = model.reader.assignments(x, c)
            return -np.log(np.maximum((w * model.head[:, block[pos]].T).sum(1), 1e-300))

        for j in omit:
            if j == 0:
                losses[j].append(loss(np.full_like(x, np.nan), c))
            elif j is not None and j <= c.shape[1]:
                masked = c.copy()
                masked[:, j - 1] = -1
                losses[j].append(loss(x, masked))
            elif j is None:
                losses[j].append(loss(x, c))
            else:
                losses[j].append(losses[None][-1])
        if ablations:
            wrong = c.copy()
            for j, p in enumerate(perms):
                ok = wrong[:, j] >= 0
                wrong[ok, j] = p[wrong[ok, j]]
            shuffled.append(loss(x, wrong))
            shuffled_x = model.vectors[token_permutation[block[pos - 1]]].copy()
            shuffled_x[age[pos] == 1] = np.nan
            shuffled_all.append(loss(shuffled_x, wrong))
    base = np.stack(losses[None])
    result = dict(
        nll=float(base.mean()),
        tokens=base.size,
        bands={
            f"{a}-{b - 1}": float(base[:, a - 1 : b - 1].mean())
            for a, b in [(1, 32), (32, 128), (128, 256), (256, 512)]
        },
    )
    if ablations:
        result["omit_scale"] = {str(1 << j): float(np.mean(losses[j])) for j in omit if j is not None}
        result["shuffle_concept_identities"] = float(np.mean(shuffled))
        result["shuffle_concepts_and_token_embedding"] = float(np.mean(shuffled_all))
    return result


def article_context(model, ids, starts, offsets):
    levels, age = model.levels(ids, starts)
    positions = (offsets[:, None] + np.arange(1, 512)).ravel()
    scores = np.zeros(model.scales + 1)
    available = np.zeros(model.scales, int)
    for begin in range(0, len(positions), 512):
        where = positions[begin : begin + 512]
        x, c = model.context(levels, where, age)
        for index in range(-1, model.scales):
            xx = x
            cc = c
            if index == 0:
                available[0] += np.isfinite(x).any(1).sum()
                xx = np.full_like(x, np.nan)
            elif 0 < index <= c.shape[1]:
                available[index] += (c[:, index - 1] >= 0).sum()
                cc = c.copy()
                cc[:, index - 1] = -1
            w = model.reader.assignments(xx, cc)
            scores[index + 1] -= np.log(np.maximum((w * model.head[:, ids[where]].T).sum(1), 1e-300)).sum()
    scores /= len(positions)
    return dict(
        nll=float(scores[0]),
        omit_scale={
            str(1 << i): dict(nll=float(scores[i + 1]), available=int(available[i]))
            for i in range(model.scales)
        },
    )


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
                from copy import copy

                branch = copy(cursor)
                branch.last = cursor.last.copy()
                branch.buffers = [b.copy() for b in cursor.buffers]
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--checkpoint-output", type=Path, help="Salva il checkpoint locale della sola prova selezionata"
    )
    p.add_argument("--pair-rows", type=int, default=2048)
    p.add_argument("--target-rows", type=int, default=12000)
    args = p.parse_args()
    tokenizer = Tokenizer.from_file(str(args.data / "bpe-8000.json"))
    data = {s: corpus(args.data, s, tokenizer) for s in ("train", "validation", "test")}
    train, starts = data["train"]
    validation, vstarts = data["validation"]
    test, tstarts = data["test"]
    _, valid_offsets = blocks(validation)
    control = json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())
    test_offsets = np.array(control["fresh_offsets"])
    report = dict(
        train_tokens=len(train),
        articles=len(starts),
        max_article_tokens=int(np.diff(np.r_[starts, len(train)]).max()),
        source_sha256=sources(args.data),
        validation_offsets=valid_offsets.tolist(),
        test_offsets=test_offsets.tolist(),
        pair_sample=args.pair_rows,
        target_structure_sample=args.target_rows,
        scales=20,
        seed=42,
        embedding={},
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    models = {}
    for mode in ("cp", "svd"):
        begin = time.monotonic()
        v, layer = embeddings(
            train,
            starts,
            tokenizer.get_vocab_size(),
            mode,
            dimensions=(models["cp"].vectors.shape[1] if mode == "svd" else 12),
        )
        print(mode, "embedding", v.shape, flush=True)
        model = Dyadic(v).fit(train, starts, args.pair_rows, args.target_rows)
        models[mode] = model
        elapsed = time.monotonic() - begin
        tuning = evaluate(model, validation, vstarts, valid_offsets[:32])
        report["embedding"][mode] = dict(
            embedding_dimensions=v.shape[1],
            embedding_levels=layer,
            fit_seconds=elapsed,
            tuning=tuning,
            terminal_levels=model.reader.layer_sizes(),
            eligible_endpoints=model.support,
            scale_levels={str(2**j): pair.reader.layer_sizes() for j, pair in enumerate(model.pairs, 1)},
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        print(mode, "validation", tuning["nll"], flush=True)
        save()
    winner = min(models, key=lambda m: report["embedding"][m]["tuning"]["nll"])
    model = models[winner]
    report["selected"] = winner
    if args.checkpoint_output:
        import pickle

        with args.checkpoint_output.open("wb") as stream:
            pickle.dump(model, stream, protocol=5)
        probe = model.cursor().prefill(test[:37]).distribution()
        with args.checkpoint_output.open("rb") as stream:
            model = pickle.load(stream)
        np.testing.assert_allclose(probe, model.cursor().prefill(test[:37]).distribution(), atol=1e-12)
        report["checkpoint_bytes"] = args.checkpoint_output.stat().st_size
    report["validation_control"] = evaluate(model, validation, vstarts, valid_offsets[32:])
    report["test"] = evaluate(model, test, tstarts, test_offsets, True)
    report["test_with_article_prefix"] = article_context(model, test, tstarts, test_offsets)
    save()
    report["recall"] = recall(model, train, test, tokenizer)
    report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save()


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
