"""Aggiunge la coordinata della parola al checkpoint selezionato, usando train e validazione."""

import argparse
import json
import pickle
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .conditional_cursor import ConditionalText
from .conditional_eval import losses
from .conditional_run import heldout
from .dyadic_run import corpus
from .word_regions import WordProfiles, judge_words, read_words


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True, help="JSON di conditional_run con il .pkl accanto")
    p.add_argument("--output", type=Path, required=True, help="JSON con il nuovo checkpoint .pkl accanto")
    args = p.parse_args()
    clock = time.perf_counter()
    r = json.loads(args.base.read_text())
    with args.base.with_suffix(".pkl").open("rb") as stream:
        model = ConditionalText(*pickle.load(stream))
    model.geometry.tokenizer_sha256 = r["source_sha256"]["bpe-8000.json"]
    tokenizer = Tokenizer.from_file(str(args.data / "bpe-8000.json"))

    def lines():
        for batch in pq.ParquetFile(args.data / "train.parquet").iter_batches(
            batch_size=512, columns=["text"]
        ):
            yield from (line for line in batch.column(0).to_pylist() if line and line.strip())

    profiles = WordProfiles.fit(lines, tokenizer)
    valid, starts = corpus(args.data, "validation", tokenizer)
    tuning = heldout(model.geometry, valid, starts, np.asarray(r["tuning_offsets"]))
    control = heldout(model.geometry, valid, starts, np.asarray(r["control_offsets"]))
    old_tune = np.exp(-losses(model, tuning)())
    old_control = np.exp(-losses(model, control)())
    new = judge_words(profiles, model.reading.prior, tuning, old_tune)
    check = read_words(profiles, model.reading.prior, control, old_control)
    r["word_coordinate"] = dict(
        training_words=len(profiles.index),
        dimensions=profiles.vectors.shape[1],
        slots_kappa_beta=profiles.weights,
        tuning_before=r["tuning"],
        control_before=r["validation_control"],
        extra_seconds=time.perf_counter() - clock,
    )
    r["tuning"], r["validation_control"] = float(-np.log(new).mean()), float(-np.log(check).mean())
    with args.output.with_suffix(".pkl").open("wb") as stream:
        pickle.dump((model.geometry, model.graph, model.memory, model.base, profiles), stream, protocol=5)
    args.output.write_text(json.dumps(r, indent=2) + "\n")


if __name__ == "__main__":
    main()
