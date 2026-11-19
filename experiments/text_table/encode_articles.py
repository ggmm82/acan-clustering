"""Encode WikiText training shards with the reference tokenizer and record article boundaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


def encode(shards, tokenizer_path, output, boundaries):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos = tokenizer.token_to_id("[EOS]")
    if eos is None or tokenizer.get_vocab_size() > 65536:
        raise ValueError("Il protocollo richiede EOS e ID di token a 16 bit.")
    starts = []
    count = 0
    with output.open("wb") as stream:
        for path in shards:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
                lines = [x for x in batch.column(0).to_pylist() if x and x.strip()]
                part = []
                for line, encoded in zip(lines, tokenizer.encode_batch(lines)):
                    title = line.strip()
                    if count == 0 or (
                        title.startswith("= ") and title.endswith(" =") and not title.startswith("= =")
                    ):
                        starts.append(count)
                    part.extend(encoded.ids)
                    part.append(eos)
                    count += len(encoded.ids) + 1
                stream.write(np.asarray(part, dtype="<u2").tobytes())
    np.save(boundaries, starts)

    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    return dict(
        tokens=count, articles=len(starts), sha256={p.name: digest(p) for p in [*shards, tokenizer_path]}
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=Path, nargs="+", required=True)
    for k in ("tokenizer", "ids", "starts"):
        p.add_argument("--" + k, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(encode(args.shards, args.tokenizer, args.ids, args.starts), indent=2))
