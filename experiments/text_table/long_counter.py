"""Six-token benchmark with the original target offsets and the full causal article prefix."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .counter import ConceptLM
from .common import corpus
from .common import blocks


def evaluation(data, split, tokenizer):
    ids, starts = corpus(data, split, tokenizer)
    offsets = (
        json.loads((Path(__file__).parent / "results/wikitext2-refinement-1024.json").read_text())[
            "fresh_offsets"
        ]
        if split == "test"
        else blocks(ids)[1]
    )

    def groups(which):
        positions = np.concatenate([np.arange(a + 1, a + 512) for a in which])
        texts, indices = [], []
        for a, b in zip(starts, np.r_[starts[1:], len(ids)]):
            chosen = positions[(positions >= a) & (positions < b)]
            if len(chosen):
                texts.append(ids[a:b])
                indices.append(chosen - a)
        return texts, indices

    return [groups(offsets)] if split == "test" else [groups(offsets[:32]), groups(offsets[32:])]


def run(data):
    tok = Tokenizer.from_file(str(data / "bpe-8000.json"))
    train, starts = corpus(data, "train", tok)
    model = ConceptLM(tok.get_vocab_size(), 6, 4, budget_bytes=1 << 30)
    for doc in np.split(train, starts[1:]):
        model.add(doc)
    model.settle()
    tune, ctrl = evaluation(data, "validation", tok)
    (test,) = evaluation(data, "test", tok)
    answer = {}
    for d in (0, 4):
        model.document = d
        model.a.fill(1)
        model.b.fill(1)
        model.judge(*tune)
        answer[str(d)] = {}
        for name, (texts, where) in (("control", ctrl), ("test", test)):
            losses = np.concatenate([-model.logp(t, i) for t, i in zip(texts, where)])
            answer[str(d)][name] = float(losses.mean())
            if name == "test":
                answer[str(d)]["blocks"] = losses.reshape(-1, 511).mean(1).tolist()
    return answer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.data), indent=2) + "\n")
