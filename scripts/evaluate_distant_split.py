"""Compare three continuations of an exported checkpoint on matched Bash documents and intent pairs.

Run as python -m scripts.evaluate_distant_split CHECKPOINT.zip TRAIN.jsonl TEST.jsonl
where the JSONL records contain nl and cmd. Runs on CPU. Reports next-token NLL on the
command and counts out of 12 for intent sensitivity and both alternatives correct.
"""
import argparse
import json
import random
import zipfile

import numpy as np
import torch
from safetensors.numpy import load
from tokenizers import Tokenizer

from acan.lm.concept_lm import from_checkpoint_arrays
from acan.lm.concept_net_training import Learner, Network


CASES = [
 ('ls -a','ls -t',['Include entries whose names begin with a dot.','Show hidden files as well.'],['Put the most recently modified entries first.','List files in modification time order.']),
 ('sort -r','sort -n',['Reverse the usual order of the lines.','Put lines in descending lexical order.'],['Order the lines by numerical value.','Treat the input as numbers when sorting.']),
 ('grep -i','grep -v',['Match uppercase and lowercase letters alike.','Ignore the case of the pattern.'],['Print the lines that do not match.','Exclude lines containing the pattern.']),
 ('wc -l','wc -c',['Count how many lines the input contains.','Give the number of newlines.'],['Count the bytes in the input.','Give the size in bytes.']),
 ('du -s','du -h',['Print only the total disk usage.','Summarize storage with a single total.'],['Show disk usage in human-readable units.','Display storage sizes using familiar units.']),
 ('head -n','tail -n',['Return lines from the beginning.','Take the first lines of the file.'],['Return lines from the end.','Take the last lines of the file.'])]


def sample(path, seed):
    with open(path, encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    random.Random(seed).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "train", "test"):
        parser.add_argument(name)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with zipfile.ZipFile(args.checkpoint) as archive:
        tokenizer = Tokenizer.from_str(archive.read("tokenizer/tokenizer.json").decode())
        checkpoint = from_checkpoint_arrays(load(archive.read("concepts.safetensors"))).model.net

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    gap_tokens = encode(" Use the shell and give a concise answer.") * 32

    def prefix(question, gap):
        return (encode("<|im_start|>user\n" + question) + gap_tokens[:gap]
                + encode("<|im_end|>\n<|im_start|>assistant\n"))

    held = list({row["nl"]: row for row in sample(args.test, 4371)}.values())[:96]
    held_questions = {row["nl"] for row in held}
    train = [row for row in sample(args.train, 956) if row["nl"] not in held_questions][:384]
    docs = [np.array(prefix(row["nl"], 0) + encode(row["cmd"] + "<|im_end|>")) for row in train]
    for seed in (7, 17):
        for mode in ("reference", "double", "local"):
            torch.manual_seed(seed)
            net = Network.restore(checkpoint)
            if mode == "local":
                proposal = net.distant_split(0, docs)
                if proposal is None:
                    raise ValueError("No repeatable direction in this training sample.")
                net.split_one(0, proposal[0], noise=.22, axis=proposal[1])
            elif mode == "double":
                net.split_concepts(0)
            Learner(net, lr=.0003, tokens=512, seed=seed).learn_documents(docs, warmup=4)
            fitted = net.export()

            def loss(question, gap, command):
                before = prefix(question, gap)
                return -np.log(fitted.targets(before + encode(command))[len(before)-1:]).sum()

            result = dict(mode=mode, seed=seed, answer_nll=(
                sum(loss(row["nl"], 0, row["cmd"]) for row in held)
                / sum(len(encode(row["cmd"])) for row in held)))
            for gap in (0, 64, 128):
                sensitive = correct = 0
                for a, b, questions_a, questions_b in CASES:
                    for qa, qb in zip(questions_a, questions_b):
                        margin_a = loss(qa, gap, b) - loss(qa, gap, a)
                        margin_b = loss(qb, gap, b) - loss(qb, gap, a)
                        sensitive += int(margin_a > margin_b)
                        correct += int(margin_a > 0 and margin_b < 0)
                result[gap] = dict(sensitive=sensitive, both_correct=correct)
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
