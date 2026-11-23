"""Measure next-token transfer to held-out alias/template pairs with document memory disabled."""
import json
import numpy as np
import torch

from acan.lm.concept_net_training import Learner, Network, _pad

VOCABULARY = 64
CLASSES = 4
TEMPLATES = (1, 2)
FIRST_LABEL = 4
FIRST_ALIAS = 8
ALIASES_PER_CLASS = 8


def examples(seed, count, held_out, distance):
    rng = np.random.default_rng(seed)
    documents = []
    for index in range(count):
        category = index % CLASSES
        template = TEMPLATES[-1] if held_out else rng.choice(TEMPLATES)
        middle = ALIASES_PER_CLASS // 2
        if held_out:
            offset = rng.integers(middle, ALIASES_PER_CLASS)
        else:
            offset = rng.integers(middle if template == TEMPLATES[-1] else ALIASES_PER_CLASS)
        alias = FIRST_ALIAS + category * ALIASES_PER_CLASS + offset
        fillers = rng.integers(FIRST_ALIAS + CLASSES * ALIASES_PER_CLASS, VOCABULARY, distance)
        documents.append(np.r_[0, alias, fillers, template, FIRST_LABEL + category, 0])
    return documents


def paired_examples(seed, pairs, distance):
    """Hold the suffix fixed and replace only the distant word with one from another category."""
    documents = examples(seed, pairs, True, distance)
    rng = np.random.default_rng(seed + 1)
    paired = []
    for original in documents:
        category = (original[-2] - FIRST_LABEL + rng.integers(1, CLASSES)) % CLASSES
        replacement = original.copy()
        offset = rng.integers(ALIASES_PER_CLASS // 2, ALIASES_PER_CLASS)
        replacement[1] = FIRST_ALIAS + category * ALIASES_PER_CLASS + offset
        replacement[-2] = FIRST_LABEL + category
        paired.extend((original, replacement))
    return paired


@torch.no_grad()
def evaluate(network, documents, permute_aliases=False, paired=False):
    labels = torch.arange(FIRST_LABEL, FIRST_LABEL + CLASSES)
    generator = torch.Generator().manual_seed(31)
    correct, losses = [], []
    for start in range(0, len(documents), 64):
        ids, mask = _pad(documents[start:start + 64])
        if permute_aliases:
            ids[:, 1] = ids[torch.randperm(len(ids), generator=generator), 1]
        state = network.states(ids[:, :-2], mask[:, :-2])[:, -1]
        logits = network._logits(state, network.centre.weight, network.log_mass)
        targets = ids[:, -2]
        correct.extend((labels[logits[:, labels].argmax(-1)] == targets).tolist())
        losses.extend((-logits.log_softmax(-1).gather(1, targets[:, None])[:, 0]).tolist())
    scores = dict(accuracy=float(np.mean(correct)), nll=float(np.mean(losses)))
    if paired:
        scores['both_correct'] = float(np.asarray(correct).reshape(-1, 2).all(axis=1).mean())
    return scores


def main():
    torch.set_num_threads(1)
    for distance in (2, 8, 16, 32):
        training = [examples(100 + epoch, 2048, False, distance) for epoch in range(4)]
        test = examples(99, 1024, True, distance)
        paired_test = paired_examples(101, 512, distance)
        for seed in (7, 17):
            for concepts in (1, 8):
                torch.manual_seed(seed)
                network = Network(dict(dim=32, layers=2, heads=1, concepts=concepts, window=3,
                                       vocabulary=VOCABULARY, cache=0))
                learner = Learner(network, tokens=1024)
                learner.warmup = 24
                for documents in training:
                    learner.learn_documents(documents)
                print(json.dumps(dict(seed=seed, distance=distance, concepts=concepts,
                                      held_out=evaluate(network, test),
                                      permuted=evaluate(network, test, permute_aliases=True),
                                      identical_suffix=evaluate(network, paired_test, paired=True))), flush=True)


if __name__ == '__main__':
    main()
