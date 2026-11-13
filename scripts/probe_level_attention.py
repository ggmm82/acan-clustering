"""Frozen-weight study: reading the old tokens of a context through two levels of concepts.

Nothing is trained and the model runs unchanged (dense). For chosen layers the
queries, keys and values are captured, and the attention output of the last
``recent`` positions is recomputed with the first tokens and the recent window
exact and the older tokens summarized, always with the same number of stored
vectors:

* flat: ``budget`` concepts (centre of the keys, mean value, mass);
* two levels: ``budget/2`` general concepts and ``budget/2`` behaviours. A
  behaviour is a concept of the keys seen through the general concepts (each
  key minus the centre of its concept), shared by all of them. The two levels
  are joined by the dense matrix N of how many tokens are in each pair, and a
  query reads every pair with weight N * exp(q.(centre + behaviour));
* the first tokens and the window alone;
* flat with ``budget**2/4`` concepts: as many concepts as the pairs of the
  two-level graph, but stored one by one.

Concepts are found with the same assignment (nearest centre), offline; this is
a study of the representation, not of the online layer.

Example: python scripts/probe_level_attention.py --text corpus.txt --output reports/levels.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.probe_concept_tree import capture, rotate


def concepts(points, count, rng, steps=12):
    """Nearest-centre concepts (Lloyd); returns the assignment of every point."""
    count = min(count, len(points))
    centres = points[rng.choice(len(points), count, replace=False)].copy()
    for _ in range(steps):
        distance = (points*points).sum(1)[:, None]-2*points@centres.T+(centres*centres).sum(1)[None]
        label = distance.argmin(1)
        for i in range(count):
            members = points[label == i]
            centres[i] = members.mean(0) if len(members) else points[rng.integers(len(points))]
    distance = (points*points).sum(1)[:, None]-2*points@centres.T+(centres*centres).sum(1)[None]
    return distance.argmin(1)


def sums(label, count, *arrays):
    onehot = np.eye(count)[label]
    return (onehot.sum(0), *[onehot.T@array for array in arrays])


def read(query, logits, values, exact_logits, exact_values):
    """Softmax over summarized and exact entries together; logits already contain the log masses."""
    both = np.concatenate((logits, exact_logits), 1)
    weight = np.exp(both-both.max(1, keepdims=True))
    weight = weight/weight.sum(1, keepdims=True)
    return weight@np.concatenate((values, exact_values), 0)


def flat_view(raw, values, positions, count, frequency, rng):
    label = concepts(raw, count, rng)
    count = label.max()+1
    mass, u, v, p = sums(label, count, raw, values, positions[:, None].astype(float))
    alive = mass > 0
    centre = rotate(u[alive]/mass[alive, None], p[alive, 0]/mass[alive], frequency)
    return centre, np.log(mass[alive]), v[alive]/mass[alive, None]


def level_view(raw, values, positions, first, second, frequency, rng):
    general = concepts(raw, first, rng)
    first = general.max()+1
    mass, u, v, p = sums(general, first, raw, values, positions[:, None].astype(float))
    alive = mass > 0
    centre, mean, place = np.zeros_like(u), np.zeros_like(v), np.zeros(first)
    centre[alive], mean[alive], place[alive] = u[alive]/mass[alive, None], v[alive]/mass[alive, None], p[alive, 0]/mass[alive]
    # Seen through the general concepts: each key and value minus the centre of its concept.
    key_rest, value_rest = raw-centre[general], values-mean[general]
    behaviour = concepts(key_rest, second, rng)
    second = behaviour.max()+1
    size, du, dv = sums(behaviour, second, key_rest, value_rest)
    present = size > 0
    shift, value_shift = np.zeros_like(du), np.zeros_like(dv)
    shift[present], value_shift[present] = du[present]/size[present, None], dv[present]/size[present, None]
    pairs = np.zeros((first, second))                             # the dense matrix between the two levels
    np.add.at(pairs, (general, behaviour), 1)
    i, b = np.nonzero(pairs)
    keys = rotate(centre[i]+shift[b], place[i], frequency)
    return keys, np.log(pairs[i, b]), mean[i]+value_shift[b], pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    parser.add_argument('--text', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--length', type=int, default=768)
    parser.add_argument('--segments', type=int, default=3)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 9, 15, 21, 27])
    parser.add_argument('--recent', type=int, default=64)
    parser.add_argument('--budgets', type=int, nargs='+', default=[16, 32, 64])
    parser.add_argument('--first', type=int, default=4)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    frequency = model.model.rotary_emb.inv_freq.double().numpy()
    ids = tokenizer(args.text.read_text(), return_tensors='pt').input_ids[0]
    rng = np.random.default_rng(0)
    names = ('window', 'flat', 'levels', 'flat_pairs')
    error = {budget: {name: [] for name in names} for budget in args.budgets}
    filled = {budget: [] for budget in args.budgets}
    for segment in range(args.segments):
        store = capture(model, ids[None, segment*args.length:(segment+1)*args.length], args.layers)
        for layer, (rq, raw, rk, values) in store.items():
            scale = raw.shape[-1]**-.5
            length = raw.shape[1]
            old = np.arange(args.first, length-args.recent)
            exact = np.r_[np.arange(args.first), np.arange(length-args.recent, length)]
            queries = np.arange(length-args.recent//2, length)       # every query sees the whole window before it... nearly
            for head in range(raw.shape[0]):
                q = rq[head, queries]*scale
                causal = np.where(exact[None] <= queries[:, None], 0., -np.inf)
                exact_logits = q@rk[head, exact].T+causal
                dense = read(q, q@rk[head, old].T, values[head, old], exact_logits, values[head, exact])
                norm = (dense*dense).sum()
                outputs = {}
                for budget in args.budgets:
                    outputs['window'] = read(q, np.zeros((len(q), 0)), np.zeros((0, values.shape[-1])), exact_logits, values[head, exact])
                    for name, count in (('flat', budget), ('flat_pairs', budget*budget//4)):
                        keys, log_mass, mean = flat_view(raw[head, old], values[head, old], old, count, frequency, rng)
                        outputs[name] = read(q, q@keys.T+log_mass, mean, exact_logits, values[head, exact])
                    keys, log_mass, mean, pairs = level_view(raw[head, old], values[head, old], old, budget//2, budget//2, frequency, rng)
                    outputs['levels'] = read(q, q@keys.T+log_mass, mean, exact_logits, values[head, exact])
                    filled[budget].append(float((pairs > 0).mean()))
                    for name in names:
                        error[budget][name].append(float(((outputs[name]-dense)**2).sum()/norm))
        print(f'segment {segment} done', flush=True)
    result = dict(model=args.model, length=args.length, recent=args.recent, first=args.first, layers=args.layers,
                  segments=args.segments, heads_measured=len(error[args.budgets[0]]['flat']),
                  note='relative squared error of the attention output against dense attention; lower is better',
                  budgets={str(budget): dict(**{name: float(np.mean(error[budget][name])) for name in names},
                                             levels_better_than_flat=float(np.mean(np.asarray(error[budget]['levels']) < np.asarray(error[budget]['flat']))),
                                             pairs_in_use=float(np.mean(filled[budget])))
                           for budget in args.budgets})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['budgets'], indent=2))


if __name__ == '__main__':
    main()
