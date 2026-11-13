"""Frozen-weight study of a concept tree over the keys of a pretrained LM.

Nothing is trained and the model itself runs unchanged (dense). For chosen
layers the queries, keys and values are captured and the attention output of
the last ``recent`` positions is recomputed from compressed views of the older
tokens, all with the same number of read vectors:

* recent tokens only;
* recent + flat concepts (average-linkage tree cut once, the same for every query);
* recent + the same tree cut per query: the node with the largest estimated
  error (mass x spread, the quantity of the error bound) is opened first;
* twice as many exact recent tokens;
* the first tokens of the context kept exact as well (pretrained models park
  attention there), with one or two windows of recent tokens.

It also measures whether one partition of the tokens can serve every layer:
the tree is built from the keys of all captured layers together and compared
with a tree built from each layer's own keys.

Example: python scripts/probe_concept_tree.py --text corpus.txt --output reports/concept-tree.json
"""
import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage


def capture(model, ids, layers):
    """Rotated queries, raw and rotated keys, values of the chosen layers (heads x L x d)."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    store, handles = {}, []
    for index in layers:
        attention = model.model.layers[index].self_attn

        def hook(module, args, kwargs, index=index):
            hidden = kwargs.get('hidden_states', args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q = module.q_proj(hidden).view(shape).transpose(1, 2)
            k = module.k_proj(hidden).view(shape).transpose(1, 2)
            v = module.v_proj(hidden).view(shape).transpose(1, 2)
            rq, rk = apply_rotary_pos_emb(q, k, *kwargs['position_embeddings'])
            store[index] = tuple(t[0].double().numpy() for t in (rq, k, rk, v))
        handles.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
    with torch.no_grad():
        model(ids)
    for handle in handles:
        handle.remove()
    return store


def rotate(keys, positions, frequency):
    angles = positions[..., None]*frequency
    angles = np.concatenate((angles, angles), -1)
    half = keys.shape[-1]//2
    return keys*np.cos(angles)+np.concatenate((-keys[..., half:], keys[..., :half]), -1)*np.sin(angles)


class Tree:
    """Average-linkage tree; every node holds additive statistics of its tokens."""

    def __init__(self, features, raw, values):
        n = len(features)
        self.n = n
        z = linkage(features, 'average', metric='sqeuclidean')
        size = 2*n-1
        self.left, self.right = np.full(size, -1), np.full(size, -1)
        self.C = np.ones(size)
        self.U, self.V = np.zeros((size, raw.shape[1])), np.zeros((size, values.shape[1]))
        self.Q, self.P = np.zeros(size), np.zeros(size)
        self.U[:n], self.V[:n], self.Q[:n], self.P[:n] = raw, values, (raw*raw).sum(1), np.arange(n)
        for step, (a, b, _, _) in enumerate(z):
            a, b, node = int(a), int(b), n+step
            self.left[node], self.right[node] = a, b
            for name in ('C', 'U', 'V', 'Q', 'P'):
                getattr(self, name)[node] = getattr(self, name)[a]+getattr(self, name)[b]
        self.root = size-1
        centre = self.U/self.C[:, None]
        self.spread = np.sqrt(np.maximum(self.Q/self.C-(centre*centre).sum(1), 0))   # rms distance of members from the centre

    def flat(self, budget):
        """The same cut for every query: undo the last merges."""
        nodes = [self.root]
        while len(nodes) < budget:
            node = max((x for x in nodes if self.left[x] >= 0), default=None)       # latest merge first
            if node is None:
                break
            nodes.remove(node)
            nodes += [self.left[node], self.right[node]]
        return nodes

    def adaptive(self, score, norm, budget):
        """Open first the node whose summary is estimated to cost most: mass x spread."""
        key = lambda node: -(np.exp(score[node])*self.C[node]*norm*self.spread[node]) if self.left[node] >= 0 else 0.
        heap = [(key(self.root), self.root)]
        done = []
        while heap and len(heap)+len(done) < budget:
            priority, node = heapq.heappop(heap)
            if self.left[node] < 0 or priority == 0.:
                done.append(node)
                continue
            for child in (self.left[node], self.right[node]):
                heapq.heappush(heap, (key(child), child))
        return done+[node for _, node in heap]


def explained(tree_nodes, tree, keys):
    """Fraction of the variance of ``keys`` explained by a partition given as tree nodes."""
    members = lambda node: [node] if tree.left[node] < 0 else members(tree.left[node])+members(tree.right[node])
    total = ((keys-keys.mean(0))**2).sum()
    within = sum(((keys[m]-keys[m].mean(0))**2).sum() for m in map(members, tree_nodes))
    return 1-within/total


def total(pairs):
    pairs = np.asarray(pairs)
    return float(pairs[:, 0].sum()/pairs[:, 1].sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    parser.add_argument('--text', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--segments', type=int, default=3)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 9, 15, 21, 27])
    parser.add_argument('--recent', type=int, default=64)
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--first', type=int, default=4)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    frequency = model.model.rotary_emb.inv_freq.double().numpy()
    ids = tokenizer(args.text.read_text(), return_tensors='pt').input_ids[0]
    names = ('recent', 'first+recent', 'first+recent x2', 'recent x2', 'recent+flat', 'recent+adaptive',
             'recent+flat shared', 'recent+adaptive shared')
    errors = {layer: {name: [] for name in names} for layer in args.layers}
    variance = {layer: dict(own=[], shared=[]) for layer in args.layers}
    for segment in range(args.segments):
        store = capture(model, ids[segment*args.length:(segment+1)*args.length][None], args.layers)
        old = args.length-args.recent
        heads = store[args.layers[0]][0].shape[0]
        groups = store[args.layers[0]][1].shape[0]
        repeat = heads//groups
        scale = store[args.layers[0]][1].shape[-1]**-.5
        # One partition for all layers: every captured layer and head is an attribute of the token.
        joint = np.concatenate([store[layer][1][g, :old]/store[layer][1][g, :old].std() for layer in args.layers
                                for g in range(groups)], 1)
        for layer in args.layers:
            rq, raw, rk, values = store[layer]
            for g in range(groups):
                own = Tree(raw[g, :old], raw[g, :old], values[g, :old])
                shared = Tree(joint, raw[g, :old], values[g, :old])
                variance[layer]['own'].append(explained(own.flat(args.budget), own, raw[g, :old]))
                variance[layer]['shared'].append(explained(shared.flat(args.budget), shared, raw[g, :old]))
                trees = {'': own, ' shared': shared}
                centres = {name: rotate(tree.U/tree.C[:, None], tree.P/tree.C, frequency) for name, tree in trees.items()}
                flats = {name: tree.flat(args.budget) for name, tree in trees.items()}
                for head in range(g*repeat, (g+1)*repeat):
                    for t in range(old, args.length, 4):
                        q = rq[head, t]
                        dense = np.exp(rk[g, :t+1]@q*scale-(rk[g, :t+1]@q*scale).max())
                        target = dense@values[g, :t+1]/dense.sum()
                        exact = np.arange(old, t+1)

                        def read(exact_index, nodes=(), tree=None, centre=None):
                            logits = [rk[g, exact_index]@q*scale]
                            vals = [values[g, exact_index]]
                            if len(nodes):
                                nodes = np.asarray(nodes)
                                logits.append(centre[nodes]@q*scale+np.log(tree.C[nodes]))
                                vals.append(tree.V[nodes]/tree.C[nodes][:, None])
                            logits, vals = np.concatenate(logits), np.concatenate(vals)
                            w = np.exp(logits-logits.max())
                            return np.linalg.norm(w@vals/w.sum()-target), np.linalg.norm(target)

                        errors[layer]['recent'].append(read(exact))
                        errors[layer]['recent x2'].append(read(np.arange(old-args.budget, t+1)))
                        first = np.arange(args.first)
                        errors[layer]['first+recent'].append(read(np.concatenate((first, exact))))
                        errors[layer]['first+recent x2'].append(read(np.concatenate((first, np.arange(old-args.budget, t+1)))))
                        for name, tree in trees.items():
                            errors[layer]['recent+flat'+name].append(read(exact, flats[name], tree, centres[name]))
                            cut = tree.adaptive(centres[name]@q*scale, np.linalg.norm(q)*scale, args.budget)
                            errors[layer]['recent+adaptive'+name].append(read(exact, cut, tree, centres[name]))
        print('segment', segment, {layer: {name: round(total(v), 3) for name, v in errors[layer].items()} for layer in args.layers}, flush=True)
    report = dict(model=args.model, length=args.length, segments=args.segments, recent=args.recent, budget=args.budget,
                  metric='sum of output error norms over sum of dense output norms, over heads and queries',
                  error={str(layer): {name: total(v) for name, v in errors[layer].items()} for layer in args.layers},
                  explained_key_variance={str(layer): {name: float(np.mean(v)) for name, v in variance[layer].items()} for layer in args.layers})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1)+'\n')
    print(json.dumps(report['explained_key_variance']))


if __name__ == '__main__':
    main()
