"""Frozen-weight probe: dense attention replaced by a recent window plus concept slots.

No parameter is trained. In every layer and key head of a pretrained causal LM,
keys older than the exact window are summarized by k concepts holding the same
additive statistics as the tabular core (count, key sum, key square sum, value
sum). A new key is a new concept; above the budget the pair with the smallest
generalization cost is merged (same description-length cost as
acan/core/concepts.py). Readout is ONE softmax over exact keys and concepts,
with log-count multiplicity (docs/MATHEMATICAL-FOUNDATION.md, C.3).

Example:
  python scripts/probe_slot_attention.py --model HuggingFaceTB/SmolLM2-135M \
      --text corpus.txt --output reports/slot-probe.json
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch


class Slots:
    """Additive concept statistics for one key head."""

    def __init__(self, budget, policy, kappa=1.):
        self.budget, self.policy, self.kappa = budget, policy, kappa
        self.C = self.U = self.Q = self.V = None
        self.root = None                                        # count, sum, square sum of all absorbed keys

    def absorb(self, keys, values):
        ones = keys.new_ones(len(keys))
        parts = (ones, keys, keys*keys, values)
        if self.C is None:
            self.C, self.U, self.Q, self.V = parts
            self.root = [float(len(keys)), keys.sum(0), (keys*keys).sum(0)]
        else:
            self.C, self.U, self.Q, self.V = (torch.cat(pair) for pair in zip((self.C, self.U, self.Q, self.V), parts))
            self.root = [self.root[0]+len(keys), self.root[1]+keys.sum(0), self.root[2]+(keys*keys).sum(0)]
        while len(self.C) > self.budget:
            excess = len(self.C)-self.budget
            pairs = self._adjacent(excess) if self.policy == 'chunk' else self._cheapest(max(min(excess, len(self.C)//8), 1))
            keep = torch.ones(len(self.C), dtype=torch.bool)
            left = torch.tensor([i for i, _ in pairs])
            right = torch.tensor([j for _, j in pairs])
            for name in ('C', 'U', 'Q', 'V'):
                value = getattr(self, name)
                value[left] = value[left]+value[right]
            keep[right] = False
            self.C, self.U, self.Q, self.V = self.C[keep], self.U[keep], self.Q[keep], self.V[keep]

    def _variance(self, c, u, q):
        mean = self.root[1]/self.root[0]
        root = (self.root[2]/self.root[0]-mean*mean).clamp_min(1e-8)
        g = u/c[..., None]
        return ((q-c[..., None]*g*g).clamp_min(0)+self.kappa*root)/(c[..., None]+self.kappa)

    def _cheapest(self, count):
        own = .5*self.C*self._variance(self.C, self.U, self.Q).log().sum(-1)-self.C*self.C.log()
        c = self.C[:, None]+self.C[None]
        together = .5*c*self._variance(c, self.U[:, None]+self.U[None], self.Q[:, None]+self.Q[None]).log().sum(-1)-c*c.log()
        cost = together-own[:, None]-own[None]
        n = len(self.C)
        upper = torch.triu_indices(n, n, 1)
        order = cost[upper[0], upper[1]].argsort()
        used, pairs = set(), []
        for position in order.tolist():
            i, j = int(upper[0][position]), int(upper[1][position])
            if i not in used and j not in used:
                used.update((i, j))
                pairs.append((i, j))
                if len(pairs) == count:
                    break
        return pairs

    def _adjacent(self, count):
        """Control: pool neighbours in time, ignoring content (smallest combined count first)."""
        combined = self.C[:-1]+self.C[1:]
        used, pairs = set(), []
        for i in combined.argsort().tolist():
            if i not in used and i+1 not in used:
                used.update((i, i+1))
                pairs.append((i, i+1))
                if len(pairs) == count:
                    break
        return pairs

    def read(self, queries, scale, mass, gaussian):
        mean = self.U/self.C[:, None]
        logits = queries@mean.T*scale
        if mass:
            logits = logits+self.C.log()
        if gaussian:
            variance = (self.Q/self.C[:, None]-mean*mean).clamp_min(0)
            logits = logits+.5*(queries*queries)@variance.T*scale*scale
        return logits, self.V/self.C[:, None]


def attend(q, k, v, scale, cfg):
    """q: heads x L x d; k, v: key heads x L x d. Causal; exact keys in [block_start-window, t]."""
    heads, length, _ = q.shape
    groups = k.shape[0]
    repeat = heads//groups
    out = torch.empty_like(q)
    slots = [Slots(cfg['slots'], cfg['policy']) for _ in range(groups)] if cfg['slots'] else None
    absorbed, block = 0, cfg['block']
    window = length if cfg['dense'] else cfg['window']
    for start in range(0, length, block):
        end = min(start+block, length)
        low = max(0, start-window)
        if slots and low > absorbed:
            for g in range(groups):
                slots[g].absorb(k[g, absorbed:low].clone(), v[g, absorbed:low].clone())
            absorbed = low
        mask = torch.arange(low, end)[None] > torch.arange(start, end)[:, None]
        for g in range(groups):
            queries = q[g*repeat:(g+1)*repeat, start:end]
            logits = (queries@k[g, low:end].T*scale).masked_fill(mask, -torch.inf)
            values = v[g, low:end]
            if slots and absorbed:
                flat = queries.reshape(-1, queries.shape[-1])
                slot_logits, slot_values = slots[g].read(flat, scale, cfg['mass'], cfg['gaussian'])
                logits = torch.cat((logits, slot_logits.view(repeat, end-start, -1)), -1)
                values = torch.cat((values, slot_values))
            out[g*repeat:(g+1)*repeat, start:end] = logits.softmax(-1)@values
    return out


def install(model, cfg):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_proj(hidden_states).view(shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)   # slots hold keys as a KV cache stores them
        out = attend(q[0], k[0], v[0], self.scaling, cfg)
        return self.o_proj(out.transpose(0, 1).reshape(1, hidden_states.shape[1], -1)), None

    layers = model.model.layers
    for layer in layers[cfg.get('first_layer', 0):]:
        layer.self_attn.forward = forward.__get__(layer.self_attn)


CONFIGS = {
    'denso': dict(dense=True, slots=0),
    'finestra 64': dict(window=64, slots=0),
    'finestra 128': dict(window=128, slots=0),
    'finestra 64 + 64 concetti': dict(window=64, slots=64),
    'finestra 64 + 64 blocchi temporali': dict(window=64, slots=64, policy='chunk'),
    'finestra 64 + 64 concetti, senza log C': dict(window=64, slots=64, mass=False),
    'finestra 64 + 64 concetti, correzione gaussiana': dict(window=64, slots=64, gaussian=True),
    'finestra 64 + 16 concetti': dict(window=64, slots=16),
    'finestra 64 + 256 concetti': dict(window=64, slots=256),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='HuggingFaceTB/SmolLM2-135M')
    parser.add_argument('--text', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--segments', type=int, default=4)
    parser.add_argument('--skip', type=int, default=256, help='Positions before this are not scored.')
    parser.add_argument('--block', type=int, default=16)
    parser.add_argument('--configs', nargs='*', default=list(CONFIGS))
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.text.read_text(), return_tensors='pt').input_ids[0]
    segments = [ids[i*args.length:(i+1)*args.length] for i in range(args.segments)]
    if len(segments[-1]) < args.length:
        parser.error('text too short for the requested segments')
    report = dict(model=args.model, length=args.length, segments=args.segments, scored_from=args.skip,
                  block=args.block, note='exact keys cover [block_start-window, t]: mean exact span is window+block/2',
                  results={})
    for name in args.configs:
        cfg = dict(dense=False, window=0, slots=0, policy='concept', mass=True, gaussian=False, block=args.block)
        cfg.update(CONFIGS[name])
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
        install(model, cfg)
        losses, started = [], time.monotonic()
        with torch.no_grad():
            for segment in segments:
                logits = model(segment[None]).logits[0, args.skip:-1]
                losses.append(torch.nn.functional.cross_entropy(logits, segment[args.skip+1:], reduction='none'))
        loss = torch.cat(losses)
        report['results'][name] = dict(nll=float(loss.mean()), standard_error=float(loss.std()/math.sqrt(len(loss))),
                                       per_segment=[float(part.mean()) for part in losses],
                                       seconds=time.monotonic()-started,
                                       state_vectors_per_key_head=(cfg['window']+cfg['slots']) if not cfg['dense'] else args.length)
        print(name, round(report['results'][name]['nll'], 4), round(report['results'][name]['seconds']), 's', flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
