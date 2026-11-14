"""A small, readable test: is a language model trained from scratch with concept attention any good?

Three tiny transformers, identical except for attention, are trained from scratch on the CPU:

* ``dense``: ordinary attention over every previous token (the best one can do);
* ``window``: only the last ``window`` tokens (the same memory budget as concepts, no concepts);
* ``concepts``: the last ``window`` tokens exact, older ones summarized in ``slots`` concepts
  (the concept-graph layer of the generative laboratory, acan/lm/concept_attention.py).

Two tasks, each with a plain measure:

* ``recall``: pairs "key value" at the start, filler, then the asked keys. The answer lies far
  beyond the exact window, so it has to come from memory. Measure: percentage of right answers
  (chance is 1 / number of values).
* ``text``: characters of a real text. Measure: bits per character on text never seen in training
  (lower is better), and a sample of generated text.

Example: python scripts/probe_small_lm.py --text .data/wikitext2-test.txt --output reports/small-lm.json
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn


class Attention(nn.Module):
    """Dense or sliding-window causal attention with rotary positions."""

    def __init__(self, d, heads, window=None):
        super().__init__()
        self.heads, self.head_dim, self.window = heads, d//heads, window
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, nn.Linear(d, d, bias=False))

    def attend(self, h, position_embeddings):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        b, length, d = h.shape
        split = lambda value: value.view(b, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        logits = q@k.transpose(-1, -2)*self.head_dim**-.5
        i = torch.arange(length)
        mask = i[None] > i[:, None]
        if self.window is not None:                              # only the last ``window`` tokens
            mask = mask | (i[None] < i[:, None]-self.window+1)
        weights = logits.masked_fill(mask, -torch.inf).softmax(-1)
        return self.o_proj((weights@v).transpose(1, 2).reshape(b, length, d))


class Block(nn.Module):
    def __init__(self, d, attention):
        super().__init__()
        self.attention, self.norm1, self.norm2 = attention, nn.RMSNorm(d), nn.RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x, position_embeddings):
        x = x+self.attention.attend(self.norm1(x), position_embeddings)
        return x+self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, kind, d=64, heads=4, layers=2, window=16, slots=8, length=256):
        super().__init__()
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        from acan.lm.concept_attention import ConceptGraphAttention
        config = LlamaConfig(vocab_size=vocab, hidden_size=d, intermediate_size=4*d, num_hidden_layers=layers,
                             num_attention_heads=heads, num_key_value_heads=heads, max_position_embeddings=length)
        self.rotary = LlamaRotaryEmbedding(config)
        self.embed = nn.Embedding(vocab, d)

        def attention(index):
            if kind == 'dense':
                return Attention(d, heads)
            if kind == 'window':
                return Attention(d, heads, window)
            layer = ConceptGraphAttention(config, index, slots=slots, window=window)
            layer.rope_frequency = self.rotary.inv_freq
            return layer
        self.blocks = nn.ModuleList([Block(d, attention(index)) for index in range(layers)])
        self.norm, self.head = nn.RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        position = self.rotary(x, torch.arange(ids.shape[1])[None])
        for block in self.blocks:
            x = block(x, position)
        return self.head(self.norm(x))


# ---- tasks -----------------------------------------------------------------------
def recall_batch(batch, pairs, keys, values, filler, questions, generator):
    """[k v]*pairs, filler, then the asked keys; right after each asked key the model must say its value."""
    noise = keys+values+1                                        # filler tokens are not keys or values
    ids, targets = [], []
    for _ in range(batch):
        chosen = torch.randperm(keys, generator=generator)[:pairs]
        answer = torch.randint(keys, keys+values, (pairs,), generator=generator)
        sequence = torch.stack((chosen, answer), 1).flatten()
        middle = torch.randint(noise, noise+8, (filler,), generator=generator)
        asked = torch.randperm(pairs, generator=generator)[:questions]
        row = torch.cat((sequence, middle, chosen[asked]))
        target = torch.full_like(row, -100)
        target[len(sequence)+filler+torch.arange(questions)] = answer[asked]
        ids.append(row)
        targets.append(target)
    return torch.stack(ids), torch.stack(targets)


def text_batch(data, batch, length, generator):
    start = torch.randint(0, len(data)-length-1, (batch,), generator=generator)
    rows = torch.stack([data[s:s+length+1] for s in start.tolist()])
    return rows[:, :-1], rows[:, 1:]


def train(model, batches, steps, rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=rate, total_steps=steps, pct_start=.1)
    model.train()
    for step in range(steps):
        ids, targets = batches()
        logits = model(ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        schedule.step()
    return float(loss)


@torch.no_grad()
def sample(model, prompt, count, generator, length):
    model.eval()
    ids = prompt.clone()
    for _ in range(count):
        logits = model(ids[:, -length:])[:, -1]
        probability = (logits/.8).softmax(-1)
        ids = torch.cat((ids, torch.multinomial(probability, 1, generator=generator)), 1)
    return ids[0, prompt.shape[1]:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', type=Path, help='a text file for the language task')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kinds', nargs='+', default=['dense', 'window', 'concepts'])
    parser.add_argument('--window', type=int, default=16)
    parser.add_argument('--slots', type=int, default=16)
    parser.add_argument('--recall-steps', type=int, default=1500)
    parser.add_argument('--text-steps', type=int, default=1500)
    parser.add_argument('--pairs', type=int, default=8)
    parser.add_argument('--filler', type=int, default=48)
    parser.add_argument('--keys', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = dict(window=args.window, slots=args.slots, recall=dict(), text=dict())
    keys = values = args.keys
    questions = 4
    length = 2*args.pairs+args.filler+questions
    result['recall_setup'] = dict(pairs=args.pairs, filler=args.filler, keys=keys, values=values,
                                  chance=1/values, sequence=length, nearest_answer_distance=args.filler+1)
    for kind in args.kinds:
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        model = TinyLM(keys+values+1+8+1, kind, d=args.dim, window=args.window, slots=args.slots, length=length)
        started = time.monotonic()
        train(model, lambda: recall_batch(32, args.pairs, keys, values, args.filler, questions, generator), args.recall_steps, args.rate)
        test = torch.Generator().manual_seed(10_000+args.seed)
        ids, targets = recall_batch(512, args.pairs, keys, values, args.filler, questions, test)
        model.eval()
        with torch.no_grad():
            predicted = model(ids).argmax(-1)
        asked = targets != -100
        accuracy = float((predicted[asked] == targets[asked]).float().mean())
        result['recall'][kind] = dict(accuracy=accuracy, seconds=time.monotonic()-started,
                                      parameters=sum(p.numel() for p in model.parameters()))
        print(f"memoria · {kind}: {100*accuracy:.1f}% risposte giuste (caso {100/values:.1f}%) in {time.monotonic()-started:.0f}s", flush=True)
    if args.text:
        raw = args.text.read_text(encoding='utf-8', errors='ignore')
        alphabet = sorted(set(raw))
        index = {c: i for i, c in enumerate(alphabet)}
        data = torch.tensor([index[c] for c in raw], dtype=torch.long)
        split = int(.9*len(data))
        train_data, held = data[:split], data[split:]
        block = 128
        for kind in args.kinds:
            torch.manual_seed(args.seed)
            generator = torch.Generator().manual_seed(args.seed)
            model = TinyLM(len(alphabet), kind, d=args.dim, window=args.window, slots=args.slots, length=block)
            started = time.monotonic()
            train(model, lambda: text_batch(train_data, 16, block, generator), args.text_steps, args.rate)
            evaluation = torch.Generator().manual_seed(20_000)
            losses = []
            model.eval()
            with torch.no_grad():
                for _ in range(8):
                    ids, targets = text_batch(held, 16, block, evaluation)
                    logits = model(ids)[:, block//2:]               # positions with a long context behind
                    losses.append(float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                                    targets[:, block//2:].reshape(-1))))
            bits = sum(losses)/len(losses)/math.log(2)
            prompt = held[:block][None]
            text = ''.join(alphabet[i] for i in sample(model, prompt, 200, torch.Generator().manual_seed(1), block).tolist())
            result['text'][kind] = dict(bits_per_character=bits, seconds=time.monotonic()-started, sample=text)
            print(f"testo · {kind}: {bits:.3f} bit per carattere in {time.monotonic()-started:.0f}s", flush=True)
            print('   ', text[:160].replace('\n', ' '), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
