"""A small test of a language model trained from scratch with concept attention.

Three tiny transformers, identical except for attention, are trained from scratch on the CPU:

* ``dense``: ordinary attention over every previous token (the best one can do);
* ``window``: only the last ``window`` tokens (the same memory as concepts, no concepts);
* ``concepts``: the concept model itself (acan/lm/pair_text.py), no network: the training sequences
  become rows of concepts of their pairs, the weights of its readings are chosen by the judgment.

Two tasks with a plain measure each:

* memory: pairs "key value", filler, then the asked keys; the answer lies far beyond the exact
  window. Measure: percentage of right answers (chance is 1 / number of values);
* text: characters of a text. Measure: bits per character on text never seen in training, and
  a generated sample.
"""
import json
import math
import time

import torch
from torch import nn

KINDS = ('dense', 'window', 'concepts')
BUDGETS = {
    'quick': dict(memory_steps=800, text_steps=500, far=False),
    'full': dict(memory_steps=2000, text_steps=1500, far=True),
}
SETTINGS = dict(dim=64, heads=4, layers=2, window=16, slots=16, rate=3e-3, pairs=8, filler=48, keys=16, block=128,
                far_filler=176, far_layers=3, cost_lengths=(512, 1024, 2048, 4096))


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
            if kind == 'concepts':                               # the layer of new projects (acan/lm/graph_scan.py)
                from acan.lm.graph_scan import GraphScanAttention
                return GraphScanAttention(d, heads, window=window, bins=slots)
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


def train(model, batches, steps, rate, check=lambda step: None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=rate, total_steps=steps, pct_start=.1)
    model.train()
    curve, window = [], []
    every = max(1, steps//40)
    for step in range(steps):
        check(step)
        ids, targets = batches()
        logits = model(ids)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        optimizer.zero_grad()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        schedule.step()
        window.append(float(loss.detach()))
        if len(window) == every or step == steps-1:
            curve.append(dict(step=step+1, loss=sum(window)/len(window), gradient_norm=float(norm)))
            window = []
    return curve


@torch.no_grad()
def sample(model, prompt, count, generator, length):
    model.eval()
    ids = prompt.clone()
    for _ in range(count):
        logits = model(ids[:, -length:])[:, -1]
        probability = (logits/.8).softmax(-1)
        ids = torch.cat((ids, torch.multinomial(probability, 1, generator=generator)), 1)
    return ids[0, prompt.shape[1]:]



def read_corpus(path, jsonl, limit=3*1024*1024):
    """At most ``limit`` bytes of text of a corpus: plain text, or the ``text`` field of JSONL records."""
    with open(path, 'rb') as stream:
        raw = stream.read(limit).decode('utf-8', errors='ignore')
    if not jsonl:
        return raw
    texts = []
    for line in raw.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue                                             # the last line may be cut by the limit
        if isinstance(record, dict) and isinstance(record.get('text'), str):
            texts.append(record['text'])
    return '\n\n'.join(texts)


def concept_model(train, tune, vocabulary, eos, check):
    """The generative concept model (acan/lm/pair_text.py) learned on ``train`` and judged on ``tune``."""
    import numpy as np
    from .pair_text import PairText
    train = np.asarray(train, np.int32)
    model = PairText(vocabulary, eos, 256 << 20)
    for start in range(0, len(train), 200_000):
        check()
        model.add(train[start:start+200_000], train, tune, check)
    model.judge(tune)
    return model


def concept_memory(pairs, filler, values, settings, seed, check, varied=False):
    """The memory task with the concept model (acan/lm/pair_text.py): the training sequences are its table.

    The same sequences a network would read in its training steps; the weights of the levels are chosen
    by the judgment on other training sequences; the measure is the same as for the networks.
    """
    import numpy as np
    keys, questions = max(values, 2*pairs), 4
    vocabulary = keys+values+1+8
    generator = torch.Generator().manual_seed(seed)

    def sequences(count, gap_generator, fixed):
        rows, targets = [], []
        for _ in range(count):
            gap = fixed if fixed is not None else int(torch.randint(0, filler+1, (1,), generator=gap_generator))
            ids, target = recall_batch(1, pairs, keys, values, gap, questions, gap_generator)
            rows.append(ids[0].numpy()); targets.append(target[0].numpy())
        return rows, targets
    started = time.monotonic()
    count = settings['memory_steps']*32                           # the sequences of a network's training
    rows, _ = sequences(count, generator, None if varied else filler)
    check()
    tune_rows, tune_targets = sequences(200, torch.Generator().manual_seed(1000+seed), None if varied else filler)
    texts = [np.r_[row, vocabulary-1] for row in tune_rows]
    model = concept_model(np.concatenate([np.r_[row, vocabulary-1] for row in rows]), texts, vocabulary, vocabulary-1, check)
    check()
    ids, targets = recall_batch(512, pairs, keys, values, filler, questions, torch.Generator().manual_seed(10_000+seed))
    ids, targets = ids.numpy(), targets.numpy()
    right, from_sequence = [], []
    for row, target in zip(ids, targets):
        stored = set(row[1:2*pairs:2].tolist())
        for at in np.flatnonzero(target != -100):
            guess = keys+int(np.argmax(model.distribution(row[:at+1])[keys:keys+values]))
            right.append(guess == target[at]); from_sequence.append(guess != target[at] and guess in stored)
    right, from_sequence = np.array(right), np.array(from_sequence)
    by_question = right.reshape(len(ids), -1).mean(0)
    return dict(accuracy=float(right.mean()), seconds=time.monotonic()-started, parameters=0, concepts=sum(model.concepts()),
                wrong_but_from_the_sequence=float(from_sequence.mean()), wrong_not_in_the_sequence=float((~right & ~from_sequence).mean()),
                accuracy_by_question=[float(value) for value in by_question], training_curve=[])


def memory(kind, pairs, filler, values, settings, shape, seed, check, varied=False):
    """Train one model on the memory task; what it recalls and how it fails.

    ``varied``: training batches have fillers from 0 to ``filler`` tokens; the measure is at ``filler``.
    """
    if kind == 'concepts':
        return concept_memory(pairs, filler, values, settings, seed, check, varied)
    keys, questions = max(values, 2*pairs), 4
    length = 2*pairs+filler+questions
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = TinyLM(keys+values+1+8, kind, length=length, **shape)
    started = time.monotonic()
    def batch():
        gap = int(torch.randint(0, filler+1, (1,), generator=generator)) if varied else filler
        return recall_batch(32, pairs, keys, values, gap, questions, generator)
    curve = train(model, batch, settings['memory_steps'], settings['rate'], lambda step: check())
    ids, targets = recall_batch(512, pairs, keys, values, filler, questions, torch.Generator().manual_seed(10_000+seed))
    model.eval()
    with torch.no_grad():
        predicted = model(ids).argmax(-1)
    asked = targets != -100
    right = predicted[asked] == targets[asked]
    # Debugging: a wrong answer that is another value of the same sequence means the model found
    # the pairs but read the wrong one; any other wrong answer means it did not find them.
    stored = ids[:, 1:2*pairs:2]
    present = (predicted[:, :, None] == stored[:, None, :]).any(-1)[asked]
    by_question = right.reshape(len(ids), -1).float().mean(0)
    return dict(accuracy=float(right.float().mean()), seconds=time.monotonic()-started,
                parameters=sum(item.numel() for item in model.parameters()),
                wrong_but_from_the_sequence=float((~right & present).float().mean()),
                wrong_not_in_the_sequence=float((~right & ~present).float().mean()),
                accuracy_by_question=[float(value) for value in by_question], training_curve=curve)


def kept(length, window, slots):
    """Vectors a concept layer holds after ``length`` tokens: the window and the blocks of its older tokens."""
    old = max(length-window, 0)
    return min(length, window)+sum(min(1 << level, slots) for level in range(old.bit_length()) if old >> level & 1)


@torch.no_grad()
def cost(shape, lengths, check):
    """Time per token of reading contexts of growing length, and what each model keeps.

    Ordinary attention compares every token with all the previous ones: its time per token grows with
    the context and its memory holds every token. The concept model reads, for every token, one concept
    per level of context and the concepts of the text in hand: its time per token does not depend on how
    long the text is. The network is untrained, the concept model counts random text: only the cost is measured.
    """
    import numpy as np
    rows = []
    TinyLM(64, 'dense', length=64, **shape).eval()(torch.zeros(1, 64, dtype=torch.long))   # warm-up
    text = np.random.default_rng(0).integers(0, 63, 100_000)
    concept = concept_model(text, [text[-2048:]], 64, 63, lambda: None)
    for length in lengths:
        check()
        row = dict(length=length)
        model = TinyLM(64, 'dense', length=length, **shape).eval()
        ids = torch.randint(0, 64, (1, length), generator=torch.Generator().manual_seed(0))
        started = time.perf_counter()
        model(ids)
        row['dense'] = dict(milliseconds_per_token=1000*(time.perf_counter()-started)/length, kept_vectors=length)
        started = time.perf_counter()
        concept.logp(ids[0].numpy())
        row['concepts'] = dict(milliseconds_per_token=1000*(time.perf_counter()-started)/length,
                               kept_vectors=concept.context_tokens())
        rows.append(row)
    x = torch.tensor([row['length'] for row in rows], dtype=torch.float64)
    y = torch.tensor([row['dense']['milliseconds_per_token'] for row in rows], dtype=torch.float64)
    slope = float(((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum())
    intercept = float(y.mean())-slope*float(x.mean())
    flat = float(torch.tensor([row['concepts']['milliseconds_per_token'] for row in rows]).mean())
    crossing = (flat-intercept)/slope if slope > 0 else None
    return dict(rows=rows, dense_slope=slope, concepts_flat=flat,
                faster_beyond=None if crossing is None else max(0., crossing))


def run(text, budget='quick', check=lambda: None, report=lambda message: None, seed=0, settings=None):
    """The three models on the memory task and on ``text``; returns what they achieved."""
    settings = {**SETTINGS, **BUDGETS[budget], **(settings or {})}
    torch.set_num_threads(1)
    if len(text) < 20*settings['block']:
        raise ValueError('Testo troppo breve per la prova: servono almeno qualche migliaio di caratteri.')
    result = dict(budget=budget, settings=settings, memory={}, text={})
    values = settings['keys']
    result['memory_setup'] = dict(pairs=settings['pairs'], chance=1/values, nearest_answer_distance=settings['filler']+1,
                                  exact_tokens=settings['window'])
    shape = dict(d=settings['dim'], heads=settings['heads'], layers=settings['layers'], window=settings['window'], slots=settings['slots'])
    for kind in KINDS:
        report(f'Memoria · {kind}')
        result['memory'][kind] = memory(kind, settings['pairs'], settings['filler'], values, settings, shape, seed, check)
    # The answer much farther away (full budget only). Measured during development: with two layers
    # and a fixed distance not even ordinary attention learns it in 2000 steps (25%, as concepts), so
    # both models get a third layer and are trained on distances from 0 to ``far_filler``; then
    # ordinary attention reaches 100%. More facts than concepts was not learned by ordinary attention
    # either at this size (16 or 32 pairs: 14-18%), so it is not measured here.
    result['memory_cases'] = {}
    if settings['far']:
        filler = settings['far_filler']
        case = dict(pairs=settings['pairs'], filler=filler, chance=1/values, nearest_answer_distance=filler+1,
                    slots=settings['slots'], layers=settings['far_layers'])
        for kind in ('dense', 'concepts'):
            report(f'Memoria lontana · {kind}')
            case[kind] = memory(kind, settings['pairs'], filler, values, settings, dict(shape, layers=settings['far_layers']),
                                seed, check, varied=True)
        result['memory_cases']['distance'] = case
    report('Costo per token')
    result['cost'] = cost(shape, settings['cost_lengths'], check)
    common = sorted(set(text), key=lambda character: -text.count(character))[:200]   # rarer characters become one symbol
    alphabet = common+['\ufffd']
    index = {character: place for place, character in enumerate(common)}
    data = torch.tensor([index.get(character, len(common)) for character in text], dtype=torch.long)
    block = settings['block']
    if len(data) < 20*block:
        raise ValueError('Testo troppo breve per la prova: servono almeno qualche migliaio di caratteri.')
    split = int(.9*len(data))
    train_data, held = data[:split], data[split:]
    for kind in KINDS:
        report(f'Scrittura · {kind}')
        if kind == 'concepts':
            result['text'][kind] = concept_text(train_data, held, alphabet, block, check)
            continue
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = TinyLM(len(alphabet), kind, length=block, **shape)
        started = time.monotonic()
        curve = train(model, lambda: text_batch(train_data, 16, block, generator), settings['text_steps'], settings['rate'], lambda step: check())
        evaluation = torch.Generator().manual_seed(20_000)
        losses = []
        model.eval()
        with torch.no_grad():
            for _ in range(8):
                ids, targets = text_batch(held, 16, block, evaluation)
                logits = model(ids)[:, block//2:]                   # positions with a long context behind
                losses.append(float(nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets[:, block//2:].reshape(-1))))
        prompt = held[:block][None]
        produced = ''.join(alphabet[i] for i in sample(model, prompt, 300, torch.Generator().manual_seed(1), block).tolist())
        result['text'][kind] = dict(bits_per_character=sum(losses)/len(losses)/math.log(2), seconds=time.monotonic()-started,
                                    prompt=''.join(alphabet[i] for i in prompt[0, -80:].tolist()), sample=produced,
                                    parameters=sum(item.numel() for item in model.parameters()), training_curve=curve)
    result['efficiency'] = efficiency(result, 16*block)
    return result


def concept_text(train_data, held, alphabet, block, check):
    """The text with the concept model: the training text is read once, measured as the networks are."""
    import numpy as np
    started = time.monotonic()
    data, held = train_data.numpy(), held.numpy()
    counted, tune = data[:-len(data)//10], data[-len(data)//10:]       # the weights are judged on text never counted
    pieces = [tune[start:start+block] for start in range(0, len(tune)-block, block)][:200]
    # The alphabet has no end-of-text character: one more symbol, never seen, marks the end of a document.
    model = concept_model(counted, pieces, len(alphabet)+1, len(alphabet), check)
    evaluation = torch.Generator().manual_seed(20_000)
    values = []
    for _ in range(8):
        ids, targets = text_batch(torch.from_numpy(held), 16, block, evaluation)
        for row, target in zip(ids.numpy(), targets.numpy()):
            values.append(model.logp(np.r_[row, target[-1:]], np.arange(block//2+1, block+1)))
    prompt = held[:block]
    produced = model.generate(prompt, 300, stop={len(alphabet)}, temperature=.8, seed=1)
    return dict(bits_per_character=float(-np.concatenate(values).mean()/math.log(2)), seconds=time.monotonic()-started,
                prompt=''.join(alphabet[i] for i in prompt[-80:].tolist()), sample=''.join(alphabet[i] for i in produced),
                parameters=0, concepts=sum(model.concepts()), training_curve=[])


def reached(curve, target):
    """The first step at which the training loss is at or below ``target``; None if never."""
    return next((point['step'] for point in curve if point['loss'] <= target), None)


def efficiency(result, characters_per_step):
    """How much training each model needs to reach the same level: the question of the data needed.

    Text: the level is the worse of the two final training losses (ordinary attention, concepts),
    so both reach it; the measure is the characters read before reaching it. Memory: the level is
    a loss of 0.5 nats, well below guessing (log 16 = 2.77), i.e. answers mostly right; the measure
    is the training steps. The training loss is measured on batches not seen before, averaged over
    a fortieth of the training, so differences under a few percent are noise.
    """
    if not result['text']['concepts']['training_curve']:          # the concept model counts the text once
        level = result['text']['dense']['training_curve'][-1]['loss'] if result['text']['dense']['training_curve'] else None
        return dict(text=dict(level=level, characters=dict(dense=None, concepts=None)), memory=dict(level=.5, steps={}, ratio=None),
                    counted_once=True)
    text = {kind: result['text'][kind]['training_curve'] for kind in ('dense', 'concepts')}
    level = max(curve[-1]['loss'] for curve in text.values())
    steps = {kind: reached(curve, level) for kind, curve in text.items()}
    out = dict(text=dict(level=level, characters={kind: None if step is None else step*characters_per_step for kind, step in steps.items()}))
    if all(steps.values()):
        out['text']['ratio'] = steps['concepts']/steps['dense']
    memory = {kind: reached(result['memory'][kind]['training_curve'], .5) for kind in ('dense', 'concepts')}
    out['memory'] = dict(level=.5, steps=memory, ratio=memory['concepts']/memory['dense'] if all(memory.values()) else None)
    return out


class Stopped(Exception):
    pass


def run_in_worker(root, identifier, stop_event=None):
    """The test requested from the laboratory: reads the corpus, reports the phase, stores the result."""
    from .store import Store
    store = Store(root)
    test = store.small_test(identifier)
    corpus = store.corpus(test['request']['corpus_id'], test['owner'], ready=True)

    def check():
        if stop_event is not None and stop_event.is_set():
            raise Stopped()
        if store.small_test(identifier)['status'] == 'cancelling':
            raise Stopped()
    try:
        text = read_corpus(store.directory('corpora', corpus['id'])/'corpus', corpus['name'].lower().endswith('.jsonl'))
        result = run(text, test['request']['budget'], check=check,
                     report=lambda phase: store.update_small_test(identifier, phase=phase))
        result['corpus'] = dict(name=corpus['name'], characters=len(text))
        store.update_small_test(identifier, status='completed', phase='Finita', result=result)
    except Stopped:
        store.update_small_test(identifier, status='cancelled', phase='Fermata')
    except Exception as exc:
        store.update_small_test(identifier, status='failed', error=str(exc))
        raise
