"""Learning the parameters of the concept network (acan/lm/concept_net.py) as a language model learns them.

The objective is the log-likelihood of the next token under the reading of the concept network; the parameters
(centres, dispersions and masses of the concepts of every layer, the attention between concepts, the token concepts)
move along its gradient. The text is read once, in order, a piece at a time: a piece of the queue teaches the
network after it has been counted. torch is imported here only: reading a trained network needs numpy alone.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .concept_net import ConceptNet


def memberships_logits(q, g, log_s, log_c):
    """log C_k + log N(q; g_k, s_k) up to a constant, for every position and concept of every head, with the square
    expanded into products of matrices (no table position x concept x dimension). q: ... x H x d; g, log_s: H x K x d."""
    inv = torch.exp(-2 * log_s)                                              # H K d
    quad = torch.einsum('...hd,hkd->...hk', q * q, inv) - 2 * torch.einsum('...hd,hkd->...hk', q, g * inv) \
        + (g * g * inv).sum(-1)
    return log_c - .5 * quad - log_s.sum(-1)


def causal_read(weight, t, v, chunk):
    """read_i = sum_k weight_ik S_k(i), S_k(i) = sum_{j<=i} t_jk v_j, without building S for every position: inside a
    block of positions with a product of matrices, across blocks with the sums of the concepts carried along.
    weight, t: B x T x H x K;  v: B x T x H x d."""
    B, T, H, K = t.shape
    d = v.shape[-1]
    out = []
    state = v.new_zeros(B, H, K, d)
    causal = torch.tril(torch.ones(chunk, chunk, dtype=v.dtype, device=v.device))
    for start in range(0, T, chunk):
        w, tc, vc = (x[:, start:start + chunk].transpose(1, 2) for x in (weight, t, v))   # B H C .
        C = w.shape[2]
        scores = (w @ tc.transpose(-1, -2)) * causal[:C, :C]                               # B H C C
        out.append((scores @ vc + w @ state).transpose(1, 2))
        state = state + tc.transpose(-1, -2) @ vc
    return torch.cat(out, 1)


class ConceptAttention(nn.Module):
    def __init__(self, dim, heads, concepts, window):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.g = nn.Parameter(torch.randn(heads, concepts, self.d) * .5)
        self.log_s = nn.Parameter(torch.zeros(heads, concepts, self.d))
        self.log_c = nn.Parameter(torch.zeros(heads, concepts))
        self.A = nn.Parameter(torch.zeros(heads, concepts, concepts))
        self.query = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.near = nn.Parameter(torch.randn(window, dim, dim) * .02)
        self.out = nn.Linear(2 * dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.window = window
        self.chunk = 64

    def forward(self, h, mask):
        B, T, D = h.shape
        q = self.query(h).view(B, T, self.h, self.d)
        v = self.value(h).view(B, T, self.h, self.d)
        t = torch.softmax(memberships_logits(q, self.g, self.log_s, self.log_c), -1) * mask[:, :, None, None]
        mass = torch.cumsum(t, 1) + 1e-6
        weight = torch.einsum('bthk,hkj->bthj', t, torch.softmax(self.A, -1)) / mass
        read = causal_read(weight, t, v, self.chunk).reshape(B, T, D)
        near = sum(F.pad(h, (0, 0, w, 0))[:, :T] @ self.near[w - 1] for w in range(1, self.window + 1))
        h = self.norm1(h + self.out(torch.cat([read, near], -1)))
        return self.norm2(h)


class Network(nn.Module):
    def __init__(self, config, centres=None, frequency=None, groups=None):
        super().__init__()
        V, dim = config['vocabulary'], config['dim']
        self.config = dict(config)
        self.centre = nn.Embedding(V, dim)
        nn.init.normal_(self.centre.weight, std=.1)
        if centres is not None:                              # the coordinates the concept model found for the tokens
            c = torch.as_tensor(np.asarray(centres, np.float32))[:V]
            c = (c - c.mean(0)) / (c.std(0) + 1e-6) * .1
            with torch.no_grad():
                self.centre.weight[:, :min(c.shape[1], dim)] = c[:, :dim]
        f = torch.as_tensor(np.asarray(frequency, np.float32)) if frequency is not None else torch.full((V,), 1. / V)
        self.log_mass = nn.Parameter(torch.log(f / f.sum()))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.layers = nn.ModuleList([ConceptAttention(dim, config['heads'], config['concepts'], config['window'])
                                     for _ in range(config['layers'])])
        self.final = nn.Linear(dim, dim)
        self.grouped = groups is not None
        if self.grouped:
            group = torch.as_tensor(np.asarray(groups), dtype=torch.long)
            self.register_buffer('group', group)
            G = int(group.max()) + 1
            with torch.no_grad():
                centre = torch.zeros(G, dim).index_add_(0, group, self.centre.weight) / torch.bincount(group, minlength=G)[:, None]
            self.group_centre = nn.Parameter(centre)
            self.group_log_mass = nn.Parameter(torch.log(torch.zeros(G).index_add_(0, group, f / f.sum()) + 1e-9))
            order = torch.argsort(group, stable=True)
            counts = torch.bincount(group, minlength=G)
            start = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
            rank = torch.empty_like(order); rank[order] = torch.arange(len(order))
            self._order, self._start, self._size, self._rank = order, start, counts, rank - start[group]

    def states(self, ids, mask):
        h = self.centre(ids)
        for layer in self.layers:
            h = layer(h, mask)
        return self.final(h)

    def _logits(self, h, centres, log_mass):
        return (h @ centres.T - .5 * (centres * centres).sum(-1)) * torch.exp(-2 * self.log_scale) + log_mass

    def target_log_probs(self, ids, mask, targets):
        h = self.states(ids, mask)
        flat, tgt = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        if not self.grouped:
            return torch.log_softmax(self._logits(flat, self.centre.weight, self.log_mass), -1).gather(1, tgt[:, None])[:, 0].view(targets.shape)
        g = self.group[tgt]
        lp = torch.log_softmax(self._logits(flat, self.group_centre, self.group_log_mass), -1).gather(1, g[:, None])[:, 0]
        inside = torch.zeros_like(lp)
        for k in torch.unique(g).tolist():
            sel = (g == k).nonzero()[:, 0]
            members = self._order[self._start[k]:self._start[k] + self._size[k]]
            logits = self._logits(flat[sel], self.centre.weight[members], self.log_mass[members])
            inside = inside.index_put((sel,), torch.log_softmax(logits, -1).gather(1, self._rank[tgt[sel]][:, None])[:, 0])
        return (lp + inside).view(targets.shape)

    def export(self):
        """The parameters as numpy arrays, read by ConceptNet."""
        keep = {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}
        return ConceptNet(self.config, keep)

    @classmethod
    def restore(cls, net):
        """A network that goes on learning from a ConceptNet (the parameters saved in a checkpoint)."""
        groups = net.p['group'].astype(np.int64) if 'group' in net.p else None
        model = cls(net.config, groups=groups)
        state = {k: torch.as_tensor(v) for k, v in net.p.items()}
        if groups is not None:
            state['group'] = torch.as_tensor(groups)
        model.load_state_dict(state)
        return model


def output_groups(token_classes, frequency, limit=512):
    """The groups of the output: the concepts of the tokens, a concept larger than ``limit`` split into blocks of
    tokens of similar frequency."""
    token_classes, frequency = np.asarray(token_classes), np.asarray(frequency)
    group = np.zeros(len(token_classes), np.int64); n = 0
    for k in np.unique(token_classes):
        members = np.flatnonzero(token_classes == k)
        members = members[np.argsort(-frequency[members], kind='stable')]
        for start in range(0, len(members), limit):
            group[members[start:start + limit]] = n; n += 1
    return group


def documents(ids, eos, longest):
    ids = np.asarray(ids, np.int64)
    return [doc[i:i + longest] for doc in np.split(ids, np.flatnonzero(ids == eos) + 1) if len(doc) > 2
            for i in range(0, len(doc) - 1, longest - 1)]


def batches(docs, tokens, shuffle, rng):
    order = rng.permutation(len(docs)) if shuffle else np.arange(len(docs))
    chunk, size = [], 0
    for j in order:
        chunk.append(docs[j]); size += len(docs[j])
        if size >= tokens:
            yield _pad(chunk); chunk, size = [], 0
    if chunk:
        yield _pad(chunk)


def _pad(chunk):
    n = max(len(c) for c in chunk)
    ids = np.zeros((len(chunk), n), np.int64); mask = np.zeros((len(chunk), n), np.float32)
    for r, c in enumerate(chunk):
        ids[r, :len(c)] = c; mask[r, :len(c)] = 1
    return torch.as_tensor(ids), torch.as_tensor(mask)


class Learner:
    """The network and its optimizer across the pieces of a queue (the state is kept between pieces)."""

    def __init__(self, network, lr=3e-3, tokens=2048, longest=512, seed=0):
        self.network, self.tokens, self.longest = network, tokens, longest
        self.opt = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=.01)
        self.lr, self.steps = lr, 0
        self.rng = np.random.default_rng(seed)

    def learn(self, ids, eos, check=None, warmup=200):
        """One pass over a piece of text, in batches of about ``tokens`` tokens. Returns the training loss."""
        self.network.train()
        total, n = 0., 0
        for ids_, mask in batches(documents(ids, eos, self.longest), self.tokens, True, self.rng):
            if check is not None:
                check()
            for group in self.opt.param_groups:
                group['lr'] = self.lr * min(1., (self.steps + 1) / warmup)
            lp = self.network.target_log_probs(ids_[:, :-1], mask[:, :-1], ids_[:, 1:])
            m = mask[:, 1:]
            loss = -(lp * m).sum() / m.sum()
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.opt.step(); self.steps += 1
            total += loss.item() * m.sum().item(); n += m.sum().item()
        self.network.eval()
        return total / max(n, 1)


def concepts_from_states(network, docs, concept_model, rows=8192, seed=0):
    """Start every layer from the concepts the concept model finds on the states the layer reads.

    Layer by layer, the network reads a sample of documents; for every group of concepts the queries of that group are
    a table, and the concept model finds its concepts by its sums (how many, centres, dispersions, masses). The layer
    starts from them; places beyond the concepts found start with a mass close to zero, so that the gradient can make
    them grow only if the prediction needs them. ``concept_model(rows)`` builds the ConceptModel to fit.
    Returns the number of concepts found in every group of every layer.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    found = []
    ids, mask = _pad([docs[j] for j in rng.permutation(len(docs))[:max(1, rows // 64)]])
    with torch.no_grad():
        h = network.centre(ids)
        keep = mask.bool()
        for layer in network.layers:
            B, T, D = h.shape
            q = layer.query(h).view(B, T, layer.h, layer.d)[keep]            # positions x heads x d
            q = q[torch.as_tensor(rng.permutation(len(q))[:rows])].numpy().astype(np.float64)
            counts = []
            for head in range(layer.h):
                frame = pd.DataFrame(q[:, head], columns=[f'q{i}' for i in range(layer.d)])
                model = concept_model(len(frame)).fit_counted_numeric(frame, np.ones(len(frame), np.int64))
                s = model.levels[-1]['table']                   # the working concepts summed over the rows as they are
                C = np.asarray(s['C'], np.float64)
                centre = np.asarray(s['U'] / np.maximum(s['Cm'], 1e-12))
                spread = np.sqrt(np.maximum(np.asarray(s['Q'] / np.maximum(s['Cm'], 1e-12)) - centre ** 2, 1e-6))
                # the statistics are in the scale of the table: back to the scale of the queries
                centre = centre * model.encoder.std + model.encoder.mean
                spread = spread * model.encoder.std
                k = min(len(C), layer.g.shape[1])
                order = np.argsort(-C)[:k]
                layer.g[head, :k] = torch.as_tensor(centre[order], dtype=layer.g.dtype)
                layer.log_s[head, :k] = torch.as_tensor(np.log(spread[order]), dtype=layer.g.dtype)
                mass = np.full(layer.g.shape[1], C[order].sum() * 1e-6)
                mass[:k] = C[order]
                layer.log_c[head] = torch.as_tensor(np.log(mass / mass.sum()), dtype=layer.g.dtype)
                counts.append(k)
            found.append(counts)
            h = layer(h, mask)
    return found
