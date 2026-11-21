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
        concepts = config['concepts'] if isinstance(config['concepts'], list) else [config['concepts']] * config['layers']
        self.config['concepts'] = list(concepts)
        self.layers = nn.ModuleList([ConceptAttention(dim, config['heads'], k, config['window']) for k in concepts])
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

    # ---- growth: a bigger network that reads exactly as the smaller one did -----------------------------------------
    def split_concepts(self, i, scale=1e-3):
        """Every concept of layer i becomes two, with nearly the same centre and half the mass each: the memberships
        split in halves, the sums too, and the reading of the layer is the same. The gradient can then pull them apart."""
        layer = self.layers[i]
        with torch.no_grad():
            g = torch.cat([layer.g, layer.g + torch.randn_like(layer.g) * scale], 1)
            log_s = torch.cat([layer.log_s, layer.log_s], 1)
            log_c = torch.cat([layer.log_c, layer.log_c], 1) - math.log(2)
            A = torch.cat([torch.cat([layer.A, layer.A], 2)] * 2, 1)    # a concept looks at both halves as at the whole
        layer.g, layer.log_s, layer.log_c, layer.A = (nn.Parameter(x.contiguous()) for x in (g, log_s, log_c, A))
        self.config['concepts'][i] = int(g.shape[1])

    def add_layer(self):
        """A new last layer that leaves the states as they are: what it adds starts at zero."""
        dim, last = self.config['dim'], self.layers[-1]
        layer = ConceptAttention(dim, self.config['heads'], int(last.g.shape[1]), self.config['window'])
        with torch.no_grad():
            layer.out.weight.zero_(); layer.out.bias.zero_()
            for norm in (layer.norm1, layer.norm2):                    # a state normalized twice is the same state
                norm.weight.fill_(1.); norm.bias.zero_()
        self.layers.append(layer)
        self.config['layers'] = len(self.layers)
        self.config['concepts'].append(int(last.g.shape[1]))

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



class Grower:
    """Learns a piece at a time and lets the size come from the data, by the rule of every choice of the model.

    After every piece the network is scored on held-out text (``score(net)``: the log-probability of every held-out
    token). If it reads better than the best network so far by more than one standard error of the paired difference,
    it goes on. If not, the best network is taken again and grown, first by splitting every concept in two, then, if
    that did not help either, by one more layer; a growth is kept if the next piece makes it read better. When neither
    helps, the network stops learning (``done``) and ``best`` is the network to keep.
    """

    def __init__(self, network, **learner):
        self.learner_options = learner
        self.learner = Learner(network, **learner)
        self.best, self.best_scores, self.trial, self.done = None, None, None, False
        self.history = []

    def _better(self, scores):
        gain = scores - self.best_scores
        return len(gain) > 1 and gain.mean() > gain.std(ddof=1) / math.sqrt(len(gain) - 1)

    def _grow(self, kind):
        network = Network.restore(self.best)
        if kind == 'concepts':
            for i in range(len(network.layers)):
                network.split_concepts(i)
        else:
            network.add_layer()
        self.learner = Learner(network, **self.learner_options)
        self.trial = kind

    def learn(self, ids, eos, score, check=None):
        if self.done:
            return self.best
        loss = self.learner.learn(ids, eos, check)
        net = self.learner.network.export()
        scores = score(net)
        size = dict(layers=net.config['layers'], concepts=list(net.config['concepts']))
        if self.best is None or self._better(scores):
            self.best, self.best_scores, self.trial = net, scores, None
            self.history.append(dict(size, loss=loss, held_out=float(-scores.mean()), kept=True))
            return self.best
        self.history.append(dict(size, loss=loss, held_out=float(-scores.mean()), kept=False, trial=self.trial))
        if self.trial is None:
            self._grow('concepts')
        elif self.trial == 'concepts':
            self._grow('layer')
        else:
            self.done, self.learner = True, None
        return self.best
