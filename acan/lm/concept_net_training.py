"""Learning the parameters of the concept network (acan/lm/concept_net.py) as a language model learns them.

The objective is the log-likelihood of the next token under the reading of the concept network; the parameters
(centres, dispersions and masses of the concepts of every layer, the attention between concepts, the token concepts)
move along its gradient. The text is read once, in order, a piece at a time: a piece of the queue teaches the
network after it has been counted. torch is imported here only: reading a trained network needs numpy alone.
"""
import copy
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
    def __init__(self, dim, heads, concepts, window, head_dim=None):
        super().__init__()
        self.h, self.d = heads, int(head_dim or dim // heads)
        self.g = nn.Parameter(torch.randn(heads, concepts, self.d) * .5)
        self.log_s = nn.Parameter(torch.zeros(heads, concepts, self.d))
        self.log_c = nn.Parameter(torch.zeros(heads, concepts))
        self.A = nn.Parameter(torch.zeros(heads, concepts, concepts))
        self.query = nn.Linear(dim, heads * self.d, bias=False)
        self.value = nn.Linear(dim, heads * self.d, bias=False)
        self.near = nn.Parameter(torch.randn(window, dim, dim) * .02)
        self.out = nn.Linear(heads * self.d + dim, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        # how open the layer is: its state is h + gate (layer(h) - h). An existing layer is open (1); a layer just
        # added is closed (0), so it leaves the state as it is, whatever the state is, and the gradient opens it
        self.gate = nn.Parameter(torch.ones(()))
        self.window = window
        self.chunk = 64

    def forward(self, h, mask):
        B, T, D = h.shape
        q = self.query(h).view(B, T, self.h, self.d)
        v = self.value(h).view(B, T, self.h, self.d)
        t = torch.softmax(memberships_logits(q, self.g, self.log_s, self.log_c), -1) * mask[:, :, None, None]
        mass = torch.cumsum(t, 1) + 1e-6
        weight = torch.einsum('bthk,hkj->bthj', t, torch.softmax(self.A, -1)) / mass
        read = causal_read(weight, t, v, self.chunk).reshape(B, T, self.h * self.d)
        near = sum(F.pad(h, (0, 0, w, 0))[:, :T] @ self.near[w - 1] for w in range(1, self.window + 1))
        new = self.norm2(self.norm1(h + self.out(torch.cat([read, near], -1))))
        return h + self.gate * (new - h)


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
        self.config.setdefault('head_dim', dim // config['heads'])
        self.layers = nn.ModuleList([ConceptAttention(dim, config['heads'], k, config['window'], self.config['head_dim'])
                                     for k in concepts])
        self.final = nn.Linear(dim, dim)
        # The memory of the document, learned: every position read leaves an address (a projection of its state) with
        # the token that followed it; a position reads the tokens that followed the addresses most like its own. One
        # more address, "none is like mine" (its likeness learned from the state), stands for the reading of the
        # network: the memory counts only as far as some address is more alike than that one.
        self.config.setdefault('cache', dim)                   # its addresses are as wide as the state
        self.config.setdefault('sentinel', True)
        c = self.config['cache']
        if c:
            self.cache_query = nn.Linear(dim, c, bias=False)
            self.cache_key = nn.Linear(dim, c, bias=False)
            self.cache_valve = nn.Linear(dim, 1)
            with torch.no_grad():                               # it starts by counting little: "none is like mine"
                self.cache_valve.bias.fill_(math.log(512.) if self.config['sentinel'] else -2.)   # outweighs a piece of 512 addresses
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
            # the members of every group in one table (padded): the token inside its group is read for all the
            # positions at once, a product over the group's members only
            S = int(counts.max())
            slots = torch.arange(S)[None]
            self._members = torch.where(slots < counts[:, None], order[(start[:, None] + slots).clamp(max=len(order) - 1)],
                                        torch.zeros((), dtype=torch.long))
            self._valid = slots < counts[:, None]

    def states(self, ids, mask):
        h = self.centre(ids)
        for layer in self.layers:
            h = layer(h, mask)
        return self.final(h)

    def _logits(self, h, centres, log_mass):
        return (h @ centres.T - .5 * (centres * centres).sum(-1)) * torch.exp(-2 * self.log_scale) + log_mass

    def _cache(self, h, mask, targets, net_lp):
        """Mix the reading of the network with the memory of the document (only the true token is needed)."""
        if not self.config.get('cache'):
            return net_lp
        B, T, _ = h.shape
        q, k = self.cache_query(h), self.cache_key(h)
        scores = q @ k.transpose(1, 2) / math.sqrt(q.shape[-1])                  # B T T
        earlier = torch.tril(torch.ones(T, T, dtype=torch.bool, device=h.device), -1)
        allowed = earlier[None] & mask[:, None, :].bool()
        scores = scores.masked_fill(~allowed, float('-inf'))
        anything = allowed.any(-1)
        a = torch.softmax(scores.masked_fill(~anything[..., None], 0.), -1) * anything[..., None]
        same = (targets[:, None, :] == targets[:, :, None]).to(a.dtype)          # the token after j is the true one
        z = self.cache_valve(h)[..., 0]
        if self.config.get('sentinel'):                        # "none is like mine" is one more address
            both = torch.softmax(torch.cat([z[..., None], scores], -1), -1)
            return torch.log(both[..., 0] * net_lp.exp() + (both[..., 1:] * same).sum(-1) + 1e-30)
        p_cache = (a * same).sum(-1)
        g = torch.sigmoid(z) * anything
        return torch.log((1 - g) * net_lp.exp() + g * p_cache + 1e-30)

    def target_log_probs(self, ids, mask, targets):
        h = self.states(ids, mask)
        flat, tgt = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        if not self.grouped:
            lp = torch.log_softmax(self._logits(flat, self.centre.weight, self.log_mass), -1).gather(1, tgt[:, None])[:, 0].view(targets.shape)
            return self._cache(h, mask, targets, lp)
        g = self.group[tgt]
        lp = torch.log_softmax(self._logits(flat, self.group_centre, self.group_log_mass), -1).gather(1, g[:, None])[:, 0]
        members, valid = self._members[g], self._valid[g]                         # N x S: the tokens of each group
        e = self.centre.weight[members]                                          # N x S x D
        logits = ((flat[:, None, :] * e).sum(-1) - .5 * (e * e).sum(-1)) * torch.exp(-2 * self.log_scale) + self.log_mass[members]
        inside = torch.log_softmax(logits.masked_fill(~valid, float('-inf')), -1).gather(1, self._rank[tgt][:, None])[:, 0]
        return self._cache(h, mask, targets, (lp + inside).view(targets.shape))

    # ---- growth: a bigger network that reads exactly as the smaller one did -----------------------------------------
    def split_concepts(self, i, scale=.5):
        """Every concept of layer i becomes two with half the mass each, their centres on either side of the old one:
        the memberships split, the sums too, and the reading of the layer is nearly the same. The gradient then takes
        them apart."""
        layer = self.layers[i]
        with torch.no_grad():
            # the two halves apart by a share of their dispersion, in opposite ways: together they still read as the
            # concept did, and they differ enough for the gradient to take each its own way
            e = torch.randn_like(layer.g) * torch.exp(layer.log_s) * scale
            g = torch.cat([layer.g + e, layer.g - e], 1)
            log_s = torch.cat([layer.log_s, layer.log_s], 1)
            log_c = torch.cat([layer.log_c, layer.log_c], 1) - math.log(2)
            A = torch.cat([torch.cat([layer.A, layer.A], 2)] * 2, 1)    # a concept looks at both halves as at the whole
        layer.g, layer.log_s, layer.log_c, layer.A = (nn.Parameter(x.contiguous()) for x in (g, log_s, log_c, A))
        self.config['concepts'][i] = int(g.shape[1])

    def add_layer(self):
        """A new last layer that leaves the states as they are: it starts closed, and what it adds starts at zero."""
        dim, last = self.config['dim'], self.layers[-1]
        layer = ConceptAttention(dim, self.config['heads'], int(last.g.shape[1]), self.config['window'], self.config['head_dim'])
        with torch.no_grad():
            layer.out.weight.zero_(); layer.out.bias.zero_()
            layer.gate.zero_()                                         # closed: the state goes through as it is
        self.layers.append(layer)
        self.config['layers'] = len(self.layers)
        self.config['concepts'].append(int(last.g.shape[1]))

    def _like(self, **change):
        """A network with the configuration changed, its own parameters at their start (to be filled)."""
        config = dict(self.config, **change)
        groups = self.group.numpy() if self.grouped else None
        return Network(config, groups=groups)

    def widened(self, noise=1e-2):
        """Twice as wide (the state and every group of concepts): every coordinate is copied, so the states are the old
        ones written twice; the maps that read them take half from each copy, the dispersions grow by sqrt 2 and the
        masses of the concepts are corrected, so memberships, readings and the next token are the same. The two copies
        of what is written get opposite perturbations: their mean, which the next maps read, is the old value, but the
        copies differ, so the gradient can pull them apart (copied exactly they would move together forever)."""
        grown = self._like(dim=2 * self.config['dim'], head_dim=2 * self.config['head_dim'],
                           cache=2 * (self.config.get('cache') or 0))
        grown.load_state_dict(self._widen({k: v.detach().clone() for k, v in self.state_dict().items()}, noise=noise))
        return grown

    def _widen(self, values, moments=False, noise=0.):
        """The map of ``widened`` on a dictionary of tensors named as the parameters: the parameters themselves, or
        (``moments``) the statistics the optimizer keeps for them. A copy of a coordinate moves as the original did, so
        it takes the original's statistics, without the corrections that keep the reading the same."""
        D, H, d = self.config['dim'], self.config['heads'], self.config['head_dim']
        half = (lambda W: W) if moments else (lambda W: W / 2)
        def apart(W, dim):                                     # two copies along dim, perturbed in opposite ways
            e = 0. if moments or not noise else torch.randn_like(W) * noise * (W.std() + 1e-6)
            return torch.cat([W + e, W - e], dim)
        two = lambda x: apart(x, -1)
        half_in = lambda W: torch.cat([half(W), half(W)], -1)          # out x in: reads both copies of the input
        rows = lambda W: apart(W, 0)                                   # writes its output twice
        heads_rows = lambda W: torch.cat([W.view(H, d, -1)] * 2, 1).reshape(2 * H * d, -1)
        new = {}
        for k, v in values.items():
            if k in ('centre.weight', 'group_centre'):
                v = two(v)
            elif k == 'log_scale':
                v = v if moments else v + math.log(2) / 2              # |h|, |e|: the products count twice
            elif k.startswith('layers.'):
                part = k.split('.', 2)[2]
                if part == 'g':
                    v = two(v)
                elif part == 'log_s':
                    v = torch.cat([v, v], -1) if moments else torch.cat([v + math.log(2) / 2] * 2, -1)
                elif part == 'log_c' and not moments:
                    log_s = values[k.replace('log_c', 'log_s')]
                    v = v + log_s.sum(-1) + d * math.log(2)
                elif part in ('query.weight', 'value.weight'):
                    v = half_in(heads_rows(v))
                elif part == 'near':
                    v = torch.cat([torch.cat([half(v), half(v)], -1)] * 2, -2)
                elif part == 'out.weight':
                    read, near = v[:, :H * d], v[:, H * d:]
                    read = torch.cat([half(read.view(D, H, d))] * 2, -1).reshape(D, 2 * H * d)
                    v = rows(torch.cat([read, half_in(near)], -1))
                elif part in ('out.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
                    v = torch.cat([v, v], -1)
            elif k == 'final.weight':
                v = rows(half_in(v))
            elif k == 'final.bias':
                v = torch.cat([v, v], -1)
            elif k in ('cache_query.weight', 'cache_key.weight'):
                v = rows(half_in(v)) * (1 / math.sqrt(2) if k == 'cache_key.weight' and not moments else 1)
            elif k == 'cache_valve.weight':
                v = half_in(v)
            new[k] = v
        return new

    def more_heads(self):
        """Twice the groups of concepts in every layer: the new groups start with nothing to say (their reading enters
        the layer with zero weight), so the network reads as before; the gradient decides what they hold."""
        H, d, D = self.config['heads'], self.config['head_dim'], self.config['dim']
        grown = self._like(heads=2 * H)
        fresh = grown.state_dict()
        new = {}
        for k, v in self.state_dict().items():
            if not k.startswith('layers.'):
                new[k] = v; continue
            part, f = k.split('.', 2)[2], fresh[k]
            if part in ('g', 'log_s', 'log_c', 'A'):
                v = torch.cat([v, f[H:]], 0)
            elif part in ('query.weight', 'value.weight'):
                v = torch.cat([v, f[H * d:]], 0)
            elif part == 'out.weight':
                v = torch.cat([v[:, :H * d], torch.zeros(D, H * d), v[:, H * d:]], -1)
            new[k] = v
        grown.load_state_dict(new)
        return grown

    def wider_window(self):
        """Twice the distances among the nearby states; the maps of the new ones start at zero."""
        grown = self._like(window=2 * self.config['window'])
        new = {k: (torch.cat([v, torch.zeros_like(v)], 0) if k.endswith('.near') else v)
               for k, v in self.state_dict().items()}
        grown.load_state_dict(new)
        return grown

    def export(self):
        """The parameters as numpy arrays, read by ConceptNet."""
        keep = {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}   # a copy: it must not change as the network learns on
        return ConceptNet(self.config, keep)

    @classmethod
    def restore(cls, net):
        """A network that goes on learning from a ConceptNet (the parameters saved in a checkpoint)."""
        groups = net.p['group'].astype(np.int64) if 'group' in net.p else None
        config = dict(net.config, cache=net.config.get('cache', 0) if 'cache_query.weight' in net.p else 0)
        config.setdefault('sentinel', False)                   # older networks: the memory with its valve
        model = cls(config, groups=groups)
        state = {k: torch.as_tensor(v) for k, v in net.p.items()}
        for i in range(len(model.layers)):                             # older networks: every layer open
            state.setdefault(f'layers.{i}.gate', torch.ones(()))
        if groups is not None:
            state['group'] = torch.as_tensor(groups)
        model.load_state_dict(state)
        return model


def frequency_groups(frequency):
    """Groups of the output from the frequencies alone: the tokens from the most frequent down, in groups of about
    the square root of the vocabulary, so that a token is read among as many groups as tokens of its group."""
    frequency = np.asarray(frequency, float)
    size = max(1, int(math.ceil(math.sqrt(len(frequency)))))
    rank = np.empty(len(frequency), np.int64)
    rank[np.argsort(-frequency, kind='stable')] = np.arange(len(frequency))
    return rank // size


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


def documents(ids, eos, longest, begin=False):
    """The documents of a stream (each ends with ``eos``), in pieces of at most ``longest`` tokens. With ``begin``
    every document starts with ``eos`` too, so that its first token is read after the end of the one before."""
    ids = np.asarray(ids, np.int64)
    docs = np.split(ids, np.flatnonzero(ids == eos) + 1)
    if begin:
        docs = [np.r_[eos, doc] for doc in docs if len(doc)]
    return [doc[i:i + longest] for doc in docs if len(doc) > 2 for i in range(0, len(doc) - 1, longest - 1)]


def held_out(network, docs, limit=None, batch=4096):
    """log p of the true token at every position of held-out documents (arrays of ids), in their order: the network
    reads them in batches, as it learns, without the gradient. ``limit``: only the first documents, about that many
    tokens."""
    network.eval()
    if limit is not None:
        kept, size = [], 0
        for d in docs:
            if size >= limit:
                break
            kept.append(d); size += len(d)
        docs = kept
    docs = [np.asarray(d, np.int64) for d in docs if len(d) >= 2]
    # documents of like length together, and as many as fit ``batch`` tokens once padded: a batch of short documents
    # and one long one would pad them all to the long one (the memory of the document is the square of the length)
    order = sorted(range(len(docs)), key=lambda j: len(docs[j]))
    chunks, chunk = [], []
    for j in order:
        if chunk and (len(chunk) + 1) * len(docs[j]) > batch:
            chunks.append(chunk); chunk = []
        chunk.append(j)
    if chunk:
        chunks.append(chunk)
    read = [None] * len(docs)
    with torch.no_grad():
        for chunk in chunks:
            ids, mask = _pad([docs[j] for j in chunk])
            lp = network.target_log_probs(ids[:, :-1], mask[:, :-1], ids[:, 1:])
            m = mask[:, 1:].bool()
            for r, j in enumerate(chunk):
                read[j] = lp[r][m[r]].numpy()
    return np.concatenate(read) if read else np.zeros(0)


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

    def __init__(self, network, lr=3e-3, tokens=2048, longest=512, seed=0, weight=None):
        """``weight``: how much every token weighs when it is the one to read (one per token of the vocabulary),
        as the concept model weighs its attributes: a target as much as all the features together."""
        self.network, self.tokens, self.longest = network, tokens, longest
        self.weight = None if weight is None else torch.as_tensor(np.asarray(weight, np.float32))
        self.opt = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=0.)
        self.decay = .01
        self.anchor = None                                     # where the parameters are pulled back to (None: zero)
        self.lr, self.steps = lr, 0
        self.seen = 0                                          # tokens read so far, batch after batch
        self.warmup = 200                                      # steps of the start, at most one pass (Grower)
        self.rng = np.random.default_rng(seed)

    def learn(self, ids, eos, check=None, warmup=None):
        """One pass over a piece of text, in batches of about ``tokens`` tokens. Returns the training loss."""
        return self.learn_documents(documents(ids, eos, self.longest), check, warmup)

    def learn_documents(self, docs, check=None, warmup=None):
        warmup = self.warmup if warmup is None else warmup
        self.network.train()
        total, n = 0., 0
        for ids_, mask in batches(docs, self.tokens, True, self.rng):
            if check is not None:
                check()
            for group in self.opt.param_groups:
                group['lr'] = self.lr * min(1., (self.steps + 1) / warmup)
            lp = self.network.target_log_probs(ids_[:, :-1], mask[:, :-1], ids_[:, 1:])
            m = mask[:, 1:] if self.weight is None else mask[:, 1:] * self.weight[ids_[:, 1:]]
            loss = -(lp * m).sum() / m.sum()
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.opt.step(); self.steps += 1
            self._pull_back(self.opt.param_groups[0]['lr'])
            total += loss.item() * m.sum().item(); n += m.sum().item()
            self.seen += int(mask[:, 1:].sum())
        self.network.eval()
        return total / max(n, 1)

    def carry(self, before, kind):
        """The optimizer of a grown copy goes on from the one of the network it grew from (``before``, a Learner):
        a fresh optimizer moves wildly at first, and a copy starting with one would lose a race for that alone. Every
        parameter that is there before takes its statistics, mapped as the parameter was; a new one starts from zero
        at the same count of steps."""
        old = {name: before.opt.state.get(p) for name, p in before.network.named_parameters()}
        steps = next((state['step'] for state in old.values() if state), None)
        if steps is None:
            return
        maps = {}
        for moment in ('exp_avg', 'exp_avg_sq'):
            values = {name: state[moment] for name, state in old.items() if state}
            if kind == 'larghezza':
                values = before.network._widen(values, moments=True)
            elif kind == 'concetti':
                for name in list(values):
                    if name.endswith(('.g', '.log_s', '.log_c')):
                        values[name] = torch.cat([values[name]] * 2, 1)
                    elif name.endswith('.A'):
                        values[name] = torch.cat([torch.cat([values[name]] * 2, 2)] * 2, 1)
            maps[moment] = values
        for name, p in self.network.named_parameters():
            state = {'step': steps.clone() if torch.is_tensor(steps) else steps}
            for moment in ('exp_avg', 'exp_avg_sq'):
                v = maps[moment].get(name)
                state[moment] = v.clone() if v is not None and v.shape == p.shape else torch.zeros_like(p)
            self.opt.state[p] = state

    def hold(self):
        """From now on the parameters are pulled back to where they are, not to zero: as a concept of the concept
        model starts equal to the level above and leaves it only as far as the data take it."""
        self.anchor = [p.detach().clone() for p in self.network.parameters()]

    def _pull_back(self, lr):
        with torch.no_grad():
            if self.anchor is None:
                for p in self.network.parameters():
                    p.mul_(1 - lr * self.decay)
            else:
                for p, a in zip(self.network.parameters(), self.anchor):
                    p.sub_(lr * self.decay * (p - a))



class _Cycle:
    """The documents of a piece in order, then again and again (every pass in a new order)."""

    def __init__(self, docs, rng):
        self.docs, self.rng = docs, rng
        self.order, self.i, self.passes = list(range(len(docs))), 0, 0

    @property
    def fresh(self):
        return self.passes == 0

    def take(self, tokens, within_pass=False):
        out, size = [], 0
        while size < tokens and self.docs:
            if self.i == len(self.order):
                if within_pass:
                    break
                self.order, self.i, self.passes = list(self.rng.permutation(len(self.docs))), 0, self.passes + 1
            doc = self.docs[self.order[self.i]]; self.i += 1
            out.append(doc); size += len(doc)
            if within_pass and self.i == len(self.order):
                break
        return out, size


class Grower:
    """Learns a piece at a time and lets every size come from the data, reconfiguring often at first and ever more
    rarely after, by the rule of every choice of the model.

    The network starts as small as it can be. The decisions follow a doubling calendar of the tokens learned: after
    one batch, two, four, ... At a decision the network as it is and its grown copies (every concept split in two;
    one more layer; twice as wide; twice the groups of concepts; twice the nearby distances: all read exactly as the
    network did) learn the same next stretch of text, so long that a decision costs at most as much as all the text
    learned so far (at most ``TRIAL`` tokens and one pass over the piece). As the concept model compares its sizes by
    what each one reaches and not by how fast it gets there, every candidate learns as far as it goes on the stretch
    (judged after 1/8, 1/4, 1/2 and the whole of it, kept at its best) and is held at the network it starts from, so
    that a grown copy leaves the smaller network only as far as the data take it; a wider copy starts from half the
    step, since its coordinates move in pairs. A copy wins only if its best reads held-out text better than the best
    of the network that did not grow, by more than one standard error of the paired difference. When the new text of
    a piece is over, the piece is read again, with decisions after one more pass, two, four, ..., while held-out text
    is read better; the passes that did not help are undone. When two decisions in a row (the second on twice the
    text) do not improve, the network stops learning (``done``) or stops reading the piece again; ``best`` is the
    network to keep.
    """

    TRIAL = 65536                                              # longest trial of a decision: its cost
    KINDS = ('concepts', 'layer', 'width', 'heads', 'window')

    def __init__(self, network, trial=None, passes=True, **learner):
        self.passes = passes                                   # read a piece again while it helps
        self.learner_options = learner
        self.learner = Learner(network, **learner)
        self.best, self.best_scores, self.best_learner, self.done = None, None, None, False
        self.trial = int(trial or self.TRIAL)
        self.batch = self.learner.tokens                       # the largest batch
        self.learned = 0                                       # tokens learned, passes included
        self.decision = self.learner.tokens                    # the first decision: after one batch
        self.history, self.passes_made = [], []
        self.begin = False                                     # documents start after an eos (the network alone)
        self.misses = 0                                        # decisions in a row that did not improve

    @staticmethod
    def _better(scores, than):
        gain = scores - than
        return len(gain) > 1 and gain.mean() > gain.std(ddof=1) / math.sqrt(len(gain) - 1)

    def _grown(self, kind):
        network = copy.deepcopy(self.learner.network)
        if kind == 'concepts':
            for i in range(len(network.layers)):
                network.split_concepts(i)
            return network
        if kind == 'layer':
            network.add_layer()
            return network
        return {'width': network.widened, 'heads': network.more_heads, 'window': network.wider_window}[kind]()

    def _trial_tokens(self):
        """A decision costs at most what was learned so far: every candidate learns that much over their number."""
        racers = 1 + len(self.KINDS)
        return max(1, min(self.trial, self.learned // racers, self.piece))

    def _keep(self, net, scores):
        self.best, self.best_scores, self.best_learned = net, scores, self.learned   # with the tokens it had learned
        self.best_learner = copy.deepcopy(self.learner)

    def _decide(self, trial, score, check):
        """The network and its grown copies learn the same stretch, each as far as it goes; the best one goes on.
        True if it improved.

        As the concept model compares sizes by what each reaches, not by how fast it gets there, every candidate is
        judged after 1/8, 1/4, 1/2 and the whole of the stretch, stops after two judgments in a row that do not improve,
        and is kept at its best; the winner is the best of the bests. Every candidate is held at the network it
        starts from: a grown copy leaves the smaller network only as far as the data take it."""
        candidates = [("com'è", self.learner)]
        for kind in self.KINDS:
            learner = Learner(self._grown(kind), **self.learner_options)
            learner.steps = self.learner.steps                 # the same pace: no warm-up again, or it could not win
            learner.tokens, learner.warmup = self.learner.tokens, self.learner.warmup
            learner.lr = self.learner.lr
            if kind == 'width':
                learner.lr /= 2                                # its coordinates move in pairs: half the step moves it as much
            candidates.append((kind, learner))
        size = sum(map(len, trial))
        rounds, done, cut = [], 0, size // 8
        for d in trial:                                        # the stretch in doubling parts: 1/8, 1/8, 1/4, 1/2
            if not rounds or (done >= cut and len(rounds) < 4):
                rounds.append([]); cut = cut * 2 if len(rounds) > 1 else cut
            rounds[-1].append(d); done += len(d)
        results = []
        for kind, learner in candidates:
            learner.hold()
            best, best_scores, misses = learner, None, 0
            for part in rounds:
                learner.learn_documents(part, check)
                scores = score(learner.network, size)          # on as much held-out text as the stretch
                if best_scores is None or self._better(scores, best_scores):
                    best, best_scores, misses = copy.deepcopy(learner), scores, 0
                else:
                    misses += 1
                    if misses >= 2:
                        break
            results.append((kind, best, best_scores))
        winner = results[0]
        for result in results[1:]:
            if self._better(result[2], winner[2]):
                winner = result
        kind, self.learner, _ = winner
        net, scores = self.learner.network.export(), score(self.learner.network)   # the winner, on all of it
        improved = self.best_scores is None or self._better(scores, self.best_scores)
        c = net.config
        self.history.append(dict(tokens=self.learned, layers=c['layers'], concepts=list(c['concepts']), dim=c['dim'],
                                 heads=c['heads'], window=c['window'], choice=kind, lr=self.learner.lr,
                                 held_out=float(-scores.mean())))
        if improved:
            self._keep(net, scores)
        return improved

    def _stretch_and_decide(self, cycle, tokens, score, check, within_pass):
        stretch, size = cycle.take(tokens, within_pass)
        if stretch:
            self.learner.learn_documents(stretch, check); self.learned += size
        trial, size = cycle.take(self._trial_tokens())
        self.learned += size
        return self._decide(trial, score, check)

    def learn(self, ids, eos, score, check=None):
        """``score(network, tokens=None)``: the log-probabilities of held-out text (the first documents, about
        ``tokens`` of them, when given) under the network that is learning (torch), as ``held_out`` gives them."""
        if self.done:
            return self.best
        docs = documents(ids, eos, self.learner.longest, self.begin)
        if not docs:
            return self.best
        cycle = _Cycle(docs, self.learner.rng)
        decided = False
        # as many batches in a pass as tokens in a batch: a small piece is learned in many small steps
        self.piece = sum(map(len, docs))                       # a trial is at most one pass over the piece
        self.learner.tokens = min(self.batch, max(1, math.isqrt(self.piece)))
        if not self.learned:                                   # the start lasts at most one pass over the piece
            self.learner.warmup = min(self.learner.warmup, max(1, -(-self.piece // self.learner.tokens)))
        if not self.learned:
            self.decision = self.learner.tokens
        while self.decision <= self.learned:                   # tokens learned elsewhere (a race of growth): the
            self.decision *= 2                                 # calendar goes on from them, it does not decide at once
        while cycle.fresh and not self.done:                   # the new text, on the calendar of the tokens learned
            stretch, size = cycle.take(self.decision - self.learned, within_pass=True)
            if stretch:
                self.learner.learn_documents(stretch, check); self.learned += size
            if self.learned < self.decision or not cycle.fresh or cycle.i == len(cycle.order):
                break                                          # the new text is over before the decision
            trial, size = cycle.take(self._trial_tokens(), within_pass=True)
            self.learned += size; decided = True
            # it no longer learns and growing does not help: confirmed by a second decision, on twice the text
            self.misses = 0 if self._decide(trial, score, check) else self.misses + 1
            self.done = self.misses >= 2
            self.decision *= 2
        if self.done:
            return self.best
        if not decided:                                        # the network at the end of the new text, if better
            scores = score(self.learner.network)
            if self.best_scores is None or self._better(scores, self.best_scores):
                self._keep(self.learner.network.export(), scores)
        if self.passes:
            piece, made = sum(map(len, docs)), 0
            interval = piece
            cycle.i, cycle.passes = len(cycle.order), 0       # the next take starts a new pass
            misses = 0
            while misses < 2:                                  # a pass that did not help, confirmed by a second
                made += interval / piece
                misses = 0 if self._stretch_and_decide(cycle, interval, score, check, False) else misses + 1
                interval *= 2
            self.passes_made.append(round(made, 2))
            if self.best_learner is not None:
                self.learner = copy.deepcopy(self.best_learner)   # undo what did not help
        return self.best
