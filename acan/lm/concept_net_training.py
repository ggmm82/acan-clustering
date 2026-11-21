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
        self.config.setdefault('head_dim', dim // config['heads'])
        self.layers = nn.ModuleList([ConceptAttention(dim, config['heads'], k, config['window'], self.config['head_dim'])
                                     for k in concepts])
        self.final = nn.Linear(dim, dim)
        # The memory of the document, learned: every position read leaves an address (a projection of its state) with
        # the token that followed it; a position reads the tokens that followed the addresses most like its own, and
        # a valve (learned from its state) says how much that reading counts.
        self.config.setdefault('cache', dim)                   # its addresses are as wide as the state
        c = self.config['cache']
        if c:
            self.cache_query = nn.Linear(dim, c, bias=False)
            self.cache_key = nn.Linear(dim, c, bias=False)
            self.cache_valve = nn.Linear(dim, 1)
            with torch.no_grad():
                self.cache_valve.bias.fill_(-2.)                # it starts by counting little
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
        p_cache = (a * same).sum(-1)
        g = torch.sigmoid(self.cache_valve(h))[..., 0] * anything
        return torch.log((1 - g) * net_lp.exp() + g * p_cache + 1e-30)

    def target_log_probs(self, ids, mask, targets):
        h = self.states(ids, mask)
        flat, tgt = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        if not self.grouped:
            lp = torch.log_softmax(self._logits(flat, self.centre.weight, self.log_mass), -1).gather(1, tgt[:, None])[:, 0].view(targets.shape)
            return self._cache(h, mask, targets, lp)
        g = self.group[tgt]
        lp = torch.log_softmax(self._logits(flat, self.group_centre, self.group_log_mass), -1).gather(1, g[:, None])[:, 0]
        inside = torch.zeros_like(lp)
        for k in torch.unique(g).tolist():
            sel = (g == k).nonzero()[:, 0]
            members = self._order[self._start[k]:self._start[k] + self._size[k]]
            logits = self._logits(flat[sel], self.centre.weight[members], self.log_mass[members])
            inside = inside.index_put((sel,), torch.log_softmax(logits, -1).gather(1, self._rank[tgt[sel]][:, None])[:, 0])
        return self._cache(h, mask, targets, (lp + inside).view(targets.shape))

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
        layer = ConceptAttention(dim, self.config['heads'], int(last.g.shape[1]), self.config['window'], self.config['head_dim'])
        with torch.no_grad():
            layer.out.weight.zero_(); layer.out.bias.zero_()
            for norm in (layer.norm1, layer.norm2):                    # a state normalized twice is the same state
                norm.weight.fill_(1.); norm.bias.zero_()
        self.layers.append(layer)
        self.config['layers'] = len(self.layers)
        self.config['concepts'].append(int(last.g.shape[1]))

    def _like(self, **change):
        """A network with the configuration changed, its own parameters at their start (to be filled)."""
        config = dict(self.config, **change)
        groups = self.group.numpy() if self.grouped else None
        return Network(config, groups=groups)

    def widened(self, noise=1e-3):
        """Twice as wide (the state and every group of concepts): every coordinate is copied, so the states are the old
        ones written twice; the maps that read them take half from each copy, the dispersions grow by sqrt 2 and the
        masses of the concepts are corrected, so memberships, readings and the next token are the same."""
        D, H, d = self.config['dim'], self.config['heads'], self.config['head_dim']
        c = self.config.get('cache') or 0
        two = lambda x: torch.cat([x, x], -1)
        half_in = lambda W: torch.cat([W, W], -1) / 2                 # out x in: reads both copies of the input
        rows = lambda W: torch.cat([W, W], 0)                          # writes its output twice
        heads_rows = lambda W: torch.cat([W.view(H, d, -1)] * 2, 1).reshape(2 * H * d, -1)
        new = {}
        for k, v in self.state_dict().items():
            v = v.detach().clone()
            name = k.split('.')[-1] if k.startswith('layers.') else k
            if k in ('centre.weight', 'group_centre'):
                v = two(v)
            elif k == 'log_scale':
                v = v + math.log(2) / 2                                # |h|, |e|: the products count twice
            elif k.startswith('layers.'):
                part = k.split('.', 2)[2]
                if part == 'g':
                    v = two(v)
                elif part == 'log_s':
                    v = two(v + math.log(2) / 2)
                elif part == 'log_c':
                    log_s = self.state_dict()[k.replace('log_c', 'log_s')]
                    v = v + log_s.sum(-1) + d * math.log(2)
                elif part in ('query.weight', 'value.weight'):
                    v = half_in(heads_rows(v))
                elif part == 'near':
                    v = torch.cat([torch.cat([v, v], -1)] * 2, -2) / 2
                elif part == 'out.weight':
                    read, near = v[:, :H * d], v[:, H * d:]
                    read = torch.cat([read.view(D, H, d) / 2] * 2, -1).reshape(D, 2 * H * d)
                    v = rows(torch.cat([read, half_in(near)], -1))
                elif part in ('out.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias'):
                    v = two(v)
            elif k == 'final.weight':
                v = rows(half_in(v))
            elif k == 'final.bias':
                v = two(v)
            elif k in ('cache_query.weight', 'cache_key.weight'):
                v = rows(half_in(v)) * (1 / math.sqrt(2) if k == 'cache_key.weight' else 1)
            elif k == 'cache_valve.weight':
                v = half_in(v)
            new[k] = v
        grown = self._like(dim=2 * D, head_dim=2 * d, cache=2 * c)
        grown.load_state_dict(new)
        with torch.no_grad():                                          # the copies may then part
            for k, p in grown.named_parameters():
                if p.dim() > 1:
                    p.add_(torch.randn_like(p) * noise * (p.std() + 1e-6))
        return grown

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
        model = cls(config, groups=groups)
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

    def __init__(self, network, lr=3e-3, tokens=2048, longest=512, seed=0, weight=None):
        """``weight``: how much every token weighs when it is the one to read (one per token of the vocabulary),
        as the concept model weighs its attributes: a target as much as all the features together."""
        self.network, self.tokens, self.longest = network, tokens, longest
        self.weight = None if weight is None else torch.as_tensor(np.asarray(weight, np.float32))
        self.opt = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=.01)
        self.lr, self.steps = lr, 0
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
            total += loss.item() * m.sum().item(); n += m.sum().item()
        self.network.eval()
        return total / max(n, 1)



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
    network did) learn the same next stretch of text, each at its step and at half of it (the pace is chosen as the
    size is; a wider copy starts from half the step, since its coordinates move in pairs), as long as all the text learned so far (at most ``TRIAL``
    tokens, what a decision may cost, and at most one pass over the piece); a copy wins only if it then reads held-out text better than the network that
    did not grow, by more than one standard error of the paired difference: the size that learns more from the same
    text. When the new text of a piece is over, the piece is read again, with decisions after one more pass, two,
    four, ..., while held-out text is read better; the passes that did not help are undone. When a decision on new
    text does not improve and no growth helps, the network stops learning (``done``); ``best`` is the network to keep.
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

    def _keep(self, net, scores):
        self.best, self.best_scores, self.best_learned = net, scores, self.learned   # with the tokens it had learned
        self.best_learner = copy.deepcopy(self.learner)

    def _decide(self, trial, score, check):
        """The network and its grown copies learn the same stretch; the best one goes on. True if it improved."""
        candidates = [("com'è", self.learner)]
        for kind in self.KINDS:
            learner = Learner(self._grown(kind), **self.learner_options)
            learner.steps = self.learner.steps                 # the same pace: no warm-up again, or it could not win
            learner.tokens, learner.warmup = self.learner.tokens, self.learner.warmup
            learner.lr = self.learner.lr
            if kind == 'width':
                # every coordinate is now two that move together: half the step moves the network as much as
                # before, or the wider copy would win only because it learns faster, not because it needs to
                learner.lr /= 2
            candidates.append((kind, learner))
        # the pace is chosen as the size is: every candidate also learns the stretch at half its step
        slower = []
        for kind, learner in candidates:
            twin = copy.deepcopy(learner)
            twin.lr /= 2
            slower.append((kind, twin))
        results = []
        for half, (kind, learner) in [(False, c) for c in candidates] + [(True, c) for c in slower]:
            learner.learn_documents(trial, check)
            net = learner.network.export()
            results.append((kind, learner, net, score(net), half))
        winner = results[0]
        for result in results[1:]:
            if self._better(result[3], winner[3]):
                winner = result
        kind, self.learner, net, scores, half = winner
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
        trial, size = cycle.take(min(self.trial, self.learned, self.piece))
        self.learned += size
        return self._decide(trial, score, check)

    def learn(self, ids, eos, score, check=None):
        if self.done:
            return self.best
        docs = documents(ids, eos, self.learner.longest)
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
        while cycle.fresh and not self.done:                   # the new text, on the calendar of the tokens learned
            stretch, size = cycle.take(self.decision - self.learned, within_pass=True)
            if stretch:
                self.learner.learn_documents(stretch, check); self.learned += size
            if self.learned < self.decision or not cycle.fresh or cycle.i == len(cycle.order):
                break                                          # the new text is over before the decision
            trial, size = cycle.take(min(self.trial, self.learned, self.piece), within_pass=True)
            self.learned += size; decided = True
            # it no longer learns and growing does not help: confirmed by a second decision, on twice the text
            self.misses = 0 if self._decide(trial, score, check) else self.misses + 1
            self.done = self.misses >= 2
            self.decision *= 2
        if self.done:
            return self.best
        if not decided:                                        # the network at the end of the new text, if better
            net = self.learner.network.export()
            scores = score(net)
            if self.best_scores is None or self._better(scores, self.best_scores):
                self._keep(net, scores)
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
