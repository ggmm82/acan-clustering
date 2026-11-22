"""The concept network: the concept model in layers, with its parameters learned as a language model learns.

Every object is a concept of the concept model (acan/MODEL.md):

* a token is a concept with a centre e_y and a mass C_y. The state of a position starts at the centre of its token;
  the next token is read as the membership of the final state to the token concepts, in two steps when the tokens
  are grouped: first the group, then the token inside it;
* in every layer, each group of concepts (a "head") has K concepts with centre g_k, dispersion s_k and mass C_k; a
  position belongs to them with T_k(h) = softmax_k(log C_k + log N(h; g_k, s_k));
* the document is read through the sums of the concepts: M_k(i) = sum_{j<=i} T_k(h_j), S_k(i) = sum_{j<=i} T_k(h_j) v_j,
  and a position reads sum_k T_k(h_i) sum_k' A_kk' S_k'(i) / M_k'(i). A cursor keeps only these sums and the last
  states of every layer, so the cost of a token and the memory do not grow with the length of the document;
* the order of the nearby tokens enters through the last W states, one map per distance.

This module reads the network with numpy (the parameters are plain arrays); acan/lm/concept_net_training.py learns
them, maximizing the log-likelihood of the next token, with torch.
"""
import json
import numpy as np

EPS_MASS = 1e-6
EPS_NORM = 1e-5


def _softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def _layer_norm(x, w, b):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS_NORM) * w + b


class ConceptNet:
    """The parameters of a trained network and the reading of a document (numpy, float32)."""

    def __init__(self, config, params):
        self.config = dict(config)
        self.p = {k: np.asarray(v, np.float32) for k, v in params.items()}
        self.layers = int(config['layers'])
        self.heads = int(config['heads'])
        self.window = int(config['window'])
        self.dim = int(config['dim'])
        self.vocabulary = int(config['vocabulary'])
        self.groups = None
        if 'group' in self.p:
            self.groups = self.p['group'].astype(np.int64)
            order = np.argsort(self.groups, kind='stable')
            counts = np.bincount(self.groups)
            self._members = np.split(order, np.cumsum(counts)[:-1])

    # ---- one layer -------------------------------------------------------------------------------------
    def _memberships(self, i, q):
        g, log_s, log_c = self.p[f'layers.{i}.g'], self.p[f'layers.{i}.log_s'], self.p[f'layers.{i}.log_c']
        inv = np.exp(-2 * log_s)                                             # the square expanded: no position x concept x d
        quad = np.einsum('...hd,hkd->...hk', q * q, inv) - 2 * np.einsum('...hd,hkd->...hk', q, g * inv) + (g * g * inv).sum(-1)
        return _softmax(log_c - .5 * quad - log_s.sum(-1))

    def shape(self, i):
        """Groups, coordinates of a group and nearby distances of layer i."""
        H, _, d = self.p[f'layers.{i}.g'].shape
        return H, d, self.p[f'layers.{i}.near'].shape[0]

    def _project(self, i, h):
        H, d, _ = self.shape(i)
        q = (h @ self.p[f'layers.{i}.query.weight'].T).reshape(*h.shape[:-1], H, d)
        v = (h @ self.p[f'layers.{i}.value.weight'].T).reshape(*h.shape[:-1], H, d)
        return q, v

    def _finish(self, i, h, read, near):
        out = np.concatenate([read, near], -1) @ self.p[f'layers.{i}.out.weight'].T + self.p[f'layers.{i}.out.bias']
        new = _layer_norm(h + out, self.p[f'layers.{i}.norm1.weight'], self.p[f'layers.{i}.norm1.bias'])
        new = _layer_norm(new, self.p[f'layers.{i}.norm2.weight'], self.p[f'layers.{i}.norm2.bias'])
        gate = self.p.get(f'layers.{i}.gate')                      # how open the layer is (older networks: open)
        return new if gate is None else h + gate * (new - h)

    def _looked(self, i):
        return _softmax(self.p[f'layers.{i}.A'])

    # ---- a whole document --------------------------------------------------------------------------------
    def states(self, ids, memberships=None):
        """The final state of every position of one document (T x D). ``memberships``, a list, receives the
        memberships of every layer (T x H x K)."""
        h = self.p['centre.weight'][np.asarray(ids, np.int64)]
        T = len(h)
        for i in range(self.layers):
            q, v = self._project(i, h)
            t = self._memberships(i, q)                                       # T H K
            if memberships is not None:
                memberships.append(t)
            mass = np.cumsum(t, 0) + EPS_MASS
            sums = np.cumsum(t[..., None] * v[:, :, None, :], 0)              # T H K d
            weight = np.einsum('thk,hkj->thj', t, self._looked(i)) / mass
            read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, -1)
            near = np.zeros_like(h)
            for w in range(1, self.shape(i)[2] + 1):
                if w < T:
                    near[w:] += h[:-w] @ self.p[f'layers.{i}.near'][w - 1]
            h = self._finish(i, h, read, near)
        return h @ self.p['final.weight'].T + self.p['final.bias']

    def _logits(self, h, centres, log_mass):
        inv = np.exp(-2 * self.p['log_scale'])
        return (h @ centres.T - .5 * (centres * centres).sum(-1)) * inv + log_mass

    def probabilities(self, h):
        """p(next token) for states h (N x D) -> N x V, float64."""
        h = np.atleast_2d(h)
        e, log_mass = self.p['centre.weight'], self.p['log_mass']
        if self.groups is None:
            return _softmax(self._logits(h, e, log_mass).astype(np.float64))
        pg = _softmax(self._logits(h, self.p['group_centre'], self.p['group_log_mass']).astype(np.float64))
        out = np.empty((len(h), self.vocabulary))
        for k, members in enumerate(self._members):
            out[:, members] = pg[:, k:k + 1] * _softmax(self._logits(h, e[members], log_mass[members]).astype(np.float64))
        return out

    @property
    def cached(self):
        return 'cache_query.weight' in self.p

    def _addresses(self, h):
        return h @ self.p['cache_query.weight'].T, h @ self.p['cache_key.weight'].T

    def _valve(self, h):
        z = (h @ self.p['cache_valve.weight'].T + self.p['cache_valve.bias'])[..., 0]
        return 1 / (1 + np.exp(-np.clip(z, -60, 60)))

    def _share(self, h, s):
        """How much the memory counts and how it divides among the addresses (scores s, the last axis): with
        "none is like mine" as one more address, whose likeness is learned from the state; else a valve."""
        if self.config.get('sentinel'):
            z = (h @ self.p['cache_valve.weight'].T + self.p['cache_valve.bias'])[..., 0]
            top = np.maximum(np.max(s, -1, initial=-np.inf), z)
            e, e0 = np.exp(s - top[..., None]), np.exp(z - top)
            total = e.sum(-1) + e0
            return 1 - e0 / total, e / total[..., None]
        a = np.exp(s - np.max(s, -1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        return self._valve(h), a * 1.

    def _mix(self, probs, h, ids, start):
        """The memory of the document: positions start.. of h read the tokens that followed the earlier positions."""
        if not self.cached:
            return probs
        q, k = self._addresses(h)
        nxt = ids[1:]                                                            # the token after every position
        out = probs.copy()
        for r in range(len(probs)):
            i = start + r
            if i == 0:
                continue
            s = k[:i] @ q[i] / np.sqrt(q.shape[1])
            g, a = self._share(h[i], s)
            cache = np.bincount(nxt[:i], weights=a, minlength=self.vocabulary)
            out[r] = (1 - g) * probs[r] + (cache if self.config.get('sentinel') else g * cache)
        return out

    def document(self, ids):
        """p(token i | tokens before it) for i = 1..T-1 of one document: (T-1) x V."""
        ids = np.asarray(ids, np.int64)
        if len(ids) < 2:
            return np.zeros((0, self.vocabulary))
        h = self.states(ids[:-1])
        return self._mix(self.probabilities(h), h, ids, 0)

    def targets(self, ids, block=256):
        """p(token i | tokens before it) of the true token only, for i = 1..T-1 of one document, in blocks."""
        ids = np.asarray(ids, np.int64)
        if len(ids) < 2:
            return np.zeros(0)
        h = self.states(ids[:-1])
        out = np.empty(len(h))
        nxt = ids[1:]
        if self.cached:
            q, k = self._addresses(h)
            g = self._valve(h)
            z = (h @ self.p['cache_valve.weight'].T + self.p['cache_valve.bias'])[..., 0]
        for i in range(0, len(h), block):
            probs = self.probabilities(h[i:i + block])
            p = probs[np.arange(len(probs)), nxt[i:i + block]]
            if self.cached:                                  # the memory of the document, a block at a time
                B, end = len(p), i + len(p)
                s = q[i:end] @ k[:end].T / np.sqrt(q.shape[1])                     # B x end
                earlier = np.arange(end)[None, :] < np.arange(i, end)[:, None]
                s = np.where(earlier, s, -np.inf)
                same = nxt[:end][None, :] == nxt[i:end][:, None]
                if self.config.get('sentinel'):              # "none is like mine" is one more address
                    top = np.maximum(s.max(1), z[i:end])
                    e, e0 = np.where(earlier, np.exp(s - top[:, None]), 0.), np.exp(z[i:end] - top)
                    total = e.sum(1) + e0
                    p = e0 / total * p + (e * same).sum(1) / total
                else:
                    top = np.where(earlier.any(1), s.max(1), 0.)
                    a = np.where(earlier, np.exp(s - top[:, None]), 0.)
                    a /= np.maximum(a.sum(1, keepdims=True), 1e-300)
                    cache = (a * same).sum(1)
                    gate = np.where(earlier.any(1), g[i:end], 0.)
                    p = (1 - gate) * p + gate * cache
            out[i:i + block] = p
        return out

    def cursor(self):
        return NetCursor(self)

    def graph(self, documents, decode=None, labels=3):
        """The graph of the network, measured on some documents. Nodes: the concepts that hold more than an even share
        of their group, named by the tokens that enter them most and the tokens that follow them most. Edges: in a
        layer, the attention from a concept to the concepts it reads (exact, the matrix A); from a layer to the next,
        how the positions of a concept spread over the concepts of the next layer. An edge is kept when it weighs more
        than an even share and, between layers, more than the next concept holds anyway: what is shown says something."""
        found, now, after = [[] for _ in range(self.layers)], [], []
        for doc in documents:
            doc = np.asarray(doc, np.int64)
            if len(doc) < 2:
                continue
            layers = []
            self.states(doc[:-1], layers)
            for i, t in enumerate(layers):
                found[i].append(t)
            now.append(doc[:-1]); after.append(doc[1:])
        if not now:
            return dict(nodes=[], edges=[])
        now, after = np.concatenate(now), np.concatenate(after)
        word = (lambda ids: [decode([int(x)]).strip() or repr(decode([int(x)])) for x in ids]) if decode else (lambda ids: [str(int(x)) for x in ids])
        nodes, edges, kept = [], [], []
        for i in range(self.layers):
            t = np.concatenate(found[i])                                       # positions x H x K
            H, K = t.shape[1:]
            share = t.sum(0) / t.sum((0, 2))[:, None]                          # H x K
            keep = {(h, k) for h in range(H) for k in range(K) if share[h, k] > 1 / K}
            kept.append((t, keep))
            for h, k in sorted(keep):
                w = t[:, h, k]
                into = np.bincount(now, weights=w, minlength=self.vocabulary)
                out = np.bincount(after, weights=w, minlength=self.vocabulary)
                nodes.append(dict(id=f'{i + 1}.{h + 1}.{k + 1}', layer=i + 1, group=h + 1, concept=k + 1, mass=float(share[h, k]),
                                  enters=word(np.argsort(-into)[:labels]), follows=word(np.argsort(-out)[:labels])))
            A = self._looked(i)                                                # H x K x K: k reads k'
            for h, k in sorted(keep):
                for k2 in np.flatnonzero(A[h, k] > 1 / K):
                    if (h, int(k2)) in keep and k2 != k:
                        edges.append(dict(source=f'{i + 1}.{h + 1}.{k + 1}', target=f'{i + 1}.{h + 1}.{k2 + 1}',
                                          weight=float(A[h, k, k2]), kind='legge'))
        for i in range(self.layers - 1):
            (t1, keep1), (t2, keep2) = kept[i], kept[i + 1]
            H2, K2 = t2.shape[1:]
            base = t2.mean(0)                                                  # what a next concept holds anyway
            for h, k in sorted(keep1):
                w = t1[:, h, k]
                spread = (w[:, None, None] * t2).sum(0) / max(w.sum(), 1e-12)  # H2 x K2, each group sums to one
                for h2, k2 in sorted(keep2):
                    if spread[h2, k2] > 1 / K2 and spread[h2, k2] > base[h2, k2]:
                        edges.append(dict(source=f'{i + 1}.{h + 1}.{k + 1}', target=f'{i + 2}.{h2 + 1}.{k2 + 1}',
                                          weight=float(spread[h2, k2]), kind='porta a',
                                          lift=float(spread[h2, k2] / max(base[h2, k2], 1e-12))))
        return dict(nodes=nodes, edges=edges)

    def describe(self, documents, decode=None, concepts=3, examples=4):
        """What the concepts of every layer hold, on some documents: how many concepts work in every group (e to the
        entropy of their masses), how much a concept is tied to one token (the share of its most frequent token), and
        the positions that belong most to the concepts of the first group with the most mass, with their context."""
        found = [[] for _ in range(self.layers)]
        where = []
        for n, doc in enumerate(documents):
            doc = np.asarray(doc, np.int64)
            if len(doc) < 2:
                continue
            layers = []
            self.states(doc, layers)
            for i, t in enumerate(layers):
                found[i].append(t)
            where += [(n, j) for j in range(len(doc))]
        if not where:
            return []
        tokens = np.array([int(np.asarray(documents[n])[j]) for n, j in where])
        show = (lambda n, j: '…' + decode(list(np.asarray(documents[n])[max(0, j - 4):j])) + '[' + decode([int(documents[n][j])]) + ']'
                ) if decode else (lambda n, j: str(int(documents[n][j])))
        out = []
        for i in range(self.layers):
            t = np.concatenate(found[i])                                       # positions x H x K
            active, tied = [], []
            for head in range(t.shape[1]):
                share = t[:, head].sum(0) / t[:, head].sum()
                active.append(float(np.exp(-(share * np.log(share + 1e-12)).sum())))
                hard = t[:, head].argmax(1)
                for k in np.unique(hard):
                    sel = hard == k
                    if sel.sum() >= 20:
                        tied.append(np.unique(tokens[sel], return_counts=True)[1].max() / sel.sum())
            mass = t[:, 0].sum(0)
            shown = []
            for k in np.argsort(-mass)[:concepts]:
                best = np.argsort(-t[:, 0, k])[:examples]
                shown.append(dict(concept=int(k), mass=float(mass[k] / mass.sum()),
                                  examples=[show(*where[j]).replace('\n', '⏎') for j in best]))
            out.append(dict(layer=i + 1, active=[round(a, 1) for a in active],
                            tied=float(np.mean(tied)) if tied else None, concepts=shown))
        return out

    # ---- persistence -------------------------------------------------------------------------------------
    def arrays(self, prefix='net.'):
        out = {prefix + k: v for k, v in self.p.items()}
        out[prefix + 'config'] = np.frombuffer(json.dumps(self.config).encode(), np.uint8)
        return out

    @classmethod
    def from_arrays(cls, data, prefix='net.'):
        if prefix + 'config' not in data:
            return None
        config = json.loads(bytes(np.asarray(data[prefix + 'config'], np.uint8)).decode())
        params = {k[len(prefix):]: v for k, v in data.items() if k.startswith(prefix) and k != prefix + 'config'}
        return cls(config, params)


class NetCursor:
    """Reads a document one token at a time, keeping the sums of the concepts and the last states of every layer."""

    def __init__(self, net):
        self.net = net
        shapes = [net.p[f'layers.{i}.g'].shape for i in range(net.layers)]   # groups, concepts, coordinates
        self.mass = [np.full((H, k), EPS_MASS, np.float32) for H, k, _ in shapes]
        self.sums = [np.zeros((H, k, d), np.float32) for H, k, d in shapes]
        self.recent = [[] for _ in range(net.layers)]                       # inputs of the layer, newest last
        self.state = None
        self.keys, self.after = [], []                                       # the memory of the document

    def copy(self):
        other = NetCursor.__new__(NetCursor)
        other.net = self.net
        other.mass = [m.copy() for m in self.mass]
        other.sums = [s.copy() for s in self.sums]
        other.recent = [list(r) for r in self.recent]
        other.state = None if self.state is None else self.state.copy()
        other.keys, other.after = list(self.keys), list(self.after)
        return other

    def prefill(self, ids):
        """Read a whole prefix at once (in bulk), leaving the same sums and recent states as observe would."""
        ids = np.asarray(ids, np.int64)
        if not len(ids):
            return
        net = self.net
        h = net.p['centre.weight'][ids]
        T = len(h)
        for i in range(net.layers):
            q, v = net._project(i, h)
            t = net._memberships(i, q)
            mass = np.cumsum(t, 0) + self.mass[i]
            sums = np.cumsum(t[..., None] * v[:, :, None, :], 0) + self.sums[i]
            weight = np.einsum('thk,hkj->thj', t, net._looked(i)) / mass
            read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, -1)
            W = net.shape(i)[2]
            before = np.array(self.recent[i][-W:]).reshape(-1, net.dim)
            padded = np.vstack([np.zeros((W - len(before), net.dim), np.float32), before, h])
            near = np.zeros_like(h)
            for w in range(1, W + 1):
                near += padded[W - w:W - w + T] @ net.p[f'layers.{i}.near'][w - 1]
            self.mass[i], self.sums[i] = mass[-1], sums[-1]
            self.recent[i] = list(padded[-W:])
            h = net._finish(i, h, read, near)
        states = h @ net.p['final.weight'].T + net.p['final.bias']
        if net.cached:
            if self.state is not None:
                self.keys.append(net._addresses(self.state)[1]); self.after.append(int(ids[0]))
            keys = net._addresses(states[:-1])[1]
            self.keys += list(keys); self.after += [int(t) for t in ids[1:]]
        self.state = states[-1]

    def observe(self, token):
        net = self.net
        if net.cached and self.state is not None:                           # the last state was followed by token
            self.keys.append(net._addresses(self.state)[1]); self.after.append(int(token))
        h = net.p['centre.weight'][int(token)]
        for i in range(net.layers):
            q, v = net._project(i, h)
            t = net._memberships(i, q)                                         # H K
            self.mass[i] += t
            self.sums[i] += t[..., None] * v[:, None, :]
            weight = np.einsum('hk,hkj->hj', t, net._looked(i)) / self.mass[i]
            read = np.einsum('hj,hjd->hd', weight, self.sums[i]).reshape(-1)
            W = net.shape(i)[2]
            near = np.zeros_like(h)
            for w, previous in enumerate(reversed(self.recent[i][-W:]), 1):
                near += previous @ net.p[f'layers.{i}.near'][w - 1]
            self.recent[i] = (self.recent[i] + [h])[-W:]
            h = net._finish(i, h, read, near)
        self.state = h @ net.p['final.weight'].T + net.p['final.bias']

    def distribution(self):
        if self.state is None:
            return None
        p = self.net.probabilities(self.state)[0]
        if self.net.cached and self.keys:
            q = self.net._addresses(self.state)[0]
            s = np.array(self.keys) @ q / np.sqrt(len(q))
            g, a = self.net._share(self.state, s)
            cache = np.bincount(self.after, weights=a, minlength=self.net.vocabulary)
            p = (1 - g) * p + (cache if self.net.config.get('sentinel') else g * cache)
        return p
