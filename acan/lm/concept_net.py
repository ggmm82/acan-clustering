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
        z = (q[..., None, :] - g) * np.exp(-log_s)
        return _softmax(log_c - .5 * (z * z).sum(-1) - log_s.sum(-1))

    def _project(self, i, h):
        H, d = self.heads, self.dim // self.heads
        q = (h @ self.p[f'layers.{i}.query.weight'].T).reshape(*h.shape[:-1], H, d)
        v = (h @ self.p[f'layers.{i}.value.weight'].T).reshape(*h.shape[:-1], H, d)
        return q, v

    def _finish(self, i, h, read, near):
        out = np.concatenate([read, near], -1) @ self.p[f'layers.{i}.out.weight'].T + self.p[f'layers.{i}.out.bias']
        h = _layer_norm(h + out, self.p[f'layers.{i}.norm1.weight'], self.p[f'layers.{i}.norm1.bias'])
        return _layer_norm(h, self.p[f'layers.{i}.norm2.weight'], self.p[f'layers.{i}.norm2.bias'])

    def _looked(self, i):
        return _softmax(self.p[f'layers.{i}.A'])

    # ---- a whole document --------------------------------------------------------------------------------
    def states(self, ids):
        """The final state of every position of one document (T x D)."""
        h = self.p['centre.weight'][np.asarray(ids, np.int64)]
        T = len(h)
        for i in range(self.layers):
            q, v = self._project(i, h)
            t = self._memberships(i, q)                                       # T H K
            mass = np.cumsum(t, 0) + EPS_MASS
            sums = np.cumsum(t[..., None] * v[:, :, None, :], 0)              # T H K d
            weight = np.einsum('thk,hkj->thj', t, self._looked(i)) / mass
            read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, self.dim)
            near = np.zeros_like(h)
            for w in range(1, self.window + 1):
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

    def document(self, ids):
        """p(token i | tokens before it) for i = 1..T-1 of one document: (T-1) x V."""
        ids = np.asarray(ids, np.int64)
        if len(ids) < 2:
            return np.zeros((0, self.vocabulary))
        return self.probabilities(self.states(ids[:-1]))

    def targets(self, ids, block=256):
        """p(token i | tokens before it) of the true token only, for i = 1..T-1 of one document, in blocks."""
        ids = np.asarray(ids, np.int64)
        if len(ids) < 2:
            return np.zeros(0)
        h = self.states(ids[:-1])
        out = np.empty(len(h))
        for i in range(0, len(h), block):
            probs = self.probabilities(h[i:i + block])
            out[i:i + block] = probs[np.arange(len(probs)), ids[i + 1:i + 1 + block]]
        return out

    def cursor(self):
        return NetCursor(self)

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
        H, d = net.heads, net.dim // net.heads
        K = net.p['layers.0.g'].shape[1] if net.layers else 0
        self.mass = [np.full((H, K), EPS_MASS, np.float32) for _ in range(net.layers)]
        self.sums = [np.zeros((H, K, d), np.float32) for _ in range(net.layers)]
        self.recent = [[] for _ in range(net.layers)]                       # inputs of the layer, newest last
        self.state = None

    def copy(self):
        other = NetCursor.__new__(NetCursor)
        other.net = self.net
        other.mass = [m.copy() for m in self.mass]
        other.sums = [s.copy() for s in self.sums]
        other.recent = [list(r) for r in self.recent]
        other.state = None if self.state is None else self.state.copy()
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
            read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, net.dim)
            before = np.array(self.recent[i][-net.window:]).reshape(-1, net.dim)
            padded = np.vstack([np.zeros((net.window - len(before), net.dim), np.float32), before, h])
            near = np.zeros_like(h)
            for w in range(1, net.window + 1):
                near += padded[net.window - w:net.window - w + T] @ net.p[f'layers.{i}.near'][w - 1]
            self.mass[i], self.sums[i] = mass[-1], sums[-1]
            self.recent[i] = list(padded[-net.window:])
            h = net._finish(i, h, read, near)
        self.state = h[-1] @ net.p['final.weight'].T + net.p['final.bias']

    def observe(self, token):
        net = self.net
        h = net.p['centre.weight'][int(token)]
        for i in range(net.layers):
            q, v = net._project(i, h)
            t = net._memberships(i, q)                                         # H K
            self.mass[i] += t
            self.sums[i] += t[..., None] * v[:, None, :]
            weight = np.einsum('hk,hkj->hj', t, net._looked(i)) / self.mass[i]
            read = np.einsum('hj,hjd->hd', weight, self.sums[i]).reshape(net.dim)
            near = np.zeros_like(h)
            for w, previous in enumerate(reversed(self.recent[i][-net.window:]), 1):
                near += previous @ net.p[f'layers.{i}.near'][w - 1]
            self.recent[i] = (self.recent[i] + [h])[-net.window:]
            h = net._finish(i, h, read, near)
        self.state = h @ net.p['final.weight'].T + net.p['final.bias']

    def distribution(self):
        return None if self.state is None else self.net.probabilities(self.state)[0]
