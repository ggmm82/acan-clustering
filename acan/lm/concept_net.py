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


def _gaussian_logits(q, g, log_s, log_c):
    inv = np.exp(-2 * log_s)
    g_term = (g * g * inv).sum(-1)
    ls_sum = log_s.sum(-1)
    if q.ndim == 3:
        qt = np.swapaxes(q, 0, 1)                                            # H T d
        quad = (np.matmul(qt * qt, np.swapaxes(inv, 1, 2))
                - 2 * np.matmul(qt, np.swapaxes(g * inv, 1, 2)) + g_term[:, None, :])
        return np.swapaxes(log_c[:, None, :] - .5 * quad - ls_sum[:, None, :], 0, 1)
    quad = np.einsum('...hd,hkd->...hk', q * q, inv) - 2 * np.einsum('...hd,hkd->...hk', q, g * inv) + g_term
    return log_c - .5 * quad - ls_sum


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
    def _tabular_layer(self, i):
        return f'layers.{i}.key.weight' in self.p

    def _linear_layer(self, i):
        return self._tabular_layer(i) and f'layers.{i}.A' not in self.p

    def _routed_layer(self, i):
        K = self.p[f'layers.{i}.g'].shape[1]
        return self._linear_layer(i) and K > 32 and K % 16 == 0

    def _pure_layer(self, i):
        H, d, _ = self.shape(i)
        return not self._tabular_layer(i) and self.p[f'layers.{i}.out.weight'].shape[1] == H * d

    def _view_scale(self, i, d):
        if self._linear_layer(i):
            return np.float32((d * float(self.config.get('res_var_div', 1.))) ** -.5)
        return np.float32(d ** -.5)

    def _memberships(self, i, q, t_prev=None):
        g, log_s, log_c = self.p[f'layers.{i}.g'], self.p[f'layers.{i}.log_s'], self.p[f'layers.{i}.log_c']
        logits = _gaussian_logits(q, g, log_s, log_c)
        parent = self.p.get(f'layers.{i}.parent')
        if parent is not None and t_prev is not None and t_prev.shape[-2:] == (parent.shape[0], parent.shape[1]):
            lift = np.einsum('...hp,hpk->...hk', t_prev, np.exp(np.clip(parent, -10., 10.)))
            logits = logits + np.log(np.maximum(lift, 1e-12))
        return _softmax(logits)

    def _macro_clusters(self, i, G_c=16):
        g, log_s, log_c = self.p[f'layers.{i}.g'], self.p[f'layers.{i}.log_s'], self.p[f'layers.{i}.log_c']
        H, K, d = g.shape
        S_c = K // G_c
        r = np.float32(self.config.get('res_var_div', 1.))
        c_factor = np.float32(1. - 1. / float(r))
        g_g = g.reshape(H, G_c, S_c, d)
        ls_g = log_s.reshape(H, G_c, S_c, d)
        lc_g = log_c.reshape(H, G_c, S_c)
        true_lp = lc_g - c_factor * ls_g.sum(-1)
        w_sub = _softmax(true_lp, -1)[..., None]
        g_bar = (w_sub * g_g).sum(2)
        var_unit = np.maximum((w_sub * (np.exp(2 * ls_g) / r + (g_g - g_bar[:, :, None, :]) ** 2)).sum(2), np.float32(1e-5))
        ls_bar = np.float32(.5) * np.log(var_unit * r)
        lp_max = true_lp.max(-1, keepdims=True)
        lse = lp_max[..., 0] + np.log(np.exp(true_lp - lp_max).sum(-1))
        l_bar = lse + c_factor * ls_bar.sum(-1)
        return g_g, ls_g, lc_g, g_bar, ls_bar, l_bar, var_unit

    def _routed_subconcepts(self, i, q, rl_c, g_g, ls_g, lc_g, need_full=False):
        H, G_c, S_c, d = g_g.shape
        top_m = int(self.config.get('subconcept_top_m', 4))
        div = np.float32(self.config.get('res_var_div', 1.))
        grp = rl_c.argmax(-1)
        if q.ndim == 2:
            h_idx = np.arange(H)
            gs, lss, lcs = g_g[h_idx, grp], ls_g[h_idx, grp], lc_g[h_idx, grp]
            inv = np.exp(-2 * lss)
            quad = ((q * q)[:, None, :] * inv).sum(-1) - 2 * (q[:, None, :] * gs * inv).sum(-1) + (gs * gs * inv).sum(-1)
            logits = lcs - .5 * quad - lss.sum(-1)
            if 0 < top_m < S_c:
                idx = np.argpartition(-logits, top_m - 1, axis=-1)[:, :top_m]
                vals = _softmax(np.take_along_axis(logits, idx, axis=-1), -1)
                pb = np.zeros_like(logits)
                np.put_along_axis(pb, idx, vals, axis=-1)
            else:
                pb = _softmax(logits, -1)
            proto = np.einsum('hs,hsd->hd', pb, gs)
            var_q = np.einsum('hs,hsd->hd', pb, np.exp(2 * lss))
            return proto, np.sqrt(var_q / div + np.float32(1e-4)), None
        T = q.shape[0]
        BG = H * G_c
        inv_raw = np.exp(-2 * ls_g)
        inv = inv_raw.reshape(BG, S_c, d)
        s2 = np.exp(2 * ls_g).reshape(BG, S_c, d)
        gf = g_g.reshape(BG, S_c, d)
        ginv = gf * inv
        bias = (lc_g - .5 * (g_g * g_g * inv_raw).sum(-1) - ls_g.sum(-1)).reshape(BG, S_c)
        b_flat = (grp + (np.arange(H) * G_c)[None, :]).reshape(-1)
        q_flat = q.reshape(T * H, d)
        order = np.argsort(b_flat, kind='stable')
        b_sorted = b_flat[order]
        q_sorted = q_flat[order]
        q2_sorted = q_sorted * q_sorted
        counts = np.bincount(b_sorted, minlength=BG)
        starts = np.zeros(BG + 1, np.int64)
        starts[1:] = np.cumsum(counts)
        proto_sorted = np.empty((T * H, d), np.float32)
        var_sorted = np.empty((T * H, d), np.float32)
        full_t = np.zeros((T, H, G_c * S_c), np.float32) if need_full else None
        for b in np.flatnonzero(counts):
            a, e = int(starts[b]), int(starts[b + 1])
            qb, q2b = q_sorted[a:e], q2_sorted[a:e]
            logits = bias[b][None, :] - .5 * (q2b @ inv[b].T - 2 * (qb @ ginv[b].T))
            if 0 < top_m < S_c:
                idx = np.argpartition(-logits, top_m - 1, axis=-1)[:, :top_m]
                vals = _softmax(np.take_along_axis(logits, idx, axis=-1), -1)
                pb = np.zeros_like(logits)
                np.put_along_axis(pb, idx, vals, axis=-1)
            else:
                pb = _softmax(logits, -1)
            proto_sorted[a:e] = pb @ gf[b]
            var_sorted[a:e] = pb @ s2[b]
            if need_full:
                h_i, c_i = divmod(int(b), G_c)
                full_t[order[a:e] // H, h_i, c_i * S_c:(c_i + 1) * S_c] = pb
        proto_flat = np.empty_like(proto_sorted); proto_flat[order] = proto_sorted
        var_flat = np.empty_like(var_sorted); var_flat[order] = var_sorted
        proto = proto_flat.reshape(T, H, d)
        s_bar = np.sqrt(var_flat.reshape(T, H, d) / div + np.float32(1e-4))
        return proto, s_bar, full_t

    def _above_stats(self, i, t):
        div = np.float32(self.config.get('res_var_div', 1.))
        g, log_s = self.p[f'layers.{i}.g'], self.p[f'layers.{i}.log_s']
        g_bar = np.einsum('...hk,hkd->...hd', t, g)
        s_bar = np.sqrt(np.einsum('...hk,hkd->...hd', t, np.exp(2 * log_s) / div) + np.float32(1e-4))
        return g_bar, s_bar

    def shape(self, i):
        """Groups, coordinates of a group and nearby distances of layer i."""
        H, _, d = self.p[f'layers.{i}.g'].shape
        return H, d, self.p[f'layers.{i}.near'].shape[0]

    def footprint(self):
        weights = [(name, value) for name, value in self.p.items() if name != 'group']
        layer_params = sum(v.size for name, v in weights if name.startswith('layers.'))
        concept_params = sum(v.size for name, v in weights
                             if name.startswith('layers.') and name.endswith(('.g', '.log_s', '.log_c', '.A', '.parent')))
        return dict(dim=self.dim, layers=self.layers,
                    groups=[list(self.p[f'layers.{i}.g'].shape[:2]) for i in range(self.layers)],
                    parameters=sum(v.size for _, v in weights), bytes=sum(v.nbytes for _, v in weights),
                    layer_parameters=layer_params, layer_concept_parameters=concept_params)

    def _project(self, i, h):
        H, d, _ = self.shape(i)
        q = (h @ self.p[f'layers.{i}.query.weight'].T).reshape(*h.shape[:-1], H, d)
        v = (h @ self.p[f'layers.{i}.value.weight'].T).reshape(*h.shape[:-1], H, d)
        return q, v

    def _project_tabular(self, i, h, prev_ctx=None):
        H, d, _ = self.shape(i)
        h_norm = _layer_norm(h, self.p[f'layers.{i}.norm1.weight'], self.p[f'layers.{i}.norm1.bias'])
        q0 = (h_norm @ self.p[f'layers.{i}.query.weight'].T).reshape(*h.shape[:-1], H, d)
        k0 = (h_norm @ self.p[f'layers.{i}.key.weight'].T).reshape(*h.shape[:-1], H, d)
        v = (h_norm @ self.p[f'layers.{i}.value.weight'].T).reshape(*h.shape[:-1], H, d)
        q = q0 + (prev_ctx[0] - prev_ctx[1]) / prev_ctx[2] if (prev_ctx is not None and prev_ctx[0].shape[-2:] == (H, d)) else q0
        return q0, q, k0, v

    def _view_bulk(self, i, q0, k0, v, past_k=None, past_v=None):
        pos_bias = self.p[f'layers.{i}.pos_bias']
        W_v = pos_bias.shape[1]
        T, H, d = q0.shape
        keep_past = past_k[-(W_v - 1):] if (W_v > 1 and past_k) else []
        keep_val = past_v[-(W_v - 1):] if (W_v > 1 and past_v) else []
        pk = np.zeros((0, H, d), np.float32) if not keep_past else np.asarray(keep_past, np.float32).reshape(-1, H, d)
        pv = np.zeros((0, H, d), np.float32) if not keep_val else np.asarray(keep_val, np.float32).reshape(-1, H, d)
        pad_len = max(0, W_v - 1 - len(pk))
        k_pad = np.concatenate([np.zeros((pad_len, H, d), np.float32), pk, k0], axis=0)
        v_pad = np.concatenate([np.zeros((pad_len, H, d), np.float32), pv, v], axis=0)
        valid = np.concatenate([np.zeros(pad_len, bool), np.ones(len(pk) + T, bool)])
        k_win = np.lib.stride_tricks.sliding_window_view(k_pad, W_v, axis=0)      # T H d W_v
        v_win = np.lib.stride_tricks.sliding_window_view(v_pad, W_v, axis=0)      # T H d W_v
        valid_w = np.lib.stride_tricks.sliding_window_view(valid, W_v, axis=0)    # T W_v
        scores = np.einsum('thd,thdw->thw', q0, k_win) * np.float32(d ** -.5) + pos_bias[None, :, :]
        scores = np.where(valid_w[:, None, :], scores, -np.inf)
        attn = _softmax(scores, -1)
        tail_k = list((np.concatenate([pk, k0], axis=0))[-W_v:])
        tail_v = list((np.concatenate([pv, v], axis=0))[-W_v:])
        return np.einsum('thw,thdw->thd', attn, v_win), tail_k, tail_v

    def _finish_tabular(self, i, h, read, view, proto, near):
        inp = np.concatenate([read, view, proto, near], -1)
        out = inp @ self.p[f'layers.{i}.out.weight'].T + self.p[f'layers.{i}.out.bias']
        delta = _layer_norm(out, self.p[f'layers.{i}.norm2.weight'], self.p[f'layers.{i}.norm2.bias'])
        gate = self.p.get(f'layers.{i}.gate')
        return h + delta if gate is None else h + gate * delta

    def _finish(self, i, h, read, near=None):
        inp = read if near is None else np.concatenate([read, near], -1)
        out = inp @ self.p[f'layers.{i}.out.weight'].T + self.p[f'layers.{i}.out.bias']
        new = _layer_norm(h + out, self.p[f'layers.{i}.norm1.weight'], self.p[f'layers.{i}.norm1.bias'])
        new = _layer_norm(new, self.p[f'layers.{i}.norm2.weight'], self.p[f'layers.{i}.norm2.bias'])
        gate = self.p.get(f'layers.{i}.gate')                      # how open the layer is (older networks: open)
        return new if gate is None else h + gate * (new - h)

    def _looked(self, i):
        return _softmax(self.p[f'layers.{i}.A'])

    def _written(self, i, h, membership):
        gate = self.p.get(f'layers.{i}.write')
        if gate is None:                                     # checkpoints made before selective writes
            return membership
        norm = np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12)
        direction = (h / norm) * np.sqrt(np.float32(self.dim))
        logits = direction @ gate.T
        strength = .5 + 1 / (1 + np.exp(np.minimum(-logits, 80)))
        return membership * strength[..., None]

    # ---- a whole document --------------------------------------------------------------------------------
    def states(self, ids, memberships=None):
        """The final state of every position of one document (T x D). ``memberships``, a list, receives the
        memberships of every layer (T x H x K)."""
        h = self.p['centre.weight'][np.asarray(ids, np.int64)]
        T = len(h)
        prev_ctx = None
        for i in range(self.layers):
            W = self.shape(i)[2]
            near = np.zeros_like(h)
            for w in range(1, W + 1):
                if w < T:
                    near[w:] += h[:-w] @ self.p[f'layers.{i}.near'][w - 1]
            t_prev = prev_ctx[3] if prev_ctx is not None else None
            if self._linear_layer(i):
                q0, q, k0, v = self._project_tabular(i, h, prev_ctx)
                if self._routed_layer(i):
                    g_g, ls_g, lc_g, g_bar_c, ls_bar_c, l_bar, _ = self._macro_clusters(i, 16)
                    l_c = _gaussian_logits(q, g_bar_c, ls_bar_c, l_bar)
                    t_c = _softmax(l_c, -1)
                    written = self._written(i, h, t_c)
                    mass = np.cumsum(written, 0) + EPS_MASS
                    sums = np.cumsum(written[..., None] * v[:, :, None, :], 0)
                    read = np.einsum('thk,thkd->thd', t_c / mass, sums).reshape(T, -1)
                    g_bar, s_bar, full_t = self._routed_subconcepts(
                        i, q, l_c, g_g, ls_g, lc_g, need_full=(memberships is not None)
                    )
                    if memberships is not None:
                        memberships.append(full_t)
                else:
                    g, log_s, log_c = self.p[f'layers.{i}.g'], self.p[f'layers.{i}.log_s'], self.p[f'layers.{i}.log_c']
                    t = _softmax(_gaussian_logits(q, g, log_s, log_c), -1)
                    if memberships is not None:
                        memberships.append(t)
                    written = self._written(i, h, t)
                    mass = np.cumsum(written, 0) + EPS_MASS
                    sums = np.cumsum(written[..., None] * v[:, :, None, :], 0)
                    read = np.einsum('thk,thkd->thd', t / mass, sums).reshape(T, -1)
                    g_bar, s_bar = self._above_stats(i, t)
                view, _, _ = self._view_bulk(i, q0, k0, v)
                h = self._finish_tabular(i, h, read, view.reshape(T, -1), g_bar.reshape(T, -1), near)
                prev_ctx = (q, g_bar, s_bar, None)
            elif self._tabular_layer(i):
                q0, q, k0, v = self._project_tabular(i, h, prev_ctx)
                t = self._memberships(i, q, t_prev)
                if memberships is not None:
                    memberships.append(t)
                written = self._written(i, h, t)
                mass = np.cumsum(written, 0) + EPS_MASS
                sums = np.cumsum(written[..., None] * v[:, :, None, :], 0)
                weight = np.einsum('thk,hkj->thj', t, self._looked(i)) / mass
                read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, -1)
                view, _, _ = self._view_bulk(i, q0, k0, v)
                g_bar, s_bar = self._above_stats(i, t)
                h = self._finish_tabular(i, h, read, view.reshape(T, -1), g_bar.reshape(T, -1), near)
                prev_ctx = (q, g_bar, s_bar, t)
            else:
                pure = self._pure_layer(i)
                q, v = self._project(i, h + near if pure else h)
                t = self._memberships(i, q, t_prev)
                if memberships is not None:
                    memberships.append(t)
                written = self._written(i, h, t)
                mass = np.cumsum(written, 0) + EPS_MASS
                sums = np.cumsum(written[..., None] * v[:, :, None, :], 0)
                weight = np.einsum('thk,hkj->thj', t, self._looked(i)) / mass
                read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, -1)
                h = self._finish(i, h, read, None if pure else near)
                prev_ctx = (q, None, None, t)
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
        of their group, named by their own tokens: those that enter them, and that follow them, much more than in all
        the text. Edges: in a
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
        u_in, inv_in = np.unique(now, return_inverse=True)
        u_out, inv_out = np.unique(after, return_inverse=True)
        everywhere_in = np.bincount(inv_in, minlength=len(u_in)) / len(now)
        everywhere_out = np.bincount(inv_out, minlength=len(u_out)) / len(after)

        def clean_token(tok_id):
            if not decode:
                return str(int(tok_id)), True
            raw = decode([int(tok_id)])
            stripped = raw.strip()
            is_word = (len(stripped) >= 2 and '<' not in stripped and '>' not in stripped
                       and stripped.lower() not in ('user', 'assistant', 'system', 'endoftext', 'im_start', 'im_end')
                       and any(ch.isalpha() for ch in stripped))
            return (stripped or repr(raw)), is_word

        in_meta = [clean_token(x) for x in u_in]
        out_meta = [clean_token(x) for x in u_out]
        in_labels = [m[0] for m in in_meta]
        out_labels = [m[0] for m in out_meta]
        in_good = np.array([m[1] for m in in_meta], dtype=bool)
        out_good = np.array([m[1] for m in out_meta], dtype=bool)

        def telling(mass, everywhere, text_list, good_mask):
            share = mass / max(float(mass.sum()), 1e-12)
            score = np.where(mass > 0, mass * np.log(np.maximum(share, 1e-300) / np.maximum(everywhere, 1e-300)), -np.inf)
            word_scores = np.where(good_mask, score, -np.inf)
            chosen_scores = word_scores if np.isfinite(word_scores).sum() >= min(labels, len(score)) else score
            order = np.argsort(-chosen_scores)[:labels]
            return [text_list[int(idx)] for idx in order]

        nodes, edges, kept = [], [], []
        for i in range(self.layers):
            role = ('concetto' if self.layers == 1 else
                    ('astrazione' if i == 0 else
                     ('sotto-concetto particolare' if i == self.layers - 1 else 'concetto intermedio')))
            t = np.concatenate(found[i])                                       # positions x H x K
            H, K = t.shape[1:]
            share = t.sum(0) / np.maximum(t.sum((0, 2))[:, None], 1e-12)       # H x K
            threshold = 0. if K <= 1 else (max(1e-3, .25 / K) if K <= 8 else (1. - 1e-5) / K)
            keep = set()
            for h in range(H):
                above = [k for k in range(K) if share[h, k] >= threshold]
                if not above:
                    above = [int(np.argmax(share[h]))]
                elif len(above) > 24:
                    above = sorted(above, key=lambda k: -float(share[h, k]))[:24]
                for k in above:
                    keep.add((h, int(k)))
            kept.append((t, keep))
            for h, k in sorted(keep):
                w = t[:, h, k]
                into = np.bincount(inv_in, weights=w, minlength=len(u_in))
                out = np.bincount(inv_out, weights=w, minlength=len(u_out))
                nodes.append(dict(id=f'{i + 1}.{h + 1}.{k + 1}', layer=i + 1, group=h + 1, concept=k + 1,
                                  role=role, mass=float(share[h, k]),
                                  enters=telling(into, everywhere_in, in_labels, in_good),
                                  follows=telling(out, everywhere_out, out_labels, out_good)))
            if f'layers.{i}.A' in self.p:
                A = self._looked(i)                                            # H x K x K: k reads k'
                for h, k in sorted(keep):
                    for k2 in np.flatnonzero(A[h, k] > 1 / K):
                        if (h, int(k2)) in keep and k2 != k:
                            edges.append(dict(source=f'{i + 1}.{h + 1}.{k + 1}', target=f'{i + 1}.{h + 1}.{k2 + 1}',
                                              weight=float(A[h, k, k2]), kind='legge'))
            else:
                g_l, ls_l, lc_l = self.p[f'layers.{i}.g'], self.p[f'layers.{i}.log_s'], self.p[f'layers.{i}.log_c']
                for h in range(H):
                    kh = [k for h0, k in sorted(keep) if h0 == h]
                    if len(kh) <= 1:
                        continue
                    idx_h = np.asarray(kh, np.int64)
                    gh, lsh, lch = g_l[h, idx_h], ls_l[h, idx_h], lc_l[h, idx_h]
                    invh = np.exp(-2 * lsh)
                    diff = gh[:, None, :] - gh[None, :, :]
                    logits_h = lch[None, :] - .5 * (diff * diff * invh[None, :, :]).sum(-1) - lsh.sum(-1)[None, :]
                    aff = _softmax(logits_h, -1)
                    even = 1. / len(kh)
                    for a_idx, k in enumerate(kh):
                        peers = [b_idx for b_idx, k2 in enumerate(kh) if k2 != k and aff[a_idx, b_idx] > even]
                        if len(peers) > 3:
                            peers = sorted(peers, key=lambda b_idx: -float(aff[a_idx, b_idx]))[:3]
                        for b_idx in peers:
                            k2 = kh[b_idx]
                            edges.append(dict(source=f'{i + 1}.{h + 1}.{k + 1}', target=f'{i + 1}.{h + 1}.{k2 + 1}',
                                              weight=float(aff[a_idx, b_idx]), kind='legge'))
        for i in range(self.layers - 1):
            (t1, keep1), (t2, keep2) = kept[i], kept[i + 1]
            list1, list2 = sorted(keep1), sorted(keep2)
            if not list1 or not list2:
                continue
            K1 = t1.shape[2]
            K2 = t2.shape[2]
            base = t2.mean(0)                                                  # what a next concept holds anyway
            parent_mass = np.maximum(t1.sum(0), 1e-12)                         # H1 x K1
            child_mass = np.maximum(t2.sum(0), 1e-12)                          # H2 x K2
            t1_k = np.stack([t1[:, h, k] for h, k in list1], axis=1)           # positions x |keep1|
            t2_k = np.stack([t2[:, h2, k2] for h2, k2 in list2], axis=1)       # positions x |keep2|
            co_kept = t1_k.T @ t2_k                                            # |keep1| x |keep2|
            parent_mat = self.p.get(f'layers.{i + 1}.parent')
            log_c_next = self.p.get(f'layers.{i + 1}.log_c')
            struct_prob = (_softmax(np.clip(parent_mat, -10., 10.) + log_c_next[:, None, :], axis=-1)
                           if parent_mat is not None and log_c_next is not None else None)
            for idx1, (h, k) in enumerate(list1):
                pm = float(parent_mass[h, k])
                for idx2, (h2, k2) in enumerate(list2):
                    cm = float(co_kept[idx1, idx2])
                    spread_val = cm / pm
                    pshare = float(cm / child_mass[h2, k2])
                    spec = (float(struct_prob[h, k, k2])
                            if struct_prob is not None and h == h2 and h < struct_prob.shape[0] else None)
                    weight = float(.5 * (spec + spread_val) if spec is not None else spread_val)
                    lift = float(weight / max(float(base[h2, k2]), 1e-12))
                    if ((K2 == 1 and h == h2)
                            or (weight > 1 / K2 and weight > base[h2, k2])
                            or (K1 > 1 and pshare > 1 / K1 and lift > 1.05)):
                        edges.append(dict(source=f'{i + 1}.{h + 1}.{k + 1}', target=f'{i + 2}.{h2 + 1}.{k2 + 1}',
                                          weight=weight, parent_share=pshare,
                                          kind='porta a', relation='specializza', lift=lift))
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
        mem_k = [16 if net._routed_layer(i) else shapes[i][1] for i in range(net.layers)]
        self.mass = [np.full((H, mk), EPS_MASS, np.float32) for (H, _, _), mk in zip(shapes, mem_k)]
        self.sums = [np.zeros((H, mk, d), np.float32) for (H, _, d), mk in zip(shapes, mem_k)]
        self.recent = [[] for _ in range(net.layers)]                       # inputs of the layer, newest last
        self.view_k = [[] for _ in range(net.layers)]                       # perspective keys within view_window
        self.view_v = [[] for _ in range(net.layers)]                       # perspective values within view_window
        self.state = None
        self.keys, self.after = [], []                                       # the memory of the document

    def copy(self):
        other = NetCursor.__new__(NetCursor)
        other.net = self.net
        other.mass = [m.copy() for m in self.mass]
        other.sums = [s.copy() for s in self.sums]
        other.recent = [list(r) for r in self.recent]
        other.view_k = [list(r) for r in self.view_k]
        other.view_v = [list(r) for r in self.view_v]
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
        prev_ctx = None
        for i in range(net.layers):
            W = net.shape(i)[2]
            before = np.array(self.recent[i][-W:]).reshape(-1, net.dim)
            padded = np.vstack([np.zeros((W - len(before), net.dim), np.float32), before, h])
            near = np.zeros_like(h)
            for w in range(1, W + 1):
                near += padded[W - w:W - w + T] @ net.p[f'layers.{i}.near'][w - 1]
            t_prev = prev_ctx[3] if prev_ctx is not None else None
            if net._linear_layer(i):
                q0, q, k0, v = net._project_tabular(i, h, prev_ctx)
                if net._routed_layer(i):
                    g_g, ls_g, lc_g, g_bar_c, ls_bar_c, l_bar, _ = net._macro_clusters(i, 16)
                    l_c = _gaussian_logits(q, g_bar_c, ls_bar_c, l_bar)
                    t_c = _softmax(l_c, -1)
                    written = net._written(i, h, t_c)
                    mass = np.cumsum(written, 0) + self.mass[i]
                    sums = np.cumsum(written[..., None] * v[:, :, None, :], 0) + self.sums[i]
                    read = np.einsum('thk,thkd->thd', t_c / mass, sums).reshape(T, -1)
                    g_bar, s_bar, _ = net._routed_subconcepts(i, q, l_c, g_g, ls_g, lc_g, need_full=False)
                else:
                    g, log_s, log_c = net.p[f'layers.{i}.g'], net.p[f'layers.{i}.log_s'], net.p[f'layers.{i}.log_c']
                    t = _softmax(_gaussian_logits(q, g, log_s, log_c), -1)
                    written = net._written(i, h, t)
                    mass = np.cumsum(written, 0) + self.mass[i]
                    sums = np.cumsum(written[..., None] * v[:, :, None, :], 0) + self.sums[i]
                    read = np.einsum('thk,thkd->thd', t / mass, sums).reshape(T, -1)
                    g_bar, s_bar = net._above_stats(i, t)
                view, self.view_k[i], self.view_v[i] = net._view_bulk(i, q0, k0, v, self.view_k[i], self.view_v[i])
                self.mass[i], self.sums[i] = mass[-1], sums[-1]
                self.recent[i] = list(padded[-W:])
                h = net._finish_tabular(i, h, read, view.reshape(T, -1), g_bar.reshape(T, -1), near)
                prev_ctx = (q, g_bar, s_bar, None)
            elif net._tabular_layer(i):
                q0, q, k0, v = net._project_tabular(i, h, prev_ctx)
                t = net._memberships(i, q, t_prev)
                written = net._written(i, h, t)
                mass = np.cumsum(written, 0) + self.mass[i]
                sums = np.cumsum(written[..., None] * v[:, :, None, :], 0) + self.sums[i]
                weight = np.einsum('thk,hkj->thj', t, net._looked(i)) / mass
                read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, -1)
                view, self.view_k[i], self.view_v[i] = net._view_bulk(i, q0, k0, v, self.view_k[i], self.view_v[i])
                g_bar, s_bar = net._above_stats(i, t)
                self.mass[i], self.sums[i] = mass[-1], sums[-1]
                self.recent[i] = list(padded[-W:])
                h = net._finish_tabular(i, h, read, view.reshape(T, -1), g_bar.reshape(T, -1), near)
                prev_ctx = (q, g_bar, s_bar, t)
            else:
                pure = net._pure_layer(i)
                q, v = net._project(i, h + near if pure else h)
                t = net._memberships(i, q, t_prev)
                written = net._written(i, h, t)
                mass = np.cumsum(written, 0) + self.mass[i]
                sums = np.cumsum(written[..., None] * v[:, :, None, :], 0) + self.sums[i]
                weight = np.einsum('thk,hkj->thj', t, net._looked(i)) / mass
                read = np.einsum('thj,thjd->thd', weight, sums).reshape(T, -1)
                self.mass[i], self.sums[i] = mass[-1], sums[-1]
                self.recent[i] = list(padded[-W:])
                h = net._finish(i, h, read, None if pure else near)
                prev_ctx = (q, None, None, t)
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
        prev_ctx = None
        for i in range(net.layers):
            W = net.shape(i)[2]
            near = np.zeros_like(h)
            for w, previous in enumerate(reversed(self.recent[i][-W:]), 1):
                near += previous @ net.p[f'layers.{i}.near'][w - 1]
            t_prev = prev_ctx[3] if prev_ctx is not None else None
            if net._linear_layer(i):
                q0, q, k0, v = net._project_tabular(i, h, prev_ctx)
                if net._routed_layer(i):
                    g_g, ls_g, lc_g, g_bar_c, ls_bar_c, l_bar, _ = net._macro_clusters(i, 16)
                    l_c = _gaussian_logits(q, g_bar_c, ls_bar_c, l_bar)
                    t_c = _softmax(l_c, -1)
                    written = net._written(i, h, t_c)
                    self.mass[i] += written
                    self.sums[i] += written[..., None] * v[:, None, :]
                    read = np.einsum('hk,hkd->hd', t_c / self.mass[i], self.sums[i]).reshape(-1)
                    g_bar, s_bar, _ = net._routed_subconcepts(i, q, l_c, g_g, ls_g, lc_g, need_full=False)
                else:
                    g, log_s, log_c = net.p[f'layers.{i}.g'], net.p[f'layers.{i}.log_s'], net.p[f'layers.{i}.log_c']
                    t = _softmax(_gaussian_logits(q, g, log_s, log_c), -1)
                    written = net._written(i, h, t)
                    self.mass[i] += written
                    self.sums[i] += written[..., None] * v[:, None, :]
                    read = np.einsum('hk,hkd->hd', t / self.mass[i], self.sums[i]).reshape(-1)
                    g_bar, s_bar = net._above_stats(i, t)
                pos_bias = net.p[f'layers.{i}.pos_bias']
                W_v = pos_bias.shape[1]
                self.view_k[i] = (self.view_k[i] + [k0])[-W_v:]
                self.view_v[i] = (self.view_v[i] + [v])[-W_v:]
                L_cur = len(self.view_k[i])
                k_buf = np.stack(self.view_k[i], axis=0)                    # L_cur H d
                v_buf = np.stack(self.view_v[i], axis=0)                    # L_cur H d
                scores = np.einsum('hd,whd->hw', q0, k_buf) * np.float32(q0.shape[-1] ** -.5) + pos_bias[:, W_v - L_cur:]
                attn = _softmax(scores, -1)
                view = np.einsum('hw,whd->hd', attn, v_buf).reshape(-1)
                self.recent[i] = (self.recent[i] + [h])[-W:]
                h = net._finish_tabular(i, h, read, view, g_bar.reshape(-1), near)
                prev_ctx = (q, g_bar, s_bar, None)
            elif net._tabular_layer(i):
                q0, q, k0, v = net._project_tabular(i, h, prev_ctx)
                t = net._memberships(i, q, t_prev)
                written = net._written(i, h, t)
                self.mass[i] += written
                self.sums[i] += written[..., None] * v[:, None, :]
                weight = np.einsum('hk,hkj->hj', t, net._looked(i)) / self.mass[i]
                read = np.einsum('hj,hjd->hd', weight, self.sums[i]).reshape(-1)
                pos_bias = net.p[f'layers.{i}.pos_bias']
                W_v = pos_bias.shape[1]
                self.view_k[i] = (self.view_k[i] + [k0])[-W_v:]
                self.view_v[i] = (self.view_v[i] + [v])[-W_v:]
                L_cur = len(self.view_k[i])
                k_buf = np.stack(self.view_k[i], axis=0)                    # L_cur H d
                v_buf = np.stack(self.view_v[i], axis=0)                    # L_cur H d
                scores = np.einsum('hd,whd->hw', q0, k_buf) * np.float32(q0.shape[-1] ** -.5) + pos_bias[:, W_v - L_cur:]
                attn = _softmax(scores, -1)
                view = np.einsum('hw,whd->hd', attn, v_buf).reshape(-1)
                g_bar, s_bar = net._above_stats(i, t)
                self.recent[i] = (self.recent[i] + [h])[-W:]
                h = net._finish_tabular(i, h, read, view, g_bar.reshape(-1), near)
                prev_ctx = (q, g_bar, s_bar, t)
            else:
                pure = net._pure_layer(i)
                q, v = net._project(i, h + near if pure else h)
                t = net._memberships(i, q, t_prev)
                written = net._written(i, h, t)
                self.mass[i] += written
                self.sums[i] += written[..., None] * v[:, None, :]
                weight = np.einsum('hk,hkj->hj', t, net._looked(i)) / self.mass[i]
                read = np.einsum('hj,hjd->hd', weight, self.sums[i]).reshape(-1)
                self.recent[i] = (self.recent[i] + [h])[-W:]
                h = net._finish(i, h, read, None if pure else near)
                prev_ctx = (q, None, None, t)
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
