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

from acan.concept_algebra import natural_em_parameters
from .concept_net import ConceptNet


class _GroupedTokenLogProb(torch.autograd.Function):
    """Exact log p(y | h, G_y) for grouped token concepts without materializing N x S x D gathers."""

    @staticmethod
    def forward(ctx, flat, rank_tgt, g, centre, log_mass, log_scale, members, valid):
        N, D = flat.shape
        if N == 0:
            ctx.empty = True
            ctx.centre_rows, ctx.dim = centre.shape
            return flat.new_empty(0)
        ctx.empty = False
        S = members.shape[1]
        inv_scale = torch.exp(-2 * log_scale)
        order = torch.argsort(g)
        g_s, x_s, r_s = g[order], flat[order], rank_tgt[order]
        ug, counts = torch.unique_consecutive(g_s, return_counts=True)
        mem_u, val_u = members[ug], valid[ug]
        C_u = centre[mem_u]
        norms_u = C_u.square().sum(-1)
        mass_u = log_mass[mem_u]
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        tail_mask = counts <= 16
        head_idx = torch.nonzero(~tail_mask, as_tuple=False).squeeze(-1)
        tail_idx = torch.nonzero(tail_mask, as_tuple=False).squeeze(-1)
        P_s = flat.new_empty(N, S)
        out_s = flat.new_empty(N)
        for u in head_idx.tolist():
            a = int(starts[u]); b = a + int(counts[u])
            lg = (x_s[a:b] @ C_u[u].T - .5 * norms_u[u]) * inv_scale + mass_u[u]
            lg = lg.masked_fill(~val_u[u], float('-inf'))
            lse = torch.logsumexp(lg, dim=-1, keepdim=True)
            P_s[a:b] = torch.exp(lg - lse)
            out_s[a:b] = lg.gather(1, r_s[a:b, None])[:, 0] - lse[:, 0]
        if len(tail_idx):
            t_counts = counts[tail_idx]
            B_max, T_u = int(t_counts.max().item()), len(tail_idx)
            u_rep = torch.repeat_interleave(torch.arange(T_u, device=flat.device), t_counts)
            local_offsets = torch.cat([t_counts.new_zeros(1), t_counts.cumsum(0)[:-1]])
            slot = torch.arange(int(t_counts.sum().item()), device=flat.device) - local_offsets[u_rep]
            src_pos = starts[tail_idx][u_rep] + slot
            X_pad = flat.new_zeros(T_u, B_max, D)
            X_pad[u_rep, slot] = x_s[src_pos]
            lg = (torch.bmm(X_pad, C_u[tail_idx].transpose(1, 2)) - .5 * norms_u[tail_idx, None, :]) * inv_scale + mass_u[tail_idx, None, :]
            lg = lg.masked_fill(~val_u[tail_idx, None, :], float('-inf'))
            lse = torch.logsumexp(lg, dim=-1, keepdim=True)
            P_s[src_pos] = torch.exp(lg - lse)[u_rep, slot]
            out_s[src_pos] = lg[u_rep, slot].gather(1, r_s[src_pos, None])[:, 0] - lse[u_rep, slot, 0]
        else:
            B_max = 0
            u_rep = slot = src_pos = flat.new_empty(0, dtype=torch.long)
        out = flat.new_empty(N)
        out[order] = out_s
        ctx.save_for_backward(order, x_s, r_s, P_s, counts, starts, head_idx, tail_idx,
                              mem_u, val_u, C_u, inv_scale, u_rep, slot, src_pos)
        ctx.B_max = B_max
        ctx.centre_rows = centre.shape[0]
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty:
            z_c = grad_out.new_zeros(ctx.centre_rows, ctx.dim)
            return grad_out.new_empty(0, ctx.dim), None, None, z_c, grad_out.new_zeros(ctx.centre_rows), grad_out.new_zeros(()), None, None
        (order, x_s, r_s, P_s, counts, starts, head_idx, tail_idx,
         mem_u, val_u, C_u, inv_scale, u_rep, slot, src_pos) = ctx.saved_tensors
        N, D = x_s.shape
        U, S, _ = C_u.shape
        go_s = grad_out[order]
        D_s = -P_s * go_s[:, None]
        D_s.scatter_add_(1, r_s[:, None], go_s[:, None])
        grad_x_s = x_s.new_empty(N, D)
        grad_C_u = C_u.new_empty(U, S, D)
        grad_mass_u = C_u.new_empty(U, S)
        for u in head_idx.tolist():
            a = int(starts[u]); b = a + int(counts[u])
            dk, xk, ck = D_s[a:b], x_s[a:b], C_u[u]
            ms = dk.sum(0)
            grad_mass_u[u] = ms
            grad_x_s[a:b] = (dk @ ck) * inv_scale
            grad_C_u[u] = (dk.T @ xk - ms[:, None] * ck) * inv_scale
        if len(tail_idx):
            T_u, B_max = len(tail_idx), ctx.B_max
            D_pad = x_s.new_zeros(T_u, B_max, S)
            X_pad = x_s.new_zeros(T_u, B_max, D)
            D_pad[u_rep, slot] = D_s[src_pos]
            X_pad[u_rep, slot] = x_s[src_pos]
            C_t = C_u[tail_idx]
            ms_t = D_pad.sum(1)
            grad_mass_u[tail_idx] = ms_t
            grad_x_s[src_pos] = (torch.bmm(D_pad, C_t) * inv_scale)[u_rep, slot]
            grad_C_u[tail_idx] = (torch.bmm(D_pad.transpose(1, 2), X_pad) - ms_t[:, :, None] * C_t) * inv_scale
        grad_log_scale = -2. * ((x_s * grad_x_s).sum() - .5 * inv_scale * (grad_mass_u * C_u.square().sum(-1)).sum())
        grad_flat = x_s.new_empty(N, D)
        grad_flat[order] = grad_x_s
        flat_mem = mem_u[val_u]
        grad_centre = C_u.new_zeros(ctx.centre_rows, D)
        grad_centre[flat_mem] = grad_C_u[val_u]
        grad_log_mass = C_u.new_zeros(ctx.centre_rows)
        grad_log_mass[flat_mem] = grad_mass_u[val_u]
        return grad_flat, None, None, grad_centre, grad_log_mass, grad_log_scale, None, None


def memberships_logits(q, g, log_s, log_c):
    """log C_k + log N(q; g_k, s_k) up to a constant, for every position and concept of every head, with the square
    expanded into batched BLAS matrix products (no table position x concept x dimension). q: ... x H x d; g, log_s: H x K x d."""
    inv = torch.exp(-2 * log_s)                                              # H K d
    if q.ndim >= 3:
        shape = q.shape[:-2]
        H, K, d = g.shape
        qf = q.reshape(-1, H, d).transpose(0, 1)                             # H N d
        ginv = (g * inv).transpose(1, 2)                                     # H d K
        quad = torch.bmm(qf.square(), inv.transpose(1, 2)) - 2. * torch.bmm(qf, ginv) + (g.square() * inv).sum(-1)[:, None, :]
        out = log_c[:, None, :] - .5 * quad - log_s.sum(-1)[:, None, :]
        return out.transpose(0, 1).view(*shape, H, K)
    quad = torch.einsum('...hd,hkd->...hk', q * q, inv) - 2 * torch.einsum('...hd,hkd->...hk', q, g * inv) \
        + (g * g * inv).sum(-1)
    return log_c - .5 * quad - log_s.sum(-1)


def _macro_clusters(g, log_s, log_c, res_var_div=1., G_c=16):
    """Aggregate K = G_c * S_c fine Gaussian sub-concepts into G_c exact moment-matched macro-clusters,
    preserving exact invariance under coordinate widening (res_var_div = 2 ** widens)."""
    H, K, d = g.shape
    S_c = K // G_c
    r = float(res_var_div)
    g_g = g.view(H, G_c, S_c, d)
    ls_g = log_s.view(H, G_c, S_c, d)
    lc_g = log_c.view(H, G_c, S_c)
    true_lp = lc_g - (1. - 1. / r) * ls_g.sum(-1)
    w_sub = torch.softmax(true_lp, dim=-1).unsqueeze(-1)
    g_bar = (w_sub * g_g).sum(2)
    var_unit = (w_sub * (torch.exp(2. * ls_g) / r + (g_g - g_bar.unsqueeze(2)).square())).sum(2).clamp_min(1e-5)
    ls_bar = .5 * torch.log(var_unit * r)
    l_bar = torch.logsumexp(true_lp, dim=-1) + (1. - 1. / r) * ls_bar.sum(-1)
    return g_g, ls_g, lc_g, g_bar, ls_bar, l_bar, var_unit


class _HybridRoutedSubConcepts(torch.autograd.Function):
    """Evaluate S fine Gaussian sub-concepts inside each position's routed macro-cluster in O(N * S * d) BLAS
    (splitting heavy buckets >= 32 from padded tail buckets < 32, with Top-m sparse activation and zero K^2)."""

    @staticmethod
    def forward(ctx, q, grp, g, log_s, log_c, top_m, valid_mask):
        N, H, d = q.shape
        _, G, S, _ = g.shape
        BG = H * G
        inv_raw = torch.exp(-2. * log_s)
        inv = inv_raw.reshape(BG, S, d)
        s2 = torch.exp(2. * log_s).reshape(BG, S, d)
        gf = g.reshape(BG, S, d)
        ginv = gf * inv
        bias = (log_c - .5 * (g.square() * inv_raw).sum(-1) - log_s.sum(-1)).reshape(BG, S)

        b_flat = (grp + (torch.arange(H, device=q.device) * G)[None, :]).reshape(-1)
        q_flat = q.reshape(N * H, d)
        m_flat = valid_mask[:, None].expand(N, H).reshape(-1)
        order = torch.argsort(b_flat)
        b_sorted = b_flat[order]
        q_sorted = q_flat[order]
        q2_sorted = q_sorted.square()
        m_sorted = m_flat[order]
        counts = torch.bincount(b_sorted, minlength=BG)
        starts = torch.zeros(BG + 1, dtype=torch.long, device=q.device)
        starts[1:] = counts.cumsum(0)

        probs_sorted = q.new_empty(N * H, S)
        proto_sorted = q.new_empty(N * H, d)
        var_sorted = q.new_empty(N * H, d)
        bc = q.new_zeros(BG, S)
        bu = q.new_zeros(BG, S, d)
        bq = q.new_zeros(BG, S, d)

        counts_cpu = counts.tolist()
        starts_cpu = starts.tolist()
        heavy = [b for b, c in enumerate(counts_cpu) if c >= 32]
        tail = [b for b, c in enumerate(counts_cpu) if 0 < c < 32]

        for b in heavy:
            a, e = starts_cpu[b], starts_cpu[b + 1]
            qb, q2b, mb = q_sorted[a:e], q2_sorted[a:e], m_sorted[a:e, None]
            logits = bias[b][None, :] - .5 * (q2b @ inv[b].T - 2. * (qb @ ginv[b].T))
            if 0 < top_m < S:
                tv, ti = logits.topk(top_m, dim=-1)
                pb = logits.new_zeros(e - a, S).scatter_(1, ti, torch.softmax(tv, -1))
            else:
                pb = torch.softmax(logits, -1)
            probs_sorted[a:e] = pb
            proto_sorted[a:e] = pb @ gf[b]
            var_sorted[a:e] = pb @ s2[b]
            pbm = pb * mb
            bc[b] = pbm.sum(0)
            bu[b] = pbm.T @ qb
            bq[b] = pbm.T @ q2b

        tail_info = None
        if tail:
            tail_t = torch.tensor(tail, dtype=torch.long, device=q.device)
            T_u = len(tail)
            t_counts = counts[tail_t]
            B_max = int(t_counts.max())
            t_starts = starts[tail_t]
            u_rep = torch.repeat_interleave(torch.arange(T_u, device=q.device), t_counts)
            total_tail = int(t_counts.sum())
            cum_t = torch.zeros(T_u + 1, dtype=torch.long, device=q.device)
            cum_t[1:] = t_counts.cumsum(0)
            slot = torch.arange(total_tail, device=q.device) - cum_t[:-1][u_rep]
            src_pos = t_starts[u_rep] + slot
            Q_pad = q.new_zeros(T_u, B_max, d); Q_pad[u_rep, slot] = q_sorted[src_pos]
            Q2_pad = Q_pad.square()
            M_pad = q.new_zeros(T_u, B_max, 1); M_pad[u_rep, slot, 0] = m_sorted[src_pos]
            valid = torch.arange(B_max, device=q.device)[None, :] < t_counts[:, None]
            inv_t, ginv_t, gf_t, s2_t, bias_t = inv[tail_t], ginv[tail_t], gf[tail_t], s2[tail_t], bias[tail_t]
            logits = bias_t[:, None, :] - .5 * (torch.bmm(Q2_pad, inv_t.transpose(1, 2)) - 2. * torch.bmm(Q_pad, ginv_t.transpose(1, 2)))
            if 0 < top_m < S:
                tv, ti = logits.topk(top_m, dim=-1)
                P_pad = logits.new_zeros(T_u, B_max, S).scatter_(2, ti, torch.softmax(tv, -1) * valid.unsqueeze(-1))
            else:
                P_pad = torch.softmax(logits, -1) * valid.unsqueeze(-1)
            probs_sorted[src_pos] = P_pad[u_rep, slot]
            proto_sorted[src_pos] = torch.bmm(P_pad, gf_t)[u_rep, slot]
            var_sorted[src_pos] = torch.bmm(P_pad, s2_t)[u_rep, slot]
            Pm = P_pad * M_pad
            bc[tail_t] = Pm.sum(1)
            bu[tail_t] = torch.bmm(Pm.transpose(1, 2), Q_pad)
            bq[tail_t] = torch.bmm(Pm.transpose(1, 2), Q2_pad)
            tail_info = (tail_t, u_rep, slot, src_pos, B_max, Q_pad, P_pad)

        proto_flat = q.new_empty(N * H, d); proto_flat[order] = proto_sorted
        var_flat = q.new_empty(N * H, d); var_flat[order] = var_sorted
        ctx.save_for_backward(q_sorted, q2_sorted, probs_sorted, order, gf, inv)
        ctx.heavy = [(b, starts_cpu[b], starts_cpu[b + 1]) for b in heavy]
        ctx.tail_info = tail_info
        ctx.shapes = (N, H, G, S, d)
        return (proto_flat.view(N, H, d), var_flat.view(N, H, d).detach(),
                bc.view(H, G * S).detach(), bu.view(H, G * S, d).detach(), bq.view(H, G * S, d).detach())

    @staticmethod
    def backward(ctx, grad_proto, *_):
        q_sorted, q2_sorted, probs_sorted, order, gf, inv = ctx.saved_tensors
        N, H, G, S, d = ctx.shapes
        BG = H * G
        gp_sorted = grad_proto.reshape(N * H, d)[order]
        gq_sorted = q_sorted.new_empty(N * H, d)
        gg = gf.new_zeros(BG, S, d)
        gls = gf.new_zeros(BG, S, d)
        glc = gf.new_zeros(BG, S)
        ginv = gf * inv

        for b, a, e in ctx.heavy:
            qb, q2b, pb, gpb = q_sorted[a:e], q2_sorted[a:e], probs_sorted[a:e], gp_sorted[a:e]
            gb, invb, ginvb = gf[b], inv[b], ginv[b]
            gg[b] = pb.T @ gpb
            dpb = gpb @ gb.T
            dl = pb * (dpb - (pb * dpb).sum(-1, keepdim=True))
            ms = dl.sum(0)
            glc[b] = ms
            gq_sorted[a:e] = dl @ ginvb - qb * (dl @ invb)
            dl_q = dl.T @ qb
            gg[b].add_((dl_q - ms[:, None] * gb) * invb)
            gls[b] = (dl.T @ q2b - 2. * dl_q * gb + ms[:, None] * gb.square()) * invb - ms[:, None]

        if ctx.tail_info is not None:
            tail_t, u_rep, slot, src_pos, B_max, Q_pad, P_pad = ctx.tail_info
            T_u = len(tail_t)
            GP_pad = Q_pad.new_zeros(T_u, B_max, d); GP_pad[u_rep, slot] = gp_sorted[src_pos]
            gf_t, inv_t, ginv_t = gf[tail_t], inv[tail_t], ginv[tail_t]
            gg_t = torch.bmm(P_pad.transpose(1, 2), GP_pad)
            dP = torch.bmm(GP_pad, gf_t.transpose(1, 2))
            dL = P_pad * (dP - (P_pad * dP).sum(-1, keepdim=True))
            ms_t = dL.sum(1)
            glc[tail_t] = ms_t
            gq_pad = torch.bmm(dL, ginv_t) - Q_pad * torch.bmm(dL, inv_t)
            gq_sorted[src_pos] = gq_pad[u_rep, slot]
            dL_Q = torch.bmm(dL.transpose(1, 2), Q_pad)
            gg_t.add_((dL_Q - ms_t[:, :, None] * gf_t) * inv_t)
            gg[tail_t] = gg_t
            diff2 = torch.bmm(dL.transpose(1, 2), Q_pad.square()) - 2. * dL_Q * gf_t + ms_t[:, :, None] * gf_t.square()
            gls[tail_t] = diff2 * inv_t - ms_t[:, :, None]

        gq_flat = q_sorted.new_empty(N * H, d); gq_flat[order] = gq_sorted
        return gq_flat.view(N, H, d), None, gg.view(H, G, S, d), gls.view(H, G, S, d), glc.view(H, G, S), None, None


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
    def __init__(self, dim, heads, concepts, window, head_dim=None, prev_concepts=0,
                 pure_concepts=False, tabular=True, linear_concepts=True, subconcept_top_m=4,
                 view_window=32, res_var_div=1.):
        super().__init__()
        self.h, self.d = heads, int(head_dim or max(1, dim // max(heads, 1)))
        self.pure_concepts = bool(pure_concepts)
        self.tabular = bool(tabular)
        self.linear_concepts = bool(linear_concepts) and self.tabular
        self.subconcept_top_m = int(subconcept_top_m)
        self.view_window = max(1, int(view_window))
        self.res_var_div = float(res_var_div)
        self.g = nn.Parameter(torch.randn(heads, concepts, self.d) * .5)
        self.log_s = nn.Parameter(torch.zeros(heads, concepts, self.d))
        self.log_c = nn.Parameter(torch.zeros(heads, concepts))
        if self.linear_concepts:
            self.register_parameter('parent', None)
            self.register_parameter('A', None)
        else:
            if prev_concepts > 0:
                self.parent = nn.Parameter(torch.zeros(heads, prev_concepts, concepts))
            else:
                self.register_parameter('parent', None)
            eye = torch.eye(concepts).unsqueeze(0).expand(heads, -1, -1).clone() if concepts > 1 else torch.zeros(heads, 1, 1)
            self.A = nn.Parameter(1.5 * eye)
        self.query = nn.Linear(dim, heads * self.d, bias=False)
        self.value = nn.Linear(dim, heads * self.d, bias=False)
        if self.tabular:
            self.key = nn.Linear(dim, heads * self.d, bias=False)
            slope = torch.linspace(-1.5, 0., self.view_window).unsqueeze(0).repeat(heads, 1)
            self.pos_bias = nn.Parameter(slope)
            out_in = 3 * heads * self.d + dim
        else:
            self.register_parameter('key', None)
            self.register_parameter('pos_bias', None)
            out_in = heads * self.d if self.pure_concepts else heads * self.d + dim
        self.write = nn.Parameter(torch.zeros(heads, dim))  # one content-dependent write strength per head
        self.near = nn.Parameter(torch.randn(window, dim, dim) * .02)
        self.out = nn.Linear(out_in, dim)
        if self.tabular:
            with torch.no_grad():
                self.out.weight[:, 2 * heads * self.d:3 * heads * self.d].zero_()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        # how open the layer is: an existing layer is open (1); a layer just added starts closed (0) and opens by gradient.
        self.gate = nn.Parameter(torch.ones(()))
        self.window = window
        self.chunk = 128 if self.linear_concepts else 64
        self._em_stats = None

    def _local(self, h):
        if not self.pure_concepts or self.tabular:
            return h
        T = h.shape[1]
        return h + sum(F.pad(h, (0, 0, w, 0))[:, :T] @ self.near[w - 1] for w in range(1, self.window + 1))

    def _logits(self, q, t_prev=None):
        logits = memberships_logits(q, self.g, self.log_s, self.log_c)
        if self.parent is not None and t_prev is not None and t_prev.shape[-2:] == (self.h, self.parent.shape[1]):
            lift = torch.einsum('bthp,hpk->bthk', t_prev, torch.exp(self.parent.clamp(-10., 10.)))
            logits = logits + torch.log(lift.clamp_min(1e-12))
        return logits

    def _above_stats(self, t_prob):
        g_bar = torch.einsum('...hk,hkd->...hd', t_prob, self.g)
        var = torch.einsum('...hk,hkd->...hd', t_prob, torch.exp(2. * self.log_s) / self.res_var_div)
        s_bar = torch.sqrt(var + 1e-4)
        return g_bar, s_bar

    def query_state(self, h, prev_ctx=None):
        B, T, _ = h.shape
        if self.tabular:
            h_norm = self.norm1(h)
            q0 = self.query(h_norm).view(B, T, self.h, self.d)
            if prev_ctx is not None and prev_ctx[0].shape[-2:] == (self.h, self.d):
                return q0 + (prev_ctx[0] - prev_ctx[1]) / prev_ctx[2]
            return q0
        return self.query(self._local(h)).view(B, T, self.h, self.d)

    def _view_read(self, q0, k, v, mask):
        B, T, H, d = q0.shape
        W_v = self.pos_bias.shape[1]
        if W_v <= 1:
            return v * mask[:, :, None, None]
        rem = (-T) % W_v
        if rem:
            q_p = F.pad(q0, (0, 0, 0, 0, 0, rem))
            k_p = F.pad(k, (0, 0, 0, 0, 0, rem))
            v_p = F.pad(v, (0, 0, 0, 0, 0, rem))
            m_p = F.pad(mask, (0, rem))
        else:
            q_p, k_p, v_p, m_p = q0, k, v, mask
        nb = q_p.shape[1] // W_v
        q_b = q_p.transpose(1, 2).reshape(B, H, nb, W_v, d)
        k_b = k_p.transpose(1, 2).reshape(B, H, nb, W_v, d)
        v_b = v_p.transpose(1, 2).reshape(B, H, nb, W_v, d)
        z_kv = k_b.new_zeros(B, H, 1, W_v - 1, d)
        k_ctx = torch.cat([torch.cat([z_kv, k_b[:, :, :-1, 1:, :]], dim=2), k_b], dim=3)
        v_ctx = torch.cat([torch.cat([z_kv, v_b[:, :, :-1, 1:, :]], dim=2), v_b], dim=3)
        m_b = m_p.reshape(B, nb, W_v)
        m_ctx = torch.cat([torch.cat([m_b.new_zeros(B, 1, W_v - 1), m_b[:, :-1, 1:]], dim=1), m_b], dim=2)
        span = 2 * W_v - 1
        cache_key = (W_v, q0.device)
        if getattr(self, '_band_key', None) != cache_key:
            r_idx = torch.arange(W_v, device=q0.device)[:, None]
            c_idx = torch.arange(span, device=q0.device)[None, :]
            off = c_idx - r_idx
            self._band_off = off.clamp(0, W_v - 1)
            self._band_in = (off >= 0) & (off < W_v)
            self._band_diag = (off == W_v - 1)
            self._band_key = cache_key
        scores = torch.matmul(q_b, k_ctx.transpose(-1, -2)) * (d ** -.5)
        pb = self.pos_bias[:, self._band_off]
        valid = ((m_ctx[:, None, :, None, :] > 0) | self._band_diag[None, None, None, :, :]) & self._band_in[None, None, None, :, :]
        scores = (scores + pb[None, :, None, :, :]).masked_fill(~valid, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_ctx).reshape(B, H, nb * W_v, d)[:, :, :T, :].transpose(1, 2)
        return out * mask[:, :, None, None]

    def _record_em_dense(self, x_coord, t_masked, mask):
        with torch.no_grad():
            xd, td = x_coord.detach(), t_masked.detach()
            tf = td.flatten(0, 1).permute(1, 2, 0)
            xf = xd.flatten(0, 1).transpose(0, 1)
            bc = tf.sum(-1)
            bu = torch.bmm(tf, xf)
            x2 = xf.square()
            bq = torch.bmm(tf, x2)
            mv = mask.reshape(1, -1, 1)
            n_valid = mv.sum().clamp_min(1.)
            mean_x = (xf * mv).sum(1, keepdim=True) / n_valid
            root_v = ((x2 * mv).sum(1, keepdim=True) / n_valid - mean_x.square()).clamp_min(1e-4)
            self._em_stats = (bc, bu, bq, root_v)

    def forward(self, h, mask, prev_ctx=None, return_context=False):
        B, T, D = h.shape
        near = sum(F.pad(h, (0, 0, w, 0))[:, :T] @ self.near[w - 1] for w in range(1, self.window + 1))
        if self.linear_concepts:
            H, d, K = self.h, self.d, self.g.shape[1]
            h_norm = self.norm1(h)
            w_qkv = torch.cat([self.query.weight, self.key.weight, self.value.weight], 0)
            q0, k0, v = F.linear(h_norm, w_qkv).view(B, T, 3, H, d).unbind(2)
            q = q0 + (prev_ctx[0] - prev_ctx[1]) / prev_ctx[2] if (prev_ctx is not None and prev_ctx[0].shape[-2:] == (H, d)) else q0
            direction = F.normalize(h, dim=-1) * math.sqrt(D)
            strength = .5 + torch.sigmoid(F.linear(direction, self.write))
            if K <= 32 or K % 16 != 0:
                t_prob = torch.softmax(memberships_logits(q, self.g, self.log_s, self.log_c), -1)
                t = t_prob * mask[:, :, None, None]
                if self.training:
                    self._record_em_dense(q, t, mask)
                written = t * strength[..., None]
                mass = torch.cumsum(written, 1) + 1e-6
                read = causal_read(t / mass, written, v, self.chunk).reshape(B, T, H * d)
                g_bar, s_bar = self._above_stats(t_prob)
            else:
                g_g, ls_g, lc_g, g_bar_c, ls_bar_c, l_bar, _ = _macro_clusters(
                    self.g, self.log_s, self.log_c, self.res_var_div, 16
                )
                l_c = memberships_logits(q, g_bar_c, ls_bar_c, l_bar)
                t_c_prob = torch.softmax(l_c, -1)
                t_c = t_c_prob * mask[:, :, None, None]
                written = t_c * strength[..., None]
                mass = torch.cumsum(written, 1) + 1e-6
                read = causal_read(t_c / mass, written, v, self.chunk).reshape(B, T, H * d)
                grp = l_c.detach().argmax(-1).reshape(B * T, H)
                g_bar_f, var_f, bc, bu, bq = _HybridRoutedSubConcepts.apply(
                    q.reshape(B * T, H, d), grp, g_g, ls_g, lc_g, self.subconcept_top_m, mask.reshape(-1)
                )
                if self.training:
                    with torch.no_grad():
                        qf = q.detach().flatten(0, 1).transpose(0, 1)
                        mv = mask.reshape(1, -1, 1)
                        n_valid = mv.sum().clamp_min(1.)
                        mean_q = (qf * mv).sum(1, keepdim=True) / n_valid
                        root_v = ((qf.square() * mv).sum(1, keepdim=True) / n_valid - mean_q.square()).clamp_min(1e-4)
                        self._em_stats = (bc, bu, bq, root_v)
                g_bar = g_bar_f.view(B, T, H, d)
                s_bar = torch.sqrt(var_f.view(B, T, H, d) / self.res_var_div + 1e-4)
            view = self._view_read(q0, k0, v, mask).reshape(B, T, H * d)
            proto = (g_bar * mask[:, :, None, None]).reshape(B, T, H * d)
            delta = self.norm2(self.out(torch.cat([read, view, proto, near], -1)))
            out_h = h + self.gate * delta
            return (out_h, (q, g_bar, s_bar, None)) if return_context else out_h

        t_prev = prev_ctx[3] if prev_ctx is not None else None
        if self.tabular:
            h_norm = self.norm1(h)
            q0 = self.query(h_norm).view(B, T, self.h, self.d)
            k = self.key(h_norm).view(B, T, self.h, self.d)
            v = self.value(h_norm).view(B, T, self.h, self.d)
            if prev_ctx is not None and prev_ctx[0].shape[-2:] == (self.h, self.d):
                q = q0 + (prev_ctx[0] - prev_ctx[1]) / prev_ctx[2]
            else:
                q = q0
        else:
            h_in = h + near if self.pure_concepts else h
            q = self.query(h_in).view(B, T, self.h, self.d)
            v = self.value(h_in).view(B, T, self.h, self.d)
        t_prob = torch.softmax(self._logits(q, t_prev), -1)
        t = t_prob * mask[:, :, None, None]
        if self.training:
            self._record_em_dense(q, t, mask)
        direction = F.normalize(h, dim=-1) * math.sqrt(D)
        strength = .5 + torch.sigmoid(F.linear(direction, self.write))
        written = t * strength[..., None]
        mass = torch.cumsum(written, 1) + 1e-6
        weight = torch.einsum('bthk,hkj->bthj', t, torch.softmax(self.A, -1)) / mass
        read = causal_read(weight, written, v, self.chunk).reshape(B, T, self.h * self.d)
        g_bar, s_bar = self._above_stats(t_prob)
        if self.tabular:
            view = self._view_read(q0, k, v, mask).reshape(B, T, self.h * self.d)
            proto = (g_bar * mask[:, :, None, None]).reshape(B, T, self.h * self.d)
            delta = self.norm2(self.out(torch.cat([read, view, proto, near], -1)))
            out_h = h + self.gate * delta
        else:
            out_proj = self.out(read) if self.pure_concepts else self.out(torch.cat([read, near], -1))
            new = self.norm2(self.norm1(h + out_proj))
            out_h = h + self.gate * (new - h)
        return (out_h, (q, g_bar, s_bar, t_prob)) if return_context else out_h


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
        self.config.setdefault('head_dim', max(1, dim // max(int(config['heads']), 1)))
        tabular = bool(config.get('tabular_hierarchy', True))
        pure = bool(config.get('pure_concepts', False)) and not tabular
        linear_concepts = bool(config.get('linear_concepts', True)) and tabular
        subconcept_top_m = int(config.get('subconcept_top_m', 4))
        default_view = int(config['window']) if (pure or config.get('cache', dim) == 0) else 24
        view_window = max(1, int(config.get('view_window', default_view)))
        res_var_div = float(config.get('res_var_div', 1.))
        self.config.update(tabular_hierarchy=tabular, pure_concepts=pure, linear_concepts=linear_concepts,
                           subconcept_top_m=subconcept_top_m, view_window=view_window, res_var_div=res_var_div)
        self.layers = nn.ModuleList([
            ConceptAttention(dim, config['heads'], k, config['window'], self.config['head_dim'],
                             prev_concepts=(concepts[idx - 1] if idx > 0 else 0),
                             pure_concepts=pure, tabular=tabular, linear_concepts=linear_concepts,
                             subconcept_top_m=subconcept_top_m, view_window=view_window, res_var_div=res_var_div)
            for idx, k in enumerate(concepts)
        ])
        self.final = nn.Linear(dim, dim)
        # The memory of the document, learned: every position read leaves an address (a projection of its state) with
        # the token that followed it; a position reads the tokens that followed the addresses most like its own. One
        # more address, "none is like mine" (its likeness learned from the state), stands for the reading of the
        # network: the memory counts only as far as some address is more alike than that one.
        self.config.setdefault('cache', 0 if pure else min(dim, 16 if linear_concepts else dim))
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
            self.register_buffer('_rank', rank - start[group], persistent=False)
            # Padded group membership, one gather per target token.
            slots = torch.arange(int(counts.max()))[None]
            valid = slots < counts[:, None]
            members = order[(start[:, None] + slots).clamp(max=len(order) - 1)]
            self.register_buffer('_members', torch.where(valid, members, 0), persistent=False)
            self.register_buffer('_valid', valid, persistent=False)

    @property
    def device(self):
        return self.centre.weight.device

    def states(self, ids, mask):
        h = self.centre(ids)
        prev_ctx = None
        for layer in self.layers:
            h, prev_ctx = layer(h, mask, prev_ctx=prev_ctx, return_context=True)
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
        same = targets[:, None, :] == targets[:, :, None]                       # the token after j is the true one
        z = self.cache_valve(h)[..., 0]
        if self.config.get('sentinel'):                        # "none is like mine" is one more address
            both = torch.softmax(torch.cat([z[..., None], scores], -1), -1)
            return torch.log(both[..., 0] * net_lp.exp() + (both[..., 1:] * same).sum(-1) + 1e-30)
        anything = allowed.any(-1)
        a = torch.softmax(scores.masked_fill(~anything[..., None], 0.), -1) * anything[..., None]
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
        if flat.shape[0] * self._members.shape[1] <= 16_384:
            members, valid = self._members[g], self._valid[g]
            centres = self.centre.weight
            matching = torch.bmm(centres[members], flat.unsqueeze(-1)).squeeze(-1)
            norms = centres.square().sum(-1)[members]
            logits = (matching - .5 * norms) * torch.exp(-2 * self.log_scale) + self.log_mass[members]
            inside = torch.log_softmax(logits.masked_fill(~valid, float('-inf')), -1).gather(1, self._rank[tgt][:, None])[:, 0]
        else:
            inside = _GroupedTokenLogProb.apply(flat, self._rank[tgt], g, self.centre.weight, self.log_mass,
                                                self.log_scale, self._members, self._valid)
        return self._cache(h, mask, targets, (lp + inside).view(targets.shape))

    # ---- growth: a bigger network that reads exactly as the smaller one did -----------------------------------------
    def split_concepts(self, i, scale=.16):
        """Every concept of layer i becomes two with half the mass each, moment-matched around the parent so the reading
        at step zero stays nearly identical while narrower dispersions and antipodal shifts pull the two halves apart."""
        layer = self.layers[i]
        was_single = layer.g.shape[1] == 1
        with torch.no_grad():
            base_g, base_log_s = layer.g, layer.log_s
            if was_single and scale and getattr(layer, '_k1_stats', None) is not None:
                mean_q, log_std_q = layer._k1_stats
                base_g = mean_q.to(device=self.device, dtype=layer.g.dtype).unsqueeze(1).clone()
                base_log_s = log_std_q.to(device=self.device, dtype=layer.log_s.dtype).unsqueeze(1).clone()
            K = base_g.shape[1]
            if scale:
                step = float(scale) / math.sqrt(max(layer.d, 1))
                u = (torch.randn_like(base_g) * step).clamp(-.45, .45)
                e = u * torch.exp(base_log_s)
                child_log_s = base_log_s + .5 * torch.log1p(-u.square())
            else:
                e = torch.zeros_like(base_g)
                child_log_s = base_log_s
            if layer.A is None:
                # Interleave children (2k, 2k+1) so each macro-cluster preserves exact mass, mean, and total variance!
                g = torch.stack([base_g + e, base_g - e], dim=2).reshape(layer.h, 2 * K, layer.d)
                log_s = torch.stack([child_log_s, child_log_s], dim=2).reshape(layer.h, 2 * K, layer.d)
                log_c = torch.stack([layer.log_c, layer.log_c], dim=2).reshape(layer.h, 2 * K) - math.log(2)
                A = new_parent = next_parent = None
            else:
                g = torch.cat([base_g + e, base_g - e], 1)
                log_s = torch.cat([child_log_s, child_log_s], 1)
                log_c = torch.cat([layer.log_c, layer.log_c], 1) - math.log(2)
                A = torch.cat([torch.cat([layer.A, layer.A], 2)] * 2, 1) - math.log(2)
                if scale:
                    p_self = .5 + min(.5 * float(scale), .12)
                    idx = torch.arange(K, device=self.device)
                    stay, cross = math.log(2. * p_self), math.log(2. * (1. - p_self))
                    A[:, idx, idx] += stay
                    A[:, idx + K, idx + K] += stay
                    A[:, idx, idx + K] += cross
                    A[:, idx + K, idx] += cross
                new_parent = None
                if layer.parent is not None:
                    p_eps = torch.randn_like(layer.parent) * (float(scale) * .04) if scale else 0.
                    new_parent = torch.cat([layer.parent + p_eps, layer.parent - p_eps], 2)
                next_parent = None
                if i + 1 < len(self.layers) and self.layers[i + 1].parent is not None:
                    nxt_p = self.layers[i + 1].parent
                    next_parent = torch.cat([nxt_p, nxt_p], 1)
        layer.g, layer.log_s, layer.log_c = (nn.Parameter(x.contiguous()) for x in (g, log_s, log_c))
        if A is not None:
            layer.A = nn.Parameter(A.contiguous())
        if new_parent is not None:
            layer.parent = nn.Parameter(new_parent.contiguous())
        if next_parent is not None:
            self.layers[i + 1].parent = nn.Parameter(next_parent.contiguous())
            if scale:
                self.layers[i + 1]._parent_boost = True
        if scale:
            layer._awakened = True
        self.config['concepts'][i] = int(g.shape[1])

    def split_one(self, layer_index, selected, noise=.06):
        """Split one concept per head in a single layer, preserving its mass and attention."""
        layer = self.layers[layer_index]
        prefix = f'layers.{layer_index}.'
        next_prefix = f'layers.{layer_index + 1}.'
        with torch.no_grad():
            raw = dict(layer.state_dict(prefix=prefix))
            if layer_index + 1 < len(self.layers) and self.layers[layer_index + 1].parent is not None:
                raw[next_prefix + 'parent'] = self.layers[layer_index + 1].parent.detach().clone()
            if layer.g.shape[1] == 1 and noise and getattr(layer, '_k1_stats', None) is not None:
                mean_q, log_std_q = layer._k1_stats
                raw[prefix+'g'] = mean_q.to(device=self.device, dtype=layer.g.dtype).unsqueeze(1).clone()
                raw[prefix+'log_s'] = log_std_q.to(device=self.device, dtype=layer.log_s.dtype).unsqueeze(1).clone()
            values = self._copy_split(raw, layer_index, selected)
            heads = torch.arange(layer.h, device=self.device)
            mass = torch.zeros_like(values[prefix+'log_c'])
            mass[heads, selected] = mass[:, -1] = math.log(2)
            values[prefix+'log_c'] -= mass
            if prefix + 'A' in values:
                values[prefix+'A'] -= mass[:, None, :]
            if noise:
                scale = values[prefix+'log_s'][heads, selected].exp()
                offset = torch.randn_like(values[prefix+'g'][heads, selected]) * scale * noise
                values[prefix+'g'][heads, selected] += offset
                values[prefix+'g'][:, -1] -= offset
                values[prefix+'log_s'][heads, selected] -= .15
                values[prefix+'log_s'][:, -1] -= .15
                if prefix + 'A' in values:
                    nudge = min(float(noise) * 4., .25)
                    values[prefix+'A'][heads, selected, selected] += nudge
                    values[prefix+'A'][heads, selected, -1] -= nudge
                    values[prefix+'A'][heads, -1, -1] += nudge
                    values[prefix+'A'][heads, -1, selected] -= nudge
        was_single = layer.g.shape[1] == 1
        for name in ('g', 'log_s', 'log_c'):
            setattr(layer, name, nn.Parameter(values[prefix+name].contiguous()))
        if layer.A is not None and prefix + 'A' in values:
            layer.A = nn.Parameter(values[prefix+'A'].contiguous())
        if layer.parent is not None and prefix + 'parent' in values:
            layer.parent = nn.Parameter(values[prefix + 'parent'].contiguous())
        if layer_index + 1 < len(self.layers) and next_prefix + 'parent' in values:
            self.layers[layer_index + 1].parent = nn.Parameter(values[next_prefix + 'parent'].contiguous())
        if was_single and noise:
            layer._awakened = True
        self.config['concepts'][layer_index] += 1

    def _copy_split(self, values, layer_index, selected):
        """Duplicate the chosen slices of both parameters and optimizer moments."""
        layer = self.layers[layer_index]
        prefix = f'layers.{layer_index}.'
        heads = torch.arange(layer.h, device=self.device)
        order = torch.arange(layer.g.shape[1], device=self.device).expand(layer.h, -1)
        order = torch.cat((order, selected[:, None]), dim=1)
        copied = dict(values)
        for name in ('g', 'log_s', 'log_c'):
            key = prefix+name
            if key in values:
                copied[key] = values[key][heads[:, None], order]
        key = prefix+'A'
        if key in values:
            copied[key] = values[key][heads[:, None, None], order[:, :, None], order[:, None, :]]
        pkey = prefix + 'parent'
        if pkey in values:
            prev_idx = torch.arange(values[pkey].shape[1], device=self.device)[None, :, None]
            copied[pkey] = values[pkey][heads[:, None, None], prev_idx, order[:, None, :]]
        nkey = f'layers.{layer_index + 1}.parent'
        if nkey in values:
            next_idx = torch.arange(values[nkey].shape[2], device=self.device)[None, None, :]
            copied[nkey] = values[nkey][heads[:, None, None], order[:, :, None], next_idx]
        return copied

    @torch.no_grad()
    def concepts_to_split(self, layer_index, docs):
        """Use the most occupied concept of each head on a spread of training documents."""
        layer = self.layers[layer_index]
        if layer.g.shape[1] == 1:
            if docs:
                sum_q = torch.zeros(layer.h, layer.d, device=self.device)
                sum_q2 = torch.zeros(layer.h, layer.d, device=self.device)
                count = 0
                for index in np.linspace(0, len(docs)-1, min(8, len(docs)), dtype=int):
                    if len(docs[index]) < 2:
                        continue
                    tokens, mask = _pad([docs[index][:-1]], self.device)
                    hidden = self.centre(tokens)
                    prev_ctx = None
                    for previous in self.layers[:layer_index]:
                        hidden, prev_ctx = previous(hidden, mask, prev_ctx=prev_ctx, return_context=True)
                    q = layer.query_state(hidden, prev_ctx).reshape(-1, layer.h, layer.d)
                    sum_q += q.sum(0)
                    sum_q2 += q.square().sum(0)
                    count += int(q.shape[0])
                if count > 1:
                    mean_q = sum_q / count
                    var_q = (sum_q2 / count - mean_q.square()).clamp_min(1e-4)
                    layer._k1_stats = (mean_q.detach(), (.5 * var_q.log()).clamp(-4., 3.).detach())
            return torch.zeros(layer.h, dtype=torch.long, device=self.device)

        usage = torch.zeros_like(layer.log_c)
        for index in np.linspace(0, len(docs)-1, min(8, len(docs)), dtype=int):
            tokens, mask = _pad([docs[index][:-1]], self.device)
            hidden = self.centre(tokens)
            prev_ctx = None
            for previous in self.layers[:layer_index]:
                hidden, prev_ctx = previous(hidden, mask, prev_ctx=prev_ctx, return_context=True)
            q = layer.query_state(hidden, prev_ctx)
            t_prev = prev_ctx[3] if (prev_ctx is not None and len(prev_ctx) == 4) else None
            logits = layer._logits(q, t_prev)
            usage += logits.softmax(-1)[0].sum(0)
        return usage.argmax(dim=-1)

    def add_layer(self):
        """A new sub-concept layer specialized from the layer above: it starts closed, and the gradient opens it."""
        dim, last = self.config['dim'], self.layers[-1]
        prev_k = int(last.g.shape[1])
        tabular = bool(self.config.get('tabular_hierarchy', True))
        pure = bool(self.config.get('pure_concepts', False)) and not tabular
        linear_concepts = bool(self.config.get('linear_concepts', True)) and tabular
        layer = ConceptAttention(dim, self.config['heads'], prev_k, self.config['window'], self.config['head_dim'],
                                 prev_concepts=prev_k, pure_concepts=pure, tabular=tabular,
                                 linear_concepts=linear_concepts,
                                 subconcept_top_m=int(self.config.get('subconcept_top_m', 4)),
                                 view_window=int(self.config.get('view_window', 24)),
                                 res_var_div=float(self.config.get('res_var_div', 1.))).to(self.device)
        with torch.no_grad():
            if tabular:
                # Residualized coordinates (q_prev - g_prev) / s_prev are already centred at 0 per parent concept.
                nn.init.normal_(layer.query.weight, std=.02)
                if linear_concepts:
                    nn.init.normal_(layer.key.weight, std=.02)
                elif layer.key.weight.shape == last.key.weight.shape:
                    layer.key.weight.copy_(last.key.weight)
                if layer.value.weight.shape == last.value.weight.shape:
                    layer.value.weight.copy_(last.value.weight)
                nn.init.normal_(layer.g, std=.25)
                layer.log_s.zero_()
                layer.log_c.copy_(last.log_c)
            else:
                if layer.query.weight.shape == last.query.weight.shape:
                    layer.query.weight.copy_(last.query.weight)
                if layer.value.weight.shape == last.value.weight.shape:
                    layer.value.weight.copy_(last.value.weight)
                if layer.g.shape == last.g.shape:
                    jitter = torch.randn_like(last.g) * .01 * (last.g.std() + 1e-6)
                    layer.g.copy_(last.g + jitter)
                    layer.log_s.copy_(last.log_s)
                    layer.log_c.copy_(last.log_c)
            nn.init.normal_(layer.out.weight, std=.02); layer.out.bias.zero_()
            if tabular:
                layer.out.weight[:, 2 * layer.h * layer.d:3 * layer.h * layer.d].normal_(std=.005)
            if layer.parent is not None and prev_k > 1:
                eye = torch.eye(prev_k, device=self.device).unsqueeze(0).expand(layer.h, -1, -1).clone()
                layer.parent.copy_((.5 if tabular else 1.) * eye)
            layer.gate.zero_()                                         # closed: the state goes through as it is
        layer._layer_boost = True
        self.layers.append(layer)
        self.config['layers'] = len(self.layers)
        self.config['concepts'].append(prev_k)

    def _like(self, **change):
        """A network with the configuration changed, its own parameters at their start (to be filled)."""
        config = dict(self.config, **change)
        groups = self.group.cpu().numpy() if self.grouped else None
        return Network(config, groups=groups).to(self.device)

    def widened(self, noise=2e-3):
        """Twice as wide (the state and every group of concepts): every coordinate is copied, so the states are the old
        ones written twice; the maps that read them take half from each copy, the dispersions grow by sqrt 2 and the
        masses of the concepts are corrected, so memberships, readings and the next token are the same."""
        grown = self._like(dim=2 * self.config['dim'], head_dim=2 * self.config['head_dim'],
                           cache=2 * (self.config.get('cache') or 0),
                           res_var_div=2. * float(self.config.get('res_var_div', 1.)))
        grown.load_state_dict(self._widen({k: v.detach().clone() for k, v in self.state_dict().items()}, noise=noise))
        return grown

    def _widen(self, values, moments=False, noise=0.):
        """The map of ``widened`` on a dictionary of tensors named as the parameters: the parameters themselves, or
        (``moments``) the statistics the optimizer keeps for them."""
        D, H, d = self.config['dim'], self.config['heads'], self.config['head_dim']
        linear_c = bool(self.config.get('linear_concepts', True)) and bool(self.config.get('tabular_hierarchy', True))
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
                elif part == 'key.weight':
                    v = half_in(heads_rows(v)) * (1. if moments else 1 / math.sqrt(2))
                elif part == 'write':
                    v = half_in(v)
                elif part == 'near':
                    v = torch.cat([v, v], -1) if v.ndim == 2 else torch.cat([torch.cat([half(v), half(v)], -1)] * 2, -2)
                elif part == 'out.weight':
                    if v.shape[1] == 3 * H * d + D:
                        blocks = [torch.cat([half(v[:, j * H * d:(j + 1) * H * d].view(D, H, d))] * 2, -1).reshape(D, 2 * H * d)
                                  for j in range(3)]
                        near = v[:, 3 * H * d:]
                        v = rows(torch.cat([*blocks, half_in(near)], -1))
                    else:
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

    def more_heads(self, noise=2e-3):
        """Twice the groups of concepts in every layer: each head is duplicated with half its output weight and small
        opposite perturbations so the layer reading is preserved at step zero while every head receives full non-zero
        gradients from the first batch."""
        grown = self._like(heads=2 * self.config['heads'])
        grown.load_state_dict(self._grow_heads({k: v.detach().clone() for k, v in self.state_dict().items()}, noise=noise))
        return grown

    def _grow_heads(self, values, moments=False, moments_power=1, noise=0.):
        """Duplicate head slices of parameters or optimizer moments (1x first moment, 4x second moment -> 0.5x step per copy)."""
        heads, width = self.config['heads'], self.config['head_dim']
        read_width = heads * width
        head_scale = (.5 ** moments_power) if moments else 1.
        out_scale = (4. if moments_power == 2 else 1.) if moments else .5
        def apart(W, scale_ref=None):
            if moments or not noise:
                return torch.cat((W * head_scale, W * head_scale), dim=0)
            amp = scale_ref if scale_ref is not None else (W.std() + 1e-6)
            e = torch.randn_like(W) * noise * amp
            return torch.cat((W + e, W - e), dim=0)
        changed = {}
        for key, value in values.items():
            if key.startswith('layers.'):
                part = key.split('.', 2)[2]
                if part == 'g':
                    log_s = values.get(key.replace('.g', '.log_s'))
                    ref = log_s.exp() if (log_s is not None and not moments) else None
                    value = apart(value, ref)
                elif part in ('query.weight', 'key.weight', 'value.weight'):
                    value = apart(value)
                elif part in ('log_s', 'log_c', 'A', 'parent', 'write', 'pos_bias'):
                    value = torch.cat((value * head_scale, value * head_scale), dim=0)
                elif part == 'out.weight':
                    if value.shape[1] == 3 * read_width + self.config['dim']:
                        parts = []
                        for j in range(3):
                            blk = value[:, j * read_width:(j + 1) * read_width]
                            if moments or not noise:
                                parts.append(torch.cat((blk * out_scale, blk * out_scale), dim=1))
                            else:
                                e = torch.randn_like(blk) * noise * (blk.std() + 1e-6)
                                parts.append(torch.cat((blk * .5 + e, blk * .5 - e), dim=1))
                        near = value[:, 3 * read_width:]
                        value = torch.cat((*parts, near), dim=1)
                    else:
                        read, near = value[:, :read_width], value[:, read_width:]
                        if moments or not noise:
                            read_new = torch.cat((read * out_scale, read * out_scale), dim=1)
                        else:
                            e = torch.randn_like(read) * noise * (read.std() + 1e-6)
                            read_new = torch.cat((read * .5 + e, read * .5 - e), dim=1)
                        value = torch.cat((read_new, near), dim=1)
            changed[key] = value
        return changed

    def wider_window(self):
        """Twice the distances among the nearby states; the maps of the new ones start at zero."""
        grown = self._like(window=2 * self.config['window'])
        new = {k: (torch.cat([v, torch.zeros_like(v)], 0) if k.endswith('.near') else v)
               for k, v in self.state_dict().items()}
        grown.load_state_dict(new)
        return grown

    def footprint(self):
        """Architecture of the live network without exporting or moving its weights."""
        named = tuple(self.named_parameters())
        layer_params = sum(w.numel() for name, w in named if name.startswith('layers.'))
        concept_params = sum(w.numel() for name, w in named
                             if name.startswith('layers.') and name.endswith(('.g', '.log_s', '.log_c', '.A', '.parent')))
        return dict(dim=int(self.config['dim']), layers=len(self.layers),
                    groups=[list(layer.g.shape[:2]) for layer in self.layers],
                    parameters=sum(w.numel() for _, w in named),
                    bytes=sum(w.numel() * w.element_size() for _, w in named),
                    layer_parameters=layer_params, layer_concept_parameters=concept_params)

    def export(self):
        """The parameters as numpy arrays, read by ConceptNet."""
        keep = {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}   # a copy: it must not change as the network learns on
        return ConceptNet(self.config, keep)

    @classmethod
    def restore(cls, net):
        """A network that goes on learning from a ConceptNet (the parameters saved in a checkpoint)."""
        groups = net.p['group'].astype(np.int64) if 'group' in net.p else None
        has_key = 'layers.0.key.weight' in net.p
        tabular = bool(net.config.get('tabular_hierarchy', has_key)) and has_key
        has_A = 'layers.0.A' in net.p
        linear_concepts = bool(net.config.get('linear_concepts', not has_A)) and tabular and not has_A
        pure_default = (not tabular and 'layers.0.out.weight' in net.p and 'layers.0.g' in net.p
                        and net.p['layers.0.out.weight'].shape[1] == net.p['layers.0.g'].shape[0] * net.p['layers.0.g'].shape[2])
        pure = bool(net.config.get('pure_concepts', pure_default)) and not tabular
        view_window = (int(net.p['layers.0.pos_bias'].shape[1]) if 'layers.0.pos_bias' in net.p
                       else int(net.config.get('view_window', 64)))
        config = dict(net.config, tabular_hierarchy=tabular, linear_concepts=linear_concepts,
                      pure_concepts=pure, view_window=view_window,
                      cache=net.config.get('cache', 0) if 'cache_query.weight' in net.p else 0)
        config.setdefault('sentinel', False)                   # older networks: the memory with its valve
        model = cls(config, groups=groups)
        state = {k: torch.as_tensor(v) for k, v in net.p.items()}
        for i in range(len(model.layers)):                             # older networks: every layer open
            state.setdefault(f'layers.{i}.gate', torch.ones(()))
            state.setdefault(f'layers.{i}.write', torch.zeros_like(model.layers[i].write))
            if model.layers[i].parent is not None:
                state.setdefault(f'layers.{i}.parent', torch.zeros_like(model.layers[i].parent))
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


def lexical_concepts(sample, vocabulary, dim, eos=None):
    """Discover distributional coordinates and semantic lexical sub-concepts of the vocabulary using ACAN's Fisher-EM
    algebra on context co-occurrences."""
    sample = np.asarray(sample, np.int64)
    counts = np.bincount(sample, minlength=vocabulary).astype(np.float64)
    frequency = counts + 1.
    if len(sample) < 64 or vocabulary < 16:
        return None, frequency_groups(frequency)
    order = np.argsort(-counts, kind='stable')
    anchors = [int(t) for t in order if t != eos and counts[t] > 0][:min(32, vocabulary)]
    if len(anchors) < 4:
        return None, frequency_groups(frequency)
    k_anc = len(anchors)
    rank = np.full(vocabulary, -1, np.int32)
    rank[anchors] = np.arange(k_anc, dtype=np.int32)
    width = 2 * k_anc
    cooc = np.zeros((vocabulary, width), np.float64)
    left_r = rank[sample[:-1]]
    sel_l = left_r >= 0
    if np.any(sel_l):
        idx_l = sample[1:][sel_l] * width + left_r[sel_l]
        cooc += np.bincount(idx_l, minlength=vocabulary * width).reshape(vocabulary, width)
    right_r = rank[sample[1:]]
    sel_r = right_r >= 0
    if np.any(sel_r):
        idx_r = sample[:-1][sel_r] * width + (k_anc + right_r[sel_r])
        cooc += np.bincount(idx_r, minlength=vocabulary * width).reshape(vocabulary, width)
    total = max(cooc.sum(), 1.)
    ppmi = np.maximum(np.log((cooc * total) / ((cooc.sum(1, keepdims=True) + 1e-9) * (cooc.sum(0, keepdims=True) + 1e-9)) + 1e-9), 0.)
    ppmi /= np.maximum(np.linalg.norm(ppmi, axis=1, keepdims=True), 1e-6)
    _, evecs = np.linalg.eigh(ppmi.T @ ppmi)
    coord_dim = min(int(dim), width)
    coords = ppmi @ evecs[:, ::-1][:, :coord_dim]
    em_dim = min(coord_dim, 16)
    x = coords[:, :em_dim]
    seen = counts > 0
    seen_idx = np.flatnonzero(seen)
    if len(seen_idx) < 4:
        return coords.astype(np.float32), frequency_groups(frequency)
    n_clusters = max(2, min(128, int(round(math.sqrt(vocabulary))), len(seen_idx) // 2))
    x_seen = x[seen_idx]
    rng = np.random.default_rng(0)
    picks = np.linspace(0, len(seen_idx) - 1, n_clusters, dtype=int)
    g = x_seen[picks].copy() + rng.normal(0., 1e-3, size=(n_clusters, em_dim))
    root_v = np.maximum(x_seen.var(axis=0, keepdims=True), 1e-3)
    log_s = .5 * np.log(np.repeat(root_v, n_clusters, axis=0))
    log_c = np.full(n_clusters, -math.log(n_clusters))
    token_w = np.log1p(counts[seen_idx])
    for _ in range(6):
        inv_v = np.exp(-2. * log_s)
        quad = (x_seen * x_seen) @ inv_v.T - 2. * x_seen @ (g * inv_v).T + (g * g * inv_v).sum(-1)[None, :]
        logits = log_c[None, :] - .5 * quad - log_s.sum(-1)[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        resp = np.exp(logits)
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-12)
        w_resp = resp * token_w[:, None]
        bc = w_resp.sum(0)
        bu = w_resp.T @ x_seen
        bq = w_resp.T @ (x_seen * x_seen)
        log_c, g, log_s = natural_em_parameters(log_c, g, log_s, bc, bu, bq, root_v, 1e-2, 1.0, np)
        log_s = np.clip(log_s, -4., 3.)
    inv_v = np.exp(-2. * log_s)
    quad_all = (x_seen * x_seen) @ inv_v.T - 2. * x_seen @ (g * inv_v).T + (g * g * inv_v).sum(-1)[None, :]
    logits_all = log_c[None, :] - .5 * quad_all - log_s.sum(-1)[None, :]
    classes = np.full(vocabulary, n_clusters, dtype=np.int64)
    classes[seen_idx] = logits_all.argmax(axis=1)
    limit = max(16, int(math.ceil(1.25 * math.sqrt(vocabulary))))
    return coords.astype(np.float32), output_groups(classes, frequency, limit=limit)


def documents(ids, eos, longest, begin=False):
    """The documents of a stream (each ends with ``eos``), in pieces of at most ``longest`` tokens. With ``begin``
    every document starts with ``eos`` too, so that its first token is read after the end of the one before."""
    ids = np.asarray(ids, np.int64)
    docs = np.split(ids, np.flatnonzero(ids == eos) + 1)
    if begin:
        docs = [np.r_[eos, doc] for doc in docs if len(doc)]
    return [doc[i:i + longest] for doc in docs if len(doc) > 2 for i in range(0, len(doc) - 1, longest - 1)]


def held_out(network, docs, limit=None, batch=4096, weight=None):
    """log p of the true token at every position of held-out documents (arrays of ids), in their order: the network
    reads them in batches, as it learns, without the gradient. ``limit``: only the first documents, about that many
    tokens. When ``weight`` is given, returns one weighted log-likelihood per document."""
    network.eval()
    if network.device.type == 'cuda':
        batch = min(batch, 256)
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
    w_arr = None if weight is None else np.asarray(weight, np.float64)
    read = [None] * len(docs)
    with torch.no_grad():
        for chunk in chunks:
            ids, mask = _pad([docs[j] for j in chunk], network.device)
            lp = network.target_log_probs(ids[:, :-1], mask[:, :-1], ids[:, 1:])
            m = mask[:, 1:].bool()
            for r, j in enumerate(chunk):
                vals = lp[r][m[r]].cpu().numpy()
                if w_arr is None:
                    read[j] = vals
                else:
                    ws = w_arr[ids[r, 1:][m[r]].cpu().numpy()]
                    tw = ws.sum()
                    read[j] = np.array([(vals * ws).sum() / tw]) if tw > 0 else np.zeros(0)
    return np.concatenate(read) if read else np.zeros(0)


def batches(docs, tokens, shuffle, rng, device=None):
    order = rng.permutation(len(docs)) if shuffle else np.arange(len(docs))
    chunk, size = [], 0
    longest = 0
    for j in order:
        length = len(docs[j])
        if chunk and device is not None and device.type == 'cuda' and (len(chunk) + 1) * max(longest, length) > tokens:
            yield _pad(chunk, device)
            chunk, size, longest = [], 0, 0
        longest = max(longest, length)
        chunk.append(docs[j]); size += length
        if size >= tokens:
            yield _pad(chunk, device); chunk, size, longest = [], 0, 0
    if chunk:
        yield _pad(chunk, device)


def _pad(chunk, device=None):
    n = max(len(c) for c in chunk)
    ids = np.zeros((len(chunk), n), np.int64); mask = np.zeros((len(chunk), n), np.float32)
    for r, c in enumerate(chunk):
        ids[r, :len(c)] = c; mask[r, :len(c)] = 1
    return torch.as_tensor(ids, device=device), torch.as_tensor(mask, device=device)


class Learner:
    """The network and its optimizer across the pieces of a queue (the state is kept between pieces)."""

    _DECAYED = ('query.weight', 'key.weight', 'value.weight', 'out.weight', '.near',
                'final.weight', 'cache_query.weight', 'cache_key.weight')

    def __init__(self, network, lr=3e-3, tokens=4096, longest=512, seed=0, weight=None):
        """``weight``: how much every token weighs when it is the one to read (one per token of the vocabulary),
        as the concept model weighs its attributes: a target as much as all the features together."""
        self.network, self.tokens, self.longest = network, tokens, longest
        self.weight = None if weight is None else torch.as_tensor(np.asarray(weight, np.float32), device=network.device)
        gates = [p for name, p in network.named_parameters() if name.endswith('.gate')]
        seen_special = {id(p) for p in gates}
        awakened, grown_params = [], []
        def add_unique(target, params):
            for p in params:
                if id(p) not in seen_special:
                    seen_special.add(id(p))
                    target.append(p)
        for layer in network.layers:
            was_awakened = getattr(layer, '_awakened', False)
            was_parent = getattr(layer, '_parent_boost', False)
            was_layer = getattr(layer, '_layer_boost', False)
            layer._awakened = layer._parent_boost = layer._layer_boost = False
            if was_awakened:
                cand = (layer.query.weight, layer.g, layer.log_s, layer.log_c, layer.A, layer.parent)
                add_unique(awakened, [p for p in cand if p is not None])
            elif was_parent and layer.parent is not None:
                add_unique(awakened, [layer.parent])
            if was_layer:
                add_unique(grown_params, list(layer.parameters()))
        regular = [p for p in network.parameters() if id(p) not in seen_special]
        groups = [{'params': regular, 'lr': lr, 'scale': 1.}]
        if awakened:
            groups.append({'params': awakened, 'lr': lr * 4., 'scale': 4., 'boost': True})
        if grown_params:
            groups.append({'params': grown_params, 'lr': lr * 2., 'scale': 2., 'boost': True})
        if gates:
            groups.append({'params': gates, 'lr': lr * 8., 'scale': 8.})
        self.opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.,
                                     foreach=(network.device.type == 'cpu'))
        self.decay = .01
        self.anchor = None                                     # where the parameters are pulled back to (None: projections only)
        self.lr, self.steps = lr, 0
        self._boost_steps = 0
        self.horizon = 2000.                                   # Robbins-Monro online EM step decay horizon (in steps)
        self.em_scale = .5                                     # online natural-EM rate matched to gradient step scale
        self.seen = 0                                          # tokens read so far, batch after batch
        self.warmup = 200                                      # steps of the start, at most one pass (Grower)
        self.rng = np.random.default_rng(seed)

    def _step_lr(self, warmup):
        warm = min(1., (self.steps + 1) / max(warmup, 1))
        cool = max(.2, 1. / math.sqrt(1. + max(0, self.steps - warmup) / self.horizon))
        return self.lr * warm * cool

    def learn(self, ids, eos, check=None, warmup=None):
        """One pass over a piece of text, in batches of about ``tokens`` tokens. Returns the training loss."""
        return self.learn_documents(documents(ids, eos, self.longest), check, warmup)

    def learn_documents(self, docs, check=None, warmup=None):
        warmup = self.warmup if warmup is None else warmup
        self.network.train()
        total, n = 0., 0
        use_foreach = self.network.device.type == 'cpu'
        for ids_, mask in batches(docs, self.tokens, True, self.rng, self.network.device):
            if check is not None:
                check()
            step_lr = self._step_lr(warmup)
            anneal = 1. / (1. + getattr(self, '_boost_steps', 0) / 48.)
            for group in self.opt.param_groups:
                if group.get('boost'):
                    warm_lr = max(step_lr * group.get('scale', 2.), min(self.lr, 3e-3))
                    group['lr'] = step_lr + anneal * (warm_lr - step_lr)
                else:
                    group['lr'] = step_lr * group.get('scale', 1.)
            lp = self.network.target_log_probs(ids_[:, :-1], mask[:, :-1], ids_[:, 1:])
            m = mask[:, 1:] if self.weight is None else mask[:, 1:] * self.weight[ids_[:, 1:]]
            denom = m.sum()
            if float(denom) <= 0.:
                continue
            loss = -(lp * m).sum() / denom
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1., foreach=use_foreach)
            with torch.no_grad():
                for p, state in self.opt.state.items():
                    if p.grad is not None and state and 'exp_avg_sq' in state:
                        step_val = int(state['step'].item()) if torch.is_tensor(state['step']) else int(state['step'])
                        if step_val > 0:
                            sq = state['exp_avg_sq']
                            zero_mask = (sq == 0) & (p.grad != 0)
                            if bool(zero_mask.any()):
                                sq[zero_mask] = p.grad[zero_mask].square()
            self.opt.step(); self.steps += 1
            self._boost_steps = getattr(self, '_boost_steps', 0) + 1
            with torch.no_grad():
                for layer in self.network.layers:
                    layer.gate.clamp_(-1.5, 1.5)
                    if layer.parent is not None:
                        layer.parent.clamp_(-8., 8.)
            self._concept_em_step(step_lr)
            self._pull_back(step_lr)
            total += loss.item() * m.sum().item(); n += m.sum().item()
            self.seen += int(mask[:, 1:].sum())
        self.network.eval()
        return total / max(n, 1)

    def _concept_em_step(self, lr):
        """Apply the shared Fisher natural-gradient / online-EM update to the Gaussian concepts of every layer."""
        if self.anchor is not None or self.em_scale <= 0.:
            return
        rate = min(.01, lr * self.em_scale)
        with torch.no_grad():
            for layer in self.network.layers:
                stats = getattr(layer, '_em_stats', None)
                layer._em_stats = None
                if stats is None or layer.g.shape[1] <= 1:
                    continue
                bc, bu, bq, root_v = stats
                if float(bc.sum()) <= 1e-6:
                    continue
                new_log_c, new_g, new_log_s = natural_em_parameters(
                    layer.log_c, layer.g, layer.log_s, bc, bu, bq, root_v, 1e-2, rate, torch
                )
                layer.log_c.copy_(new_log_c)
                layer.g.copy_(new_g)
                layer.log_s.copy_(new_log_s.clamp(-4., 3.))

    def carry(self, before, kind, split=None):
        """The optimizer of a grown copy goes on from the one of the network it grew from (``before``, a Learner):
        a fresh optimizer moves wildly at first, and a copy starting with one would lose a race for that alone. Every
        parameter that is there before takes its statistics, mapped as the parameter was; a brand-new parameter tensor
        starts cleanly at step 0 with full bias correction."""
        old = {name: before.opt.state.get(p) for name, p in before.network.named_parameters()}
        steps = next((state['step'] for state in old.values() if state), None)
        if steps is None:
            return
        maps = {}
        for moment in ('exp_avg', 'exp_avg_sq'):
            power = 1 if moment == 'exp_avg' else 2
            values = {name: state[moment] for name, state in old.items() if state}
            if kind == 'larghezza':
                values = before.network._widen(values, moments=True)
            elif kind == 'teste':
                values = before.network._grow_heads(values, moments=True, moments_power=power)
            elif isinstance(split, tuple):
                values = before.network._copy_split(values, *split)
            elif kind == 'concetti':
                m_scale, a_scale = .5 ** power, .25 ** power
                def dup_k(v, interleaved):
                    if interleaved:
                        return torch.stack([v * m_scale, v * m_scale], dim=2).reshape(v.shape[0], 2 * v.shape[1], *v.shape[2:])
                    return torch.cat([v * m_scale] * 2, 1)
                if isinstance(split, int):
                    prefix = f'layers.{split}.'
                    interleaved = f'layers.{split}.A' not in old
                    next_parent = f'layers.{split + 1}.parent'
                    for name in list(values):
                        if name == next_parent:
                            values[name] = torch.cat([values[name] * m_scale] * 2, 1)
                        elif name.startswith(prefix):
                            if name.endswith(('.g', '.log_s', '.log_c')):
                                values[name] = dup_k(values[name], interleaved)
                            elif name.endswith('.parent'):
                                values[name] = torch.cat([values[name] * m_scale] * 2, 2)
                            elif name.endswith('.A'):
                                values[name] = torch.cat([torch.cat([values[name] * a_scale] * 2, 2)] * 2, 1)
                else:
                    for name in list(values):
                        if not name.startswith('layers.'):
                            continue
                        idx = int(name.split('.', 2)[1])
                        interleaved = f'layers.{idx}.A' not in old
                        if name.endswith(('.g', '.log_s', '.log_c')):
                            values[name] = dup_k(values[name], interleaved)
                        elif name.endswith(('.A', '.parent')):
                            values[name] = torch.cat([torch.cat([values[name] * a_scale] * 2, 2)] * 2, 1)
            maps[moment] = values
        for name, p in self.network.named_parameters():
            v_sq = maps['exp_avg_sq'].get(name)
            if v_sq is None or v_sq.shape != p.shape or float(v_sq.max()) <= 0.:
                continue
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
                # Only regularize linear projections; never pull priors, dispersions, norms, gates or K=1 queries/keys toward zero.
                single_k = {f'layers.{i}.{part}' for i, layer in enumerate(self.network.layers)
                            if layer.g.shape[1] <= 1 for part in ('query.weight', 'key.weight')}
                for name, p in self.network.named_parameters():
                    if name.endswith(self._DECAYED) and name not in single_k:
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
            best.anchor = None
            results.append((kind, best, best_scores))
        self.learner.anchor = None
        winner = results[0]
        for result in results[1:]:
            if self._better(result[2], winner[2]):
                winner = result
        kind, self.learner, _ = winner
        self.learner.anchor = None
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
        # as many batches in a pass as tokens in a batch on small pieces; full batches on large pieces
        self.piece = sum(map(len, docs))                       # a trial is at most one pass over the piece
        self.learner.tokens = min(self.batch, max(1, self.piece // 64) if self.piece >= 16384 else max(1, math.isqrt(self.piece)))
        if not self.learned:                                   # the start lasts at most one pass over the piece
            self.learner.warmup = min(self.learner.warmup, max(1, -(-self.piece // self.learner.tokens)))
        if not self.learned:
            self.decision = self.learner.tokens
        while self.decision <= self.learned:                   # tokens learned elsewhere (a race of growth): the
            self.decision *= 2                                 # calendar goes on from them, it does not decide at once
        if not self.KINDS and not self.passes:
            # Fixed architecture on a fresh single-pass piece (NetText non-race step): learn the whole piece cleanly
            # without anchoring weights to a sub-stretch or prematurely stopping the queue.
            self.learner.anchor = None
            self.learner.learn_documents(docs, check)
            self.learned += self.piece
            while self.decision <= self.learned:
                self.decision *= 2
        else:
            while cycle.fresh and not self.done:               # the new text, on the calendar of the tokens learned
                stretch, size = cycle.take(self.decision - self.learned, within_pass=True)
                if stretch:
                    self.learner.learn_documents(stretch, check); self.learned += size
                if self.learned < self.decision or not cycle.fresh or cycle.i == len(cycle.order):
                    break                                      # the new text is over before the decision
                trial, size = cycle.take(self._trial_tokens(), within_pass=True)
                self.learned += size; decided = True
                # it no longer learns and growing does not help: confirmed by a second decision, on twice the text
                self.misses = 0 if self._decide(trial, score, check) else self.misses + 1
                self.done = self.misses >= 2
                self.decision *= 2
            if self.done:
                return self.best
        if not decided:                                        # same architecture: keep any held-out improvement
            scores = score(self.learner.network)
            if self.best_scores is None or scores.mean() > self.best_scores.mean():
                self._keep(self.learner.network.export(), scores)
            elif self.best_learner is not None and self._better(self.best_scores, scores):
                self.learner = copy.deepcopy(self.best_learner)
                self.learner.lr = max(1e-4, self.learner.lr * .5)
            elif not self.passes:
                return self.learner.network.export()
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
