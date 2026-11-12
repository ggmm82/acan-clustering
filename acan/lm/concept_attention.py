"""Attention as the cp-v9 concept model (acan/concept_algebra.py).

The prefix is a table: each past token is a row with key attributes and a value
to be read. The query is a row whose value is unknown. Membership is

    T_i(q)  ∝  C_i · E_{k ~ concept i}[exp(scale · q·k)]

* The last ``window`` tokens (plus the current block) are singleton concepts:
  for them the expectation is exp(scale · q·k), i.e. dense attention.
* Older tokens are generalized bottom-up, exactly as table rows are: each
  enters as its own concept, and above the ``slots`` budget the pair with the
  smallest increase of description length is added together.
* Keys are summarized before the positional rotation; a concept is rotated to
  the mean position of its members when it is read.

With window >= length, or slots >= length, the layer equals per-head dense
softmax attention with the copied projections. Statistics only ever contain
earlier tokens, so no token supports its own prediction. Absorption happens on
a fixed grid of absolute positions: full-prefix and incremental decoding agree.
"""
import math

import torch

from acan.concept_algebra import centre_variance, description_length, dot_similarity
from .attention import ACANConceptAttentionLayer

PROFILE = 'cp-v9'
STATISTICS = ('C', 'U', 'Q', 'V', 'P')


def rotate(keys, positions, inverse_frequency):
    """Rotary position applied at (possibly fractional) positions; matches the library at integers."""
    angles = positions[..., None]*inverse_frequency
    angles = torch.cat((angles, angles), -1)
    half = keys.shape[-1]//2
    return keys*angles.cos()+torch.cat((-keys[..., half:], keys[..., :half]), -1)*angles.sin()


def merge_costs(state, kappa):
    """Increase of description length for every pair of concepts (batch x heads x n x n)."""
    root = (state['root_q']/state['root_n']-(state['root_u']/state['root_n'])**2).clamp_min(1e-8)[:, :, None]
    c = state['C'][..., None]
    own = description_length(c, centre_variance(c, state['U'], state['Q'], root, kappa, torch)[1], state['C'], torch)
    pair_c = state['C'][:, :, :, None]+state['C'][:, :, None]
    pair_variance = centre_variance(pair_c[..., None], state['U'][:, :, :, None]+state['U'][:, :, None],
                                    state['Q'][:, :, :, None]+state['Q'][:, :, None], root[:, :, None], kappa, torch)[1]
    together = description_length(pair_c[..., None], pair_variance, pair_c, torch)
    return together-own[:, :, :, None]-own[:, :, None]


def generalize(state, budget, kappa):
    """Add together the cheapest pair, in every batch element and head, until the budget holds."""
    while state['C'].shape[2] > budget:
        n = state['C'].shape[2]
        with torch.no_grad():                                   # the structure is discrete; contents stay differentiable
            parts = [merge_costs({key: value[i:i+1] for key, value in state.items() if key != 'root_n'} | {'root_n': state['root_n']}, kappa)
                     for i in range(state['C'].shape[0])]
            cost = torch.cat(parts).masked_fill(torch.ones(n, n, dtype=torch.bool, device=state['C'].device).tril(), torch.inf)
            flat = cost.flatten(2).argmin(-1)
            first, second = flat//n, flat % n
            kept = torch.arange(n-1, device=flat.device)
            kept = kept+(kept >= second[..., None])
        for name in STATISTICS:
            value = state[name]
            tail = value.shape[3:]
            index = lambda where: where.reshape(*where.shape, *([1]*len(tail))).expand(*where.shape, *tail)
            value = value.scatter_add(2, index(first[..., None]), value.gather(2, index(second[..., None])))
            state[name] = value.gather(2, index(kept))
    return state


class ConceptSlotAttention(ACANConceptAttentionLayer):
    """Same construction and graft boundary as the base layer; only ``attend`` differs."""
    block = 16
    kappa = 1.

    def attend(self, h, position_embeddings=None, use_cache=False):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        b, length, d = h.shape
        heads = d//self.head_dim
        previous = self.cache if use_cache else None
        past = previous['seen'] if previous else 0
        split = lambda value: value.view(b, length, heads, self.head_dim).transpose(1, 2)
        q, raw, value = split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))
        key = raw
        if position_embeddings is not None:
            q, key = apply_rotary_pos_emb(q, raw, *position_embeddings)
        q, key, raw, value = q.float(), key.float(), raw.float(), value.float()
        frequency = getattr(self, 'rope_frequency', None)        # the model's own rotary frequencies
        if position_embeddings is None:
            frequency = torch.zeros(self.head_dim//2, device=h.device)
        elif frequency is None:
            frequency = 1./10000.**(torch.arange(0, self.head_dim, 2, dtype=torch.float32)/self.head_dim)
        frequency = frequency.to(device=h.device, dtype=torch.float32)
        scale = self.head_dim**-.5
        # Buffers hold every token not yet absorbed; ``low`` is the absolute position of their first entry.
        low = previous['low'] if previous else 0
        state = previous['state'] if previous else None
        if previous:
            key, raw_all, value_all = (torch.cat((previous[name], current), 2)
                                       for name, current in (('keys', key), ('raw', raw), ('values', value)))
        else:
            raw_all, value_all = raw, value
        outputs, slot_weight = [], None
        position = past
        while position < past+length:
            start = self.block*(position//self.block)
            end = min(start+self.block, past+length)
            target = max(0, start-self.window)
            if target > low:                                    # generalize the tokens leaving the exact window
                part = slice(0, target-low)
                state = self._absorb(state, raw_all[:, :, part], value_all[:, :, part],
                                     torch.arange(low, target, device=h.device, dtype=torch.float32))
                key, raw_all, value_all = key[:, :, target-low:], raw_all[:, :, target-low:], value_all[:, :, target-low:]
                low = target
            queries = q[:, :, position-past:end-past]
            exact = slice(0, end-low)
            logits = queries@key[:, :, exact].transpose(-1, -2)*scale
            mask = (torch.arange(low, end, device=h.device)[None] > torch.arange(position, end, device=h.device)[:, None])
            logits = logits.masked_fill(mask, -torch.inf)
            values = value_all[:, :, exact]
            if state is not None:
                centre = rotate(state['U']/state['C'][..., None], state['P']/state['C'], frequency)
                concept = dot_similarity(queries[:, :, :, None], centre[:, :, None], scale, torch)+state['C'].log()[:, :, None]
                logits = torch.cat((logits, concept), -1)
                values = torch.cat((values, state['V']/state['C'][..., None]), 2)
            weights = logits.softmax(-1)
            outputs.append(weights@values)
            if state is not None:
                slot_weight = weights[:, :, -1, end-low:]
            position = end
        output = self.o_proj(torch.cat(outputs, 2).transpose(1, 2).reshape(b, length, d).to(h.dtype))
        if use_cache:
            detach = lambda item: item.detach().clone()
            self.cache = dict(seen=past+length, low=low, keys=detach(key), raw=detach(raw_all), values=detach(value_all),
                              state=None if state is None else {name: (detach(item) if torch.is_tensor(item) else item)
                                                                for name, item in state.items()})
        self.last_info = self._info(state, slot_weight, b, d, h.device)
        return output

    def _absorb(self, state, keys, values, positions):
        b, heads, count, _ = keys.shape
        new = dict(C=keys.new_ones(b, heads, count), U=keys, Q=keys*keys, V=values,
                   P=positions.expand(b, heads, count))
        if state is None:
            state = dict(new, root_n=float(count), root_u=keys.sum(2), root_q=(keys*keys).sum(2))
        else:
            state = dict({name: torch.cat((state[name], new[name]), 2) for name in STATISTICS},
                         root_n=state['root_n']+count, root_u=state['root_u']+keys.sum(2),
                         root_q=state['root_q']+(keys*keys).sum(2))
        return generalize(state, self.k, self.kappa)

    def _info(self, state, slot_weight, b, d, device):
        """Inspection in the shape the workbench expects: masses, last-token memberships, relevance per channel."""
        masses = torch.zeros(b, self.k, device=device)
        assignments = torch.zeros(b, self.k, device=device)
        relevance = torch.zeros(b, self.k, d, device=device)
        active = torch.zeros(b, dtype=torch.long, device=device)
        if state is not None:
            n = state['C'].shape[2]
            root = (state['root_q']/state['root_n']-(state['root_u']/state['root_n'])**2).clamp_min(1e-8)[:, :, None]
            within = centre_variance(state['C'][..., None], state['U'], state['Q'], root, 0., torch)[1]
            masses[:, :n] = state['C'].detach().mean(1)
            relevance[:, :n] = (1-within/root).clamp(0, 1).detach().permute(0, 2, 1, 3).reshape(b, n, d)
            active[:] = n
            if slot_weight is not None:
                assignments[:, :n] = slot_weight.detach().mean(1)
        return dict(active_slots=active, relevance=relevance, masses=masses, assignments=assignments)
