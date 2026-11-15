"""Concept attention without loops: every concept of every reader from prefix sums, in one pass.

The concepts are those of the concept graph (acan/core/concepts.py, acan/lm/concept_attention.py):
nodes holding only additive statistics of the observations they contain,

    C = sum T,   U = sum T z,   Q = sum T z^2,   Y = sum T y,   P = sum T position,

read with mass times expected similarity, log C + E[q.z]. What changes is how observations are
assigned to concepts. The present layer merges, one at a time, the pair cheapest for the reader:
a greedy search whose every step depends on the previous one. Here the assignment is a formula,
known before reading, so that every statistic of every concept for every position is a difference
of prefix sums, computed at once:

* by age, as levels: the older tokens of a reader at position t, those before its exact window,
  are covered by the dyadic segments of the binary writing of their number n = t - window + 1
  (Fenwick decomposition). The most recent old tokens are in the smallest segments, the oldest in
  the largest: fine resolution near, coarse far, with at most log2(n) + 1 segments. The levels of
  the tabular graph (general concepts above, particular below) here are the sizes of the segments;
* by content, inside each segment: a key goes to one of 2^r bins by the signs of r fixed
  projections (a hash fixed by the seed, no parameter), so tokens with different keys stay apart.

A reader sees its last ``window`` tokens exactly and at most (log2(T) + 1) * 2^r concepts. Training
reads all positions in parallel; generation can keep the segment sums as a binary counter (two
equal segments add up when a new one arrives), so its memory grows with log T, not with T.

Similarity: with ``variance`` the reading uses the second-order term of the expected similarity,
as the tabular model does: log E[exp(s q.z)] ~ s q.g + s^2 q^2 . var / 2, the variance from Q.
The observation z of a token is its key with its rotary position, so every statistic is a sum.
"""
import math

import torch


def rotate(keys, positions, inverse_frequency):
    angles = positions[..., None]*inverse_frequency
    angles = torch.cat((angles, angles), -1)
    half = keys.shape[-1]//2
    return keys*angles.cos()+torch.cat((-keys[..., half:], keys[..., :half]), -1)*angles.sin()


def fenwick_segments(length, window, device):
    """For every reader t: the start and end of its old segments, one per level, and which exist.

    The old tokens of t are 0 .. n-1 with n = t - window + 1. Level j holds [s, s + 2^j) when bit j
    of n is set, with s = n with its bits up to j cleared.
    """
    levels = max(1, math.ceil(math.log2(max(length, 2))) + 1)
    n = (torch.arange(length, device=device) - window + 1).clamp(min=0)              # old tokens of each reader
    j = torch.arange(levels, device=device)
    present = ((n[:, None] >> j) & 1).bool()
    start = (n[:, None] >> (j+1)) << (j+1)
    end = start + (1 << j)
    return start.clamp(max=length), end.clamp(max=length), present


def concept_scan(q, raw, v, inverse_frequency, window, directions, scale, variance=True):
    """Attention output (b, heads, T, d) of every position over its window and its concepts.

    q: queries with rotary positions; raw: keys before rotary positions; v: values. ``directions``
    (d, r) are the fixed projections of the content hash. A concept sums the keys with their
    positions (rotated keys), so its centre and variance are plain sums: high-frequency components
    of far segments average away, low-frequency ones (coarse position) remain.
    """
    b, heads, length, d = q.shape
    device, dtype = q.device, q.dtype
    keys = rotate(raw, torch.arange(length, device=device, dtype=dtype), inverse_frequency)
    # Exact window, as a band: every position reads the ``window`` tokens ending at itself.
    pad = lambda x: torch.nn.functional.pad(x, (0, 0, window-1, 0))
    band_keys = pad(keys).unfold(2, window, 1)                                      # (b, h, T, d, window)
    band_values = pad(v).unfold(2, window, 1)
    band_logits = scale*torch.einsum('bhtd,bhtdw->bhtw', q, band_keys)
    exists = (torch.arange(length, device=device)[:, None] - window + 1 + torch.arange(window, device=device)) >= 0
    band_logits = band_logits.masked_fill(~exists, -torch.inf)
    # Content bins by fixed signs of the keys before positions; one-hot assignment of every token.
    bins = 2**directions.shape[1]
    code = (((raw @ directions) > 0).long()*(2**torch.arange(directions.shape[1], device=device))).sum(-1)
    assign = torch.nn.functional.one_hot(code, bins).to(dtype)                      # (b, h, T, bins)
    start, end, present = fenwick_segments(length, window, device)                  # (T, levels)
    levels = start.shape[1]

    def segment(x):
        """Sum of x (b, h, T, bins[, d]) over every (reader, level) segment: prefix[end] - prefix[start]."""
        table = torch.nn.functional.pad(x.cumsum(2), (0,)*(2*(x.dim()-3)) + (1, 0))
        pick = lambda index: table.index_select(2, index.reshape(-1)).reshape(b, heads, length, levels, *x.shape[3:])
        return pick(end) - pick(start)
    count = segment(assign)*present[..., None]                                      # (b, h, T, levels, bins)
    alive = count > .5
    safe = count.clamp(min=1.)
    centre = segment(assign[..., None]*keys[..., None, :])/safe[..., None]          # (b, h, T, levels, bins, d)
    logits = scale*torch.einsum('bhtd,bhtlkd->bhtlk', q, centre) + safe.log()
    if variance:                                                                    # log E[exp(s q.z)] to second order
        spread = (segment(assign[..., None]*(keys*keys)[..., None, :])/safe[..., None] - centre*centre).clamp(min=0.)
        logits = logits + .5*scale*scale*torch.einsum('bhtd,bhtlkd->bhtlk', q*q, spread)
    logits = logits.masked_fill(~alive, -torch.inf).reshape(b, heads, length, levels*bins)
    weights = torch.cat((band_logits, logits), -1).softmax(-1)
    mean_value = (segment(assign[..., None]*v[..., None, :])/safe[..., None]).reshape(b, heads, length, levels*bins, d)
    return (torch.einsum('bhtw,bhtdw->bhtd', weights[..., :window], band_values)
            + torch.einsum('bhtc,bhtcd->bhtd', weights[..., window:], mean_value))


def concept_scan_blocks(q, raw, v, inverse_frequency, window, directions, scale, variance=True):
    """The same reading as ``concept_scan``, organised by shared segments: no per-reader copies.

    At level j the readers come in blocks of 2^(j+1) consecutive old-token counts n; the second half
    of each block (bit j of n set) reads the same segment, the first 2^j old tokens of the block.
    So a level is one batched matrix product between the queries of a block and the bins of its
    segment, the way ordinary attention is a product between queries and keys. The levels are a
    fixed structure of log2(T) + 1 steps, like the layers of a network, not a search.
    """
    b, heads, length, d = q.shape
    device, dtype = q.device, q.dtype
    keys = rotate(raw, torch.arange(length, device=device, dtype=dtype), inverse_frequency)
    pad = lambda x: torch.nn.functional.pad(x, (0, 0, window-1, 0))
    band_keys, band_values = pad(keys).unfold(2, window, 1), pad(v).unfold(2, window, 1)   # (b, h, T, d, window)
    band_logits = scale*torch.einsum('bhtd,bhtdw->bhtw', q, band_keys)
    exists = (torch.arange(length, device=device)[:, None] - window + 1 + torch.arange(window, device=device)) >= 0
    band_logits = band_logits.masked_fill(~exists, -torch.inf)
    readers = length - window + 1                                                   # reader t has n = t - window + 1 old tokens
    if readers <= 1:
        weights = band_logits.softmax(-1)
        return torch.einsum('bhtw,bhtdw->bhtd', weights, band_values)
    bins = 2**directions.shape[1]
    code = (((raw @ directions) > 0).long()*(2**torch.arange(directions.shape[1], device=device))).sum(-1)
    assign = torch.nn.functional.one_hot(code, bins).to(dtype)                      # (b, h, T, bins)
    tables = dict(count=assign, keys=assign[..., None]*keys[..., None, :], values=assign[..., None]*v[..., None, :])
    if variance:
        tables['squares'] = assign[..., None]*(keys*keys)[..., None, :]
    prefix = {name: torch.nn.functional.pad(x.cumsum(2), (0,)*(2*(x.dim()-3)) + (1, 0)) for name, x in tables.items()}
    queries = q[:, :, window-1:]                                                    # (b, h, readers, d), reader index = n
    levels = max(1, math.ceil(math.log2(readers)) + 1)
    pieces, logits_levels = [], []
    for level in range(levels):                                                     # fixed structure: log2(T) + 1 steps
        half = 1 << level
        block = 2*half
        blocks = (readers + block - 1)//block
        starts = torch.arange(blocks, device=device)*block                          # segment of block i: [i*block, i*block + half)
        ends = (starts + half).clamp(max=length)
        stats = {name: table.index_select(2, ends.clamp(max=length)) - table.index_select(2, starts.clamp(max=length))
                 for name, table in prefix.items()}
        count = stats['count']                                                      # (b, h, blocks, bins)
        safe = count.clamp(min=1.)
        centre = stats['keys']/safe[..., None]                                      # (b, h, blocks, bins, d)
        padded = torch.nn.functional.pad(queries, (0, 0, 0, blocks*block - readers))
        second = padded.reshape(b, heads, blocks, block, d)[:, :, :, half:]         # readers with bit `level` set
        logits = scale*torch.einsum('bhird,bhikd->bhirk', second, centre) + safe.log()[:, :, :, None]
        if variance:
            spread = (stats['squares']/safe[..., None] - centre*centre).clamp(min=0.)
            logits = logits + .5*scale*scale*torch.einsum('bhird,bhikd->bhirk', second*second, spread)
        logits = logits.masked_fill(~(count > .5)[:, :, :, None], -torch.inf)
        # Back to one row per reader: the first half of every block reads nothing at this level.
        full = torch.full((b, heads, blocks, block, bins), -torch.inf, device=device, dtype=dtype)
        full[:, :, :, half:] = logits
        logits_levels.append(full.reshape(b, heads, blocks*block, bins)[:, :, :readers])
        pieces.append((half, block, blocks, stats['values']/safe[..., None]))
    concept_logits = torch.cat(logits_levels, -1)                                  # (b, h, readers, levels*bins)
    concept_logits = torch.cat((torch.full((b, heads, window-1, levels*bins), -torch.inf, device=device, dtype=dtype),
                                concept_logits), 2)                               # readers before the window read nothing old
    weights = torch.cat((band_logits, concept_logits), -1).softmax(-1)
    out = torch.einsum('bhtw,bhtdw->bhtd', weights[..., :window], band_values)
    old = weights[:, :, window-1:, window:]                                         # (b, h, readers, levels*bins)
    reading = []
    for level, (half, block, blocks, value) in enumerate(pieces):
        share = old[..., level*bins:(level+1)*bins]
        share = torch.nn.functional.pad(share, (0, 0, 0, blocks*block - readers)).reshape(b, heads, blocks, block, bins)
        reading.append(torch.einsum('bhirk,bhikd->bhird', share, value).reshape(b, heads, blocks*block, d)[:, :, :readers])
    concepts = torch.stack(reading).sum(0)
    return out + torch.cat((torch.zeros(b, heads, window-1, d, device=device, dtype=dtype), concepts), 2)


class AlgebraicConceptAttention(torch.nn.Module):
    """A drop-in attention for the small models: projections, rotary positions, concept_scan."""

    def __init__(self, d, heads, window=16, hash_bits=2, seed=0, variance=True):
        super().__init__()
        self.heads, self.head_dim, self.window, self.variance = heads, d//heads, window, variance
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, torch.nn.Linear(d, d, bias=False))
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('directions', torch.randn(self.head_dim, hash_bits, generator=generator), persistent=False)
        self.register_buffer('inverse_frequency', 1./10000.**(torch.arange(0, self.head_dim, 2, dtype=torch.float32)/self.head_dim),
                             persistent=False)

    def attend(self, h, position_embeddings=None):
        b, length, d = h.shape
        split = lambda value: value.view(b, length, self.heads, self.head_dim).transpose(1, 2)
        q, raw, v = split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))
        positions = torch.arange(length, device=h.device, dtype=h.dtype)
        q = rotate(q, positions, self.inverse_frequency)
        out = concept_scan_blocks(q, raw, v, self.inverse_frequency, self.window, self.directions, self.head_dim**-.5, self.variance)
        return self.o_proj(out.transpose(1, 2).reshape(b, length, d))
