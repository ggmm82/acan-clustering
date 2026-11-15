"""The concept graph of a context as a fixed dyadic structure: no search, no data-dependent loop.

Nodes are the concepts of the concept graph (MODEL.md): additive statistics of the observations they
contain, C = sum 1, U = sum z, Q = sum z^2, V = sum y, read with mass times expected similarity.
The observation z of a token is its key with its rotary position, y its value.

Levels. Level 0 holds the tokens. A node block of level j covers 2^j consecutive tokens and holds
at most ``bins`` concepts; it is the generalization of its two children blocks of level j-1:

* while the children hold together at most ``bins`` concepts they are kept as they are (exact);
* otherwise their 2*bins concepts are added together into ``bins`` concepts. Which ones is the
  partition that lowers the spread of the keys inside the concepts, i.e. raises the CP of the thesis
  (within-concept variance is what CP^2 measures, and what bounds the reader's error): the signs
  along the leading directions of the children's centres around their mean (power iterations from
  fixed directions), then Lloyd steps, each of which can only lower the spread. The structure is
  chosen without gradient, as in the present layer; the sums are differentiable.

Every concept of level j is the sum of concepts of level j-1: the levels are nested and additive, as
in the tabular graph. A reader at position t reads its last ``window`` tokens exactly and, for its
n = t - window + 1 older tokens, the blocks of the binary writing of n (one per set bit): fine near,
coarse far, at most ``bins`` * (log2 n + 1) concepts. With bins >= n the reading is dense attention.

Training builds every level once for all readers (log2 T batched steps, a fixed structure like the
layers of a network). Generation keeps the same blocks as a binary counter: when a token leaves the
window it enters level 0 and equal blocks are generalized by the same function, so both agree.
"""
import math

import torch
from torch.nn import functional as F


def rotate(keys, positions, inverse_frequency):
    angles = positions[..., None]*inverse_frequency
    angles = torch.cat((angles, angles), -1)
    half = keys.shape[-1]//2
    return keys*angles.cos()+torch.cat((-keys[..., half:], keys[..., :half]), -1)*angles.sin()


@torch.no_grad()
def partition(count, centre_sum, bins, directions, power=2, lloyd=1, weight=None):
    """Which of ``bins`` concepts each child concept joins (..., m): the spread inside is lowered.

    count (..., m), centre_sum (..., m, d): masses and key sums of the children. Children with no
    mass are ignored. Signs along the leading directions of the centres (power iterations started
    from the fixed ``directions`` (d, r), bins = 2^r), then Lloyd steps on the centres.
    """
    mass = count[..., None]
    centre = centre_sum/mass.clamp(min=1.)
    mean = centre_sum.sum(-2, keepdim=True)/mass.sum(-2, keepdim=True).clamp(min=1.)
    x = (centre-mean)*(mass > 0)
    if weight is not None:                                       # the spread as the readers see it
        x = x*weight.sqrt()[..., None, :]
    weighted = x*mass
    code = torch.zeros(count.shape, dtype=torch.long, device=count.device)
    previous = []
    for bit in range(directions.shape[1]):
        w = directions[:, bit].expand(*x.shape[:-2], -1) if power else directions[:, bit]
        for _ in range(power):
            w = (weighted.transpose(-1, -2)@(x@w[..., None])).squeeze(-1)
            for other in previous:                               # the next direction, across the previous ones
                w = w-other*(w*other).sum(-1, keepdim=True)
            w = w/w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        previous.append(w)
        code = code+(((x@w[..., None]).squeeze(-1) > 0).long() << bit)
    for _ in range(lloyd):
        member = F.one_hot(code, bins).to(x.dtype)*mass          # (..., m, bins)
        total = member.sum(-2)                                   # (..., bins)
        middle = member.transpose(-1, -2)@x/total.clamp(min=1e-12)[..., None]
        distance = (x*x).sum(-1, keepdim=True)-2*x@middle.transpose(-1, -2)+(middle*middle).sum(-1)[..., None, :]
        code = distance.masked_fill((total <= 0)[..., None, :], torch.inf).argmin(-1)
    return code


@torch.no_grad()
def aggregate(count, centre_sum, bins):
    """Which of ``bins`` concepts each child joins (..., m), adding together the pair that costs least.

    The cost of adding two concepts is the growth of the spread inside them,
    C_a C_b / (C_a + C_b) * |g_a - g_b|^2: with additive statistics it is a closed form, and the
    least growth of the spread is the least loss of CP (the aggregation of the thesis). The steps
    are m - bins, fixed by the sizes and not by the data, for all blocks at once.
    """
    m = count.shape[-1]
    count, total = count.clone(), centre_sum.clone()
    label = torch.arange(m, device=count.device).expand(count.shape).clone()
    dead = torch.zeros_like(count, dtype=torch.bool)
    eye = torch.eye(m, dtype=torch.bool, device=count.device)
    for _ in range(m-bins):
        centre = total/count.clamp(min=1.)[..., None]
        square = (centre*centre).sum(-1)
        distance = (square[..., :, None]+square[..., None, :]-2*centre@centre.transpose(-1, -2)).clamp(min=0.)
        both = count[..., :, None]+count[..., None, :]
        cost = count[..., :, None]*count[..., None, :]/both.clamp(min=1.)*distance
        cost = cost.masked_fill(eye | dead[..., :, None] | dead[..., None, :], torch.inf)
        flat = cost.flatten(-2).argmin(-1)
        first, second = torch.minimum(flat//m, flat % m), torch.maximum(flat//m, flat % m)
        into, gone = F.one_hot(first, m).to(count.dtype), F.one_hot(second, m).to(count.dtype)
        moved = (count*gone).sum(-1, keepdim=True)
        moved_sum = (total*gone[..., None]).sum(-2, keepdim=True)
        count = count*(1-gone)+into*moved
        total = total*(1-gone[..., None])+into[..., None]*moved_sum
        label = torch.where(label == second[..., None], first[..., None], label)
        dead = dead | gone.bool()
    rank = (~dead).long().cumsum(-1)-1                           # the concepts that remain, numbered in order
    return rank.gather(-1, label)


def generalize(node, bins, directions, power=2, lloyd=1, weight=None):
    """A block of 2*bins concepts (or fewer) as at most ``bins`` concepts: sums of the children."""
    if node['C'].shape[-1] <= bins:
        return node
    code = (aggregate(node['C'].detach(), node['U'].detach(), bins) if lloyd == 'aggregate' else
            partition(node['C'].detach(), node['U'].detach(), bins, directions, power, lloyd, weight))
    member = F.one_hot(code, bins).to(node['C'].dtype)           # (..., m, bins)
    gather = member.transpose(-1, -2)
    out = dict(C=(gather@node['C'][..., None]).squeeze(-1), P=(gather@node['P'][..., None]).squeeze(-1))
    for name in ('U', 'V'):
        out[name] = gather@node[name]
    return out


def pair(node):
    """Consecutive blocks two by two: (b, h, blocks, m[, d]) -> (b, h, blocks//2, 2m[, d])."""
    out = {}
    for name, value in node.items():
        blocks = value.shape[2]//2
        value = value[:, :, :2*blocks]
        out[name] = value.reshape(*value.shape[:2], blocks, 2*value.shape[3], *value.shape[4:])
    return out


def read(queries, node, scale, frequency):
    """Logits of queries (..., r, d) over the concepts of their block: log C + s q.centre.

    Keys are summed before their rotary position, as in the present layer: a concept keeps the
    content of its members whatever their positions, and its centre is rotated to their mean
    position when it is read.
    """
    mass = node['C'].clamp(min=1.)
    centre = rotate(node['U']/mass[..., None], node['P']/mass, frequency)
    logits = scale*queries@centre.transpose(-1, -2)+mass.log()[..., None, :]
    return logits.masked_fill((node['C'] <= .5)[..., None, :], -torch.inf)


def graph_scan(q, raw, keys, v, window, bins, directions, scale, frequency, power=2, lloyd=1, energy=None):
    """Attention output (b, h, T, d) of every position over its window and its concept graph.

    q, keys: queries and keys with their rotary positions; raw: keys before them; v: values. Blocks of up to ``bins`` tokens
    are exact (one concept per token), so they are read together with the window, as ordinary
    attention between chunks of ``window`` queries and the keys just behind them; the levels above
    are one batched product each.
    """
    b, heads, length, d = q.shape
    device = q.device
    # Exact part: reader t reads the tokens from the last multiple of ``bins`` of its old count n to t.
    back = window+bins-1
    chunks = (length+window-1)//window
    front = lambda x: F.pad(x, (0, 0, back, chunks*window-length))
    key_windows = front(keys).unfold(2, back+window, window).transpose(-1, -2)      # (b, h, chunks, back+window, d)
    value_windows = front(v).unfold(2, back+window, window).transpose(-1, -2)
    chunked = F.pad(q, (0, 0, 0, chunks*window-length)).reshape(b, heads, chunks, window, d)
    exact = scale*chunked@key_windows.transpose(-1, -2)                             # (b, h, chunks, window, back+window)
    reader = torch.arange(chunks*window, device=device).reshape(chunks, window, 1)
    place = (torch.arange(chunks, device=device)*window-back)[:, None, None]+torch.arange(back+window, device=device)
    count = (reader-window+1).clamp(min=0)                                         # old tokens of each reader
    exact = exact.masked_fill((place > reader) | (place < count-count % bins), -torch.inf)
    exact = exact.reshape(b, heads, chunks*window, back+window)[:, :, :length]
    old = length-window
    first = (bins-1).bit_length()                                                  # level whose blocks hold ``bins`` tokens
    logits, values = [], []
    if old >= bins:
        readers = old+1                                                            # reader with n old tokens, n = 0 .. old
        queries = q[:, :, window-1:]
        blocks = old//bins
        cut = lambda x: x[:, :, :blocks*bins].reshape(b, heads, blocks, bins, d)
        z, y = cut(raw), cut(v)
        node = dict(C=z.new_ones(b, heads, blocks, bins), U=z, V=y,
                    P=torch.arange(blocks*bins, device=device, dtype=z.dtype).reshape(blocks, bins).expand(b, heads, -1, -1))
        for level in range(first, old.bit_length()):                               # fixed structure: log2(T) steps
            if level > first:
                last = (torch.arange(node['C'].shape[2]//2, device=device)+1)*(1 << level)-1   # last token of each block
                node = generalize(pair(node), bins, directions, power, lloyd,
                                  None if energy is None else energy[:, :, last])
            half = 1 << level
            block = 2*half
            blocks = (readers+block-1)//block
            # Readers with bit ``level`` of n set are the second half of each block of 2^(level+1)
            # counts and read the first 2^level tokens of it: the even blocks of this level.
            even = {name: value[:, :, ::2][:, :, :blocks] for name, value in node.items()}
            missing = blocks-even['C'].shape[2]
            if missing:
                even = {name: F.pad(value, (0, 0)*(value.dim()-3)+(0, missing)) for name, value in even.items()}
            second = F.pad(queries, (0, 0, 0, blocks*block-readers)).reshape(b, heads, blocks, block, d)[:, :, :, half:]
            part = read(second, even, scale, frequency)                            # (b, h, blocks, half, bins)
            part = F.pad(part, (0, 0, half, 0), value=-torch.inf).reshape(b, heads, blocks*block, bins)[:, :, :readers]
            logits.append(part)
            values.append((half, block, blocks, even['V']/even['C'].clamp(min=1.)[..., None]))
    concept = F.pad(torch.cat(logits, -1), (0, 0, window-1, 0), value=-torch.inf) if logits else exact[..., :0]
    weights = torch.cat((exact, concept), -1).softmax(-1)
    share = F.pad(weights[..., :back+window], (0, 0, 0, chunks*window-length)).reshape(b, heads, chunks, window, back+window)
    out = (share@value_windows).reshape(b, heads, chunks*window, d)[:, :, :length]
    if logits:
        total = 0
        readers = old+1
        for share, (half, block, blocks, value) in zip(weights[:, :, window-1:, back+window:].split(bins, -1), values):
            share = F.pad(share, (0, 0, 0, blocks*block-readers)).reshape(b, heads, blocks, block, bins)[:, :, :, half:]
            total = total+F.pad(share@value, (0, 0, half, 0)).reshape(b, heads, blocks*block, d)[:, :, :readers]
        out = out+F.pad(total, (0, 0, window-1, 0))
    return out


class GraphStream:
    """The same reading one token at a time: ``window`` exact tokens and a binary counter of blocks."""

    def __init__(self, window, bins, directions, scale, frequency, power=2, lloyd=1):
        self.window, self.bins, self.directions, self.scale, self.frequency = window, bins, directions, scale, frequency
        self.power, self.lloyd = power, lloyd
        self.keys, self.values, self.raw, self.counter, self.table, self.seen, self.energy = None, None, None, [], None, 0, None

    def step(self, q, raw, key, v, energy=None):
        """q, key: (b, h, d) of the new token with its rotary position; raw: its key before; v: its value."""
        self.raw = raw[:, :, None] if self.raw is None else torch.cat((self.raw, raw[:, :, None]), 2)
        self.seen += 1
        if energy is not None:
            self.energy = energy[:, :, None] if self.energy is None else torch.cat((self.energy, energy[:, :, None]), 2)
        self.keys = key[:, :, None] if self.keys is None else torch.cat((self.keys, key[:, :, None]), 2)
        self.values = v[:, :, None] if self.values is None else torch.cat((self.values, v[:, :, None]), 2)
        if self.keys.shape[2] > self.window:                     # the oldest exact token becomes old
            z, y = self.raw[:, :, :1], self.values[:, :, :1]
            self.keys, self.values, self.raw = self.keys[:, :, 1:], self.values[:, :, 1:], self.raw[:, :, 1:]
            weight = None
            if self.energy is not None:                          # what the queries asked up to the token that leaves
                weight, self.energy = self.energy[:, :, 0], self.energy[:, :, 1:]
            node = dict(C=z.new_ones(*z.shape[:2], 1), U=z, V=y, P=z.new_full((*z.shape[:2], 1), self.seen-self.window-1))
            level = 0
            while level < len(self.counter) and self.counter[level] is not None:   # a carry: at most log2(n) steps
                older = self.counter[level]
                node = generalize({name: torch.cat((older[name], node[name]), 2) for name in node},
                                  self.bins, self.directions, self.power, self.lloyd, weight)
                self.counter[level] = None
                level += 1
            if level == len(self.counter):
                self.counter.append(None)
            self.counter[level] = node
            self.table = None
        if self.table is None:                                   # every block of the counter side by side: one reading
            nodes = [node for node in self.counter if node is not None]
            self.table = {name: torch.cat([node[name] for node in nodes], 2) for name in nodes[0]} if nodes else {}
        logits, values = self.scale*(self.keys@q[..., None]).squeeze(-1), self.values
        if self.table:
            logits = torch.cat((logits, read(q[:, :, None], self.table, self.scale, self.frequency).squeeze(2)), -1)
            values = torch.cat((values, self.table['V']/self.table['C'].clamp(min=1.)[..., None]), 2)
        return (logits.softmax(-1)[..., None]*values).sum(2)

    def kept_vectors(self):
        return self.keys.shape[2]+sum(node['C'].shape[-1] for node in self.counter if node is not None)


class GraphScanAttention(torch.nn.Module):
    """Attention for the small models: projections, rotary positions, graph_scan."""

    def __init__(self, d, heads, window=16, bins=8, seed=0, power=2, lloyd=1, reader=False):
        super().__init__()
        self.heads, self.head_dim, self.window, self.bins = heads, d//heads, window, bins
        self.power, self.lloyd, self.reader = power, lloyd, reader
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, torch.nn.Linear(d, d, bias=False))
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('directions', torch.randn(self.head_dim, int(math.log2(bins)), generator=generator), persistent=False)
        self.register_buffer('inverse_frequency', 1./10000.**(torch.arange(0, self.head_dim, 2, dtype=torch.float32)/self.head_dim),
                             persistent=False)

    def project(self, h, first=0, asked=None):
        """Queries and keys with positions, keys before them, values, and what the queries asked so far.

        The last is the running sum of the squared queries (before positions, the two halves of each
        rotary pair together, so that it does not depend on the position): the reader's error on a
        concept is s q.(z - centre), so the spread that matters is the one weighted by it.
        """
        b, length, d = h.shape
        split = lambda value: value.view(b, length, self.heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(first, first+length, device=h.device, dtype=h.dtype)
        frequency = self.inverse_frequency.to(h.dtype)
        question, raw = split(self.q_proj(h)), split(self.k_proj(h))
        energy = None
        if self.reader:
            with torch.no_grad():
                square = question*question
                half = self.head_dim//2
                square = square[..., :half]+square[..., half:]
                energy = torch.cat((square, square), -1).cumsum(2)+(0 if asked is None else asked[:, :, None])+1e-12
        return rotate(question, positions, frequency), raw, rotate(raw, positions, frequency), split(self.v_proj(h)), energy

    def attend(self, h, position_embeddings=None):
        b, length, d = h.shape
        q, raw, keys, v, energy = self.project(h)
        out = graph_scan(q, raw, keys, v, self.window, self.bins, self.directions.to(h.dtype), self.head_dim**-.5,
                         self.inverse_frequency.to(h.dtype), self.power, self.lloyd, energy)
        return self.o_proj(out.transpose(1, 2).reshape(b, length, d))

    def stream(self):
        return GraphStream(self.window, self.bins, self.directions, self.head_dim**-.5, self.inverse_frequency, self.power, self.lloyd)
