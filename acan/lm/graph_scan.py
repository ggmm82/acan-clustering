"""The concept graph of a context as a fixed dyadic structure: no search over the context.

One model for tables and for language (MODEL.md, acan/concept_algebra.py). Nodes are concepts:
additive statistics of the observations they contain,

    C = sum 1,  U = sum z,  S = sum |z|^2,  P = sum position,  V = sum y,  A = sum reads,

read with mass times similarity, log C + s q.centre. The observation z of a token is its key before
the rotary position, y its value; the centre is rotated to the mean position of the members when it
is read. Concepts are generalized by adding together the pair the reader loses least on:
reads received times radius (reader_cost), the same cost that generalizes the rows of a table.

What is fixed is *when* and *among which* concepts the generalization happens:

* a block of level j covers 2^j consecutive tokens and holds at most ``bins`` concepts. It is the
  generalization of its two children of level j-1: while they hold together at most ``bins``
  concepts they stay as they are (exact); otherwise their concepts are added together, pair by
  pair, until ``bins`` remain. Every concept of a level is a sum of concepts of the level below:
  the levels are nested and additive, as in the tabular graph;
* a reader at position t reads its last ``window`` tokens exactly and, for its n = t - window + 1
  older tokens, the blocks of the binary writing of n (one per set bit): fine near, coarse far, at
  most ``bins`` * (log2 n + 1) concepts. With bins >= n the reading is dense attention;
* the reads of a token are the shares it received from the readers that had it in their window
  (known when it leaves the window), so that tokens that are read stay distinct.

Training builds every level once for all readers: log2 T batched steps, a structure fixed by the
length like the layers of a network. Generation keeps the same blocks as a binary counter: a token
that leaves the window enters level 0, and equal blocks are generalized by the same function, so
training and generation give the same reading. The structure is chosen without gradient, as in the
present layer; the sums are differentiable.
"""
import torch
from torch.nn import functional as F

from acan.concept_algebra import reader_cost, reader_radius

SCALARS, VECTORS = ('C', 'S', 'P', 'A'), ('U', 'V')


def rotate(keys, positions, inverse_frequency):
    """Rotary position applied at (possibly fractional) positions; matches the library at integers."""
    angles = positions[..., None]*inverse_frequency
    angles = torch.cat((angles, angles), -1)
    half = keys.shape[-1]//2
    return keys*angles.cos()+torch.cat((-keys[..., half:], keys[..., :half]), -1)*angles.sin()


def tokens(raw, values, received, positions):
    """Every token as a concept of its own: (..., n, d) and (..., n) -> statistics with one member each."""
    return dict(C=torch.ones_like(received), S=(raw*raw).sum(-1), P=positions.expand(received.shape).to(raw.dtype),
                A=received, U=raw, V=values)


@torch.no_grad()
def aggregate(count, centre_sum, square_sum, bins, reads=None):
    """Which of ``bins`` concepts each child joins (..., m), adding together the pair the reader loses least on.

    The cost is the one of the tabular model and of the present layer (acan/concept_algebra.py,
    reader_cost): reads received times radius, the radius being the spread of the keys around the
    centre. With additive statistics the cost of every pair is a closed form of the counts, the
    products of the key sums and the sums of squares; after a pair is added together only its row
    changes. The steps are m - bins, fixed by the sizes and not by the data, for all blocks at once.
    count (..., m), centre_sum (..., m, d), square_sum (..., m): C, U and sum |z|^2; reads (..., m):
    what the members were read while they were exact (unread concepts weigh a little, as in the
    present layer).
    """
    m, size = count.shape[-1], centre_sum.shape[-1]
    count, square = count.clone(), square_sum.clone()
    reads = torch.zeros_like(count) if reads is None else reads.clone()
    floor = reads.mean(-1, keepdim=True)*1e-3+1e-12
    gram = centre_sum@centre_sum.transpose(-1, -2)               # products of the key sums: additive in both indices
    norm = gram.diagonal(dim1=-2, dim2=-1).clone()
    radius = lambda c, n, q: reader_radius(((q/c.clamp(min=1.)-n/(c*c).clamp(min=1.)).clamp(min=0.))/size, torch)
    own = radius(count, norm, square)
    rate = reads+floor
    together = radius(count[..., :, None]+count[..., None, :], norm[..., :, None]+norm[..., None, :]+2*gram,
                      square[..., :, None]+square[..., None, :])
    cost = reader_cost(rate[..., :, None], own[..., :, None], rate[..., None, :], own[..., None, :], together)
    cost.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
    label = torch.arange(m, device=count.device).expand(count.shape).clone()
    dead = torch.zeros_like(count, dtype=torch.bool)
    for _ in range(m-bins):
        flat = cost.flatten(-2).argmin(-1)
        first, second = torch.minimum(flat//m, flat % m)[..., None], torch.maximum(flat//m, flat % m)[..., None]
        for value in (count, square, reads, norm):                   # the second concept joins the first
            value.scatter_add_(-1, first, value.gather(-1, second))
        row, column = first[..., None].expand(*first.shape, m), first[..., None].expand(*first.shape[:-1], m, 1)
        gram.scatter_add_(-2, row, gram.gather(-2, second[..., None].expand(*second.shape, m)))
        gram.scatter_add_(-1, column, gram.gather(-1, second[..., None].expand(*second.shape[:-1], m, 1)))
        mixed = gram.gather(-2, row).squeeze(-2)                     # products of the new sum with every other
        norm.scatter_(-1, first, mixed.gather(-1, first))
        dead.scatter_(-1, second, True)
        mass, spread, read, length = (value.gather(-1, first) for value in (count, square, reads, norm))
        alone = radius(mass, length, spread)
        own.scatter_(-1, first, alone)
        fresh = reader_cost(read+floor, alone, reads+floor, own, radius(mass+count, length+norm+2*mixed, spread+square))
        fresh = fresh.masked_fill(dead, torch.inf).scatter_(-1, first, torch.inf)
        cost.scatter_(-2, row, fresh[..., None, :])
        cost.scatter_(-1, column, fresh[..., :, None])
        gone = second[..., None]
        cost.scatter_(-2, gone.expand(*second.shape, m), torch.inf)
        cost.scatter_(-1, gone.expand(*second.shape[:-1], m, 1), torch.inf)
        label = torch.where(label == second, first, label)
    rank = (~dead).long().cumsum(-1)-1                           # the concepts that remain, numbered in order
    return rank.gather(-1, label)


def generalize(node, bins):
    """A block of concepts (..., m[, d]) as at most ``bins`` concepts: sums of the children."""
    if node['C'].shape[-1] <= bins:
        return node
    code = aggregate(node['C'].detach(), node['U'].detach(), node['S'].detach(), bins, node['A'].detach())
    gather = F.one_hot(code, bins).to(node['C'].dtype).transpose(-1, -2)          # (..., bins, m)
    out = {name: (gather@node[name][..., None]).squeeze(-1) for name in SCALARS}
    out.update({name: gather@node[name] for name in VECTORS})
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
    """Logits of queries (..., r, d) over the concepts of their block (..., k[, d]): log C + s q.centre."""
    mass = node['C'].clamp(min=1.)
    centre = rotate(node['U']/mass[..., None], node['P']/mass, frequency)
    logits = scale*queries@centre.transpose(-1, -2)+mass.log()[..., None, :]
    return logits.masked_fill((node['C'] <= .5)[..., None, :], -torch.inf)


def unit_of(bins):
    """Tokens of the largest block that is still exact: the largest power of two not above ``bins``."""
    return 1 << (bins.bit_length()-1)


def graph_scan(q, raw, keys, v, window, bins, scale, frequency, details=False):
    """Attention output (b, h, T, d) of every position over its window and its concept graph.

    q, keys: queries and keys with their rotary positions; raw: keys before them; v: values.
    Blocks of up to ``bins`` tokens are exact (one concept per token), so they are read together
    with the window, as ordinary attention between chunks of ``window`` queries and the keys just
    behind them; every level above is one batched product. ``details`` also returns the reads
    each token received inside the windows (b, h, T).
    """
    b, heads, length, d = q.shape
    device = q.device
    unit = unit_of(bins)
    # Exact part: reader t reads the tokens from the last multiple of ``unit`` of its old count n to t.
    back = window+unit-1
    chunks = (length+window-1)//window
    front = lambda x: F.pad(x, (0, 0, back, chunks*window-length))
    key_windows = front(keys).unfold(2, back+window, window).transpose(-1, -2)      # (b, h, chunks, back+window, d)
    value_windows = front(v).unfold(2, back+window, window).transpose(-1, -2)
    chunked = F.pad(q, (0, 0, 0, chunks*window-length)).reshape(b, heads, chunks, window, d)
    exact = scale*chunked@key_windows.transpose(-1, -2)                             # (b, h, chunks, window, back+window)
    reader = torch.arange(chunks*window, device=device).reshape(chunks, window, 1)
    place = (torch.arange(chunks, device=device)*window-back)[:, None, None]+torch.arange(back+window, device=device)
    count = (reader-window+1).clamp(min=0)                                         # old tokens of each reader
    causal = exact.masked_fill(place > reader, -torch.inf)
    with torch.no_grad():                                        # what each token is read while it is in a window
        near = causal.masked_fill((place < reader-window+1) | (place < 0), -torch.inf).softmax(-1).nan_to_num(0.)
        near = near.masked_fill(reader >= length, 0.).sum(3)     # readers of the padding do not exist
        received = near.new_zeros(b, heads, chunks*window+back+window)
        received.index_add_(2, (place[:, 0]+back).reshape(-1), near.reshape(b, heads, -1))
        received = received[:, :, back:back+length]
    exact = causal.masked_fill(place < count-count % unit, -torch.inf)
    exact = exact.reshape(b, heads, chunks*window, back+window)[:, :, :length]
    old = length-window
    logits, values = [], []
    if old >= unit:
        readers = old+1                                                            # reader with n old tokens, n = 0 .. old
        queries = q[:, :, window-1:]
        blocks = old//unit
        cut = lambda x: x[:, :, :blocks*unit].reshape(b, heads, blocks, unit, *x.shape[3:])
        node = tokens(cut(raw), cut(v), cut(received), torch.arange(blocks*unit, device=device).reshape(blocks, unit))
        for level in range(unit.bit_length()-1, old.bit_length()):                 # fixed structure: log2(T) steps
            if node['C'].shape[2] == 0:
                break
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
            part = read(second, even, scale, frequency)                            # (b, h, blocks, half, concepts)
            part = F.pad(part, (0, 0, half, 0), value=-torch.inf).reshape(b, heads, blocks*block, -1)[:, :, :readers]
            logits.append(part)
            values.append((half, block, blocks, even['V']/even['C'].clamp(min=1.)[..., None]))
            node = generalize(pair(node), bins)
    concept = F.pad(torch.cat(logits, -1), (0, 0, window-1, 0), value=-torch.inf) if logits else exact[..., :0]
    weights = torch.cat((exact, concept), -1).softmax(-1)
    share = F.pad(weights[..., :back+window], (0, 0, 0, chunks*window-length)).reshape(b, heads, chunks, window, back+window)
    out = (share@value_windows).reshape(b, heads, chunks*window, d)[:, :, :length]
    if logits:
        total, readers = 0, old+1
        shares = weights[:, :, window-1:, back+window:].split([part.shape[-1] for part in logits], -1)
        for share, (half, block, blocks, value) in zip(shares, values):
            share = F.pad(share, (0, 0, 0, blocks*block-readers)).reshape(b, heads, blocks, block, -1)[:, :, :, half:]
            total = total+F.pad(share@value, (0, 0, half, 0)).reshape(b, heads, blocks*block, d)[:, :, :readers]
        out = out+F.pad(total, (0, 0, window-1, 0))
    return (out, received) if details else out


class GraphStream:
    """The same reading one token at a time: ``window`` exact tokens and a binary counter of blocks."""

    def __init__(self, window, bins, scale, frequency):
        self.window, self.bins, self.scale, self.frequency = window, bins, scale, frequency
        self.keys = self.values = self.raw = self.received = self.table = None
        self.counter, self.seen = [], 0

    def load(self, raw, keys, values, received):
        """Start from a whole prefix (b, h, n, d): what ``step`` would hold after its n tokens."""
        length = raw.shape[2]
        old = max(length-self.window, 0)
        self.seen, self.counter, self.table = length, [None]*old.bit_length(), None
        self.keys, self.values, self.raw, self.received = keys[:, :, old:], values[:, :, old:], raw[:, :, old:], received[:, :, old:].clone()
        unit, start = unit_of(self.bins), 0
        for level in reversed(range(old.bit_length())):          # the blocks of the binary writing of ``old``
            size = 1 << level
            if not old & size:
                continue
            part = slice(start, start+size)
            shape = (max(size//unit, 1), min(size, unit))
            cut = lambda x: x[:, :, part].reshape(*x.shape[:2], *shape, *x.shape[3:])
            node = tokens(cut(raw), cut(values), cut(received), torch.arange(start, start+size, device=raw.device).reshape(shape))
            while node['C'].shape[2] > 1:
                node = generalize(pair(node), self.bins)
            self.counter[level] = {name: value[:, :, 0] for name, value in node.items()}
            start += size
        return self

    def step(self, q, raw, key, v):
        """q, key: (b, h, d) of the new token with its rotary position; raw: its key before; v: its value."""
        add = lambda held, new: new[:, :, None] if held is None else torch.cat((held, new[:, :, None]), 2)
        self.keys, self.values, self.raw = add(self.keys, key), add(self.values, v), add(self.raw, raw)
        self.seen += 1
        exact = self.scale*(self.keys@q[..., None]).squeeze(-1)
        near = exact.softmax(-1).detach()                        # the reader's shares inside its window
        self.received = add(self.received, near.new_zeros(near.shape[:2]))
        if self.keys.shape[2] > self.window:                     # the oldest exact token becomes old
            self.received[:, :, 1:] += exact[:, :, 1:].softmax(-1).detach()
            node = tokens(self.raw[:, :, :1], self.values[:, :, :1], self.received[:, :, :1],
                          torch.full((1,), self.seen-self.window-1, device=q.device))
            self.keys, self.values, self.raw = self.keys[:, :, 1:], self.values[:, :, 1:], self.raw[:, :, 1:]
            self.received, exact = self.received[:, :, 1:], exact[:, :, 1:]
            level = 0
            while level < len(self.counter) and self.counter[level] is not None:   # a carry: at most log2(n) steps
                older = self.counter[level]
                node = generalize({name: torch.cat((older[name], node[name]), 2) for name in node}, self.bins)
                self.counter[level] = None
                level += 1
            if level == len(self.counter):
                self.counter.append(None)
            self.counter[level] = node
            self.table = None
        else:
            self.received += near
        if self.table is None:                                   # every block of the counter side by side: one reading
            nodes = [node for node in self.counter if node is not None]
            self.table = {name: torch.cat([node[name] for node in nodes], 2) for name in nodes[0]} if nodes else {}
        logits, values = exact, self.values
        if self.table:
            logits = torch.cat((logits, read(q[:, :, None], self.table, self.scale, self.frequency).squeeze(2)), -1)
            values = torch.cat((values, self.table['V']/self.table['C'].clamp(min=1.)[..., None]), 2)
        self.shares = logits.softmax(-1)
        return (self.shares[..., None]*values).sum(2)

    def kept_vectors(self):
        return self.keys.shape[2]+sum(node['C'].shape[-1] for node in self.counter if node is not None)

    def tensors(self):
        held = [self.keys, self.values, self.raw, self.received]
        return [item for item in held if item is not None]+[value for node in self.counter if node is not None for value in node.values()]


class GraphScanAttention(torch.nn.Module):
    """Attention for the small models: projections, rotary positions, graph_scan."""

    def __init__(self, d, heads, window=16, bins=16):
        super().__init__()
        self.heads, self.head_dim, self.window, self.bins = heads, d//heads, window, bins
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self, name, torch.nn.Linear(d, d, bias=False))
        self.register_buffer('inverse_frequency', 1./10000.**(torch.arange(0, self.head_dim, 2, dtype=torch.float32)/self.head_dim),
                             persistent=False)

    def project(self, h, first=0):
        """Queries and keys with their positions, keys before them, values: (b, heads, T, head_dim) each."""
        b, length, d = h.shape
        split = lambda value: value.view(b, length, self.heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(first, first+length, device=h.device, dtype=h.dtype)
        frequency = self.inverse_frequency.to(h.dtype)
        raw = split(self.k_proj(h))
        return rotate(split(self.q_proj(h)), positions, frequency), raw, rotate(raw, positions, frequency), split(self.v_proj(h))

    def attend(self, h, position_embeddings=None):
        b, length, d = h.shape
        out = graph_scan(*self.project(h), self.window, self.bins, self.head_dim**-.5, self.inverse_frequency.to(h.dtype))
        return self.o_proj(out.transpose(1, 2).reshape(b, length, d))

    def stream(self, dtype=torch.float32):
        return GraphStream(self.window, self.bins, self.head_dim**-.5, self.inverse_frequency.to(dtype))
