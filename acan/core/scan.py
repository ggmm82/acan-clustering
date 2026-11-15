"""Specific concepts of a table from one pass over its rows: the dyadic blocks of the language layer.

The rows are read once, in blocks of ``bins`` rows in which every row is a concept of its own. Two
equal blocks become one block of twice the rows: their concepts are added together, pair by pair
where the reader loses least (reader_cost, acan/concept_algebra.py, with the tabular radius of
ConceptModel), until ``bins`` remain. The same structure and the same rule read the context of
the language model (acan/lm/graph_scan.py); a table has no age, so what is read is only the top:
the concepts of the blocks that are left, at most ``bins`` for every power of two in the number of
rows. They are sums over every row of the table, and they are where the generalization of
ConceptModel starts. Memory is one chunk of rows and the blocks left; time is linear in the rows.
"""
import numpy as np

from acan.concept_algebra import reader_cost

CONSTANT = 1e-6                                                 # as in ConceptModel: dispersion below which an attribute is constant

KEYS = ('C', 'Cm', 'U', 'Q', 'N', 'PV', 'PP', 'PG')


def rows_as_concepts(model, data):
    x, mask, onehot = data[:3]
    n = len(x)
    s = dict(C=np.ones(n), Cm=mask.copy(), U=x.copy(), Q=x*x, N=onehot.copy())
    if len(data) > 3:
        scale, frequency, impurity = data[3:]
        s.update(PV=mask*scale, PP=np.array(frequency), PG=np.array(impurity))
    else:
        s.update(PV=mask.copy(), PP=np.broadcast_to(model.root_p, onehot.shape).copy(), PG=np.broadcast_to(model.root_g, (n, len(model.root_g))).copy())
    return s


class Radius:
    """The radius of ConceptModel for sums of concepts, written out on the arrays (the same numbers, less work)."""

    def __init__(self, model):
        encoder = model.encoder
        read = np.array([name in model.labels for name in encoder.numeric+encoder.categorical], bool)
        recognized = ~read if (~read).any() else np.ones_like(read)
        numeric = len(encoder.numeric)
        self.numeric = np.flatnonzero(recognized[:numeric])
        self.blocks = [encoder.block(name) for name in encoder.categorical]
        self.starts = np.array([block.start for block in self.blocks], int)
        self.categorical = np.flatnonzero(recognized[numeric:])
        self.count = int(recognized.sum())

    def __call__(self, s, other=None):
        """Radius of every concept of ``s``, or of every concept of ``s`` added to ``other`` (broadcast)."""
        take = (lambda key: s[key]) if other is None else (lambda key: s[key]+other[key])
        total = 0.
        if len(self.numeric):
            members = take('Cm')[..., self.numeric]
            safe = np.maximum(members, 1e-12)
            variance = np.maximum(take('Q')[..., self.numeric]/safe-(take('U')[..., self.numeric]/safe)**2, 0)
            prior = np.maximum(np.where(members > 1e-9, take('PV')[..., self.numeric]/safe, 1.), 1e-9)
            total = np.where(prior > CONSTANT, variance/np.maximum(prior, CONSTANT), 1.).sum(-1)
        if len(self.categorical):
            counts = take('N')
            size = np.maximum(np.add.reduceat(counts, self.starts, axis=-1), 1e-12)
            gini = 1-np.add.reduceat(counts*counts, self.starts, axis=-1)/(size*size)
            base = take('PG')/np.maximum(take('C'), 1e-12)[..., None]
            ratio = np.where(base > CONSTANT, gini/np.maximum(base, CONSTANT), 1.)
            total = total+ratio[..., self.categorical].sum(-1)
        return np.sqrt(total/self.count)


def aggregate(model, s, bins, radius=None):
    """s: statistics (blocks, m, ...). Adds together, in every block, the pair the reader loses least on, until ``bins`` remain.

    Reads received are the mass, as for the rows of a table. After a pair is added together only
    the costs of the new concept change. The steps are m - bins, for all the blocks at once.
    """
    radius = radius or Radius(model)
    blocks, m = s['C'].shape
    s = {key: value.copy() for key, value in s.items()}
    rows = np.arange(blocks)
    own = radius(s)                                              # (blocks, m)
    cost = np.empty((blocks, m, m))
    for a in range(m-1):                                         # the first costs: every pair once
        one = {key: value[:, a:a+1] for key, value in s.items()}
        rest = {key: value[:, a+1:] for key, value in s.items()}
        cost[:, a, a+1:] = cost[:, a+1:, a] = reader_cost(one['C'], own[:, a:a+1], rest['C'], own[:, a+1:], radius(rest, one))
    cost[:, np.arange(m), np.arange(m)] = np.inf
    dead = np.zeros((blocks, m), bool)
    for _ in range(m-bins):
        flat = cost.reshape(blocks, -1).argmin(1)
        first, second = np.minimum(flat//m, flat % m), np.maximum(flat//m, flat % m)
        for value in s.values():
            value[rows, first] += value[rows, second]
            value[rows, second] = 0
        dead[rows, second] = True
        one = {key: value[rows, first][:, None] for key, value in s.items()}
        alone = radius(one)
        own[rows, first] = alone[:, 0]
        fresh = reader_cost(one['C'], alone, s['C'], own, radius(s, one))
        fresh[dead] = np.inf
        fresh[rows, first] = np.inf
        cost[rows, first] = fresh
        cost[rows, :, first] = fresh
        cost[rows, second] = np.inf
        cost[rows, :, second] = np.inf
    keep = np.argsort(dead, axis=1, kind='stable')[:, :bins]      # the concepts that remain, in order
    return {key: np.take_along_axis(value, keep.reshape(blocks, bins, *([1]*(value.ndim-2))), 1) for key, value in s.items()}


class Scan:
    """The blocks of a table read a chunk at a time: a binary counter of blocks and the rows not yet in one."""

    def __init__(self, model, bins, chunk=4096):
        self.model, self.bins, self.chunk, self.radius = model, bins, chunk, Radius(model)
        self.counter, self.pending, self.rows = [], None, 0

    def _carry(self, level, node):
        """A block (bins, ...) of bins * 2^level rows enters the counter; equal blocks are generalized."""
        while level < len(self.counter) and self.counter[level] is not None:
            node = self._join(self.counter[level], node)
            self.counter[level] = None
            level += 1
        self.counter.extend([None]*(level+1-len(self.counter)))
        self.counter[level] = node

    def _join(self, older, node):
        both = {key: np.concatenate((older[key], node[key]))[None] for key in KEYS}
        return {key: value[0] for key, value in aggregate(self.model, both, self.bins, self.radius).items()}

    def push(self, data):
        """Rows of the table, already encoded: every one is read once."""
        s = rows_as_concepts(self.model, data)
        if self.pending is not None:
            s = {key: np.concatenate((self.pending[key], s[key])) for key in KEYS}
        self.rows += len(data[0])
        blocks = len(s['C'])//self.bins
        self.pending = {key: value[blocks*self.bins:] for key, value in s.items()}
        node = {key: value[:blocks*self.bins].reshape(blocks, self.bins, *value.shape[1:]) for key, value in s.items()}
        level = 0
        while node['C'].shape[0]:
            if node['C'].shape[0] % 2:                           # the odd block is a digit of the binary counter
                self._carry(level, {key: value[-1] for key, value in node.items()})
                node = {key: value[:-1] for key, value in node.items()}
                if not node['C'].shape[0]:
                    break
            paired = {key: value.reshape(value.shape[0]//2, 2*value.shape[1], *value.shape[2:]) for key, value in node.items()}
            parts = [aggregate(self.model, {key: value[start:start+self.chunk] for key, value in paired.items()}, self.bins, self.radius)
                     for start in range(0, paired['C'].shape[0], self.chunk)]
            node = {key: np.concatenate([part[key] for part in parts]) for key in KEYS}
            level += 1
        return self

    def absorb(self, other):
        """The blocks of another part of the table, read elsewhere, join this counter (sums are additive)."""
        self.rows += other.rows
        for level, node in enumerate(other.counter):
            if node is not None:
                self._carry(level, node)
        if other.pending is not None and len(other.pending['C']):
            waiting = other.pending if self.pending is None else {key: np.concatenate((self.pending[key], other.pending[key])) for key in KEYS}
            blocks = len(waiting['C'])//self.bins
            for block in range(blocks):
                self._carry(0, {key: value[block*self.bins:(block+1)*self.bins] for key, value in waiting.items()})
            self.pending = {key: value[blocks*self.bins:] for key, value in waiting.items()}
        return self

    def result(self):
        """The concepts left: those of the blocks of the counter and the rows that fill no block."""
        parts = [node for node in self.counter if node is not None]+([self.pending] if self.pending is not None else [])
        out = {key: np.concatenate([part[key] for part in parts]) for key in KEYS}
        alive = out['C'] > 0
        return {key: value[alive] for key, value in out.items()}


def scan(model, data, bins, order=None, chunk=4096):
    """One pass over rows held in memory, in the given order. Returns the concepts left."""
    if order is not None:
        data = tuple(part[order] for part in data)
    return Scan(model, bins, chunk).push(data).result()


def scan_part(arguments):
    """One chunk of a table read in another process: its blocks, to be absorbed by the counter of the whole."""
    model, bins, frame = arguments
    part = Scan(model, bins).push(model.encoder.transform(frame))
    part.model = part.radius = None                              # only the sums travel back
    return part
