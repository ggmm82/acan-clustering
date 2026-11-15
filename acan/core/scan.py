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


def aggregate(model, s, bins):
    """s: statistics (blocks, m, ...). Adds together, in every block, the pair the reader loses least on, until ``bins`` remain."""
    blocks, m = s['C'].shape
    s = {key: value.copy() for key, value in s.items()}
    rows = np.arange(blocks)
    own = model._radius(s)                                       # (blocks, m)
    cost = np.empty((blocks, m, m))
    for a in range(m):                                           # the first costs: every pair once
        together = model._radius({key: value[:, a:a+1]+value for key, value in s.items()})
        cost[:, a] = reader_cost(s['C'][:, a:a+1], own[:, a:a+1], s['C'], own, together)
    cost[:, np.arange(m), np.arange(m)] = np.inf
    dead = np.zeros((blocks, m), bool)
    for _ in range(m-bins):
        flat = cost.reshape(blocks, -1).argmin(1)
        first, second = np.minimum(flat//m, flat % m), np.maximum(flat//m, flat % m)
        for key, value in s.items():
            value[rows, first] += value[rows, second]
            value[rows, second] = 0
        dead[rows, second] = True
        one = {key: value[rows, first][:, None] for key, value in s.items()}
        own[rows, first] = model._radius(one)[:, 0]
        fresh = reader_cost(one['C'], own[rows, first][:, None], s['C'], own, model._radius({key: one[key]+value for key, value in s.items()}))
        fresh[dead] = np.inf
        fresh[rows, first] = np.inf
        cost[rows, first] = fresh
        cost[rows, :, first] = fresh
        cost[rows, second] = np.inf
        cost[rows, :, second] = np.inf
    keep = np.argsort(dead, axis=1, kind='stable')[:, :bins]      # the concepts that remain, in order
    return {key: np.take_along_axis(value, keep.reshape(blocks, bins, *([1]*(value.ndim-2))), 1) for key, value in s.items()}


def scan(model, data, bins, order, chunk=4096):
    """One pass: rows in blocks of ``bins``, equal blocks generalized two by two. Returns the concepts left."""
    s = rows_as_concepts(model, data)
    s = {key: value[order] for key, value in s.items()}
    n = len(order)
    blocks = n//bins
    node = {key: value[:blocks*bins].reshape(blocks, bins, *value.shape[1:]) for key, value in s.items()}
    left = [{key: value[blocks*bins:] for key, value in s.items()}]          # rows that do not fill a block
    while node['C'].shape[0] > 1:
        count = node['C'].shape[0]
        if count % 2:                                            # the odd block waits, as a digit of the binary counter
            left.append({key: value[-1] for key, value in node.items()})
            node = {key: value[:-1] for key, value in node.items()}
        paired = {key: value.reshape(value.shape[0]//2, 2*value.shape[1], *value.shape[2:]) for key, value in node.items()}
        parts = [aggregate(model, {key: value[start:start+chunk] for key, value in paired.items()}, bins)
                 for start in range(0, paired['C'].shape[0], chunk)]
        node = {key: np.concatenate([part[key] for part in parts]) for key in KEYS}
    left.append({key: value[0] for key, value in node.items()} if node['C'].shape[0] else {key: value[:0, 0] for key, value in node.items()})
    out = {key: np.concatenate([part[key] for part in left]) for key in KEYS}
    alive = out['C'] > 0
    return {key: value[alive] for key, value in out.items()}
