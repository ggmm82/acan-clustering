"""Affinamenti CP di un tratto: geometria osservata e lettura categoriale su tutte le righe."""

import numpy as np

from acan.core.conditional import SemanticCuts


class SpanGeometry:
    """Each wider span sees only the learned numeric descriptions of its two ordered halves."""

    def __init__(self, vectors, scales=20):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.scales = scales

    def fit(self, tokens, starts, sample=16384, precision=14, directory=None):
        from .dyadic import frequencies

        starts = np.asarray(starts, dtype=np.int64)
        n = len(tokens)
        if not n or not len(starts) or starts[0] != 0 or np.any(np.diff(starts) <= 0) or starts[-1] >= n:
            raise ValueError("Servono token e inizi di documento ordinati a partire da zero.")
        ends = np.r_[starts[1:], n]

        def allocate(name, dtype):
            if directory is None:
                return np.empty(n, dtype=dtype)
            return np.lib.format.open_memmap(
                directory / (name + ".npy"), mode="w+", dtype=dtype, shape=(n,)
            )

        rng = np.random.default_rng(42)
        age = allocate("age", np.int64)
        for begin in range(0, n, 65536):
            at = np.arange(begin, min(n, begin + 65536))
            age[at] = at - starts[np.searchsorted(starts, at, side="right") - 1] + 1
        frequency = frequencies(tokens, len(self.vectors))
        cuts = SemanticCuts().fit(self.vectors, frequency, precision)
        self.token_codes = cuts.encode(self.vectors)
        self.depths = [precision]
        self.pairs = []
        child = tokens
        vectors = self.vectors
        col = allocate("span-1", np.uint32)
        for begin in range(0, n, 65536):
            col[begin : begin + 65536] = self.token_codes[tokens[begin : begin + 65536]]
        columns = [col]
        for j in range(1, self.scales):
            half = 1 << (j - 1)
            # The sampled row may be followed by a target only inside the same article.
            cumulative = np.cumsum(np.maximum(ends - starts - 2 * half, 0))
            total = int(cumulative[-1])
            if not total:
                break
            ranks = rng.choice(total, min(sample, total), replace=False)
            docs = np.searchsorted(cumulative, ranks, side="right")
            end = starts[docs] + 2 * half - 1 + ranks - np.r_[0, cumulative[:-1]][docs]
            x = np.column_stack((vectors[child[end - half]], vectors[child[end]]))
            x, mass = np.unique(x, axis=0, return_counts=True)
            depth = min(precision, max(1, int(np.ceil(np.log2(len(end))))))
            tree = SemanticCuts().fit(x, mass, depth)
            codes = self.compose(child, age, half, vectors, tree, allocate(f"span-{2*half}", np.uint32))
            # The vector of a learned span is its categorical target mean in semantic space.
            # Scan all eligible rows, not just the rows used to propose the partition.
            size = 2 ** (depth + 1)
            count = np.zeros(size, np.int64)
            moments = np.zeros((size, self.vectors.shape[1]), dtype=np.float64)
            for begin in range(0, n - 1, 131072):
                at = np.arange(begin, min(n - 1, begin + 131072))
                at = at[age[at + 1] > 2 * half]
                groups = codes[at]
                count += np.bincount(groups, minlength=size)
                following = self.vectors[tokens[at + 1]]
                for d in range(moments.shape[1]):
                    moments[:, d] += np.bincount(groups, following[:, d], minlength=size)
            sums = moments.astype(np.float32)
            # Generalize sparse/unseen leaves to their ancestors, one prior member per cell.
            for h in range(depth, 0, -1):
                lo = 1 << h
                count[lo // 2 : lo] += count[lo : lo * 2].reshape(-1, 2).sum(1)
                sums[lo // 2 : lo] += sums[lo : lo * 2].reshape(-1, 2, sums.shape[1]).sum(1)
            sums[1] /= max(1, count[1])
            for h in range(1, depth + 1):
                lo = 1 << h
                sums[lo : lo * 2] += np.repeat(sums[lo // 2 : lo], 2, axis=0)
                sums[lo : lo * 2] /= (count[lo : lo * 2] + 1)[:, None]
            self.pairs.append((tree, vectors))
            self.depths.append(depth)
            vectors, child = sums, codes
            columns.append(codes)
        return columns

    @staticmethod
    def compose(child, age, half, vectors, tree, output=None):
        codes = np.empty(len(child), np.uint32) if output is None else output
        codes[:] = 0
        for begin in range(0, len(child), 32768):
            at = np.arange(begin, min(begin + 32768, len(child)))
            at = at[age[at] >= 2 * half]
            if len(at):
                x = np.column_stack((vectors[child[at - half]], vectors[child[at]]))
                codes[at] = tree.encode(x)
        return codes

    def transform(self, tokens, starts):
        from .dyadic import ages

        age = ages(len(tokens), starts)
        child = np.asarray(tokens)
        columns = [self.token_codes[child]]
        for j, (tree, vectors) in enumerate(self.pairs):
            child = self.compose(child, age, 1 << j, vectors, tree)
            columns.append(child)
        return columns


def region(columns, age, positions, depths, parts, age_at_positions=False):
    """Ordered rectangular region in the learned cuts; 0 denotes a missing span."""
    key = np.ones(len(positions), np.uint64)
    valid = np.ones(len(key), bool)
    bits = 1
    for lag, scale, precision in parts:
        width = 1 << scale
        bits += precision
        if precision > depths[scale] or bits > 64:
            raise ValueError("Risoluzione oltre la rappresentazione della cella.")
        valid &= (age if age_at_positions else age[positions]) > lag + width
        at = np.maximum(positions - 1 - lag, 0)
        value = columns[scale][at] >> (depths[scale] - precision)
        key = (key << np.uint64(precision)) | (value & ((1 << precision) - 1)).astype(np.uint64)
    key[~valid] = 0
    return key


def proposals(depths):
    """From broad to narrow. Alternate refinements of the two halves, then older dyadic spans."""

    def part(lag, scale, bits):
        return lag, scale, min(depths[scale], bits)

    stages = []
    recent = part(0, 0, depths[0])
    for bits in (3, 6, 10, depths[0]):
        stages.append([(part(0, 0, bits),)])
    if len(depths) > 1:
        # Intersect the two semantic half-spans in either direction. A broad recent
        # class lets distinct words share evidence about a more specific older half.
        for near, old in ((3, 6), (6, 10), (10, 6)):
            stages.append([(part(0, 0, near), part(1, 0, old))])
        for bits in (3, 6, 10, depths[0]):
            stages.append([(part(0, 1, bits),), (recent, part(1, 0, bits))])
        for j in range(1, len(depths) - 1):
            half = 1 << j
            for overlap in (False, True):
                if overlap:
                    stages.append([(recent, part(1, 0, 10), part(half, j, b)) for b in (3, 7)])
                else:
                    stages.append([(recent, part(half, j, b)) for b in (4, 10)])
    return stages
