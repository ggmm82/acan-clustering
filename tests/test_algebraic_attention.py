"""The loop-free concept attention against a plain reference with loops, and causality."""
import importlib.util
import math
import unittest

TORCH = importlib.util.find_spec('torch') is not None


def reference(q, raw, v, frequency, window, directions, scale, variance):
    """The same reading written the obvious way: for every reader, its window and its concepts."""
    import torch
    from acan.lm.algebraic_attention import rotate
    b, heads, length, d = q.shape
    out = torch.zeros_like(q)
    positions = torch.arange(length, dtype=q.dtype)
    keys = rotate(raw, positions, frequency)
    bits = directions.shape[1]
    for bi in range(b):
        for h in range(heads):
            code = (((raw[bi, h] @ directions) > 0).long()*(2**torch.arange(bits))).sum(-1)
            for t in range(length):
                logits, values = [], []
                for s in range(max(0, t-window+1), t+1):
                    logits.append(scale*(q[bi, h, t]@keys[bi, h, s])); values.append(v[bi, h, s])
                n, level = max(0, t-window+1), 0
                while (n >> level) > 0:
                    if (n >> level) & 1:
                        start = (n >> (level+1)) << (level+1)
                        for k in range(2**bits):
                            members = [s for s in range(start, start+2**level) if code[s] == k]
                            if not members:
                                continue
                            idx = torch.tensor(members)
                            g, y = keys[bi, h, idx].mean(0), v[bi, h, idx].mean(0)
                            value = scale*(q[bi, h, t]@g) + math.log(len(members))
                            if variance:
                                spread = (keys[bi, h, idx]**2).mean(0) - g*g
                                value = value + .5*scale*scale*(q[bi, h, t]**2*spread.clamp(min=0)).sum()
                            logits.append(value); values.append(y)
                    level += 1
                weights = torch.stack(logits).softmax(0)
                out[bi, h, t] = weights@torch.stack(values)
    return out


@unittest.skipUnless(TORCH, 'Requires torch')
class AlgebraicAttentionTest(unittest.TestCase):
    def setUp(self):
        import torch
        torch.manual_seed(0)
        self.b, self.h, self.T, self.d = 2, 2, 37, 8
        self.q, self.raw, self.v = (torch.randn(self.b, self.h, self.T, self.d, dtype=torch.float64) for _ in range(3))
        self.frequency = 1./10000.**(torch.arange(0, self.d, 2, dtype=torch.float64)/self.d)
        self.directions = torch.randn(self.d, 2, dtype=torch.float64)

    def test_same_as_the_reference_with_loops(self):
        from acan.lm.algebraic_attention import concept_scan, concept_scan_blocks
        for variance in (False, True):
            for window in (1, 4, 9, 40):
                slow = reference(self.q, self.raw, self.v, self.frequency, window, self.directions, .3, variance)
                for fast in (concept_scan, concept_scan_blocks):
                    value = fast(self.q, self.raw, self.v, self.frequency, window, self.directions, .3, variance)
                    self.assertLess(float((value-slow).abs().max()), 1e-9, (fast.__name__, variance, window))

    def test_no_position_reads_the_future(self):
        import torch
        from acan.lm.algebraic_attention import concept_scan_blocks
        q, raw, v = (x.clone().requires_grad_() for x in (self.q, self.raw, self.v))
        out = concept_scan_blocks(q, raw, v, self.frequency, 4, self.directions, .3, True)
        out[:, :, 20].sum().backward()
        for grad in (q.grad, raw.grad, v.grad):
            self.assertEqual(float(grad[:, :, 21:].abs().sum()), 0.)
        self.assertGreater(float(raw.grad[:, :, :5].abs().sum()), 0.)            # the oldest tokens are read, through concepts

    def test_memory_of_a_reader_grows_with_the_logarithm(self):
        from acan.lm.algebraic_attention import fenwick_segments
        start, end, present = fenwick_segments(4096, 16, 'cpu')
        self.assertLessEqual(int(present.sum(1).max()), 12)                       # at most log2(4096) segments per reader
        n = 4096-16                                                               # the last reader: segments tile its old tokens
        lengths = ((end-start)*present)[-1]
        self.assertEqual(int(lengths.sum()), n)
