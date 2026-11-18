"""Invariants of the optional CP reading, for both scoring and generation."""

import unittest

import numpy as np

from acan.lm.concept_lm import ConceptLM
from acan.lm.context_refinement import ContextRefinement, SuffixCursor, pool


def fitted(ids):
    model = ConceptLM(9, depth=4, document=2, budget_bytes=1 << 20)
    for piece in np.array_split(ids, 11):
        model.add_context_piece(piece)
    model.settle()
    model.context_trained_at = model.tokens
    model.refinement = ContextRefinement(np.arange(9) // 3, [1.0, 2.0, 3.0], model.tokens).build(model)
    return model


class ContextRefinementTests(unittest.TestCase):
    def test_stream_counts_are_independent_of_io_boundaries(self):
        ids = np.random.default_rng(21).integers(0, 9, 300)
        streamed = fitted(ids)
        whole = ConceptLM(9, depth=4, document=2)
        whole.add(ids)
        whole.settle()
        for a, b in zip(streamed.levels, whole.levels):
            for x, y in zip(a.pairs(), b.pairs()):
                np.testing.assert_array_equal(x, y)
        for depth, coarse in streamed.refinement.tables.items():
            assert coarse.counts.sum() == streamed.levels[depth].counts.sum()
        # Checkpoint in the middle of a suffix, then append more evidence.
        restored = ConceptLM.from_arrays(streamed.arrays())
        for model in (streamed, restored):
            model.add_context_piece(ids[:19])
            model.settle()
        np.testing.assert_array_equal(restored.context_tail, streamed.context_tail)
        np.testing.assert_allclose(restored.logp(ids), streamed.logp(ids))
        same = pool(streamed.levels[2], streamed.refinement.mapping)
        np.testing.assert_array_equal(restored.refinement.tables[2].counts, same.counts)

    def test_distribution_is_causal_normalized_and_agrees_with_scoring(self):
        ids = np.tile([1, 4, 7, 2, 4, 8], 60)
        model = fitted(ids)
        probe = np.array([1, 4, 7, 2, 4, 8, 1, 4, 8])
        expected = model.logp(probe)
        cursor = model.prediction_cursor([])
        for t, y in enumerate(probe):
            p = cursor.distribution()
            np.testing.assert_allclose(p.sum(), 1.0)
            np.testing.assert_allclose(cursor.probability(y), p[y])
            np.testing.assert_allclose(np.log(p[y]), expected[t])
            np.testing.assert_allclose(model.distribution(probe[:t]), p)
            cursor.observe(y)
        assert model.distribution([1, 4]).argmax() == 7
        assert model.distribution([2, 4]).argmax() == 8
        # Force eviction to exercise the sequential validation path and optional smoothing.
        model.document_limit = lambda: 3
        cursor = SuffixCursor(model)
        a, b = 2 * model.a, 3 * model.b
        long = np.arange(90) % 9
        scores = model.logp(long, a=a, b=b)
        for y, score in zip(long, scores):
            np.testing.assert_allclose(np.log(cursor.probability(y, a, b)), score)
            cursor.observe(y)
            assert all(n <= 3 for n in cursor.cells[1:])

    def test_a_late_piece_can_change_the_learned_partition(self):
        from acan.lm.context_refinement import partitions

        model = ConceptLM(12, depth=2, document=0)
        model.add_context_piece(np.tile(np.arange(6), 60))
        model.settle()
        first = partitions(model)
        assert first and all(len(set(cut[6:])) == 1 for cut in first)
        model.add_context_piece(np.tile(np.arange(6, 12), 200))
        model.settle()
        assert any(len(set(cut[6:])) > 1 for cut in partitions(model))
