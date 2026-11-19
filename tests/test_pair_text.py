import unittest

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptModel
from acan.lm.concept_lm import from_checkpoint_arrays
from acan.lm.pair_text import PairText


class PairTextTest(unittest.TestCase):
    def test_counted_tabular_masses(self):
        x = pd.DataFrame({"x": [0.0, 1.0, 8.0, 9.0]})
        n = np.array([1, 9, 3, 5])
        model = ConceptModel(max_concepts=1, covariance=False).fit_counted_numeric(x, n)
        self.assertEqual(model.rows, n.sum())
        self.assertAlmostEqual(model.statistics["C"].sum(), n.sum())
        self.assertAlmostEqual(model.encoder.mean[0], np.average(x.x, weights=n))
        for invalid in ([1, 0, 2, 3], [0.5, 1, 2, 3], [1, 2, 3]):
            with self.assertRaises(ValueError):
                model.fit_counted_numeric(x, invalid)

    def test_checkpoint_prefix_document_and_resume(self):
        model = PairText(11, 10, 1 << 20)
        ids = np.tile([1, 2, 3, 1, 2, 4, 10, 5, 6, 7, 1, 5, 7, 10], 24)
        tune = [ids[14:70], ids[70:126]]
        model.add(ids[:211], ids, tune, lambda: None)
        model.judge(tune)
        other = from_checkpoint_arrays(model.arrays())
        for obj in (model, other):
            obj.add(ids[211:], ids, tune, lambda: None)
            obj.judge(tune)
        q = model.logp(ids)
        np.testing.assert_allclose(q, other.logp(ids))
        np.testing.assert_allclose(q[:35], model.logp(ids[:35]))
        np.testing.assert_allclose(q[7:14], model.logp(ids[7:14]))
        self.assertEqual(model.tokens, len(ids))
        self.assertEqual(model.generate(ids, 4), other.generate(ids, 4))
        cursor = other.model.cursor()
        for i, t in enumerate(ids[:7]):
            p = cursor.distribution()
            self.assertAlmostEqual(p.sum(), 1.0)
            self.assertAlmostEqual(np.log(p[t]), q[i])
            cursor.observe(t)
        bulk = other.model.cursor(ids[:6])
        fork = bulk.fork()
        before = bulk.distribution()
        fork.observe(0)
        np.testing.assert_array_equal(bulk.distribution(), before)
        self.assertAlmostEqual(np.log(before[ids[6]]), q[6])


class QuestionColumnTest(unittest.TestCase):
    """The answer sees the last user turn: the same columns in bulk and while writing."""

    def test_question_columns_follow_turns_and_the_cursor_agrees_with_bulk(self):
        from acan.lm.pair_model import EncodedPrefix, question_columns
        y = np.array([5, 1, 9, 10, 11, 2, 1, 8, 20, 21, 2, 1, 9, 12, 2, 0, 1, 9, 13, 2])
        row = EncodedPrefix(y, np.array([0, 15]))
        row.spans['c8'] = np.arange(1, len(y)+1).astype(np.uint32)
        question_columns(row, (1, 9, 2), [8])
        # asked at 1..5, kept through the answer, replaced by the next question, gone in a new document
        np.testing.assert_array_equal(row.spans['q8'], [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 14, 0, 0, 0, 0, 19])

        rng = np.random.default_rng(0)
        start, user, end, assistant, eos = 11, 12, 13, 14, 15
        dialogues = []
        for _ in range(200):
            q = rng.integers(1, 6, 3)
            # A fixed preamble longer than every exact context, then the word the question asked for.
            dialogues += [start, user, *q, end, start, assistant, *[6]*10, q[-1]+5, end, eos]
        ids = np.array(dialogues)
        tune = [ids[:300], ids[300:600]]
        model = PairText(16, eos, 1 << 20, max_context=8, turns=(start, user, end))
        model.add(ids, ids, tune, lambda: None)
        model.judge(tune)
        self.assertTrue(any(f.startswith('q') for f in model.model.weights))      # the question was selected
        doc = ids[:ids.tolist().index(eos)+1]
        q = model.logp(doc)
        cursor = model.model.cursor()
        for i, t in enumerate(doc[:-1]):
            self.assertAlmostEqual(np.log(cursor.distribution()[t]), q[i])
            cursor.observe(t)
        restored = from_checkpoint_arrays(model.arrays())
        np.testing.assert_allclose(restored.logp(doc), q)
        self.assertEqual(restored.turns, (start, user, end))
