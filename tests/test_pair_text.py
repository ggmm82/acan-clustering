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
