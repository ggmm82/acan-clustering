import pickle
import unittest

import numpy as np
import pandas as pd

from acan.core.category_pairs import CategoryPairs, judged_subconcepts
from acan.core.concepts import ConceptModel


class CategoryPairTest(unittest.TestCase):
    def test_order_missing_and_large_ids(self):
        a = np.array([0xFFFFFFFF, 1, 0, 7], dtype=np.uint32)
        b = np.array([1, 0xFFFFFFFF, 7, 0], dtype=np.uint32)
        encoder = CategoryPairs().fit(a, b)
        codes = encoder.transform(a, b)
        self.assertNotEqual(codes[0], codes[1])
        np.testing.assert_array_equal(codes[2:], [0, 0])
        self.assertEqual(encoder.transform(a[:1], a[:1])[0], 0)

    def test_judgment_and_retry_use_only_fitted_categories(self):
        p, c = np.tile([1, 1, 2, 2], 90), np.tile([1, 2, 1, 2], 90)
        y = (p == c).astype(int)
        accepted, report = judged_subconcepts((p, c, y), (p, c, y), 2)
        self.assertEqual(report["accepted"], 4)
        self.assertEqual(len(accepted), 4)
        frame = pd.DataFrame({"p": p.astype(str), "c": c.astype(str), "class": y.astype(str)})
        model = ConceptModel(covariance=False).fit(frame, labels=["class"])
        model.fit_category_pair("class", "p", "c", frame.iloc[:200], frame.iloc[200:])
        output, classes = model.predict_proba(frame.assign(**{"class": "wrong"}), "class")
        model.fit_category_pair("class", "p", "c", frame.iloc[:200], frame.iloc[200:])
        np.testing.assert_allclose(model.predict_proba(frame, "class")[0], output)
        np.testing.assert_allclose(output.sum(1), 1.0)
        np.testing.assert_allclose(
            model.predict_proba(frame.assign(p="unknown"), "class")[0].sum(1), 1.0
        )
        self.assertEqual(set(classes), {"0", "1"})

    def test_native_local_router_predicts_from_observations_alone(self):
        x = pd.DataFrame(np.random.default_rng(12).normal(size=(100, 2)), columns=["a", "b"])
        model = ConceptModel(max_concepts=12, covariance=False).fit(x)
        router = model.fit_subconcepts(x)
        codes = router.transform(x)
        np.testing.assert_array_equal(codes[:10], pickle.loads(pickle.dumps(router)).transform(x.iloc[:10]))
        self.assertTrue(np.all((0 <= codes) & (codes < router.concepts)))
