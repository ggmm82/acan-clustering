import unittest

import numpy as np
import pandas as pd

from acan.core.category_table import SharedProfiles, Table
from acan.core.concepts import ConceptModel


class ConditionalTests(unittest.TestCase):
    def test_equivalent_histograms_share_predictions_and_chunk_sums(self):
        keys = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 6], dtype=np.uint64)
        ys = np.array([4, 3, 3, 3, 3, 2, 1, 2, 2, 2, 2])
        tab = Table.count_regions(((keys[i : i + 3], ys[i : i + 3]) for i in range(0, len(keys), 3)), 6)
        packed = SharedProfiles(tab)
        self.assertLess(len(packed.C), len(tab.C))
        expected = Table.count(keys[1:], ys[1:], 6)
        region, target = np.indices((8, 6))
        for a, b in zip(
            packed.lookup(region.ravel(), target.ravel()), expected.lookup(region.ravel(), target.ravel())
        ):
            np.testing.assert_array_equal(a, b)
        for region in range(1, 7):
            for a, b in zip(
                packed.followers(packed.find([region])[0]), expected.followers(expected.find([region])[0])
            ):
                np.testing.assert_array_equal(a, b)

    def test_refit_and_old_graph_without_refinements(self):
        frame = pd.DataFrame({"x": [0, 0, 1, 1] * 10, "token": ["a", "b", "a", "b"] * 10})
        graph = ConceptModel(max_concepts=1).fit(frame, labels=["token"])
        graph.refine_category("token")
        graph.fit(frame, labels=["token"])
        self.assertEqual(graph.category_regions, {})
        before = graph.predict_proba(frame, "token")[0]
        del graph.category_regions
        np.testing.assert_array_equal(before, graph.predict_proba(frame, "token")[0])
