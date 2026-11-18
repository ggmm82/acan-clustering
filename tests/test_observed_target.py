"""La nuova categoria viene letta e aggiornata con le appartenenze del modello tabulare."""

import unittest
import numpy as np
import pandas as pd

from acan.concept_algebra import categorical_reading
from acan.core.concepts import ConceptModel


class ObservedTargetTests(unittest.TestCase):
    def test_streamed_counts_use_observed_memberships_and_the_public_prediction(self):
        x = pd.DataFrame({"misura": np.tile([0.0, 0.1, 0.8, 1.0], 15)})
        y = np.tile([0, 0, 1, 1], 15)
        graph = ConceptModel(max_concepts=12, covariance=False).fit(x)
        original = graph.assignments(x)
        graph.fit_observed_target(
            "esito",
            ["a", "b", "raro"],
            [(x.iloc[:13], y[:13]), (x.iloc[13:], y[13:])],
            [(x.iloc[::5], y[::5])],
        )
        expected = original.T @ np.eye(3)[y]
        prior = (expected.sum(0) + 1) / (len(y) + 3)
        k = graph.observed_targets["esito"]["kappa"]
        read = categorical_reading(
            expected, expected.sum(1)[:, None], (expected > 1e-9).sum(1)[:, None], prior, k, np
        )
        p, labels = graph.predict_proba(x.assign(esito="raro"), "esito")
        np.testing.assert_allclose(p, original @ read)
        np.testing.assert_array_equal(graph.predict(x, "esito"), np.array(labels)[p.argmax(1)])
        self.assertTrue(np.all(p[:, 2] > 0))
        np.testing.assert_allclose(p.sum(1), 1.0)
        with self.assertRaises(ValueError):
            graph.fit_observed_target("esito", labels, [(x, y + 3)], [(x, y)])
        np.testing.assert_array_equal(graph.predict_proba(x, "esito")[0], p)
        graph.fit(x)
        self.assertFalse(graph.observed_targets)

    def test_exact_category_refines_the_streamed_head_and_refitting_discards_old_counts(self):
        from acan.core.category_table import Table

        frame = pd.DataFrame({"x": np.tile([0.0, 0.2, 1.0], 16)})
        keys = np.tile([1, 2], 24)
        y = keys - 1
        graph = ConceptModel(max_concepts=8, covariance=False).fit(frame)
        graph.fit_observed_target("target", ["a", "b"], [(frame, y)], [(frame, y)])
        prior, _ = graph.predict_proba(frame, "target")
        table = Table.count(keys, y, 2)
        reading = graph.refine_category("target")
        _, (counts, smoothing, expected) = reading.propose(table, keys, y, prior[np.arange(len(y)), y])
        reading.levels.append(("key", counts, smoothing))
        posterior, categories = graph.predict_proba(frame.assign(key=keys), "target")
        self.assertEqual(categories, ["a", "b"])
        np.testing.assert_allclose(posterior[np.arange(len(y)), y], expected)
        np.testing.assert_allclose(posterior.sum(1), 1)
        np.testing.assert_allclose(graph.predict_proba(frame.assign(key=99), "target")[0], prior)
        graph.fit_observed_target("target", categories, [(frame, y)], [(frame, y)])
        np.testing.assert_allclose(graph.predict_proba(frame.assign(key=keys), "target")[0], prior)
