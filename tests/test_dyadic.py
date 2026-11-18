"""Causalità e parità delle tabelle diadiche con la lettura tabulare pubblica."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptModel
from experiments.text_table.dyadic import Dyadic, TARGET, frame, names


class DyadicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.vectors = rng.normal(size=(10, 3))
        cls.ids = np.tile(np.array([1, 3, 6, 2, 3, 8, 5, 7, 9, 0, 4]), 41)
        cls.starts = np.array([0, 143])
        cls.model = Dyadic(cls.vectors, scales=5).fit(cls.ids, cls.starts, pair_rows=150, target_rows=280)

    def test_scale_endpoints_and_categorical_target_are_causal(self):
        model = self.model
        ids = self.ids[:48].copy()
        starts = np.array([0, 19])
        levels, age = model.levels(ids, starts)
        for j, pair in enumerate(model.pairs, 1):
            where = np.flatnonzero(age >= 1 << j)
            self.assertTrue(np.all(levels[j][age < 1 << j] == -1))
            child = levels[j - 1]
            x, c = pair.inputs(child[where - (1 << (j - 1))], child[where])
            expected = pair.graph.assignments(frame(x, c, pair.cols)).argmax(1)
            np.testing.assert_array_equal(levels[j][where], expected)
        t = 26
        x, c = model.context(levels, np.array([t]), age)
        ids[t:] = (ids[t:] + 2) % 10
        altered, age1 = model.levels(ids, starts)
        xx, cc = model.context(altered, np.array([t]), age1)
        np.testing.assert_array_equal(x, xx)
        np.testing.assert_array_equal(c, cc)
        _, missing = model.context(levels, np.array([19, 20]), age)
        self.assertTrue((missing == -1).all())

    def test_cursor_matches_public_tabular_target_after_every_token(self):
        model = self.model
        ids = self.ids[:57]
        levels, age = model.levels(np.r_[ids, 0])
        x, c = model.context(levels, np.arange(len(ids) + 1), age)
        actual, labels = model.graph.predict_proba(frame(x, c, model.columns), TARGET)
        self.assertEqual(labels, names("t", len(model.vectors)))
        cursor = model.cursor()
        for i, token in enumerate(ids):
            np.testing.assert_allclose(cursor.distribution(), actual[i], atol=1e-10)
            cursor.observe(token)
        np.testing.assert_allclose(cursor.distribution(), actual[-1], atol=1e-10)
        vectorized = model.cursor().prefill(ids[:11]).prefill(ids[11:])
        np.testing.assert_array_equal(vectorized.last, cursor.last)
        for a, b in zip(vectorized.buffers, cursor.buffers):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(cursor.distribution().sum(), 1.0)

    def test_full_pass_matches_the_model_sufficient_statistics(self):
        ids = self.ids
        m = Dyadic(self.vectors, scales=2)
        with patch.object(Dyadic, "finish", lambda *args: None):
            m.fit(ids, self.starts, pair_rows=128, target_rows=120)
        expected = deepcopy(m.graph)
        levels, age = m.levels(ids, self.starts)
        x, c = m.context(levels, np.arange(len(ids)), age)
        rows = frame(x, c, m.columns)
        rows[TARGET] = pd.Categorical.from_codes(ids, categories=names("t", 10))
        Dyadic.complete_dictionary(expected, ids, 10)
        expected._finish(lambda: iter([rows]))
        m.finish(ids, age, levels)
        for key in expected.statistics:
            np.testing.assert_allclose(
                m.graph.statistics[key], expected.statistics[key], rtol=1e-8, atol=1e-8
            )


class JudgmentTests(unittest.TestCase):
    def test_categorical_deletion_agrees_with_literal_dense_counts(self):
        from acan.core.concepts import Encoder, _softmax

        rng = np.random.default_rng(22)
        df = pd.DataFrame(
            {
                "x": rng.normal(size=31),
                "feature": rng.choice(["m", "n", None], 31),
                "read": rng.choice(["a", "b", "c", None], 31),
            }
        )
        m = ConceptModel(covariance=False)
        m.encoder = Encoder().fit(df)
        m.labels = ["read"]
        data = m.encoder.transform(df)
        m._roots(data)
        s = m._statistics(rng.dirichlet(np.ones(4), len(df)), data)
        m.statistics = s
        for distinct in (False, True):
            m.distinct = distinct
            w = m._weights(2)
            route = m._route(s, data, w)[0]
            expected = []
            for i, (x, mask, onehot) in enumerate(zip(*data)):
                r = route[i][:, None]
                rest = dict(
                    C=s["C"] - r[:, 0],
                    Cm=np.maximum(s["Cm"] - r * mask, 0),
                    U=s["U"] - r * x,
                    Q=np.maximum(s["Q"] - r * x * x, 0),
                    N=np.maximum(s["N"] - r * onehot, 0),
                )
                g, v, p = m._parameters(rest, m._prior(s))
                num = (-0.5 * ((x - g) ** 2 / v + np.log(v) + np.log(2 * np.pi)) * mask).sum(1)
                f, y = m._blocks()
                membership = _softmax((np.log(rest["C"]) + num + p[:, f] @ onehot[f])[None])[0][0]
                expected.append(np.log(membership @ np.exp(p[:, y] @ onehot[y])))
            np.testing.assert_allclose(m._score(s, data, w, np.arange(len(df))), expected, atol=1e-11)
