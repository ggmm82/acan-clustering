"""Raffinamenti CP con lettura tabulare, lettura incrementale e confini di documento."""

import pickle
import unittest

import numpy as np
import pandas as pd

from acan.core.category_table import SharedProfiles, Table
from acan.core.concepts import ConceptModel
from acan.core.conditional import SemanticCuts
from experiments.text_table.conditional import SpanGeometry
from experiments.text_table.conditional_cursor import ConditionalText
from experiments.text_table.conditional_memory import probabilities
from experiments.text_table.conditional_run import Rows


class ConditionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        centers = rng.normal(size=(6, 3))
        centers[5] = centers[0]
        cls.tokens = rng.integers(0, 6, 240)
        cls.starts = np.array([0, 100, 175])
        cls.geometry = SpanGeometry(centers, scales=4)
        columns = cls.geometry.fit(cls.tokens, cls.starts, sample=200, precision=4)
        rows = Rows(cls.geometry, cls.tokens, cls.starts, np.arange(240), columns)
        graph = ConceptModel(max_concepts=1, covariance=False).fit(
            pd.DataFrame({"token": [str(i) for i in cls.tokens]}), labels=["token"]
        )
        r = graph.refine_category("token")
        features = [((0, 0, 3),), ((0, 0, 3), (1, 0, 3)), ((2, 1, 3),)]
        for f in features:
            keys = rows.keys(f)
            ok = keys != 0
            r.levels.append((f, Table.count(keys[ok], rows.y[ok], 6), 2.0))
        cls.model = ConditionalText(cls.geometry, graph, [((), 2.0), (features[1], 1.0)])

    def test_missing_and_unseen_profiles_follow_the_same_numeric_rule(self):
        g = self.geometry
        self.assertEqual(g.token_codes[0], g.token_codes[5])
        dead = SemanticCuts().fit(np.zeros((4, 3)), np.ones(4), 4)
        self.assertEqual(len(np.unique(dead.encode(np.ones((7, 3))))), 1)
        a = self.tokens.copy()
        a[25:] = (a[25:] + 1) % 6
        first = g.transform(a, self.starts)
        original = g.transform(self.tokens, self.starts)
        for i, (x, y) in enumerate(zip(first, original)):
            np.testing.assert_array_equal(x[:25], y[:25])
            for start in self.starts[1:]:
                np.testing.assert_array_equal(y[start : start + (1 << i) - 1], 0)

    def test_online_chunked_public_api_and_serialization(self):
        model = pickle.loads(pickle.dumps(self.model))
        tokens = self.tokens[:35]
        rows = Rows(model.geometry, tokens, np.array([0]), np.arange(len(tokens)))
        p = probabilities(model.reading, model.memory, rows)
        cursor = model.cursor()
        fs = [f for f, t, k in model.reading.levels]
        frame = pd.DataFrame(
            np.column_stack([rows.keys(f) for f in fs]), columns=pd.Index(fs, tupleize_cols=False)
        )
        native, _ = model.graph.predict_proba(frame, "token")
        frame["token"] = "wrong"
        np.testing.assert_allclose(native, model.graph.predict_proba(frame, "token")[0])
        for i, y in enumerate(tokens):
            np.testing.assert_allclose(cursor.distribution().sum(), 1.0, atol=1e-10)
            self.assertAlmostEqual(p[i], cursor.distribution()[y])
            np.testing.assert_allclose(model.reading.distribution(cursor.current.__getitem__), native[i])
            np.testing.assert_allclose(
                model.cursor().prefill(tokens[:i]).distribution(), cursor.distribution()
            )
            cursor.observe(y)
        bulk = model.cursor().prefill(tokens[:17]).prefill(tokens[17:])
        np.testing.assert_allclose(bulk.distribution(), cursor.distribution())
        clone = bulk.fork()
        state = bulk.distribution().copy()
        clone.observe(3)
        np.testing.assert_allclose(state, bulk.distribution())

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
        graph = pickle.loads(pickle.dumps(self.model.graph))
        frame = pd.DataFrame({"x": [0, 0, 1, 1] * 10, "token": ["a", "b", "a", "b"] * 10})
        graph.fit(frame, labels=["token"])
        self.assertEqual(graph.category_regions, {})
        before = graph.predict_proba(frame, "token")[0]
        del graph.category_regions
        np.testing.assert_array_equal(before, graph.predict_proba(frame, "token")[0])


class WordRegionTests(unittest.TestCase):
    def test_synonym_composed_of_two_pieces_recalls_a_previously_observed_answer(self):
        from experiments.text_table.word_regions import WordDocument, WordProfiles

        pieces = ["Ġalpha", "Ġbeta", "Ġmedic", "Ġsays", "Ġanswer", "Ġphys", "ician", "[EOS]"]

        class Tokenizer:
            def id_to_token(self, i):
                return pieces[i]

            def get_vocab_size(self):
                return len(pieces)

        index = {s: i for i, s in enumerate(("alpha", "beta", "medic", "says", "answer", "physician"))}
        vectors = np.eye(len(index))
        vectors[5] = vectors[2]
        profiles = WordProfiles(index, vectors, np.arange(len(index)), Tokenizer())
        profiles.weights = [(2, 1.0, 0.25)]
        prior = np.full(len(pieces), 1 / len(pieces))
        doc = WordDocument(profiles, prior, window=32)
        # The first query only learns the answer after it has been consumed.
        for y in [0, 1, 2, 3]:
            doc.observe(y)
        np.testing.assert_array_equal(doc.distribution(prior), prior)
        doc.observe(4)
        other = doc.fork()
        for y in [7, 0, 1, 5, 6, 3]:
            other.observe(y)
        self.assertEqual(other.distribution(prior).argmax(), 4)
        self.assertAlmostEqual(other.distribution(prior).sum(), 1.0)
        parent = doc.distribution(prior).copy()
        other.observe(0)
        np.testing.assert_array_equal(parent, doc.distribution(prior))
