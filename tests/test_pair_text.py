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


class OwnWordsTest(unittest.TestCase):
    def test_the_words_the_model_writes_do_not_feed_the_memory_of_the_document(self):
        model = PairText(11, 10, 1 << 20)
        ids = np.tile([1, 2, 3, 1, 2, 4, 10, 5, 6, 7, 1, 5, 7, 10], 24)
        tune = [ids[14:70], ids[70:126]]
        model.add(ids, ids, tune, lambda: None)
        model.judge(tune)
        model.model.memory = model.model.memory or {'e1': 1.0}
        cursor = model.model.cursor(ids[:20])
        before = cursor.distribution()
        quiet = cursor.fork()
        quiet.observe(3, remember=False)
        self.assertEqual({f: dict(v) for f, v in quiet.overlay.items()}, {f: dict(v) for f, v in cursor.overlay.items()})
        loud = cursor.fork()
        loud.observe(3)
        self.assertNotEqual({f: dict(v) for f, v in loud.overlay.items()}, {f: dict(v) for f, v in cursor.overlay.items()})
        np.testing.assert_array_equal(cursor.distribution(), before)


class TopicTest(unittest.TestCase):
    """What the text talks about multiplies the reading; writing and scoring agree, and checkpoints keep it."""

    def test_topics_are_judged_read_by_multiplying_and_agree_token_by_token(self):
        rng = np.random.default_rng(1)
        eos, docs = 40, []
        for i in range(80):                                    # two subjects with their own words, shared grammar
            words = np.arange(1, 20) if i % 2 else np.arange(20, 39)
            body = []
            for _ in range(12):
                body += [int(rng.choice(words)), int(rng.choice(words)), 39]   # 39: a full stop in both
            docs += body + [eos]
        ids = np.array(docs)
        tune = [ids[:370], ids[370:740]]
        model = PairText(41, eos, 1 << 22, max_context=8)
        model.add(ids, ids, tune, lambda: None)
        model.judge(tune)
        self.assertTrue(model.model.topics)                     # the scales have topic concepts
        for f in model.model.topics:
            model.model.topics[f] = [4.0, 1.0]                 # read them, to check the reading itself
        doc = ids[:ids.tolist().index(eos)+1]
        q = model.logp(doc)
        cursor = model.model.cursor()
        for i, t in enumerate(doc[:-1]):
            p = cursor.distribution()
            self.assertAlmostEqual(p.sum(), 1.0)
            self.assertAlmostEqual(np.log(p[t]), q[i])
            cursor.observe(t)
        restored = from_checkpoint_arrays(model.arrays())
        np.testing.assert_allclose(restored.logp(doc), q)
        self.assertEqual(restored.model.topics, model.model.topics)


class StructureReportTest(unittest.TestCase):
    def test_the_part_of_the_structure_done_never_goes_back(self):
        from acan.vis.long_training import structure_progress
        rng = np.random.default_rng(2)
        eos, docs = 40, []
        for i in range(60):
            words = np.arange(1, 20) if i % 2 else np.arange(20, 39)
            docs += [int(rng.choice(words)) for _ in range(30)] + [eos]
        ids = np.array(docs)
        seen = []
        model = PairText(41, eos, 1 << 22, max_context=8)
        model.report = lambda phase, width, seconds, **facts: seen.append(
            (phase, structure_progress(dict(phase=phase, width=width, seconds=seconds, **facts))[0]))
        model.add(ids, ids, [ids[:300], ids[300:600]], lambda: None)
        parts = [part for _, part in seen]
        self.assertEqual(parts, sorted(parts))                  # the bar only moves forward
        self.assertEqual(seen[-1], ('done', 1.))
        self.assertIn('topics', {phase for phase, _ in seen})
