import json
import unittest
import numpy as np

from acan.generative import ConceptLanguageModel


class TestLanguagePrototype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ConceptLanguageModel(context=4, slots=8, seed=42).fit(
            '\n'.join('ab' * i for i in range(10, 35)), epochs=15)

    def test_held_out_sequence_learning_against_unigram(self):
        report = self.model.report
        self.assertLess(report['test']['perplexity'], report['unigram_perplexity'] * .8)
        self.assertFalse(report['test_used_for_selection'])
        self.assertNotEqual(report['train_sha256'], report['test_sha256'])
        self.assertGreater(report['test']['tokens'], 0)
        # Repeated grammar windows are disclosed, not described as novel contexts.
        self.assertGreater(report['train_test_window_overlap'], 0)

    def test_probability_mass_causal_context_and_gradient(self):
        model = self.model
        x, y = model._windows(['abba'])
        probabilities, _ = model._forward(x)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-12)
        self.assertTrue((probabilities > 0).all())
        self.assertEqual(x[0].tolist(), [model.bos] * 4)
        self.assertEqual(x[1, -1], model.lookup['a'])
        self.assertEqual(y[-1], 1)  # EOS is scored as a real next-token target.
        _, gradients = model.loss_and_grad(x, y)
        for key in model.params:
            for index in list(np.ndindex(model.params[key].shape))[:6]:
                old, eps = model.params[key][index], 1e-5
                model.params[key][index] = old + eps
                a = model.loss_and_grad(x, y)[0]
                model.params[key][index] = old - eps
                b = model.loss_and_grad(x, y)[0]
                model.params[key][index] = old
                self.assertAlmostEqual(gradients[key][index], (a - b) / (2 * eps), delta=2e-6)

    def test_seeded_generation_and_model_roundtrip(self):
        original = self.model.generate('a', length=30, seed=7)
        restored = ConceptLanguageModel.from_json(self.model.to_json())
        self.assertEqual(restored.generate('a', length=30, seed=7), original)
        self.assertGreater(len(original[1]), 0)
        self.assertTrue(all(entry['concept'] < self.model.slots for entry in original[1]))

    def test_no_held_out_vocabulary_leakage(self):
        corpus = 'ab' * 200 + 'Ω' * 60
        model = ConceptLanguageModel(context=2, slots=4).fit(corpus, epochs=2)
        self.assertNotIn('Ω', model.chars)
        self.assertGreater(model.report['test']['unknown_tokens'], 0)
        self.assertTrue(np.isfinite(model.report['test']['perplexity']))

    def test_cancellation_and_invalid_models(self):
        with self.assertRaisesRegex(ValueError, 'cancelled'):
            ConceptLanguageModel().fit('ab' * 100, cancelled=lambda: True)
        value = json.loads(self.model.to_json())
        value['parameters']['mass'] = [float('nan')] * self.model.slots
        with self.assertRaises(ValueError):
            ConceptLanguageModel.from_json(json.dumps(value))
        with self.assertRaises(ValueError):
            ConceptLanguageModel(context=10000)
