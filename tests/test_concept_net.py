import importlib.util
import unittest

import numpy as np

from acan.lm.concept_net import ConceptNet

TORCH = importlib.util.find_spec('torch') is not None


def config(**kw):
    return dict(dict(vocabulary=50, dim=16, layers=2, heads=2, concepts=8, window=3), **kw)


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class ConceptNetTest(unittest.TestCase):
    def network(self, grouped):
        import torch
        from acan.lm.concept_net_training import Network, output_groups
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        groups = output_groups(rng.integers(0, 4, 50), rng.random(50), limit=6) if grouped else None
        net = Network(config(), centres=rng.normal(size=(50, 8)), frequency=rng.random(50) + .1, groups=groups)
        with torch.no_grad():                                  # parameters away from their start
            for p in net.parameters():
                p.add_(torch.randn_like(p) * .1)
        return net

    def test_numpy_reads_as_torch_and_the_cursor_as_the_whole_document(self):
        import torch
        ids = np.random.default_rng(1).integers(0, 50, 23)
        for grouped in (False, True):
            net = self.network(grouped)
            x = torch.as_tensor(ids[None, :-1])
            with torch.no_grad():
                expected = net.target_log_probs(x, torch.ones_like(x, dtype=torch.float32), torch.as_tensor(ids[None, 1:]))[0].numpy()
            reading = net.export()
            whole = reading.document(ids)
            np.testing.assert_allclose(whole.sum(1), 1, rtol=1e-6)
            np.testing.assert_allclose(np.log(whole[np.arange(len(ids) - 1), ids[1:]]), expected, atol=2e-4)
            cursor = reading.cursor()
            for i, token in enumerate(ids[:-1]):
                cursor.observe(token)
                np.testing.assert_allclose(cursor.distribution(), whole[i], atol=1e-5)

    def test_saved_and_restored_network_reads_the_same_and_goes_on_learning(self):
        from acan.lm.concept_net_training import Learner, Network
        net = self.network(True)
        reading = net.export()
        again = ConceptNet.from_arrays(reading.arrays())
        ids = np.r_[np.random.default_rng(2).integers(0, 49, 40), 49]
        np.testing.assert_allclose(again.document(ids[:-1]), reading.document(ids[:-1]), atol=1e-6)
        restored = Network.restore(again)
        before = restored.export().document(ids[:-1])
        loss = Learner(restored, tokens=64).learn(np.tile(ids, 20), eos=49)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.allclose(restored.export().document(ids[:-1]), before))


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class PairTextWithNetworkTest(unittest.TestCase):
    def test_the_network_starts_the_chain_in_bulk_as_token_by_token_and_after_saving(self):
        from acan.lm.concept_lm import from_checkpoint_arrays
        from acan.lm.pair_text import PairText
        rng = np.random.default_rng(1)
        eos, docs = 40, []
        for i in range(80):
            words = np.arange(1, 20) if i % 2 else np.arange(20, 39)
            body = []
            for _ in range(12):
                body += [int(rng.choice(words)), int(rng.choice(words)), 39]
            docs += body + [eos]
        ids = np.array(docs)
        tune = [ids[:370], ids[370:740]]
        model = PairText(41, eos, 1 << 22, max_context=8, network=dict(dim=16, layers=2, heads=2, concepts=8))
        model.add(ids, ids, tune, lambda: None)
        model.judge(tune)
        self.assertIsNotNone(model.model.net)
        self.assertTrue(np.isfinite(model.network_loss))
        doc = ids[:ids.tolist().index(eos) + 1]
        q = model.logp(doc)
        cursor = model.model.cursor()
        for i, t in enumerate(doc[:-1]):
            p = cursor.distribution()
            self.assertAlmostEqual(p.sum(), 1.0)
            self.assertAlmostEqual(np.log(p[t]), q[i], places=5)
            cursor.observe(t)
        prefilled = model.model.cursor(doc[:10]).distribution()
        stepped = model.model.cursor()
        for t in doc[:10]:
            stepped.observe(t)
        np.testing.assert_allclose(prefilled, stepped.distribution(), atol=1e-6)
        restored = from_checkpoint_arrays(model.arrays())
        np.testing.assert_allclose(restored.logp(doc), q, atol=1e-6)


@unittest.skipUnless(TORCH, 'the queue options load the trainer, which needs torch')
class QueueOptionTest(unittest.TestCase):
    def test_the_queue_can_learn_the_network_and_refuses_anything_but_a_choice(self):
        from acan.lm.research_options import validate_options
        base = dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-pairs-v1')
        self.assertFalse(validate_options(base)['concept_network'])
        self.assertTrue(validate_options(dict(base, concept_network=True))['concept_network'])
        with self.assertRaises(ValueError):
            validate_options(dict(base, concept_network='sì'))


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class DescribeTest(unittest.TestCase):
    def test_every_layer_says_how_many_concepts_work_and_shows_examples(self):
        from acan.lm.concept_net_training import Network
        net = Network(config()).export()
        docs = [np.random.default_rng(i).integers(0, 50, 60) for i in range(5)]
        described = net.describe(docs, decode=lambda ids: ' '.join(map(str, ids)))
        self.assertEqual([layer['layer'] for layer in described], [1, 2])
        for layer in described:
            self.assertEqual(len(layer['active']), 2)
            self.assertTrue(all(0 < a <= 8 for a in layer['active']))
            self.assertTrue(layer['concepts'] and all(c['examples'] for c in layer['concepts']))
