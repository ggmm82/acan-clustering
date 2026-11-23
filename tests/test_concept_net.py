import importlib.util
import unittest

import numpy as np

from acan.lm.concept_net import ConceptNet

TORCH = importlib.util.find_spec('torch') is not None


def config(**kw):
    return dict(dict(vocabulary=50, dim=16, layers=2, heads=2, concepts=8, window=3), **kw)


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class ConceptNetTest(unittest.TestCase):
    def network(self, grouped, sentinel=True):
        import torch
        from acan.lm.concept_net_training import Network, output_groups
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        groups = output_groups(rng.integers(0, 4, 50), rng.random(50), limit=6) if grouped else None
        net = Network(dict(config(), sentinel=sentinel), centres=rng.normal(size=(50, 8)), frequency=rng.random(50) + .1, groups=groups)
        with torch.no_grad():                                  # parameters away from their start
            for p in net.parameters():
                p.add_(torch.randn_like(p) * .1)
        return net

    def test_cuda_probe_rejects_a_visible_device_without_kernels(self):
        from unittest.mock import patch

        from acan.lm.resources import cuda_usable

        cuda_usable.cache_clear()
        self.addCleanup(cuda_usable.cache_clear)
        with patch('torch.cuda.is_available', return_value=True), patch(
            'torch.ones', side_effect=RuntimeError('no kernel image')
        ) as kernel, self.assertLogs('acan.lm.resources', level='WARNING'):
            self.assertFalse(cuda_usable())
            self.assertFalse(cuda_usable())
            kernel.assert_called_once_with(1, device='cuda')

    def test_cuda_training_growth_and_cpu_export(self):
        import torch
        from acan.lm.concept_net_training import Learner, Network, held_out

        if not torch.cuda.is_available():
            self.skipTest('requires a CUDA GPU')
        net = self.network(grouped=True).to('cuda')
        docs = [np.arange(1, 17), np.arange(2, 12)]
        learner = Learner(net, tokens=32)
        self.assertTrue(np.isfinite(learner.learn_documents(docs)))
        grown = net.widened().more_heads()
        grown.add_layer()
        self.assertTrue(all(p.is_cuda for p in grown.parameters()))
        resumed = Network.restore(grown.export())
        self.assertEqual(resumed.device.type, 'cpu')
        np.testing.assert_allclose(held_out(grown, docs), held_out(resumed, docs), atol=3e-5)

    def test_numpy_reads_as_torch_and_the_cursor_as_the_whole_document(self):
        import torch
        ids = np.random.default_rng(1).integers(0, 50, 23)
        for grouped, sentinel in ((False, True), (True, True), (False, False)):   # (False, False): an older network's valve
            net = self.network(grouped, sentinel)
            x = torch.as_tensor(ids[None, :-1])
            with torch.no_grad():
                expected = net.target_log_probs(x, torch.ones_like(x, dtype=torch.float32), torch.as_tensor(ids[None, 1:]))[0].numpy()
            reading = net.export()
            np.testing.assert_allclose(np.log(reading.targets(ids)), expected, atol=2e-4)
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
        model = PairText(41, eos, 1 << 22, max_context=8, network=dict(dim=16, layers=2, heads=2, concepts=8, passes=False))
        model.add(ids, ids, tune, lambda: None)
        model.judge(tune)
        self.assertIsNotNone(model.model.net)
        self.assertTrue(model.network_history is not None)
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
    def test_the_graph_names_the_concepts_and_keeps_edges_that_say_something(self):
        net = ConceptNetTest.network(self, False).export()
        docs = [np.random.default_rng(i).integers(0, 50, 40) for i in range(4)]
        g = net.graph(docs)
        ids = {n['id'] for n in g['nodes']}
        self.assertTrue(g['nodes'] and all(n['enters'] and n['follows'] for n in g['nodes']))
        self.assertTrue(all(e['source'] in ids and e['target'] in ids for e in g['edges']))
        self.assertTrue(all(e['kind'] in ('legge', 'porta a') for e in g['edges']))
        from acan.vis.long_training import network_graph_view
        self.assertIsNotNone(network_graph_view(g))

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


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class GrowthTest(unittest.TestCase):
    def test_more_concepts_or_one_more_layer_read_as_before_and_go_on_learning(self):
        import torch
        from acan.lm.concept_net_training import Learner, Network
        torch.manual_seed(0)
        net = Network(config(layers=1, concepts=4))
        ids = np.r_[np.random.default_rng(3).integers(0, 49, 60), 49]
        Learner(net, tokens=64).learn(np.tile(ids, 5), eos=49)
        with torch.no_grad():                                  # normalizations far from the identity
            for norm in (net.layers[0].norm2,):
                norm.weight.mul_(3.); norm.bias.add_(.5)
        before = net.export().document(ids[:-1])
        net.split_concepts(0)
        self.assertEqual(net.config['concepts'], [8])
        split = net.export().document(ids[:-1])
        np.testing.assert_allclose(split, before, atol=2e-3)      # the halves apart: nearly the same reading
        net.add_layer()
        self.assertEqual((net.config['layers'], net.config['concepts']), (2, [8, 8]))
        grown = net.export()
        np.testing.assert_allclose(grown.document(ids[:-1]), split, atol=1e-5)   # a closed layer: exactly the same
        cursor = grown.cursor()
        for t in ids[:10]:
            cursor.observe(t)
        np.testing.assert_allclose(cursor.distribution(), grown.document(ids[:11])[9], atol=1e-5)
        again = Network.restore(grown)
        self.assertTrue(np.isfinite(Learner(again, tokens=64).learn(np.tile(ids, 5), eos=49)))

    def test_split_a_single_concept_preserves_reading_and_optimizer_slices(self):
        import copy
        import torch
        from acan.lm.concept_net_training import Learner, Network

        net = Network(config(dim=8, layers=2, heads=2, concepts=3, window=1))
        tokens = np.tile(np.r_[np.arange(1, 10), 49], 4)
        base = Learner(net, tokens=32)
        base.learn(tokens, eos=49)
        selected = torch.tensor([1, 0])
        grown = copy.deepcopy(net)
        grown.split_one(1, selected, noise=0)
        self.assertEqual(grown.config['concepts'], [3, 4])
        np.testing.assert_allclose(grown.export().document(tokens[:10]), net.export().document(tokens[:10]),
                                   atol=1e-6)
        trial = Learner(grown, tokens=32)
        trial.carry(base, 'concetti', (1, selected))
        heads = torch.arange(2)
        old = base.opt.state[net.layers[1].g]['exp_avg']
        new = trial.opt.state[grown.layers[1].g]['exp_avg']
        torch.testing.assert_close(new[:, :-1], old)
        torch.testing.assert_close(new[:, -1], old[heads, selected])
        self.assertTrue(np.isfinite(trial.learn(tokens, eos=49)))

    def test_added_heads_keep_old_adam_moments_and_start_with_zero_output(self):
        import torch
        from acan.lm.concept_net_training import Learner, Network

        net = Network(config(dim=8, layers=1, heads=1, concepts=2, window=1))
        ids = np.tile(np.r_[np.arange(1, 10), 49], 4)
        base = Learner(net, tokens=32)
        base.learn(ids, eos=49)
        grown = net.more_heads()
        trial = Learner(grown, tokens=32)
        trial.carry(base, 'teste')
        np.testing.assert_allclose(grown.export().document(ids[:10]), net.export().document(ids[:10]), atol=1e-6)
        for name in ('g', 'log_s', 'log_c', 'A'):
            old = base.opt.state[getattr(net.layers[0], name)]['exp_avg']
            new = trial.opt.state[getattr(grown.layers[0], name)]['exp_avg']
            torch.testing.assert_close(new[:1], old)
            self.assertEqual(new[1:].count_nonzero().item(), 0)
        for moment in ('exp_avg', 'exp_avg_sq'):
            old = base.opt.state[net.layers[0].out.weight][moment]
            new = trial.opt.state[grown.layers[0].out.weight][moment]
            torch.testing.assert_close(new[:, :8], old[:, :8])
            torch.testing.assert_close(new[:, 16:], old[:, 8:])
            self.assertEqual(new[:, 8:16].count_nonzero().item(), 0)
        self.assertTrue(np.isfinite(trial.learn(ids, eos=49)))


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class CalendarTest(unittest.TestCase):
    def test_decisions_double_and_a_growth_wins_only_if_it_reads_better(self):
        import torch
        from acan.lm.concept_net_training import Grower, Network, held_out
        torch.manual_seed(0)
        rng = np.random.default_rng(4)
        docs = []
        for i in range(300):                                   # two kinds of documents: something to learn
            words = np.arange(1, 25) if i % 2 else np.arange(25, 49)
            docs += list(rng.choice(words, 30)) + [49]
        ids, held = np.array(docs[:-1500]), np.array(docs[-1500:])

        held_docs = [d for d in np.split(held, np.flatnonzero(held == 49) + 1) if len(d) > 2]
        score = lambda network, tokens=None: held_out(network, held_docs, tokens)
        grower = Grower(Network(config(dim=4, heads=1, layers=1, concepts=2, window=1)), trial=512, passes=False, tokens=128)
        for start in range(0, len(ids), 2000):
            grower.learn(ids[start:start + 2000], 49, score)
        decisions = [h['tokens'] for h in grower.history]
        self.assertGreaterEqual(len(decisions), 3)
        self.assertTrue(all(b > a for a, b in zip(decisions, decisions[1:])))
        self.assertTrue(all(h['choice'] in ("com'è", *Grower.KINDS) for h in grower.history))
        self.assertIsNotNone(grower.best)


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class DirectionsTest(unittest.TestCase):
    def test_wider_more_groups_and_a_longer_window_read_as_before(self):
        import torch
        from acan.lm.concept_net_training import Network, output_groups
        torch.manual_seed(0)
        rng = np.random.default_rng(5)
        ids = rng.integers(0, 50, 30)
        for grouped in (False, True):
            groups = output_groups(rng.integers(0, 4, 50), rng.random(50), limit=6) if grouped else None
            net = Network(config(dim=4, layers=2, heads=1, concepts=2, window=1), groups=groups)
            with torch.no_grad():
                for p in net.parameters():
                    p.add_(torch.randn_like(p) * .3)
            before = net.export().document(ids)
            for grown, change in ((net.widened(), dict(dim=8, head_dim=8)), (net.more_heads(), dict(heads=2)),
                                  (net.wider_window(), dict(window=2))):
                self.assertTrue(all(grown.config[k] == v for k, v in change.items()))
                reading = grown.export()
                np.testing.assert_allclose(reading.document(ids), before, atol=2e-3)
                cursor = reading.cursor()
                for t in ids[:-1]:
                    cursor.observe(t)
                np.testing.assert_allclose(cursor.distribution(), reading.document(ids)[-1], atol=1e-5)


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class PassesTest(unittest.TestCase):
    def test_a_small_piece_is_read_again_while_it_helps_and_the_last_pass_is_undone(self):
        import torch
        from acan.lm.concept_net_training import Grower, Network, held_out
        torch.manual_seed(0)
        rng = np.random.default_rng(6)
        docs = []
        for i in range(60):
            words = np.arange(1, 25) if i % 2 else np.arange(25, 49)
            docs += list(rng.choice(words, 8)) + [49]
        ids, held = np.array(docs[:450]), np.array(docs[450:])
        held_docs = [d for d in np.split(held, np.flatnonzero(held == 49) + 1) if len(d) > 2]
        score = lambda network, tokens=None: held_out(network, held_docs, tokens)
        grower = Grower(Network(config(dim=4, layers=1, heads=1, concepts=2, window=1)), trial=256, tokens=256)
        Grower.KINDS, kinds = ('width',), Grower.KINDS
        try:
            best = grower.learn(ids, 49, score)
        finally:
            Grower.KINDS = kinds
        self.assertGreater(grower.passes_made[0], 1)
        numpy = lambda net: np.concatenate([np.log(net.targets(d)) for d in held_docs])
        np.testing.assert_allclose(score(grower.learner.network), numpy(best), atol=2e-4)


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class NetTextTest(unittest.TestCase):
    def test_the_network_alone_learns_reads_token_by_token_as_in_bulk_and_after_saving(self):
        from acan.lm.concept_lm import from_checkpoint_arrays
        from acan.lm.net_text import NetText
        rng = np.random.default_rng(7)
        eos, docs = 40, []
        for i in range(80):
            words = np.arange(1, 20) if i % 2 else np.arange(20, 39)
            docs += [int(rng.choice(words)) for _ in range(20)] + [eos]
        ids = np.array(docs)
        tune = [ids[:420], ids[420:840]]
        model = NetText(41, eos, 1 << 22, max_context=64, network=dict(dim=8, layers=1, heads=1, concepts=4, window=1, passes=False))
        before = model.logp(ids[:60], np.arange(1, 60)).mean()
        model.add(ids, ids, tune, lambda: None)
        self.assertIsNotNone(model.model)
        self.assertGreater(model.logp(ids[:60], np.arange(1, 60)).mean(), before)
        self.assertTrue(model.trace and all(len(point) == 2 for point in model.trace))   # the chart inside the steps
        xs = [point[0] for point in model.trace]
        self.assertEqual(xs, sorted(xs)); self.assertGreater(xs[-1], xs[0])            # it moves on as the network reads
        doc = ids[:ids.tolist().index(eos) + 1]
        q = model.logp(doc)
        cursor = model.prediction_cursor(())
        for i, t in enumerate(doc):
            p = cursor.distribution()
            self.assertAlmostEqual(p.sum(), 1.0, places=6)
            self.assertAlmostEqual(np.log(p[t]), q[i], places=4)
            cursor.observe(t)
        np.testing.assert_allclose(model.prediction_cursor(doc[:10]).distribution(),
                                   model.distribution(doc[:10]), atol=1e-6)
        restored = from_checkpoint_arrays(model.arrays())
        np.testing.assert_allclose(restored.logp(doc), q, atol=1e-6)
        self.assertEqual(len(model.generate(doc[:5], 8, temperature=.7, seed=1)), 8)
        self.assertEqual(model.concepts(), [4])

    def test_the_queue_accepts_the_network_alone_as_a_choice(self):
        from acan.lm.research_options import validate_options
        base = dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-pairs-v1')
        self.assertFalse(validate_options(base)['concept_network_only'])
        self.assertTrue(validate_options(dict(base, concept_network_only=True))['concept_network_only'])
        with self.assertRaises(ValueError):
            validate_options(dict(base, concept_network_only='sì'))


@unittest.skipUnless(TORCH, 'torch is needed to learn the network')
class NetTextGrowthTest(unittest.TestCase):
    def test_growth_starts_with_one_head_and_can_select_more_heads(self):
        from unittest.mock import patch
        from acan.lm.concept_lm import from_checkpoint_arrays
        from acan.lm.net_text import NetText

        eos = 40
        rng = np.random.default_rng(3)
        ids = rng.integers(0, eos, size=(8, 9))
        ids[:, -1] = eos
        stream = ids.ravel()
        model = NetText(41, eos, 1 << 22, growth=True)
        self.assertEqual(model.network['heads'], 1)
        model.network['passes'] = False

        # Keep training real and make the evaluation select the candidate with two heads.
        def score(network, *_):
            return np.full(2, -1. if network.config['heads'] == 2 else -2.)

        with patch('acan.lm.concept_net_training.held_out', side_effect=score):
            model.add(stream, stream, [stream], lambda: None)
        self.assertEqual([race['choice'] for race in model.races], ['teste', "com'è"])
        self.assertTrue(model.races[-1]['retained'])
        self.assertEqual(model.model.net.shape(0)[0], 2)
        restored = from_checkpoint_arrays(model.arrays())
        self.assertEqual(restored.model.net.shape(0)[0], 2)
        np.testing.assert_allclose(restored.logp(ids[0]), model.logp(ids[0]), atol=1e-6)

    def test_growth_starts_with_one_concept_and_tests_layers_in_turn(self):
        from unittest.mock import patch
        from acan.lm.concept_lm import from_checkpoint_arrays
        from acan.lm.net_text import NetText

        model = NetText(41, 40, 1 << 22, growth=True, network=dict(layers=2, passes=False))
        self.assertEqual(model.network['concepts'], 1)
        ids = np.tile(np.r_[np.arange(1, 10), 40], 8)

        def score(network, *_):
            concepts = sum(network.config['concepts'])
            return np.full(2, {2: -4., 3: -3., 4: -2.}.get(concepts, -5.))

        reported = []
        model.report = lambda *args, **facts: reported.append(facts)
        with patch('acan.lm.concept_net_training.held_out', side_effect=score):
            model.add(ids[:40], ids, [ids[40:]], lambda: None)
        self.assertEqual([race['tokens'] for race in model.races], [40, 40, 40])
        self.assertEqual([race['round'] for race in model.races], [1, 2, 3])
        self.assertEqual([race['layer'] for race in model.races], [1, 2, None])
        self.assertEqual([race['choice'] for race in model.races], ['concetti', 'concetti', "com'è"])
        self.assertEqual([race['held_out'] for race in model.races], [3., 2., 2.])
        self.assertTrue(model.races[-1]['retained'])
        self.assertEqual((model.read, model.tokens), (40, 40))
        self.assertEqual({entry['race_round'] for entry in reported}, {1, 2, 3})
        validating = next(entry for entry in reported if entry['race_round'] == 1 and entry['race'] == 4
                          and entry['race_evaluating'])
        self.assertEqual(validating['race_scores'], [4., 4., 3.])
        self.assertEqual(validating['race_candidates'][3:], ['concetti · strato 1', 'teste'])
        for race in model.races:
            final = next(entry for entry in reported if entry['race_round'] == race['round'] and len(entry['race_scores']) == 5)
            self.assertEqual(final['race_projection']['held_out'], race['held_out'])
            self.assertEqual(final['race_projection']['grows'], race['choice'] != "com'è")
            self.assertEqual(final['race_projection']['retained'], race['retained'])
        self.assertEqual({p[3] for p in model.trace if len(p) == 4}, {1, 2, 3})
        self.assertLessEqual(max(point[0] for point in model.trace), 1.)
        restored = from_checkpoint_arrays(model.arrays())
        self.assertEqual(restored.model.net.config['concepts'], [2, 2])

    def test_the_growing_network_races_when_the_text_doubles_and_reads_as_before_after_saving(self):
        from acan.lm.concept_lm import from_checkpoint_arrays
        from acan.lm.net_text import NetText
        rng = np.random.default_rng(8)
        eos, docs = 40, []
        for i in range(60):
            words = np.arange(1, 20) if i % 2 else np.arange(20, 39)
            docs += [int(rng.choice(words)) for _ in range(12)] + [eos]
        ids = np.array(docs)
        tune = [ids[:390]]
        model = NetText(41, eos, 1 << 22, max_context=32, growth=True,
                        network=dict(dim=4, layers=1, heads=1, concepts=2, window=1, passes=False))
        reported = []
        model.report = lambda *args, **facts: reported.append(facts)   # the page: the race reports, the chart has its lines
        model._reported = -1e9
        model.add(ids[:390], ids, tune, lambda: None)
        model.add(ids[390:], ids, tune, lambda: None)
        self.assertTrue(any(len(point) == 4 for point in model.trace))
        # the decisions read the validation in stretches as long as in learning, whatever its documents
        long_text = np.r_[np.arange(1, 39).repeat(30), eos]               # one document of over a thousand tokens
        self.assertLessEqual(max(map(len, model._held([long_text], 32))), 32)
        # a piece after a race goes on with the calendar of the tokens learned: no decisions at once, no stop
        grower = model._learner
        self.assertGreater(grower.decision, grower.learned)
        self.assertEqual(list(dict.fromkeys(race['tokens'] for race in model.races)), [390, len(ids)])
        self.assertTrue(all(race['choice'] in NetText.KINDS for race in model.races))
        for race in model.races:
            candidates = {candidate['choice']: candidate for candidate in race['candidates']}
            self.assertEqual(set(candidates), set(NetText.KINDS))
            self.assertEqual(candidates['teste']['heads'], 2 * candidates["com'è"]['heads'])
            self.assertEqual(race['heads'], candidates[race['choice']]['heads'])
            previous = candidates["com'è"]['concepts']
            split = candidates['concetti']
            expected = previous.copy()
            expected[split['layer']-1] += 1
            self.assertEqual(split['concepts'], expected)
        self.assertTrue(all(entry['race_total'] == 5 for entry in reported if 'race' in entry))
        doc = ids[:ids.tolist().index(eos) + 1]
        q = model.logp(doc)
        cursor = model.prediction_cursor(())
        for i, t in enumerate(doc):
            self.assertAlmostEqual(np.log(cursor.distribution()[t]), q[i], places=4)
            cursor.observe(t)
        restored = from_checkpoint_arrays(model.arrays())
        self.assertTrue(restored.growth)
        np.testing.assert_allclose(restored.logp(doc), q, atol=1e-6)
