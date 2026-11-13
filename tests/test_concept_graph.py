import itertools
import unittest

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptMixture
from acan.core.graph import ConceptGraph, KEYS


def nested(seed=0, rows=40):
    """Two distant families, each made of three close groups, plus one irrelevant attribute."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], float)
    group = np.repeat(np.arange(6), rows)
    x = centres[group]+rng.normal(0, .08, (len(group), 2))
    return pd.DataFrame(dict(a=x[:, 0], b=x[:, 1], noise=rng.normal(size=len(group)))), group


class ConceptGraphTest(unittest.TestCase):
    def fit(self, frame, labels):
        """Leaves given by a hard assignment, so the test does not depend on a search."""
        from acan.core.concepts import Encoder
        model = ConceptMixture()
        model.labels, model.encoder = [], Encoder().fit(frame)
        model.root_p = np.zeros(0)
        data = model.encoder.transform(frame)
        weights = model._weights(1.)
        statistics = model._statistics(np.eye(labels.max()+1)[labels], data)
        return model, ConceptGraph(model, statistics, weights)

    def halves(self, group, seed=0):
        return group*2+np.random.default_rng(seed).integers(0, 2, len(group))

    def test_levels_are_nested_and_additive(self):
        frame, group = nested()
        model, graph = self.fit(frame, self.halves(group))        # twelve leaves: every true group arbitrarily halved
        for node in range(graph.leaves, graph.root+1):
            below = graph.members(node)
            for key in KEYS:
                np.testing.assert_allclose(graph.s[key][node], graph.s[key][below].sum(0), rtol=1e-9, atol=1e-9)
        levels = graph.levels()
        self.assertEqual(levels[0]['concepts'], 2)                # the two families: finer groups are below the resolution of the root
        self.assertEqual(levels[1]['concepts'], 6)                # resolved one level down, against each family
        self.assertLess(levels[0]['pair_criterion'], .5)
        for coarse, fine in zip(levels, levels[1:]):
            cover = [set(graph.members(node)) for node in coarse['nodes']]
            for node in fine['nodes']:
                self.assertEqual(sum(set(graph.members(node)) <= part for part in cover), 1)
        self.assertEqual(sorted(leaf for node in levels[-1]['nodes'] for leaf in graph.members(node)), list(range(graph.leaves)))

    def test_cut_is_the_best_partition_of_the_subtree(self):
        frame, group = nested(seed=1, rows=12)
        model, graph = self.fit(frame, self.halves(group)[:len(frame)] % 7)
        chosen, value = graph._cut(graph.root)

        def partitions(node):
            yield [node]
            if graph.left[node] >= 0:
                for a, b in itertools.product(partitions(graph.left[node]), partitions(graph.right[node])):
                    yield a+b
        from acan.concept_algebra import pair_mass
        score = lambda nodes: float((pair_mass(graph.s['C'][nodes], np)*graph.relevance(np.asarray(nodes), graph.root)).sum())
        best = max(score(part) for part in partitions(graph.root) if part != [graph.root])
        self.assertAlmostEqual(value, best, places=9)
        self.assertAlmostEqual(score(chosen), best, places=9)
        self.assertAlmostEqual(score([graph.root]), 0., places=9)  # one concept explains nothing about itself

    def test_a_level_without_support_falls_back_on_its_parent(self):
        frame, group = nested()
        model, graph = self.fit(frame, self.halves(group))
        g, v, _ = graph.parameters()
        self.assertTrue((v > 0).all() and np.isfinite(v).all())
        parent = {child: node for node, kids in graph.children.items() for child in kids}
        leaf = 0
        for key in KEYS:
            graph.s[key][leaf] = graph.s[key][leaf]*1e-9             # no evidence of its own
        _, weak, _ = graph.parameters()
        up = parent[leaf]
        if up == graph.root:
            expected = np.ones(v.shape[1])
        else:
            saved = {key: graph.s[key][:graph.leaves].copy() for key in KEYS}
            chain, node = [], up
            while node != graph.root:
                chain.append(node); node = parent[node]
            from acan.concept_algebra import centre_variance
            expected = np.ones(v.shape[1])
            for node in reversed(chain):
                _, expected = centre_variance(graph.s['Cm'][node], graph.s['U'][node], graph.s['Q'][node], expected, model.kappa, np)
        np.testing.assert_allclose(weak[leaf], expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()


class ConceptTreeTest(unittest.TestCase):
    def test_levels_resolve_two_scales_without_a_target(self):
        from acan.core.concepts import fit_tree
        for seed in range(3):
            frame, group = nested(rows=30, seed=seed)
            tree = fit_tree(frame, seed=0)
            # At the scale of the whole table only the two families are distinct; inside each
            # family, on its own scale, the three groups are. Nothing was asked for.
            self.assertEqual((tree.model.concepts, tree.concepts, tree.levels), (2, 6, 2))
            family = tree.model.assignments(frame).argmax(1)
            self.assertTrue((pd.crosstab(family, group < 3) > 0).sum(axis=1).eq(1).all())

    def test_normal_scores_trade_noise_shape_for_gaps(self):
        import sys
        sys.path.insert(0, 'tests')
        from test_concepts import game
        from acan.core.concepts import fit_tree
        self.assertEqual(fit_tree(game(seed=0), seed=0, normal_scores=True).concepts, 4)   # uniform noise no longer splits
        frame, _ = nested(rows=30)
        self.assertNotEqual(fit_tree(frame, seed=0, normal_scores=True).model.concepts, 2)  # but the two scales are lost

    def test_scores_use_training_ranks_and_resist_one_extreme_value(self):
        from acan.core.concepts import NormalScores
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(dict(x=rng.random(200), label=rng.integers(0, 2, 200)))
        scores = NormalScores().fit(frame, skip=['label'])
        z = scores.transform(frame)
        self.assertTrue((z.label == frame.label).all())
        self.assertLess(abs(z.x.mean()), .05)
        self.assertLess(abs(z.x.std()-1), .1)
        far = scores.transform(pd.DataFrame(dict(x=[1e9], label=[0])))
        self.assertLess(float(far.x.iloc[0]), 4.)               # an extreme value cannot stretch the scale
        self.assertTrue(np.isnan(scores.transform(pd.DataFrame(dict(x=[np.nan], label=[0]))).x.iloc[0]))

    def test_levels_resolve_what_one_level_cannot_and_stop_by_judgment(self):
        from acan.core.concepts import fit_tree
        frame, group = nested(rows=30)
        frame['family'] = np.where(group < 3, 'left', 'right')
        frame['kind'] = pd.Series(group).map(dict(enumerate('abcabc'))).to_numpy()    # depends on the fine group
        flat = fit_tree(frame, ['kind'], seed=0)
        self.assertGreaterEqual(flat.concepts, 6)
        classes = sorted(frame.kind.unique())
        test, test_group = nested(seed=9, rows=10)
        test['family'] = np.where(test_group < 3, 'left', 'right')
        rows = test
        predicted = np.asarray(classes)[flat.predict_proba(rows, 'kind', classes).argmax(1)]
        self.assertGreater((predicted == pd.Series(test_group).map(dict(enumerate('abcabc'))).to_numpy()).mean(), .95)
        # Root concepts blur a few rows of a neighbouring group; each kept child separates them.
        for concept, child in flat.children.items():
            rows = frame.iloc[flat.rows[concept]].reset_index(drop=True)
            routed = child.model.assignments(rows).argmax(1)
            table = pd.crosstab(routed, rows.kind)
            self.assertGreater(table.max(axis=1).sum(), rows.kind.value_counts().iloc[0])   # purer than the parent concept

    def test_application_builds_judged_levels_and_predicts_through_them(self):
        from acan import GraphClusterModel
        from acan.vis.inference import infer
        from acan.vis.training import graph_stats, train_native
        options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
                       min_samples_split=12, similarity_threshold=.92, routing_profile='cp-v9')
        frame, group = nested(rows=30)
        frame['kind'] = pd.Series(group).map(dict(enumerate('abcabc'))).to_numpy()
        graph = train_native(frame, ['kind'], options)
        root = graph.clusterModel[graph.rootId]
        self.assertEqual(root.routing_parameters['level_prior'], 'parent-concept-v1')
        self.assertTrue(root.routing_parameters['pair_levels'])
        self.assertEqual(len(graph.clusterModel), root.routing_parameters['nodes'])
        for node in graph.clusterModel.values():
            for child in node.children.values():
                self.assertIn(child, graph.clusterModel)
        test, test_group = nested(seed=9, rows=10)
        payload = graph.to_json()
        predicted = infer(payload, test)['kind'].astype(str).to_numpy()
        truth = pd.Series(test_group).map(dict(enumerate('abcabc'))).to_numpy()
        self.assertGreater((predicted == truth).mean(), .95)
        again = infer(GraphClusterModel.loadModel(payload).to_json(), test)['kind'].astype(str).to_numpy()
        self.assertTrue((again == predicted).all())
        self.assertGreaterEqual(graph_stats(graph)['groups'], 6)


class CovarianceTest(unittest.TestCase):
    def setup(self, labels=()):
        from acan.core.concepts import Encoder
        rng = np.random.default_rng(3)
        mix = rng.normal(size=(3, 3))
        x = rng.normal(size=(60, 3))@mix
        frame = pd.DataFrame(x, columns=list('abc'))
        if labels:
            frame['y'] = np.where(x[:, 0]+x[:, 1] > 0, 'p', 'n')
        model = ConceptMixture(covariance=True)
        model.labels, model.encoder = list(labels), Encoder().fit(frame)
        data = model.encoder.transform(frame)
        counts = np.maximum(data[2].sum(0), 1)
        model.root_p = np.maximum(data[2].sum(0)/counts.sum() if len(counts) else np.zeros(0), 1e-12)
        model.cov = np.arange(3)
        t = np.eye(3)[rng.integers(0, 3, 60)]*.9+.1/3            # soft memberships
        weights = model._weights(3. if labels else 1.)
        return model, data, t, weights

    def test_block_density_is_the_gaussian_of_the_shrunk_covariance(self):
        from scipy.stats import multivariate_normal
        model, data, t, weights = self.setup()
        s = model._statistics(t, data)
        g, sigma = model._covariance(s['C'], s['U'], s['S'])
        expected = np.stack([multivariate_normal(g[i], sigma[i]).logpdf(data[0]) for i in range(3)], 1)
        np.testing.assert_allclose(model._block_density(s, data[0], data[1]), expected, rtol=1e-9, atol=1e-9)
        missing = data[1].copy(); missing[:, 1] = 0             # an unobserved attribute is marginalized exactly
        expected = np.stack([multivariate_normal(g[i][[0, 2]], sigma[i][[0, 2]][:, [0, 2]]).logpdf(data[0][:, [0, 2]])
                             for i in range(3)], 1)
        np.testing.assert_allclose(model._block_density(s, data[0], missing), expected, rtol=1e-9, atol=1e-9)

    def test_judgment_in_closed_form_equals_removing_each_row(self):
        for labels in ((), ('y',)):
            model, data, t, weights = self.setup(labels)
            s = model._statistics(t, data)
            routed = model._route(s, data, weights)[0]
            fast = model._score(s, data, weights, np.arange(8))
            for row in range(8):
                one = tuple(part[row:row+1] for part in data)
                removed = model._statistics(routed[row:row+1], one)
                rest = {key: s[key]-removed[key] for key in s}
                feature_weights = model._weights(0.) if labels else weights
                logits = model._log_density(rest, one, feature_weights)+np.log(rest['C'])
                if labels:
                    post = np.exp(logits-logits.max()); post /= post.sum()
                    _, _, log_p = model._parameters(rest)
                    block = model.encoder.block('y')
                    slow = np.log((post[0]*(np.exp(log_p[:, block])@one[2][0, block])).sum())
                else:
                    slow = np.log(np.exp(logits-logits.max()).sum())+logits.max()-np.log(len(data[0])-1)
                self.assertAlmostEqual(fast[row], slow, places=8)

    def test_correlated_concepts_are_chosen_only_when_judged_better(self):
        from acan.core.concepts import fit_tree
        rng = np.random.default_rng(0)
        base = rng.normal(size=(300, 1))
        along = pd.DataFrame(dict(a=base[:, 0], b=base[:, 0]+rng.normal(0, .1, 300)))
        along['side'] = np.where(along.b-along.a > 0, 'above', 'below')      # only the joint position decides
        tree = fit_tree(along, ['side'], seed=0)
        self.assertTrue(tree.covariance)
        test = pd.DataFrame(dict(a=np.linspace(-2, 2, 60)))
        test['b'] = test.a+np.tile([.15, -.15], 30)
        predicted = np.asarray(['above', 'below'])[tree.predict_proba(test, 'side', ['above', 'below']).argmax(1)]
        self.assertGreater((predicted == np.where(test.b > test.a, 'above', 'below')).mean(), .9)
        import sys
        sys.path.insert(0, 'tests')
        from test_concepts import game
        self.assertFalse(fit_tree(game(seed=2), ['play'], seed=0).covariance)   # nothing to gain: the simpler concept stays

    def test_application_uses_and_saves_correlated_concepts(self):
        from acan import GraphClusterModel
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
                       min_samples_split=12, similarity_threshold=.92, routing_profile='cp-v9')
        rng = np.random.default_rng(0)
        base = rng.normal(size=300)
        frame = pd.DataFrame(dict(a=base, b=base+rng.normal(0, .1, 300)))
        frame['side'] = np.where(frame.b-frame.a > 0, 'above', 'below')
        graph = train_native(frame, ['side'], options)
        root = graph.clusterModel[graph.rootId]
        self.assertIsNotNone(root.routing_parameters['correlated'])
        test = pd.DataFrame(dict(a=np.linspace(base.min()*.8, base.max()*.8, 60)))
        test['b'] = test.a+np.tile([.15, -.15], 30)
        payload = graph.to_json()
        predicted = infer(payload, test)['side'].astype(str).to_numpy()
        self.assertGreater((predicted == np.where(test.b > test.a, 'above', 'below')).mean(), .9)
        again = infer(GraphClusterModel.loadModel(payload).to_json(), test)['side'].astype(str).to_numpy()
        self.assertTrue((again == predicted).all())
        partial = infer(payload, test.assign(b=np.nan))['side']     # one correlated attribute unknown: marginalized
        self.assertEqual(len(partial), len(test))
