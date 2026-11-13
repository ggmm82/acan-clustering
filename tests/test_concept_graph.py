import itertools
import unittest

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptMixture
from acan.core.graph import ConceptGraph


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
            for key in graph.keys:
                np.testing.assert_allclose(graph.s[key][node], graph.s[key][below].sum(0), rtol=1e-9, atol=1e-9)
        levels = graph.levels()
        self.assertEqual(levels[0]['concepts'], 2)                # the two families: finer groups are below the resolution of the root
        self.assertGreater(levels[1]['concepts'], 2)              # finer levels exist below the two families
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
        for key in graph.keys:
            graph.s[key][leaf] = graph.s[key][leaf]*1e-9             # no evidence of its own
        _, weak, _ = graph.parameters()
        up = parent[leaf]
        if up == graph.root:
            expected = np.ones(v.shape[1])
        else:
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


class SavedGraphTest(unittest.TestCase):
    def test_application_saves_nodes_edges_and_levels(self):
        from acan import GraphClusterModel
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
                       min_samples_split=12, similarity_threshold=.92, routing_profile='concept-graph-v1')
        rng = np.random.default_rng(0)
        centre = np.repeat(np.arange(4), 60)
        frame = pd.DataFrame(dict(x=centre*2+rng.normal(0, .6, 240), noise=rng.normal(size=240)))
        frame['kind'] = np.asarray(list('abab'))[centre]
        graph = train_native(frame, ['kind'], options)
        root = graph.clusterModel[graph.rootId]
        state = root.routing_parameters
        self.assertEqual(root.routing_profile, 'concept-graph-v1')
        self.assertTrue(state['edges'] and all(0 <= edge['first'] < edge['second'] < root.k() for edge in state['edges']))
        levels = state['pair_levels']
        self.assertTrue(levels)
        for level in levels:                                        # every level is a partition of the concepts
            self.assertEqual(sorted(i for group in level['members'] for i in group), list(range(root.k())))
        payload = graph.to_json()
        predicted = infer(payload, frame.drop(columns=['kind']).assign(kind=np.nan).iloc[:40], 'complete')['kind']
        self.assertGreater((predicted.astype(str).to_numpy() == frame.kind.iloc[:40].to_numpy()).mean(), .8)
        self.assertEqual(GraphClusterModel.loadModel(payload).clusterModel[graph.rootId].routing_parameters['edges'],
                         state['edges'])


if __name__ == '__main__':
    unittest.main()
