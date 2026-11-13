import unittest

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptModel, Encoder


def nested(seed=0, rows=40):
    """Two distant families, each made of three close groups, plus one irrelevant attribute."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], float)
    group = np.repeat(np.arange(6), rows)
    x = centres[group]+rng.normal(0, .08, (len(group), 2))
    return pd.DataFrame(dict(a=x[:, 0], b=x[:, 1], noise=rng.normal(size=len(group)))), group


def shared(seed=0, rows=60):
    """Two families that differ in ``family``; the same two behaviours (``mode``) occur in both."""
    rng = np.random.default_rng(seed)
    family, mode = np.repeat([0, 1], 2*rows), np.tile(np.repeat([0, 1], rows), 2)
    return pd.DataFrame(dict(family=family*10+rng.normal(0, .3, 4*rows), other=family*10+rng.normal(0, .3, 4*rows),
                             mode=mode*1.5+rng.normal(0, .1, 4*rows), again=mode*1.5+rng.normal(0, .1, 4*rows))), family, mode


class ConceptLevelsTest(unittest.TestCase):
    def given(self, frame):
        model = ConceptModel()
        model.labels, model.encoder = [], Encoder().fit(frame)
        model.root_p = model.root_g = np.zeros(0)
        return model, model.encoder.transform(frame), model._weights(1.)

    def test_general_concepts_then_particular_behaviours(self):
        frame, group = nested()
        model = ConceptModel(seed=0).fit(frame)
        self.assertGreaterEqual(len(model.levels), 2)
        self.assertEqual(model.levels[0]['concepts'], 2)             # the families: all that is distinct at the scale of the root
        family = model._route(model.levels[0]['statistics'], model.encoder.transform(frame), model.train_weights)[0].argmax(1)
        self.assertEqual(len(set(zip(family, group//3))), 2)         # level 1 never mixes the two families
        # The three groups are placed in the same way inside both families: seen through the
        # families they are three behaviours, not six fragments, and each descends from both.
        self.assertEqual(model.levels[-1]['concepts'], 3)
        concept = model.assignments(frame, observed_only=False).argmax(1)
        self.assertEqual(len(set(zip(concept, group % 3))), 3)       # a behaviour never mixes two placements
        descent = model.levels[1]['parents']                          # P(family | behaviour), through the levels between
        for level in model.levels[2:]:
            descent = descent@level['parents']
        np.testing.assert_allclose(descent, .5, atol=.05)
        self.assertTrue(all(level['pair_criterion'] > 0 for level in model.levels))

    def test_levels_are_joined_by_dense_matrices(self):
        frame, _ = nested(seed=1)
        model = ConceptModel(seed=0).fit(frame)
        for above, level in zip(model.levels, model.levels[1:]):
            self.assertEqual(level['parents'].shape, (above['concepts'], level['concepts']))
            np.testing.assert_allclose(level['parents'].sum(0), 1, atol=1e-9)        # P(parent | concept)
            np.testing.assert_allclose(level['parents']@level['mass'], above['mass'], atol=1e-6)   # mass flows up exactly
        self.assertIsNone(model.levels[0]['parents'])
        self.assertAlmostEqual(sum(model.levels[-1]['mass']), 1.)

    def test_one_behaviour_found_in_two_families_is_one_concept_with_two_parents(self):
        frame, family, mode = shared()
        model, data, weights = self.given(frame)
        # Level 1: the two families. Level 2, on the scale of the families: the two behaviours.
        families = model._statistics(np.eye(2)[family], data)
        scale = model._context(families, data, weights, hard=True)
        behaviours = model._statistics(np.eye(2)[mode], scale)
        fragments = model._statistics(np.eye(4)[2*family+mode], scale)        # a tree: each behaviour once per family
        # The pair criterion against the level above prefers one node per behaviour to its fragments.
        self.assertGreater(model.pair_criterion(behaviours), model.pair_criterion(fragments))
        parents = (np.eye(2)[family].T@np.eye(2)[mode])/np.eye(2)[mode].sum(0)
        np.testing.assert_allclose(parents, .5)                               # each behaviour descends from both families

    def test_a_level_adds_nothing_to_itself(self):
        frame, group = nested(seed=2)
        model, data, weights = self.given(frame)
        level = model._statistics(np.eye(6)[group], data)
        scale = model._context(level, data, weights, hard=True)
        own = model._hard(model._route(level, data, weights)[0])              # the level as a partition
        self.assertAlmostEqual(model.pair_criterion(model._statistics(own, scale)), 0., places=9)
        self.assertGreater(model.pair_criterion(level), 0)                    # against the root it is a level
        halves = group*2+np.random.default_rng(0).integers(0, 2, len(group))  # arbitrary halves of every group
        self.assertLess(abs(model.pair_criterion(model._statistics(np.eye(12)[halves], scale))), .01)

    def test_statistics_of_the_level_above_are_additive(self):
        frame, group = nested(seed=3)
        model, data, weights = self.given(frame)
        families = model._statistics(np.eye(2)[group//3], data)
        scale = model._context(families, data, weights, hard=True)
        fine, coarse = model._statistics(np.eye(6)[group], scale), model._statistics(np.eye(2)[group//3], scale)
        for key in fine:
            np.testing.assert_allclose(fine[key].reshape(2, 3, -1).sum(1), coarse[key].reshape(2, -1), rtol=1e-9, atol=1e-9)
        # A concept without members of its own takes the scale of the level above.
        empty = {key: value[:1]*1e-6 for key, value in fine.items()}
        np.testing.assert_allclose(model._parameters(empty)[1], model._prior(fine)[0][:1], rtol=1e-3)


class SavedGraphTest(unittest.TestCase):
    def test_application_saves_levels_matrices_and_edges(self):
        from acan import GraphClusterModel
        from acan.vis import graph_view
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
                       min_samples_split=12, similarity_threshold=.92, routing_profile='concept-graph-v1')
        frame, group = nested()
        frame['kind'] = np.asarray(list('abcabc'))[group]             # the same three kinds in both families
        graph = train_native(frame, ['kind'], options)
        root = graph.clusterModel[graph.rootId]
        state = root.routing_parameters
        self.assertEqual(root.routing_profile, 'concept-graph-v1')
        self.assertTrue(all(0 <= edge['first'] < edge['second'] < root.k() for edge in state['edges']))
        levels = state['levels']
        self.assertGreaterEqual(len(levels), 2)
        self.assertEqual(levels[-1]['concepts'], root.k())
        for above, level in zip(levels, levels[1:]):
            parents = np.asarray(level['parents'])
            self.assertEqual(parents.shape, (above['concepts'], level['concepts']))
            np.testing.assert_allclose(parents.sum(0), 1, atol=1e-3)
        drawn = graph_view.elements(graph)
        classes = [element.get('classes', '') for element in drawn]
        self.assertEqual(classes.count('level-concept'), sum(level['concepts'] for level in levels[:-1]))
        shifted = min(element['position']['y'] for element in drawn if element.get('classes', '').startswith('local-group'))
        self.assertGreater(shifted, max(element['position']['y'] for element in drawn if element.get('classes') == 'level-concept'))
        self.assertIn('level-edge', classes)
        self.assertIn('livello 1', graph_view.navigation(graph)[4])
        payload = graph.to_json()
        predicted = infer(payload, frame.drop(columns=['kind']).assign(kind=np.nan).iloc[::6], 'complete')['kind']
        self.assertGreater((predicted.astype(str).to_numpy() == frame.kind.iloc[::6].to_numpy()).mean(), .8)
        self.assertEqual(GraphClusterModel.loadModel(payload).clusterModel[graph.rootId].routing_parameters['levels'],
                         levels)


if __name__ == '__main__':
    unittest.main()
