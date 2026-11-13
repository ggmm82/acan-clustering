import unittest

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptMixture


def game(rows=120, seed=0):
    """Two attributes decide the outcome through an interaction; three are noise."""
    rng = np.random.default_rng(seed)
    weather, place = rng.integers(0, 2, rows), rng.integers(0, 2, rows)
    return pd.DataFrame(dict(weather=np.where(weather, 'rain', 'sun'), place=np.where(place, 'hall', 'field'),
                             day=rng.integers(1, 29, rows), month=rng.integers(1, 13, rows),
                             year=rng.integers(1980, 2010, rows),
                             play=np.where(weather == place, 'yes', 'no')))


class ConceptMixtureTest(unittest.TestCase):
    def test_unsupervised_concepts_and_relevance(self):
        frame = game()
        model = ConceptMixture(seed=1).fit(frame)
        # The count is never given. Uniform noise is fitted slightly better by extra
        # Gaussians, so one or two surplus concepts can pass the one-standard-error
        # rule (documented limit); the four combinations must never be mixed.
        self.assertEqual(model.levels[0]['concepts'], 4)             # the general concepts are the four combinations
        self.assertTrue(4 <= model.concepts <= 6)
        concept = model.assignments(frame, observed_only=False).argmax(1)
        table = pd.crosstab([frame.weather, frame.place], concept)
        self.assertTrue(((table > 0).sum(axis=0) == 1).all())
        relevance = model.explained_variance(frame)
        self.assertGreater(min(relevance[c] for c in ('weather', 'place', 'play')), .9)
        self.assertLess(max(relevance[c] for c in ('day', 'month', 'year')), .5)      # a surplus split raises one
        self.assertTrue(0 <= relevance['CP'] <= 1)

    def test_interaction_is_predicted_without_reading_the_target(self):
        train, test = game(seed=2), game(60, seed=3)
        model = ConceptMixture(seed=0).fit(train, ['play'])
        self.assertEqual(float((model.predict(test, 'play') == test.play).mean()), 1.)
        scrambled = test.assign(play=test.play.sample(frac=1, random_state=0).to_numpy())
        self.assertTrue((model.predict(scrambled, 'play') == model.predict(test, 'play')).all())

    def test_generalization_adds_statistics_and_copies_cost_nothing(self):
        frame = game()
        model = ConceptMixture(seed=0).fit(frame)
        k = model.concepts
        doubled = {key: np.concatenate([value/2, value/2]) for key, value in model.statistics.items()}
        cost = model._merge_costs(doubled)
        finite = cost[np.isfinite(cost)]
        self.assertTrue((finite > -1e-9).all())                          # generalizing never helps the reader for free
        self.assertTrue(all(abs(cost[i, i+k]) < 1e-9 for i in range(k)))    # except between two copies of one concept
        data = model.encoder.transform(frame)
        merged = model._generalize(doubled, data, model.train_weights, k)
        for key, value in model.statistics.items():
            self.assertTrue(np.allclose(np.sort(merged[key], axis=0), np.sort(value, axis=0)))

    def test_concepts_are_connected_where_observations_hesitate(self):
        rng = np.random.default_rng(0)
        centre = np.repeat(np.arange(4.), 60)                            # four groups in a row: 0-1-2-3
        frame = pd.DataFrame(dict(x=centre*2+rng.normal(0, .6, 240), noise=rng.normal(size=240)))
        model = ConceptMixture(seed=0).fit(frame)
        membership = model.assignments(frame, observed_only=False)
        table = model._statistics(membership, model.encoder.transform(frame))
        centres = table['U']/table['Cm']                                 # in the units of the table
        distance = np.sqrt(((centres[:, None]-centres[None])**2).sum(-1))
        self.assertTrue(model.edges)
        strongest = max(model.edges, key=lambda edge: edge[2])
        typical = np.median(distance[np.triu_indices(model.concepts, 1)])
        self.assertLess(distance[strongest[0], strongest[1]], typical)    # neighbours, not distant groups
        self.assertTrue(all(first < second and weight > 0 for first, second, weight in model.edges))
        # Linear in the attributes: sums of values, squares and counts, and of the scale of the level above.
        self.assertFalse(any(key not in ('C', 'Cm', 'U', 'Q', 'N', 'PV', 'PP', 'PG') for key in model.statistics))
        self.assertTrue(all(value.ndim <= 2 for value in model.statistics.values()))

    def test_features_that_vary_together_are_read_together_when_judged_better(self):
        def crossed(rows, seed):
            """Two classes with the same margins: they differ only in how two attributes vary together."""
            rng = np.random.default_rng(seed)
            kind = rng.integers(0, 2, rows)
            a = rng.normal(size=rows)
            b = np.where(kind, a, -a)+rng.normal(0, .3, rows)
            return pd.DataFrame(dict(a=a, b=b, noise=rng.normal(size=rows), kind=np.where(kind, 'along', 'across')))
        train, test = crossed(400, 0), crossed(300, 1)
        together = ConceptMixture(seed=0).fit(train, ['kind'])
        alone = ConceptMixture(seed=0, covariance=False).fit(train, ['kind'])
        self.assertIsNotNone(together.cross)                             # adopted: unseen rows are read better
        self.assertIsNone(alone.cross)
        accuracy = lambda model: float((model.predict(test, 'kind') == test.kind).mean())
        self.assertGreater(accuracy(together), .8)                       # the two classes cross at the origin
        self.assertGreater(accuracy(together), accuracy(alone)-.03)       # here many concepts do as well; see Letter
        self.assertLessEqual(together.concepts, alone.concepts)
        # The score of an unseen row equals the score after taking the row out of the sums by hand.
        data = together.encoder.transform(train)
        t = together._route(together.statistics, data, together.train_weights)[0]
        full_s, full_c = together.statistics, together.cross
        together.statistics, together.cross = together._statistics(t, data), together._cross(t, data)
        fast = together._block_unseen(data[0][:3], data[1][:3], t[:3])
        for row in range(3):
            keep = np.ones(len(train), bool)
            keep[row] = False
            part = tuple(value[keep] for value in data)
            together.statistics = together._statistics(t[keep], part)
            together.cross = together._cross(t[keep], part)
            np.testing.assert_allclose(fast[row], together._block_density(data[0][row:row+1], data[1][row:row+1])[0], rtol=1e-6, atol=1e-6)
        together.statistics, together.cross = full_s, full_c
        # A missing feature is dropped from the Gaussian: predictions stay finite and sensible.
        partial = together.predict_proba(test.assign(noise=np.nan), 'kind')[0]
        self.assertTrue(np.isfinite(partial).all())
        self.assertGreater(float((np.asarray(together.encoder.categories['kind'])[partial.argmax(1)] == test.kind).mean()), .8)

    def test_missing_attributes_are_marginalized(self):
        train = game(seed=4)
        train.loc[::7, 'day'] = np.nan
        model = ConceptMixture(seed=0).fit(train, ['play'])
        test = game(40, seed=5)
        full = model.predict_proba(test, 'play')[0]
        without_noise = model.predict_proba(test.assign(day=np.nan, month=np.nan, year=np.nan), 'play')[0]
        self.assertTrue(np.allclose(full.sum(1), 1) and np.isfinite(without_noise).all())
        self.assertTrue((full.argmax(1) == without_noise.argmax(1)).all())

    def test_score_excludes_the_scored_row(self):
        frame = game(40)
        model = ConceptMixture(seed=0).fit(frame)
        data = model.encoder.transform(frame)
        rows = np.arange(len(frame))
        one = model._statistics(np.ones((len(frame), 1)), data)
        each = model._statistics(np.eye(len(frame)), data)               # every row is its own concept
        seen = model._route(each, data, model.train_weights)[1]
        self.assertGreater(seen.mean(), model._route(one, data, model.train_weights)[1].mean())   # memorizing fits
        unseen = model._score(each, data, model.train_weights, rows)
        self.assertLess(unseen.mean(), model._score(one, data, model.train_weights, rows).mean())  # and reads nothing


if __name__ == '__main__':
    unittest.main()


OPTIONS = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
               min_samples_split=12, similarity_threshold=.92, routing_profile='concept-graph-v1')


class ConceptApplicationTest(unittest.TestCase):
    def test_application_reads_features_together(self):
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        rng = np.random.default_rng(0)
        kind = rng.integers(0, 2, 500)
        a = rng.normal(size=500)
        frame = pd.DataFrame(dict(a=a, b=np.where(kind, a, -a)+rng.normal(0, .3, 500), noise=rng.normal(size=500),
                                  kind=np.where(kind, 'along', 'across')))
        graph = train_native(frame.iloc[:350], ['kind'], OPTIONS)
        state = graph.clusterModel[graph.rootId].routing_parameters
        self.assertEqual(len(state['correlated']['columns']), 3)
        test = frame.iloc[350:]
        predicted = infer(graph.to_json(), test.assign(kind=np.nan), 'complete')['kind']
        self.assertGreater((predicted.astype(str).to_numpy() == test.kind.to_numpy()).mean(), .8)

    def test_application_path_classification_round_trip(self):
        from acan import GraphClusterModel
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        train, test = game(seed=2), game(60, seed=3)
        graph = train_native(train, ['play'], OPTIONS)
        payload = graph.to_json()
        root = GraphClusterModel.loadModel(payload).clusterModel[graph.rootId]
        self.assertEqual(root.routing_profile, 'concept-graph-v1')
        self.assertEqual(root.routing_parameters['generalization'], 'reader-loss-v1')
        self.assertIn('edges', root.routing_parameters)
        self.assertEqual(root.routing_parameters['cp_definition'], 'explained-variance-v1')
        self.assertEqual(len(graph.clusterModel), 1)                     # no automatic hierarchy
        self.assertEqual(root.k(), 4)
        predicted = infer(payload, test)
        self.assertTrue((predicted['play'].astype(str) == test['play']).all())
        again = infer(GraphClusterModel.loadModel(payload).to_json(), test)
        self.assertTrue((again['play'] == predicted['play']).all())
        self.assertTrue(((root.Relevance >= 0) & (root.Relevance <= 1)).all())

    def test_application_path_without_target_and_with_quantity(self):
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        frame = game(seed=6)
        graph = train_native(frame, [], OPTIONS)
        self.assertTrue(4 <= graph.clusterModel[graph.rootId].k() <= 6)
        hidden = game(40, seed=8)
        completed = infer(graph.to_json(), hidden.assign(play=np.nan), 'complete')
        self.assertTrue((completed['play'].astype(str) == hidden['play']).all())   # any attribute can be the unknown
        rng = np.random.default_rng(0)
        priced = frame.assign(price=np.where(frame.play == 'yes', 30., 10.)+rng.normal(0, 1, len(frame)))
        graph = train_native(priced.iloc[:90], ['price'], OPTIONS)
        root = graph.clusterModel[graph.rootId]
        self.assertEqual(root.routing_parameters['numeric_targets'], [root.cols.index('price')])
        self.assertEqual(root.routing_parameters['numeric_readout'], 'weighted-training-mean')
        predicted = infer(graph.to_json(), priced.iloc[90:])['price'].astype(float)
        error = np.abs(predicted.to_numpy()-priced.price.iloc[90:].to_numpy()).mean()
        self.assertLess(error, 3.)

    def test_requested_refinement_uses_the_same_core(self):
        from acan import GraphClusterModel
        from acan.vis.refinement import refine
        from acan.vis.training import train_native
        frame = game(200, seed=7)
        graph = train_native(frame, ['play'], OPTIONS)
        root = graph.clusterModel[graph.rootId]
        refined = refine(graph.to_json(), frame, f'{root.id}_G0', OPTIONS)
        child = refined.clusterModel[refined.clusterModel[refined.rootId].children[0]]
        self.assertEqual(child.routing_profile, 'concept-graph-v1')
