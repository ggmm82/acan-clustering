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

    def test_generalization_adds_statistics_and_removes_duplicates(self):
        frame = game()
        model = ConceptMixture(seed=0).fit(frame)
        doubled = {key: np.concatenate([value/2, value/2]) for key, value in model.statistics.items()}
        weights = model.train_weights
        cost = model._merge_costs(doubled, weights)
        k = model.concepts
        self.assertTrue(all(cost[i, i+k] < 0 for i in range(k)))          # merging a copy always pays
        merged, _ = model._generalize(doubled, weights, k)
        for key, value in model.statistics.items():
            self.assertTrue(np.allclose(np.sort(merged[key], axis=0), np.sort(value, axis=0)))

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
        own = {key: value.copy() for key, value in model.statistics.items()}
        rows = np.arange(len(frame))
        seen = model._route(own, data, model.train_weights)[1]
        unseen = model._score(own, data, model.train_weights, rows)
        self.assertLess(unseen.mean(), seen.mean())                      # no self-support


if __name__ == '__main__':
    unittest.main()


OPTIONS = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=3, max_epochs=100,
               min_samples_split=12, similarity_threshold=.92, routing_profile='cp-v9')


class ConceptApplicationTest(unittest.TestCase):
    def test_application_path_classification_round_trip(self):
        from acan import GraphClusterModel
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        train, test = game(seed=2), game(60, seed=3)
        graph = train_native(train, ['play'], OPTIONS)
        payload = graph.to_json()
        root = GraphClusterModel.loadModel(payload).clusterModel[graph.rootId]
        self.assertEqual(root.routing_profile, 'cp-v9')
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
        self.assertEqual(child.routing_profile, 'cp-v9')
