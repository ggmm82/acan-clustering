"""Quality guards: what the concept graph achieves today must not silently get worse.

The project's own datasets are not in the repository, so these use the tables bundled with
scikit-learn and the application's own split. Each threshold sits a little below today's
value (in parentheses), far enough for numerical differences between library versions and
close enough to catch a real regression. When a change is meant to trade quality for
something else, update the threshold and the reason here, never delete the test.
"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SKLEARN = importlib.util.find_spec('sklearn') is not None


def bundled(name):
    from sklearn import datasets
    from acan.vis.datasets import clean_frame
    data = getattr(datasets, f'load_{name}')(as_frame=True)
    frame = data.frame.rename(columns={'target': 'kind'})
    frame['kind'] = frame['kind'].map(lambda value: f'c{value}')
    return clean_frame(frame)


def measure(name, seeds):
    from acan.core.concepts import ConceptModel
    from acan.vis.datasets import split_for_training
    frame = bundled(name)
    accuracy, models = [], []
    for seed in seeds:
        train, test, _ = split_for_training(frame, ['kind'], [], seed=seed)
        model = ConceptModel(seed=seed).fit(train, ['kind'])
        accuracy.append(float((model.predict(test, 'kind') == test.kind).mean()))
        models.append(model)
    return np.array(accuracy), models


@unittest.skipUnless(SKLEARN, 'Requires scikit-learn (installed with the ui extra)')
class BundledTablesTest(unittest.TestCase):
    def test_iris(self):
        accuracy, models = measure('iris', range(10))
        self.assertGreaterEqual(accuracy.mean(), .95)                   # (0.973)
        self.assertGreaterEqual(accuracy.min(), .86)                    # (0.900)
        for model in models:
            self.assertLessEqual(model.concepts, 6)                     # (3-5) three species, not a cloud of prototypes
            self.assertIn(model.levels[0]['concepts'], (2, 3))          # setosa apart, then the two close species

    def test_wine(self):
        accuracy, models = measure('wine', range(10))
        self.assertGreaterEqual(accuracy.mean(), .95)                   # (0.978)
        self.assertTrue(all(model.concepts <= 12 for model in models))  # (3-8)

    def test_breast_cancer(self):
        accuracy, models = measure('breast_cancer', range(2))
        self.assertGreaterEqual(accuracy.mean(), .92)                   # (0.939)
        self.assertTrue(all(model.concepts <= 45 for model in models))  # (10-29)
        self.assertTrue(all(model.levels[0]['concepts'] <= 5 for model in models))  # few general concepts

    def test_same_seed_same_model(self):
        _, first = measure('iris', [3])
        _, second = measure('iris', [3])
        for key, value in first[0].statistics.items():
            np.testing.assert_allclose(value, second[0].statistics[key])
        self.assertEqual([level['concepts'] for level in first[0].levels],
                         [level['concepts'] for level in second[0].levels])


@unittest.skipUnless(SKLEARN, 'Requires scikit-learn (installed with the ui extra)')
class ApplicationPathTest(unittest.TestCase):
    """The same quality through the interface: encoding, saving, reloading, inference."""

    def test_iris_through_the_application(self):
        from acan import GraphClusterModel
        from acan.vis import graph_view
        from acan.vis.datasets import split_for_training
        from acan.vis.inference import infer
        from acan.vis.training import train_native
        options = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=1, max_epochs=100,
                       min_samples_split=12, similarity_threshold=.92, routing_profile='concept-graph-v1')
        frame = bundled('iris')
        scores = []
        for seed in range(3):
            train, test, _ = split_for_training(frame, ['kind'], [], seed=seed)
            graph = train_native(train, ['kind'], options)
            payload = graph.to_json()
            predicted = infer(payload, test.assign(kind=np.nan))['kind'].astype(str)
            scores.append((predicted.to_numpy() == test.kind.to_numpy()).mean())
            reloaded = GraphClusterModel.loadModel(payload)
            structure = graph_view.model_structure(reloaded)
            # Every group with rows descends from some concept of the level above, and is drawn so.
            counts = np.asarray(reloaded.clusterModel[reloaded.rootId].Counts).ravel()
            for group in structure['groups']:
                if counts[int(group['name'][1:])] > 0 and structure['levels']:
                    self.assertTrue(group['parents'])
            drawn = graph_view.elements(reloaded)
            targets = {element['data']['target'] for element in drawn if element.get('classes') == 'level-edge'}
            for index, count in enumerate(counts):
                if count > 0 and structure['levels']:
                    self.assertIn(f'{reloaded.rootId}_G{index}', targets)
        self.assertGreaterEqual(np.mean(scores), .93)                   # (0.97)


DATA = Path(__file__).parent/'data'


def accuracy(path, seeds):
    from acan.core.concepts import ConceptModel
    from acan.vis.datasets import clean_frame, read_arff, split_for_training, target_and_filters
    path = str(path)
    raw = (pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path) if path.endswith('.xlsx')
           else read_arff(Path(path).read_bytes()))
    frame = clean_frame(raw)
    labels, filters = target_and_filters(frame)
    scores = []
    for seed in seeds:
        train, test, _ = split_for_training(frame, labels, filters, seed=seed)
        model = ConceptModel(seed=seed).fit(train, labels)
        scores.append((model.predict(test, labels[0]) == test[labels[0]].astype(str)).mean())
    return float(np.mean(scores))


class PublicTablesTest(unittest.TestCase):
    """Real public tables kept in tests/data (UCI, CC BY 4.0), run in CI in about a minute.

    Thresholds are set so that a known regression fails: with the target weighted as a single
    attribute, Sonar falls to 0.781 and Vehicle to 0.704 on these splits.
    """

    def test_sonar(self):
        self.assertGreaterEqual(accuracy(DATA/'sonar.csv', range(1, 6)), .83)        # (0.867)

    def test_vehicle(self):
        self.assertGreaterEqual(accuracy(DATA/'vehicle.arff', range(1, 4)), .75)     # (0.790)

    def test_vote(self):
        self.assertGreaterEqual(accuracy(DATA/'vote.arff', range(1, 6)), .94)        # (0.961) categorical, missing values


def local(name):
    """A dataset of the project that is not public enough to be in the repository."""
    import os
    for folder in (os.environ.get('ACAN_DATA', ''), '.data', '.data/nuovi'):
        path = os.path.join(folder, name)
        if folder and os.path.exists(path):
            return path
    return None


@unittest.skipUnless(local('titanic.csv'), 'Needs the project datasets (.data/ or ACAN_DATA); skipped in CI')
class ProjectDatasetsTest(unittest.TestCase):
    """The longer measurements of docs/MODEL-RESULTS.md, ten splits each (a few minutes)."""

    def test_sonar_ten_splits(self):
        self.assertGreaterEqual(accuracy(DATA/'sonar.csv', range(1, 11)), .83)       # (0.864)

    def test_titanic(self):
        self.assertGreaterEqual(accuracy(local('titanic.csv'), range(1, 11)), .80)   # (0.818); without the resolution floor 0.781

    @unittest.skipUnless(local('credit-g.arff'), 'Needs credit-g.arff')
    def test_credit(self):
        self.assertGreaterEqual(accuracy(local('credit-g.arff'), range(1, 6)), .63)  # (0.67): known weakness, not to worsen


class BehaviourTest(unittest.TestCase):
    """Properties measured during development, on data built for the purpose."""

    def test_unsupervised_game_finds_the_four_combinations(self):
        import sys
        sys.path.insert(0, 'tests')
        from test_concepts import game
        from acan.core.concepts import ConceptModel
        found = [ConceptModel(seed=0).fit(game(seed=seed)).levels[0]['concepts'] for seed in range(8)]
        self.assertGreaterEqual(found.count(4), 7)                     # (8 of 8)

    def test_two_families_resolve_at_two_scales(self):
        from acan.core.concepts import ConceptModel
        rng = np.random.default_rng(0)
        parts = []
        for family, centre in enumerate((-6., 6.)):
            for group in range(3):
                angle = 2*np.pi*group/3+family
                parts.append(np.r_[centre, centre, 1.5*np.cos(angle), 1.5*np.sin(angle)]+rng.normal(0, .3, (150, 4)))
        model = ConceptModel(seed=0).fit(pd.DataFrame(np.vstack(parts), columns=list('abcd')))
        self.assertEqual(model.levels[0]['concepts'], 2)               # the families first
        self.assertGreaterEqual(model.levels[-1]['concepts'], 5)       # (5-6) the groups below them


if __name__ == '__main__':
    unittest.main()
