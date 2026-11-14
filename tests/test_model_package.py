"""The downloadable Python project: same predictions as the application, usable without installing."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from acan.core.export import build_package
from acan.vis.datasets import clean_frame, read_arff, split_for_training, target_and_filters
from acan.vis.inference import infer
from acan.vis.training import train_native

DATA = Path(__file__).parent/'data'
OPTIONS = dict(lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300, max_depth=1, max_epochs=100,
               min_samples_split=12, similarity_threshold=.92, routing_profile='concept-graph-v1')


def fixture():
    frame = clean_frame(read_arff((DATA/'vote.arff').read_bytes()))
    labels, filters = target_and_filters(frame)
    train, test, _ = split_for_training(frame, labels, filters, seed=1)
    return train_native(train, labels, OPTIONS), train, test, labels


class ModelPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.train, cls.test, cls.labels = fixture()
        cls.blob = build_package(cls.graph)
        cls.folder = tempfile.TemporaryDirectory()
        cls.zip = Path(cls.folder.name)/'modello_concetti.zip'
        cls.zip.write_bytes(cls.blob)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def load(self):
        sys.path.insert(0, str(self.zip))                       # straight from the zip, nothing installed
        try:
            for name in [name for name in sys.modules if name.startswith('modello_concetti')]:
                del sys.modules[name]
            import modello_concetti
            return modello_concetti.load()
        finally:
            sys.path.remove(str(self.zip))

    def test_project_layout(self):
        names = set(zipfile.ZipFile(io.BytesIO(self.blob)).namelist())
        self.assertEqual(names, {'requirements.txt', 'README.md', '__main__.py', 'modello_concetti/__init__.py',
                                 'modello_concetti/model.py', 'modello_concetti/concepts.py',
                                 'modello_concetti/concept_algebra.py', 'modello_concetti/encoding.py',
                                 'modello_concetti/state.json'})                # only what is needed
        requirements = zipfile.ZipFile(io.BytesIO(self.blob)).read('requirements.txt').decode()
        self.assertIn('numpy', requirements)
        self.assertIn('pandas', requirements)
        self.assertNotIn('acan', requirements)                   # self-contained: no dependency on the library

    def test_same_predictions_as_the_application(self):
        model = self.load()
        features = self.test.drop(columns=self.labels)
        packaged = model.predict(features)[self.labels[0]].astype(str).to_numpy()
        application = infer(self.graph.to_json(), features.assign(**{label: np.nan for label in self.labels}))
        np.testing.assert_array_equal(packaged, application[self.labels[0]].astype(str).to_numpy())
        probability = model.predict_proba(features)
        np.testing.assert_allclose(probability.sum(1), 1, atol=1e-9)
        self.assertEqual([level['concepts'] for level in model.levels()],
                         [level['concepts'] for level in self.graph.clusterModel[self.graph.rootId].routing_parameters['levels']])

    def test_completion_fills_only_empty_cells(self):
        model = self.load()
        rows = self.test.drop(columns=self.labels).iloc[:10].copy()
        column = rows.columns[0]
        rows.loc[rows.index[:3], column] = np.nan
        done = model.complete(rows)
        self.assertFalse(done.isna().any().any())                         # every empty cell is filled
        known = rows.notna()
        self.assertTrue((done[known] == rows[known]).sum().sum() == known.sum().sum())   # and nothing else changes

    def test_update_adds_rows_to_the_sums(self):
        model = self.load()
        before = model.core.statistics['C'].sum()
        model.update(self.test.iloc[:40])
        self.assertAlmostEqual(model.core.statistics['C'].sum(), before+40, places=6)
        with tempfile.TemporaryDirectory() as folder:
            model.save(folder)
            again = type(model).load(folder)
            features = self.test.drop(columns=self.labels)
            pd.testing.assert_frame_equal(again.predict(features), model.predict(features))

    def test_refit_discovers_concepts_again(self):
        model = self.load()
        model.refit(self.train, seed=1)
        features = self.test.drop(columns=self.labels)
        accuracy = (model.predict(features)[self.labels[0]].astype(str) == self.test[self.labels[0]].astype(str)).mean()
        self.assertGreater(accuracy, .9)

    def test_runs_from_the_command_line_without_installing(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = Path(folder)/'righe.csv'
            self.test.drop(columns=self.labels).to_csv(rows, index=False)
            out = Path(folder)/'previsioni.csv'
            subprocess.run([sys.executable, str(self.zip), 'prevedi', str(rows), '-o', str(out)], check=True, cwd=folder)
            predicted = pd.read_csv(out)
            self.assertIn(f'previsto_{self.labels[0]}', predicted)
            self.assertEqual(len(predicted), len(self.test))
            levels = subprocess.run([sys.executable, str(self.zip), 'livelli'], check=True, capture_output=True,
                                    text=True, cwd=folder).stdout
            self.assertTrue(json.loads(levels))

    def test_older_model_is_explained(self):
        older = json.loads(self.graph.to_json())
        older['clusterModel'][str(self.graph.rootId)]['routing_parameters'].pop('core')
        from acan import GraphClusterModel
        with self.assertRaisesRegex(ValueError, 'riaddestralo'):
            build_package(GraphClusterModel.loadModel(json.dumps(older)))


if __name__ == '__main__':
    unittest.main()
