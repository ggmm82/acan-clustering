"""Same-user-data comparisons, genuine modern estimators and reversible branches."""
import importlib.util
import unittest

import numpy as np
import pandas as pd

from acan import GraphClusterModel
from acan.vis.training import train_native
from acan.vis.refinement import refine, remove_branch, root_signature

OPTIONS = dict(max_depth=1, max_epochs=25, min_samples_split=5, similarity_threshold=.92,
               lr=.08, tau=.2, alpha=.5, device='numpy', batch_size=300)


class TestOnDemandRefinement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.DataFrame(np.random.default_rng(0).uniform(size=(100, 3)), columns=['x', 'y', 'z'])
        cls.graph = train_native(cls.train, [], OPTIONS)

    def test_one_child_no_duplicate_root_and_exact_undo(self):
        original = self.graph.to_json()
        parent = self.graph.clusterModel[self.graph.rootId]
        group = next(k for k, (sub, _) in parent.splitDataset(self.train).items() if 5 <= len(sub) < len(self.train))
        grown = refine(original, self.train, f'{parent.id}_G{group}', OPTIONS)
        self.assertEqual(grown.rootId, self.graph.rootId)
        self.assertEqual(len(grown.clusterModel), 2)
        child = grown.clusterModel[parent.id].children[group]
        self.assertNotEqual(child, grown.rootId)
        self.assertEqual(root_signature(original), root_signature(remove_branch(grown.to_json(), str(child)).to_json()))
        restored = remove_branch(grown.to_json(), f'{parent.id}_G{group}')
        self.assertEqual(restored.to_json(), GraphClusterModel.loadModel(original).to_json())
        with self.assertRaises(ValueError):
            remove_branch(original, str(parent.id))
        with self.assertRaises(ValueError):
            refine(grown.to_json(), self.train, f'{parent.id}_G{group}', OPTIONS)

    def test_global_benchmark_limit_and_session_owned_stop(self):
        import threading
        from acan.vis.jobs import TrainingJobs
        class Work:
            def __init__(self):
                self.stop = threading.Event()
            def nLearn(self, **kwargs):
                self.stop.wait(3)
            def request_stop(self):
                self.stop.set()
        jobs = TrainingJobs(max_running=1)
        first = Work()
        owned = jobs.start('first-page', first, {}, [])
        try:
            with self.assertRaises(ValueError):
                jobs.start('other-page', Work(), {}, [])
            self.assertFalse(jobs.stop('other-page'))
            self.assertFalse(first.stop.is_set())
        finally:
            jobs.stop('first-page')
            self.assertTrue(owned['done'].wait(2))
        second = Work()
        next_job = jobs.start('other-page', second, {}, [])
        jobs.stop('other-page')
        self.assertTrue(next_job['done'].wait(2))


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Requires UI benchmarks')
class TestUserDataEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cls.data = pd.DataFrame({'x': rng.uniform(size=100), 'y': rng.uniform(size=100)})
        cls.data['class'] = np.where(cls.data.x > .5, 'high', 'low')
        cls.data['number'] = cls.data.x * 10 + cls.data.y
        cls.train, cls.test = cls.data.iloc[:80], cls.data.iloc[80:]
        cls.graph = train_native(cls.train, ['class', 'number'], OPTIONS)

    def test_current_and_previous_same_rows_independent_targets(self):
        from acan.vis.evaluation import evaluate, encode
        report = evaluate(self.graph.to_json(), self.test, before_json=self.graph.to_json(), baselines=False)
        self.assertEqual(report['rows'], 20)
        self.assertTrue(all(row['N'] == 20 for row in report['scores']))
        self.assertTrue(all(item['Delta'] == 0 for item in report['comparisons']))
        self.assertEqual({d['kind'] for d in report['details']}, {'regression', 'classification'})
        a, b = encode(pd.DataFrame({'x': [0., 1.], 'cat': ['a', 'b']}), pd.DataFrame({'x': [10.], 'cat': ['unseen']}))
        self.assertEqual(a.shape[1], b.shape[1])
        self.assertEqual(b[0, 0], 10.)  # held-out values did not change fitted scale

    @unittest.skipUnless(importlib.util.find_spec('xgboost') and importlib.util.find_spec('catboost'), 'Requires modern UI extras')
    def test_real_modern_estimators_classification_and_regression(self):
        from threadpoolctl import threadpool_limits
        from acan.vis.evaluation import evaluate
        with threadpool_limits(limits=1):
            report = evaluate(self.graph.to_json(), self.test, self.train)
        names = {r['Modello'] for r in report['scores']}
        self.assertTrue({'KNN', 'CART', 'Random Forest', 'Extra Trees', 'HistGradientBoosting', 'XGBoost', 'CatBoost'}.issubset(names))
        self.assertFalse([r for r in report['timings'] if r['Stato'] != 'ok'])
        self.assertEqual(len([r for r in report['timings'] if r['Modello'] == 'CatBoost']), 2)

    def test_unsupervised_masks_only_held_out_observations(self):
        from acan.vis.evaluation import evaluate
        graph = GraphClusterModel.loadModel(self.graph.to_json())
        graph.labels = []
        report = evaluate(graph.to_json(), self.test)
        self.assertEqual(report['mode'], 'masked-root')
        self.assertGreater(report['masked_cells'], 0)
        self.assertEqual(report['rows'], len(self.test))
