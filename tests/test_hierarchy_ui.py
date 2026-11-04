"""Real native hierarchy, serialization, rendering and per-tab cancellation."""
import importlib.util
import threading
import unittest

import numpy as np
import pandas as pd

from acan import GraphClusterModel
from acan.vis.training import NativeTraining, graph_stats, train_native

OPTIONS = dict(max_depth=3, max_epochs=25, min_samples_split=12,
               similarity_threshold=.92, lr=.08, tau=.2, alpha=.5,
               device='numpy', batch_size=300)


class TestNativeHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame(np.random.default_rng(0).uniform(size=(100, 3)), columns=['x', 'y', 'z'])
        cls.graph = train_native(cls.data, [], OPTIONS)

    def test_native_hierarchy_survives_roundtrip(self):
        loaded = GraphClusterModel.loadModel(self.graph.to_json())
        self.assertGreater(graph_stats(loaded)['depth'], 1)
        self.assertEqual(graph_stats(loaded), graph_stats(self.graph))
        for node in loaded.clusterModel.values():
            self.assertTrue(set(node.children.values()).issubset(loaded.clusterModel))

    def test_prediction_uses_native_hierarchy_and_returns_paths(self):
        from acan.vis.inference import infer
        graph = GraphClusterModel.loadModel(self.graph.to_json())
        graph.labels = ['z']
        inputs = self.data[['x', 'y']].iloc[:20].copy()
        native, confidence, paths = graph.predictDataset(inputs, ['z'])
        result = infer(graph.to_json(), inputs, 'predict')
        lo, hi = graph.minMax['z']
        np.testing.assert_allclose(result.z, native.z * (hi - lo) + lo)
        self.assertEqual(result.ACAN_path.tolist(), list(paths))
        np.testing.assert_allclose(result.ACAN_confidence, confidence.mean(axis=1))
        self.assertTrue(any(not path.startswith(f'{graph.rootId}_G') for path in paths))

    @unittest.skipUnless(importlib.util.find_spec('dash'), 'Install ui extra')
    def test_both_views_have_valid_edges_and_child_edge_is_inspectable(self):
        from acan.vis.render import to_elements, edge_info, describe_graph
        for show_groups in (True, False):
            elements = list(to_elements(self.graph, show_groups).values())
            ids = {e['data']['id'] for e in elements if 'source' not in e['data']}
            for edge in elements:
                if 'source' in edge['data']:
                    self.assertIn(edge['data']['source'], ids)
                    self.assertIn(edge['data']['target'], ids)
        child_edge = next(e['data'] for e in to_elements(self.graph).values()
                          if 'source' in e['data'] and '_G' in e['data']['source'])
        self.assertIsNotNone(edge_info([child_edge], self.graph.to_json()))
        _, layout = describe_graph(self.graph.to_json(), 'breadthfirst')
        self.assertIn(str(self.graph.rootId), layout['roots'])

    def test_subprocess_returns_whole_graph_and_cancel_preserves_it(self):
        learner = NativeTraining(self.data, [], OPTIONS)
        learner.nLearn()
        self.assertEqual(graph_stats(learner.graph), graph_stats(self.graph))
        cancelled = NativeTraining(self.data, [], dict(OPTIONS, max_epochs=1000))
        errors = []
        def run():
            try:
                cancelled.nLearn()
            except ValueError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=run)
        thread.start()
        cancelled.request_stop()
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(cancelled.graph)
        self.assertTrue(errors)
        self.assertIsNotNone(learner.graph)


if __name__ == '__main__':
    unittest.main()
