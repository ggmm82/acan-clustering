"""Trained models stay on the server; the page holds a short reference (acan/vis/model_cache.py)."""
import os
import tempfile
import unittest
from unittest import mock

from acan.vis.model_cache import PREFIX, keep, load_graph, resolve


class ModelCacheTests(unittest.TestCase):
    def test_a_reference_gives_back_the_same_model_and_other_values_pass_through(self):
        from acan import GraphClusterModel
        graph = GraphClusterModel()
        graph.labels = ['y']
        text = graph.to_json()
        self.assertNotIn('\n', text)                           # compact JSON: no indentation
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {'ACAN_MODEL_CACHE': folder}):
            reference = keep(text)
            self.assertTrue(reference.startswith(PREFIX))
            self.assertLess(len(reference), 80)
            self.assertEqual(keep(text), reference)            # the same model, the same reference
            self.assertEqual(keep(reference), reference)
            self.assertEqual(resolve(reference), text)
            self.assertEqual(resolve(text), text)              # a model JSON as it was stored before
            self.assertIsNone(resolve(None))
            self.assertEqual(load_graph(reference).labels, ['y'])
            with self.assertRaises(ValueError):
                resolve(PREFIX+'../../etc/passwd')
            with self.assertRaises(ValueError):
                resolve(PREFIX+'0'*64)                         # removed from the server: a clear message


if __name__ == '__main__':
    unittest.main()
