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

    def test_the_oldest_models_go_beyond_the_limit_and_a_full_disk_keeps_the_json_in_the_page(self):
        from acan.vis import model_cache
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {'ACAN_MODEL_CACHE': folder}), \
                mock.patch.object(model_cache, 'PERSISTENT_BYTES', 2500):
            references = []
            for i in range(5):
                references.append(keep('{"model": %d, "padding": "%s"}' % (i, 'x'*1000)))
                os.utime(os.path.join(folder, references[-1][len(PREFIX):]+'.json'), (i, i))
            kept = sorted(os.listdir(folder))
            self.assertLessEqual(len(kept), 3)
            self.assertIn(references[-1][len(PREFIX):]+'.json', kept)         # the newest always stays
            with mock.patch.object(model_cache, 'folder', side_effect=OSError('disco pieno')):
                self.assertEqual(keep('{"model": 9}'), '{"model": 9}')       # as before: the JSON in the page


if __name__ == '__main__':
    unittest.main()
