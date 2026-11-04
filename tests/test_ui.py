"""Exercise Dash callback requests through Flask, without a running web server."""
import base64
import importlib.util
import unittest

import pandas as pd

from acan import AlgebraicClusterLearner


@unittest.skipUnless(importlib.util.find_spec('dash'), 'Install the ui extra')
class TestUICallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from acan.vis.render import app
        cls.app = app
        cls.model = AlgebraicClusterLearner.learn(pd.DataFrame({
            'x': [0., .1, .9, 1.] * 5, 'label': ['A', 'A', 'B', 'B'] * 5,
        }), labels=['label'], max_epochs=40, device='numpy').to_json()

    def setUp(self):
        self.client = self.app.server.test_client()
        self.assertEqual(self.client.get('/_dash-layout').status_code, 200)

    def invoke(self, output_prefix, inputs, states=(), changed=None):
        key = next(key for key, callback in self.app.callback_map.items()
                   if key.startswith(output_prefix) and
                   [(item['id'], item['property']) for item in callback['inputs']] ==
                   [(item[0], item[1]) for item in inputs])
        outputs = self.app.callback_map[key]['output']
        if isinstance(outputs, list):
            outputs = [dict(id=o.component_id, property=o.component_property) for o in outputs]
        else:
            outputs = dict(id=outputs.component_id, property=outputs.component_property)
        def fields(items):
            return [dict(id=i, property=p, value=v) for i, p, v in items]
        response = self.client.post('/_dash-update-component', json={
            'output': key, 'outputs': outputs, 'inputs': fields(inputs),
            'state': fields(states),
            'changedPropIds': [changed or f'{inputs[0][0]}.{inputs[0][1]}'],
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()['response']

    def test_load_restores_labels_and_invalid_model_preserves_state(self):
        contents = 'data:application/json;base64,' + base64.b64encode(self.model.encode()).decode()
        result = self.invoke('..model-store.data', [('upload-model', 'contents', contents)],
                             [('upload-model', 'filename', 'model.json')])
        self.assertEqual(result['dpdnLabel']['value'], ['label'])
        result = self.invoke('..model-store.data', [('upload-model', 'contents', 'data:;base64,e30=')],
                             [('upload-model', 'filename', 'bad.json')])
        self.assertNotIn('model-store', result)
        self.assertIn('❌', result['learning-status']['children'])

    def test_inference_callback_and_stale_results_clear(self):
        inputs = [('inference-run', 'n_clicks', 1), ('model-store', 'data', self.model),
                  ('inference-input', 'data', 'x\n0.1\n0.9\n'),
                  ('inference-mode', 'value', 'predict')]
        result = self.invoke('..inference-result.data', inputs)
        self.assertFalse(result['inference-export']['disabled'])
        self.assertIn('label', result['inference-result']['data'])
        cleared = self.invoke('..inference-result.data', inputs, changed='model-store.data')
        self.assertIsNone(cleared['inference-result']['data'])
        self.assertTrue(cleared['inference-export']['disabled'])

    def test_sessions_receive_distinct_server_identifiers(self):
        other = self.app.server.test_client()
        other.get('/_dash-layout')
        with self.client.session_transaction() as first:
            first_id = first['acan_session']
        with other.session_transaction() as second:
            self.assertNotEqual(first_id, second['acan_session'])
