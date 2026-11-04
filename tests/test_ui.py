"""Exercise Dash callback requests through Flask, without a running web server."""
import base64
import importlib.util
import unittest
import threading

import pandas as pd

from acan import AlgebraicClusterLearner


def find_tab(node):
    if isinstance(node, dict):
        props = node.get('props', {})
        if props.get('id') == 'tab-session':
            return props['data']
        return find_tab(props.get('children'))
    if isinstance(node, list):
        return next((token for child in node if (token := find_tab(child))), None)


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
        response = self.client.get('/_dash-layout')
        self.assertEqual(response.status_code, 200)
        self.tab_id = find_tab(response.get_json())

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
                             [('upload-model', 'filename', 'model.json'),
                              ('tab-session', 'data', self.tab_id)])
        self.assertEqual(result['dpdnLabel']['value'], ['label'])
        result = self.invoke('..model-store.data', [('upload-model', 'contents', 'data:;base64,e30=')],
                             [('upload-model', 'filename', 'bad.json'),
                              ('tab-session', 'data', self.tab_id)])
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

    def test_same_browser_tabs_do_not_share_training_or_stop(self):
        from acan.vis.render import training_jobs
        second_tab = find_tab(self.client.get('/_dash-layout').get_json())
        self.assertNotEqual(self.tab_id, second_tab)
        with self.client.session_transaction() as session:
            owner = session['acan_session']

        class Learner:
            iteration = 0
            def __init__(self):
                self.stop = threading.Event()
            def nLearn(self, **kwargs):
                self.stop.wait(5)
            def request_stop(self):
                self.stop.set()

        a, b = Learner(), Learner()
        first = training_jobs.start((owner, self.tab_id), a, {}, ['target_a'])
        second = training_jobs.start((owner, second_tab), b, {}, ['target_b'])
        try:
            self.invoke('..learning-monitor.disabled', [('btn-stop-learn', 'n_clicks', 1)],
                        [('tab-session', 'data', self.tab_id)])
            self.assertTrue(first['done'].wait(2))
            self.assertFalse(second['done'].is_set())
            self.assertFalse(b.stop.is_set())
            response = self.invoke('..model-store.data', [('learning-monitor', 'n_intervals', 1)],
                                   [('tab-session', 'data', second_tab)])
            self.assertNotIn('model-store', response)
            self.assertFalse(response['learning-monitor']['disabled'])
        finally:
            a.request_stop()
            b.request_stop()
            second['done'].wait(2)
