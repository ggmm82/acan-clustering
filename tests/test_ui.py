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

    def test_upload_keeps_row_order_and_reports_suggested_target(self):
        frame = pd.DataFrame({'PassengerId': range(120), 'Survived': [0, 1]*60,
                              'measurement': [i/120 for i in range(120)]})
        contents = 'data:text/csv;base64,' + base64.b64encode(frame.to_csv(index=False).encode()).decode()
        result = self.invoke('..data-store.data', [('upload-data', 'contents', contents)],
                             [('upload-data', 'filename', 'sample.csv'),
                              ('upload-data', 'last_modified', 0), ('label-store', 'data', [])])
        data = result['data-store']['data']
        self.assertEqual(data['PassengerId'], list(range(120)))
        result = self.invoke('..dpdnLabel.options', [('data-store', 'data', data)],
                             [('label-store', 'data', [])])
        self.assertEqual(result['dpdnLabel']['value'], ['Survived'])
        self.assertEqual(result['dpdnFilter']['value'], ['PassengerId'])

    def test_loaded_model_target_options_keep_uploaded_feature_filters(self):
        # Loading a JSON model restores only its target dropdown options.
        data = {'Name':['A','B'], 'Ticket':['X','Y'], 'Cabin':['C1','C2'],
                'PassengerId':[1,2], 'Age':[20,30], 'Survived':[0,1]}
        result = self.invoke('..label-store.data', [('dpdnLabel', 'value', ['Survived'])],
            [('dpdnLabel', 'options', [{'label':'Survived', 'value':'Survived'}]),
             ('data-store', 'data', data)])
        self.assertEqual({item['value'] for item in result['dpdnFilter']['options']},
                         {'Name','Ticket','Cabin','PassengerId','Age'})
        self.assertNotIn('value', result['dpdnFilter'])

    def test_upload_sends_filter_options_and_values_together(self):
        data = {'PassengerId':list(range(100)), 'Name':[f'person{i}' for i in range(100)],
                'Ticket':[f'ticket{i}' for i in range(100)], 'Age':list(range(100)),
                'Survived':[0,1]*50}
        result = self.invoke('..dpdnLabel.options', [('data-store', 'data', data)],
                             [('label-store', 'data', [])])
        options = {item['value'] for item in result['dpdnFilter']['options']}
        self.assertTrue(set(result['dpdnFilter']['value']).issubset(options))
        self.assertEqual(set(result['dpdnFilter']['value']), {'PassengerId','Name','Ticket'})
        self.assertNotIn('Survived', options)

    def test_restore_suggested_filters_after_clearing(self):
        data = {'PassengerId':list(range(100)), 'Name':[f'person{i}' for i in range(100)],
                'Ticket':[f'ticket{i}' for i in range(100)], 'Age':list(range(100)), 'Survived':[0,1]*50}
        result = self.invoke('dpdnFilter.value', [('btn-suggest-filter', 'n_clicks', 1)],
                             [('data-store', 'data', data), ('label-store', 'data', ['Survived'])])
        self.assertEqual(set(result['dpdnFilter']['value']), {'PassengerId','Name','Ticket'})

    def test_public_experiment_requires_explicit_click_and_passes_only_preset(self):
        from unittest.mock import patch
        with patch('acan.vis.generative.backend', return_value=dict(job={'id':'new-public-job'},
                message='Training avviato', prompt='Nel mezzo del cammin')) as backend:
            result = self.invoke('..lm-active-job.data', [('lm-public-start', 'n_clicks', 1)],
                [('lm-workspace', 'data', 'a'*64), ('lm-public-dataset', 'value', 'dante-it'),
                 ('lm-public-preset', 'value', 'check')])
        backend.assert_called_once_with('a'*64, '/public-experiments', 'POST', dict(dataset='dante-it', preset='check'))
        self.assertEqual(result['lm-active-job']['data'], 'new-public-job')
        self.assertEqual(result['lm-prompt']['value'], 'Nel mezzo del cammin')

    def test_lm_chart_explains_baseline_and_repeated_validation(self):
        job = dict(id='job', progress=dict(step=12, tokens_trained=12288, baseline_nll=4.683,
                   validation_nll=6.62), history=[dict(step=12, training_nll=6.69, validation_nll=6.62)],
                   snapshots=[dict(step=7, name='saved')], status='running',
                   config=dict(total_steps=1000), checkpoint='saved', error=None)
        result = self.invoke('..lm-progress.children', [('lm-state', 'data', {'jobs':[job]}),
                             ('lm-job', 'value', 'job')], [('lm-checkpoint', 'value', None)])
        figure = result['lm-loss']['figure']
        self.assertEqual(figure['layout']['shapes'][0]['y0'], 4.683)
        text = str(result['lm-progress']['children'])
        self.assertIn('ancora peggiore', text)
        self.assertIn('checkpoint passo 7', text)
        self.assertIn('token elaborati', text)

    def test_public_defaults_survive_worker_catalog_loading(self):
        result = self.invoke('..lm-public-dataset.options', [('lm-state', 'data', None),
                             ('lm-public-dataset', 'value', 'dante-it')])
        self.assertNotIn('lm-public-dataset', result)
        self.assertNotIn('lm-public-preset', result)
        self.assertTrue(result['lm-public-start']['disabled'])

    def test_new_public_project_is_selected_over_previous_project(self):
        state = dict(jobs=[dict(id='old', config={'name':'Old'}, status='paused', progress={}),
                           dict(id='new', config={'name':'New'}, status='created', progress={})],
                     corpora=[], select_job='new')
        result = self.invoke('..lm-corpus.options', [('lm-state', 'data', state)],
            [('lm-active-job', 'data', 'new'), ('lm-upload-state', 'data', None),
             ('lm-corpus', 'value', None), ('lm-job', 'value', 'old')])
        self.assertEqual(result['lm-job']['value'], 'new')

    def test_sessions_receive_distinct_server_identifiers(self):
        other = self.app.server.test_client()
        other.get('/_dash-layout')
        with self.client.session_transaction() as first:
            first_id = first['acan_session']
        with other.session_transaction() as second:
            self.assertNotEqual(first_id, second['acan_session'])

    def test_model_change_never_starts_benchmark(self):
        from unittest.mock import patch
        from acan.vis.render import benchmark_jobs
        with patch.object(benchmark_jobs, 'start') as start:
            result = self.invoke('..benchmark-status.children', [
                ('benchmark-run', 'n_clicks', 0), ('benchmark-stop', 'n_clicks', 0),
                ('model-store', 'data', self.model)], [('tab-session', 'data', self.tab_id)],
                changed='model-store.data')
        start.assert_not_called()
        self.assertTrue(result['benchmark-monitor']['disabled'])
        self.assertIsNone(result['benchmark-report']['data'])

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
