import base64
import importlib.util
import time
import unittest

from acan.generative import ConceptLanguageModel


@unittest.skipUnless(importlib.util.find_spec('dash'), 'Install the ui extra')
class TestLaboratoryUI(unittest.TestCase):
    def setUp(self):
        from acan.vis.render import app
        from test_ui import find_tab
        self.app = app
        self.client = app.server.test_client()
        self.tab = find_tab(self.client.get('/_dash-layout').get_json())

    def invoke(self, prefix, inputs, states, changed):
        key = next(key for key in self.app.callback_map if key.startswith(prefix))
        result_fields = self.app.callback_map[key]['output']
        outputs = [dict(id=o.component_id, property=o.component_property) for o in result_fields]
        def fields(values):
            return [dict(id=i, property=p, value=v) for i, p, v in values]
        result = self.client.post('/_dash-update-component', json=dict(
            output=key, outputs=outputs, inputs=fields(inputs), state=fields(states), changedPropIds=[changed]))
        self.assertIn(result.status_code, (200, 204), result.get_data(as_text=True))
        return result.get_json()['response'] if result.status_code == 200 else {}

    def test_training_model_transfer_and_tab_isolation(self):
        from test_ui import find_tab
        other = find_tab(self.client.get('/_dash-layout').get_json())
        inputs = [('lab-train', 'n_clicks', 1), ('lab-stop', 'n_clicks', 0), ('lab-monitor', 'n_intervals', 0)]
        states = [('lab-corpus', 'value', 'ab' * 200), ('lab-context', 'value', 4),
                  ('lab-slots', 'value', 8), ('lab-epochs', 'value', 3),
                  ('lab-lr', 'value', .04), ('lab-seed', 'value', 42), ('tab-session', 'data', self.tab)]
        result = self.invoke('..lab-status.children', inputs, states, 'lab-train.n_clicks')
        self.assertFalse(result['lab-monitor']['disabled'])
        for i in range(100):
            inputs[-1] = ('lab-monitor', 'n_intervals', i + 1)
            result = self.invoke('..lab-status.children', inputs, states, 'lab-monitor.n_intervals')
            if result['lab-monitor']['disabled']:
                break
            time.sleep(.01)
        self.assertIn('Completato', result['lab-status']['children'])
        inputs = [('lab-monitor', 'disabled', True), ('lab-model-upload', 'contents', None)]
        own = self.invoke('..lab-model.data', inputs, [('tab-session', 'data', self.tab)], 'lab-monitor.disabled')
        self.assertIn('acan-concept-character-lm-v1', own['lab-model']['data'])
        foreign = self.invoke('..lab-model.data', inputs, [('tab-session', 'data', other)], 'lab-monitor.disabled')
        self.assertNotIn('lab-model', foreign)

    def test_load_generate_and_invalidate_stale_output(self):
        text = ConceptLanguageModel(context=2, slots=4).fit('ab' * 100, epochs=3).to_json()
        upload = 'data:application/json;base64,' + base64.b64encode(text.encode()).decode()
        loaded = self.invoke('..lab-model.data', [('lab-monitor', 'disabled', True),
                             ('lab-model-upload', 'contents', upload)],
                             [('tab-session', 'data', self.tab)], 'lab-model-upload.contents')
        inputs = [('lab-generate', 'n_clicks', 1), ('lab-model', 'data', loaded['lab-model']['data'])]
        states = [('lab-prompt', 'value', 'a'), ('lab-length', 'value', 20),
                  ('lab-temperature', 'value', .8), ('lab-top-k', 'value', 0),
                  ('lab-top-p', 'value', .95), ('lab-generation-seed', 'value', 7)]
        result = self.invoke('..lab-output.children', inputs, states, 'lab-generate.n_clicks')
        self.assertTrue(result['lab-output']['children'].startswith('a'))
        self.assertIn('✅', result['lab-generation-status']['children'])
        cleared = self.invoke('..lab-output.children', inputs, states, 'lab-model.data')
        self.assertEqual(cleared['lab-output']['children'], '')
