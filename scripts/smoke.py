"""Check the real HTTP application and saved-model inference, not just a port."""
import argparse
import base64
import http.cookiejar
import json
import time
import urllib.request

import pandas as pd

def check(url):
    client = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def get(path):
        with client.open(url.rstrip('/') + path, timeout=10) as response:
            return json.load(response)

    for attempt in range(30):
        try:
            assert get('/healthz')['status'] == 'ok'
            break
        except (OSError, AssertionError):
            if attempt == 29:
                raise
            time.sleep(1)
    layout = get('/_dash-layout')
    assert 'Inference' in json.dumps(layout)

    def tab_token(node):
        if isinstance(node, dict):
            props = node.get('props', {})
            if props.get('id') == 'tab-session':
                return props['data']
            return tab_token(props.get('children'))
        if isinstance(node, list):
            return next((token for child in node if (token := tab_token(child))), None)

    tab_id = tab_token(layout)
    assert tab_id
    dependencies = get('/_dash-dependencies')
    load = next(item['output'] for item in dependencies
                if any(i['id'] == 'upload-model' for i in item['inputs']))
    data = pd.DataFrame({'x': [0., .1, .9, 1.] * 5,
                         'label': ['A', 'A', 'B', 'B'] * 5})

    def post(output, outputs, inputs, states=()):
        body = json.dumps(dict(output=output, outputs=outputs,
                               inputs=inputs, state=list(states),
                               changedPropIds=[inputs[0]['id'] + '.' + inputs[0]['property']])).encode()
        request = urllib.request.Request(url.rstrip('/') + '/_dash-update-component',
                                         data=body, headers={'Content-Type': 'application/json'})
        with client.open(request, timeout=30) as response:
            return json.load(response)['response']

    def field(name, prop, value):
        return dict(id=name, property=prop, value=value)

    def outputs(*pairs):
        return [dict(id=name, property=prop) for name, prop in pairs]

    train = next(item['output'] for item in dependencies
                 if any(i['id'] == 'button-learn' for i in item['inputs']))
    started = post(train, outputs(('model-store', 'data'), ('test-store', 'data'),
                                 ('learning-status', 'children'), ('btn-stop-learn', 'disabled'),
                                 ('learning-monitor', 'disabled')),
                   [field('button-learn', 'n_clicks', 1)],
                   [field('data-store', 'data', data.to_dict()),
                    field('label-store', 'data', ['label']),
                    field('dpdnFilter', 'value', []), field('dpdnCategory', 'value', []),
                    field('input-max-batch', 'value', 300), field('input-lr', 'value', .08),
                    field('tab-session', 'data', tab_id)])
    assert started['learning-monitor']['disabled'] is False, started
    monitor = next(item['output'] for item in dependencies
                   if any(i['id'] == 'learning-monitor' for i in item['inputs']))
    for attempt in range(30):
        progress = post(monitor, outputs(('model-store', 'data'), ('org-chart', 'elements'),
                                        ('learning-status', 'children'), ('btn-stop-learn', 'disabled'),
                                        ('learning-monitor', 'disabled'), ('evaluation-trigger', 'data')),
                        [field('learning-monitor', 'n_intervals', attempt + 1)],
                        [field('tab-session', 'data', tab_id)])
        if progress['learning-monitor']['disabled']:
            assert 'model-store' in progress, progress
            saved = progress['model-store']['data']
            break
        time.sleep(1)
    else:
        raise AssertionError('HTTP training did not finish in time')

    loaded = post(load, outputs(('model-store', 'data'), ('dpdnLabel', 'options'),
                               ('dpdnLabel', 'value'), ('learning-status', 'children')),
                  [field('upload-model', 'contents', 'data:application/json;base64,' +
                         base64.b64encode(saved.encode()).decode())],
                  [field('upload-model', 'filename', 'model.json'),
                   field('tab-session', 'data', tab_id)])
    assert loaded['dpdnLabel']['value'] == ['label']
    endpoint = next(item['output'] for item in dependencies
                    if item['output'].startswith('..inference-result.data'))
    for mode, csv in [('predict', 'x\n0.1\n0.9\n'),
                      ('complete', 'x,label\n0.1,\n0.9,\n')]:
        result = post(endpoint, outputs(('inference-result', 'data'), ('inference-table', 'children'),
                                        ('inference-status', 'children'), ('inference-export', 'disabled')),
                      [field('inference-run', 'n_clicks', 1), field('model-store', 'data', saved),
                       field('inference-input', 'data', csv), field('inference-mode', 'value', mode)])
        assert result['inference-export']['disabled'] is False, result
        content = result['inference-result']['data']
        assert '0.1,A' in content and '0.9,B' in content, content
    print('HTTP health, layout, training, model reload, prediction and completion passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:4000')
    check(parser.parse_args().url)
