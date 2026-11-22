"""The OpenAI-compatible interface (acan/lm/openai_api.py): keys, models, chat and text, streaming, stop, the official client."""
import importlib.util
import json
import tempfile
import threading
import unittest

import numpy as np

TOKEN = 's'*40


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers')
                     and importlib.util.find_spec('flask') and importlib.util.find_spec('safetensors'), 'Service dependencies required')
class OpenAICompatibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from acan.lm.concept_trainer import run_training
        from acan.lm.research_options import validate_options
        from acan.lm.service import create_app
        from acan.lm.store import Store, owner_key
        cls.folder = tempfile.TemporaryDirectory()
        root = cls.folder.name
        store = Store(root)
        cls.workspace, cls.other = 'a'*64, 'b'*64
        owner = owner_key(cls.workspace)
        words = 'il gatto dorme sul tetto e la luna guarda la città mentre il cane abbaia piano'.split()
        rng = np.random.default_rng(0)
        lines = [json.dumps(dict(text=f'<|im_start|>user\nCiao {i}<|im_end|>\n<|im_start|>assistant\n'
                                      f"{' '.join(rng.choice(words, 30))}<|im_end|>")) for i in range(800)]
        raw = ('\n'.join(lines)+'\n').encode()
        corpus = store.create_corpus(owner, 'chat.jsonl', len(raw))
        store.append_corpus(corpus['id'], owner, 0, raw); store.finish_corpus(corpus['id'], owner)
        config = validate_options(dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-pairs-v1',
                                       vocabulary=4000, sequence_length=64, validation_batches=4, save_steps=2,
                                       concept_piece=10_000, concept_context=8, device='cpu', concept_network_growth=False,
                                       corpus_id=corpus['id'], name='Prova API'))
        job = store.create_job(owner, config)
        store.update(job['id'], status='queued')
        run_training(root, job['id'])
        cls.job = store.job(job['id'])
        assert cls.job['status'] == 'completed', cls.job['error']
        cls.app = create_app(root, TOKEN)
        cls.client = cls.app.test_client()
        service = {'Authorization': 'Bearer '+TOKEN, 'X-ACAN-Workspace': cls.workspace}
        cls.key = cls.client.post('/api-keys', json=dict(name='prova'), headers=service).get_json()
        cls.service = service

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def call(self, path, body=None, key=None):
        headers = {'Authorization': 'Bearer '+(key or self.key['key'])}
        if body is None:
            return self.client.get(path, headers=headers)
        return self.client.post(path, json=body, headers=headers)

    def test_keys_belong_to_a_workspace_and_can_be_revoked(self):
        self.assertTrue(self.key['key'].startswith('acan-'))
        listed = self.client.get('/api-keys', headers=self.service).get_json()['keys']
        self.assertTrue(any(item['id'] == self.key['id'] for item in listed))
        self.assertNotIn(self.key['key'], json.dumps(listed))                 # only its digest is kept
        bad = self.call('/v1/models', key='acan-'+'0'*48)
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(bad.get_json()['error']['code'], 'invalid_api_key')
        self.assertEqual(self.client.get('/v1/models', headers={'Authorization': 'Bearer '+TOKEN}).status_code, 401)
        stranger = self.client.post('/api-keys', json=dict(name='altro'),
                                    headers=dict(self.service, **{'X-ACAN-Workspace': self.other})).get_json()
        self.assertEqual(self.call('/v1/models', key=stranger['key']).get_json()['data'], [])
        self.assertEqual(self.call('/v1/chat/completions', dict(model=self.job['id'], messages=[dict(role='user', content='Ciao')]),
                                   key=stranger['key']).status_code, 404)
        temporary = self.client.post('/api-keys', json=dict(name='breve'), headers=self.service).get_json()
        self.assertEqual(self.call('/v1/models', key=temporary['key']).status_code, 200)
        self.client.post(f"/api-keys/{temporary['id']}/revoke", headers=self.service)
        self.assertEqual(self.call('/v1/models', key=temporary['key']).status_code, 401)

    def test_models_are_the_projects_with_a_checkpoint(self):
        data = self.call('/v1/models').get_json()['data']
        self.assertEqual([item['id'] for item in data], [self.job['id']])
        self.assertEqual(self.call('/v1/models/'+'f'*32).status_code, 404)

    def test_chat_answer_is_the_model_writing_after_the_chat_prompt(self):
        from transformers import AutoTokenizer
        from acan.lm.concept_lm import load
        from acan.lm.probes import chat_prompt, stop_tokens
        body = dict(model=self.job['id'], messages=[dict(role='user', content='Ciao 3')], temperature=0, max_tokens=12)
        answer = self.call('/v1/chat/completions', body).get_json()
        self.assertEqual(answer['object'], 'chat.completion')
        self.assertEqual(answer['choices'][0]['message']['role'], 'assistant')
        root = self.app.store.directory('jobs', self.job['id'])
        model, _ = load(root/'checkpoints'/self.job['checkpoint'])
        tokenizer = AutoTokenizer.from_pretrained(str(root/'tokenizer'), local_files_only=True)
        ids = tokenizer(chat_prompt('Ciao 3'))['input_ids']
        produced = model.generate(ids, 12, stop=set(stop_tokens(tokenizer)), temperature=0.)
        self.assertEqual(answer['choices'][0]['message']['content'], tokenizer.decode(produced, skip_special_tokens=True))
        self.assertEqual(answer['usage']['completion_tokens'], len(produced))
        self.assertEqual(answer['usage']['prompt_tokens'], len(ids))
        self.assertIn(answer['choices'][0]['finish_reason'], ('stop', 'length'))

    def test_streaming_gives_the_same_text_in_pieces(self):
        body = dict(model=self.job['id'], messages=[dict(role='user', content='Ciao 7')], temperature=.8, seed=5, max_tokens=20)
        whole = self.call('/v1/chat/completions', body).get_json()['choices'][0]['message']['content']
        streamed = self.call('/v1/chat/completions', dict(body, stream=True, stream_options=dict(include_usage=True)))
        self.assertTrue(streamed.content_type.startswith('text/event-stream'))
        events = [line[6:] for line in streamed.get_data(as_text=True).splitlines() if line.startswith('data: ')]
        self.assertEqual(events[-1], '[DONE]')
        chunks = [json.loads(item) for item in events[:-1]]
        text = ''.join(chunk['choices'][0]['delta'].get('content', '') for chunk in chunks if chunk['choices'])
        self.assertEqual(text, whole)
        self.assertTrue(any(chunk.get('usage') for chunk in chunks))
        self.assertTrue(any(chunk['choices'] and chunk['choices'][0]['finish_reason'] for chunk in chunks))

    def test_stop_strings_end_the_answer_before_them(self):
        body = dict(model=self.job['id'], prompt='il gatto dorme sul tetto e la luna', temperature=0, max_tokens=40)
        free = self.call('/v1/completions', body).get_json()['choices'][0]['text']
        words = free.split()
        self.assertGreater(len(words), 3)
        stop = ' '+words[2]
        cut = self.call('/v1/completions', dict(body, stop=[stop])).get_json()
        self.assertEqual(cut['object'], 'text_completion')
        self.assertEqual(cut['choices'][0]['text'], free[:free.find(stop)])
        self.assertEqual(cut['choices'][0]['finish_reason'], 'stop')
        streamed = self.call('/v1/completions', dict(body, stop=[stop], stream=True)).get_data(as_text=True)
        pieces = [json.loads(line[6:]) for line in streamed.splitlines() if line.startswith('data: {')]
        self.assertEqual(''.join(item['choices'][0]['text'] for item in pieces), free[:free.find(stop)])

    def test_invalid_requests_answer_as_openai_does(self):
        for body, param in ((dict(model=self.job['id'], messages=[]), 'messages'),
                            (dict(model=self.job['id'], messages=[dict(role='user', content='x')], temperature=3), 'temperature'),
                            (dict(model=self.job['id'], messages=[dict(role='user', content='x')], n=2), 'n'),
                            (dict(model=self.job['id'], messages=[dict(role='user', content='x')], max_tokens=0), 'max_tokens'),
                            (dict(messages=[dict(role='user', content='x')]), 'model')):
            reply = self.call('/v1/chat/completions', body)
            self.assertEqual(reply.status_code, 400, body)
            self.assertEqual(reply.get_json()['error']['param'], param)

    @unittest.skipUnless(importlib.util.find_spec('openai'), 'The official client is not installed')
    def test_the_official_client_works_against_a_running_server(self):
        from openai import OpenAI
        from werkzeug.serving import make_server
        server = make_server('127.0.0.1', 0, self.app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAI(base_url=f'http://127.0.0.1:{server.server_port}/v1', api_key=self.key['key'], max_retries=0)
            self.assertEqual([item.id for item in client.models.list()], [self.job['id']])
            answer = client.chat.completions.create(model=self.job['id'], messages=[{'role': 'user', 'content': 'Ciao 1'}],
                                                    temperature=0, max_tokens=10)
            self.assertIsInstance(answer.choices[0].message.content, str)
            pieces = client.chat.completions.create(model=self.job['id'], messages=[{'role': 'user', 'content': 'Ciao 1'}],
                                                    temperature=0, max_tokens=10, stream=True)
            text = ''.join(chunk.choices[0].delta.content or '' for chunk in pieces if chunk.choices)
            self.assertEqual(text, answer.choices[0].message.content)
        finally:
            server.shutdown()

    @unittest.skipUnless(importlib.util.find_spec('dash'), 'The site is not installed')
    def test_the_site_passes_v1_to_the_service_with_the_clients_key(self):
        import os
        from unittest import mock
        from werkzeug.serving import make_server
        server = make_server('127.0.0.1', 0, self.app, threaded=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with mock.patch.dict(os.environ, {'ACAN_LM_URL': f'http://127.0.0.1:{server.server_port}', 'ACAN_LM_SERVICE_TOKEN': TOKEN}):
                from acan.vis.render import app as site
                web = site.server.test_client()
                headers = {'Authorization': 'Bearer '+self.key['key']}
                self.assertEqual([item['id'] for item in web.get('/v1/models', headers=headers).get_json()['data']], [self.job['id']])
                self.assertEqual(web.get('/v1/models', headers={'Authorization': 'Bearer '+TOKEN}).status_code, 401)
                body = dict(model=self.job['id'], messages=[dict(role='user', content='Ciao 2')], temperature=0, max_tokens=8)
                whole = web.post('/v1/chat/completions', json=body, headers=headers).get_json()['choices'][0]['message']['content']
                streamed = web.post('/v1/chat/completions', json=dict(body, stream=True), headers=headers).get_data(as_text=True)
                chunks = [json.loads(line[6:]) for line in streamed.splitlines() if line.startswith('data: {')]
                self.assertEqual(''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices']), whole)
        finally:
            server.shutdown()


if __name__ == '__main__':
    unittest.main()
