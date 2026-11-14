"""The small test of the generative laboratory: three tiny models trained from scratch."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

TORCH = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('transformers') is not None


@unittest.skipUnless(TORCH, 'Torch required')
class SmallTestTest(unittest.TestCase):
    def test_runs_on_a_text_and_reports_plain_measures(self):
        from acan.lm.small_test import KINDS, run
        text = ('Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, '
                'ché la diritta via era smarrita. ')*80
        phases = []
        result = run(text, 'quick', report=phases.append, settings=dict(memory_steps=3, text_steps=3, dim=16, block=32))
        self.assertEqual(set(result['memory']), set(KINDS))
        self.assertEqual(set(result['text']), set(KINDS))
        for kind in KINDS:
            self.assertTrue(0 <= result['memory'][kind]['accuracy'] <= 1)
            self.assertGreater(result['text'][kind]['bits_per_character'], 0)
            self.assertEqual(len(result['text'][kind]['sample']), 300)
        self.assertEqual(len(phases), 6)
        self.assertGreater(result['memory_setup']['nearest_answer_distance'], result['memory_setup']['exact_tokens'])
        json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(ValueError, 'troppo breve'):
            run('breve', 'quick', settings=dict(memory_steps=1, text_steps=1, dim=16, block=32))

    def test_reads_plain_text_and_jsonl(self):
        from acan.lm.small_test import read_corpus
        with tempfile.TemporaryDirectory() as folder:
            plain, records = Path(folder)/'a', Path(folder)/'b'
            plain.write_text('uno\n\ndue', encoding='utf-8')
            records.write_text('\n'.join(json.dumps(dict(text=word)) for word in ('uno', 'due'))+'\n{"text": "tron', encoding='utf-8')
            self.assertEqual(read_corpus(plain, False), 'uno\n\ndue')
            self.assertEqual(read_corpus(records, True), 'uno\n\ndue')          # a record cut by the limit is skipped


@unittest.skipUnless(TORCH and importlib.util.find_spec('flask'), 'Requires LM service dependencies')
class SmallTestServiceTest(unittest.TestCase):
    def test_request_follow_and_stop(self):
        from unittest.mock import patch
        from acan.lm.service import create_app
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root, 'x'*40)
            client = app.test_client()
            headers = {'Authorization': 'Bearer '+'x'*40, 'X-ACAN-Workspace': 'a'*64}
            content = 'ciao '*2000
            created = client.post('/corpora', headers=headers, json=dict(name='dante.txt', size=len(content.encode()))).get_json()
            client.post(f"/corpora/{created['id']}/chunks", headers={**headers, 'X-Offset': '0'}, data=content.encode())
            client.post(f"/corpora/{created['id']}/finish", headers=headers)
            with patch('acan.lm.manager.launch') as launch:
                answer = client.post('/small-tests', headers=headers, json=dict(corpus_id=created['id'], budget='quick'))
            self.assertEqual(answer.status_code, 200, answer.get_data(as_text=True))
            identifier = answer.get_json()['id']
            self.assertEqual(launch.call_args.args[2], 'small-test')
            test = client.get(f'/small-tests/{identifier}', headers=headers).get_json()
            self.assertEqual((test['status'], test['request']['corpus_name'], test['request']['budget']), ('running', 'dante.txt', 'quick'))
            self.assertEqual(len(client.get('/small-tests', headers=headers).get_json()), 1)
            other = {**headers, 'X-ACAN-Workspace': 'b'*64}
            self.assertEqual(client.get(f'/small-tests/{identifier}', headers=other).status_code, 404)
            self.assertEqual(client.post(f'/small-tests/{identifier}/cancel', headers=headers).get_json()['status'], 'cancelling')
            self.assertEqual(client.post('/small-tests', headers=headers, json=dict(corpus_id=created['id'], budget='forever')).status_code, 400)
