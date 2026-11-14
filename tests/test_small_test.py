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
            memory = result['memory'][kind]                                   # data kept for debugging
            self.assertAlmostEqual(memory['accuracy']+memory['wrong_but_from_the_sequence']+memory['wrong_not_in_the_sequence'], 1, places=5)
            self.assertEqual(memory['training_curve'][-1]['step'], 3)
            self.assertGreater(result['text'][kind]['parameters'], 0)
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


def finished_test(dense, window, concepts, bits=(2.40, 2.40, 2.42)):
    kinds = ('dense', 'window', 'concepts')
    return dict(id='f'*32, status='completed', phase='Finita', request=dict(corpus_name='wikitext-2.jsonl', budget='full'),
                result=dict(memory_setup=dict(chance=1/16),
                            memory={kind: dict(accuracy=value, seconds=1.) for kind, value in zip(kinds, (dense, window, concepts))},
                            text={kind: dict(bits_per_character=value, prompt='p', sample='s') for kind, value in zip(kinds, bits)}))


@unittest.skipUnless(importlib.util.find_spec('dash_bootstrap_components'), 'Requires the ui extra')
class ValidationPathTest(unittest.TestCase):
    def test_the_criteria_of_the_small_test(self):
        from acan.vis.generative import small_checks
        self.assertEqual([passed for _, passed, _ in small_checks(finished_test(1., .06, .95)['result'])], [True, True])
        # Today's measure: concepts recall 24% against 100%: the memory criterion fails.
        self.assertEqual([passed for _, passed, _ in small_checks(finished_test(1., .06, .24)['result'])], [False, True])
        self.assertIsNone(small_checks(finished_test(.3, .06, .3)['result'])[0][1])      # the test itself did not work
        self.assertFalse(small_checks(finished_test(1., .06, .95, (2.40, 2.40, 2.50))['result'])[1][1])

    def test_decision_needs_every_step(self):
        from acan.vis.generative import decision_view, validation_pair
        def job(identifier, variant, status='completed'):
            return dict(id=identifier, status=status, progress=dict(step=600),
                        config=dict(name='Validazione · x', series=dict(id='s', variant=variant)))
        def tested(nll):
            return [dict(records=dict(selected_checkpoint=dict(quality=dict(nll=nll, tokens=9000))))]
        dense = job('b', 'dense')
        dense['config']['name'] = 'Controllo denso · Validazione · x'          # as the worker renames it
        pair = validation_pair([job('a', 'chosen'), dense, dict(id='c', status='completed', progress={}, config=dict(name='Altro'))])
        self.assertEqual([item['id'] for item in pair], ['a', 'b'])
        text = lambda view: str(view)
        self.assertIn('Pronto per la GPU', text(decision_view(finished_test(1., .06, .95), pair, dict(a=tested(3.40), b=tested(3.35)))))
        self.assertIn('Non ancora', text(decision_view(finished_test(1., .06, .24), pair, dict(a=tested(3.40), b=tested(3.35)))))
        self.assertIn('Non ancora', text(decision_view(finished_test(1., .06, .95), pair, dict(a=tested(3.60), b=tested(3.35)))))
        self.assertIn('sospesa', text(decision_view(finished_test(1., .06, .95), pair, {})))         # verification missing
        self.assertIn('sospesa', text(decision_view(None)))
        self.assertIsNone(validation_pair([job('a', 'chosen')]))

    def test_view_of_a_small_test(self):
        from acan.vis.generative import small_test_view
        line, view = small_test_view(finished_test(1., .06, .24))
        self.assertIn('Finita', line)
        self.assertIn('24.0%', str(view))
        line, view = small_test_view(dict(id='x', status='failed', error='Testo troppo breve', request={}))
        self.assertIn('Testo troppo breve', str(view))
        self.assertIsNone(small_test_view(dict(id='x', status='running', phase='Memoria · dense', request={}))[1])
