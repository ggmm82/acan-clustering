"""Italian and English data for conversation: formats, splits, caps, resume after a pause."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ARROW = importlib.util.find_spec('pyarrow') is not None


def parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)


class FormatTest(unittest.TestCase):
    def test_chat_format_and_split(self):
        from acan.lm.dialogue_data import chat, split_of
        text = chat([('human', 'Ciao'), ('gpt', 'Ciao! Come posso aiutarti?'), ('user', '')])
        self.assertEqual(text, '<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\nCiao! Come posso aiutarti?<|im_end|>')
        self.assertIsNone(chat([('user', 'solo una domanda')]))
        self.assertEqual(split_of('qualunque', official_test=True), 'test')
        roles = [split_of(f'documento {i}') for i in range(5000)]
        self.assertAlmostEqual(roles.count('validation')/5000, .01, delta=.005)
        self.assertEqual(split_of('documento 7'), split_of('documento 7'))        # the reference never moves

    def test_catalog_pins_every_file(self):
        from acan.lm.dialogue_data import MIXTURES, SOURCES, catalog
        for source in SOURCES.values():
            for file in source['files']:
                self.assertRegex(file['sha256'], '^[0-9a-f]{64}$')
                self.assertRegex(file['url'], r'^https://huggingface\.co/datasets/[^/]+/[^/]+/resolve/[0-9a-f]{40}/')
        for mixture in MIXTURES.values():
            for name, part in mixture['parts'].items():
                self.assertTrue(all(index < len(SOURCES[name]['files']) for index in part['files']))
        self.assertEqual([item['id'] for item in catalog()['mixtures']], ['prova', 'base', 'esteso'])


@unittest.skipUnless(ARROW, 'Requires pyarrow')
class BuildTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        sources = self.root/'sources'
        sources.mkdir()
        parquet(sources/'synth.parquet', [dict(language='it' if i % 3 else 'en', query=f'Domanda {i}?', synthetic_answer=f'Risposta {i}.')
                                           for i in range(300)])
        parquet(sources/'wiki.parquet', [dict(title=f'Voce {i}', text='Testo della voce. '*20) for i in range(200)])
        parquet(sources/'oasst.parquet', [
            dict(message_id='a', parent_id=None, text='Come stai?', role='prompter', lang='it', deleted=False),
            dict(message_id='b', parent_id='a', text='Bene, grazie!', role='assistant', lang='it', deleted=False),
            dict(message_id='c', parent_id='a', text='Benissimo.', role='assistant', lang='it', deleted=False),
            dict(message_id='d', parent_id=None, text='Hola', role='prompter', lang='es', deleted=False)])
        self.files = {}
        for name in ('synth', 'wiki', 'oasst'):
            path = sources/f'{name}.parquet'
            self.files[name] = dict(url='file://'+str(path), path=f'data/{name}.parquet', bytes=path.stat().st_size,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.sources = dict(synth=dict(title='S', languages=['it', 'en'], license='x', page='x', files=[self.files['synth']]),
                            wiki_it=dict(title='W', languages=['it'], license='x', page='x', files=[self.files['wiki']]),
                            oasst=dict(title='O', languages=['it'], license='x', page='x', files=[self.files['oasst']]))
        self.mixtures = dict(piccola=dict(title='piccola', parts=dict(
            synth=dict(files=[0], caps=dict(it=1, en=1)), wiki_it=dict(files=[0], caps=dict(it=1)), oasst=dict(files=[0], caps=dict(it=1)))))

    def tearDown(self):
        self.folder.cleanup()

    def patched(self):
        from acan.lm import dialogue_data
        def download(file, target, check=lambda: None, attempts=5):
            Path(target).write_bytes(Path(file['url'][7:]).read_bytes())
            return Path(target)
        return [mock.patch.object(dialogue_data, 'SOURCES', self.sources), mock.patch.object(dialogue_data, 'MIXTURES', self.mixtures),
                mock.patch.object(dialogue_data, 'download', download)]

    def test_builds_resumes_and_respects_caps(self):
        from acan.lm import dialogue_data
        patches = self.patched()
        for patch in patches:
            patch.start()
        try:
            with self.assertRaises(dialogue_data.Paused):
                self._pause_after_first_file()
            state = dialogue_data.build(self.root/'build', 'piccola')
        finally:
            for patch in patches:
                patch.stop()
        lines = [json.loads(line) for line in (self.root/'build'/'corpus.jsonl').read_text().splitlines()]
        texts = [line['text'] for line in lines]
        self.assertEqual(len(texts), len(set(texts)))                    # the resume wrote nothing twice
        self.assertTrue(state['done'])
        self.assertEqual(sum(state['records'].values()), len(lines))
        self.assertTrue(any(text.startswith('Voce ') for text in texts))
        oasst = [text for text in texts if 'Come stai?' in text]
        self.assertEqual(len(oasst), 2)                                   # one per answer; Spanish left out
        self.assertTrue(all('<|im_start|>assistant' in text for text in oasst))
        self.assertEqual(sum('Domanda' in text for text in texts), 300)  # all of it: under the cap of 1 MB per language

    def _pause_after_first_file(self):
        from acan.lm import dialogue_data
        original = dialogue_data.records
        def records(source, path, file):
            for item in original(source, path, file):
                yield item
            if source == 'synth':
                raise dialogue_data.Paused()                            # stop after reading the first file, before saving its end
        with mock.patch.object(dialogue_data, 'records', records):
            dialogue_data.build(self.root/'build', 'piccola')

    def test_caps_stop_a_language(self):
        from acan.lm import dialogue_data
        self.mixtures['piccola']['parts']['synth']['caps'] = dict(it=0.001, en=1)    # about 1 KB of Italian
        patches = self.patched()
        for patch in patches:
            patch.start()
        try:
            dialogue_data.build(self.root/'caps', 'piccola')
        finally:
            for patch in patches:
                patch.stop()
        texts = [json.loads(line)['text'] for line in (self.root/'caps'/'corpus.jsonl').read_text().splitlines()]
        italian = sum(len((json.dumps(dict(text=text))+'\n').encode()) for text in texts if 'Domanda' in text and not text.startswith('<|im_start|>user\nDomanda 0?')
                      and int(text.split('Domanda ')[1].split('?')[0]) % 3)
        self.assertLess(italian, 2048)


@unittest.skipUnless(ARROW and importlib.util.find_spec('flask') and importlib.util.find_spec('torch'), 'Requires LM service dependencies')
class BuildServiceTest(unittest.TestCase):
    def test_request_pause_and_resume(self):
        from acan.lm.service import create_app
        with tempfile.TemporaryDirectory() as root:
            client = create_app(root, 'x'*40).test_client()
            headers = {'Authorization': 'Bearer '+'x'*40, 'X-ACAN-Workspace': 'b'*64}
            catalog = client.get('/dialogue-data', headers=headers).get_json()
            self.assertEqual(len(catalog['mixtures']), 3)
            with mock.patch('acan.lm.manager.launch') as launch:
                answer = client.post('/builds', headers=headers, json=dict(mixture='prova'))
                self.assertEqual(answer.status_code, 200, answer.get_data(as_text=True))
                identifier = answer.get_json()['id']
                self.assertEqual(launch.call_args.args[2:], ('build', identifier))
                self.assertEqual(client.post('/builds', headers=headers, json=dict(mixture='enorme')).status_code, 400)
                item = client.post(f'/builds/{identifier}/pause', headers=headers).get_json()
                self.assertEqual(item['status'], 'pausing')
                from acan.lm.store import Store
                Store(root).update_build(identifier, status='paused')
                item = client.post(f'/builds/{identifier}/resume', headers=headers).get_json()
                self.assertEqual(item['status'], 'queued')
            self.assertEqual(client.get('/builds', headers={**headers, 'X-ACAN-Workspace': 'c'*64}).get_json(), [])


if __name__ == '__main__':
    unittest.main()
