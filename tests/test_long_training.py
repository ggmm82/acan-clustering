"""Long training: documents as text, permanent milestones, learning queue, restart after maintenance."""
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

SERVICE = all(importlib.util.find_spec(name) for name in ('flask', 'torch', 'transformers'))


class DocumentsTest(unittest.TestCase):
    def convert(self, name, raw):
        from acan.lm.documents import convert
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'corpus'
            path.write_bytes(raw)
            count = convert(path, name)
            records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(count, len(records))
        return records

    def test_text_markdown_and_python(self):
        self.assertEqual(self.convert('note.md', '# Titolo\n\nUn paragrafo.'.encode())[0]['text'], '# Titolo\n\nUn paragrafo.')
        self.assertIn('def saluta', self.convert('codice.py', b'def saluta():\n    return "ciao"\n')[0]['text'])

    def test_docx_epub_and_zip(self):
        docx = io.BytesIO()
        with zipfile.ZipFile(docx, 'w') as archive:
            archive.writestr('word/document.xml', '<w:document><w:body><w:p><w:r><w:t>Primo</w:t></w:r><w:r><w:t xml:space="preserve"> paragrafo</w:t></w:r></w:p>'
                                                  '<w:p><w:r><w:t>Secondo &amp; ultimo</w:t></w:r></w:p></w:body></w:document>')
        self.assertEqual(self.convert('lettera.docx', docx.getvalue())[0]['text'], 'Primo paragrafo\n\nSecondo & ultimo')
        epub = io.BytesIO()
        with zipfile.ZipFile(epub, 'w') as archive:
            archive.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
            archive.writestr('OEBPS/book.opf', '<package><manifest><item id="b" href="b.xhtml"/><item id="a" href="a.xhtml"/></manifest>'
                                               '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>')
            archive.writestr('OEBPS/a.xhtml', '<html><head><style>x{}</style></head><body><h1>Capitolo uno</h1><p>Era una notte.</p></body></html>')
            archive.writestr('OEBPS/b.xhtml', '<html><body><p>Fine.</p></body></html>')
        text = self.convert('libro.epub', epub.getvalue())[0]['text']
        self.assertLess(text.index('Capitolo uno'), text.index('Fine.'))           # reading order from the spine
        self.assertNotIn('x{}', text)
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, 'w') as archive:
            archive.writestr('a.txt', 'uno')
            archive.writestr('dir/b.md', 'due')
            archive.writestr('libro.epub', epub.getvalue())
            archive.writestr('immagine.png', b'\x89PNG')
            archive.writestr('__MACOSX/a.txt', 'spazzatura')
        sources = [record['source'] for record in self.convert('tutto.zip', bundle.getvalue())]
        self.assertEqual(sources, ['a.txt', 'dir/b.md', 'libro.epub'])

    @unittest.skipUnless(importlib.util.find_spec('pypdf'), 'Requires pypdf')
    def test_pdf_without_text_is_refused(self):
        from pypdf import PdfWriter
        stream = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.write(stream)
        with self.assertRaisesRegex(ValueError, 'Nessun testo'):
            self.convert('scansione.pdf', stream.getvalue())

    def test_long_text_is_cut_at_paragraphs(self):
        from acan.lm.documents import PIECE, pieces
        text = '\n\n'.join(['parola '*1000]*400)
        cut = list(pieces(text))
        self.assertTrue(all(len(piece) <= PIECE for piece in cut))
        self.assertEqual(' '.join(cut).split(), text.split())                    # nothing lost or repeated


class MilestoneTest(unittest.TestCase):
    def test_old_checkpoints_on_a_milestone_are_kept(self):
        from acan.lm.checkpoints import retain
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for step in (500, 1000, 1500, 2000, 2500, 3000):
                (root/f'step-{step:012d}-0-x').mkdir()
            retain(root, keep=3, milestone=1000)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ['milestone-000000001000-0-x', 'step-000000002000-0-x', 'step-000000002500-0-x', 'step-000000003000-0-x'])
            retain(root, keep=1)
            self.assertEqual(sorted(path.name for path in root.iterdir()), ['milestone-000000001000-0-x', 'step-000000003000-0-x'])


@unittest.skipUnless(SERVICE, 'Requires LM service dependencies')
class QueueTest(unittest.TestCase):
    def setUp(self):
        from acan.lm.service import create_app
        self.folder = tempfile.TemporaryDirectory()
        self.root = self.folder.name
        self.client = create_app(self.root, 'x'*40).test_client()
        self.headers = {'Authorization': 'Bearer '+'x'*40, 'X-ACAN-Workspace': 'd'*64}

    def tearDown(self):
        self.folder.cleanup()

    def corpus(self, name, content):
        created = self.client.post('/corpora', headers=self.headers, json=dict(name=name, size=len(content))).get_json()
        self.client.post(f"/corpora/{created['id']}/chunks", headers={**self.headers, 'X-Offset': '0'}, data=content)
        return self.client.post(f"/corpora/{created['id']}/finish", headers=self.headers).get_json()

    def test_documents_upload_becomes_jsonl(self):
        corpus = self.corpus('appunti.md', '# Appunti\n\nTesto.'.encode())
        self.assertEqual((corpus['status'], corpus['name']), ('ready', 'appunti.md.jsonl'))
        refused = self.client.post('/corpora', headers=self.headers, json=dict(name='foto.png', size=10))
        self.assertEqual(refused.status_code, 400)

    def test_queue_runs_texts_in_order_with_warm_start(self):
        first, second, third = (self.corpus(f'{name}.txt', ('testo '*500).encode()) for name in ('uno', 'due', 'tre'))
        base = dict(protocol='controlled-v1', routing_profile='concept-graph-v1', graft_fraction=1., sequence_length=64,
                    device='cpu', name='Dialogo', passes=1, chat_probes=True, milestone_steps=1000, auto_resume=True,
                    early_stop_patience=0)
        with mock.patch('acan.lm.service.launch') as launch:
            answer = self.client.post('/queues', headers=self.headers, json=dict(base=base, corpora=[first['id'], second['id']]))
            self.assertEqual(answer.status_code, 200, answer.get_data(as_text=True))
            body = answer.get_json()
            self.assertEqual(launch.call_count, 1)
        jobs = body['jobs']
        self.assertEqual([job['config']['series']['order'] for job in jobs], [0, 1])
        self.assertEqual([job['config']['warm_start'] for job in jobs], [False, True])
        self.assertEqual({job['config']['graft_fraction'] for job in jobs}, {1.})               # all concepts
        added = self.client.post(f"/queues/{body['queue']}/add", headers=self.headers, json=dict(corpus_id=third['id'])).get_json()
        self.assertEqual((added['config']['series']['order'], added['config']['warm_start'], added['config']['passes']), (2, True, 1))
        self.assertTrue(added['config']['name'].startswith('Dialogo · 3 · tre.txt'))
        from acan.lm.research_trainer import warm_source
        from acan.lm.store import Store
        store = Store(self.root)
        with self.assertRaisesRegex(ValueError, 'non ha ancora un checkpoint'):
            warm_source(store, store.job(jobs[1]['id']))
        with self.assertRaisesRegex(ValueError, 'Primo file'):
            warm_source(store, store.job(jobs[0]['id']))

    def test_interrupted_long_training_resumes_by_itself_but_not_forever(self):
        from acan.lm.service import resume_after_restart
        from acan.lm.store import Store
        corpus = self.corpus('uno.txt', ('testo '*500).encode())
        store = Store(self.root)
        create = lambda **extra: self.client.post('/jobs', headers=self.headers, json=dict(
            corpus_id=corpus['id'], protocol='controlled-v1', routing_profile='concept-graph-v1', device='cpu', sequence_length=64, **extra)).get_json()
        job = create(auto_resume=True)
        store.update(job['id'], status='interrupted', progress=dict(step=40))
        with mock.patch('acan.lm.manager.launch') as launch:
            for _ in range(5):
                resume_after_restart(store)
        self.assertEqual(launch.call_count, 3)                               # three tries at the same step, then it waits
        manual = create()
        store.update(manual['id'], status='interrupted', progress=dict(step=1))
        with mock.patch('acan.lm.manager.launch') as launch:
            resume_after_restart(store)
        self.assertEqual(launch.call_count, 0)


class ProbesTest(unittest.TestCase):
    def test_chat_prompt_and_answers_file(self):
        from acan.lm.probes import chat_prompt, read_answers
        self.assertEqual(chat_prompt(' Ciao '), '<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\n')
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(read_answers(folder), [])
            path = Path(folder)/'answers.jsonl'
            path.write_text('\n'.join(json.dumps(dict(step=step, answers=[])) for step in range(60))+'\n')
            self.assertEqual([item['step'] for item in read_answers(folder, limit=3)], [57, 58, 59])


if __name__ == '__main__':
    unittest.main()


@unittest.skipUnless(importlib.util.find_spec('dash_bootstrap_components'), 'Requires the ui extra')
class LongTrainingPageTest(unittest.TestCase):
    def job(self, order, status, step=0, checkpoint='step-1', **progress):
        return dict(id=f'j{order}', status=status, checkpoint=checkpoint if status != 'created' else None, error=None,
                    progress=dict(step=step, **progress),
                    config=dict(name='LLM a concetti · x', total_steps=1000, passes=1, batch_size=1, accumulation=8, sequence_length=64,
                                window=16, slots=16, graft_fraction=1., series=dict(id='q1', order=order, variant='queue', label=f't{order}.txt')))

    def test_queue_status_and_current_item(self):
        from acan.vis.long_training import current_item, queue_table, queues
        jobs = [self.job(1, 'running', 400, validation_nll=3.1, best_validation_nll=3.05), self.job(0, 'completed', 1000), self.job(2, 'created'),
                dict(id='other', status='paused', checkpoint=None, progress={}, config=dict(name='altro'))]
        found = queues(jobs)
        self.assertEqual([job['id'] for job in found['q1']], ['j0', 'j1', 'j2'])
        self.assertEqual(current_item(found['q1'])['id'], 'j1')
        page = str(queue_table(found['q1']))
        for text in ('appreso', 'sta imparando', 'in attesa', '40%', 'passo 400 di 1,000', '3.100 (migliore 3.050)'):
            self.assertIn(text, page)
        self.assertEqual(current_item([self.job(0, 'completed'), self.job(1, 'created')])['id'], 'j1')

    def test_sizes_and_answers(self):
        from acan.vis.long_training import answers_view, footprint_view
        progress = dict(step=200, tokens_trained=102400, tokens_per_second=24., trainable_parameters=2_442_240, total_parameters=136_957_440,
                        training_nll=3.0, validation_nll=3.2, best_validation_nll=3.1, baseline_nll=3.6, learning_rate=1e-4, gradient_norm=7.,
                        process_peak_rss_kib=1_500_000, graft=dict(replaced_layers=list(range(30)), total_layers=30))
        item = dict(progress=progress, config=self.job(0, 'running')['config'], status='running', checkpoint_bytes=30*1024**2,
                    checkpoints=dict(step=dict(count=3, bytes=90*1024**2), milestone=dict(count=1, bytes=30*1024**2), best=dict(count=1, bytes=30*1024**2)),
                    token_bytes=4*1024**2, corpus_bytes=10*1024**2, disk_free=3*1024**3, disk_total=100*1024**3)
        page = str(footprint_view(item))
        for text in ('136,957,440', '1.8%', '30 su 30', '16 token esatti + 16 concetti', '24.0 token al secondo', '3 recenti, 1 permanenti, 1 migliore',
                     'Meno di 5 GB liberi', 'abbassare il passo'):
            self.assertIn(text, page)
        self.assertIn('Ancora nessuna risposta', str(answers_view([])))
        entries = [dict(step=100, validation_nll=3.4, answers=[dict(question='Ciao!', answer='ciao ciao ciao')]),
                   dict(step=200, validation_nll=3.2, answers=[dict(question='Ciao!', answer='Ciao! Come posso aiutarti?')])]
        page = str(answers_view(entries))
        self.assertIn('ciao ciao ciao', page)
        self.assertIn('Come posso aiutarti?', page)
        self.assertIn('confronto con il passo 100', page)


@unittest.skipUnless(SERVICE, 'Requires LM service dependencies')
class NativeModelTest(unittest.TestCase):
    """A concept model born from zero: its own tokenizer, random weights, every weight learns."""

    def test_tokenizer_model_and_options(self):
        import torch
        from transformers import AutoTokenizer
        from acan.lm.native import build, parameters, train_tokenizer
        from acan.lm.research_options import validate_options
        from acan.lm.resources import check_profile
        with tempfile.TemporaryDirectory() as folder:
            corpus = Path(folder)/'corpus'
            corpus.write_text('\n'.join(json.dumps(dict(text=f'<|im_start|>user\nCiao {i}<|im_end|>\n<|im_start|>assistant\nSalve {i}! Hello.<|im_end|>'))
                                        for i in range(300)), encoding='utf-8')
            train_tokenizer(corpus, Path(folder)/'tokenizer', vocabulary=4000)
            tokenizer = AutoTokenizer.from_pretrained(Path(folder)/'tokenizer')
            ids = tokenizer('<|im_start|>assistant\nCiao è là')['input_ids']
            self.assertEqual(tokenizer.convert_ids_to_tokens(ids)[:2], ['<|im_start|>', 'assistant'])   # turn markers stay whole
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), 'assistant\nCiao è là')      # any character, byte level
            options = validate_options(dict(corpus_id='a'*32, protocol='controlled-v1', routing_profile='concept-graph-v1',
                                            architecture='native', native_size='micro', vocabulary=4000, graft_fraction=1.,
                                            sequence_length=128, slots=16, window=16, device='cpu'))
            check_profile(options, False)
            with self.assertRaisesRegex(ValueError, 'GPU'):
                check_profile(dict(options, native_size='base'), False)
            with self.assertRaisesRegex(ValueError, 'grafo dei concetti'):
                validate_options(dict(options, routing_profile='cp-v9'))
            first, report = build(str(Path(folder)/'tokenizer'), options)
            second, _ = build(str(Path(folder)/'tokenizer'), options)
            trainable = {key: value for key, value in first.named_parameters() if value.requires_grad}
            self.assertGreater(sum(value.numel() for value in trainable.values()), .99*sum(value.numel() for value in first.parameters()))
            self.assertFalse(any('lora' in key for key in trainable))                                    # no adapters
            for (key, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items()):       # the seed fixes the model
                self.assertTrue(torch.equal(a, b), key)
            self.assertEqual((report['profile'], len(report['replaced_layers'])), ('acan-native-v1', 4))
            from acan.lm.concept_attention import ConceptGraphAttention
            self.assertEqual(sum(isinstance(module, ConceptGraphAttention) for module in first.modules()), 4)  # every layer
            batch = torch.tensor([ids*8])
            loss = first(batch, labels=batch).loss
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertAlmostEqual(parameters('micro', report['vocabulary'])/1e6, sum(value.numel() for value in first.parameters() if value.requires_grad)/1e6, delta=.1)
