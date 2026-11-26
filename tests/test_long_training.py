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
        from acan.lm.retention import retain_with_milestones as retain
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
        refused = self.client.post('/corpora', headers=self.headers, json=dict(name='programma.exe', size=10))
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
        added = self.client.post(f"/queues/{body['queue']}/add", headers=self.headers,
                                 json=dict(corpus_id=third['id'], concept_network_growth=True,
                                           concept_max_dim=192, concept_max_layers=4,
                                           concept_max_heads=4, concept_max_concepts=32)).get_json()
        self.assertEqual((added['config']['series']['order'], added['config']['warm_start'], added['config']['passes']), (2, True, 1))
        self.assertEqual((added['config']['concept_max_dim'], added['config']['concept_max_layers'],
                          added['config']['concept_max_heads'], added['config']['concept_max_concepts']),
                         (192, 4, 4, 32))
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
            # New projects read through concept-graph-v2: the same weights, another reading layer.
            from acan.lm.graph_attention import ConceptGraphScanAttention
            newer, later = build(str(Path(folder)/'tokenizer'), dict(options, routing_profile='concept-graph-v2'))
            self.assertEqual(later['algebra'], 'concept-graph-v2')
            self.assertEqual(sum(isinstance(module, ConceptGraphScanAttention) for module in newer.modules()), 4)
            for (key, a), (_, b) in zip(first.state_dict().items(), newer.state_dict().items()):
                self.assertTrue(torch.equal(a, b), key)
            newer.eval()
            torch.manual_seed(0)
            mixed = torch.randint(0, report['vocabulary'], (1, 96))
            with torch.no_grad():                                 # a prompt read at once and continued token by token
                whole = newer(mixed, use_cache=False).logits
                start = newer(mixed[:, :40], use_cache=True)
                rest = newer(mixed[:, 40:], past_key_values=start.past_key_values, use_cache=True).logits
            # In single precision a nearly tied pair may be added together differently in the two
            # ways of reading: the readings stay close, they are not bit for bit the same.
            gap = (whole[:, 40:]-rest).abs()
            self.assertLess(float(gap.mean()), 5e-3)
            self.assertLess(float(gap.max()), .1*float(whole.abs().max()))
            self.assertAlmostEqual(parameters('micro', report['vocabulary'])/1e6, sum(value.numel() for value in first.parameters() if value.requires_grad)/1e6, delta=.1)


class ContractTest(unittest.TestCase):
    """Saved projects must stay readable after an update."""

    def test_the_checkpoint_contract_files_are_unchanged(self):
        # Every checkpoint is verified against a hash of attention.py, grafting.py, trainer.py, data.py and
        # checkpoints.py: changing any of them makes every saved project unreadable. Put new behaviour in
        # other modules (retention.py, native.py ...). Update this value only together with a migration.
        from acan.lm import checkpoints
        self.assertEqual(checkpoints.contract(), 'b43c8db4a1027a352ab8bf8a3b896116247b6a784e9989b18dfc7d00d17b2365')

    @unittest.skipUnless(SERVICE, 'Requires LM service dependencies')
    def test_projects_of_release_8_6_0_are_accepted(self):
        from acan.lm.compatibility import PRE_NATIVE_INFERENCE
        from acan.lm.research_model import check_inference_contract
        check_inference_contract(dict(routing_profile='concept-graph-v1', graft_fraction=1.), dict(model_contract=PRE_NATIVE_INFERENCE))
        with self.assertRaises(ValueError):
            check_inference_contract(dict(routing_profile='concept-graph-v1', architecture='native'), dict(model_contract=PRE_NATIVE_INFERENCE))


@unittest.skipUnless(SERVICE, 'Requires LM service dependencies')
class NativeControlTest(unittest.TestCase):
    def test_control_is_the_same_native_model_with_ordinary_attention(self):
        import shutil
        from acan.lm.native import build, train_tokenizer
        from acan.lm.service import create_app
        from acan.lm.store import Store
        with tempfile.TemporaryDirectory() as root:
            client = create_app(root, 'x'*40).test_client()
            headers = {'Authorization': 'Bearer '+'x'*40, 'X-ACAN-Workspace': 'e'*64}
            text = '\n'.join(json.dumps(dict(text=f'Frase numero {i}, una frase qualunque.')) for i in range(400)).encode()
            corpus = client.post('/corpora', headers=headers, json=dict(name='t.jsonl', size=len(text))).get_json()
            client.post(f"/corpora/{corpus['id']}/chunks", headers={**headers, 'X-Offset': '0'}, data=text)
            client.post(f"/corpora/{corpus['id']}/finish", headers=headers)
            job = client.post('/jobs', headers=headers, json=dict(corpus_id=corpus['id'], protocol='controlled-v1', routing_profile='concept-graph-v1',
                                                                  architecture='native', native_size='micro', vocabulary=4000, graft_fraction=1.,
                                                                  device='cpu', sequence_length=64)).get_json()
            store = Store(root)
            self.assertEqual(client.post(f"/jobs/{job['id']}/dense-control", headers=headers).status_code, 400)  # no tokenizer yet
            train_tokenizer(store.directory('corpora', corpus['id'])/'corpus', store.directory('jobs', job['id'])/'tokenizer', 4000)
            store.update(job['id'], status='paused', checkpoint='step-000000000001-0-x')
            with mock.patch('acan.lm.service.launch'):
                control = client.post(f"/jobs/{job['id']}/dense-control", headers=headers).get_json()
            self.assertEqual((control['config']['architecture'], control['config']['native_attention'], control['config']['control_of']),
                             ('native', 'dense', job['id']))
            from acan.lm.native import resolve
            path, base = resolve(store, store.job(control['id']), control['config'])
            self.assertEqual(base['files_sha256'], json.loads(json.dumps(
                {item.name: __import__('hashlib').sha256(item.read_bytes()).hexdigest()
                 for item in sorted((store.directory('jobs', job['id'])/'tokenizer').iterdir())})))    # the same tokenizer
            model, report = build(path, control['config'])
            from acan.lm.attention import ACANConceptAttentionLayer
            self.assertFalse(any(isinstance(module, ACANConceptAttentionLayer) for module in model.modules()))
            self.assertEqual(report['replaced_layers'], [])


@unittest.skipUnless(importlib.util.find_spec('PIL') and importlib.util.find_spec('numpy'), 'Requires Pillow')
class MediaTest(unittest.TestCase):
    """Pictures, sounds and animations as tokens, and back."""

    def picture(self=None, colour=(255, 0, 0), size=(64, 48), mode='PNG', **extra):
        from PIL import Image
        stream = io.BytesIO()
        Image.new('RGB', size, colour).save(stream, mode, **extra)
        return stream.getvalue()

    def test_image_round_trip(self):
        from PIL import Image
        from acan.lm.media import IMAGE_SIZE, encode, parts
        text = encode('foto.png', self.picture())
        self.assertEqual((text.count('<|row|>'), len(text.split('<|row|>')[3].split('|><|'))), (IMAGE_SIZE, IMAGE_SIZE))
        self.assertIn('<|p224|>', text)                                   # pure red: 3 bits red, nothing else
        kind, data = parts('Ecco: '+text+' fine')[1]
        self.assertEqual(kind, 'image')
        image = Image.open(io.BytesIO(data)).convert('RGB')
        self.assertEqual(image.getpixel((image.width//2, image.height//2)), (255, 0, 0))

    def test_sound_round_trip(self):
        import numpy as np
        import wave
        from acan.lm.media import AUDIO_RATE, encode, parts
        rate = 16000
        tone = (np.sin(2*np.pi*440*np.arange(rate//2)/rate)*20000).astype('<i2')
        stream = io.BytesIO()
        with wave.open(stream, 'wb') as file:
            file.setnchannels(1); file.setsampwidth(2); file.setframerate(rate); file.writeframes(tone.tobytes())
        text = encode('la.wav', stream.getvalue())
        self.assertEqual(len(__import__('re').findall(r'<\|a\d+\|>', text)), AUDIO_RATE//2)   # half a second at 8 kHz
        kind, data = parts(text)[0]
        with wave.open(io.BytesIO(data)) as file:
            back = np.frombuffer(file.readframes(file.getnframes()), dtype='<i2').astype(float)
        spectrum = np.abs(np.fft.rfft(back))
        self.assertAlmostEqual(np.argmax(spectrum)*AUDIO_RATE/len(back), 440, delta=5)   # still an A

    def test_animation_and_documents(self):
        from PIL import Image
        from acan.lm.media import VIDEO_FPS, encode, parts
        frames = [Image.new('RGB', (20, 20), colour) for colour in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
        stream = io.BytesIO()
        frames[0].save(stream, 'GIF', save_all=True, append_images=frames[1:], duration=int(1000/VIDEO_FPS), loop=0)
        text = encode('anim.gif', stream.getvalue())
        self.assertEqual(text.count('<|frame|>'), 3)
        self.assertEqual(parts(text)[0][0], 'video')
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, 'w') as archive:
            archive.writestr('a.png', self.picture())
            archive.writestr('nota.txt', 'una nota')
        from acan.lm.documents import convert
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'corpus'
            path.write_bytes(bundle.getvalue())
            convert(path, 'tutto.zip')
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(records[0]['text'].startswith('<|image|>') and records[0]['text'].endswith('<|/image|>'))   # one whole record

    @unittest.skipUnless(SERVICE, 'Requires LM service dependencies')
    def test_tokenizer_reads_media_whole_and_learns_from_every_text(self):
        from transformers import AutoTokenizer
        from acan.lm.media import encode, readable
        from acan.lm.native import train_tokenizer
        with tempfile.TemporaryDirectory() as folder:
            italian, code = Path(folder)/'it', Path(folder)/'code'
            italian.write_text('\n'.join(json.dumps(dict(text='La casa è bella e la strada è lunga.')) for _ in range(300)))
            code.write_text('\n'.join(json.dumps(dict(text='def somma(a, b):\n    return a + b\n')) for _ in range(300)))
            train_tokenizer([italian, code], Path(folder)/'tok', vocabulary=2000)
            tokenizer = AutoTokenizer.from_pretrained(Path(folder)/'tok')
            self.assertLessEqual(len(tokenizer('    return a + b')['input_ids']), 6)          # code got its own pieces
            self.assertLessEqual(len(tokenizer('La casa è bella')['input_ids']), 6)          # and so did Italian
            image = encode('x.png', self.picture())
            ids = tokenizer(image)['input_ids']
            self.assertEqual(len(ids), 32*32+32+2)                                           # one token per pixel and marker
            self.assertEqual(readable(tokenizer, ids), image)                                # kept for display


@unittest.skipUnless(SERVICE, 'Requires LM service dependencies')
class NativeSeriesTest(unittest.TestCase):
    def test_validation_series_on_native_models_and_page_profiles(self):
        from acan.lm.manager import series_configs
        from acan.lm.research_options import validate_options
        from acan.lm.resources import check_profile
        from acan.vis.long_training import profile
        native = dict(protocol='controlled-v1', routing_profile='concept-graph-v2', architecture='native', native_size='micro',
                      graft_fraction=1., vocabulary=32000, distillation_weight=0., sequence_length=256, batch_size=1,
                      accumulation=8, slots=16, window=16, learning_rate=1e-3, device='cpu')
        base = dict(native, corpus_id='a'*32, name='Validazione')
        (_, _, concepts), (_, _, dense) = series_configs(base, ['chosen', 'dense'])
        self.assertEqual((concepts.get('native_attention', 'concepts'), dense['native_attention'], dense['architecture']), ('concepts', 'dense', 'native'))
        check_profile(validate_options(dict(native, corpus_id='a'*32, sequence_length=1088)), False)   # a whole image
        for cuda in (False, True):                                          # the page learns the concept model, with or without GPU
            config = validate_options(dict(profile(cuda), corpus_id='a'*32))
            check_profile(config, cuda)
            self.assertEqual(config['device'], 'auto')
            self.assertEqual((config['architecture'], config['routing_profile']), ('concepts', 'concept-pairs-v1'))
        with self.assertRaisesRegex(ValueError, 'concept-pairs-v1'):
            validate_options(dict(profile(False), corpus_id='a'*32, routing_profile='concept-graph-v2'))



@unittest.skipUnless(importlib.util.find_spec('PIL') and importlib.util.find_spec('numpy'), 'Requires Pillow')
class FfmpegOnDemandTest(unittest.TestCase):
    """ffmpeg is not part of the worker: fetched, verified and kept only when a file needs it."""

    def test_only_compressed_media_ask_for_it_and_it_is_kept(self):
        import hashlib, platform
        from acan.lm import media
        with mock.patch.object(media, 'ffmpeg', side_effect=AssertionError('not needed')):
            media.encode('a.png', MediaTest.picture(None))                    # pictures, WAV and GIF never ask for it
        with tempfile.TemporaryDirectory() as folder:
            wheel = io.BytesIO()
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('imageio_ffmpeg/binaries/ffmpeg-linux-test', '#!/bin/sh\necho ffmpeg')
            raw = wheel.getvalue()
            pinned = {platform.machine().lower(): dict(url='x', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())}
            def download(file, target, check=lambda: None, attempts=5):
                Path(target).write_bytes(raw)
                return Path(target)
            with mock.patch('shutil.which', return_value=None), mock.patch.object(media, 'FFMPEG_WHEELS', {**pinned, 'x86_64': pinned[platform.machine().lower()],
                                                                                                          'aarch64': pinned[platform.machine().lower()]}), \
                    mock.patch('acan.lm.dialogue_data.download', side_effect=download) as fetched:
                first = media.ffmpeg(Path(folder)/'tools')
                second = media.ffmpeg(Path(folder)/'tools')
            self.assertEqual(first, second)
            self.assertEqual(fetched.call_count, 1)                            # downloaded once, then kept
            self.assertTrue(Path(first).read_text().startswith('#!/bin/sh'))
            self.assertEqual(sorted(path.name for path in (Path(folder)/'tools').iterdir() if not path.name.endswith('.lock')),
                             [Path(first).name])                               # the package itself is removed


class StructureProgressTest(unittest.TestCase):
    def test_the_structure_shows_how_far_it_is_and_what_is_left(self):
        from acan.vis.long_training import concept_summary, structure_progress
        start = dict(phase='tokens', width=1, seconds=0., step=1, steps=4)
        self.assertEqual(structure_progress(start), (0., None))
        half, left = structure_progress(dict(phase='scale', width=4, seconds=60., step=3, steps=4, inside_permille=0))
        self.assertAlmostEqual(half, .5)
        self.assertAlmostEqual(left, 60.)
        self.assertEqual(structure_progress(dict(phase='subconcepts', width=8, seconds=120., step=4, steps=4))[0], 1.)
        self.assertAlmostEqual(structure_progress(dict(phase='topics', width=0, seconds=60., step=4, steps=4,
                                                       part_permille=750))[0], .75)   # as learned, never back
        progress = dict(graft=dict(vocabulary=8), step=0,
                        structure=dict(phase='scale', width=4, seconds=60., step=3, steps=4, concepts=10, start=100))
        text = str(concept_summary(progress, dict(total_steps=45), status='running'))
        self.assertIn('struttura', text)
        self.assertIn('scala 3 di 4', text)
        progress['structure'] = dict(phase='network', race=2, done=500, piece=1000)
        self.assertIn('gara di crescita', str(concept_summary(progress, dict(total_steps=45), status='running')))
        self.assertNotIn('gara di crescita', str(concept_summary(progress, dict(total_steps=45), status='paused')))
        progress['structure'] = dict(phase='network', race=5, race_total=5, done=500, piece=1000)
        text = str(concept_summary(progress, dict(total_steps=45), status='running'))
        self.assertIn('gara di crescita al 90%', text)
        self.assertIn('5° di 5, il doppio delle teste', text)
        progress['structure'].update(race=4, candidate='concetti · strato 2')
        text = str(concept_summary(progress, dict(total_steps=45), status='running'))
        self.assertIn('4° di 5, concetti · strato 2', text)

    def test_growth_panel_shows_completed_scores_and_what_would_happen(self):
        from acan.vis.long_training import concept_summary

        network = dict(phase='network', race=4, race_total=5, race_round=2, candidate='concetti · strato 2',
                       done=750, piece=1000, race_candidates=["com'è", 'larghezza', 'strato', 'concetti · strato 2', 'teste'],
                       race_scores=[6.02, 5.993, 6.01],
                       network=dict(dim=32, layers=1, groups=[[1, 16]], parameters=2048, bytes=8192),
                       candidate_network=dict(dim=32, layers=1, groups=[[1, 17]], parameters=2304, bytes=9216),
                       race_projection=dict(candidate='larghezza', held_out=5.993, grows=True, retained=False))
        progress = dict(graft=dict(vocabulary=8), structure=network)
        config = dict(total_steps=45)
        text = str(concept_summary(progress, config, status='running'))
        for detail in ('tornata 2', 'In testa per NLL', '5.993', 'In prova · 75%', 'il doppio delle teste', 'In attesa',
                       'Se i punteggi confermati restano decisivi', 'Nuova tornata sullo stesso pezzo',
                       'Rete corrente', '1×16', '2,048 parametri', '8.0 KB', '1×17', '2,304 parametri', '9.0 KB'):
            self.assertIn(detail, text)

        network.update(race_evaluating=True, race_reference_nll=5.98,
                       race_projection=dict(candidate="com'è", held_out=5.98, grows=False, retained=True))
        text = str(concept_summary(progress, config, status='running'))
        self.assertIn('In validazione', text)
        self.assertIn('NLL della rete di partenza 5.980', text)
        self.assertIn('si conserva la rete di partenza (5.980)', text)
        self.assertIn('La gara per questo pezzo si chiude', text)
        self.assertNotIn('In testa per NLL', str(concept_summary(progress, config, status='paused')))
