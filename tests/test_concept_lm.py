"""The generative concept model: document memory, and a long learning of the queue from start to end."""
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from acan.lm.concept_lm import OLD_CHECKPOINT, document_counts, from_checkpoint_arrays


def context_keys(ids, depth):
    """Exact keys of the last j tokens for j = 0..depth; a position with fewer tokens is invalid."""
    ids = np.asarray(ids, np.int64)
    keys, valid = [np.zeros(len(ids), np.uint64)], [np.ones(len(ids), bool)]
    key = np.zeros(len(ids), np.uint64)
    for j in range(1, depth+1):
        shifted = np.zeros(len(ids), np.uint64)
        shifted[j:] = (ids[:-j]+1).astype(np.uint64)
        key = key*np.uint64(1000)+shifted
        keys.append(key.copy())
        ok = np.zeros(len(ids), bool); ok[j:] = True
        valid.append(ok)
    return keys, valid


class DocumentMemoryTests(unittest.TestCase):
    def test_document_counts_see_only_earlier_positions(self):
        rng = np.random.default_rng(1)
        ids = rng.integers(0, 4, 200)
        keys, valid = context_keys(ids, 3)
        for j in (1, 2, 3):
            fast = document_counts(keys[j], valid[j], ids)
            for t in range(len(ids)):
                if t < j:
                    self.assertEqual(fast[:, t].sum(), 0); continue
                earlier = [u for u in range(j, t) if tuple(ids[u-j:u]) == tuple(ids[t-j:t])]
                expected = (len(earlier), sum(ids[u] == ids[t] for u in earlier), len({ids[u] for u in earlier}))
                self.assertEqual(tuple(fast[:, t]), expected)

    def test_checkpoints_of_removed_models_say_so(self):
        for arrays in ({'level0.keys': np.zeros(1)}, {'text_graph': np.zeros(1)}):
            with self.assertRaisesRegex(ValueError, 'non fa più parte del prodotto'):
                from_checkpoint_arrays(arrays)
        self.assertIn('Riaddestra', OLD_CHECKPOINT)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('transformers', 'tokenizers', 'torch')),
                     'Worker dependencies required')
class ConceptLongLearningTests(unittest.TestCase):
    def test_pair_graph_queue_resume_generation_and_export(self):
        self._exercise_queue('concept-pairs-v1')

    def _exercise_queue(self, profile):
        from acan.lm.concept_trainer import evaluate, generate, run_training
        from acan.lm.net_text import NetText
        from acan.lm.research_options import validate_options
        from acan.lm.store import Store, owner_key
        words = 'il gatto dorme sul tetto e la luna guarda la città mentre il cane abbaia piano'.split()
        rng = np.random.default_rng(0)

        def corpus(store, owner, name, documents):
            lines = []
            for i in range(documents):
                body = ' '.join(rng.choice(words, 40))
                lines.append(json.dumps(dict(text=f'<|im_start|>user\nCiao {i}<|im_end|>\n<|im_start|>assistant\n{body}<|im_end|>')))
            raw = ('\n'.join(lines)+'\n').encode()
            item = store.create_corpus(owner, name, len(raw))
            store.append_corpus(item['id'], owner, 0, raw); store.finish_corpus(item['id'], owner)
            return item
        with tempfile.TemporaryDirectory() as root:
            store = Store(root); owner = owner_key('a'*64)
            first, second = corpus(store, owner, 'uno.jsonl', 1500), corpus(store, owner, 'due.jsonl', 300)
            base = dict(protocol='controlled-v1', architecture='concepts', routing_profile=profile, vocabulary=4000,
                        sequence_length=64, validation_batches=4, save_steps=2, concept_piece=10_000, concept_context=8,
                        chat_probes=True, probe_prompts=['Ciao!'], device='cpu')
            job = store.create_job(owner, validate_options(dict(base, corpus_id=first['id'], name='Coda · 1')))
            store.update(job['id'], status='queued')
            stop = threading.Event()

            def halt():
                while store.job(job['id'])['progress'].get('step', 0) < 1:
                    time.sleep(.05)
                stop.set()
            threading.Thread(target=halt, daemon=True).start()
            original_add = NetText.add

            def report_after_compaction(model, ids, *args):
                if not model.trace:
                    model.piece = len(ids)
                    for position in range(514):
                        model._point(position, np.array([-4.0]), 'larghezza')
                    model.report('network', 0, 0., race=2, done=513, piece=len(ids))
                    live = store.job(job['id'])['progress']
                    self.assertEqual(live['step'], 0)
                    self.assertEqual(live['network_trace'], model.trace)
                    self.assertEqual(live['network_trace'][-1][2], 'larghezza')
                return original_add(model, ids, *args)

            with patch.object(NetText, 'add', report_after_compaction):
                run_training(root, job['id'], stop)
            paused = store.job(job['id'])
            self.assertEqual(paused['status'], 'paused', paused['error'])
            self._transfer_paused_project(store, job, paused)
            run_training(root, job['id'])
            done = store.job(job['id'])
            self.assertEqual(done['status'], 'completed', done['error'])
            progress = done['progress']
            self.assertLess(progress['validation_nll'], progress['baseline_nll'])          # it learned
            self.assertEqual(progress['tokens_trained'], progress['dataset']['tokens']['train'])
            self.assertTrue(progress['concepts_per_level'] and progress['concepts_total'] > 0)
            from acan.lm.concept_lm import load
            saved = store.directory('jobs', job['id'])/'checkpoints'/done['checkpoint']
            graph, saved_state = load(saved)
            self.assertEqual(saved_state['network_trace'], progress['network_trace'])
            self.assertNotIn('structure', saved_state)
            self.assertIsInstance(graph, NetText)                                         # the concept network alone
            self.assertIsNotNone(graph.model.net)
            self.assertEqual(graph.tokens, progress['tokens_trained'])
            history = store.history(job['id'], owner)
            measured = [row['step'] for row in history if 'validation_nll' in row]
            self.assertEqual(measured[0], 0)                                              # the error before reading
            self.assertEqual(measured[-1], progress['step'])
            self.assertGreater(len(measured), 2)                                          # measured along the way, not only at saves
            used = [row['concepts_used'] for row in history if 'concepts_used' in row]   # concepts used, step by step
            self.assertTrue(used and len(used[-1]) == len(graph.model.net.config['concepts']))
            self.assertTrue(all(0 < u <= k for u, k in zip(used[-1], graph.model.net.config['concepts'])))
            self.assertLess(len(measured), len(history)+1)
            answers = [json.loads(line) for line in (store.directory('jobs', job['id'])/'answers.jsonl').read_text().splitlines()]
            self.assertTrue(answers and answers[-1]['answers'][0]['question'] == 'Ciao!')
            # How the last answers were written, token by token, for the walk of the page (latest save only).
            walked = json.loads((store.directory('jobs', job['id'])/'answer_paths.json').read_text())
            self.assertEqual((walked['step'], walked['questions'][0]), (answers[-1]['step'], 'Ciao!'))
            first = next(path for path in walked['paths'] if path)
            self.assertTrue(first['steps'] and first['steps'][0]['levels'])
            checkpoint = store.directory('jobs', job['id'])/'checkpoints'/done['checkpoint']
            result = generate(store, done, dict(prompt='Ciao!', chat=True, max_new_tokens=20, temperature=0., seed=0,
                                                checkpoint=done['checkpoint']), checkpoint)
            self.assertTrue(result['concept_model'] and result['generated_tokens'] <= 20)
            scores = evaluate(store, done, dict(test_batches=4, context_lengths=[32, 400], new_tokens=8), checkpoint, lambda: None)
            self.assertTrue(np.isfinite(scores['nll']) and scores['tokens'] > 0)
            self.assertEqual([row['prompt_tokens'] for row in scores['performance']], [32, 400])
            self.assertTrue(all(row['cache']['total'] > 0 and row['decode_tokens_per_second'] > 0 for row in scores['performance']))
            from acan.lm.exporting import export_bundle
            import zipfile
            archive = zipfile.ZipFile(export_bundle(store, done, done['checkpoint']))
            names = archive.namelist()
            self.assertIn('concepts.safetensors', names)
            self.assertTrue(any(name.startswith('tokenizer/') for name in names))
            with tempfile.TemporaryDirectory() as unpacked:          # the downloaded model writes on its own
                archive.extractall(unpacked)
                import subprocess, sys
                written = subprocess.run([sys.executable, str(Path(unpacked)/'scrivi.py'), 'Ciao!', '--token', '12'],
                                         capture_output=True, text=True, timeout=120)
                self.assertEqual(written.returncode, 0, written.stderr[-2000:])
                self.assertTrue(written.stdout.strip())
            # The next text of the queue starts from the network of the first and goes on learning.
            config = validate_options(dict(base, corpus_id=second['id'], name='Coda · 2', warm_start=True))
            config['series'] = dict(id='q', order=1, variant='queue', label='due')
            store.set_config(job['id'], dict(done['config'], series=dict(id='q', order=0, variant='queue', label='uno')))
            following = store.create_job(owner, config)
            run_training(root, following['id'])
            after = store.job(following['id'])
            self.assertEqual(after['status'], 'completed', after['error'])
            self.assertEqual(after['progress']['warm_start']['job'], job['id'])
            self.assertGreater(after['progress']['tokens_trained'], 0)

    def _transfer_paused_project(self, source, job, paused):
        from acan.lm.concept_lm import load
        from acan.lm.concept_trainer import run_training
        from acan.lm.service import create_app
        from acan.lm.store import owner_key

        token = 'x'*32
        source_headers = {'Authorization': 'Bearer '+token, 'X-ACAN-Workspace': 'a'*64}
        target_headers = dict(source_headers, **{'X-ACAN-Workspace': 'b'*64})
        source_client = create_app(str(source.root), token).test_client()
        response = source_client.get(f"/jobs/{job['id']}/project-archive", headers=source_headers)
        self.assertEqual(response.status_code, 200)
        with response:
            archive = response.data
        with tempfile.TemporaryDirectory() as target_root:
            target = create_app(target_root, token)
            client = target.test_client()
            transfer = client.post('/project-imports', headers=target_headers, json={'size': len(archive)})
            self.assertEqual(transfer.status_code, 200)
            url = '/project-imports/'+transfer.json['id']
            self.assertEqual(client.post(url+'/finish', headers=source_headers).status_code, 404)
            self.assertEqual(client.post(url+'/finish', headers=target_headers).status_code, 400)
            for offset in range(0, len(archive), 1024**2):
                result = client.post(url+'/chunk', headers=dict(target_headers, **{'X-Offset': str(offset)}),
                                     data=archive[offset:offset+1024**2])
                self.assertEqual(result.status_code, 200, result.get_data(as_text=True))
            result = client.post(url+'/finish', headers=target_headers)
            self.assertEqual(result.status_code, 200, result.get_data(as_text=True))
            identifier = result.json['id']
            self.assertNotEqual(identifier, job['id'])
            imported = target.store.job(identifier, owner_key(target_headers['X-ACAN-Workspace']))
            self.assertEqual(imported['status'], 'paused')
            step = paused['progress']['step']
            self.assertEqual(imported['progress']['step'], step)
            src_checkpoint = source.directory('jobs', job['id'])/'checkpoints'/paused['checkpoint']
            dst_checkpoint = target.store.directory('jobs', identifier)/'checkpoints'/imported['checkpoint']
            original, _ = load(src_checkpoint)
            restored, _ = load(dst_checkpoint)
            np.testing.assert_array_equal(original.logp(np.arange(1, 11)), restored.logp(np.arange(1, 11)))
            stop = threading.Event()

            def halt_after_step():
                while not stop.wait(.02):
                    if target.store.job(identifier)['progress']['step'] > step:
                        stop.set()
            watcher = threading.Thread(target=halt_after_step, daemon=True)
            watcher.start()
            try:
                run_training(target_root, identifier, stop)
            finally:
                stop.set()
                watcher.join()
            resumed = target.store.job(identifier)
            self.assertGreater(resumed['progress']['step'], step, resumed['error'])


if __name__ == '__main__':
    unittest.main()
