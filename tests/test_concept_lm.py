"""The generative concept model: document memory, and a long learning of the queue from start to end."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

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


class ConceptLongLearningTests(unittest.TestCase):
    def test_pair_graph_queue_resume_generation_and_export(self):
        self._exercise_queue('concept-pairs-v1')

    def _exercise_queue(self, profile):
        from acan.lm.concept_trainer import evaluate, generate, run_training
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
                        sequence_length=64, validation_batches=4, save_steps=2, concept_piece=10_000,
                        chat_probes=True, probe_prompts=['Ciao!'], device='cpu')
            job = store.create_job(owner, validate_options(dict(base, corpus_id=first['id'], name='Coda · 1')))
            store.update(job['id'], status='queued')
            stop = threading.Event()

            def halt():
                while store.job(job['id'])['progress'].get('step', 0) < 1:
                    time.sleep(.05)
                stop.set()
            threading.Thread(target=halt, daemon=True).start()
            run_training(root, job['id'], stop)
            paused = store.job(job['id'])
            self.assertEqual(paused['status'], 'paused', paused['error'])
            run_training(root, job['id'])
            done = store.job(job['id'])
            self.assertEqual(done['status'], 'completed', done['error'])
            progress = done['progress']
            self.assertLess(progress['validation_nll'], progress['baseline_nll'])          # it learned
            self.assertEqual(progress['tokens_trained'], progress['dataset']['tokens']['train'])
            self.assertTrue(progress['concepts_per_level'] and progress['concepts_total'] > 0)
            from acan.lm.concept_lm import load
            from acan.core.concepts import ConceptModel
            saved = store.directory('jobs', job['id'])/'checkpoints'/done['checkpoint']
            graph, _ = load(saved)
            self.assertIsInstance(graph.model.root, ConceptModel)                         # the tabular concept model
            self.assertEqual(graph.tokens, progress['tokens_trained'])
            history = store.history(job['id'], owner)
            measured = [row['step'] for row in history if 'validation_nll' in row]
            self.assertEqual(measured[0], 0)                                              # the error before reading
            self.assertEqual(measured[-1], progress['step'])
            self.assertGreater(len(measured), 2)                                          # measured along the way, not only at saves
            self.assertLess(len(measured), len(history)+1)
            answers = [json.loads(line) for line in (store.directory('jobs', job['id'])/'answers.jsonl').read_text().splitlines()]
            self.assertTrue(answers and answers[-1]['answers'][0]['question'] == 'Ciao!')
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
            # The next text of the queue starts from the sums of the first and adds its own.
            config = validate_options(dict(base, corpus_id=second['id'], name='Coda · 2', warm_start=True))
            config['series'] = dict(id='q', order=1, variant='queue', label='due')
            store.set_config(job['id'], dict(done['config'], series=dict(id='q', order=0, variant='queue', label='uno')))
            following = store.create_job(owner, config)
            run_training(root, following['id'])
            after = store.job(following['id'])
            self.assertEqual(after['status'], 'completed', after['error'])
            self.assertEqual(after['progress']['warm_start']['job'], job['id'])
            self.assertGreater(after['progress']['tokens_trained'], 0)


if __name__ == '__main__':
    unittest.main()
