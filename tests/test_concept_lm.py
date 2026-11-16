"""The concept language model: sums, reading, budget, saving, and a long learning from start to end."""
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from acan.lm.concept_lm import ConceptLM, Table, context_keys, document_counts


def brute(ids, depth, document, a, b, V):
    """The reading written position by position, from its definition."""
    out = []
    for t in range(1, len(ids)):
        p = 1./V
        for j in range(depth+1):
            if t < j:
                continue
            context = tuple(ids[t-j:t])
            members = [u for u in range(j, len(ids)) if tuple(ids[u-j:u]) == context]
            C = len(members); N = sum(ids[u] == ids[t] for u in members); D = len({ids[u] for u in members})
            if C:
                w = a[j]*max(D, 1); p = (N+w*p)/(C+w)
        for j in range(1, document+1):
            if t < j:
                continue
            context = tuple(ids[t-j:t])
            members = [u for u in range(j, t) if tuple(ids[u-j:u]) == context]
            C = len(members); N = sum(ids[u] == ids[t] for u in members); D = len({ids[u] for u in members})
            w = b[j]*max(D, 1); p = (N+w*p)/(C+w)
        out.append(np.log(p))
    return np.array(out)


class ConceptLMTests(unittest.TestCase):
    def test_reading_is_the_definition(self):
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 4, 120)
        model = ConceptLM(5, depth=3, document=2)
        model.add(ids); model.settle()
        model.a, model.b = np.array([.5, 1., 2., 4.]), np.array([1., .5, 2.])
        np.testing.assert_allclose(model.logp(ids, np.arange(1, len(ids))), brute(ids, 3, 2, model.a, model.b, 5), atol=1e-12)

    def test_document_counts_see_only_earlier_positions(self):
        rng = np.random.default_rng(1)
        ids = rng.integers(0, 4, 200)
        keys, valid = context_keys(ids, 3, 5)
        for j in (1, 2, 3):
            fast = document_counts(keys[j], valid[j], ids)
            for t in range(len(ids)):
                if t < j:
                    self.assertEqual(fast[:, t].sum(), 0); continue
                earlier = [u for u in range(j, t) if tuple(ids[u-j:u]) == tuple(ids[t-j:t])]
                expected = (len(earlier), sum(ids[u] == ids[t] for u in earlier), len({ids[u] for u in earlier}))
                self.assertEqual(tuple(fast[:, t]), expected)

    def test_pieces_added_in_any_order_are_the_whole_text(self):
        rng = np.random.default_rng(2)
        ids = rng.integers(0, 7, 3000)
        whole = ConceptLM(8, depth=4, document=0); whole.add(ids); whole.settle()
        pieces = ConceptLM(8, depth=4, document=0)
        for start in range(0, 3000, 300):                       # contexts do not cross the pieces
            pieces.add(ids[start:start+300])
        pieces.settle()
        self.assertEqual([int(t.C.sum()) for t in pieces.levels][0], 3000)
        for j in range(5):
            np.testing.assert_array_equal(np.sort(whole.levels[j].keys), whole.levels[j].keys)
            self.assertLessEqual(int(pieces.levels[j].C.sum()), int(whole.levels[j].C.sum()))
            self.assertGreaterEqual(int(pieces.levels[j].C.sum()), int(whole.levels[j].C.sum())-10*j)
        merged = ConceptLM(8, depth=4, document=0); merged.add(ids[:1500]); merged.settle()
        other = ConceptLM(8, depth=4, document=0); other.add(ids[1500:]); other.settle()
        merged.merge(other)
        self.assertEqual(merged.tokens, 3000)
        self.assertEqual(int(merged.levels[0].C.sum()), 3000)

    def test_budget_generalizes_the_cheapest_concepts_and_keeps_level_zero(self):
        rng = np.random.default_rng(3)
        ids = [0]                                               # the next token depends on the previous one only
        for _ in range(20000):
            ids.append((ids[-1]*7+rng.integers(0, 3)) % 30)
        ids = np.array(ids)
        free = ConceptLM(31, depth=4, document=0); free.add(ids); free.settle()
        tight = ConceptLM(31, depth=4, document=0, budget_bytes=free.nbytes()//4); tight.add(ids); tight.settle()
        self.assertLessEqual(tight.nbytes(), free.nbytes()//4)
        self.assertGreater(tight.generalized, 0)
        np.testing.assert_array_equal(tight.levels[0].counts, free.levels[0].counts)
        self.assertEqual(tight.concepts()[1], free.concepts()[1])      # the contexts that carry the information stay
        self.assertLess(sum(tight.concepts()[2:]), sum(free.concepts()[2:]))    # longer ones add nothing and go first

    def test_budget_holds_for_all_the_sums_while_learning(self):
        rng = np.random.default_rng(6)
        ids = rng.integers(0, 50, 40000)
        free = ConceptLM(51, depth=5, document=0)
        for start in range(0, 40000, 2000):
            free.add(ids[start:start+2000])
        budget = free.nbytes()//3
        model = ConceptLM(51, depth=5, document=0, budget_bytes=budget)
        for start in range(0, 40000, 2000):                     # every piece, not only at the end
            model.add(ids[start:start+2000])
            self.assertLessEqual(model.nbytes(), budget)
        model.settle()
        self.assertEqual(int(model.levels[0].C.sum()), 40000)   # the sums of level 0 are all there
        self.assertGreater(model.generalized, 0)

    def test_sum_of_tables_is_the_sum_of_their_cells(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            x = Table.count(rng.integers(0, 40, 500).astype(np.uint64), rng.integers(0, 6, 500), 6)
            y = Table.count(rng.integers(20, 60, 300).astype(np.uint64), rng.integers(0, 6, 300), 6)
            z = x+y
            expected = {}
            for t in (x, y):
                for k, v, n in zip(*t.pairs()):
                    expected[(int(k), int(v))] = expected.get((int(k), int(v)), 0)+int(n)
            self.assertEqual({(int(k), int(v)): int(n) for k, v, n in zip(*z.pairs())}, expected)
            self.assertEqual(dict(zip(z.keys.tolist(), z.C.tolist())),
                             {k: dict(zip(x.keys.tolist(), x.C.tolist())).get(k, 0)+dict(zip(y.keys.tolist(), y.C.tolist())).get(k, 0)
                              for k in set(x.keys.tolist()) | set(y.keys.tolist())})
            np.testing.assert_array_equal(z.D, np.bincount(z.cells//7, minlength=len(z.keys)))

    @unittest.skipUnless(importlib.util.find_spec('safetensors'), 'Checkpoints need safetensors (the worker has it)')
    def test_saved_and_loaded_model_reads_the_same(self):
        from acan.lm.concept_lm import load, save
        rng = np.random.default_rng(4)
        ids = rng.integers(0, 9, 2000)
        model = ConceptLM(10, depth=3, document=2); model.add(ids); model.settle()
        model.a[2], model.b[1] = 3., .25
        with tempfile.TemporaryDirectory() as folder:
            path = save(folder, model, dict(step=1, validation_nll=1.), 'identity')
            again, state = load(path, 'identity')
        np.testing.assert_allclose(again.logp(ids), model.logp(ids))
        self.assertEqual(state['step'], 1)
        self.assertEqual(again.generate(ids[:10], 20, temperature=.7, seed=3), model.generate(ids[:10], 20, temperature=.7, seed=3))

    def test_the_text_being_read_is_remembered_at_any_distance(self):
        rng = np.random.default_rng(5)
        model = ConceptLM(40, depth=2, document=2)
        model.add(rng.integers(0, 30, 5000)); model.settle()
        model.b[:] = .03
        pair = [31, 35]                                         # a key and its value said once, far before
        text = np.r_[pair, rng.integers(0, 30, 400), 31]
        self.assertEqual(int(np.argmax(model.distribution(text))), 35)


@unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'), 'Tokenizer required')
class ConceptLongLearningTests(unittest.TestCase):
    def test_queue_pause_resume_answers_generation_test_and_export(self):
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
            base = dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-lm-v1', vocabulary=4000,
                        sequence_length=64, validation_batches=4, save_steps=2, concept_piece=10_000, concept_depth=4,
                        concept_document=2, chat_probes=True, probe_prompts=['Ciao!'], device='cpu')
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
            names = zipfile.ZipFile(export_bundle(store, done, done['checkpoint'])).namelist()
            self.assertIn('concepts.safetensors', names)
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
