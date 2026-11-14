"""Offline gates for causal grafting, safe persistence and exact CPU continuation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

LM = all(importlib.util.find_spec(name) for name in ('torch','transformers','peft','safetensors'))


class TestDurableStore(unittest.TestCase):
    def test_corpus_streaming_ownership_and_restart(self):
        from acan.lm.store import Store, owner_key
        with tempfile.TemporaryDirectory() as root:
            store = Store(root)
            owner, other = owner_key('a'*64), owner_key('b'*64)
            corpus = store.create_corpus(owner, 'notes.txt', 6)
            store.append_corpus(corpus['id'], owner, 0, b'hello ')
            with self.assertRaises(KeyError):
                store.corpus(corpus['id'], other)
            ready = store.finish_corpus(corpus['id'], owner)
            job = store.create_job(owner, dict(corpus_id=corpus['id']))
            store.update(job['id'], status='running', checkpoint='immutable-snapshot')
            restarted = Store(root)
            self.assertEqual(restarted.job(job['id'], owner)['checkpoint'], 'immutable-snapshot')
            restarted.reconcile()
            self.assertEqual(restarted.job(job['id'], owner)['status'], 'interrupted')
            with self.assertRaises(KeyError):
                restarted.job(job['id'], other)
            self.assertEqual(len(ready['sha256']), 64)
            from acan.lm.checkpoints import for_resume
            empty=Path(root)/'empty-checkpoints';empty.mkdir()
            self.assertIsNone(for_resume(empty,'fixture'))
            with self.assertRaises(ValueError):
                for_resume(empty,'fixture',saved_name='missing-learned-checkpoint')
            with self.assertRaises(ValueError):
                for_resume(empty,'fixture',completed_steps=1)



@unittest.skipUnless(LM, 'Requires the separate llm extra')
class TestCausalHybrid(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def model(self):
        from transformers import LlamaConfig
        from acan.lm.grafting import build_from_config
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=4, num_attention_heads=4,num_key_value_heads=2,
                             max_position_embeddings=128)
        return build_from_config(config, dict(slots=4,window=4,lora_rank=2))[0]

    def test_future_suffix_cannot_change_prefix_logits(self):
        import torch
        model = self.model().eval()
        x = torch.randint(0,32,(2,12)); y = x.clone(); y[:,6:] = torch.randint(0,32,(2,6))
        with torch.inference_mode():
            a=model(x,use_cache=False).logits; b=model(y,use_cache=False).logits
        torch.testing.assert_close(a[:,:6],b[:,:6], rtol=0,atol=0)
        self.assertGreater(float((a[:,6:]-b[:,6:]).abs().max()), 0)

    def test_streaming_matches_prefix_and_allocated_slot_cache_is_bounded(self):
        import torch
        from acan.lm.attention import ACANConceptAttentionLayer
        from acan.lm.grafting import reset_streams
        model=self.model().eval(); x=torch.randint(0,32,(1,24))
        with torch.inference_mode():
            expected=model(x,use_cache=False).logits
            reset_streams(model); cache=None; logits=[]; sizes=[]
            for i in range(x.shape[1]):
                result=model(x[:,i:i+1],past_key_values=cache,use_cache=True)
                cache=result.past_key_values; logits.append(result.logits)
                if i in (7,15,23):
                    sizes.append(sum(module.cache_bytes() for module in model.modules() if isinstance(module,ACANConceptAttentionLayer)))
        # A detached last-position view must not retain an entire prefill storage.
        with torch.inference_mode():
            reset_streams(model);model(x,use_cache=True)
        for module in model.modules():
            if isinstance(module,ACANConceptAttentionLayer):
                for item in (module.cache, module.last_info):
                    for tensor in item.values():
                        if isinstance(tensor,torch.Tensor):
                            self.assertEqual(tensor.untyped_storage().nbytes(),tensor.numel()*tensor.element_size())
        torch.testing.assert_close(torch.cat(logits,1),expected,atol=2e-6,rtol=1e-5)
        self.assertEqual(len(set(sizes)),1)
        self.assertGreater(sizes[0],0)

    def test_graft_preserves_embeddings_ffn_and_only_adapter_parameters_train(self):
        import torch
        from transformers import LlamaConfig,AutoModelForCausalLM
        from acan.lm.grafting import graft,add_lora
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
                           num_attention_heads=4,num_key_value_heads=2)
        base=AutoModelForCausalLM.from_config(config)
        embedding=base.model.embed_tokens.weight.detach().clone()
        ffn=base.model.layers[3].mlp.up_proj.weight.detach().clone()
        report=graft(base,4,4)
        self.assertEqual(report['replaced_layers'],[2,3])
        torch.testing.assert_close(base.model.embed_tokens.weight,embedding)
        torch.testing.assert_close(base.model.layers[3].mlp.up_proj.weight,ffn)
        model=add_lora(base,2)
        trainables={name for name,p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all('lora_' in name or name.endswith('.G') or name.endswith('.omega') for name in trainables))
        x=torch.randint(0,32,(1,8)); model(x,labels=x,use_cache=False).loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_resume_matches_uninterrupted_optimizer_and_rng(self):
        import random
        import numpy as np
        import torch
        from acan.lm import checkpoints
        model=self.model().train()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        x=torch.randint(0,32,(1,8))
        def step(m,opt):
            opt.zero_grad();loss=m(x,labels=x,use_cache=False).loss;loss.backward();opt.step()
        step(model,optimizer)
        with tempfile.TemporaryDirectory() as root:
            identity=checkpoints.fingerprint(dict(base='offline-fixture',seed=42))
            path=checkpoints.save(root,model,optimizer,dict(step=1,cursor=dict(epoch=0,offset=1)),identity)
            step(model,optimizer)
            expected={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
            expected_random=(random.random(),float(np.random.random()),float(torch.rand(())))
            torch.manual_seed(42)
            resumed=self.model().train()
            opt=torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=.001)
            state=checkpoints.restore(path,resumed,opt,identity)
            self.assertEqual(state['cursor']['offset'],1)
            step(resumed,opt)
            for name,p in resumed.named_parameters():
                if p.requires_grad:
                    torch.testing.assert_close(p,expected[name],atol=0,rtol=0)
            actual=(random.random(),float(np.random.random()),float(torch.rand(())))
            self.assertEqual(actual,expected_random)
            (Path(root)/'.step-999999.tmp').mkdir()
            self.assertEqual(checkpoints.latest(root,identity),path)
            with (path/'trainables.safetensors').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaises(ValueError):
                checkpoints.verify(path,identity)

    def test_wsd_continuation_and_deterministic_data_cursor(self):
        import numpy as np
        import torch
        from acan.lm.data import Blocks
        from acan.lm.trainer import wsd_lr
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'tokens.bin'; np.arange(1000,dtype='<u4').tofile(path)
            blocks=Blocks(path,16)
            state=dict(epoch=0,offset=0)
            first=blocks.batch(3,state); copied=dict(state)
            expected=blocks.batch(3,state)
            resumed=Blocks(path,16)
            actual=resumed.batch(3,copied)
            torch.testing.assert_close(actual,expected)
            self.assertLess(int(first.max()),min(blocks.validation_ids)*17)
            rates=[wsd_lr(i,100,.001) for i in range(100)]
            self.assertLess(rates[0],rates[5]);self.assertEqual(rates[10],rates[70]);self.assertLess(rates[-1],rates[80])


@unittest.skipUnless(LM and importlib.util.find_spec('flask'), 'Requires LM service dependencies')
class TestLMService(unittest.TestCase):
    def test_api_restart_preserves_live_lease_and_pause_then_marks_dead_worker(self):
        import fcntl
        from acan.lm.service import create_app
        from acan.lm.store import Store, owner_key
        token='internal-test-token-'+'x'*32
        headers={'Authorization':'Bearer '+token,'X-ACAN-Workspace':'a'*64}
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);owner=owner_key('a'*64)
            corpus=store.create_corpus(owner,'sample.txt',1)
            store.append_corpus(corpus['id'],owner,0,b'x');store.finish_corpus(corpus['id'],owner)
            job=store.create_job(owner,dict(corpus_id=corpus['id']))
            store.update(job['id'],status='running',checkpoint='last-good')
            lease=store.resource();fcntl.flock(lease,fcntl.LOCK_EX)
            try:
                client=create_app(root,token).test_client()
                self.assertEqual(client.get('/state',headers=headers).get_json()['jobs'][0]['status'],'running')
                self.assertEqual(client.post(f"/jobs/{job['id']}/pause",headers=headers).get_json()['status'],'pausing')
            finally:
                lease.close()
            restarted=create_app(root,token).test_client()
            restored=restarted.get('/state',headers=headers).get_json()['jobs'][0]
            self.assertEqual(restored['status'],'interrupted')
            self.assertEqual(restored['checkpoint'],'last-good')

    def test_api_isolation_no_implicit_training_and_persistent_restart(self):
        from unittest.mock import patch
        from acan.lm.service import create_app
        token='internal-test-token-'+'x'*32
        headers={'Authorization':'Bearer '+token,'X-ACAN-Workspace':'a'*64}
        with tempfile.TemporaryDirectory() as root:
            app=create_app(root,token); client=app.test_client()
            self.assertEqual(client.get('/state').status_code,401)
            corpus=client.post('/corpora',headers=headers,json=dict(name='sample.txt',size=6)).get_json()
            cid=corpus['id']
            self.assertEqual(client.post(f'/corpora/{cid}/chunks',headers=dict(headers,**{'X-Offset':'0'}),data=b'hello ').status_code,200)
            client.post(f'/corpora/{cid}/finish',headers=headers)
            other=dict(headers,**{'X-ACAN-Workspace':'b'*64})
            self.assertEqual(client.post('/jobs',headers=other,json=dict(corpus_id=cid)).status_code,404)
            with patch('acan.lm.service.launch') as launch:
                created=client.post('/jobs',headers=headers,json=dict(corpus_id=cid,total_steps=2)).get_json()
                launch.assert_not_called()
                self.assertEqual(client.post(f"/jobs/{created['id']}/start",headers=headers).status_code,200)
                launch.assert_called_once()
            # A requested max token count must override defaults, not raise a
            # duplicate-keyword error before the worker can be queued.
            store=app.store
            snapshot=store.directory('jobs',created['id'])/'checkpoints'/'step-000000000000-fixture'
            snapshot.mkdir(parents=True)
            (snapshot/'manifest.json').write_text(json.dumps(dict(step=0)))
            store.update(created['id'],status='paused')
            with patch('acan.lm.manager.launch') as generation_launch:
                response=client.post(f"/jobs/{created['id']}/generate",headers=headers,
                    json=dict(checkpoint=snapshot.name,prompt='The river',max_new_tokens=4))
                self.assertEqual(response.status_code,200,response.get_data(as_text=True))
                generation_launch.assert_called_once()
                generation_id=response.get_json()['id']
            self.assertEqual(client.get('/generations/'+generation_id,headers=other).status_code,404)
            self.assertEqual(client.get('/state',headers=other).get_json()['jobs'],[])
            fresh=create_app(root,token).test_client()
            self.assertEqual(fresh.get('/state',headers=headers).get_json()['jobs'][0]['id'],created['id'])


@unittest.skipUnless(LM and importlib.util.find_spec('flask'), 'Requires LM service dependencies')
class TestPublicExperiments(unittest.TestCase):
    def fixture(self):
        import hashlib
        raw = b'The river flows through the valley.\n'*100
        spec = dict(title='Fixture', language='en', url='https://example.invalid/text',
                    source='https://example.invalid', license='Test only', license_url='https://example.invalid/license',
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), format='text',
                    note='Fixture only', prompt='The river')
        return raw, spec

    def test_explicit_download_start_reuse_provenance_and_isolation(self):
        from unittest.mock import patch
        from acan.lm import public_corpus as public
        from acan.lm.service import create_app
        raw, spec = self.fixture()
        token = 'internal-test-token-'+'x'*32
        headers = {'Authorization':'Bearer '+token, 'X-ACAN-Workspace':'a'*64}
        other = dict(headers, **{'X-ACAN-Workspace':'b'*64})
        with tempfile.TemporaryDirectory() as root, patch.dict(public.CATALOG, fixture=spec):
            app = create_app(root, token); client = app.test_client()
            with patch.object(public, 'download', return_value=raw) as download, patch.object(public, 'launch') as launch:
                self.assertEqual(client.get('/public-corpora', headers=headers).status_code, 200)
                client.get('/state', headers=headers)
                download.assert_not_called(); launch.assert_not_called()
                result = client.post('/public-experiments', headers=headers, json=dict(dataset='fixture', preset='check'))
                self.assertEqual(result.status_code, 200, result.get_data(as_text=True))
                data = result.get_json(); job = data['job']
                self.assertEqual(job['config']['total_steps'], 20)
                self.assertEqual(job['config']['accumulation'], 1)
                self.assertEqual(job['config']['base_model'], 'HuggingFaceTB/SmolLM2-135M')
                download.assert_called_once(); launch.assert_called_once()
                self.assertEqual(client.get('/state', headers=other).get_json()['corpora'], [])
                self.assertEqual(client.post('/jobs/'+job['id']+'/pause', headers=other).status_code, 404)
                second = client.post('/public-experiments', headers=headers, json=dict(dataset='fixture', preset='practice')).get_json()
                self.assertEqual(second['corpus']['id'], data['corpus']['id'])
                self.assertEqual(second['job']['config']['total_steps'], 200)
                self.assertEqual(download.call_count, 1)
                self.assertEqual(data['source']['source_sha256'], spec['sha256'])
                metadata = app.store.directory('jobs', job['id'])/'public-source.json'
                self.assertTrue(metadata.exists())
                fresh = create_app(root, token).test_client()
                self.assertEqual(len(fresh.get('/state', headers=headers).get_json()['jobs']), 2)
                with patch('acan.lm.service.launch'):             # the control names the project it controls
                    control = client.post('/jobs/'+job['id']+'/dense-control', headers=headers)
                self.assertEqual(control.status_code, 200, control.get_data(as_text=True))
                self.assertEqual(control.get_json()['config']['control_of'], job['id'])
                self.assertEqual(control.get_json()['config']['architecture'], 'dense-control')

    def test_bad_hash_unknown_source_and_busy_worker_cannot_start(self):
        import fcntl
        from unittest.mock import patch
        from acan.lm import public_corpus as public
        from acan.lm.service import create_app
        raw, spec = self.fixture()
        headers = {'Authorization':'Bearer '+'x'*40, 'X-ACAN-Workspace':'a'*64}
        with tempfile.TemporaryDirectory() as root, patch.dict(public.CATALOG, fixture=spec):
            app = create_app(root, 'x'*40); client = app.test_client()
            with patch.object(public, 'download', return_value=raw+b'corrupt') as download, patch.object(public, 'launch') as launch:
                result = client.post('/public-experiments', headers=headers, json=dict(dataset='fixture', preset='check'))
                self.assertEqual(result.status_code, 400)
                self.assertIn('checksum', result.get_json()['error'])
                self.assertEqual(client.get('/state', headers=headers).get_json()['jobs'], [])
                for dataset in ('https://example.invalid/custom', [], None):
                    self.assertEqual(client.post('/public-experiments', headers=headers,
                        json=dict(dataset=dataset, preset='check')).status_code, 400)
                lease = app.store.resource(); fcntl.flock(lease, fcntl.LOCK_EX)
                try:
                    self.assertEqual(client.post('/public-experiments', headers=headers,
                        json=dict(dataset='fixture', preset='check')).status_code, 400)
                    self.assertEqual(download.call_count, 1)
                finally:
                    lease.close()
                launch.assert_not_called()

    def test_worker_race_keeps_a_resumable_project(self):
        from unittest.mock import patch
        from acan.lm import public_corpus as public
        from acan.lm.store import Store, owner_key
        raw, spec = self.fixture()
        with tempfile.TemporaryDirectory() as root, patch.dict(public.CATALOG, fixture=spec), \
                patch.object(public, 'download', return_value=raw), \
                patch.object(public, 'launch', side_effect=ValueError('Worker occupato')):
            store = Store(root)
            result = public.experiment(store, owner_key('a'*64), 'fixture', 'check')
            self.assertEqual(result['job']['status'], 'created')
            self.assertIn('Training non avviato', result['message'])
            self.assertEqual(result['corpus']['status'], 'ready')


if __name__=='__main__':
    unittest.main()


@unittest.skipUnless(LM and importlib.util.find_spec('flask'), 'Requires LM service dependencies')
class TestProjectManagementAndSeries(unittest.TestCase):
    """Rename, delete, verifications of a project, and series run one after the other."""

    def setUp(self):
        import hashlib
        from unittest.mock import patch
        from acan.lm import public_corpus as public
        from acan.lm.service import create_app
        raw = b'The river flows through the valley.\n'*100
        spec = dict(title='Fixture', language='en', url='https://example.invalid/text', source='https://example.invalid',
                    license='Test only', license_url='https://example.invalid/license', bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), format='text', note='Fixture only', prompt='The river')
        self.root = tempfile.TemporaryDirectory()
        self.headers = {'Authorization': 'Bearer '+'x'*40, 'X-ACAN-Workspace': 'a'*64}
        self.patches = [patch.dict(public.CATALOG, fixture=spec), patch.object(public, 'download', return_value=raw),
                        patch.object(public, 'launch')]
        for item in self.patches:
            item.start()
        self.app = create_app(self.root.name, 'x'*40)
        self.client = self.app.test_client()
        self.job = self.client.post('/public-experiments', headers=self.headers, json=dict(dataset='fixture', preset='check')).get_json()['job']

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.root.cleanup()

    def test_rename_and_delete(self):
        other = {**self.headers, 'X-ACAN-Workspace': 'b'*64}
        renamed = self.client.post(f"/jobs/{self.job['id']}/rename", headers=self.headers, json=dict(name='Dante, prima prova'))
        self.assertEqual(renamed.get_json()['config']['name'], 'Dante, prima prova')
        self.assertEqual(self.client.post(f"/jobs/{self.job['id']}/rename", headers=self.headers, json=dict(name='')).status_code, 400)
        self.assertEqual(self.client.post(f"/jobs/{self.job['id']}/delete", headers=other).status_code, 404)
        self.app.store.update(self.job['id'], status='running')
        self.assertEqual(self.client.post(f"/jobs/{self.job['id']}/delete", headers=self.headers).status_code, 400)   # not while running
        self.app.store.update(self.job['id'], status='paused')
        folder = self.app.store.directory('jobs', self.job['id'])
        self.assertEqual(self.client.post(f"/jobs/{self.job['id']}/delete", headers=self.headers).status_code, 200)
        self.assertFalse(folder.exists())
        self.assertEqual(self.client.get('/state', headers=self.headers).get_json()['jobs'], [])

    def test_verifications_of_a_project_are_listed(self):
        import json
        with self.app.store.db() as db:
            db.execute("INSERT INTO evaluations(id,job,owner,status,request,result) VALUES('e1',?,?,'completed','{}',?)",
                       (self.job['id'], self.job['owner'] if 'owner' in self.job else self.app.store.job(self.job['id'])['owner'],
                        json.dumps(dict(step=20, records=dict(selected_checkpoint=dict(quality=dict(nll=3.2)))))))
        listed = self.client.get(f"/jobs/{self.job['id']}/evaluations", headers=self.headers).get_json()
        self.assertEqual(listed[0]['records']['selected_checkpoint']['quality']['nll'], 3.2)

    def test_series_runs_one_project_after_the_other(self):
        from unittest.mock import patch
        from acan.lm.manager import continue_series
        base = dict(self.job['config'])
        with patch('acan.lm.service.launch') as launch:
            created = self.client.post('/series', headers=self.headers, json=dict(base=base, variants=['chosen', 'fewer_concepts', 'dense']))
        self.assertEqual(created.status_code, 200, created.get_data(as_text=True))
        jobs = created.get_json()['jobs']
        self.assertEqual([job['config']['series']['order'] for job in jobs], [0, 1, 2])
        self.assertEqual(jobs[1]['config']['slots'], base['slots']//2)
        dense = jobs[2]['config']
        self.assertEqual(dense['architecture'], 'dense-control')
        self.assertEqual(dense['control_of'], jobs[0]['id'])                      # drawn with the chosen settings
        self.assertTrue(dense['name'].startswith('Controllo denso · '))
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[1]['id'], jobs[0]['id'])
        store = self.app.store
        with patch('acan.lm.manager.launch') as following:
            store.update(jobs[0]['id'], status='paused')                        # paused by the user: the series waits
            self.assertIsNone(continue_series(store, jobs[0]['id']))
            store.update(jobs[0]['id'], status='completed')
            self.assertEqual(continue_series(store, jobs[0]['id']), jobs[1]['id'])
            store.update(jobs[1]['id'], status='paused', progress=dict(early_stop_reason='Validazione senza miglioramenti'))
            self.assertEqual(continue_series(store, jobs[1]['id']), jobs[2]['id'])
        self.assertEqual(following.call_count, 2)
        for wrong in (['chosen'], ['chosen', 'chosen'], ['chosen', 'unknown']):
            self.assertEqual(self.client.post('/series', headers=self.headers, json=dict(base=base, variants=wrong)).status_code, 400)

    def test_requests_accept_full_length_and_long_contexts(self):
        from unittest.mock import patch
        from acan.lm import manager
        job = self.app.store.job(self.job['id'])
        with patch.object(manager, 'snapshots', return_value=[dict(name='step-1')]), patch.object(manager, 'launch'):
            manager.request_generation(self.app.store, dict(job, status='paused'), job['owner'],
                                       dict(prompt='The river', checkpoint='step-1', full_length=True))
            with self.assertRaises(ValueError):
                manager.request_generation(self.app.store, dict(job, status='paused'), job['owner'],
                                           dict(prompt='The river', checkpoint='step-1', full_length='yes'))
            manager.request_evaluation(self.app.store, dict(job, status='paused'), job['owner'],
                                       dict(checkpoint='step-1', max_batches=1, context_lengths=[512, 2048, 4096]))
            with self.assertRaises(ValueError):
                manager.request_evaluation(self.app.store, dict(job, status='paused'), job['owner'],
                                           dict(checkpoint='step-1', context_lengths=[9000]))
