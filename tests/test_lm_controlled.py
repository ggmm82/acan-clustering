"""Offline safeguards for controlled training, retention, export and diagnosis."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

LM=all(importlib.util.find_spec(name) for name in ('torch','transformers','peft','safetensors'))


class Tokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False):
        return [2+ord(c)%30 for c in text]


@unittest.skipUnless(LM,'Requires LM dependencies')
class ControlledTrainingTests(unittest.TestCase):
    def fixture(self,root,patience=0,distillation=.5,steps=4,base_path=None,architecture='hybrid',fraction=.5,routing='shared-algebra-v1',cp_weight=0.,checkpointing=False):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from acan.lm.store import Store,owner_key
        from acan.lm.research_options import validate_options
        torch.set_num_threads(1)
        store=Store(root);owner=owner_key('a'*64)
        raw=('\n'.join(json.dumps(dict(text=(role+' document '+str(i)+' abcdef '*30),split=role))
             for role in ('train','validation','test') for i in range(4))+'\n').encode()
        corpus=store.create_corpus(owner,'corpus.jsonl',len(raw))
        store.append_corpus(corpus['id'],owner,0,raw);store.finish_corpus(corpus['id'],owner)
        options=validate_options(dict(corpus_id=corpus['id'],total_steps=steps,sequence_length=16,batch_size=1,
            accumulation=1,lora_rank=2,slots=2,window=4,save_steps=1,device='cpu',validation_batches=1,
            early_stop_patience=patience,distillation_weight=distillation,architecture=architecture,graft_fraction=fraction,routing_profile=routing,cp_weight=cp_weight,gradient_checkpointing=checkpointing))
        job=store.create_job(owner,options)
        if base_path is None:
            base_path=Path(root)/'fixture-base'
            torch.manual_seed(7)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128))
            model.save_pretrained(base_path,safe_serialization=True)
        base=dict(repo='fixture',revision='a'*40,files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(base_path).iterdir() if p.is_file()})
        (store.directory('jobs',job['id'])/'base.json').write_text(json.dumps(base))
        return store,job,base_path,base

    def run_job(self,store,job,path,base,stop=None,losses=None):
        from acan.lm import research_trainer as trainer
        with patch.object(trainer,'resolve_base',return_value=(path,base)), \
                patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
            if losses is not None:
                with patch.object(trainer,'validation_loss',side_effect=losses):
                    trainer.run_training(store.root,job['id'],stop)
            else:
                trainer.run_training(store.root,job['id'],stop)

    def test_teacher_objective_and_data_splits_are_measured(self):
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root)
            self.run_job(store,job,path,base)
            result=store.job(job['id'])
            self.assertEqual(result['status'],'completed',result['error'])
            self.assertEqual(result['progress']['step'],4)
            self.assertGreater(result['progress']['teacher_kl'],0)
            self.assertGreater(result['progress']['acan_gradient_norm'],0)
            self.assertGreater(result['progress']['lora_gradient_norm'],0)
            self.assertEqual(result['progress']['dataset']['documents'],{'train':4,'validation':4,'test':4})

    def test_early_stop_preserves_best_after_rolling_prune_and_can_resume(self):
        from acan.lm.manager import snapshots
        from acan.lm.checkpoints import verify
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,patience=2,distillation=0)
            self.run_job(store,job,path,base,losses=[4.,3.,3.1,3.2])
            result=store.job(job['id'])
            self.assertEqual(result['status'],'paused',result['error'])
            self.assertEqual(result['progress']['step'],2)
            self.assertIn('Validazione',result['progress']['early_stop_reason'])
            choices=snapshots(store,result);best=next(item for item in choices if item['best'])
            self.assertEqual(best['step'],0);self.assertEqual(best['validation_nll'],3.)
            checkpoint_root=store.directory('jobs',job['id'])/'checkpoints'
            self.assertLessEqual(len(list(checkpoint_root.glob('step-*'))),3)
            verify(checkpoint_root/best['name'])
            store.update(job['id'],pause=0)
            self.run_job(store,job,path,base,losses=[2.9,2.8])
            result=store.job(job['id'])
            self.assertEqual(result['status'],'completed',result['error'])
            self.assertEqual(next(item for item in snapshots(store,result) if item['best'])['step'],4)

    def test_distillation_resume_preserves_exact_parameters(self):
        from safetensors.torch import load_file
        from acan.lm import research_trainer as trainer
        from acan.lm.store import Store
        with tempfile.TemporaryDirectory() as one,tempfile.TemporaryDirectory() as two:
            s1,j1,path,base=self.fixture(one);s2,j2,_,_=self.fixture(two,base_path=path)
            self.run_job(s1,j1,path,base)
            stop=threading.Event();original=Store.record
            def record(store,id,step,values):
                original(store,id,step,values)
                if str(store.root)==str(s2.root) and step==2:stop.set()
            with patch.object(Store,'record',record):self.run_job(s2,j2,path,base,stop)
            self.assertEqual(s2.job(j2['id'])['status'],'paused')
            s2.update(j2['id'],pause=0);self.run_job(s2,j2,path,base)
            import torch
            a=load_file(s1.directory('jobs',j1['id'])/'checkpoints'/s1.job(j1['id'])['checkpoint']/'trainables.safetensors')
            b=load_file(s2.directory('jobs',j2['id'])/'checkpoints'/s2.job(j2['id'])['checkpoint']/'trainables.safetensors')
            self.assertEqual(set(a),set(b))
            for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_concept_profile_trains_and_resumes_exactly(self):
        from safetensors.torch import load_file
        from acan.lm import research_trainer as trainer
        from acan.lm.store import Store
        with tempfile.TemporaryDirectory() as one,tempfile.TemporaryDirectory() as two:
            s1,j1,path,base=self.fixture(one,routing='cp-v9',distillation=0);s2,j2,_,_=self.fixture(two,base_path=path,routing='cp-v9',distillation=0)
            self.run_job(s1,j1,path,base)
            done=s1.job(j1['id'])
            self.assertEqual(done['status'],'completed')
            self.assertEqual(done['progress']['graft']['algebra'],'cp-v9')
            stop=threading.Event();original=Store.record
            def record(store,id,step,values):
                original(store,id,step,values)
                if str(store.root)==str(s2.root) and step==2:stop.set()
            with patch.object(Store,'record',record):self.run_job(s2,j2,path,base,stop)
            self.assertEqual(s2.job(j2['id'])['status'],'paused')
            s2.update(j2['id'],pause=0);self.run_job(s2,j2,path,base)
            import torch
            a=load_file(s1.directory('jobs',j1['id'])/'checkpoints'/done['checkpoint']/'trainables.safetensors')
            b=load_file(s2.directory('jobs',j2['id'])/'checkpoints'/s2.job(j2['id'])['checkpoint']/'trainables.safetensors')
            self.assertEqual(set(a),set(b))
            for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_cp_routing_resume_preserves_exact_parameters(self):
        from safetensors.torch import load_file
        from acan.lm import research_trainer as trainer
        from acan.lm.store import Store
        with tempfile.TemporaryDirectory() as one,tempfile.TemporaryDirectory() as two:
            s1,j1,path,base=self.fixture(one,routing='cp-joint-v1',distillation=0,cp_weight=.1,checkpointing=True);s2,j2,_,_=self.fixture(two,base_path=path,routing='cp-joint-v1',distillation=0,cp_weight=.1,checkpointing=True)
            self.run_job(s1,j1,path,base)
            metrics=s1.job(j1['id'])['progress']
            self.assertEqual(metrics['cp_weight'],.1)
            self.assertGreater(metrics['training_cp_penalty'],0.)
            self.assertAlmostEqual(metrics['training_objective'],metrics['training_nll']+.1*metrics['training_cp_penalty'],places=6)
            stop=threading.Event();original=Store.record
            def record(store,id,step,values):
                original(store,id,step,values)
                if str(store.root)==str(s2.root) and step==2:stop.set()
            with patch.object(Store,'record',record):self.run_job(s2,j2,path,base,stop)
            self.assertEqual(s2.job(j2['id'])['status'],'paused')
            s2.update(j2['id'],pause=0);self.run_job(s2,j2,path,base)
            import torch
            a=load_file(s1.directory('jobs',j1['id'])/'checkpoints'/s1.job(j1['id'])['checkpoint']/'trainables.safetensors')
            b=load_file(s2.directory('jobs',j2['id'])/'checkpoints'/s2.job(j2['id'])['checkpoint']/'trainables.safetensors')
            self.assertEqual(set(a),set(b))
            for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_export_contains_inspectable_deltas_and_reloads_exact_weights(self):
        import zipfile
        import torch
        from safetensors.torch import load_file
        from acan.lm.exporting import export_bundle,load_export
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,distillation=0,steps=1)
            self.run_job(store,job,path,base);job=store.job(job['id'])
            with patch('huggingface_hub.snapshot_download',return_value=str(path)):
                archive=export_bundle(store,job,job['checkpoint'])
                target=Path(root)/'export';target.mkdir()
                with zipfile.ZipFile(archive) as z:
                    self.assertIn('architecture.json',z.namelist());self.assertIn('trainables.safetensors',z.namelist())
                    self.assertNotIn('optimizer.pt',z.namelist());self.assertNotIn('corpus',z.namelist())
                    z.extractall(target)
                with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                    model,_=load_export(target)
                values=load_file(target/'trainables.safetensors')
                for key,parameter in model.named_parameters():
                    if parameter.requires_grad:self.assertTrue(torch.equal(values[key],parameter.cpu()),key)
                original_manifest=(target/'export-manifest.json').read_text()
                manifest=json.loads(original_manifest);manifest['model_contract']='0'*64
                (target/'export-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'altra versione|altra|versione'):load_export(target)
                (target/'export-manifest.json').write_text(original_manifest)
                (target/'config.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'modificato'):load_export(target)

    def test_full_acan_is_causal_streaming_and_uses_no_dense_kv_cache(self):
        import torch
        from acan.lm.research_model import load_model
        from acan.lm.grafting import reset_streams
        from acan.lm.evaluation import cache_bytes
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0,fraction=1.)
            torch.manual_seed(42);model,graft=load_model(path,job['config']);model.eval()
            self.assertEqual(graft['fraction'],1.)
            ids=torch.tensor([[2+i%25 for i in range(20)]])
            other=ids.clone();other[:,8:]=7
            with torch.no_grad():
                full=model(ids,use_cache=False).logits
                changed=model(other,use_cache=False).logits
                torch.testing.assert_close(full[:,:8],changed[:,:8],atol=1e-6,rtol=1e-6)
                reset_streams(model);cache=None;pieces=[]
                for index in range(ids.shape[1]):
                    output=model(ids[:,index:index+1],past_key_values=cache,use_cache=True)
                    cache=output.past_key_values;pieces.append(output.logits)
                    self.assertEqual(cache.get_seq_length(),index+1)
                torch.testing.assert_close(full,torch.cat(pieces,1),atol=2e-5,rtol=2e-5)
                self.assertEqual(cache_bytes(model,cache)['dense_layers'],0)
                before=cache_bytes(model,cache)['total']
                for _ in range(10):cache=model(ids[:,:1],past_key_values=cache,use_cache=True).past_key_values
                self.assertEqual(cache_bytes(model,cache)['total'],before)
                reset_streams(model)
                generated=model.generate(ids[:,:4],max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0)
                self.assertGreater(generated.shape[1],4)

    def test_dense_lora_control_has_no_acan_layers_and_keeps_the_same_data_protocol(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0,architecture='dense-control')
            self.run_job(store,job,path,base);result=store.job(job['id'])
            self.assertEqual(result['status'],'completed',result['error'])
            self.assertEqual(result['progress']['graft']['fraction'],0.)
            self.assertEqual(result['progress']['acan_gradient_norm'],0.)
            tensors=load_file(store.directory('jobs',job['id'])/'checkpoints'/result['checkpoint']/'trainables.safetensors')
            self.assertTrue(all('lora_' in key for key in tensors))

    def test_independent_test_comparison_measures_both_quality_and_whole_cache(self):
        from acan.lm import evaluation
        from acan.lm.manager import request_evaluation
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0)
            self.run_job(store,job,path,base);job=store.job(job['id'])
            with patch('acan.lm.manager.launch'):
                identifier=request_evaluation(store,job,job['owner'],dict(checkpoint=job['checkpoint'],
                    max_batches=1,new_tokens=2,context_lengths=[16,32]))
            with patch.object(evaluation,'resolve_base',return_value=(path,base)):
                evaluation.evaluate_checkpoint(root,identifier)
            with store.db() as db:
                row=dict(db.execute('SELECT * FROM evaluations WHERE id=?',(identifier,)).fetchone())
            self.assertEqual(row['status'],'completed',row['error'])
            result=json.loads(row['result'])
            self.assertEqual(result['records']['original']['quality']['tokens'],16)
            self.assertEqual(result['records']['selected_checkpoint']['quality']['tokens'],16)
            self.assertGreater(result['records']['selected_checkpoint']['performance'][0]['cache']['acan_layers'],0)
            self.assertEqual(result['records']['original']['performance'][0]['cache']['acan_layers'],0)
            self.assertEqual(len(result['comparison']),2)
            self.assertIn('never used',result['protocol']['test'])

    def test_web_diagnostics_and_export_are_owner_scoped_and_redact_secrets(self):
        from acan.lm.service import create_app
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0)
            self.run_job(store,job,path,base);job=store.job(job['id'])
            secret='test-service-secret-'+'x'*32
            (store.directory('jobs',job['id'])/'worker.log').write_text('Authorization: Bearer '+secret+' hf_'+'a'*30)
            app=create_app(root,secret);client=app.test_client()
            headers={'Authorization':'Bearer '+secret,'X-ACAN-Workspace':'a'*64}
            other=dict(headers,**{'X-ACAN-Workspace':'b'*64})
            endpoint='/jobs/'+job['id']+'/diagnostics'
            self.assertEqual(client.get(endpoint,headers=other).status_code,404)
            response=client.get(endpoint,headers=headers)
            self.assertEqual(response.status_code,200)
            self.assertNotIn(secret,response.get_data(as_text=True))
            self.assertNotIn('hf_'+'a'*30,response.get_data(as_text=True))
            endpoint='/jobs/'+job['id']+'/export?checkpoint='+job['checkpoint']
            self.assertEqual(client.get(endpoint,headers=other).status_code,404)
            with patch('huggingface_hub.snapshot_download',return_value=str(path)):
                response=client.get(endpoint,headers=headers)
                self.assertEqual(response.status_code,200);response.close()

    def test_wikitext_import_keeps_official_article_splits(self):
        import io
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        from acan.lm import public_corpus as public
        files=[];raw={}
        for role in ('train','validation','test'):
            stream=io.BytesIO();parquet.write_table(arrow.table({'text':[' = '+role+' article = ','body '+role+' words',' = = Section = = ','more words']}),stream)
            raw[role]=stream.getvalue()
            files.append(dict(split=role,url='https://example.invalid',bytes=len(raw[role]),sha256=hashlib.sha256(raw[role]).hexdigest()))
        with patch.object(public,'download',side_effect=lambda spec:raw[spec['split']]):
            content=public.prepare_wikitext(dict(files=files))
        records=[json.loads(line) for line in content.decode().splitlines()]
        self.assertEqual([item['split'] for item in records],['train','validation','test'])
        self.assertTrue(all('Section' in item['text'] for item in records))

    def test_web_cancel_does_not_load_a_model_or_modify_training(self):
        from acan.lm.service import create_app
        from acan.lm import evaluation
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0)
            identifier='b'*32
            with store.db() as db:db.execute('INSERT INTO evaluations(id,job,owner,status,request) VALUES(?,?,?,?,?)',
                (identifier,job['id'],job['owner'],'running',json.dumps({'checkpoint':'unused'})))
            app=create_app(root,'c'*40);client=app.test_client()
            # No live lease: creation reconciles a stale running request. Restore
            # the fixture to emulate an actual in-progress evaluation.
            with store.db() as db:db.execute("UPDATE evaluations SET status='running' WHERE id=?",(identifier,))
            headers={'Authorization':'Bearer '+'c'*40,'X-ACAN-Workspace':'a'*64}
            other=dict(headers,**{'X-ACAN-Workspace':'b'*64})
            endpoint='/evaluations/'+identifier+'/cancel'
            self.assertEqual(client.post(endpoint,headers=other).status_code,404)
            self.assertEqual(client.post(endpoint,headers=headers).status_code,200)
            with patch.object(evaluation,'resolve_base') as resolve:
                evaluation.evaluate_checkpoint(root,identifier)
                resolve.assert_not_called()
            with store.db() as db:status=db.execute('SELECT status FROM evaluations WHERE id=?',(identifier,)).fetchone()[0]
            self.assertEqual(status,'cancelled');self.assertEqual(store.job(job['id'])['status'],'created')

    def test_legacy_creation_cannot_train_official_test_split(self):
        from acan.lm.service import create_app
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0)
            corpus_id=job['config']['corpus_id']
            (store.directory('corpora',corpus_id)/'public-source.json').write_text(json.dumps(dict(preprocessing='wikitext-articles-official-splits-v1')))
            client=create_app(root,'c'*40).test_client();headers={'Authorization':'Bearer '+'c'*40,'X-ACAN-Workspace':'a'*64}
            result=client.post('/jobs',headers=headers,json=dict(corpus_id=corpus_id))
            self.assertEqual(result.status_code,400)
            self.assertIn('split ufficiali',result.get_json()['error'])

    def test_known_legacy_controlled_resume_preserves_exact_weights(self):
        import torch
        from safetensors.torch import load_file
        from acan.lm import research_trainer as trainer
        from acan.lm.compatibility import LEGACY_TRAINING,LEGACY_INFERENCE
        from acan.lm.research_model import check_inference_contract
        from acan.lm.store import Store
        with tempfile.TemporaryDirectory() as one,tempfile.TemporaryDirectory() as two:
            a,ja,path,base=self.fixture(one,steps=4)
            b,jb,_,_=self.fixture(two,steps=4,base_path=path)
            for store,job in ((a,ja),(b,jb)):
                config=dict(job['config']);config.pop('routing_profile')
                with store.db() as db:db.execute('UPDATE jobs SET config=? WHERE id=?',(json.dumps(config),job['id']))
                job['config']=config
            with patch.object(trainer,'contract',return_value=LEGACY_TRAINING[0]),patch.object(trainer,'inference_contract',return_value=LEGACY_INFERENCE):
                self.run_job(a,ja,path,base)
                stop=threading.Event();original=Store.record
                def record(store,id,step,values):
                    original(store,id,step,values)
                    if str(store.root)==str(b.root) and step==2:stop.set()
                with patch.object(Store,'record',record):self.run_job(b,jb,path,base,stop)
            b.update(jb['id'],pause=0);self.run_job(b,jb,path,base)
            self.assertEqual(b.job(jb['id'])['status'],'completed')
            outputs=[]
            for store,job in ((a,ja),(b,jb)):
                current=store.job(job['id']);check_inference_contract(current['config'],current['progress'])
                outputs.append(load_file(store.directory('jobs',job['id'])/'checkpoints'/current['checkpoint']/'trainables.safetensors'))
            for key in outputs[0]:self.assertTrue(torch.equal(outputs[0][key],outputs[1][key]),key)
            with self.assertRaisesRegex(ValueError,'versione'):
                check_inference_contract(dict(jb['config'],routing_profile='shared-algebra-v1'),dict(model_contract=LEGACY_INFERENCE))

    def test_layer_inspection_preserves_exact_training_and_is_owner_scoped(self):
        import torch
        from safetensors.torch import load_file
        from acan.lm.inspection import latest
        from acan.lm.service import create_app
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            a,ja,path,base=self.fixture(one,steps=2,distillation=0)
            b,jb,_,_=self.fixture(two,steps=2,distillation=0,base_path=path)
            app=create_app(two,'c'*40);client=app.test_client()
            headers={'Authorization':'Bearer '+'c'*40,'X-ACAN-Workspace':'a'*64}
            other=dict(headers,**{'X-ACAN-Workspace':'b'*64})
            endpoint='/jobs/'+jb['id']+'/layers'
            b.update(jb['id'],status='running')
            self.assertEqual(client.post(endpoint,headers=other,json={}).status_code,404)
            self.assertEqual(client.get(endpoint,headers=other).status_code,404)
            self.assertEqual(client.post(endpoint,headers=headers,json={}).status_code,200)
            self.run_job(a,ja,path,base);self.run_job(b,jb,path,base)
            report=client.get(endpoint,headers=headers).get_json()
            self.assertFalse(report['pending']);report=report['report']
            self.assertEqual(report['step'],1)
            self.assertEqual(len(report['frames']),1)
            self.assertEqual(len(report['frames'][0]['layers']),2)
            self.assertEqual(set(report['gradients']),{'0','1'})
            self.assertGreater(report['gradients']['1']['norm'],0)
            self.assertEqual(report['frames'][0]['layers'][1]['kind'],'ACAN')
            self.assertNotIn('document',json.dumps(report))
            self.assertIsNone(latest(a,ja)['report'])
            for store,job in ((a,ja),(b,jb)):
                current=store.job(job['id'])
                values=load_file(store.directory('jobs',job['id'])/'checkpoints'/current['checkpoint']/'trainables.safetensors')
                if store is a:first=values
                else:
                    for key in first:self.assertTrue(torch.equal(first[key],values[key]),key)

    def test_layer_inspection_tracks_prefill_decoding_and_does_not_change_logits(self):
        import torch
        from acan.lm.inspection import Capture
        from acan.lm.research_model import load_model
        from acan.lm.grafting import reset_streams
        with tempfile.TemporaryDirectory() as root:
            store,job,path,base=self.fixture(root,steps=1,distillation=0)
            torch.manual_seed(42);model,_=load_model(path,job['config']);model.eval()
            ids=torch.tensor([[2,3,4,5]])
            with torch.inference_mode():
                expected=model.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0,eos_token_id=None)
                reset_streams(model)
                with Capture(model,'inference',max_frames=3) as capture:
                    actual=model.generate(ids,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0,eos_token_id=None)
                self.assertTrue(torch.equal(expected,actual))
            report=capture.report(0)
            self.assertEqual(len(report['frames']),3)
            self.assertEqual(report['gradients'],{})
            for frame in report['frames']:
                self.assertEqual(len(frame['layers']),2)
                self.assertTrue(frame['logits_finite'])
                self.assertEqual(len(frame['next_token_candidates']),5)
                self.assertAlmostEqual(sum(frame['layers'][1]['mass_fractions']),1.,places=5)
            self.assertFalse(capture.handles)
            self.assertFalse(model.get_base_model().model._forward_pre_hooks)
            self.assertFalse(model.get_base_model().lm_head._forward_hooks)
            # An exception must also remove hooks and must not retain tensors.
            with self.assertRaisesRegex(RuntimeError,'test error'):
                with Capture(model,'inference'):raise RuntimeError('test error')
            self.assertFalse(model.get_base_model().model._forward_pre_hooks)
            self.assertFalse(model.get_base_model().lm_head._forward_hooks)
            json.dumps(report,allow_nan=False)
            core=model.get_base_model()
            def corrupt(module,inputs,output):
                return torch.full_like(output,float('nan'))
            hook=core.model.layers[0].register_forward_hook(corrupt)
            try:
                with torch.inference_mode(),Capture(model,'inference') as broken:
                    model(ids,use_cache=False)
                failure=broken.report()
                self.assertFalse(failure['frames'][0]['logits_finite'])
                self.assertFalse(failure['frames'][0]['layers'][0]['output']['finite'])
                json.dumps(failure,allow_nan=False)
            finally:
                hook.remove()

    def test_document_deduplication_and_pause_recovery_do_not_cross_splits(self):
        from acan.lm.research_data import prepare
        from acan.lm.data import PreparationPaused
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);source=root/'corpus.jsonl'
            source.write_text('\n'.join(json.dumps(dict(text=text,split=role)) for text,role in [
                ('duplicate words','train'),('duplicate  words','validation'),('validation unique','validation'),('test unique','test')]))
            count=[0]
            def pause():count[0]+=1;return count[0]>2
            with self.assertRaises(PreparationPaused):prepare(source,'corpus.jsonl',Tokenizer(),root,{},pause)
            result=prepare(source,'corpus.jsonl',Tokenizer(),root,{})
            self.assertEqual(result['documents'],{'train':1,'validation':1,'test':1})
            self.assertEqual(result['duplicates'],1)


class CPUResourceProfileTests(unittest.TestCase):
    def test_cpu_profile_rejects_oversized_jobs_before_weight_loading(self):
        from acan.lm.resources import check_profile
        options=dict(base_model='HuggingFaceTB/SmolLM2-135M',device='auto',batch_size=1,sequence_length=128,
                     slots=16,window=16,lora_rank=8,graft_fraction=.5)
        check_profile(options,False)
        for changes in ({'batch_size':8},{'sequence_length':2048},{'graft_fraction':1.},{'base_model':'HuggingFaceTB/SmolLM2-360M'}):
            with self.assertRaises(ValueError):check_profile(dict(options,**changes),False)
        check_profile(dict(options,graft_fraction=1.,sequence_length=64),False)
        check_profile(dict(options,batch_size=8,sequence_length=2048),True)
        with self.assertRaises(ValueError):check_profile(dict(options,device='cpu',batch_size=8),True)
