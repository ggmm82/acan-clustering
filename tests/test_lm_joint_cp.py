"""The joint adapters must share values, derivatives and causal CP mechanics."""
import importlib.util
import unittest
import copy
import numpy as np

LM=all(importlib.util.find_spec(name) for name in ('torch','transformers','peft'))


@unittest.skipUnless(LM,'LM dependencies required')
class JointCPTests(unittest.TestCase):
    def test_coalescing_equal_slots_preserves_prefix_cp_readout_and_gradients(self):
        import torch
        from acan.algebra import concept_step
        torch.set_num_threads(1)
        x=torch.tensor([[.1,.8],[.7,.3],[.4,.6]],dtype=torch.float64)
        anchors=torch.tensor([[.2,.7],[.8,.1]],dtype=torch.float64,requires_grad=True)
        copies=torch.tensor([0,0,1,1,1])
        def run(a,log_multiplicity=None):
            count=sums=None; readout=[]; cps=[]
            for t in range(len(x)):
                prefix=x[:t+1]; root=prefix.mean(0,keepdim=True)
                lo,hi=prefix.amin(0,keepdim=True),prefix.amax(0,keepdim=True)
                den=(hi-lo)/2+(root-(hi+lo)/2).abs()
                result=concept_step(x[t:t+1],a,root,den.clamp_min(1e-6),
                    torch.ones_like(root),8.,slice(None),lambda v,f:v.clamp_min(f),
                    torch.clamp,lambda v:v.softmax(0),base_counts=count,base_sums=sums,
                    log_multiplicity=log_multiplicity)
                count,sums=result['C'],result['sums']
                readout.append(result['T'].T@a); cps.append(result['normalized_cp'])
            return torch.cat(readout),torch.stack(cps)
        expanded,cp_expanded=run(anchors[copies])
        compact,cp_compact=run(anchors,torch.tensor([[2.],[3.]],dtype=x.dtype).log())
        torch.testing.assert_close(expanded,compact,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(cp_expanded,cp_compact,atol=1e-12,rtol=1e-12)
        grad_expanded=torch.autograd.grad(expanded.sum()+cp_expanded.sum(),anchors)[0]
        grad_compact=torch.autograd.grad(compact.sum()+cp_compact.sum(),anchors)[0]
        torch.testing.assert_close(grad_expanded,grad_compact,atol=1e-12,rtol=1e-12)

    def test_joint_core_numpy_torch_values_and_gradients_agree(self):
        import torch
        from acan.algebra import concept_step,prediction_cp_objective
        from acan.core.supervised import objective,probabilities
        torch.set_num_threads(1)
        rng=np.random.default_rng(4)
        x=np.column_stack([rng.uniform(.1,.9,(7,3)),np.eye(2)[[0,1,0,1,1,0,1]]])
        params=dict(G=rng.normal(size=(3,5)),log_beta=np.asarray(3.))
        state=dict(features=np.array([1,1,1,0,0],bool),root=x.mean(0,keepdims=True),den=np.ones((1,5))*.5,
                   omega=np.ones((1,5)),lo=np.zeros((1,5)),span=np.ones((1,5)),
                   log_multiplicity=np.log(np.array([[1.],[2.],[3.]])))
        expected=objective(params,x,[[3,4]],state)
        g=torch.tensor(params['G'],requires_grad=True)
        beta=torch.tensor(params['log_beta'],requires_grad=True)
        anchors=torch.cat((g[:,:3].sigmoid(),g[:,3:].softmax(1)),1)
        actual=concept_step(torch.tensor(x),anchors,torch.tensor(state['root']),torch.tensor(state['den']),
            torch.tensor(state['omega']),beta.exp(),state['features'],lambda v,f:v.clamp_min(f),torch.clamp,lambda v:v.softmax(0),
            log_multiplicity=torch.tensor(state['log_multiplicity']))
        for key in ('T','R','F','normalized_cp'):
            np.testing.assert_allclose(expected[key],actual[key].detach(),atol=1e-12,rtol=1e-12)
        prediction=actual['T'].T@anchors[:,3:]
        nll=-(torch.tensor(x[:,3:])*prediction.log()).sum()/len(x)
        loss=prediction_cp_objective(nll,actual['normalized_cp'],.1)
        loss.backward()
        np.testing.assert_allclose(expected['gradients']['G'],g.grad,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(expected['gradients']['log_beta'],beta.grad,atol=1e-10,rtol=1e-10)
        self.assertAlmostEqual(expected['loss'],float(loss.detach()),places=12)

    def test_joint_lm_cp_is_causal_differentiable_and_streaming_equivalent(self):
        import torch
        from types import SimpleNamespace
        from acan.lm.cp_attention import CPJointAttention
        torch.set_num_threads(1);torch.manual_seed(7)
        layer=CPJointAttention(SimpleNamespace(hidden_size=8,num_attention_heads=2),0,slots=3,window=4)
        x=torch.randn(2,6,8,requires_grad=True)
        output=layer.attend(x);cp=layer.last_cp
        self.assertTrue(torch.isfinite(cp).all())
        self.assertTrue(((cp>=0)&(cp<=1)).all())
        # CP alone must teach the prototypes and never depend on future tokens.
        gradient=torch.autograd.grad(cp[:,:4].mean(),(layer.G,x),retain_graph=True)
        self.assertGreater(float(gradient[0].abs().sum()),0.)
        torch.testing.assert_close(gradient[1][:,4:],torch.zeros_like(x[:,4:]))
        changed=x.detach().clone();changed[:,4:]+=10
        changed_output=layer.attend(changed)
        torch.testing.assert_close(cp[:,:4],layer.last_cp[:,:4],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(output[:,:4],changed_output[:,:4],atol=1e-6,rtol=1e-5)
        # Compare a CP derivative against the numerical derivative.
        analytic=torch.autograd.grad(cp.mean(),layer.G)[0][0,0]
        original=layer.G[0,0].item()
        with torch.no_grad():layer.G[0,0]=original+1e-3
        layer.attend(x.detach());plus=layer.last_cp.mean().item()
        with torch.no_grad():layer.G[0,0]=original-1e-3
        layer.attend(x.detach());minus=layer.last_cp.mean().item()
        with torch.no_grad():layer.G[0,0]=original
        self.assertAlmostEqual(float(analytic),(plus-minus)/2e-3,delta=5e-5)
        layer.reset_stream();pieces=[];cps=[]
        for index in range(x.shape[1]):
            pieces.append(layer.attend(x[:,index:index+1],use_cache=True));cps.append(layer.last_cp)
        torch.testing.assert_close(output,torch.cat(pieces,1),atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(cp,torch.cat(cps,1),atol=2e-6,rtol=2e-5)

    def test_joint_cp_mean_is_independent_of_layer_and_batch_replication(self):
        import torch
        from types import SimpleNamespace
        from acan.lm.cp_attention import CPJointAttention
        from acan.lm.cp_objective import regularize
        layer=CPJointAttention(SimpleNamespace(hidden_size=4,num_attention_heads=1),0,slots=2,window=2)
        layer.last_cp=torch.tensor([[.2,.4,.8]])
        loss=torch.tensor(2.)
        a,penalty=regularize(torch.nn.ModuleList([layer]),loss,.1)
        self.assertAlmostEqual(float(penalty),.7,places=6)
        layer.last_cp=layer.last_cp.repeat(3,1)
        other=copy.deepcopy(layer)
        b,_=regularize(torch.nn.ModuleList([layer,other]),loss,.1)
        torch.testing.assert_close(a,b)
        layer.last_cp=torch.cat((layer.last_cp[:,:-1].repeat(1,4),layer.last_cp[:,-1:]),1)
        replicated,_=regularize(torch.nn.ModuleList([layer]),loss,.1)
        torch.testing.assert_close(a,replicated)
        zero,_=regularize(torch.nn.ModuleList([layer]),loss,0.)
        torch.testing.assert_close(zero,loss,atol=0,rtol=0)

    def test_joint_grafted_model_uses_same_core_at_both_fractions(self):
        import tempfile
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from acan.lm.research_model import load_model
        from acan.lm.cp_attention import CPJointAttention
        from acan.lm.cp_objective import regularize
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64))
            base.save_pretrained(directory,safe_serialization=True)
            for fraction in (.5,1.):
                options=dict(architecture='hybrid',graft_fraction=fraction,sequence_length=8,slots=2,window=4,
                    lora_rank=2,routing_profile='cp-joint-v1')
                model,report=load_model(directory,options)
                self.assertTrue(any(isinstance(m,CPJointAttention) for m in model.modules()))
                tokens=torch.tensor([[1,2,3,4]])
                result=model(tokens,labels=tokens,use_cache=False)
                loss,penalty=regularize(model,result.loss,.1);loss.backward()
                self.assertGreater(float(penalty.detach()),0.)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                model.eval()
                with torch.no_grad():
                    full=model(tokens,use_cache=False).logits
                    first=model(tokens[:,:2],use_cache=True)
                    rest=model(tokens[:,2:],past_key_values=first.past_key_values,use_cache=True)
                    torch.testing.assert_close(full[:,2:],rest.logits,atol=2e-5,rtol=2e-4)
                self.assertEqual(report['algebra'],'cp-joint-v1')
