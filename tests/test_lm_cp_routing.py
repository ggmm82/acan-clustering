import unittest
import importlib.util
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required')
class UnifiedRoutingTests(unittest.TestCase):
    def test_numpy_torch_router_parity_and_differentiability(self):
        import torch
        from acan.core.variational import solve
        torch.set_num_threads(1)
        rng=np.random.default_rng(42)
        x=rng.uniform(.1,.9,(5,4))
        state=dict(anchors=rng.uniform(.1,.9,(3,4)),omega=np.ones((1,4)),root=np.full((1,4),.5),den=np.full((1,4),.5))
        expected=solve(x,state,3)
        tensor=torch.tensor(x,requires_grad=True)
        actual=solve(tensor,state,3)
        np.testing.assert_allclose(expected['T'],actual['T'].detach(),atol=1e-10)
        actual['T'].square().sum().backward()
        plus,minus=x.copy(),x.copy();plus[0,0]+=1e-5;minus[0,0]-=1e-5
        finite_difference=((solve(plus,state,3)['T']**2).sum()-(solve(minus,state,3)['T']**2).sum())/2e-5
        self.assertAlmostEqual(float(tensor.grad[0,0]),finite_difference,delta=2e-5)
        self.assertTrue(all(b['energy']>=a['energy'] for a,b in zip(actual['history'],actual['history'][1:])))

    def test_prefix_streaming_and_finite_training_gradients(self):
        import torch
        from types import SimpleNamespace
        from acan.lm.cp_attention import CPUnifiedAttention
        torch.set_num_threads(1);torch.manual_seed(7)
        net=CPUnifiedAttention(SimpleNamespace(hidden_size=8,num_attention_heads=2),0,slots=3,window=4)
        x=torch.randn(1,5,8)
        y=net.attend(x)
        changed=x.clone();changed[:,3:]+=10
        torch.testing.assert_close(y[:,:3],net.attend(changed)[:,:3],atol=1e-6,rtol=1e-5)
        y.square().mean().backward()
        for parameter in net.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        net.reset_stream()
        pieces=[net.attend(x[:,i:i+1],use_cache=True) for i in range(5)]
        torch.testing.assert_close(y,torch.cat(pieces,1),atol=2e-6,rtol=2e-5)

    def test_grafted_model_trains_and_streams_with_both_graft_fractions(self):
        import tempfile
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from acan.lm.research_model import load_model
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64))
            base.save_pretrained(directory,safe_serialization=True)
            for fraction in (.5,1.):
                options=dict(architecture='hybrid',graft_fraction=fraction,sequence_length=8,slots=2,window=4,lora_rank=2,routing_profile='cp-routing-v1')
                model,report=load_model(directory,options)
                tokens=torch.tensor([[1,2,3,4]])
                result=model(input_ids=tokens,labels=tokens,use_cache=False)
                result.loss.backward()
                parameters=[p for p in model.parameters() if p.requires_grad]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in parameters))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None))
                torch.optim.Adam(parameters,lr=1e-4).step()
                model.eval()
                with torch.no_grad():
                    full=model(input_ids=tokens,use_cache=False).logits
                    first=model(input_ids=tokens[:,:2],use_cache=True)
                    rest=model(input_ids=tokens[:,2:],past_key_values=first.past_key_values,use_cache=True)
                    torch.testing.assert_close(full[:,2:],rest.logits,atol=2e-5,rtol=2e-4)
                self.assertEqual(report['algebra'],'cp-routing-v1')
