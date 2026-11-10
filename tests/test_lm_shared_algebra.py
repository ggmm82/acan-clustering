"""Explicit equation tests shared by both paper audits; no quality target fitting."""
import importlib.util
import unittest
import numpy as np
from acan.algebra import weighted_squared_distance,routing_logits,mass_modulation
from acan.core.paper_routing import PaperRoutingNumpy

TORCH=importlib.util.find_spec('torch') is not None


class SharedEquationTests(unittest.TestCase):
    def fixture(self,alpha=2.):
        rng=np.random.default_rng(7);x=rng.uniform(.15,.85,(8,4))
        net=PaperRoutingNumpy(8,4,k=3,relevance_alpha=alpha)
        net.set_root_statistics(x.mean(0),x.min(0),x.max(0))
        net.omega=np.array([[.7,.9,.5,1.]])
        return net,x

    def test_paper_squared_distance_and_log_mass_match_literal_equations(self):
        net,x=self.fixture();result=net(x);g=1/(1+np.exp(-net.matrices['G']))
        r=net._compute_relevance(g)[0];h=r**2
        distance=np.empty((3,8))
        for i in range(3):
            for t in range(8):
                distance[i,t]=sum(h[i,j]/(h[i].mean()+1e-6)*(x[t,j]-g[i,j])**2 for j in range(4))/4
        first=np.exp(-8*distance-(-8*distance).max(0));first/=first.sum(0)
        logits=-8*distance+1.15*np.log(first.mean(1,keepdims=True)+1e-4)
        expected=np.exp(logits-logits.max(0));expected/=expected.sum(0)
        np.testing.assert_allclose(result['T'],expected,atol=1e-12)
        np.testing.assert_allclose(result['G'],result['T']@x/result['T'].sum(1,keepdims=True),atol=1e-12)

    def test_paper_routing_numpy_gradient_matches_its_objective(self):
        for alpha in (1.,2.,3.):
            net,x=self.fixture(alpha);net(x)
            gradients={k:v.copy() for k,v in net.backward().items()}
            for key in gradients:
                for index in np.ndindex(net.matrices[key].shape):
                    original=net.matrices[key][index];epsilon=1e-6
                    net.matrices[key][index]=original+epsilon;plus=net(x)['neg_cp']
                    net.matrices[key][index]=original-epsilon;minus=net(x)['neg_cp']
                    net.matrices[key][index]=original
                    self.assertAlmostEqual(gradients[key][index],(plus-minus)/(2*epsilon),delta=2e-6)

    def test_paper_profile_survives_export_and_uses_the_same_inference_metric(self):
        import pandas as pd
        from acan.vis.training import train_native
        from acan import GraphClusterModel
        frame=pd.DataFrame({'x':[0.,.1,.9,1.]*10,'class':['a','a','b','b']*10})
        options=dict(max_depth=1,max_epochs=25,min_samples_split=5,similarity_threshold=.92,
            lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300,routing_profile='paper-routing-v1')
        graph=train_native(frame,['class'],options)
        loaded=GraphClusterModel.loadModel(graph.to_json())
        root=loaded.clusterModel[loaded.rootId]
        self.assertEqual(root.routing_profile,'paper-routing-v1')
        cix=[root.cols.index('x')]
        weights,_,_=root.predictTupleSoft([.2],cix)
        p=root.routing_parameters
        logits=routing_logits(np.array([[.2]]),root.Groups[:,cix],root.Relevance[:,cix],p['alpha'],p['beta'])
        logits=mass_modulation(logits,root.Freqs.reshape(-1),p['gamma'],np.log)
        expected=np.exp(logits-logits.max());expected/=expected.sum()
        np.testing.assert_allclose(weights,expected,atol=1e-12)
        before=root.predictTupleSoft([.2],cix)[0]
        after=GraphClusterModel.loadModel(loaded.to_json()).clusterModel[loaded.rootId].predictTupleSoft([.2],cix)[0]
        np.testing.assert_array_equal(before,after)

    @unittest.skipUnless(TORCH,'CPU Torch required in both dedicated audits')
    def test_shared_paper_numpy_torch_values_and_gradients_agree(self):
        import torch
        from acan.core.paper_routing import PaperRoutingTorch
        for alpha in (1.,2.,3.):
            net,x=self.fixture(alpha);other=PaperRoutingTorch(8,4,k=3,relevance_alpha=alpha).double()
            other.set_root_statistics(x.mean(0),x.min(0),x.max(0))
            with torch.no_grad():
                other.omega.copy_(torch.tensor(net.omega))
                for k,v in net.matrices.items():other.matrices[k].copy_(torch.tensor(v))
            a,b=net(x),other(torch.tensor(x))
            for k in ('T','G','R','F','cp'):
                np.testing.assert_allclose(a[k],b[k].detach().numpy(),atol=1e-10,rtol=1e-9)
            b['neg_cp'].backward()
            for k,v in net.backward().items():
                np.testing.assert_allclose(v,other.matrices[k].grad.numpy(),atol=1e-9,rtol=1e-8)

    @unittest.skipUnless(TORCH,'CPU Torch required in LM audit')
    def test_shared_lm_adapter_preserves_legacy_logits_gradients_and_causality(self):
        import torch
        from types import SimpleNamespace
        from acan.lm.attention import ACANConceptAttentionLayer
        from acan.lm.shared_attention import SharedAlgebraAttention
        torch.set_num_threads(1);torch.manual_seed(42)
        config=SimpleNamespace(hidden_size=8,num_attention_heads=2,head_dim=4)
        old=ACANConceptAttentionLayer(config,0,slots=3,window=4)
        new=SharedAlgebraAttention(config,0,slots=3,window=4);new.load_state_dict(old.state_dict())
        x=torch.randn(1,9,8)
        a=old.attend(x);b=new.attend(x)
        torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        a.square().sum().backward();b.square().sum().backward()
        for (ka,pa),(kb,pb) in zip(old.named_parameters(),new.named_parameters()):
            self.assertEqual(ka,kb);torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-5)
        altered=x.clone();altered[:,5:]=10
        torch.testing.assert_close(new.attend(x)[:,:5],new.attend(altered)[:,:5],atol=1e-7,rtol=1e-6)
        new.reset_stream();pieces=[]
        for i in range(x.shape[1]):pieces.append(new.attend(x[:,i:i+1],use_cache=True))
        torch.testing.assert_close(b,torch.cat(pieces,1),atol=2e-6,rtol=2e-5)
