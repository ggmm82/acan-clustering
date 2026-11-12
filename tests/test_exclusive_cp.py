"""Exclusive CP: support selection, duplicate evidence, and backend derivatives."""
import importlib.util
import unittest
import numpy as np
from acan.algebra import concept_step, simplex_projection, simplex_coordinate_candidate
from acan.core.supervised import objective


class ExclusiveCPTests(unittest.TestCase):
    def test_adam_handles_finite_gradient_whose_square_overflows(self):
        from acan.core.learner import _NumpyAdam
        value=np.array([0.])
        optimizer=_NumpyAdam({'mass':value},lr=.08)
        with np.errstate(over='raise',invalid='raise'):
            optimizer.step({'mass':np.array([1e200])})
            self.assertAlmostEqual(float(value[0]),-.08)
            optimizer.step({'mass':np.array([1e200])})
            self.assertAlmostEqual(float(value[0]),-.16)
        self.assertTrue(np.isfinite(value).all())

    def fixture(self):
        rng=np.random.default_rng(9)
        x=np.column_stack([rng.uniform(.1,.9,(9,3)),np.eye(2)[[0,1,1,0,1,0,0,1,0]]])
        params=dict(G=rng.normal(size=(3,5)),log_beta=np.asarray(1.2),masses=np.array([.2,.5,.3]))
        state=dict(features=np.array([1,1,1,0,0],bool),root=x.mean(0,keepdims=True),
            den=np.ones((1,5))*.6,omega=np.array([[.8,1.,.7,.9,.9]]),
            lo=np.zeros((1,5)),span=np.ones((1,5)))
        return x,params,state

    def test_exclusive_cp_gradients_include_overlap_and_masses(self):
        x,params,state=self.fixture()
        result=objective(params,x,[[3,4]],state)
        for key in params:
            for index in np.ndindex(params[key].shape):
                original=params[key][index]
                params[key][index]=original+1e-6
                plus=objective(params,x,[[3,4]],state,gradients=False)['loss']
                params[key][index]=original-1e-6
                minus=objective(params,x,[[3,4]],state,gradients=False)['loss']
                params[key][index]=original
                self.assertAlmostEqual(result['gradients'][key][index],(plus-minus)/2e-6,delta=3e-7)

    def test_duplicate_evidence_is_discounted_without_a_group_count(self):
        x=np.array([[.1,.8],[.2,.7],[.3,.6]])
        def cp(k):
            a=np.repeat([[.2,.7]],k,axis=0)
            return concept_step(x,a,np.array([[.5,.5]]),np.ones((1,2)),np.ones((1,2)),8.,
                slice(None),np.maximum,np.clip,lambda v:np.exp(v-v.max(0))/np.exp(v-v.max(0)).sum(0),
                masses=np.ones(k)/k,exponential=np.exp,distinct=1-np.eye(k))['normalized_cp']
        self.assertGreater(cp(1),cp(2))
        self.assertAlmostEqual(cp(1)/np.sqrt(2),cp(2),places=12)
        np.testing.assert_array_equal(simplex_projection(np.array([2.,-.1,.4])),[1.,0.,0.])
        proposal = simplex_coordinate_candidate(np.array([.2,.5,.3]),0)
        np.testing.assert_array_equal(proposal,[0.,.7,.3])

    def test_pair_support_excludes_self_evidence_and_accumulates_causally(self):
        def step(x,**previous):
            return concept_step(np.asarray(x),np.array([[.8,.2]]),np.array([[.5,.5]]),
                np.ones((1,2))*.5,np.ones((1,2)),8.,slice(None),np.maximum,np.clip,
                lambda scores:np.ones_like(scores),masses=np.ones(1),exponential=np.exp,
                distinct=np.zeros((1,1)),**previous)
        one=step([[.8,.2]])
        self.assertEqual(float(one['evidence'][0,0]),0.)
        two=step([[.8,.2]],base_counts=one['C'],base_sums=one['sums'],
                 base_square_counts=one['square_counts'])
        full=step([[.8,.2],[.8,.2]])
        self.assertAlmostEqual(float(two['evidence'][0,0]),.5)
        for key in ('C','sums','square_counts','evidence','normalized_cp'):
            np.testing.assert_allclose(two[key],full[key],atol=1e-12)
        self.assertGreater(two['normalized_cp'],one['normalized_cp'])

    def test_unsupervised_relations_do_not_create_a_singleton_for_a_missing_cell(self):
        import pandas as pd
        from acan.vis.training import train_native,graph_stats
        rng=np.random.default_rng(12)
        a,b=np.tile(np.array([[0,0],[0,1],[1,0],[1,1]]),(25,1)).T
        frame=pd.DataFrame(dict(weather=np.where(a,'sun','rain'),place=np.where(b,'outside','inside'),
            day=rng.integers(1,29,len(a)),month=rng.integers(1,13,len(a)),
            year=rng.integers(1980,2010,len(a)),outcome=np.where(a==b,'yes','no')))
        incomplete=frame.iloc[[0]].copy();incomplete['outcome']=np.nan
        frame=pd.concat([frame,incomplete],ignore_index=True)
        options=dict(lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300,max_depth=1,
            max_epochs=100,min_samples_split=12,similarity_threshold=.92,routing_profile='cp-joint-v1')
        graph=train_native(frame,[],options)
        self.assertEqual(graph.labels,[])
        self.assertEqual(graph_stats(graph),dict(nodes=1,depth=1,groups=4))
        self.assertEqual(graph.clusterModel[graph.rootId].routing_parameters['training_nll'],0.)
        self.assertEqual(graph.clusterModel[graph.rootId].routing_parameters['training_rows'],101)

    def test_interaction_concepts_are_learned_with_irrelevant_dates(self):
        import pandas as pd
        from acan.vis.training import train_native,graph_stats
        from acan.vis.inference import infer
        from acan import GraphClusterModel
        rng=np.random.default_rng(12)
        def examples(repeats):
            a,b=np.tile(np.array([[0,0],[0,1],[1,0],[1,1]]),(repeats,1)).T
            return pd.DataFrame(dict(Tempo=np.where(a,'Sole','Piove'),Luogo=np.where(b,'Spiazzale','Palazzetto'),
                Giorno=rng.integers(1,29,len(a)),Mese=rng.integers(1,13,len(a)),
                Anno=rng.integers(1980,2010,len(a)),Gioco=np.where(a==b,'SI','NO')))
        train,held=examples(25),examples(40)
        options=dict(lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300,max_depth=3,
            max_epochs=100,min_samples_split=12,similarity_threshold=.92,routing_profile='cp-joint-v1')
        graph=train_native(train,['Gioco'],options)
        self.assertEqual(graph_stats(graph),dict(nodes=1,depth=1,groups=4))
        restored=GraphClusterModel.loadModel(graph.to_json())
        root=restored.clusterModel[restored.rootId]
        self.assertTrue(np.all(np.asarray(root.routing_parameters['masses'])>0))
        predicted=infer(restored.to_json(),held)
        np.testing.assert_array_equal(predicted.Gioco,held.Gioco)
        self.assertTrue((predicted.groupby(['Tempo','Luogo']).ACAN_path.nunique()==1).all())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required for structural descent')
    def test_inference_skips_pairwise_training_cost_without_changing_output(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from acan.lm.cp_attention import CPJointAttention
        torch.manual_seed(4)
        layer=CPJointAttention(SimpleNamespace(hidden_size=8,num_attention_heads=2),0,slots=3,window=3)
        x=torch.randn(2,5,8)
        training=layer.attend(x)
        self.assertIsNotNone(layer.last_cp)
        layer.eval()
        with patch('acan.algebra.exclusive_cluster_probability',side_effect=AssertionError('training-only cost')):
            inference=layer.attend(x)
        torch.testing.assert_close(training,inference,atol=0,rtol=0)
        self.assertIsNone(layer.last_cp)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required for structural descent')
    def test_generative_coordinate_uses_objective_and_preserves_rng(self):
        import torch
        from types import SimpleNamespace
        from acan.lm.cp_attention import CPJointAttention
        from acan.lm.cp_objective import coordinate_concept_step,project_concept_masses
        layer=CPJointAttention(SimpleNamespace(hidden_size=4,num_attention_heads=1),0,slots=3,window=2)
        with torch.no_grad():layer.masses.copy_(torch.tensor([.2,.5,.3]))
        state=torch.get_rng_state().clone()
        def objective_with_dropout():
            # Same random component must be used in both evaluations.
            return layer.masses[0]+torch.rand(())
        self.assertTrue(coordinate_concept_step(layer,0,objective_with_dropout))
        torch.testing.assert_close(layer.masses,torch.tensor([0.,.7,.3]))
        torch.testing.assert_close(state,torch.get_rng_state(),atol=0,rtol=0)
        original=layer.masses.detach().clone()
        self.assertFalse(coordinate_concept_step(layer,0,lambda:-layer.masses[1]))
        torch.testing.assert_close(layer.masses,original)
        with torch.no_grad():layer.masses.copy_(torch.tensor([2.,-.1,.4]))
        project_concept_masses(layer)
        torch.testing.assert_close(layer.masses,torch.tensor([1.,0.,0.]))

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Torch required for parity')
    def test_exclusive_core_torch_matches_numpy_at_interior_and_boundary(self):
        import torch
        torch.set_num_threads(1)
        for boundary in (False,True):
            x,params,state=self.fixture()
            if boundary:params['masses']=np.array([.4,0.,.6])
            expected=objective(params,x,[[3,4]],state)
            g=torch.tensor(params['G'],requires_grad=True)
            beta=torch.tensor(params['log_beta'],requires_grad=True)
            masses=torch.tensor(params['masses'],requires_grad=True)
            anchors=torch.cat((g[:,:3].sigmoid(),g[:,3:].softmax(1)),1)
            actual=concept_step(torch.tensor(x),anchors,torch.tensor(state['root']),torch.tensor(state['den']),
                torch.tensor(state['omega']),beta.exp(),state['features'],lambda v,f:v.clamp_min(f),
                torch.clamp,lambda v:v.softmax(0),masses=masses,exponential=torch.exp,
                distinct=1-torch.eye(3,dtype=torch.float64))
            for key in ('T','R','F','normalized_cp'):
                np.testing.assert_allclose(expected[key],actual[key].detach(),atol=1e-12,rtol=1e-12)
            prediction=actual['T'].T@anchors[:,3:]
            loss=-(torch.tensor(x[:,3:])*prediction.log()).sum()/len(x)+.1*(1-actual['normalized_cp'])
            loss.backward()
            for key,gradient in [('G',g.grad),('log_beta',beta.grad),('masses',masses.grad)]:
                np.testing.assert_allclose(expected['gradients'][key],gradient,atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(simplex_projection(masses.detach()+.2).numpy(),
                simplex_projection(params['masses']+.2),atol=1e-12)
