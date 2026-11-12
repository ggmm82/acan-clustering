import unittest
import numpy as np
import pandas as pd
from acan import AlgebraicClusterLearner, GraphClusterModel
from acan.core.supervised import objective


class SupervisedCPTests(unittest.TestCase):
    def test_categorical_initialization_inverts_softmax(self):
        data=pd.DataFrame(dict(x=[0.,.5,1.],label_a=[1.,0.,0.],
                               label_b=[0.,1.,0.],label_c=[0.,0.,1.]))
        learner=AlgebraicClusterLearner(data,labels=['label'],device='numpy',lr=0.,
            batch_size=3,routing_profile='cp-joint-v1')
        node=learner.nLearn(max_epochs=1)
        expected=np.array([.01,.01,.99])/1.01
        np.testing.assert_allclose(np.sort(node.Groups[:,1:],axis=1),
            np.repeat(expected[None,:],node.k(),axis=0),atol=1e-12)

    def test_joint_hierarchy_marginalizes_paths_and_preserves_old_checkpoints(self):
        from acan.core.model import AlgebraicClusterModel
        def node(identifier,readouts,masses):
            result=AlgebraicClusterModel()
            result.id=identifier;result.cols=['x','label_a','label_b']
            result.Root=np.zeros((1,3));result.RootError=np.zeros((1,3))
            result.Groups=np.column_stack([np.zeros(2),readouts])
            result.Freqs=np.asarray(masses).reshape(1,-1)
            result.routing_profile='cp-joint-v1'
            result.routing_parameters=dict(features=[0],routing_relevance=np.ones((2,3)).tolist(),
                masses=masses,beta=1.,hierarchy_readout='conditional-mixture-v1')
            return result
        root=node(0,[[1.,0.],[0.,1.]],[.6,.4])
        child=node(1,[[1.,0.],[0.,1.]],[.2,.8])
        root.children={0:1}
        graph=GraphClusterModel();graph.clusterModel={0:root,1:child};graph.rootId=0
        data=pd.DataFrame(dict(x=np.zeros(257),label_a=np.ones(257),label_b=np.zeros(257)))
        labels=['label_a','label_b']
        result,confidence,paths=graph.assignLabel(root,data,labels)
        np.testing.assert_allclose(result,np.tile([.12,.88],(257,1)),atol=1e-12)
        np.testing.assert_allclose(confidence,.48,atol=1e-12)
        self.assertTrue(np.all(paths=='1_G1'))
        restored=GraphClusterModel.loadModel(graph.to_json())
        before=restored.to_json()
        actual=restored.assignLabel(restored.clusterModel[0],data,labels)
        for expected,observed in zip((result,confidence,paths),actual):
            np.testing.assert_array_equal(expected,observed)
        self.assertEqual(before,restored.to_json())
        child.Groups[:,1:]=[1.,0.]
        expanded=graph.assignLabel(root,data,labels)[0]
        root.children={}
        unexpanded=graph.assignLabel(root,data,labels)[0]
        np.testing.assert_allclose(expanded,unexpanded,atol=1e-12)
        root.children={0:1};child.Groups[:,1:]=[[1.,0.],[0.,1.]]
        root.routing_parameters.pop('hierarchy_readout')
        historical=graph.assignLabel(root,data,labels)[0]
        np.testing.assert_allclose(historical,np.tile([.2,.8],(257,1)),atol=1e-12)

    def test_child_anchors_initialize_in_the_observed_local_range(self):
        data=pd.DataFrame(dict(x=[.6,.62,.68,.8],label_a=[1.,1.,0.,0.],label_b=[0.,0.,1.,1.]))
        learner=AlgebraicClusterLearner(data,labels=['label'],device='numpy',lr=0.,
            batch_size=4,routing_profile='cp-joint-v1')
        node=learner.nLearn(max_epochs=1)
        np.testing.assert_allclose(np.sort(node.Groups[:,0]),[.602,.62,.68,.798],atol=1e-12)

    def test_joint_objective_gradients_match_finite_differences(self):
        rng=np.random.default_rng(4)
        x=np.column_stack([rng.uniform(.1,.9,(7,3)),np.eye(2)[[0,1,0,1,1,0,1]]])
        params=dict(G=rng.normal(size=(3,5)),log_beta=np.asarray(3.))
        state=dict(features=np.array([1,1,1,0,0],bool),root=x.mean(0,keepdims=True),den=np.ones((1,5))*.5,
                   omega=np.ones((1,5)),lo=np.zeros((1,5)),span=np.ones((1,5)))
        for weight in (0.,.1):
            result=objective(params,x,[[3,4]],state,weight)
            self.assertAlmostEqual(result['loss'],result['nll']+weight*(1-result['normalized_cp']))
            for key in params:
                for index in np.ndindex(params[key].shape):
                    old=params[key][index]
                    params[key][index]=old+1e-6
                    plus=objective(params,x,[[3,4]],state,weight,False)['loss']
                    params[key][index]=old-1e-6
                    minus=objective(params,x,[[3,4]],state,weight,False)['loss']
                    params[key][index]=old
                    self.assertAlmostEqual(result['gradients'][key][index],(plus-minus)/2e-6,delta=2e-7)
            changed=x.copy();changed[:,3:]=changed[:,3:][:,::-1]
            np.testing.assert_array_equal(result['T'],objective(params,changed,[[3,4]],state,weight,False)['T'])

    def test_supervised_model_persists_and_ignores_targets_and_other_queries(self):
        from acan.vis.training import train_native
        from acan.vis.inference import infer
        data=pd.DataFrame(dict(x=[0.,.05,.95,1.]*12,label=['a','a','b','b']*12))
        options=dict(lr=.03,tau=.2,alpha=.5,device='numpy',batch_size=20,max_depth=1,max_epochs=12,
                     min_samples_split=4,similarity_threshold=.92,routing_profile='cp-joint-v1')
        graph=train_native(data,['label'],options);root=graph.clusterModel[graph.rootId]
        self.assertEqual(root.routing_parameters['training_rows'],len(data))
        self.assertEqual(root.routing_parameters['objective'],'categorical-nll-cp-v1')
        self.assertTrue(all(np.isfinite(s['loss']) for s in root.history))
        payload=graph.to_json();loaded=GraphClusterModel.loadModel(payload)
        a=infer(payload,data)['label'].to_numpy()
        np.testing.assert_array_equal(a,infer(loaded.to_json(),data)['label'])
        self.assertGreater(np.mean(a==data['label']),.95)
        changed=data.copy();changed['label']='incorrect'
        np.testing.assert_array_equal(a,infer(payload,changed)['label'])
        np.testing.assert_array_equal(a[:1],infer(payload,data.iloc[:1])['label'])
        before=loaded.to_json();loaded.clusterModel[root.id].predictTupleSoft([.5],[0])
        self.assertEqual(before,loaded.to_json())
        r=root.history[-1]
        self.assertAlmostEqual(r['loss'],r['nll']+.1*(1-r['normalized_cp']))

    def test_supervised_hierarchy_uses_feature_only_branch_assignments(self):
        from acan.vis.training import train_native,graph_stats
        rng=np.random.default_rng(8)
        data=pd.DataFrame(dict(x=rng.uniform(size=72),label=['a','b']*36))
        options=dict(lr=.01,tau=.2,alpha=.5,device='numpy',batch_size=24,max_depth=2,max_epochs=2,
                     min_samples_split=3,similarity_threshold=1.,routing_profile='cp-joint-v1')
        graph=train_native(data,['label'],options)
        self.assertEqual(graph_stats(graph)['depth'],2)
        loaded=GraphClusterModel.loadModel(graph.to_json())
        for node in loaded.clusterModel.values():
            self.assertEqual(node.routing_parameters['objective'],'categorical-nll-cp-v1')
            a=node.predictTupleSoft([.25,1.,0.],[0,1,2])[0]
            b=node.predictTupleSoft([.25,0.,1.],[0,1,2])[0]
            np.testing.assert_array_equal(a,b)

    def test_hierarchy_purity_uses_observed_labels_not_confident_readouts(self):
        from unittest.mock import patch
        from acan.vis.training import train_native,graph_stats
        rng=np.random.default_rng(17)
        data=pd.DataFrame(dict(x=rng.uniform(size=512),label=['a','b']*256))
        original=AlgebraicClusterLearner.nLearn
        def confident_readout(learner, **unused):
            node=original(learner,max_epochs=1)
            indices=learner._supervised_groups[0]
            node.Groups[:,indices]=[1.,0.]
            return node
        options=dict(lr=.01,tau=.2,alpha=.5,device='numpy',batch_size=300,max_depth=2,max_epochs=1,
                     min_samples_split=3,similarity_threshold=.92,routing_profile='cp-joint-v1')
        with patch.object(AlgebraicClusterLearner,'nLearn',confident_readout):
            graph=train_native(data,['label'],options)
        self.assertEqual(graph_stats(graph)['depth'],2)
