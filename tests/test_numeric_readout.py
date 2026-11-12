"""Continuous targets are weighted means, not a collection of class codes."""
import unittest
import numpy as np
import pandas as pd
from acan import GraphClusterModel
from acan.core.supervised import objective,predict_assignments
from acan.preprocessing import transform_features
from acan.vis.datasets import split_for_training
from acan.vis.training import train_native
from acan.vis.inference import infer
from acan.vis.evaluation import evaluate

OPTIONS=dict(lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300,max_depth=1,
             max_epochs=40,min_samples_split=12,similarity_threshold=.92,routing_profile='cp-joint-v1')

class NumericReadoutTests(unittest.TestCase):
    def test_mixed_head_gradients_include_profiled_numeric_means(self):
        rng=np.random.default_rng(19)
        x=np.column_stack([rng.uniform(.1,.9,(9,3)),np.eye(2)[[0,1,0,1,0,1,0,1,0]],rng.uniform(.1,.9,(9,2))])
        params=dict(G=rng.normal(size=(3,7)),masses=np.array([.2,.5,.3]),log_beta=np.array(2.))
        state=dict(features=np.array([1,1,1,0,0,0,0],bool),root=x.mean(0,keepdims=True),den=np.ones((1,7))*.5,
            omega=np.ones((1,7)),lo=np.zeros((1,7)),span=np.ones((1,7)),regression=[5,6])
        result=objective(params,x,[[3,4]],state)
        for key in params:
            for idx in np.ndindex(params[key].shape):
                old=params[key][idx];params[key][idx]=old+1e-6
                plus=objective(params,x,[[3,4]],state,gradients=False)['loss']
                params[key][idx]=old-1e-6
                minus=objective(params,x,[[3,4]],state,gradients=False)['loss'];params[key][idx]=old
                self.assertAlmostEqual(result['gradients'][key][idx],(plus-minus)/2e-6,delta=1e-6)

    def test_numeric_readout_is_training_mean_persistent_and_target_independent(self):
        data=pd.DataFrame(dict(x=[0.,.05,.95,1.]*20,price=[100.,110.,890.,900.]*20))
        train,test,info=split_for_training(data,['price'],task='regression')
        graph=train_native(train,['price'],OPTIONS);root=graph.clusterModel[graph.rootId]
        self.assertEqual(info['target_types'],{'price':'regression'})
        self.assertNotIn(root.cols.index('price'),root.routing_parameters['features'])
        encoded=transform_features(train.drop(columns=['price']),graph.preprocessing)
        normalized=graph.alignData(encoded,list(encoded.columns));cix=[root.cols.index(c) for c in normalized]
        weights=predict_assignments(root,normalized.to_numpy(),cix)
        lo,hi=graph.minMax['price'];expected=weights@((train.price.to_numpy()-lo)/(hi-lo))/weights.sum(1)
        np.testing.assert_allclose(root.Groups[:,root.cols.index('price')],expected,atol=1e-12)
        payload=graph.to_json();restored=GraphClusterModel.loadModel(payload)
        before=restored.to_json();pred=infer(payload,test)
        np.testing.assert_allclose(pred.price,infer(restored.to_json(),test).price)
        changed=test.copy();changed['price']=1e12
        np.testing.assert_allclose(pred.price,infer(payload,changed).price)
        self.assertTrue(pred.price.between(train.price.min(),train.price.max()).all())
        self.assertEqual(before,restored.to_json())
        report=evaluate(payload,test,train=train,baselines=False)
        self.assertEqual({row['Metrica'] for row in report['scores']},{'MAE','RMSE','R²'})
        self.assertEqual(report['details'][0]['kind'],'regression')

    def test_constant_numeric_target_and_hierarchy_are_supported(self):
        data=pd.DataFrame(dict(x=np.linspace(0,1,32),price=np.full(32,500.)))
        graph=train_native(data,['price'],dict(OPTIONS,max_depth=3,max_epochs=5))
        np.testing.assert_allclose(infer(graph.to_json(),data).price,500.)
