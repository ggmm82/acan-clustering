"""Regressions for uploaded tables and train-fitted inference dimensions."""
import unittest
import numpy as np
import pandas as pd
from acan import GraphClusterModel
from acan.vis.datasets import clean_frame, target_and_filters, split_for_training, check_budget
from acan.vis.training import train_native
from acan.vis.inference import infer
from acan.preprocessing import fit_features, transform_features

OPTIONS=dict(max_depth=1,max_epochs=25,min_samples_split=5,similarity_threshold=.92,lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300)

class DatasetPreparationTests(unittest.TestCase):
    def test_identifiers_and_target_with_pandas_string_dtype(self):
        frame=pd.DataFrame({'PassengerId':range(100),'Name':pd.Series([f'person {i}' for i in range(100)],dtype='str'), 'Ticket':[f'T{i}' for i in range(100)], 'measurement':np.arange(100)/100, 'Survived':[0,1]*50,'Embarked':['A','B']*50})
        labels,filters=target_and_filters(frame)
        self.assertEqual(labels,['Survived'])
        self.assertEqual(set(filters),{'PassengerId','Name','Ticket'})
        train,test,info=split_for_training(frame,labels,filters)
        self.assertEqual(info['target_types'],{'Survived':'classification'})
        self.assertTrue(info['stratified'])
        self.assertEqual(train.Survived.nunique(),2)
        _,_,info=split_for_training(frame,labels,filters,task='regression')
        self.assertEqual(info['target_types']['Survived'],'regression')

    def test_unknown_target_excluded_without_inventing_a_class(self):
        frame=pd.DataFrame({'x':range(11),'Gioco':['SI','NO']*5+[' ? ']})
        train,test,info=split_for_training(frame,['Gioco'])
        self.assertEqual(info['excluded_missing_targets'],1)
        self.assertEqual(set(train.Gioco)|set(test.Gioco),{'SI','NO'})
        self.assertEqual(len(train)+len(test),10)

    def test_budget_reports_expansion_before_one_hot_allocation(self):
        frame=pd.DataFrame({'identifier':[f'id{i}' for i in range(1100)]})
        with self.assertRaisesRegex(ValueError,'identifier: 1100'):
            check_budget(frame)

    def test_fixed_feature_schema_and_training_only_imputation(self):
        frame=pd.DataFrame({'x':[0.,2.,np.nan,4.], 'color':['a','b','a','b'],'label':['A','B','A','B']})
        schema=fit_features(frame,['label'])
        encoded=transform_features(pd.DataFrame({'x':[np.nan],'color':['a']}),schema)
        self.assertEqual(encoded.loc[0,'x'],2.)
        self.assertEqual(encoded.loc[0,'x_NaN'],1.)
        self.assertEqual(encoded.loc[0,'color_b'],0.)

    def test_prediction_batch_invariance_and_saved_schema(self):
        frame=pd.DataFrame({'x':[0.,.1,.9,1.]*10,'color':['a','a','b','b']*10,'label':['A','A','B','B']*10})
        graph=train_native(frame,['label'],OPTIONS)
        saved=GraphClusterModel.loadModel(graph.to_json()).to_json()
        single=infer(saved,frame.iloc[[0]])
        batch=infer(saved,frame.iloc[[0,2]])
        self.assertEqual(single.label.iloc[0],batch.label.iloc[0])
        self.assertAlmostEqual(single.ACAN_confidence.iloc[0],batch.ACAN_confidence.iloc[0],places=12)
