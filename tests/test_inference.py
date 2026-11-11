import threading
import unittest

import numpy as np
import pandas as pd

from acan import AlgebraicClusterLearner, GraphClusterModel
from acan.vis.inference import infer
from acan.vis.jobs import TrainingJobs


class TestSavedInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame([
            {'Tempo': 'Sole', 'Luogo': 'Spiazzale', 'Gioco': 'SI'},
            {'Tempo': 'Sole', 'Luogo': 'Palazzetto', 'Gioco': 'NO'},
            {'Tempo': 'Piove', 'Luogo': 'Spiazzale', 'Gioco': 'NO'},
            {'Tempo': 'Piove', 'Luogo': 'Palazzetto', 'Gioco': 'SI'},
        ] * 25)
        cls.saved = AlgebraicClusterLearner.learn(
            cls.data, labels=['Gioco'], device='numpy', max_epochs=90,
            tau=0.15, alpha=0.6).to_json()

    def test_predict_saved_model_without_targets_or_training(self):
        result = infer(self.saved, self.data.iloc[:4].drop(columns='Gioco'))
        self.assertEqual(result.Gioco.tolist(), ['SI', 'NO', 'NO', 'SI'])
        # Targets in an uploaded labelled dataset must not influence prediction.
        wrong = self.data.iloc[:4].assign(Gioco='WRONG')
        self.assertEqual(infer(self.saved, wrong).Gioco.tolist(), result.Gioco.tolist())

    def test_complete_float_nan_categorical_column_preserves_input(self):
        data = self.data.iloc[:4].assign(Gioco=np.nan)
        result = infer(self.saved, data, 'complete')
        self.assertEqual(result.Gioco.tolist(), ['SI', 'NO', 'NO', 'SI'])
        self.assertTrue(data.Gioco.isna().all())
        pd.testing.assert_frame_equal(result[['Tempo', 'Luogo']], data[['Tempo', 'Luogo']])

    def test_complete_partial_missing_and_nondefault_index(self):
        data = self.data.iloc[:4].copy()
        data.index = [7, 8, 9, 10]
        data.loc[7, 'Gioco'] = None
        model = GraphClusterModel.loadModel(self.saved)
        result = model.predict_masked(data)
        self.assertEqual(result.Gioco.tolist(), ['SI', 'NO', 'NO', 'SI'])
        self.assertEqual(result.index.tolist(), [7, 8, 9, 10])

    def test_missing_model_or_unusable_inputs_report_error(self):
        with self.assertRaises(ValueError):
            infer(None, self.data)
        with self.assertRaises(ValueError):
            infer(self.saved, pd.DataFrame({'unseen': [1]}))
        with self.assertRaises(ValueError):
            infer(self.saved, self.data.assign(unseen=np.nan), 'complete')

    def test_numeric_completion(self):
        data = pd.DataFrame({'input_value': [0., .1, .2, .8, .9, 1.],
                             'output_value': [0., .1, .2, .8, .9, 1.]})
        model = AlgebraicClusterLearner.learn(data, labels=['output_value'],
                                             device='numpy', max_epochs=40)
        new = pd.DataFrame({'input_value': [.1, .9]})
        result = infer(model.to_json(), new)
        self.assertTrue(result.output_value.notna().all())
        self.assertTrue(result.output_value.between(0, 1).all())

    def test_predict_all_missing_features_uses_train_fitted_preprocessing(self):
        from acan.vis.training import train_native
        data=pd.DataFrame(dict(age=[1.,2.,np.nan,4.]*8,vote=['y','n',None,'y']*8,
                               Class=['a','b','a','a']*8))
        options=dict(lr=.08,tau=.2,alpha=.5,device='numpy',batch_size=300,
            max_depth=1,max_epochs=10,min_samples_split=12,similarity_threshold=.92)
        payload=train_native(data,['Class'],options).to_json()
        query=pd.DataFrame(dict(age=[np.nan],vote=[None]))
        result=infer(payload,query)
        self.assertIn(result.Class.iloc[0],['a','b'])
        self.assertTrue(np.isfinite(result.ACAN_confidence).all())
        with self.assertRaisesRegex(ValueError,'attributo noto'):
            infer(payload,pd.DataFrame(dict(unrelated=[1.])))
        with self.assertRaisesRegex(ValueError,'attributo noto'):
            infer(payload,query.assign(Class=None),'complete')


class BlockingLearner:
    def __init__(self):
        self.released = threading.Event()
        self.stopped = False

    def nLearn(self, **kwargs):
        if not self.released.wait(5):
            raise RuntimeError('Test learner timed out')

    def request_stop(self):
        self.stopped = True
        self.released.set()


class TestTrainingSessions(unittest.TestCase):
    def test_jobs_and_stop_are_isolated_and_duplicate_rejected(self):
        jobs = TrainingJobs()
        a, b = BlockingLearner(), BlockingLearner()
        ja = jobs.start('session-a', a, {}, ['A'])
        jb = jobs.start('session-b', b, {}, ['B'])
        try:
            with self.assertRaises(ValueError):
                jobs.start('session-a', BlockingLearner(), {}, [])
            self.assertIsNone(jobs.get('session-c'))
            self.assertTrue(jobs.stop('session-a'))
            self.assertTrue(ja['done'].wait(2))
            self.assertFalse(b.stopped)
            self.assertFalse(jb['done'].is_set())
            self.assertEqual(jb['labels'], ['B'])
        finally:
            jobs.stop('session-a')
            jobs.stop('session-b')
            jb['done'].wait(2)

    def test_job_errors_are_reported_and_allow_retry(self):
        class Broken:
            def nLearn(self, **kwargs):
                raise ValueError('broken training')
        jobs = TrainingJobs()
        job = jobs.start('a', Broken(), {}, [])
        self.assertTrue(job['done'].wait(2))
        self.assertEqual(job['error'], 'broken training')
        replacement = jobs.start('a', Broken(), {}, [])
        self.assertIsNot(replacement, job)
        self.assertTrue(replacement['done'].wait(2))
