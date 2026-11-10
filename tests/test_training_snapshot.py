import unittest
import numpy as np
import pandas as pd
from acan import AlgebraicClusterLearner
from acan.core.network import HAS_TORCH

class TrainingSnapshotTests(unittest.TestCase):
    def test_best_objective_matches_exported_weights_and_batch(self):
        rng=np.random.default_rng(7)
        frame=pd.DataFrame(rng.uniform(size=(80,5)))
        profiles=[(p,d) for p in ('legacy-v8','paper-routing-v1') for d in (('numpy','cpu') if HAS_TORCH else ('numpy',))]
        for profile,device in profiles:
            learner=AlgebraicClusterLearner(frame,batch_size=20,device=device,routing_profile=profile)
            model=learner.nLearn(max_epochs=8,min_delta=0)
            self.assertAlmostEqual(model.cp,max(r['cp'] for r in learner.history),places=10)
            self.assertAlmostEqual(float(learner.model_net(learner.X)['cp']),model.cp,places=10)
