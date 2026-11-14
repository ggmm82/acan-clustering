import unittest
import numpy as np
import pandas as pd
from acan import AlgebraicClusterLearner, GraphClusterModel

class CPApplicationTests(unittest.TestCase):
    def test_energy_roundtrip_and_query_isolation(self):
        data=pd.DataFrame({'x':[0.,.1,.9,1.]*10,'y':[0.,0.,1.,1.]*10})
        learner=AlgebraicClusterLearner(data,device='numpy',routing_profile='cp-routing-v1')
        model=learner.nLearn(max_epochs=10)
        energies=[s['energy'] for s in learner.history]
        self.assertTrue(all(b>=a for a,b in zip(energies,energies[1:])))
        graph=GraphClusterModel();graph.clusterModel[model.id]=model;graph.rootId=model.id
        payload=graph.to_json()
        loaded=GraphClusterModel.loadModel(payload).clusterModel[model.id]
        before=loaded.to_json()
        a=loaded.predictTupleSoft([.1],[0])[0]
        loaded.predictTupleSoft([.99],[0])
        np.testing.assert_array_equal(a,loaded.predictTupleSoft([.1],[0])[0])
        np.testing.assert_array_equal(a,model.predictTupleSoft([.1],[0])[0])
        self.assertEqual(before,loaded.to_json())
        self.assertAlmostEqual(a.sum(),1.)
        self.assertTrue(np.isfinite(a).all())
        self.assertEqual(model.routing_profile,'cp-routing-v1')
        self.assertIs(learner.get_current_model(),model)

    def test_app_exposes_profile_and_math_page(self):
        from acan.vis.render import app, show_page
        layout=app.server.test_client().get('/_dash-layout')
        self.assertEqual(layout.status_code,200)
        self.assertIn(b'concept-graph-v1',layout.data)
        self.assertNotIn(b'"value":"cp-joint-v1"',layout.data.replace(b' ',b''))   # not offered for new models
        self.assertIn(b'page-mathematics',layout.data)
        self.assertEqual(show_page('Matematica')[-2],{'display':'block'})       # the last page is Informatica
        self.assertEqual(show_page('Informatica')[-1],{'display':'block'})
