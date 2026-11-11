"""Large model navigation stays bounded without discarding trained prototypes."""
import importlib.util
import unittest
import numpy as np
import pandas as pd
from acan import GraphClusterModel
from acan.core.model import AlgebraicClusterModel
from acan.vis.graph_view import elements,navigation


def fixture():
    graph=GraphClusterModel();graph.labels=['label']
    graph.minMax={col:(0.,1.) for col in ('x','label_A','label_B')}
    def node(k):
        model=AlgebraicClusterModel.summarize(pd.DataFrame(dict(x=[0.,1.],label_A=[1.,0.],label_B=[0.,1.])))
        model.n=160;model.Groups=np.tile([.5,.8,.2],(k,1));model.Relevance=np.ones((k,3))*.5
        model.Counts=np.ones((k,1))*2;model.Freqs=np.ones((k,1))/k
        return model
    root,child=node(80),node(24)
    root.Counts[65:]=0;root.children[0]=child.id
    graph.rootId=root.id;graph.clusterModel={root.id:root,child.id:child}
    return graph


@unittest.skipUnless(importlib.util.find_spec('dash'),'UI extra required')
class GraphViewTests(unittest.TestCase):
    def test_pages_are_bounded_and_do_not_mutate_the_model(self):
        graph=fixture();before=graph.to_json()
        a=elements(graph,page=1);b=elements(graph,page=2)
        nodes=lambda values:{e['data']['id'] for e in values if 'source' not in e['data']}
        self.assertEqual(len(nodes(a)),13)
        self.assertEqual(nodes(a)&nodes(b),{str(graph.rootId)})
        for edge in a:
            if 'source' in edge['data']:
                self.assertIn(edge['data']['source'],nodes(a));self.assertIn(edge['data']['target'],nodes(a))
        self.assertEqual(before,graph.to_json())
        self.assertTrue(all('position' in e for e in a if 'source' not in e['data']))
        self.assertIn('child_id',next(e for e in a if e['data']['id']==f'{graph.rootId}_G0')['data'])

    def test_empty_counts_navigation_and_outdated_pages_are_explicit(self):
        graph=fixture();root=graph.rootId;child=graph.clusterModel[root].children[0]
        _,_,pages,page,summary=navigation(graph,root,99)
        self.assertEqual(page,6);self.assertEqual(len(pages),6)
        self.assertIn('15 prototipi senza assegnazioni hard',summary)
        self.assertEqual(len(navigation(graph,root,1,True)[2]),7)
        self.assertEqual(navigation(graph,child,1,action='up')[1],root)
        self.assertEqual(navigation(graph,child,1,action='root')[1],root)
        self.assertEqual(navigation(graph,'missing')[1],root)
        self.assertEqual(len(elements(graph,root,6)),11) # Root + five cards + five edges.

    def test_overview_contains_only_concept_nodes(self):
        from acan.vis.render import update_graph
        graph=fixture()
        overview=update_graph(graph.to_json(),'hierarchy')
        self.assertEqual(len([e for e in overview if 'source' not in e['data']]),2)
