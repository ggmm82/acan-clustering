import unittest
import numpy as np
from scipy import sparse
from experiments.text_table.reading import LevelReading


class LevelReadingTests(unittest.TestCase):
    def setUp(self):
        self.counts = np.array([[8., 2., 0., 0.], [0., 3., 7., 0.], [2., 1., 6., 0.]])
        self.parents = np.array([[.8, .2, .1], [.2, .8, .9]])
        self.reader = LevelReading(sparse.csr_matrix(self.counts), [self.parents], kappa=.5)

    def test_formula_and_mass_at_every_level(self):
        r = self.reader
        root = self.counts.sum(0)
        prior = (root + .5*np.count_nonzero(root)/4)/(root.sum()+.5*np.count_nonzero(root))
        general = self.parents @ self.counts
        d = (general > 0).sum(1, keepdims=True)
        general = (general + .5*d*prior)/(general.sum(1,keepdims=True)+.5*d)
        d = (self.counts > 0).sum(1,keepdims=True)
        expected = (self.counts + .5*d*(self.parents.T @ general))/(self.counts.sum(1,keepdims=True)+.5*d)
        actual = r.probability(np.arange(4))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        np.testing.assert_allclose(actual.sum(1),1.,atol=1e-12)
        self.assertTrue(np.all(actual[:,3] > 0))
        np.testing.assert_allclose(r.counts[0].sum(), r.counts[-1].sum(), rtol=0, atol=1e-12)

    def test_document_updates_only_after_revelation_and_reset(self):
        r = self.reader; query=np.array([.6,.3,.1]); state=r.document()
        initial = query @ r.probability(np.arange(4))
        np.testing.assert_allclose(state.probability(query,np.arange(4)),initial)
        state.probability(query,[2])
        self.assertEqual(state.observations,0)
        state.observe(query,3)
        updated=state.probability(query,np.arange(4))
        self.assertGreater(updated[3],initial[3])
        np.testing.assert_allclose(updated.sum(),1.)
        np.testing.assert_allclose(r.document().probability(query,np.arange(4)),initial)

    def test_memory_has_no_position_axis_for_long_prefix(self):
        state=self.reader.document(); query=np.array([.6,.3,.1])
        for i in range(10000):state.observe(query,i%4)
        self.assertEqual(state.observations,10000)
        self.assertEqual(len(state.counts),4)
        self.assertEqual(sum(a.size for a in state.counts.values()),12)
        np.testing.assert_allclose(state.mass.sum(),10000.)
        np.testing.assert_allclose(state.probability(query,np.arange(4)).sum(),1.)

    def test_prefix_statistics_are_additive(self):
        state=self.reader.document()
        for weights,token in [(np.array([1.,0.,0.]),1),(np.array([.2,.3,.5]),2)]:state.observe(weights,token)
        np.testing.assert_allclose(state.mass,[1.2,.3,.5])
        np.testing.assert_allclose(state.distinct,[2,1,1])
        np.testing.assert_allclose(state.counts[2],[.2,.3,.5])

    def test_cursor_has_bounded_ordered_context_and_causal_memberships(self):
        from experiments.text_table.model import VectorTable
        from experiments.text_table.reading import TextCursor
        rng=np.random.default_rng(3)
        table=VectorTable(rng.normal(size=(4,2)),context=2,max_concepts=4,judged_rows=16,
            refine_rows=64,covariance=False).fit(np.tile([0,1,2,3],20),rows=64)
        reading=LevelReading(table.counts,[level['parents'] for level in table.model.levels[1:]])
        cursor=TextCursor(table,reading);ids=np.array([0,2,1,3,2,0])
        for pos,token in enumerate(ids):
            expected=table.model.assignments(table.rows.frame(ids,[pos]))[0]
            np.testing.assert_allclose(cursor.membership(),expected,atol=1e-12)
            before=cursor.distribution()
            np.testing.assert_allclose(cursor.distribution(),before)
            np.testing.assert_allclose(before.sum(),1.,atol=1e-12)
            self.assertEqual(cursor.state.observations,pos)
            cursor.observe(token)
        self.assertEqual(len(cursor.window),2)
        self.assertEqual(cursor.state.observations,len(ids))

    def test_document_formula_uses_all_levels_of_the_same_graph(self):
        r=self.reader; state=r.document(); w=np.array([.2,.3,.5]);state.observe(w,2)
        vocabulary=np.arange(4)
        def read(n, prior):
            n=np.asarray(n);d=np.maximum((n>0).sum(-1,keepdims=True),1)
            return (n+.5*d*prior)/(n.sum(-1,keepdims=True)+.5*d)
        root=read(r.root,np.full(4,.25))
        document=np.array([0.,0.,1.,0.])
        prior=read(document,root)[None]
        for i,c in enumerate(r.counts):
            if i:prior=r.parents[i-1].T@prior
            global_head=read(c.toarray(),prior)
            doc=np.zeros(c.shape)
            doc[:,2]=self.parents@w if i==0 else w
            prior=read(doc,global_head)
        np.testing.assert_allclose(state.probability(w,vocabulary),w@prior,atol=1e-12)
        for mass in state.level_mass:np.testing.assert_allclose(mass.sum(),1.)

    def test_batched_document_reading_is_exactly_causal_streaming(self):
        ids=np.array([0,2,1,2,3,0,0,2])
        weights=np.random.default_rng(8).dirichlet(np.ones(3),size=len(ids))
        weights[2]=[1.,0.,0.]
        prefix=self.reader.prefix_statistics(weights,ids)
        for kappa in (.03125,.5,8.):
            self.reader.kappa=kappa
            state=self.reader.document();expected=[]
            for t,y in zip(weights,ids):
                expected.append(state.probability(t,[y])[0]);state.observe(t,y)
            actual=self.reader.sequence_probability(weights,ids,prefix)
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            changed=ids.copy();changed[5:]=3
            np.testing.assert_allclose(self.reader.sequence_probability(weights,changed)[:5],actual[:5],atol=1e-12)
