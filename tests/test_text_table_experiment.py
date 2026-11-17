"""Garanzie del prototipo fuori dal prodotto, senza reti o contesti esatti."""
import unittest
import numpy as np
from experiments.text_table.model import TextRows,VectorTable,semantic_vectors
from acan.concept_algebra import categorical_reading

class TextTableExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vectors=np.random.default_rng(5).normal(size=(7,3)).astype(np.float32)
        cls.ids=np.tile([0,1,2,0,3,4],20)
        cls.model=VectorTable(cls.vectors,context=2,max_concepts=8,judged_rows=24,
            refine_rows=32,chunk_rows=16,covariance=False).fit(cls.ids,rows=96,chunk=16)

    def test_relative_position_and_no_current_or_future_token(self):
        rows=TextRows(self.vectors,2);positions=np.array([0,1,5])
        frame=rows.frame(self.ids,positions)
        np.testing.assert_allclose(frame.iloc[2,:3],self.vectors[self.ids[4]])
        np.testing.assert_allclose(frame.iloc[2,3:],self.vectors[self.ids[3]])
        self.assertTrue(frame.iloc[0].isna().all())
        changed=self.ids.copy();changed[5:]=6
        np.testing.assert_allclose(rows.frame(self.ids,[5]),rows.frame(changed,[5]))
        np.testing.assert_allclose(self.model.distribution(self.ids,[5]),self.model.distribution(changed,[5]),atol=1e-12)

    def test_distribution_sums_to_one_and_true_probability_matches(self):
        positions=np.arange(1,18)
        p=self.model.distribution(self.ids,positions)
        self.assertTrue(np.all(p>=0));np.testing.assert_allclose(p.sum(1),1.,atol=1e-12)
        np.testing.assert_allclose(self.model.true_probability(self.ids,positions),p[np.arange(len(positions)),self.ids[positions]])
        np.testing.assert_allclose(self.model.distribution(self.ids,[len(self.ids)]).sum(),1.,atol=1e-12)

    def test_reading_equals_literal_concept_formula(self):
        self.model.kappa=2.
        t=self.model.model.assignments(self.model.rows.frame(self.ids,[5]))[0]
        expected=np.zeros(len(self.vectors))
        for i in range(len(t)):
            expected+=t[i]*categorical_reading(self.model.counts[i].toarray()[0],self.model.mass[i],
                self.model.distinct[i],self.model.vector_prior[i],2.,np)
        np.testing.assert_allclose(self.model.distribution(self.ids,[5])[0],expected,atol=1e-12)
        self.model.kappa=1.

    def test_sparse_counts_are_only_training_assignments(self):
        positions=np.arange(2,98)
        frame=self.model.rows.frame(self.ids,positions,targets=True)
        t=self.model.model.assignments(frame)
        expected=np.zeros(self.model.counts.shape)
        for row,token in enumerate(self.ids[positions]):expected[:,token]+=t[row]
        np.testing.assert_allclose(self.model.counts.toarray(),expected,atol=1e-9)
        self.assertEqual(self.model.counts[:,6].nnz,0)
        np.testing.assert_allclose(self.model.counts.sum(1).A1,self.model.mass,atol=1e-9)
        self.assertEqual(self.model.model.rows,96)

    def test_vectors_depend_only_on_the_supplied_training_sequence(self):
        first,s=semantic_vectors(self.ids,7,dimensions=3,top=4,seed=0)
        second,_=semantic_vectors(self.ids,7,dimensions=3,top=4,seed=0)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(first.shape,(7,3));self.assertTrue(np.isfinite(s).all())

    def test_observing_the_merge_path_does_not_change_learning(self):
        observed=[]
        again=VectorTable(self.vectors,context=2,max_concepts=8,judged_rows=24,
            refine_rows=32,chunk_rows=16,covariance=False).fit(self.ids,rows=96,chunk=16,
                progress=lambda step: observed.append((step['k'],step['score'])))
        self.assertTrue(observed)
        self.assertTrue(all(k>=1 and np.isfinite(score) for k,score in observed))
        np.testing.assert_allclose(again.counts.toarray(),self.model.counts.toarray(),atol=1e-10)
        np.testing.assert_allclose(again.model.assignments(again.rows.frame(self.ids,[5])),
            self.model.model.assignments(self.model.rows.frame(self.ids,[5])),atol=1e-12)
