"""Adattatore sperimentale: righe vettoriali riconosciute da ConceptModel.

Nessuna rete o ottimizzazione per gradiente. Il riconoscimento e la ricerca
rimangono quelli del modello tabulare; la lettura conserva somme e coppie sparse.
"""
import numpy as np
from scipy import sparse
from acan.core.concepts import ConceptModel
from acan.concept_algebra import categorical_reading, centre_variance


from acan.lm.text_rows import semantic_vectors, TextRows


class VectorTable:
    """ConceptModel invariato + lettura sparsa del token e prior vettoriale."""
    def __init__(self, vectors, context=8, **options):
        self.rows=TextRows(vectors,context)
        self.model=ConceptModel(**options)
        self.kappa=1.

    def fit(self, ids, rows=200000, chunk=2048, progress=None):
        read=lambda:self.rows.chunks(ids,rows,chunk)
        self.model.fit_chunks(read,labels=self.rows.targets,progress=progress)
        return self.read_training(ids,rows,chunk)

    def read_training(self, ids, rows=200000, chunk=2048):
        """Recount the training rows after a choice along the same learned path."""
        read=lambda:self.rows.chunks(ids,rows,chunk)
        k=self.model.concepts;v=len(self.rows.vectors);d=self.rows.vectors.shape[1]
        self.counts=sparse.csr_matrix((k,v),dtype=float)
        c=np.zeros((k,1));u=np.zeros((k,d));q=np.zeros((k,d))
        cursor=self.rows.context
        for frame in read():
            target=np.asarray(ids[cursor:cursor+len(frame)]);cursor+=len(frame)
            t=self.model.assignments(frame)
            one=sparse.csr_matrix((np.ones(len(frame)),(np.arange(len(frame)),target)),shape=(len(frame),v))
            self.counts+=sparse.csr_matrix(t.T)@one
            y=self.rows.vectors[target].astype(float)
            c+=t.sum(0)[:,None];u+=t.T@y;q+=t.T@(y*y)
        self.mass=c[:,0];self.distinct=np.diff(self.counts.indptr)
        root=np.var(self.rows.vectors[np.asarray(ids[:rows+self.rows.context])],axis=0)
        self.mean,self.variance=centre_variance(c,u,q,root[None],self.model.kappa,np)
        self.variance=np.maximum(self.variance,1e-12)
        e=self.rows.vectors.astype(float)
        logits=-.5*((self.mean*self.mean/self.variance).sum(1)[:,None]
                    -2*(self.mean/self.variance)@e.T+(1/self.variance)@(e*e).T)
        logits-=logits.max(1,keepdims=True)
        self.vector_prior=np.exp(logits);self.vector_prior/=self.vector_prior.sum(1,keepdims=True)
        return self

    def distribution(self, ids, positions):
        t=self.model.assignments(self.rows.frame(ids,positions))
        head=categorical_reading(self.counts.toarray(),self.mass[:,None],self.distinct[:,None],self.vector_prior,self.kappa,np)
        return t@head

    def true_probability(self, ids, positions):
        # Bounded query batches: never allocate positions x concepts x vocabulary.
        out=[]
        for start in range(0,len(positions),256):
            where=np.asarray(positions[start:start+256])
            t=self.model.assignments(self.rows.frame(ids,where))
            targets=np.asarray(ids)[where]
            probability=categorical_reading(self.counts[:,targets].toarray().T,self.mass[None],
                self.distinct[None],self.vector_prior[:,targets].T,self.kappa,np)
            out.append((t*probability).sum(1))
        return np.concatenate(out) if out else np.zeros(0)
