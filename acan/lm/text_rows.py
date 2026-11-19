"""Causal numeric rows for the tabular concept graph."""
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd


def semantic_vectors(ids, vocabulary, dimensions=32, top=512, seed=0, starts=None):
    ids=np.asarray(ids,dtype=np.int64)
    frequency=np.bincount(ids,minlength=vocabulary)
    top=min(top,vocabulary)
    common=np.argsort(-frequency,kind='stable')[:top]
    where=np.full(vocabulary,top,dtype=int);where[common]=np.arange(top)
    left,right=ids[:-1],ids[1:]
    if starts is not None:
        valid=np.ones(len(left),bool)
        valid[np.asarray(starts)[1:]-1]=False
        left,right=left[valid],right[valid]
    before=np.bincount(right*(top+1)+where[left],minlength=vocabulary*(top+1)).reshape(vocabulary,top+1)
    after=np.bincount(left*(top+1)+where[right],minlength=vocabulary*(top+1)).reshape(vocabulary,top+1)
    profile=np.hstack([np.sqrt(a/np.maximum(a.sum(1,keepdims=True),1)) for a in (before,after)])
    center=profile[frequency>0].mean(0)
    _,singular,directions=randomized_svd((profile if starts is None else profile[frequency>0])-center,n_components=min(dimensions,profile.shape[1]),n_iter=5,random_state=seed)
    return ((profile-center)@directions.T).astype(np.float32),singular


class TextRows:
    """La riga p contiene solo E[x_(p-1)],...,E[x_(p-k)]; il target è separato."""
    def __init__(self, vectors, context=8):
        self.vectors=np.asarray(vectors);self.context=int(context)
        d=self.vectors.shape[1]
        self.features=[f'past_{lag}_{j}' for lag in range(1,context+1) for j in range(d)]
        self.targets=[f'next_{j}' for j in range(d)]

    def frame(self, ids, positions, targets=False, observed_context=None):
        ids=np.asarray(ids,dtype=np.int64);positions=np.asarray(positions,dtype=int)
        d=self.vectors.shape[1]
        width = self.context if observed_context is None else observed_context
        if not 1 <= width <= self.context:
            raise ValueError('Finestra osservata fuori dai limiti della riga.')
        columns = self.features[:width*d]
        values=np.full((len(positions),len(columns)),np.nan)
        for lag in range(1,width+1):
            valid=positions>=lag
            values[valid,(lag-1)*d:lag*d]=self.vectors[ids[positions[valid]-lag]]
        result=pd.DataFrame(values,columns=columns)
        if targets:result[self.targets]=self.vectors[ids[positions]]
        return result

    def chunks(self, ids, rows, chunk=2048):
        rows=min(rows,len(ids)-self.context)
        for start in range(self.context,self.context+rows,chunk):
            yield self.frame(ids,np.arange(start,min(start+chunk,self.context+rows)),targets=True)
