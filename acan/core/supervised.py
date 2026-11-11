"""Supervised concept prototypes: feature-only routing, categorical NLL + normalized CP.

NumPy and exact analytic gradients keep the native CPU application self-contained.
This is an explicit supervised extension, not the fixed-anchor variational solver.
"""
import numpy as np
from acan.algebra import concept_step, concept_assignments, prediction_cp_objective
from .network import _sigmoid_np
from .learner import _NumpyAdam

WEIGHT = .1


def target_groups(learner):
    groups = []
    for label in learner.labels:
        indices = [i for i, col in enumerate(learner.cols) if str(col).startswith(str(label) + '_')]
        if len(indices) < 2:
            return []
        groups.append(indices)
    return groups


def probabilities(logits, groups, lo, span):
    sigmoid = _sigmoid_np(logits)
    anchors = lo + span * sigmoid
    for indices in groups:
        values = logits[:, indices]
        p = np.exp(values-values.max(1, keepdims=True))
        anchors[:, indices] = p / p.sum(1, keepdims=True)
    return anchors, sigmoid


def objective(params, x, groups, state, weight=WEIGHT, gradients=True):
    """Return the joint objective and its exact derivatives; targets never route."""
    a, sigmoid = probabilities(params['G'], groups, state['lo'], state['span'])
    features = state['features']
    root, den, omega = (state[key] for key in ('root', 'den', 'omega'))
    beta = np.exp(np.clip(params['log_beta'], 0, 9))
    def softmax(scores):
        t = np.exp(scores-scores.max(0, keepdims=True))
        return t/t.sum(0, keepdims=True)
    shared = concept_step(x,a,root,den,omega,beta,features,np.maximum,np.clip,softmax)
    r0,w,scores,t = (shared[key] for key in ('routing_relevance','weights','scores','T'))
    h = r0[:, features]**2 + 1e-4
    ax, xx = a[:, features], x[:, features]
    n = len(x)
    counts = t.sum(1, keepdims=True)
    mass = counts.clip(1e-12)
    centroid = shared['G']
    effective = .5*(centroid+a)
    raw_r = np.abs(effective-root)/den*omega
    r = shared['R']
    s = r.sum(1, keepdims=True).clip(1e-12)
    q = (r*r).sum(1, keepdims=True)
    freq = shared['F'].T
    cp = shared['normalized_cp']
    nll = 0.
    da = np.zeros_like(a); dt = np.zeros_like(t)
    for indices in groups:
        raw_prediction = t.T@a[:, indices]
        prediction = raw_prediction.clip(1e-12)
        nll -= float((x[:, indices]*np.log(prediction)).sum())/n/len(groups)
        dp = -x[:, indices]/prediction/n/len(groups)*(raw_prediction>1e-12)
        da[:, indices] += t@dp
        dt += a[:, indices]@dp.T
    loss = prediction_cp_objective(nll, cp, weight)
    result = dict(loss=loss, nll=nll, normalized_cp=cp, T=t, G=a, R=r, F=freq.T)
    if not gradients:
        return result
    scale = -weight*.5/cp if cp > 1e-6 else 0.
    dr = scale*freq*(2*r/s-q/s**2*(r.sum(1, keepdims=True)>1e-12))
    deff = dr*np.sign(effective-root)/den*omega*(raw_r<1)
    da += .5*deff
    dpost = .5*deff
    dm = scale*q/s/n-(dpost*centroid).sum(1, keepdims=True)/mass*(counts>1e-12)
    dt += (dpost/mass)@x.T+dm
    dz = t*(dt-(t*dt).sum(0, keepdims=True))
    da[:, features] += 2*beta*w*(dz@xx-ax*dz.sum(1, keepdims=True))
    dw = -beta*(ax**2*dz.sum(1, keepdims=True)-2*ax*(dz@xx)+dz@(xx**2))
    dh = (dw-(dw*w).sum(1, keepdims=True))/h.sum(1, keepdims=True)
    da[:, features] += 2*r0[:, features]*dh*np.sign(ax-root[:, features])/den[:, features]*omega[:, features]*(r0[:,features]<1)
    dg = da*state['span']*sigmoid*(1-sigmoid)*(np.abs(params['G'])<30)
    for indices in groups:
        p = a[:, indices]
        dg[:, indices] = p*(da[:, indices]-(da[:, indices]*p).sum(1, keepdims=True))
    result['gradients'] = dict(G=dg, log_beta=np.asarray((dz*scores).sum())*((params['log_beta']>0)&(params['log_beta']<9)))
    return result


def train(learner, epochs):
    groups = target_groups(learner)
    x = learner.full_dataset.to_numpy(dtype=float)
    features = np.ones(learner.m, dtype=bool)
    for indices in groups:
        features[indices] = False
    if not features.any():
        raise ValueError('Il training supervisionato richiede attributi oltre al target.')
    rng = np.random.default_rng(learner.random_state)
    # Class-balanced anchors retain every observed class when the budget permits.
    classes = x[:, groups[0]].argmax(1) if groups else np.zeros(len(x), dtype=int)
    buckets = [list(rng.permutation(np.flatnonzero(classes == c))) for c in np.unique(classes)]
    chosen = []
    while len(chosen) < learner.k:
        for bucket in buckets:
            if bucket and len(chosen) < learner.k:
                chosen.append(bucket.pop())
    initial = x[chosen].clip(.01, .99)
    span = learner.RootMax-learner.RootMin
    # Invert the same local affine sigmoid used by probabilities(). This also
    # initializes children from their own records in global normalization units.
    scaled = np.divide(x[chosen]-learner.RootMin,span,
        out=np.full_like(x[chosen],.5),where=span>1e-8).clip(.01,.99)
    params = dict(G=np.log(scaled/(1-scaled)), log_beta=np.asarray(np.log(128.)))
    for indices in groups:
        params['G'][:, indices] = np.log(initial[:, indices])
    den = span/2+np.abs(learner.Root-(learner.RootMin+span/2))
    state = dict(features=features, root=learner.Root, den=np.where(den>1e-8,den,1.),
                 omega=learner.model_net.omega*(den>1e-8),lo=learner.RootMin,span=span)
    optimizer = _NumpyAdam(params, lr=learner.lr)
    best, best_loss = {k:v.copy() for k,v in params.items()}, float('inf')
    learner.history = []
    for epoch in range(max(1,int(epochs))):
        if learner._should_stop:
            break
        learner.iteration = epoch
        order = rng.permutation(len(x))
        for start in range(0,len(x),learner.batch_size):
            result = objective(params,x[order[start:start+learner.batch_size]],groups,state)
            if not np.isfinite(result['loss']) or not all(np.isfinite(g).all() for g in result['gradients'].values()):
                raise ValueError('Obiettivo supervisionato o gradienti non finiti.')
            optimizer.step(result['gradients'])
        result = objective(params,x,groups,state,gradients=False)
        learner.history.append(dict(epoch=epoch,loss=result['loss'],nll=result['nll'],
            normalized_cp=result['normalized_cp'],cp=result['normalized_cp']*np.sqrt(learner.m),
            cp_weight=WEIGHT,k_active=int((result['T'].sum(1)>1e-6).sum())))
        if result['loss'] < best_loss:
            best_loss = result['loss']; best = {k:v.copy() for k,v in params.items()}
    result = objective(best,x,groups,state,gradients=False)
    # Persist only a bounded visualization sample; frequencies/CP use all training rows.
    sample = learner.NormDataset
    view = objective(best,sample,groups,state,gradients=False)
    learner.Groups,learner.Relevance = result['G'],result['R']
    learner.Transformation = view['T']
    hard = np.zeros_like(view['T']); hard[view['T'].argmax(0),np.arange(len(sample))] = 1
    learner.Transformation_hard = hard
    learner.Counts = np.bincount(result['T'].argmax(0),minlength=learner.k)[:,None]
    learner.Freqs = result['F']; learner.cp = result['normalized_cp']*np.sqrt(learner.m)
    learner.k_active = int((learner.Counts>0).sum())
    anchors,_ = probabilities(best['G'],groups,state['lo'],state['span'])
    learner.variational_state = dict(objective='categorical-nll-cp-v1' if groups else 'normalized-cp-v1',cp_weight=WEIGHT,
        beta=float(np.exp(np.clip(best['log_beta'],0,9))),features=np.flatnonzero(features).tolist(),
        routing_relevance=np.clip(np.abs(anchors-state['root'])/state['den']*state['omega'],0,1).tolist(),
        training_rows=len(x),training_nll=result['nll'],training_objective=result['loss'])
    return learner.setModel()


def predict(model, row, cix):
    state = model.routing_parameters
    features = set(state['features'])
    # Never route on targets, even when hierarchy splitting passes complete rows.
    observed = [(position,index) for position,index in enumerate(cix) if index in features]
    if not observed:
        weights = np.asarray(model.Freqs).reshape(-1).copy()
    else:
        positions,indices = zip(*observed)
        a = model.Groups[:,indices]
        r = np.asarray(state['routing_relevance'])[:,indices]
        def softmax(scores):
            value = np.exp(scores-scores.max(axis=0,keepdims=True))
            return value/value.sum(axis=0,keepdims=True)
        weights,_,_ = concept_assignments(np.asarray(row).reshape(1,-1)[:,positions],a,r,
            state['beta'],slice(None),softmax)
        weights = weights[:,0]
    weights /= weights.sum()
    best = int(weights.argmax())
    return weights,float(weights[best]),best
