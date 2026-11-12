"""Shared CP concepts: feature-only routing, categorical NLL and learned sparse support.

NumPy and exact analytic gradients keep the native CPU application self-contained.
This is an explicit supervised extension, not the fixed-anchor variational solver.
"""
import numpy as np
from acan.algebra import (concept_step, concept_assignments, prediction_cp_objective,
                         simplex_projection, mass_assignments)
from acan.algebra import simplex_coordinate_candidate, simplex_coordinate_index
from .network import _sigmoid_np, compute_group_behavior_coupling
from .learner import _NumpyAdam

WEIGHT = .1


def target_groups(learner):
    groups = []
    for label in learner.labels:
        if label in learner.cols:
            continue
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
    masses = params.get('masses')
    shared = concept_step(x,a,root,den,omega,beta,features,np.maximum,np.clip,softmax,
        masses=masses,exponential=np.exp,distinct=1-np.eye(len(a)))
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
    regression=state.get('regression',[])
    target_count=max(1,len(groups)+len(regression))
    da = np.zeros_like(a); dt = np.zeros_like(t)
    for indices in groups:
        raw_prediction = t.T@a[:, indices]
        prediction = raw_prediction.clip(1e-12)
        nll -= float((x[:, indices]*np.log(prediction)).sum())/n/target_count
        dp = -x[:, indices]/prediction/n/target_count*(raw_prediction>1e-12)
        da[:, indices] += t@dp
        dt += a[:, indices]@dp.T
    mse=0.
    if regression:
        means=centroid[:,regression]
        # Profile weighted least squares: the exact optimum of each readout
        # is its assignment-weighted training mean. Its mean derivative cancels.
        errors=x[None,:,regression]-means[:,None,:]
        squared=(errors*errors).sum(axis=2)
        mse=float((t*squared).sum())/n/target_count
        dt+=squared/n/target_count
        dmeans=2*means*(counts-mass)/n/target_count
        dt+=(dmeans/mass)@x[:,regression].T-(dmeans*means).sum(1,keepdims=True)/mass*(counts>1e-12)
    loss = prediction_cp_objective(nll+mse, cp, weight)
    result = dict(loss=loss, nll=nll, mse=mse, normalized_cp=cp, T=t, G=a, R=r, F=freq.T)
    if regression:result['regression_means']=means
    if masses is not None:
        result['overlap'] = shared['overlap']
    if not gradients:
        return result
    scale = -weight*.5/cp if cp > 1e-6 else 0.
    beta_extra = 0.
    pair_dw = np.zeros_like(w)
    if masses is None:
        dquality = scale*freq
        dfrequency = scale*q/s
    else:
        distinct_cp = shared['distinct_cp']
        scale = -weight*.5*shared['probability_affinity']/distinct_cp if distinct_cp>1e-6 else 0.
        quality, roots, z, overlap = (shared[key] for key in ('quality','quality_roots','z','overlap'))
        d_z = -2*scale*(overlap@z)
        root_denominator = np.maximum(quality,1e-12)**.5
        root_derivative = 1/root_denominator-.5*quality/root_denominator**3*(quality>1e-12)
        dquality = scale*freq+d_z*freq*root_derivative
        # Differentiate the same distinct-observation support factor as Torch.
        count_square=counts*counts
        support_den=np.maximum(count_square,1e-24)
        support_numerator=count_square-shared['square_counts']
        support_raw=support_numerator/support_den
        dsupport=(2*counts-2*t)/support_den-2*counts*support_numerator/support_den**2*(count_square>1e-24)
        dt += dquality*(q/s)*dsupport*((support_raw>=0)&(support_raw<=1))
        dquality = dquality*shared['evidence']
        dfrequency = scale*quality+d_z*roots
        frequency_den = np.maximum(freq,1e-300)**.5
        frequency_derivative = 1/frequency_den-.5*freq/frequency_den/np.maximum(freq,1e-300)*(freq>1e-300)
        dfrequency += -weight*distinct_cp*shared['prior_root']*frequency_derivative
        d_overlap = -scale*(z@z.T)*overlap
        raw_pair = (shared['pair_distances']+shared['pair_distances'].T)/2
        d_pair = -.5*beta*(d_overlap+d_overlap.T)*(raw_pair>=0)
        beta_extra = float((-beta*shared['pair_symmetric']*d_overlap).sum())
        da[:,features] += 2*w*(ax*d_pair.sum(1,keepdims=True)-d_pair@ax)
        da[:,features] += 2*(ax*(d_pair.T@w)-d_pair.T@(w*ax))
        pair_dw = ax**2*d_pair.sum(1,keepdims=True)-2*ax*(d_pair@ax)+d_pair@(ax**2)
    dr = dquality*(2*r/s-q/s**2*(r.sum(1, keepdims=True)>1e-12))
    deff = dr*np.sign(effective-root)/den*omega*(raw_r<1)
    da += .5*deff
    dpost = .5*deff
    dm = dfrequency/n-(dpost*centroid).sum(1, keepdims=True)/mass*(counts>1e-12)
    dt += (dpost/mass)@x.T+dm
    dz = t*(dt-(t*dt).sum(0, keepdims=True))
    da[:, features] += 2*beta*w*(dz@xx-ax*dz.sum(1, keepdims=True))
    dw = -beta*(ax**2*dz.sum(1, keepdims=True)-2*ax*(dz@xx)+dz@(xx**2))+pair_dw
    dh = (dw-(dw*w).sum(1, keepdims=True))/h.sum(1, keepdims=True)
    da[:, features] += 2*r0[:, features]*dh*np.sign(ax-root[:, features])/den[:, features]*omega[:, features]*(r0[:,features]<1)
    dg = da*state['span']*sigmoid*(1-sigmoid)*(np.abs(params['G'])<30)
    for indices in groups:
        p = a[:, indices]
        dg[:, indices] = p*(da[:, indices]-(da[:, indices]*p).sum(1, keepdims=True))
    result['gradients'] = dict(G=dg, log_beta=np.asarray((dz*scores).sum()+beta_extra)*((params['log_beta']>0)&(params['log_beta']<9)))
    if masses is not None:
        base = shared['base_assignments']
        normalization = (base*masses[:,None]).sum(0,keepdims=True).clip(1e-300)
        result['gradients']['masses'] = (base/normalization*(dt-(t*dt).sum(0,keepdims=True))).sum(1)
        prior = masses[:,None]
        prior_den = np.maximum(prior,1e-300)**.5
        prior_derivative = 1/prior_den-.5*prior/prior_den/np.maximum(prior,1e-300)*(prior>1e-300)
        result['gradients']['masses'] += (-weight*shared['distinct_cp']*shared['frequency_root']*prior_derivative).ravel()
    return result


def train(learner, epochs):
    candidate_budget = learner.k
    groups = target_groups(learner)
    x = learner.full_dataset.to_numpy(dtype=float)
    features = np.ones(learner.m, dtype=bool)
    for indices in groups:
        features[indices] = False
    regression=getattr(learner,'_regression_indices',[])
    features[regression]=False
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
    params = dict(G=np.log(scaled/(1-scaled)), log_beta=np.asarray(np.log(128.)),
                  masses=np.ones(learner.k)/learner.k)
    for indices in groups:
        # Categorical heads use softmax, whose inverse is log probability.
        # Log odds would sharpen a binary .99/.01 seed to .9999/.0001.
        params['G'][:, indices] = np.log(initial[:, indices])
    den = span/2+np.abs(learner.Root-(learner.RootMin+span/2))
    omega = compute_group_behavior_coupling(x,~features if learner.labels else None)
    state = dict(features=features, root=learner.Root, den=np.where(den>1e-8,den,1.),
                 omega=omega*(den>1e-8),lo=learner.RootMin,span=span,regression=regression)
    optimizer = _NumpyAdam({key:params[key] for key in ('G','log_beta')}, lr=learner.lr)
    mass_optimizer = _NumpyAdam({'masses':params['masses']}, lr=learner.lr)
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
            previous = {key:params[key].copy() for key in ('G','log_beta')}
            optimizer.step(result['gradients'])
            delta_g = {key:params[key]-previous[key] for key in previous}
            directional_g = sum(float((result['gradients'][key]*delta_g[key]).sum()) for key in previous)
            if directional_g >= 0:
                delta_g = {key:-learner.lr*result['gradients'][key] for key in previous}
                directional_g = sum(float((result['gradients'][key]*delta_g[key]).sum()) for key in previous)
            accepted_g = False
            if directional_g < 0:
                for power in range(20):
                    step = 2.**(-power)
                    for key in previous:
                        params[key][...] = previous[key]+step*delta_g[key]
                    candidate = objective(params,x[order[start:start+learner.batch_size]],groups,state,gradients=False)
                    if candidate['loss'] <= result['loss']+1e-4*step*directional_g:
                        accepted_g = True
                        break
            if not accepted_g:
                for key in previous:
                    params[key][...] = previous[key]
        # Coordinate optimization: structural masses use the entire training
        # objective, so a minibatch cannot discard concepts needed elsewhere.
        result = objective(params,x,groups,state)
        old = params['masses'].copy()
        mass_optimizer.step(result['gradients'])
        delta = simplex_projection(params['masses'])-old
        directional = float(result['gradients']['masses']@delta)
        if directional >= 0:
            # Momentum can point uphill at a simplex boundary. Fall back to
            # feasible projected gradient descent, using the same objective.
            delta = simplex_projection(old-learner.lr*result['gradients']['masses'])-old
            directional = float(result['gradients']['masses']@delta)
        accepted = False
        if directional < 0:
            for power in range(20):
                step = 2.**(-power)
                params['masses'][:] = old+step*delta
                candidate = objective(params,x,groups,state,gradients=False)
                if candidate['loss'] <= result['loss']+1e-4*step*directional:
                    result = candidate
                    accepted = True
                    break
        if not accepted:
            params['masses'][:] = old
        coordinate = simplex_coordinate_index(params['masses'],epoch)
        proposal = (simplex_coordinate_candidate(params['masses'],coordinate,result['overlap'][coordinate])
                    if coordinate is not None else None)
        if proposal is not None:
            old = params['masses'].copy()
            params['masses'][:] = proposal
            candidate = objective(params,x,groups,state,gradients=False)
            if candidate['loss'] <= result['loss']:
                result = candidate
            else:
                params['masses'][:] = old
        learner.history.append(dict(epoch=epoch,loss=result['loss'],nll=result['nll'],
            mse=result['mse'],prediction_loss=result['nll']+result['mse'],
            normalized_cp=result['normalized_cp'],cp=result['normalized_cp']*np.sqrt(learner.m),
            cp_weight=WEIGHT,k_active=int((params['masses']>0).sum())))
        if result['loss'] < best_loss:
            best_loss = result['loss']; best = {k:v.copy() for k,v in params.items()}
    # This is the exact support of the constrained solution. Zero-mass slots
    # contribute nothing to routing, CP or prediction; no merge/prune tolerance.
    support = best['masses']>0
    best['G'] = best['G'][support]
    best['masses'] = best['masses'][support]
    learner.k = int(support.sum())
    result = objective(best,x,groups,state,gradients=False)
    # Persist only a bounded visualization sample; frequencies/CP use all training rows.
    sample = learner.NormDataset
    view = objective(best,sample,groups,state,gradients=False)
    learner.Groups,learner.Relevance = result['G'],result['R']
    if regression:learner.Groups[:,regression]=result['regression_means']
    learner.Transformation = view['T']
    hard = np.zeros_like(view['T']); hard[view['T'].argmax(0),np.arange(len(sample))] = 1
    learner.Transformation_hard = hard
    learner.Counts = np.bincount(result['T'].argmax(0),minlength=learner.k)[:,None]
    learner.Freqs = result['F']; learner.cp = result['normalized_cp']*np.sqrt(learner.m)
    learner.k_active = int((learner.Counts>0).sum())
    anchors,_ = probabilities(best['G'],groups,state['lo'],state['span'])
    learner.variational_state = dict(objective='mixed-nll-mse-cp-v1' if regression and groups else 'expected-mse-cp-v1' if regression else 'categorical-nll-cp-v1' if groups else 'normalized-cp-v1',cp_weight=WEIGHT,
        cp_definition='pair-supported-cp-v1',masses=best['masses'].tolist(),
        hierarchy_readout='conditional-mixture-v1',
        candidate_budget=candidate_budget,
        beta=float(np.exp(np.clip(best['log_beta'],0,9))),features=np.flatnonzero(features).tolist(),
        routing_relevance=np.clip(np.abs(anchors-state['root'])/state['den']*state['omega'],0,1).tolist(),
        numeric_targets=regression,numeric_readout='weighted-training-mean',
        numeric_anchor_values=anchors[:,regression].tolist(),training_mse=result['mse'],
        training_rows=len(x),training_nll=result['nll'],training_objective=result['loss'])
    return learner.setModel()


def predict_assignments(model, rows, cix):
    """Frozen feature-only posterior for one or more independent observations."""
    state = model.routing_parameters
    features = set(state['features'])
    # Never route on targets, even when hierarchy splitting passes complete rows.
    observed = [(position,index) for position,index in enumerate(cix) if index in features]
    if not observed:
        weights = np.repeat(np.asarray(model.Freqs).reshape(-1,1),len(rows),axis=1)
    else:
        positions,indices = zip(*observed)
        a = model.Groups[:,indices]
        r = np.asarray(state['routing_relevance'])[:,indices]
        def softmax(scores):
            value = np.exp(scores-scores.max(axis=0,keepdims=True))
            return value/value.sum(axis=0,keepdims=True)
        weights,_,_ = concept_assignments(np.asarray(rows)[:,positions],a,r,
            state['beta'],slice(None),softmax)
        if 'masses' in state:
            weights,_ = mass_assignments(weights,np.asarray(state['masses']),np.maximum)
    return weights / weights.sum(axis=0,keepdims=True)


def predict(model, row, cix):
    weights = predict_assignments(model,np.asarray(row).reshape(1,-1),cix)[:,0]
    best = int(weights.argmax())
    return weights,float(weights[best]),best
