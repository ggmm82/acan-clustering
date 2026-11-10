"""Bounded CPU CP routing. Shared batch/query energy, frozen training statistics."""
import numpy as np

PROFILE = 'cp-routing-v1'

def _scalar(value):
    return float(value.detach()) if hasattr(value,'detach') else float(value)


class _TorchOps:
    def __init__(self,x):
        import torch
        self.torch,self.x = torch,x
    def asarray(self,a,dtype=None):
        return self.torch.as_tensor(a,dtype=self.x.dtype,device=self.x.device)
    def zeros(self,shape):
        return self.x.new_zeros(shape)
    def zeros_like(self,a):
        return self.torch.zeros_like(a)
    def maximum(self,a,b):
        return self.torch.maximum(self.asarray(a),self.asarray(b))
    def __getattr__(self,name):
        return getattr(self.torch,name)


def _softmax(z):
    if hasattr(z,'detach'):
        t=z.softmax(0).clamp_min(1e-14)
        return t/t.sum(0,keepdim=True)
    t = np.exp(z-z.max(axis=0, keepdims=True))
    t = np.maximum(t, 1e-14)
    return t/t.sum(axis=0, keepdims=True)


def solve(x, state, steps=30, base_counts=None, base_sums=None):
    xp = _TorchOps(x) if hasattr(x,"detach") else np
    x = xp.asarray(x, dtype=float)
    a, omega, root, den = (xp.asarray(state[k], dtype=float) for k in ('anchors','omega','root','den'))
    k, m = a.shape
    bc = xp.zeros((k,1)) if base_counts is None else xp.asarray(base_counts).reshape(k,1)
    bs = xp.zeros_like(a) if base_sums is None else xp.asarray(base_sums)
    total = _scalar(bc.sum()+len(x))
    relevance = xp.abs(a-root)/den*omega
    h = relevance**2
    w = h/(h.mean(1,keepdims=True)+1e-6)/m
    scores = -8.*xp.maximum((w*a*a).sum(1,keepdims=True)-2*(w*a)@x.T+w@(x*x).T,0)
    def evaluate(t, gradient=False):
        raw_mass = bc+t.sum(1,keepdims=True)
        mass = xp.maximum(raw_mass,1e-6)
        centroid = (bs+t@x)/mass
        effective = .5*(centroid+a)
        r = xp.abs(effective-root)/den*omega
        s = xp.maximum(r.sum(1,keepdims=True),1e-6)
        q = (r*r).sum(1,keepdims=True)
        freq = mass/total
        raw_cp = (freq*m*q/s).sum()
        cp = xp.sqrt(xp.maximum(raw_cp,1e-12))
        entropy = -(t*xp.log(t)).sum()/total
        prior = ((freq+1e-4)*xp.log(freq+1e-4)-(freq+1e-4)).sum()
        value = cp+(t*scores).sum()/total+entropy+.5*prior
        if not gradient:
            return value, cp, centroid, r, mass
        scale = .5/cp*(raw_cp>1e-12)
        dr = scale*freq*m*(2*r/s-q/s**2*(r.sum(1,keepdims=True)>1e-6))
        dg = .5*dr*xp.sign(effective-root)/den*omega
        dm = (scale*m*q/s/total-(dg*centroid).sum(1,keepdims=True)/mass)*(raw_mass>1e-6)
        grad = (dg/mass)@x.T+dm
        grad += (scores-xp.log(t)-1+.5*xp.log(freq+1e-4))/total
        return grad
    t = _softmax(scores)
    history = []
    for iteration in range(max(1,min(int(steps),30))):
        before, cp, _, _, _ = evaluate(t)
        grad = evaluate(t,True)
        accepted = False
        for backtrack in range(20):
            candidate = _softmax(xp.log(t)+(2.**(-backtrack))*total*grad)
            after = evaluate(candidate)[0]
            direction = _scalar((grad*(candidate-t)).sum())
            if xp.isfinite(after) and direction>0 and after>=before+1e-4*direction:
                t = candidate
                accepted = True
                break
        history.append(dict(epoch=iteration,cp=_scalar(evaluate(t)[1]),energy=_scalar(evaluate(t)[0]),accepted=accepted))
        if not accepted:
            break
    value, cp, centroid, r, mass = evaluate(t)
    return dict(T=t,G=centroid,R=r,C=mass,F=(mass/total).T,cp=_scalar(cp),history=history,energy=_scalar(value))


def train(learner, steps):
    net = learner.model_net
    anchors = 1/(1+np.exp(-net.matrices['G']))
    span = learner.RootMax-learner.RootMin
    den = span/2+np.abs(learner.Root-(learner.RootMin+span/2))
    omega = net.omega.copy()*(den>1e-8)
    state = dict(anchors=anchors.tolist(),omega=omega.tolist(),root=learner.Root.tolist(),
                 den=np.where(den>1e-8,den,1.).tolist(),steps=min(30,max(1,int(steps))))
    result = solve(learner.NormDataset,state,steps)
    state['counts'] = result['C'].tolist()
    state['sums'] = (result['T']@learner.NormDataset).tolist()
    learner.Groups,learner.Relevance = result['G'],result['R']
    learner.Transformation = result['T']
    hard = np.zeros_like(result['T']);hard[result['T'].argmax(0),np.arange(learner.n)] = 1
    learner.Transformation_hard = hard
    learner.Counts = hard.sum(1,keepdims=True)
    learner.Freqs,learner.cp,learner.history = result['F'],result['cp'],result['history']
    learner.k_active = int((learner.Counts>0).sum())
    learner.variational_state = state
    return learner.setModel()


def predict(model, row, cix):
    if not cix:
        weights = np.asarray(model.Freqs).reshape(-1).copy()
        weights /= weights.sum()
    else:
        original = model.routing_parameters
        state = dict(original)
        for key in ('anchors','omega','root','den'):
            state[key] = np.asarray(original[key])[:,cix]
        # A query is appended independently to frozen training summaries. No test
        # batch transduction, target access or mutation of the persisted model.
        result = solve(np.asarray(row).reshape(1,-1),state,original['steps'],
                       original['counts'],np.asarray(original['sums'])[:,cix])
        weights = result['T'][:,0]
    best = int(weights.argmax())
    return weights,float(weights[best]),best
