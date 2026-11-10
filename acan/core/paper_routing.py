"""Versioned paper routing; legacy initialization, relevance and CP remain explicit.

This implements the paper's squared metric and two-stage log-mass competition.
It does not invent Phi feature selection, entropy coefficients or masked losses
which the supplied tabular paper does not specify sufficiently for this API.
"""
import numpy as np
from .network import _AnalyticalClusterNet, _sigmoid_np, _softmax_axis0_np, HAS_TORCH
from acan.algebra import routing_logits, mass_modulation

PROFILE = 'paper-routing-v1'


class PaperRoutingNumpy(_AnalyticalClusterNet):
    def __init__(self, *args, relevance_alpha=2., mass_gamma=1.15, beta=8., **kwargs):
        # alpha in the legacy API is a mixture coefficient, not paper alpha.
        kwargs['alpha'] = 1.
        super().__init__(*args, **kwargs)
        self.relevance_alpha, self.mass_gamma, self.beta = relevance_alpha, mass_gamma, beta

    def forward(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.n = n = len(X)
        G = _sigmoid_np(self.matrices['G'])
        relevance, _, _ = self._compute_relevance(G)
        logits = routing_logits(X[None], G[:,None], relevance[:,None], self.relevance_alpha, self.beta)
        first = _softmax_axis0_np(logits)
        mass1 = first.mean(axis=1, keepdims=True)
        T = _softmax_axis0_np(mass_modulation(logits, mass1, self.mass_gamma, np.log))
        hard = np.zeros_like(T)
        hard[np.argmax(T, axis=0), np.arange(n)] = 1.
        mass = np.clip(T.sum(axis=1, keepdims=True), 1e-6, None)
        frequency = mass / float(n)
        posterior = T @ X / mass
        effective = .5*posterior + .5*G
        R, den, valid = self._compute_relevance(effective)
        real_R = np.clip(R.sum(axis=1, keepdims=True), 1e-6, None)
        denom = np.clip(real_R / self.m, 1e-6, None)
        cp = float(np.sqrt(max(float(np.sum(frequency*(R*R).sum(axis=1,keepdims=True)/denom)),1e-12)))
        self._cache = dict(X=X,T=T,G_sig=G,R=R,F_post=frequency.T,G_post=posterior,G_eff=effective,
                           C_soft=mass,real_R=real_R,g_pr_sq=(R*R).sum(axis=1,keepdims=True),
                           safe_den=den,valid_den=valid,cp=cp,first=first,mass1=mass1,relevance=relevance)
        return dict(T=T,T_hard=hard,C=hard.sum(axis=1,keepdims=True),F=frequency.T,G=posterior,R=R,cp=cp,neg_cp=-cp)

    def backward(self):
        c = self._cache
        X,T,G,R,F = c['X'],c['T'],c['G_sig'],c['R'],c['F_post'].T
        n = float(len(X))
        raw_R = R.sum(axis=1,keepdims=True)
        denom = np.clip(c['real_R']/self.m,1e-6,None)
        objective = float(np.sum(F*c['g_pr_sq']/denom))
        scale = -.5/c['cp'] if objective>1e-12 else 0.
        dR = scale*F*(2*R/denom-c['g_pr_sq']/denom**2*(raw_R>self.m*1e-6)/self.m)
        dF = scale*c['g_pr_sq']/denom
        d_eff = dR*np.sign(c['G_eff']-self.Root)/c['safe_den']*c['valid_den']*self.omega
        d_post,dG = .5*d_eff,.5*d_eff
        dC = dF/n-(d_post*c['G_post']).sum(axis=1,keepdims=True)/c['C_soft']
        dT = (d_post/c['C_soft'])@X.T+dC*(T.sum(axis=1,keepdims=True)>1e-6)
        dz = T*(dT-(T*dT).sum(axis=0,keepdims=True))
        first = c['first']
        d_first = self.mass_gamma*dz.sum(axis=1,keepdims=True)/(c['mass1']+1e-4)/n
        dz += first*(d_first-(first*d_first).sum(axis=0,keepdims=True))
        dd = -self.beta*dz
        latent = c['relevance']
        H = latent**self.relevance_alpha
        hmean = H.mean(axis=1,keepdims=True)+1e-6
        W = H/hmean
        summed,dx,dx2 = dd.sum(axis=1,keepdims=True),dd@X,dd@(X**2)
        dG += 2*W*(G*summed-dx)/self.m
        dW = (G**2*summed-2*G*dx+dx2)/self.m
        dH = (dW-(dW*W).sum(axis=1,keepdims=True)/self.m)/hmean
        d_lat = dH*self.relevance_alpha*np.maximum(latent,1e-30)**(self.relevance_alpha-1)*(latent>0)
        _,den,valid = self._compute_relevance(G)
        dG += d_lat*np.sign(G-self.Root)/den*valid*self.omega
        self.grads['G'] = dG*G*(1-G)*(np.abs(self.matrices['G'])<30)
        # Eq. v8_routing obtains its mass from assignments, not a learned F prior.
        self.grads['F'] = np.zeros_like(self.matrices['F'])
        return self.grads


if HAS_TORCH:
    import torch
    from .network import ModularClusterNet

    class PaperRoutingTorch(ModularClusterNet):
        def __init__(self,*args,relevance_alpha=2.,mass_gamma=1.15,beta=8.,**kwargs):
            kwargs['alpha']=1.
            super().__init__(*args,**kwargs)
            self.relevance_alpha,self.mass_gamma,self.beta=relevance_alpha,mass_gamma,beta

        def forward(self,X):
            X=torch.as_tensor(X,dtype=self.matrices['G'].dtype,device=self.matrices['G'].device)
            G=torch.sigmoid(self.matrices['G'].clamp(-30,30))
            relevance=self._compute_relevance(G)
            logits=routing_logits(X[None],G[:,None],relevance[:,None],self.relevance_alpha,self.beta)
            first=logits.softmax(0)
            T=mass_modulation(logits,first.mean(1,keepdim=True),self.mass_gamma,torch.log).softmax(0)
            # Keep the legacy unused F parameter connected with exactly zero gradient.
            T=T+self.matrices['F'].sum()*0
            hard=torch.nn.functional.one_hot(T.argmax(0),self.k).t().to(T.dtype)
            mass=T.sum(1,keepdim=True).clamp_min(1e-6)
            F=(mass/len(X)).t()
            posterior=T@X/mass
            R=self._compute_relevance(.5*posterior+.5*G)
            cp=self._compute_cluster_probability(F,R)
            return dict(T=T,T_hard=hard,C=hard.sum(1,keepdim=True),F=F,G=posterior,R=R,cp=cp,neg_cp=-cp)
