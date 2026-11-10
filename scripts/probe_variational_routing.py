"""CPU mathematical probe, explicitly invoked; NOT an application routing profile.

Uses the existing tabular CP as an oracle. Optimizing free assignments here does
not implement an inductive predictor or a causal language model.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from acan.core.paper_routing import PaperRoutingTorch


def energy(t, scores, cp, mu=1., tau=1., gamma=.5, epsilon=1e-4):
    """Maximize J; t has shape slots x observations, columns on simplex."""
    f = t.mean(1)
    entropy = -(t * t.log()).sum(0).mean()
    primitive = ((f + epsilon) * (f + epsilon).log() - (f + epsilon)).sum()
    return mu * cp(t) + (t * scores).sum(0).mean() + tau * entropy + gamma * primitive


def mirror_step(t, objective, initial_step=64., max_backtracks=40):
    """Entropic proximal ascent with Armijo acceptance of the actual objective.

    A rejected/non-finite proposal leaves T unchanged. Numerical interior floor
    only bounds the probe; it is not part of the paper or an optimality proof.
    """
    source = t.detach().requires_grad_(True)
    before = objective(source)
    gradient, = torch.autograd.grad(before, source)
    if not torch.isfinite(before) or not torch.isfinite(gradient).all():
        raise ValueError('Non-finite objective/gradient at the current state')
    for backtracks in range(max_backtracks + 1):
        step = initial_step * .5 ** backtracks
        proposal = (source.log() + step * source.shape[1] * gradient).softmax(0)
        proposal = proposal.clamp_min(1e-14)
        proposal = (proposal / proposal.sum(0)).detach()
        directional = (gradient.detach() * (proposal - source.detach())).sum()
        after = objective(proposal).detach()
        if directional > 0 and torch.isfinite(after) and after >= before.detach() + 1e-4 * directional:
            return proposal, backtracks, True
    return t.detach().clone(), max_backtracks, False


def fixture(seed):
    generator = torch.Generator().manual_seed(seed)
    x = .1 + .8 * torch.rand(12, 4, generator=generator, dtype=torch.float64)
    net = PaperRoutingTorch(12, 4, k=3).double()
    net.set_root_statistics(x.mean(0), x.min(0).values, x.max(0).values)
    with torch.no_grad():
        net.omega.copy_(torch.tensor([[.7, .9, .5, 1.]], dtype=torch.float64))
        net.matrices['G'].copy_(torch.randn(3, 4, generator=generator, dtype=torch.float64))
    g = net.matrices['G'].detach().sigmoid()

    def cp(t):
        mass = t.sum(1, keepdim=True).clamp_min(1e-6)
        centroid = t @ x / mass
        relevance = net._compute_relevance(.5 * centroid + .5 * g)
        return net._compute_cluster_probability((mass / len(x)).t(), relevance)

    from acan.algebra import routing_logits
    scores = routing_logits(x[None], g[:, None], net._compute_relevance(g)[:, None]).detach()
    t = torch.randn(3, 12, generator=generator, dtype=torch.float64).softmax(0)
    return t, scores, cp, net, x


def run_probe():
    torch.set_num_threads(1)
    records = []
    for seed in range(20):
        t, scores, cp, net, x = fixture(seed)
        native = net(x)
        torch.testing.assert_close(cp(native['T']), native['cp'], atol=1e-12, rtol=1e-12)
        for mode in ('cp-only', 'joint-energy'):
            current = t.clone()
            objective = cp if mode == 'cp-only' else lambda a: energy(a, scores, cp)
            start = float(objective(current).detach())
            gains, accepted, backtracks = [], 0, 0
            for _ in range(30):
                before = float(objective(current).detach())
                current, reductions, ok = mirror_step(current, objective)
                after = float(objective(current).detach())
                assert after >= before, 'Accepted step decreased the actual objective'
                torch.testing.assert_close(current.sum(0), torch.ones(12, dtype=current.dtype))
                assert torch.isfinite(current).all() and (current > 0).all()
                gains.append(after-before)
                accepted += int(ok)
                backtracks += reductions
            records.append(dict(seed=seed, mode=mode, initial_objective=start,
                final_objective=float(objective(current).detach()), minimum_step_gain=min(gains),
                accepted_steps=accepted, backtracking_reductions=backtracks))
    epsilon = 1e-4
    stability = []
    for gamma in (.5, 1., 1.15):
        def mapping(p):
            f = torch.tensor([p, 1-p], dtype=torch.float64)
            return float((gamma * (f + epsilon).log()).softmax(0)[0])
        delta = 1e-6
        numeric = (mapping(.5+delta)-mapping(.5-delta))/(2*delta)
        analytic = gamma/(1+2*epsilon)
        assert abs(numeric-analytic) < 1e-8
        p = .51
        for _ in range(100):
            p = mapping(p)
        stability.append(dict(gamma=gamma, symmetric_gain_analytic=analytic,
            symmetric_gain_numeric=numeric, initial_mass=.51, mass_after_100_iterations=p))
    return dict(status='passed', scope='synthetic mathematical probe; no dataset/LM quality claim',
        torch=torch.__version__, dtype='float64', seeds=list(range(20)), steps_per_run=30,
        joint_coefficients=dict(mu=1., tau=1., gamma=.5, beta=8., epsilon=epsilon),
        coefficients_status='probe settings, not paper defaults or tuned application parameters',
        actual_objective_monotone=True, runs=records, duplicate_slot_stability=stability,
        note='100 repeated mass updates are a diagnostic dynamical system, not the current two-stage forward')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_probe()
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], runs=len(result['runs']),
        accepted_steps=sum(r['accepted_steps'] for r in result['runs']),
        duplicate_slot_stability=result['duplicate_slot_stability'])))


if __name__ == '__main__':
    main()
