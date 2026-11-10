"""Mathematical proposal checks; no application profile is activated."""
import importlib.util
from pathlib import Path
import sys
import unittest

# Also discoverable when launched through scripts/check_paper.py outside cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch needed for research probe')
class VariationalRoutingTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_fixed_prior_softmax_is_unique_entropy_regularized_optimum(self):
        import torch
        from scripts.probe_variational_routing import fixture
        t, scores, _, _, _ = fixture(3)
        tau, gamma = .7, 1.15
        prior = torch.tensor([.2, .3, .5], dtype=t.dtype)
        logits = scores + gamma * (prior[:, None]+1e-4).log()
        optimal = (logits/tau).softmax(0)
        def objective(a):
            return (a*logits-tau*a*a.log()).sum(0).mean()
        gap = objective(optimal)-objective(t)
        kl = (t*(t.log()-optimal.log())).sum(0).mean()
        torch.testing.assert_close(gap, tau*kl, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(gap), 0)

    def test_full_objective_gradient_on_simplex_matches_finite_difference(self):
        import torch
        from scripts.probe_variational_routing import fixture, energy
        t, scores, cp, _, _ = fixture(8)
        t.requires_grad_(True)
        gradient, = torch.autograd.grad(energy(t, scores, cp), t)
        for j in range(t.shape[1]):
            delta = torch.zeros_like(t)
            delta[0,j], delta[1,j] = 1e-6, -1e-6
            numerical = (energy(t+delta, scores, cp)-energy(t-delta, scores, cp))/2e-6
            torch.testing.assert_close(numerical, gradient[0,j]-gradient[1,j], atol=2e-7, rtol=2e-6)

    def test_cp_is_the_existing_objective_and_step_respects_slot_permutation(self):
        import torch
        from scripts.probe_variational_routing import fixture, energy, mirror_step
        t, scores, cp, net, x = fixture(2)
        native = net(x)
        torch.testing.assert_close(cp(native['T']), native['cp'], atol=1e-12, rtol=1e-12)
        permutation = torch.tensor([2,0,1]); inverse = permutation.argsort()
        a, _, _ = mirror_step(t, lambda v: energy(v, scores, cp))
        b, _, _ = mirror_step(t[permutation], lambda v: energy(v, scores[permutation], lambda q: cp(q[inverse])))
        torch.testing.assert_close(b, a[permutation], atol=1e-12, rtol=1e-12)

    def test_research_probe_checks_actual_ascent_and_mass_instability(self):
        from scripts.probe_variational_routing import run_probe
        report = run_probe()
        self.assertEqual(len(report['runs']), 40)
        self.assertTrue(all(r['final_objective'] > r['initial_objective'] for r in report['runs']))
        stable, boundary, unstable = report['duplicate_slot_stability']
        self.assertLess(abs(stable['mass_after_100_iterations']-.5), 1e-10)
        self.assertLess(boundary['symmetric_gain_analytic'], 1)
        self.assertGreater(unstable['symmetric_gain_analytic'], 1)
        self.assertGreater(unstable['mass_after_100_iterations'], .99)


if __name__ == '__main__':
    unittest.main()
