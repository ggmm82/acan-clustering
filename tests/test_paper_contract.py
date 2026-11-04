import unittest
import numpy as np

from acan.core.network import _AnalyticalClusterNet, HAS_TORCH


def fixture(alpha=.5):
    rng = np.random.default_rng(7)
    x = rng.uniform(.15, .85, (8, 4))
    net = _AnalyticalClusterNet(8, 4, k=3, alpha=alpha)
    net.set_root_statistics(x.mean(0), x.min(0), x.max(0))
    net.omega = np.array([[.7, .9, .5, 1.]])
    return net, x


class TestPaperContracts(unittest.TestCase):
    def test_numpy_gradients_match_returned_objective(self):
        for alpha in (0., .5, 1.):
            with self.subTest(alpha=alpha):
                net, x = fixture(alpha)
                net.forward(x)
                gradients = {key: value.copy() for key, value in net.backward().items()}
                for key in gradients:
                    for index in np.ndindex(net.matrices[key].shape):
                        original, eps = net.matrices[key][index], 1e-6
                        net.matrices[key][index] = original + eps
                        plus = net.forward(x)['neg_cp']
                        net.matrices[key][index] = original - eps
                        minus = net.forward(x)['neg_cp']
                        net.matrices[key][index] = original
                        self.assertAlmostEqual(gradients[key][index], (plus - minus) / (2 * eps), delta=2e-6)

    def test_attention_centroids_frequencies_and_cp_formula(self):
        net, x = fixture()
        result = net(x)
        np.testing.assert_allclose(result['T'].sum(0), 1, atol=1e-12)
        np.testing.assert_allclose(result['G'], result['T'] @ x / result['T'].sum(1, keepdims=True))
        np.testing.assert_allclose(result['F'].sum(), 1, atol=1e-12)
        relevance = result['R']
        expected = np.sqrt(np.sum(result['F'].T * (relevance**2).sum(1, keepdims=True) /
                                  np.clip(relevance.sum(1, keepdims=True) / x.shape[1], 1e-6, None)))
        self.assertAlmostEqual(result['cp'], expected, places=12)
        self.assertEqual(result['neg_cp'], -result['cp'])
        np.testing.assert_allclose(result['C'].sum(), len(x))

    def test_constant_columns_and_saturated_parameters(self):
        x = np.full((8, 4), .4)
        net = _AnalyticalClusterNet(8, 4, k=3)
        net.set_root_statistics(x.mean(0), x.min(0), x.max(0))
        self.assertTrue(np.isfinite(net(x)['cp']))
        for gradient in net.backward().values():
            np.testing.assert_allclose(gradient, 0, atol=1e-12)
        net, x = fixture()
        net.matrices['G'][0, 0] = 40
        net(x)
        self.assertEqual(net.backward()['G'][0, 0], 0)

    @unittest.skipUnless(HAS_TORCH, 'Optional torch backend; required by the dedicated audit CI job')
    def test_torch_and_numpy_forward_and_gradient_parity(self):
        import torch
        from acan.core.network import ModularClusterNet
        for alpha in (0., .5, 1.):
            net, x = fixture(alpha)
            torch_net = ModularClusterNet(8, 4, k=3, alpha=alpha).double()
            torch_net.set_root_statistics(x.mean(0), x.min(0), x.max(0))
            with torch.no_grad():
                torch_net.omega.copy_(torch.tensor(net.omega))
                for key, value in net.matrices.items():
                    torch_net.matrices[key].copy_(torch.tensor(value))
            a, b = net(x), torch_net(torch.tensor(x))
            for key in ('T', 'G', 'R', 'F', 'cp', 'neg_cp'):
                np.testing.assert_allclose(a[key], b[key].detach().numpy(), atol=2e-10, rtol=2e-9)
            b['neg_cp'].backward()
            gradients = net.backward()
            for key, value in gradients.items():
                np.testing.assert_allclose(value, torch_net.matrices[key].grad.numpy(), atol=2e-9, rtol=2e-8)
