"""Verification of the single learning rule uniting tabular EM and generative concept networks."""
import importlib.util
import unittest

import numpy as np

from acan.concept_algebra import centre_variance, natural_em_dict, natural_em_parameters, natural_em_step

TORCH = importlib.util.find_spec('torch') is not None


class UnifiedLearningAlgebraTest(unittest.TestCase):
    def test_step_one_equals_exact_tabular_batch_em_and_chunked_sums(self):
        rng = np.random.default_rng(11)
        n, k, d = 120, 4, 5
        x = rng.normal(size=(n, d))
        raw_t = rng.dirichlet(np.ones(k), size=n)
        batch_c = raw_t.sum(0)
        batch_u = raw_t.T @ x
        batch_q = raw_t.T @ (x * x)
        root = np.ones((1, d))
        kappa = 1.5

        log_c = rng.normal(size=k)
        g = rng.normal(size=(k, d))
        log_s = rng.normal(scale=.2, size=(k, d))

        new_log_c, new_g, new_log_s = natural_em_parameters(
            log_c, g, log_s, batch_c, batch_u, batch_q, root, kappa, step=1., xp=np
        )
        expected_g, expected_v = centre_variance(batch_c[:, None], batch_u, batch_q, root, kappa, np)
        expected_pi = batch_c / batch_c.sum()
        np.testing.assert_allclose(np.exp(new_log_c), expected_pi, atol=1e-12)
        np.testing.assert_allclose(new_g, expected_g, atol=1e-12)
        np.testing.assert_allclose(np.exp(2 * new_log_s), expected_v, atol=1e-12)

        # Streaming online EM with step = n_b / N_so_far equals the exact full-table additive statistics.
        running = None
        seen = 0
        for start in (0, 27, 73):
            stop = {0: 27, 27: 73, 73: n}[start]
            part_t, part_x = raw_t[start:stop], x[start:stop]
            part = dict(
                C=part_t.sum(0) / len(part_x),
                U=(part_t.T @ part_x) / len(part_x),
                Q=(part_t.T @ (part_x * part_x)) / len(part_x),
            )
            seen += len(part_x)
            running = natural_em_dict(running, part, step=len(part_x) / seen)
        np.testing.assert_allclose(running['C'] * n, batch_c, atol=1e-12)
        np.testing.assert_allclose(running['U'] * n, batch_u, atol=1e-12)
        np.testing.assert_allclose(running['Q'] * n, batch_q, atol=1e-12)

    @unittest.skipUnless(TORCH, 'torch required to verify autograd Fisher natural gradient identity')
    def test_infinitesimal_step_equals_complete_data_fisher_natural_gradient(self):
        import torch

        torch.manual_seed(7)
        n, k, d = 64, 3, 4
        z = torch.randn(n, d, dtype=torch.float64)
        log_c = torch.tensor([0.3, -0.4, 0.1], dtype=torch.float64, requires_grad=True)
        g = (torch.randn(k, d, dtype=torch.float64) * .6).detach().requires_grad_(True)
        log_s = (torch.randn(k, d, dtype=torch.float64) * .25).detach().requires_grad_(True)

        pi = torch.softmax(log_c, dim=-1)
        var = torch.exp(2 * log_s)
        log_det = 2 * log_s.sum(-1)
        quad = ((z[:, None, :] - g[None, :, :]) ** 2 / var[None, :, :]).sum(-1)
        log_comp = torch.log(pi)[None, :] - .5 * (quad + log_det[None, :] + d * np.log(2 * np.pi))
        ll = torch.logsumexp(log_comp, dim=-1).mean()
        ll.backward()

        with torch.no_grad():
            t = torch.softmax(log_comp, dim=-1)
            bc = t.sum(0)
            bu = t.T @ z
            bq = t.T @ (z * z)
            eps = 1e-7
            nlc, ng, nls = natural_em_parameters(
                log_c, g, log_s, bc, bu, bq, root=0., kappa=0., step=eps, xp=torch
            )
            em_vel_pi = (torch.exp(nlc) - pi) / eps
            em_vel_g = (ng - g) / eps
            em_vel_ls = (nls - log_s) / eps

            # Analytical inverse complete-data Fisher metric applied to Euclidean autograd gradients:
            fisher_vel_pi = log_c.grad                                 # dL / d(log c_k) = pi_hat_k - pi_k
            fisher_vel_g = (var / pi[:, None]) * g.grad                # F_g^{-1} = sigma_k^2 / pi_k
            fisher_vel_ls = (1. / (2. * pi[:, None])) * log_s.grad     # F_{log s}^{-1} = 1 / (2 pi_k)

            torch.testing.assert_close(em_vel_pi, fisher_vel_pi, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(em_vel_g, fisher_vel_g, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(em_vel_ls, fisher_vel_ls, atol=1e-6, rtol=1e-5)


@unittest.skipUnless(TORCH, 'torch required to test grouped readout and network growth dynamics')
class UnifiedNetworkMechanicsTest(unittest.TestCase):
    def test_grouped_token_log_prob_matches_dense_reference_values_and_gradients(self):
        import copy
        import torch
        from acan.lm.concept_net_training import Network, _GroupedTokenLogProb, _pad, frequency_groups

        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        v = 120
        freq = 1. / (np.arange(1, v + 1) ** .9)
        groups = frequency_groups(freq)
        net = Network(dict(vocabulary=v, dim=16, layers=1, heads=2, concepts=4, window=2, grouped=True),
                      frequency=freq / freq.sum(), groups=groups)
        ref = copy.deepcopy(net)
        docs = [rng.choice(v, size=25, p=freq / freq.sum()) for _ in range(5)]
        ids, mask = _pad(docs)
        x_in, m_in, targets = ids[:, :-1], mask[:, :-1], ids[:, 1:]

        hf = net.states(x_in, m_in)
        flat_f, tgt_f = hf.reshape(-1, hf.shape[-1]), targets.reshape(-1)
        g_f = net.group[tgt_f]
        lp_gf = torch.log_softmax(net._logits(flat_f, net.group_centre, net.group_log_mass), -1).gather(1, g_f[:, None])[:, 0]
        in_f = _GroupedTokenLogProb.apply(flat_f, net._rank[tgt_f], g_f, net.centre.weight, net.log_mass,
                                          net.log_scale, net._members, net._valid)
        lp_fast = net._cache(hf, m_in, targets, (lp_gf + in_f).view(targets.shape))
        (-lp_fast.mean()).backward()

        h = ref.states(x_in, m_in)
        flat, tgt = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
        g = ref.group[tgt]
        lp_g = torch.log_softmax(ref._logits(flat, ref.group_centre, ref.group_log_mass), -1).gather(1, g[:, None])[:, 0]
        members, valid = ref._members[g], ref._valid[g]
        matching = torch.bmm(ref.centre.weight[members], flat.unsqueeze(-1)).squeeze(-1)
        norms = ref.centre.weight.square().sum(-1)[members]
        logits = (matching - .5 * norms) * torch.exp(-2 * ref.log_scale) + ref.log_mass[members]
        inside = torch.log_softmax(logits.masked_fill(~valid, float('-inf')), -1).gather(1, ref._rank[tgt][:, None])[:, 0]
        lp_ref = ref._cache(h, m_in, targets, (lp_g + inside).view(targets.shape))
        (-lp_ref.mean()).backward()

        torch.testing.assert_close(lp_fast, lp_ref, atol=1e-5, rtol=1e-5)
        for (name, p_fast), (_, p_ref) in zip(net.named_parameters(), ref.named_parameters()):
            if p_ref.grad is not None:
                torch.testing.assert_close(p_fast.grad, p_ref.grad, atol=1e-6, rtol=1e-5, msg=name)

    def test_added_third_layer_preserves_reading_yet_receives_nonzero_opening_signal(self):
        import torch
        from acan.lm.concept_net_training import Learner, Network

        torch.manual_seed(5)
        net = Network(dict(vocabulary=40, dim=16, layers=2, heads=2, concepts=3, window=2, grouped=True),
                      groups=np.arange(40) // 6)
        ids = np.tile(np.r_[np.arange(1, 15), 39], 4)
        Learner(net, tokens=32).learn(ids, eos=39)
        before = net.export().document(ids[:14])
        net.add_layer()
        np.testing.assert_allclose(net.export().document(ids[:14]), before, atol=1e-5)
        Learner(net, tokens=32).learn(ids, eos=39)
        self.assertNotAlmostEqual(float(net.layers[-1].gate.item()), 0., places=5)

    def test_corpus_bundles_preserve_exact_token_counts_and_mix_domains(self):
        from acan.lm.concept_trainer import corpus_bundles, piece_from_bundles

        eos = 99
        # Four distinct blocks of documents concatenated sequentially (domains 0, 1, 2, 3).
        stream = np.concatenate([
            np.tile(np.r_[np.full(30, d * 10 + 1, dtype=np.int64), eos], 20)
            for d in range(4)
        ])
        read = len(stream)
        bundles, offsets = corpus_bundles(stream, read, eos, seed=42, bundle_tokens=64)
        piece = 620
        steps = int(np.ceil(read / piece))
        pieces = [piece_from_bundles(stream, bundles, offsets, s * piece, min((s + 1) * piece, read))
                  for s in range(steps)]
        self.assertEqual(sum(len(p) for p in pieces), read)
        np.testing.assert_array_equal(
            np.bincount(np.concatenate(pieces), minlength=100),
            np.bincount(stream, minlength=100),
        )
        # First piece now samples multiple domains instead of only domain 0.
        distinct_domains = {int(v) // 10 for v in np.unique(pieces[0]) if v != eos}
        self.assertGreaterEqual(len(distinct_domains), 3)

    def test_generative_net_text_solves_tabular_xor_with_irrelevant_columns(self):
        import pandas as pd
        import torch
        from acan.lm.net_text import NetText

        def make_rows(n, seed):
            rng = np.random.default_rng(seed)
            w, p = rng.integers(0, 2, n), rng.integers(0, 2, n)
            return pd.DataFrame(dict(
                w=w, p=p,
                noise1=rng.integers(0, 8, n),
                noise2=rng.integers(0, 8, n),
                noise3=rng.integers(0, 8, n),
                play=(w == p).astype(int),
            ))

        train_df, val_df, test_df = make_rows(180, 1), make_rows(40, 2), make_rows(80, 3)
        # Disjoint token offsets per column: w(0..1), p(2..3), n1(4..11), n2(12..19), n3(20..27), play(28..29), eos=30
        offsets = [0, 2, 4, 12, 20, 28]
        eos, vocab = 30, 31

        def encode(df, include_target=True):
            arr = df.to_numpy(dtype=np.int64)
            cols = 6 if include_target else 5
            shifted = arr[:, :cols] + np.array(offsets[:cols], dtype=np.int64)
            if not include_target:
                return shifted
            with_eos = np.column_stack([shifted, np.full(len(df), eos, dtype=np.int64)])
            return with_eos.ravel()

        rng = np.random.default_rng(7)
        train_stream = np.concatenate([encode(train_df.iloc[rng.permutation(len(train_df))]) for _ in range(8)])
        val_stream = encode(val_df)
        weight = np.full(vocab, .05, dtype=np.float32)
        weight[28:30] = 5.
        weight[eos] = 0.

        torch.manual_seed(0)
        net_cfg = dict(dim=32, layers=2, heads=2, concepts=8, window=3, grouped=False, cache=0, passes=False)
        model = NetText(vocab, eos, budget=1 << 22, max_context=12, network=net_cfg, growth=False, weight=weight)
        model.decide_on = val_stream
        model.add(train_stream, train_stream, [val_stream], lambda: None)

        feats = encode(test_df, include_target=False)
        preds = [int(np.argmax(model.distribution(row)[28:30])) for row in feats]
        acc = float((np.array(preds) == test_df['play'].to_numpy()).mean())
        self.assertGreaterEqual(acc, .95)


class GradualTabularEMTest(unittest.TestCase):
    def test_gradual_online_em_rate_solves_tabular_xor_interaction(self):
        import pandas as pd
        from acan.core.concepts import ConceptModel

        class GradualModel(ConceptModel):
            def _em(self, s, data, weights, steps, rate=.5):
                return super()._em(s, data, weights, max(steps * 2, 8), rate=rate)

        def game(rows, seed):
            rng = np.random.default_rng(seed)
            w, p = rng.integers(0, 2, rows), rng.integers(0, 2, rows)
            return pd.DataFrame(dict(
                weather=np.where(w, 'rain', 'sun'),
                place=np.where(p, 'hall', 'field'),
                day=rng.integers(1, 29, rows),
                month=rng.integers(1, 13, rows),
                play=np.where(w == p, 'yes', 'no'),
            ))

        train, test = game(160, 10), game(80, 11)
        exact = ConceptModel(seed=0).fit(train, ['play'])
        gradual = GradualModel(seed=0).fit(train, ['play'])
        self.assertEqual(float((exact.predict(test, 'play') == test.play).mean()), 1.)
        self.assertEqual(float((gradual.predict(test, 'play') == test.play).mean()), 1.)

    @unittest.skipUnless(TORCH, 'PyTorch required')
    def test_growth_candidates_start_at_reference_nll_and_non_race_steps_stay_unanchored(self):
        import copy
        import torch
        from acan.lm.concept_net_training import Learner, Network, documents, held_out
        from acan.lm.net_text import NetText

        torch.manual_seed(2)
        rng = np.random.default_rng(2)
        eos = 40
        docs_raw = [np.r_[rng.integers(1, 20 if i % 2 == 0 else 40, size=18), eos] for i in range(60)]
        stream = np.concatenate(docs_raw)
        net = Network(dict(vocabulary=41, dim=16, layers=1, heads=1, concepts=1, window=2, grouped=False))
        Learner(net, tokens=64).learn(stream[:600], eos=eos)
        held = documents(stream[600:], eos, 64, begin=True)
        ref = float(-held_out(net, held).mean())
        sel = net.concepts_to_split(0, held)
        split_net = copy.deepcopy(net); split_net.split_one(0, sel)
        layer_net = copy.deepcopy(net); layer_net.add_layer()
        for name, cand in (('larghezza', net.widened()), ('teste', net.more_heads()),
                           ('strato', layer_net), ('concetti', split_net)):
            start_nll = float(-held_out(cand, held).mean())
            self.assertAlmostEqual(start_nll, ref, delta=0.015, msg=name)

        model = NetText(41, eos, 1 << 22, max_context=64, growth=True,
                        network=dict(dim=8, layers=1, heads=1, concepts=1, window=1, grouped=False, passes=False))
        model.add(stream[:300], stream, [stream[900:]], lambda: None)
        model.add(stream[300:600], stream, [stream[900:]], lambda: None)
        self.assertIsNotNone(model._learner)
        self.assertIsNone(model._learner.learner.anchor)
        self.assertFalse(model._learner.done)

    def test_eos_less_stream_is_split_into_bounded_bundles(self):
        from acan.lm.concept_trainer import corpus_bundles, piece_from_bundles

        stream = np.arange(1, 40_001, dtype=np.int64) % 97
        bundles, offsets = corpus_bundles(stream, len(stream), eos=99, seed=7, bundle_tokens=1024)
        self.assertGreaterEqual(len(bundles), 9)
        self.assertTrue(all(b - a <= 4096 for a, b in bundles))
        rebuilt = piece_from_bundles(stream, bundles, offsets, 0, len(stream))
        self.assertEqual(len(rebuilt), len(stream))
        self.assertEqual(sorted(rebuilt.tolist()), sorted(stream.tolist()))


if __name__ == '__main__':
    unittest.main()
