import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'Torch required')
class ConceptAttentionTests(unittest.TestCase):
    def layers(self, slots, window, length=48):
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
        from acan.lm.concept_attention import ConceptSlotAttention
        torch.set_num_threads(1); torch.manual_seed(3)
        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256)
        dense = LlamaAttention(config, 0).eval()
        layer = ConceptSlotAttention(config, 0, slots=slots, window=window)
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            getattr(layer, name).weight.data.copy_(getattr(dense, name).weight.data)
        rotary = LlamaRotaryEmbedding(config)
        layer.rope_frequency = rotary.inv_freq
        x = torch.randn(2, length, 32)
        position = rotary(x, torch.arange(length)[None])
        causal = torch.full((length, length), float('-inf')).triu(1)[None, None]
        return dense, layer, x, position, causal

    def test_one_concept_per_token_or_a_full_window_is_dense_attention(self):
        import torch
        for slots, window in ((2, 64), (64, 4)):
            dense, layer, x, position, causal = self.layers(slots, window)
            with torch.no_grad():
                expected = dense(x, position_embeddings=position, attention_mask=causal)[0]
                torch.testing.assert_close(layer.attend(x, position), expected, atol=2e-5, rtol=2e-4)

    def test_generalized_prefix_is_causal_streams_and_has_finite_gradients(self):
        import torch
        dense, layer, x, position, causal = self.layers(3, 4)
        full = layer.attend(x, position)
        self.assertEqual(int(layer.last_info['active_slots'][0]), 3)
        self.assertAlmostEqual(float(layer.last_info['masses'][0].sum()), 28., places=4)   # tokens before 32-4
        with torch.no_grad():
            self.assertGreater(float((full-dense(x, position_embeddings=position, attention_mask=causal)[0]).abs().max()), 1e-3)
        changed = x.clone(); changed[:, 30:] += 5
        torch.testing.assert_close(full[:, :30], layer.attend(changed, position)[:, :30], atol=1e-6, rtol=1e-5)
        full.square().mean().backward()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            gradient = getattr(layer, name).weight.grad
            self.assertTrue(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)
        for sizes in ((1,)*48, (7, 20, 21), (33, 15)):
            layer.reset_stream(); pieces, start = [], 0
            with torch.no_grad():
                for size in sizes:
                    part = tuple(item[:, start:start+size] for item in position)
                    pieces.append(layer.attend(x[:, start:start+size], part, use_cache=True)); start += size
            torch.testing.assert_close(full.detach(), torch.cat(pieces, 1), atol=2e-5, rtol=2e-4)
        self.assertLess(layer.cache['keys'].shape[2], 4+2*layer.block)                 # bounded state
        from acan.lm.concept_attention import merge_costs
        state = layer.cache['state']
        torch.testing.assert_close(state['cost'], merge_costs(state, layer.kappa), atol=1e-4, rtol=1e-4)   # incremental = full

    def test_tabular_and_attention_share_the_same_operators(self):
        import torch
        from acan import concept_algebra as algebra
        from acan.core.concepts import ConceptMixture
        from acan.lm.concept_attention import merge_costs
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 5))
        t = np.eye(6)[rng.integers(0, 6, 40)]
        model = ConceptMixture()
        model.encoder = type('E', (), dict(categorical=[], width=0))()
        model.root_p = np.zeros(0)
        data = (x, np.ones_like(x), np.zeros((40, 0)))
        s = model._statistics(t, data)
        ones = (np.ones(5), np.ones(0))
        table = model._merge_costs(s, ones)
        root = np.ones(5)                                      # the tabular core standardizes: root variance one
        state = dict(C=torch.tensor(s['C'])[None, None], U=torch.tensor(s['U'])[None, None], Q=torch.tensor(s['Q'])[None, None],
                     root=torch.tensor(root)[None, None])
        attention = merge_costs(state, 1.)[0, 0].numpy()
        off = ~np.eye(6, dtype=bool)
        np.testing.assert_allclose(table[off], attention[off], rtol=1e-9, atol=1e-9)
        g, v = algebra.centre_variance(s['Cm'], s['U'], s['Q'], 1., 1., np)
        direct = algebra.exact_similarity(x[:, None], g[None], v[None], np)
        np.testing.assert_allclose(direct, model._log_density(s, data, ones), rtol=1e-9, atol=1e-9)

    def test_grafted_model_keeps_the_pretrained_function_inside_the_window(self):
        import tempfile
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from acan.lm.research_model import load_model
        torch.set_num_threads(1); torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)).eval()
            base.save_pretrained(directory, safe_serialization=True)
            tokens = torch.randint(0, 32, (1, 40))
            with torch.no_grad():
                expected = base(input_ids=tokens).logits
            for fraction in (.5, 1.):
                options = dict(architecture='hybrid', graft_fraction=fraction, sequence_length=40, slots=2, window=48,
                               lora_rank=2, routing_profile='cp-v9')
                model, report = load_model(directory, options)
                self.assertEqual(report['algebra'], 'cp-v9')
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(input_ids=tokens, use_cache=False).logits, expected, atol=5e-5, rtol=5e-4)
                options['window'] = 4
                model, _ = load_model(directory, options)
                result = model(input_ids=tokens, labels=tokens, use_cache=False)
                result.loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None))
                model.eval()
                with torch.no_grad():
                    full = model(input_ids=tokens, use_cache=False).logits
                    first = model(input_ids=tokens[:, :25], use_cache=True)
                    rest = model(input_ids=tokens[:, 25:], past_key_values=first.past_key_values, use_cache=True)
                    torch.testing.assert_close(full[:, 25:], rest.logits, atol=5e-5, rtol=5e-4)


if __name__ == '__main__':
    unittest.main()
