import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'Torch required')
class ConceptAttentionTests(unittest.TestCase):
    graph = False
    profile = 'cp-v9'

    def layers(self, slots, window, length=48):
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
        from acan.lm.concept_attention import ConceptGraphAttention, ConceptSlotAttention
        ConceptSlotAttention = ConceptGraphAttention if self.graph else ConceptSlotAttention
        torch.set_num_threads(1); torch.manual_seed(3)
        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                             attn_implementation='eager')
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

    def test_tabular_and_attention_share_the_same_criterion(self):
        import torch
        from acan.concept_algebra import reader_cost
        from acan.core.concepts import ConceptModel
        from acan.lm.concept_attention import reader_costs
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 5))
        t = np.eye(6)[rng.integers(0, 6, 40)]
        model = ConceptModel()
        model.labels = []
        model.encoder = type('E', (), dict(numeric=list('abcde'), categorical=[], width=0))()
        model.root_p = np.zeros(0)
        s = model._statistics(t, (x, np.ones_like(x), np.zeros((40, 0))))
        first, second = np.triu_indices(6, 1)
        table = model._pair_costs(s, first, second)            # reads received by a concept of a table: its mass
        tensor = lambda value: torch.tensor(value)[None, None]
        state = dict(C=tensor(s['C']), U=tensor(s['U']), Q=tensor(s['Q']), P=tensor(np.zeros(6)),
                     A=tensor(s['C']), root=torch.ones(1, 1, 5, dtype=torch.float64))
        attention = reader_costs(state, 0.)[0, 0].numpy()      # rate = A/(now-position+1) = mass
        floor = s['C'].mean()*1e-3+1e-12
        radius = model._radius(s['Cm'], s['U'], s['Q'], s['N'])
        together = model._radius(s['Cm'][first]+s['Cm'][second], s['U'][first]+s['U'][second],
                                 s['Q'][first]+s['Q'][second], s['N'][first]+s['N'][second])
        expected = reader_cost(s['C'][first]+floor, radius[first], s['C'][second]+floor, radius[second], together)
        np.testing.assert_allclose(attention[first, second], expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(table, expected, rtol=5e-3)   # identical but for the floor on the reads
        self.assertTrue((table > -1e-12).all())                  # generalizing never helps the reader for free
        copies = {key: np.concatenate([value/2, value/2]) for key, value in s.items()}
        self.assertTrue(np.allclose(model._pair_costs(copies, np.arange(6), np.arange(6)+6), 0, atol=1e-9))

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
                               lora_rank=2, routing_profile=self.profile)
                model, report = load_model(directory, options)
                self.assertEqual(report['algebra'], self.profile)
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


class ConceptGraphAttentionTests(ConceptAttentionTests):
    """The same guarantees with generalization guided by the reader."""
    graph = True
    profile = 'concept-graph-v1'

    def test_tokens_the_queries_keep_reading_stay_exact(self):
        import torch
        from acan.lm.concept_attention import reader_costs
        torch.manual_seed(0)
        keys = torch.randn(1, 1, 6, 4)
        state = dict(C=torch.ones(1, 1, 6), U=keys, Q=keys*keys, P=torch.arange(6.)[None, None],
                     A=torch.tensor([[[50., 0., 0., 0., 0., 0.]]]), root=torch.ones(1, 1, 4))
        cost = reader_costs(state, 10.)
        cheapest = divmod(int(cost.flatten().argmin()), 6)
        self.assertNotIn(0, cheapest)                              # the heavily read token is not the one generalized
        quiet = dict(state, A=torch.zeros(1, 1, 6))
        even = reader_costs(quiet, 10.)
        self.assertLess(float((cost[0, 0, 0, 1:]/even[0, 0, 0, 1:]).min()), float('inf'))
        self.assertGreater(float(cost[0, 0, 0, 1:].min()), float(cost[0, 0, 1:, 1:][torch.isfinite(cost[0, 0, 1:, 1:])].min()))


if __name__ == '__main__':
    unittest.main()
