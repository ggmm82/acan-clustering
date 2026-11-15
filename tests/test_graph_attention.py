"""concept-graph-v2 inside a model: the same guarantees as the other concept layers."""
import importlib.util
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'), 'Torch required')
class GraphAttentionTests(unittest.TestCase):
    def layers(self, slots, window, length=48):
        import torch
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
        from acan.lm.graph_attention import ConceptGraphScanAttention
        torch.set_num_threads(1); torch.manual_seed(3)
        config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                             attn_implementation='eager')
        dense = LlamaAttention(config, 0).eval()
        layer = ConceptGraphScanAttention(config, 0, slots=slots, window=window).eval()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            getattr(layer, name).weight.data.copy_(getattr(dense, name).weight.data)
        rotary = LlamaRotaryEmbedding(config)
        layer.rope_frequency = rotary.inv_freq
        x = torch.randn(2, length, 32)
        position = rotary(x, torch.arange(length)[None])
        causal = torch.full((length, length), float('-inf')).triu(1)[None, None]
        return dense, layer, x, position, causal

    def test_enough_concepts_or_a_full_window_is_dense_attention(self):
        import torch
        for slots, window in ((2, 64), (64, 4)):
            dense, layer, x, position, causal = self.layers(slots, window)
            with torch.no_grad():
                expected = dense(x, position_embeddings=position, attention_mask=causal)[0]
                torch.testing.assert_close(layer.attend(x, position), expected, atol=2e-5, rtol=2e-4)

    def test_generalized_prefix_is_causal_streams_and_has_finite_gradients(self):
        import torch
        dense, layer, x, position, causal = self.layers(4, 4)
        full = layer.attend(x, position)
        info = layer.last_info
        self.assertAlmostEqual(float(info['masses'][0].sum()), 44., places=4)      # every token before the window, once
        self.assertLessEqual(int(info['active_slots'][0]), 4*6)                    # at most ``slots`` per level
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
            torch.testing.assert_close(torch.cat(pieces, 1), full.detach(), atol=2e-5, rtol=2e-4)
            self.assertGreater(layer.cache_bytes(), 0)

    def test_memory_does_not_grow_like_the_context(self):
        import torch
        _, layer, x, position, _ = self.layers(4, 4, length=200)
        sizes = []
        with torch.no_grad():
            for length in (50, 100, 200):
                layer.reset_stream()
                layer.attend(x[:, :length], tuple(item[:, :length] for item in position), use_cache=True)
                sizes.append(layer.cache['stream'].kept_vectors())
        self.assertLessEqual(sizes[-1], 4+4*8)
        self.assertLess(sizes[-1], 2*sizes[0])

    def test_profile_in_a_model_dense_in_the_limit_trainable_and_continues_a_prompt(self):
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
                               lora_rank=2, routing_profile='concept-graph-v2')
                model, report = load_model(directory, options)
                self.assertEqual(report['algebra'], 'concept-graph-v2')
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(input_ids=tokens, use_cache=False).logits, expected, atol=5e-5, rtol=5e-4)
                options.update(window=4, slots=4)
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

    def test_earlier_projects_keep_their_receipts_and_the_new_profile_does_not_borrow_them(self):
        from acan.lm.compatibility import PRE_SCAN_INFERENCE
        from acan.lm.research_model import check_inference_contract
        check_inference_contract(dict(routing_profile='concept-graph-v1', architecture='native', graft_fraction=1.),
                                 dict(model_contract=PRE_SCAN_INFERENCE))
        with self.assertRaises(ValueError):
            check_inference_contract(dict(routing_profile='concept-graph-v2', architecture='native', graft_fraction=1.),
                                     dict(model_contract=PRE_SCAN_INFERENCE))


if __name__ == '__main__':
    unittest.main()
