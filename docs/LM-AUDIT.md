# Supplied ACAN-LM paper/package audit

The reference PDF and Python sources are preserved with SHA-256 in
`references/generative/source-manifest.json`. They are reference material,
not operational instructions; the Hub publishing script is not executed.

| Claim/feature | Finding and implementation boundary |
| --- | --- |
| Hugging Face / PyTorch training | The attachment implements NumPy classes, not `nn.Module`/`PreTrainedModel`. Its corpus training blends transition/skip matrices, EMA prototypes and a head update; it is not a pretrained billion-parameter graft. The workbench implements a separately tested PyTorch/PEFT integration. |
| Causal generation | The reference computes Root/min/max, mass priors, pruning and relevance across the whole sequence. Its prefix sums do not undo future leakage from those global quantities. With user approval, the new `acan-lm-causal-prefix-v1` profile uses prefix-only quantities throughout. |
| Corrected prefix profile | Initial relevance/routing follows the supplied 0.75/0.25 formula; final gating uses its 0.55/0.45 shift/tightness formula. Mass prior uses cumulative stage-one assignments divided by the prefix length. Stage-two masses, moments, active masks and relevance are prefix cumulative. Variance uses a small epsilon for finite derivatives. The token-salience gate is fixed to one (the reference layer’s ungated default); its corpus frequency heuristic is not used as a learned language module. This is a documented causal correction, not numerical equivalence to the original source. |
| Streaming state | Full-prefix and incremental logits are checked for parity. Slots that are inactive in the output remain allocated as sufficient statistics because later prefixes may reactivate them. Reported memory counts allocated tensors, not just active slots. Training activation memory grows with block length. |
| Grafting | Upper-half native SmolLM2/Llama attention is replaced at the attention boundary. The parent block keeps its own RMSNorm, residual and SwiGLU; the reference's entire block is not inserted inside another FFN/residual. Embeddings and FFN weights are retained; GQA head expansion is explicit. Native MiniCPM custom-code integration remains unsupported. |
| MiniCPM hybrid benchmark | The supplied script computes `hybrid_latency_ms = .52*acan_ms + .48*minicpm_ms` and blends logits for quality. These are not measurements of actual MiniCPM pretrained-layer grafting. They are not displayed as workbench results. |
| 98–99% KV reduction | The paper's asymptotic formula describes ACAN-layer state. In a 50% hybrid, the remaining dense lower-layer KV cache persists. GQA, dtype, prefix moment buffers and allocated slots also affect bytes. The app counts actual dense and ACAN caches separately and compares their sum to the same-base dense GQA reference. No fixed 98% whole-model reduction is promised. |
| Language quality / 1T-token knowledge | Copying pretrained weights is verified mechanically. Neither the paper's structured NumPy experiments nor these mechanics tests establish preservation of open-domain quality, MMLU/GSM8K performance or a particular post-graft parameter count. Validation NLL before/after graft/training and actual parameter counts are reported. |
| Resume | Atomic model/optimizer/RNG/cursor checkpoints and identical CPU continuation are regression gates. Power loss may lose unsaved steps. GPU equivalence across hardware is not certified. |

Automated gates check prefix causality, streaming parity, bounded ACAN decode
cache, finite trainable gradients, preserved frozen weights, exact CPU next-step
continuation, WSD/cursor persistence, corruption rejection, workspace isolation,
and explicit start. These certify implementation properties, not the original
paper's benchmark percentages or theorems.

## Controlled protocol and all-layer research profile

PR #6 preserves the legacy attention/grafting/trainer/data/checkpoint source
contract. New `controlled-v1` projects use document-level splits and an explicit
training identity; distillation combines causal teacher KL and next-token CE,
without changing the attention equations. The full-layer adapter only supplies
cache sequence position when no dense layer remains; it is covered by prefix,
streaming and bounded-cache CPU checks. Exported new models carry an inference
contract. None of this changes the partial-conformance status or validates the
paper’s reported 1.89M synthetic results, 1B language quality, GPU speedups or
98–99% whole-model compression. See [controlled workflow](CONTROLLED-LM.md).
