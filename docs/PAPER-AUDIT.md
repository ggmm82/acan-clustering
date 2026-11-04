# Paper/code correspondence audit

**Result: partial conformance, not a certification of the whole paper.**
The reviewed source is `paper/Algebraic_Concept_Attention_Paper.tex`, SHA-256
`314e7cb016b4de2c986b591f2e601ac4e844ee836ff2db621a74efe180320671`.
The supplied PDF `paper/Algebraic_Concept_Attention_Scientific_Paper.pdf` is also frozen
(SHA-256 `1fdd1a5f6e235d291c29364cae2dd1bd138df74db42256bb9ec36dd528f40396`).
The existing TeX and archived results/figures are preserved unchanged. An edit to the
paper invalidates the audit gate until its claim mapping is reviewed explicitly.

## Verified software contracts

| Contract | Repeated verification |
| --- | --- |
| Competitive assignment | Each tuple's attention weights sum to one over concepts. |
| Soft centroid update | G equals T X divided by the soft concept mass. |
| CP and optimization | The returned `neg_cp` is minus the stated implemented CP. |
| Analytical NumPy gradients | Central finite differences for G and F, alpha 0/.5/1, plus degenerate/saturated cases. |
| Dual backend | NumPy and PyTorch CPU forward values and gradients agree for identical parameters, in float64. |
| Runtime completion | Arbitrary-column missing-value completion, JSON round trip and rule extraction. |
| Behavioral fixtures | Compression/classification, XOR and irrelevant-attribute fixtures, masked completion. |

The former NumPy backward pass omitted derivatives through metric relevance
normalization and weighted distances, used a different dot-product scaling, and
added F terms absent from the returned objective. These are corrected. On the
small finite-difference fixture, maximum absolute errors were below 3e-10.
PyTorch CP epsilon handling and dtype preservation now agree with NumPy; its
forward entry point also accepts NumPy inputs. These checks do not establish GPU
equivalence, identical native random initialization, or universal convergence.

## Differences and unverified scientific claims

| Paper statement | Actual implementation / audit disposition |
| --- | --- |
| Eq. `v8_routing`: gamma log C minus beta squared distance | Existing tabular training uses mass multiplied by a blend of dot similarity and square-root distance, divided by tau. This is a different routing rule and remains explicitly documented. |
| Optimized omega and explicit interaction expansion Phi(X) | omega is fixed preprocessing based on first/second-order correlations; no learnable omega or explicit expanded feature tensor is present. |
| Masked tabular modeling | Completion at inference exists; the tabular objective is CP, not masked reconstruction loss. |
| Slot collapse / computational pruning | Inactive hard-assignment rows are removed from the exported model. Training arrays retain their original k. |
| Linear attention cost | The routing kernel is O(n k m) for fixed k and m; coupling initialization also has O(n m²) work. |
| Global convergence and asymptotic theorems | Early stopping bounds iterations. Adam usage and finite tests do not prove global convergence or the stated noise/collapse theorems. Independent mathematical review is required. |
| Real UCI benchmark interpretation | The historical script synthesizes Iris/Zoo/Wine/Wisconsin-like data. Those experiments are not real UCI dataset evaluations. New output labels and provenance state this explicitly. |
| Full published percentage reproduction | Selected synthetic checks pass, but dataset sizes/noise grids and baseline preprocessing differ from the prose in places. This is not an independent reproduction of every table. |
| LLM/audio/video/multimodal capability | Neither the paper's experiments nor the tabular code demonstrate it. This audit does not add an alternative generative architecture. |

`acan/paper_contract.py` contains the machine-readable version of these findings.
No theorem or published table was rewritten to make a test green.

One specific issue in the noise-proof argument also needs review: for a uniform
[0,1] feature, sigma tends to 1/sqrt(12), so the displayed concentration term
`max(0, 1 - 2*sigma)` tends to approximately **0.423**, not zero. That term is
not used by the current root-shift relevance implementation. Numerical fixture
success therefore cannot certify the proof as written.

## Historical experiment rerun

The full historical script was made runnable outside its original machine.
Outputs now go to an explicit directory (default `.runs/paper-benchmark`), not
over the archived paper results. Two benchmark defects were fixed: train/test
normalization leakage in baseline encoding, and mutation of pandas' read-only
NumPy views. Missing dummy columns are aligned without DataFrame fragmentation.
The reference backend is explicitly NumPy even when PyTorch is installed.

For the fixed seeds and current synthetic protocol, the rerun obtained:

- 100% ACAN classification over the irrelevant-feature grid through 48 noise attributes.
- 100% ACAN classification on the four noisy XOR configurations.
- Masked-cell accuracy 94.29%, 92.86%, 90.00%, 85.71% for one through four masked fields.
- Synthetic six-dataset ACAN accuracies: 92.5%, 82.5%, 100%, 100%, 100%, 100%.
- Noise/outlier experiments are variable, not a monotonic robustness guarantee
  (for example 82.5% mean at 5% corruption and 92.5% at 20%).

These are synthetic fixture results, not evidence of real-world or linguistic
performance. Baseline metrics after train-only preprocessing need not equal the
archived figures. The release archive includes fresh CSVs and provenance.

## Persistent deployment gates

Every PR, main verification and manually requested stable release must pass:

1. Normal Python/UI tests on 3.10–3.12.
2. Dedicated CPU PyTorch audit, with no skipped backend check or zero-test success.
3. Full historical synthetic rerun and selected regression thresholds: at least
   99% on clean/noisy XOR and irrelevant-attribute fixtures; masked completion
   within two percentage points of the audited synthetic values. Other tables
   must be complete and finite and are reported without fabricated guarantees.
4. Tests against the installed Docker image and HTTP tabular workflows.

`paper-audit.json` records the paper hash, source tree, run ID, clean checkout,
versions, test outcomes and known gaps.
`paper-benchmarks.zip` contains the freshly generated CSVs and their provenance.
Both are attached to preview/stable releases. The sealed runtime rejects missing,
failed, skipped, dirty, mismatched or modified audit evidence. Reusing a verified
PR image also retains the audit tied to that exact source tree and run.

Local audit commands:

```sh
python scripts/check_paper.py --require-torch --output reports/paper-audit.json
OPENBLAS_NUM_THREADS=1 python benchmarks/run_benchmarks.py --output-dir reports/benchmark
python scripts/check_benchmarks.py
```

Run the first command in an environment with CPU PyTorch installed; the official
CPU wheel index is `https://download.pytorch.org/whl/cpu`. CI currently verifies
PyTorch 2.15.0. Docker stays a CPU NumPy distribution and does not bundle Torch.
Local dirty-tree checks can help development but cannot seal a deployment.

## Scope correction

The proposed character language model and its cross-entropy objective were
withdrawn: they were an additional architecture, not the tabular model described
by the manuscript. This branch retains audit/numerical fixes only. GUI work must
call the existing ACAN APIs and must not introduce a different model or objective.
