# Generative laboratory

The **Laboratory** tab runs an experimental, local CPU character language model.
It trains from scratch on uploaded UTF-8 text; it does not call a hosted model,
download weights, or provide an LLM/audio/video/multimodal backend.

## Architecture and relationship to the paper

Each input is the previous C characters, with explicit beginning-of-document
padding. Learnable categorical concept prototypes and position relevances
define weighted squared distances. Competitive routing uses
`softmax(gamma * log(mass) - beta * distance_squared)`. Each concept has a
learnable next-character distribution; their mixture defines the prediction.
Small uniform smoothing ensures finite held-out probabilities, including UNK.

All concept, relevance, mass and emission parameters are trained with Adam on
next-token cross entropy. Numerical gradient tests cover all parameter groups.
This uses the paper's log-mass competitive routing idea but **is an extension**:
its categorical prototypes, learned position weights and linguistic objective
are not the tabular v8 CP model. It does not resolve the tabular discrepancies
listed in [Paper audit](PAPER-AUDIT.md).

## Use

1. Open **Laboratory**, try the provided synthetic corpus, or upload a text file.
   One document/sentence per line is recommended.
2. Choose context size, concepts, epochs, learning rate and seed. Click
   **Addestra prototipo**. Training runs in the background with a stop control.
3. Inspect the train/validation loss curves and held-out test metrics. Download
   the experiment report to retain settings/measurements with the model JSON.
4. Supply a prompt and sampling settings, then click **Genera testo**. EOS stops
   the sample; otherwise generation ends at the selected length limit.
5. Download the model and reload it in a later page. The same prompt, weights,
   sampling parameters and seed reproduce the same sample.

The trace distinguishes original model probability from the actual sampling
probability after temperature/top-k/top-p filtering. Routing weights and the
largest per-token concept contribution expose the mixture computation; they are
not causal explanations or calibrated confidence guarantees.

## Measurement protocol

- Identical documents are deduplicated and grouped before a seeded 70/15/15
  split. With fewer than ten unique documents, contiguous text segments are
  used, with context reset at each boundary.
- The vocabulary is learned **only** from training. Unseen held-out characters
  become UNK, and unknown-token counts are reported.
- Context windows are built after splitting; no window crosses a split boundary.
  Each split has at most 4,096 deterministically sampled windows.
- Early stopping uses validation loss (patience 8); the actual selected epoch
  and both epoch loss curves are retained. Test data never select the epoch.
- Reported NLL uses natural logs, bits/token divides NLL by log(2), and perplexity
  is exp(NLL). EOS and UNK are included in the scored targets.
- Smoothed unigram and bigram baselines use the same training targets and test
  windows. Training/test window overlap is reported: high overlap limits any
  claim about novel contexts despite document separation.

On the small default synthetic Italian demo with seed 42, context 6, concepts 12
and 25 epochs, a local run measured test perplexity about **4.11**, against
**16.58** unigram and **3.83** bigram. This improves over the unigram baseline
but does not beat the bigram model. Train/test window overlap was 100% for this
repetitive toy corpus; these figures do not demonstrate general language ability.
The regression suite also tests held-out alternating-character sequences.

## Resources, state and limitations

The prototype supports 100–40,000 corpus characters, at most 126 distinct
training characters plus UNK/EOS, context 1–16, concepts 2–64, and epochs 1–100.
Generation supports prompts up to 2,000 characters and up to 512 output tokens.
Up to four language jobs can coexist in one process; memory is bounded by the
model/window limits. Use the current single-worker deployment.

Jobs are scoped by the signed browser cookie plus the distinct page ID, as with
tabular training. Two tabs can train on different corpora without sharing their
models, stores or stop controls. There is no authentication. Refresh creates a
new page scope; export before leaving. JSON contains parameters, vocabulary,
history and aggregate metrics/hashes, not the source corpus or optimizer state.
It is an inference export, **not** a resumable training checkpoint.

Large-vocabulary training, subword tokenization, GPU/distributed optimization,
persistent job storage, optimizer resume, and audio/video/multimodal encoders
are future research work, not switches hidden in the current GUI.
