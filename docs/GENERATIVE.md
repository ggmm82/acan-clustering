# Durable ACAN-LM generative laboratory

This implementation starts from the newly supplied
[ACAN-LM paper](../paper/ACAN_LM_vs_MiniCPM_Scientific_Paper.pdf) and
[unaltered reference package](../references/generative/source-manifest.json).
It is separate from the withdrawn character-language experiment and does not
change the tabular ACAN objective. See [LM audit](LM-AUDIT.md) for differences.

## What is implemented

The supported pretrained family is **SmolLM2** (135M, 360M, 1.7B), loaded from
Hugging Face using safetensors, native Transformers Llama code and a recorded
immutable Hub commit. Attention in the upper half of all layers is replaced by the
corrected `ACANConceptAttentionLayer`; the existing embeddings, layer norms,
FFNs and lower dense layers remain. GQA K/V projections are expanded by head
replication for the supplied full-channel local/slot attention. LoRA adapts
attention and FFN projections; slot prototypes and coupling are trainable. The
training loss is genuine next-token cross-entropy, with AdamW and a specified
warmup/stable/cosine-decay schedule. This is continued pre-training, not a claim
that grafting alone preserves the original model's quality.

**MiniCPM remote-code checkpoints are not silently executed or substituted.**
Their custom architecture requires a separately reviewed adapter. SmolLM2 is
the first implemented option from the user's requested MiniCPM-or-SmolLM2 path.
The 135M profile also enables CPU mechanics checks; larger models require CUDA
in this workbench. GPU throughput, VRAM requirements and general language quality
must be measured on the actual hardware/data; no single-GPU speed claim is made.

## User workflow

1. Open **Generative** and connect the separately deployed worker. Save the
   recovery-key JSON. This random browser-local key owns the persistent workspace;
   importing it in another browser restores access without an account. Treat it
   like a private project capability. Other workspace keys cannot see the files,
   jobs or generated text. Projects are explicitly selected; unrelated projects
   do not share training state.
2. Upload UTF-8 TXT (blank-line-separated paragraphs for new controlled projects;
   the legacy protocol retains one document per line) or JSONL (`{"text":"..."}` per line).
   The browser streams 1 MiB chunks; it does not base64 the entire file. Corpus,
   SHA-256 and prepared tokens live on the worker's persistent volume. Tokenization
   is incremental and resumable. Each document is followed by EOS. Individual
   records are limited to 16 MiB; uploads to 8 GiB.
3. Select the corpus/base model, number of optimizer steps and **Crea e avvia**.
   Downloads and training begin only on this explicit action. Advanced options
   configure block length, batch, gradient accumulation, LoRA rank, slots, window,
   learning rate and save interval. The immutable budget determines WSD: 5%
   warmup, stable until 80%, cosine decay to 10% of maximum LR. Reported token
   counts are actual next-token training targets, not corpus size.
4. Close the browser if needed: the owned worker process continues. A web UI restart does not affect the separate worker. An abrupt API-process
   restart can reattach to a live lease; graceful worker-service shutdown requests
   a checkpointed pause. A container/host interruption is marked
   interrupted and requires an explicit resume. Nothing automatically launches
   a multi-day training after a restart.
5. **Pausa e salva** finishes the current optimizer update, writes an atomic
   checkpoint, then releases the accelerator. Save also occurs every configured
   N steps or 5 minutes. **Riprendi** verifies the checkpoint and restores learned
   parameters, Adam moments, WSD step/budget, shuffled data cursor, CPU/CUDA,
   NumPy and Python RNG state. Fixed config, base/tokenizer commit and corpus/token
   hashes prevent accidentally resuming on different data or architecture.
6. Select a checkpoint and generate while the project is paused/completed.
   One accelerator operation runs at a time. Generation reads an immutable
   snapshot and does not alter training weights or optimizer state. Then resume.
   The app displays generated text, measured speed, whole-model cache bytes and
   slot/channel statistics. Latent channels are not certified human explanations.

## Persistence guarantees and limits

Checkpoint files are written to a temporary directory, flushed/fsynced, hashed,
then atomically renamed. The previous complete snapshot remains available if a
new write fails. The best checkpoint is retained outside the latest-three rolling window.
Complete checkpoints use safetensors plus `weights_only=True`
optimizer/RNG loading; uploaded model pickle/code files are never accepted.
A SQLite WAL with synchronous FULL persists ownership, state, metrics and pointers.
Incomplete snapshot directories are ignored; a corrupt newest complete snapshot
is rejected instead of silently resetting the model. A recorded checkpoint that
is missing also blocks resume; it never silently starts fresh from the base. The last three complete
snapshots are retained to bound disk use. All base weights can be reconstructed
from the recorded Hub commit; checkpoints contain only trainable tensors and
optimizer/RNG state, not a fresh copy of frozen weights every save.

A sudden kill/power failure can lose work **since the last complete checkpoint**;
it cannot guarantee preservation of unsaved optimizer updates. The checkpoint
interval and the visible saved step make that boundary explicit. Graceful worker
shutdown requests a pause and allows up to 100 seconds; Compose gives it 120.
Back up the whole worker volume (database, corpus, tokens, checkpoints and Hub
cache) and keep the exact worker image/version. Browser recovery keys must also
be backed up. The worker rejects a different code contract; do not replace its
image during a long run without pausing and retaining that image for resume.
GPU bitwise reproducibility across drivers/hardware is not certified. Exact
continuation is checked on the deterministic CPU fixture.

Training uses fixed unpadded blocks and single-device accumulation. Validation
is a disjoint final 5% of complete token blocks, sampled consistently (up to 8
blocks), not a benchmark such as MMLU/GSM8K. Single-prompt generation supports
at most 4096 context tokens and 256 new tokens; padded batches and beam search
are outside this first profile. Training retains prefix activations and is not
constant-memory in sequence length. Incremental decoding keeps bounded ACAN
sufficient statistics and the local window; dense lower-layer KV still grows.

## Deployment

The normal web image stays CPU/lightweight and does not import/install Torch.
A separate persistent LM worker is enabled explicitly; Hostinger can serve the
GUI while a remote GPU machine runs it. The internal service token is a server
configuration value, not a user login, and is never sent to browser JavaScript.
See `compose.generative.yaml`, its GPU override and `Dockerfile.generative`.
Do not expose port 8000 publicly without HTTPS/private networking. Use an HTTPS
endpoint for a remote worker and set `ACAN_LM_URL` and the same service token on
the web service. The worker data volume remains on the training machine.

CPU compose/build instructions and GPU build options are in the compose comments.
CUDA execution requires a compatible driver and NVIDIA Container Toolkit. The
CPU test image does not magically gain CUDA from a GPU device mount; build the
CUDA variant with the specified Torch index. GPU operation must be verified on
the deployment host. No GPU exists in the development environment used here.

### Local CPU setup

```sh
export ACAN_LM_SERVICE_TOKEN=$(openssl rand -hex 32)
docker compose -f compose.yaml -f compose.generative.yaml --profile generative up -d --build
```

Keep this token in your deployment's private environment file so it remains the
same after restarts. Never run `docker compose down -v` on the training stack
unless intentionally deleting learned state and corpus data. Back up the named
`acan-lm-data` volume after a checkpointed pause.

For a GPU host, add `-f compose.generative.gpu.yaml`. Its default official index
is CUDA 13.2 (`torch==2.15.0`); a compatible driver/container runtime is required.
Wheel availability was checked, but execution is not GPU-validated here. Do not
replace an existing worker image while a long job is running; pin and retain the
image that created its checkpoint contract.

### Controlled releases / Hostinger

CI builds and tests both the web image and a separate CPU worker candidate. The
release manifest seals both digests, the LM audit and the complete standalone
`compose.generative.hostinger.yaml`. Deploy this single file in Hostinger
instead of `compose.hostinger.yaml`: it includes both services, with no profile
to enable. Set a private `ACAN_LM_SERVICE_TOKEN` environment variable in Hostinger;
the file passes the same internal token to both services. No user login is added.
Download the raw YAML asset instead of copying formatted text: non-breaking
spaces are not valid YAML indentation. This CPU worker
is for the 135M mechanics profile; a long training needs the reviewed code built
with CUDA on a GPU host. For a remote worker, keep its data volume there and set
`ACAN_LM_URL=https://your-private-gpu-endpoint` plus the same service token on
the Hostinger frontend. No human login is added to the GUI.

## Guided public-corpus experiments

The Generative tab now offers **Avvio guidato · dati pubblici**. Choose a source
and a bounded preset, then explicitly click **Scarica e avvia la prova**. No user
file or training parameters are required. The action verifies/downloads text,
creates a separate workspace-local corpus/project and launches the existing
durable SmolLM2-135M–ACAN training worker. It never replaces an existing project
or interrupts a busy worker. A race for the worker leaves a created project that
can be started later with **Riprendi**. First use also downloads base weights;
CPU runtime is hardware-dependent, not guaranteed to be short.

- **Divina Commedia**: Italian medieval poetry, Project Gutenberg ebook 1000.
  Original work is public domain; edition terms are linked in the UI. This is
  a mechanics experiment, not modern Italian assistant training.
- **Tiny Shakespeare**: English theatrical text from the char-rnn collection,
  pinned to commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`.
- **Prova breve**: 20 updates, sequence length 64, batch 1, accumulation 1,
  checkpoints every 5 updates: 1,280 processed next-token targets.
- **Esercizio**: 200 updates, length 128, batch 1, accumulation 4, checkpoints
  every 20 updates: 102,400 processed next-token targets.

No crawling, arbitrary URL fetching, automatic model promotion or unlimited
training loop is introduced. Source downloads are bounded to 8 MiB per file (WikiText prepared data to 32 MiB) and checked
against a fixed byte count and SHA-256 before use; Gutenberg headers/footers are
removed from training input. A changed remote source fails closed until reviewed.
Source URL, edition conditions, raw/prepared hashes and preprocessing are stored
in `public-source.json` alongside both the corpus and project. Verified corpora
are reused within the same workspace; other workspaces receive distinct corpora
and jobs. Download failure launches no training. Runtime GET/state polling never
starts training. Existing checkpoints and the model's code contract are unchanged.

After launch, use the same pause/save, resume and checkpoint generation controls.
A source-appropriate prompt is prefilled. WikiText-2 additionally preserves its official article splits; new experiments use the controlled document protocol. Validation still uses the existing
held-out tail (up to eight blocks); it is a small diagnostic sample, not an
independent broad language benchmark. The original model NLL is shown as a green
reference line; red validation values are repeated between checkpoint evaluations.
Public text may already be present in base-model pretraining, so this experiment
cannot establish unseen-data performance or ACAN superiority.

A legacy-protocol local CPU smoke completed all 20 steps on the verified Dante corpus. Its
validation NLL changed from an original-model reference of 5.7384 to 5.6065
after training. This is one small, potentially pretraining-overlapping sample,
not evidence of broad language superiority; exact configuration and source
revision are in [PUBLIC-CORPUS-SMOKE.json](PUBLIC-CORPUS-SMOKE.json).

## Controlled CPU workflow (PR #6)

See the [step-by-step controlled laboratory guide](CONTROLLED-LM.md) for document
splits, exact deduplication, optional teacher distillation, early stopping,
best-checkpoint preservation, model export, matched dense-LoRA control, full
ACAN experiments, independent test/cache/speed measurements and own-project web
diagnostics. Existing projects retain their legacy trainer and compatible model
contract. New controlled projects record their own protocol/model code identity.

Internal Compose communication sets both `NO_PROXY` and `no_proxy` to the
worker hostname and loopback. This prevents a Docker client’s lowercase proxy
default from sending private frontend/worker traffic to an external proxy.
