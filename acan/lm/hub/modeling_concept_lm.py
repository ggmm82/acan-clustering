"""A language model made of concepts, for transformers: AutoModelForCausalLM, generate and pipeline work as for any
model. The network itself is computed with numpy (concept_net.py); torch only carries the results."""
import numpy as np
import torch
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .concept_net import ConceptNet
from .configuration_concept_lm import ConceptLMConfig


class _Reader:
    """Reads a text one token at a time; the model's own words do not go into the memory of the document."""

    def __init__(self, net, eos, ids=()):
        self.cursor = net.cursor()
        self.cursor.observe(int(eos))
        if len(ids):
            self.cursor.prefill(np.asarray(ids, np.int64))

    def observe(self, token, remember=True):
        keys = len(self.cursor.keys)
        self.cursor.observe(int(token))
        if not remember and len(self.cursor.keys) > keys:
            self.cursor.keys.pop(); self.cursor.after.pop()

    def distribution(self):
        return self.cursor.distribution()


class ConceptLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ConceptLMConfig
    main_input_name = "input_ids"

    def __init__(self, config, net=None):
        super().__init__(config)
        self.net = net
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)   # device and dtype for transformers

    def _init_weights(self, module):
        pass

    @classmethod
    def from_pretrained(cls, name, *args, config=None, **kwargs):
        from safetensors.numpy import load_file
        from transformers.utils import cached_file
        hub = {key: kwargs[key] for key in ("revision", "cache_dir", "token", "local_files_only", "force_download") if key in kwargs}
        if config is None:
            config = ConceptLMConfig.from_pretrained(name, **hub)
        net = ConceptNet.from_arrays(load_file(cached_file(name, "concepts.safetensors", **hub)), "net.")
        model = cls(config, net)
        model.eval()
        return model

    def _reader(self, ids):
        ids = np.asarray(ids, np.int64)
        ends = np.flatnonzero(ids == self.config.eos_token_id)
        return _Reader(self.net, self.config.eos_token_id, ids[int(ends[-1]) + 1:] if len(ends) else ids)

    def _row(self, ids):
        """The probabilities of the next token after every position of a text."""
        out = np.empty((len(ids), self.config.vocab_size), np.float32)
        reader = _Reader(self.net, self.config.eos_token_id)
        for t, token in enumerate(ids):
            if token == self.config.eos_token_id:
                reader = _Reader(self.net, self.config.eos_token_id)
            else:
                reader.observe(token)
            out[t] = reader.distribution()
        return out

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        ids = input_ids.detach().cpu().numpy()
        mask = np.ones_like(ids) if attention_mask is None else attention_mask.detach().cpu().numpy()
        logits = np.zeros(ids.shape + (self.config.vocab_size,), np.float32)
        for b in range(len(ids)):
            kept = np.flatnonzero(mask[b])
            logits[b, kept] = np.log(np.maximum(self._row(ids[b, kept]), 1e-30))
        logits = torch.from_numpy(logits).to(self.anchor.device)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, generation_config=None, max_new_tokens=None, do_sample=None,
                 temperature=None, top_k=None, top_p=None, eos_token_id=None, pad_token_id=None, streamer=None, **kwargs):
        """Writes after every row, reading each new token once (as the model was trained to write)."""
        settings = generation_config or self.generation_config
        max_length = kwargs.get("max_length")
        temperature = settings.temperature if temperature is None else temperature
        do_sample = settings.do_sample if do_sample is None else do_sample
        top_k = settings.top_k if top_k is None else top_k
        top_p = settings.top_p if top_p is None else top_p
        stop = eos_token_id if eos_token_id is not None else settings.eos_token_id
        stop = set(stop if isinstance(stop, (list, tuple)) else [stop]) | set(self.config.stop_token_ids) - {None}
        pad = self.config.pad_token_id if pad_token_id is None else pad_token_id
        ids = input_ids.detach().cpu().numpy()
        mask = np.ones_like(ids) if attention_mask is None else attention_mask.detach().cpu().numpy()
        count = max_new_tokens or settings.max_new_tokens or ((max_length or settings.max_length or 20) - ids.shape[1])
        rows = []
        for b in range(len(ids)):
            prompt = ids[b, np.flatnonzero(mask[b])]
            reader, written = self._reader(prompt), []
            for _ in range(max(0, int(count))):
                p = torch.from_numpy(np.asarray(reader.distribution(), np.float64))
                if do_sample:
                    logits = torch.log(p.clamp_min(1e-300)) / max(float(temperature or 1.), 1e-6)
                    if top_k:
                        logits[logits < torch.topk(logits, int(top_k)).values[-1]] = -float("inf")
                    probs = torch.softmax(logits, -1)
                    if top_p is not None and top_p < 1.:
                        order = torch.argsort(probs, descending=True)
                        drop = torch.cumsum(probs[order], 0) - probs[order] > top_p
                        probs[order[drop]] = 0.
                        probs = probs / probs.sum()
                    token = int(torch.multinomial(probs, 1))
                else:
                    token = int(torch.argmax(p))
                if token in stop:
                    written.append(token)
                    break
                written.append(token)
                if streamer is not None:
                    streamer.put(torch.tensor([token]))
                reader.observe(token, remember=False)
            rows.append(list(ids[b]) + written)
        if streamer is not None:
            streamer.end()
        width = max(len(row) for row in rows)
        out = torch.full((len(rows), width), pad, dtype=torch.long)
        for b, row in enumerate(rows):
            out[b, :len(row)] = torch.tensor(row, dtype=torch.long)
        return out.to(input_ids.device)
