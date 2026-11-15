"""concept-graph-v2: the concept layer on the fixed dyadic structure of acan/lm/graph_scan.py.

Same construction, weights and graft boundary as the other concept layers; only ``attend`` differs.
The concepts, the reading and the cost of adding two concepts together are those of
concept-graph-v1 (acan/lm/concept_attention.py); what changes is that the generalization happens
inside blocks fixed by the position instead of in a search over the whole context, so that every
position of a training block is read at once and no step depends on the previous one.

``slots`` is the number of concepts of a block. A reader keeps ``window`` exact tokens and at most
``slots`` concepts for every power of two in the number of its older tokens.
"""
import torch

from .concept_attention import ConceptSlotAttention
from .graph_scan import GraphStream, graph_scan

PROFILE = 'concept-graph-v2'


class ConceptGraphScanAttention(ConceptSlotAttention):
    def attend(self, h, position_embeddings=None, use_cache=False):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        b, length, d = h.shape
        heads = d//self.head_dim
        previous = self.cache if use_cache else None
        past = previous['seen'] if previous else 0
        split = lambda value: value.view(b, length, heads, self.head_dim).transpose(1, 2)
        q, raw, value = split(self.q_proj(h)), split(self.k_proj(h)), split(self.v_proj(h))
        key = raw
        if position_embeddings is not None:
            q, key = apply_rotary_pos_emb(q, raw, *position_embeddings)
        q, key, raw, value = q.float(), key.float(), raw.float(), value.float()
        frequency = getattr(self, 'rope_frequency', None)        # the model's own rotary frequencies
        if position_embeddings is None:
            frequency = torch.zeros(self.head_dim//2, device=h.device)
        elif frequency is None:
            frequency = 1./10000.**(torch.arange(0, self.head_dim, 2, dtype=torch.float32)/self.head_dim)
        frequency = frequency.to(device=h.device, dtype=torch.float32)
        scale = self.head_dim**-.5
        stream = None
        if previous is None:                                     # a whole block: every position at once
            out, received = graph_scan(q, raw, key, value, self.window, self.k, scale, frequency, details=True)
            if use_cache or not self.training:                   # what a writer would hold after it
                with torch.no_grad():
                    stream = GraphStream(self.window, self.k, scale, frequency).load(raw, key, value, received)
        else:                                                    # continuing: one token at a time
            stream = previous['stream']
            out = torch.stack([stream.step(q[:, :, t], raw[:, :, t], key[:, :, t], value[:, :, t]) for t in range(length)], 2)
        output = self.o_proj(out.transpose(1, 2).reshape(b, length, d).to(h.dtype))
        if use_cache:
            self.cache = dict(seen=past+length, stream=stream)
        self.last_info = self._graph_info(stream, b, d, h.device)
        return output

    def cache_bytes(self):
        stream = (self.cache or {}).get('stream')
        tensors = stream.tensors() if stream is not None else []
        tensors += [item for item in (self.last_info or {}).values() if isinstance(item, torch.Tensor)]
        storages = {item.untyped_storage().data_ptr(): item.untyped_storage().nbytes() for item in tensors}
        return sum(storages.values())

    def _graph_info(self, stream, b, d, device):
        """Inspection in the shape the workbench expects, for the concepts the last reader holds.

        One row per concept, at most ``slots`` for every level in use: masses, the share the last
        token gave to each, and how tight each concept is on the scale of all of them together.
        """
        nodes = [node for node in (stream.counter if stream is not None else []) if node is not None]
        if not nodes:
            return dict(active_slots=torch.zeros(b, dtype=torch.long, device=device), relevance=torch.zeros(b, self.k, d, device=device),
                        masses=torch.zeros(b, self.k, device=device), assignments=torch.zeros(b, self.k, device=device))
        table = {name: torch.cat([node[name] for node in nodes], 2).detach() for name in ('C', 'U', 'S')}
        n = table['C'].shape[2]
        width = max(n, self.k)
        mass = table['C'].clamp(min=1.)
        within = (table['S']/mass-(table['U']*table['U']).sum(-1)/(mass*mass)).clamp(min=0.)
        total = table['C'].sum(-1, keepdim=True).clamp(min=1.)
        whole = (table['S'].sum(-1, keepdim=True)/total-(table['U'].sum(2)**2).sum(-1, keepdim=True)/(total*total)).clamp(min=1e-12)
        masses = torch.zeros(b, width, device=device)
        assignments = torch.zeros(b, width, device=device)
        relevance = torch.zeros(b, width, d, device=device)
        masses[:, :n] = table['C'].mean(1)
        relevance[:, :n] = (1-within/whole).clamp(0, 1).mean(1)[..., None].expand(b, n, d)
        shares = getattr(stream, 'shares', None)
        if shares is not None and shares.shape[-1] >= n:
            assignments[:, :n] = shares[:, :, -n:].detach().mean(1)
        active = (table['C'].mean(1) > .5).sum(-1)
        return dict(active_slots=active, relevance=relevance, masses=masses, assignments=assignments)
