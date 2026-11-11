"""Experimental shared CP routing with prefix summaries and a local output path."""
import math
import torch
from .attention import ACANConceptAttentionLayer


class CPUnifiedAttention(ACANConceptAttentionLayer):
    def attend(self, h, position_embeddings=None, use_cache=False):
        b, length, d = h.shape
        previous = self.cache if use_cache else None
        past = previous['seen'] if previous else 0
        x = h.float()  # stable prefix statistics, including for bf16 base weights
        positions = torch.arange(past + 1, past + length + 1, device=x.device, dtype=x.dtype)[None, :, None]
        sums = x.cumsum(1) + (previous['sum'][:, None] if previous else 0)
        minimum, maximum = x.cummin(1).values, x.cummax(1).values
        if previous:
            minimum = torch.minimum(minimum, previous['min'][:, None])
            maximum = torch.maximum(maximum, previous['max'][:, None])
        root = sums / positions
        center = .5 * (maximum + minimum)
        denominator = .5 * (maximum - minimum) + (root - center).abs() + 1e-5
        difference = self.G[None, None] - root[:, :, None]
        r_shift = (difference.abs() / denominator[:, :, None]).clamp(0, 1)
        from acan.core.variational import solve
        routing_x = x.sigmoid()
        if getattr(self,'joint_model',False):
            routing_x = routing_x.double()
        cp_sums = routing_x.cumsum(1)+(previous['cp_sum'][:,None] if previous else 0)
        cp_min,cp_max = routing_x.cummin(1).values,routing_x.cummax(1).values
        if previous:
            cp_min=torch.minimum(cp_min,previous['cp_min'][:,None])
            cp_max=torch.maximum(cp_max,previous['cp_max'][:,None])
        cp_root=cp_sums/positions
        cp_den=.5*(cp_max-cp_min)+(cp_root-.5*(cp_min+cp_max)).abs()
        all_assignments = []
        all_cp = []
        all_relevance = []
        saved_content = []
        saved_overlaps = []
        for batch_index in range(b):
            count = previous['mass'][batch_index,:,None] if previous else x.new_zeros(self.k,1)
            content = previous['cp_x'][batch_index] if previous else x.new_zeros(self.k,d)
            pieces = []
            prefix_cp = []
            prefix_relevance = []
            for index in range(length):
                den = cp_den[batch_index,index:index+1]
                valid = den>1e-8
                state = dict(anchors=self.G.sigmoid(),omega=self.omega.abs()[None]*valid,
                    root=cp_root[batch_index,index:index+1],den=torch.where(valid,den,torch.ones_like(den)))
                if getattr(self,'joint_model',False):
                    from acan.algebra import concept_step
                    solved = concept_step(routing_x[batch_index,index:index+1],
                        state['anchors'].double(),state['root'],state['den'],state['omega'].double(),
                        self.log_beta.clamp(0,9).exp().double(),
                        slice(None),lambda value,floor: value.clamp_min(floor),torch.clamp,
                        lambda scores: scores.softmax(0),base_counts=count,base_sums=content,
                        masses=self.masses.double(),exponential=torch.exp,
                        distinct=(1-torch.eye(self.k,device=x.device,dtype=torch.float64) if self.training else None),
                        compute_cp=self.training)
                else:
                    solved = solve(routing_x[batch_index,index:index+1],state,steps=3,
                        base_counts=count,base_sums=content)
                assignment = solved['T'][:,0]
                pieces.append(assignment)
                if getattr(self,'joint_model',False):
                    prefix_relevance.append(solved['R'])
                if self.training:
                    from acan.algebra import normalized_cluster_probability
                    # Anchors may lie outside a short prefix's observed range.
                    # Cap relevance to keep CP/sqrt(d) bounded in [0,1].
                    prefix_cp.append(solved['normalized_cp'] if 'normalized_cp' in solved else
                        normalized_cluster_probability(solved['F'].T, solved['R'].clamp(0, 1),
                            lambda value, floor: value.clamp_min(floor)))
                count = count+assignment[:,None]
                content = content+assignment[:,None]*routing_x[batch_index,index]
            all_assignments.append(torch.stack(pieces))
            if prefix_cp:
                all_cp.append(torch.stack(prefix_cp))
            if prefix_relevance:
                all_relevance.append(torch.stack(prefix_relevance))
            saved_content.append(content)
            if 'overlap' in solved:
                saved_overlaps.append(solved['overlap'].detach())
        assignments = torch.stack(all_assignments)
        self.last_cp = torch.stack(all_cp) if all_cp else None
        mass1 = assignments.cumsum(1)+(previous['mass1'][:,None] if previous else 0)
        mass = assignments.cumsum(1) + (previous['mass'][:, None] if previous else 0)
        active = mass / positions >= self.pruning
        if getattr(self,'joint_model',False):
            active = (self.masses>0)[None,None].expand(b,length,-1)
        fallback = torch.nn.functional.one_hot(mass.argmax(-1), self.k).bool()
        active = torch.where(active.any(-1, keepdim=True), active, fallback)
        q, key, value = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        if position_embeddings is not None:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = position_embeddings
            qh = q.view(b, length, -1, self.head_dim).transpose(1, 2)
            kh = key.view(b, length, -1, self.head_dim).transpose(1, 2)
            qh, kh = apply_rotary_pos_emb(qh, kh, cos, sin)
            q, key = qh.transpose(1, 2).reshape(b, length, d), kh.transpose(1, 2).reshape(b, length, d)
        q, key, value = q.float(), key.float(), value.float()
        weighted_values = (assignments[..., None] * value[:, :, None]).cumsum(1)
        weighted_x = (assignments[..., None] * x[:, :, None]).cumsum(1)
        weighted_x2 = (assignments[..., None] * x[:, :, None].square()).cumsum(1)
        if previous:
            weighted_values = weighted_values + previous['values'][:, None]
            weighted_x = weighted_x + previous['x'][:, None]
            weighted_x2 = weighted_x2 + previous['x2'][:, None]
        slot_states = weighted_values / (mass[..., None] + 1e-5)
        centroid = weighted_x / (mass[..., None] + 1e-6)
        std = (weighted_x2 / (mass[..., None] + 1e-6) - centroid.square()).clamp_min(0).add(1e-6).sqrt()
        tightness = (1 - 1.5 * std).clamp(0, 1)
        final_relevance = .55 * r_shift + .45 * tightness
        if getattr(self,'joint_model',False):
            final_relevance = torch.stack(all_relevance)
        gate = (final_relevance * active[..., None]).sum(2) / active.sum(-1, keepdim=True)
        if getattr(self,'joint_model',False):
            gate = (final_relevance*assignments[...,None]).sum(2)
        context = (assignments[..., None] * active[..., None] * slot_states).sum(2) * gate
        old_keys = previous['keys'] if previous else key[:, :0]
        old_values = previous['local_values'] if previous else value[:, :0]
        keys = torch.cat((old_keys, key), 1)
        values = torch.cat((old_values, value), 1)
        left = self.window - 1 - old_keys.shape[1]
        padded_keys = torch.nn.functional.pad(keys, (0, 0, left, 0))
        padded_values = torch.nn.functional.pad(values, (0, 0, left, 0))
        kw = padded_keys.unfold(1, self.window, 1).transpose(-1, -2)
        vw = padded_values.unfold(1, self.window, 1).transpose(-1, -2)
        scores = (q[:, :, None] * kw).sum(-1) / math.sqrt(d)
        valid = torch.arange(self.window, device=x.device)[None, :] + torch.arange(length, device=x.device)[:, None] >= left
        scores = scores.masked_fill(~valid[None], -torch.inf)
        local = (scores.softmax(-1)[..., None] * vw).sum(2)
        output = self.o_proj((.55 * context + .45 * local).to(h.dtype))
        if use_cache:
            keep = self.window - 1
            self.cache = dict(seen=past + length, sum=sums[:, -1].detach().clone(), min=minimum[:, -1].detach().clone(),
                              max=maximum[:, -1].detach().clone(), mass1=mass1[:, -1].detach().clone(), mass=mass[:, -1].detach().clone(),
                              values=weighted_values[:, -1].detach().clone(), x=weighted_x[:, -1].detach().clone(), x2=weighted_x2[:, -1].detach().clone(),
                              keys=keys[:, -keep:].detach().clone() if keep else keys[:, :0].detach().clone(),
                              local_values=values[:, -keep:].detach().clone() if keep else values[:, :0].detach().clone())
        if use_cache:
            self.cache.update(cp_sum=cp_sums[:,-1].detach().clone(),cp_min=cp_min[:,-1].detach().clone(),
                cp_max=cp_max[:,-1].detach().clone(),cp_x=torch.stack(saved_content).detach().clone())
        self.last_info = dict(active_slots=active[:, -1].sum(-1).detach().clone(),
                              relevance=final_relevance[:, -1].detach().clone(), masses=mass[:, -1].detach().clone(),
                              assignments=assignments[:, -1].detach().clone())
        if saved_overlaps:
            self.last_info['overlap'] = torch.stack(saved_overlaps).mean(0)
        return output


class CPJointAttention(CPUnifiedAttention):
    """The same learned concept model as the joint tabular adapter."""
    joint_model = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initialize_concepts()

    def initialize_concepts(self):
        if not hasattr(self,'masses'):
            self.masses = torch.nn.Parameter(self.G.new_full((self.k,),1/self.k))
            self.log_beta = torch.nn.Parameter(self.G.new_tensor(math.log(128.)))
