"""Bounded, opt-in observations. Hooks never replace tensors or modify gradients."""
from contextlib import contextmanager
import json
import math
import re


def queue(store, job):
    if job['config'].get('protocol') != 'controlled-v1':
        raise ValueError('La cattura durante il training richiede un nuovo progetto controllato. Puoi ispezionare un checkpoint legacy in generazione.')
    if job['status'] not in ('preparing', 'running', 'queued'):
        raise ValueError('Avvia o riprendi il training prima di richiedere una cattura.')
    with store.db() as db:
        db.execute('INSERT INTO layer_inspections(job,pending) VALUES(?,1) '
                   'ON CONFLICT(job) DO UPDATE SET pending=1', (job['id'],))


def latest(store, job):
    with store.db() as db:
        row = db.execute('SELECT * FROM layer_inspections WHERE job=?', (job['id'],)).fetchone()
    return dict(pending=bool(row and row['pending']), report=json.loads(row['result']) if row and row['result'] else None)


def stats(tensor):
    import torch
    x = tensor.detach().float()
    finite = bool(torch.isfinite(x).all())
    return dict(rms=float(x.double().square().mean().sqrt()) if finite else None, finite=finite)


class Capture:
    """Only final-position activations, all layers; no full attention matrices."""
    def __init__(self, model, mode, tokenizer=None, max_frames=1):
        self.model, self.mode, self.tokenizer = model, mode, tokenizer
        self.max_frames = min(max_frames, 32)
        self.frames, self.handles, self.current = [], [], None
        self.seen = 0

    def __enter__(self):
        from .attention import ACANConceptAttentionLayer
        core = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        self.handles.append(core.model.register_forward_pre_hook(self.start, with_kwargs=True))
        for index, layer in enumerate(core.model.layers):
            kind = 'ACAN' if isinstance(layer.self_attn, ACANConceptAttentionLayer) else 'dense'
            self.handles.append(layer.register_forward_hook(self.layer_hook(index, kind)))
        self.handles.append(core.lm_head.register_forward_hook(self.end))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start(self, module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        size = ids.shape[1] if ids is not None else kwargs['inputs_embeds'].shape[1]
        self.seen += size
        self.current = dict(forward=len(self.frames), input_tokens=size, prefix_tokens=self.seen, layers=[]) if len(self.frames) < self.max_frames else None
        if self.current is not None and self.tokenizer and ids is not None:
            self.current['last_input_token'] = self.tokenizer.decode([int(ids[0,-1])])

    def layer_hook(self, index, kind):
        def observe(layer, inputs, output):
            import torch
            if self.current is None:
                return
            out = output[0] if isinstance(output, tuple) else output
            before, after = inputs[0][:1, -1], out[:1, -1]
            row = dict(layer=index, kind=kind, input=stats(before), output=stats(after), change=stats(after-before))
            info = getattr(layer.self_attn, 'last_info', None)
            if kind == 'ACAN' and info:
                mass = info['masses'][0].detach().float()
                assignment = info['assignments'][0].detach().float()
                finite = bool(torch.isfinite(mass).all() and torch.isfinite(assignment).all() and torch.isfinite(info['relevance']).all())
                row['slots_finite'] = finite
                if finite:
                    p = mass / mass.sum().clamp_min(1e-12)
                    entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
                    top = torch.topk(assignment, min(3, assignment.numel()))
                    row.update(mass_fractions=p.cpu().tolist(), assignment=assignment.cpu().tolist(),
                               effective_slots=math.exp(entropy), active_slots=int(info['active_slots'][0]),
                               dominant_mass=float(p.max()), top_slots=top.indices.cpu().tolist(),
                               mean_relevance=float(info['relevance'][0].float().mean()))
                    if float(p.max()) > .95:
                        row['warning'] = 'Più del 95% della massa in uno slot: concentrazione da indagare, non prova di errore.'
            self.current['layers'].append(row)
        return observe

    def end(self, module, args, output):
        import torch
        if self.current is None:
            return
        logits = output[:1, -1].detach().float()
        self.current['logits_finite'] = bool(torch.isfinite(logits).all())
        if self.current['logits_finite']:
            p = logits.softmax(-1)[0]
            values, ids = torch.topk(p, min(5, p.numel()))
            self.current['next_token_candidates'] = [dict(token_id=int(i), probability=float(v),
                token=self.tokenizer.decode([int(i)]) if self.tokenizer else None) for i, v in zip(ids, values)]
        self.frames.append(self.current)
        self.current = None

    def report(self, step=None):
        gradients = {}
        if self.mode == 'training':
            for name, parameter in self.model.named_parameters():
                match = re.search(r'layers\.(\d+)\.', name)
                if match and parameter.requires_grad:
                    row = gradients.setdefault(int(match[1]), dict(squared=0., finite=True, parameters=0, with_gradient=0))
                    row['parameters'] += parameter.numel()
                    if parameter.grad is not None:
                        value = stats(parameter.grad)
                        row['with_gradient'] += parameter.numel()
                        row['finite'] &= value['finite']
                        if value['finite']:
                            row['squared'] += value['rms']**2 * parameter.numel()
            gradients = {str(i): dict(norm=math.sqrt(r['squared']) if r['finite'] else None,
                finite=r['finite'], trainable_parameters=r['parameters'], parameters_with_gradient=r['with_gradient']) for i,r in gradients.items()}
        return dict(schema=1, mode=self.mode, step=step, frames=self.frames, gradients=gradients,
                    scope='Ultima posizione del primo elemento del batch. Training: attivazioni del primo microbatch e gradienti accumulati prima del clipping; inferenza: un campione per forward, prima il prompt poi i token successivi. Nessun testo grezzo del training.',
                    limits='Le osservazioni non attribuiscono cause o significati linguistici. Concentrazione degli slot e gradienti nulli possono essere legittimi. La cattura aggiunge costo: i suoi tempi non sono un benchmark.')


@contextmanager
def training_probe(store, job, model, step):
    with store.db() as db:
        claimed = db.execute('UPDATE layer_inspections SET pending=0 WHERE job=? AND pending=1 RETURNING job', (job['id'],)).fetchone()
    if not claimed:
        yield None
        return
    capture = Capture(model, 'training')
    try:
        with capture:
            yield capture
        result = capture.report(step)
    except Exception as exc:
        result = dict(step=step, error=type(exc).__name__, frames=[])
        raise
    finally:
        with store.db() as db:
            db.execute('UPDATE layer_inspections SET result=? WHERE job=?', (json.dumps(result, allow_nan=False),job['id']))
