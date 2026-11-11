"""Explicit prefix CP penalty, separate from next-token NLL and teacher KL."""
import torch
from .cp_attention import CPUnifiedAttention
from acan.algebra import prediction_cp_objective


def regularize(model, language_loss, weight):
    layers = [module for module in model.modules() if isinstance(module, CPUnifiedAttention)]
    if not layers:
        if weight:
            raise ValueError('Il termine CP richiede il routing CP diretta e livelli ACAN.')
        return language_loss, language_loss.new_zeros(())
    values = [getattr(layer, 'last_cp', None) for layer in layers]
    if any(value is None or value.shape[1] < 2 for value in values):
        raise ValueError('CP di training assente o blocco senza token da prevedere.')
    # The last input position has no next-token target in the shifted LM loss.
    # Each layer contributes the mean over the same batch and predictive positions.
    mean_cp = torch.stack([value[:, :-1].mean() for value in values]).mean()
    return (prediction_cp_objective(language_loss, mean_cp, weight) if weight else language_loss), 1-mean_cp


def clear(model):
    for module in model.modules():
        if isinstance(module, CPUnifiedAttention):
            module.last_cp = None
def project_concept_masses(model):
    """Enforce the same simplex constraint as the native CP optimizer."""
    import torch
    from acan.algebra import simplex_projection
    from .cp_attention import CPJointAttention
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer,CPJointAttention):
                layer.masses.copy_(simplex_projection(layer.masses.double()).to(layer.masses.dtype))


def coordinate_concept_step(model, step, evaluate):
    """Pairwise simplex coordinate descent on the actual training objective.

    Compare both endpoints with the same dropout realization and leave all RNG
    streams unchanged, so structural optimization remains exactly resumable.
    """
    from acan.algebra import simplex_coordinate_candidate, simplex_coordinate_index
    from .cp_attention import CPJointAttention
    layers = [layer for layer in model.modules() if isinstance(layer,CPJointAttention)]
    if not layers:
        return False
    layer = layers[step % len(layers)]
    index = simplex_coordinate_index(layer.masses.detach(),step//len(layers))
    if index is None:
        return False
    original = layer.masses.detach().clone()
    devices = [layer.masses.device.index] if layer.masses.is_cuda else []
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        cpu_state = torch.get_rng_state()
        gpu_states = [torch.cuda.get_rng_state(device) for device in devices]
        try:
            before = evaluate()
            overlap = (layer.last_info or {}).get('overlap')
            proposal = simplex_coordinate_candidate(layer.masses.detach(),index,
                overlap[index] if overlap is not None else None)
            torch.set_rng_state(cpu_state)
            for device,state in zip(devices,gpu_states):
                torch.cuda.set_rng_state(state,device)
            layer.masses.copy_(proposal)
            after = evaluate()
            accepted = bool(torch.isfinite(after) and after <= before)
            if not accepted:
                layer.masses.copy_(original)
            return accepted
        except BaseException:
            layer.masses.copy_(original)
            raise
        finally:
            clear(model)
