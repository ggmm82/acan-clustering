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
