"""Explicit common operators: tabular paper Eq. v8_routing and LM Eqs. 1–2.

Array operations work on NumPy arrays and PyTorch tensors. Prefix/global mass,
relevance estimation, objectives and output heads belong to their own adapters.
"""


def weighted_squared_distance(x, groups, relevance, alpha=2., epsilon=1e-6):
    focused = relevance ** alpha
    weights = focused / (focused.mean(axis=-1, keepdims=True) + epsilon)
    return (weights * (x-groups)**2).mean(axis=-1)


def routing_logits(x, groups, relevance, alpha=2., beta=8., coupling=0., epsilon=1e-6):
    return -beta * weighted_squared_distance(x, groups, relevance, alpha, epsilon) + coupling


def mass_modulation(logits, mass_fraction, gamma, logarithm, epsilon=1e-4):
    return logits + gamma * logarithm(mass_fraction + epsilon)
