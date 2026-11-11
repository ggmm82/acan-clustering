"""Common NumPy/PyTorch concept algebra.

The joint profile shares routing, sufficient statistics, relevance and the CP
training objective. Earlier paper/legacy profiles retain their named operators.
Adapters provide training-only or causal-prefix statistics and output heads.
"""


def weighted_squared_distance(x, groups, relevance, alpha=2., epsilon=1e-6):
    focused = relevance ** alpha
    weights = focused / (focused.mean(axis=-1, keepdims=True) + epsilon)
    return (weights * (x-groups)**2).mean(axis=-1)


def routing_logits(x, groups, relevance, alpha=2., beta=8., coupling=0., epsilon=1e-6):
    return -beta * weighted_squared_distance(x, groups, relevance, alpha, epsilon) + coupling


def mass_modulation(logits, mass_fraction, gamma, logarithm, epsilon=1e-4):
    return logits + gamma * logarithm(mass_fraction + epsilon)


def normalized_cluster_probability(frequencies, relevance, maximum):
    """CP / sqrt(m); callers supply bounded relevance and backend maximum."""
    sums = maximum(relevance.sum(axis=-1, keepdims=True), 1e-12)
    squares = (relevance * relevance).sum(axis=-1, keepdims=True)
    return maximum((frequencies * squares / sums).sum(), 1e-12) ** .5


def prediction_cp_objective(prediction_loss, mean_normalized_cp, weight):
    """Common supervised objective for classification and next-token prediction."""
    return prediction_loss + weight * (1 - mean_normalized_cp)


def concept_assignments(x, anchors, relevance, beta, features, softmax):
    """One routing equation for training and frozen tabular/causal LM inference."""
    focused = relevance[:, features] ** 2 + 1e-4
    weights = focused / focused.sum(axis=1, keepdims=True)
    a, observed = anchors[:, features], x[:, features]
    distance = ((weights*a*a).sum(axis=1, keepdims=True)
                - 2*(weights*a)@observed.T + weights@(observed*observed).T)
    scores = -beta*distance
    return softmax(scores), weights, scores


def concept_step(x, anchors, root, denominator, omega, beta, features,
                 maximum, clip, softmax, base_counts=None, base_sums=None):
    """Shared concept model for a batch of rows or the next causal token.

    Targets can contribute to the training summaries, but are excluded from
    routing by ``features``. Prefix callers supply only previously read data.
    All operations retain PyTorch's graph and also work on NumPy arrays.
    """
    relevance = clip(abs(anchors-root) / denominator * omega, 0, 1)
    assignments, weights, scores = concept_assignments(x,anchors,relevance,beta,features,softmax)
    counts = assignments.sum(axis=1, keepdims=True)
    sums = assignments@x
    if base_counts is not None:
        counts = counts + base_counts
        sums = sums + base_sums
    mass = maximum(counts, 1e-12)
    centroid = sums / mass
    effective = .5*(centroid+anchors)
    final_relevance = clip(abs(effective-root) / denominator * omega, 0, 1)
    frequencies = counts / maximum(counts.sum(), 1e-12)
    normalized_cp = normalized_cluster_probability(frequencies, final_relevance, maximum)
    return dict(T=assignments, G=centroid, R=final_relevance, F=frequencies.T,
                C=counts, sums=sums, normalized_cp=normalized_cp,
                routing_relevance=relevance, weights=weights, scores=scores)
