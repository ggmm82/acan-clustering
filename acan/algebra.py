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


def simplex_projection(values):
    """Euclidean projection onto nonnegative masses summing to one.

    Zero support is a constrained optimum, not a pruning tolerance.
    """
    if hasattr(values, 'detach'):
        import torch
        ordered = values.sort(descending=True).values
        ranks = torch.arange(1, len(values)+1, device=values.device, dtype=values.dtype)
        cumulative = ordered.cumsum(0)-1
        count = (ordered-cumulative/ranks > 0).sum()
        return (values-cumulative[count-1]/count).clamp_min(0)
    import numpy as np
    ordered = np.sort(values)[::-1]
    ranks = np.arange(1, len(values)+1)
    cumulative = ordered.cumsum()-1
    count = (ordered-cumulative/ranks > 0).sum()
    return np.maximum(values-cumulative[count-1]/count, 0)


def simplex_coordinate_candidate(values, index, partner_scores=None):
    """Endpoint of one pairwise coordinate of the simplex.

    A constrained optimizer must examine its boundary, including exact zeros.
    This proposes a parameter step only; the training objective decides whether
    to accept it. No similarity threshold or requested support size is involved.
    """
    if values[index] <= 0 or (values>0).sum() <= 1:
        return None
    candidate = values.clone() if hasattr(values,'detach') else values.copy()
    source = values if partner_scores is None else partner_scores
    partners = source.clone() if hasattr(source,'detach') else source.copy()
    partners[values<=0] = -1
    partners[index] = -1
    partner = int(partners.argmax())
    candidate[partner] += candidate[index]
    candidate[index] = 0
    return candidate


def simplex_coordinate_index(values, iteration):
    """Cycle over the current active face instead of spending steps on zeros."""
    if hasattr(values,'detach'):
        indices = (values>0).nonzero().reshape(-1)
    else:
        import numpy as np
        indices = np.flatnonzero(values>0)
    return int(indices[iteration % len(indices)]) if len(indices)>1 else None


def mass_assignments(base, masses, maximum):
    """Posterior assignment with learned simplex masses, including exact zeros."""
    base = maximum(base, 1e-300)
    weighted = base*masses.reshape(-1, 1)
    return weighted/maximum(weighted.sum(axis=0, keepdims=True), 1e-300), base


def exclusive_cluster_probability(frequencies, relevance, anchors, weights,
                                  features, beta, maximum, exponential, distinct, masses):
    """CP evidence minus evidence shared by distinct overlapping slots.

    With quality q_i=||R_i||²/sum(R_i), z_i=F_i sqrt(q_i), and
    S_ij=exp(-beta*(D_ij+D_ji)/2) for i != j, V=sum(F_i q_i)-z' S z.
    Since 0<=S_ij<=1 and sum(F)=1, 0<=V<=sum(F_i q_i)<=1.
    Identical copies have V=q sum(F_i²), so duplication is no longer neutral.
    """
    sums = maximum(relevance.sum(axis=1, keepdims=True), 1e-12)
    quality = (relevance*relevance).sum(axis=1, keepdims=True)/sums
    a = anchors[:, features]
    distances = ((weights*a*a).sum(axis=1, keepdims=True)
                 - 2*(weights*a)@a.T + weights@(a*a).T)
    symmetric = maximum((distances+distances.T)/2, 0)
    overlap = exponential(-beta*symmetric)*distinct
    # Linear continuation at zero has a finite derivative and roots²<=quality.
    # In particular an irrelevant channel cannot create overlap evidence.
    roots = quality/maximum(quality, 1e-12)**.5
    z = frequencies*roots
    value = (frequencies*quality).sum()-(z.T@overlap@z).sum()
    distinct_cp = maximum(value, 1e-12)**.5
    # Probability fidelity calibrates the learned prior against observed usage.
    # It is the Bhattacharyya affinity, bounded by one on the simplex. Without
    # this factor an almost-zero prior can represent a frequent concept when
    # distances are sharp, making structural optimization poorly conditioned.
    prior = masses.reshape(-1,1)
    prior_root = prior/maximum(prior,1e-300)**.5
    frequency_root = frequencies/maximum(frequencies,1e-300)**.5
    affinity = (prior_root*frequency_root).sum()
    return dict(normalized_cp=distinct_cp*affinity,distinct_cp=distinct_cp,
                probability_affinity=affinity,prior_root=prior_root,frequency_root=frequency_root,
                quality=quality, overlap=overlap, pair_distances=distances,
                pair_symmetric=symmetric, quality_roots=roots, z=z,
                exclusive_value=value)


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
                 maximum, clip, softmax, base_counts=None, base_sums=None,
                 masses=None, exponential=None, distinct=None, compute_cp=True):
    """Shared concept model for a batch of rows or the next causal token.

    Targets can contribute to the training summaries, but are excluded from
    routing by ``features``. Prefix callers supply only previously read data.
    All operations retain PyTorch's graph and also work on NumPy arrays.
    """
    relevance = clip(abs(anchors-root) / denominator * omega, 0, 1)
    assignments, weights, scores = concept_assignments(x,anchors,relevance,beta,features,softmax)
    base_assignments = assignments
    if masses is not None:
        assignments, base_assignments = mass_assignments(assignments,masses,maximum)
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
    normalized_cp = normalized_cluster_probability(frequencies, final_relevance, maximum) if compute_cp else None
    result = dict(T=assignments, G=centroid, R=final_relevance, F=frequencies.T,
                C=counts, sums=sums, normalized_cp=normalized_cp,
                routing_relevance=relevance, weights=weights, scores=scores)
    if masses is not None and compute_cp:
        result.update(exclusive_cluster_probability(frequencies,final_relevance,
            anchors,weights,features,beta,maximum,exponential,distinct,masses))
        result['base_assignments'] = base_assignments
    return result
