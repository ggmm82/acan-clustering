"""The single concept model, for NumPy arrays and PyTorch tensors alike.

A concept is a set of additive statistics over its members z:

    C = sum of weights,   U = sum of z,   Q = sum of z*z,   (+ whatever is read out)

Three operators define the model. Table rows and attention keys use the same ones.

1. Membership.  The weight of concept i for an observation is its mass times
   the expected similarity between the observation and the members of i:

       T_i(obs)  ∝  C_i · E_{z ~ concept i}[ κ(obs, z) ]

   * κ = identity (a table row is observed exactly): the expectation is the
     Gaussian density of the concept, ``exact_similarity``.
   * κ = exp(s·q·z) (dot-product attention): the expectation is the moment
     generating function of the concept, ``dot_similarity``; with one concept
     per key it is exactly dense softmax attention.

2. Learning.  Both tables and concept networks update concepts by the complete-data
   Fisher natural gradient of the log-likelihood, ``natural_em_step``:

       S_i'  =  (1 - step) · S_i  +  step · batch_S_i

   As step -> 0 its derivative on (log C, g, log s) is F_c^{-1} grad L; at step = 1
   (fixed table rows) it is the exact M-step of batch EM in a single jump.

3. Generalization.  Two concepts are generalized by adding their statistics.
   ``description_length`` is the code length of a concept; the cost of a merge is
   the increase of the total length. Copies of one concept always have negative cost.

4. Judgment.  A concept is evaluated only on observations it has not absorbed:
   statistics are additive, so a row is subtracted before it is predicted, and
   a causal prefix never contains the token it predicts.

Levels. Concepts form nested levels: the statistics of a concept are the sum of
those of its children, and each level is the prior of the next
(``centre_variance`` with the parent's variance as ``root``). At one level the
pair criterion

    mean over ordered pairs (t,u) of [same concept] * s_tu,   sum of all s_tu = 0,

equals sum_i pair_mass(C_i) * relevance_i / (n(n-1)) with relevance measured
against the level above. It is zero for one concept and for singletons.

``xp`` is ``numpy`` or ``torch``.
"""
import math

LOG_2PI = math.log(2*math.pi)


def centre_variance(c, u, q, root, kappa, xp):
    """Centre and variance of each concept; the variance shrinks to ``root`` with ``kappa`` pseudo-members.

    kappa > 0 is the predictive variance used to judge new observations.
    kappa = 0 is the empirical variance of the stored members.
    """
    g = u/xp.clip(c, 1e-12, None)
    return g, (xp.clip(q-c*g*g, 0, None)+kappa*root)/(c+kappa)


def description_length(c_attribute, variance, c, xp, weights=1.):
    """Gaussian code length of the members' attributes minus the code length saved by a larger mass."""
    return .5*(c_attribute*xp.log(variance)*weights).sum(-1)-c*xp.log(xp.clip(c, 1e-30, None))


def exact_similarity(x, g, variance, xp, observed=1., weights=1.):
    """log E[κ] for exactly observed attributes: the Gaussian log-density, summed over observed attributes."""
    return (-.5*((x-g)**2/variance+xp.log(variance)+LOG_2PI)*observed*weights).sum(-1)


def dot_similarity(q, g, scale, xp, variance=None):
    """log E[κ] for κ = exp(scale·q·z): scale·q·g, plus the second-order term when a variance is given."""
    value = scale*(q*g).sum(-1)
    if variance is not None:
        value = value+.5*scale*scale*(q*q*variance).sum(-1)
    return value


def pair_mass(c, xp):
    """Number of ordered pairs of distinct members: a single member supports nothing."""
    return c*xp.clip(c-1, 0, None)


def unbiased(variance, c, xp):
    """Variance estimated from pairs of distinct members (population variance times c/(c-1))."""
    return variance*c/xp.clip(c-1, 1e-12, None)


def cross_similarity(ga, va, gb, vb, scale, xp):
    """Mean similarity between members of two concepts, centred on the average pair of the reference.

    1 - mean_j [(ga-gb)^2 + va + vb] / (2*scale_j), with population variances va, vb and
    ``scale`` the pair variance of the reference level. Positive: the two concepts are
    closer than an average pair, and adding them together raises the pair criterion.
    """
    return 1-(((ga-gb)**2+va+vb)/(2*scale)).mean(-1)


def reader_radius(recognition, xp):
    """Radius of a concept for whoever reads it.

    ``recognition`` is the mean squared distance of the members from the centre in the
    part that is matched (attributes of a row, key of a token), on the scale of the root.
    It is the quantity that bounds the error of reading the concept instead of its members.
    """
    return xp.sqrt(recognition)


def reader_cost(reads_first, radius_first, reads_second, radius_second, radius_together):
    """Loss of the reader when two concepts are added together: reads received times radius.

    The error of reading a concept instead of its members is bounded by the reads it
    receives times the distance of its members from the centre. The cost is never
    negative and is zero only for two copies of one concept.
    """
    return (reads_first+reads_second)*radius_together-reads_first*radius_first-reads_second*radius_second


KAPPA_GRID = tuple(2.**k for k in range(-5, 9))                  # weights of the level above tried by the judgment


def categorical_reading(counts, members, distinct, prior, kappa, xp):
    """Frequencies of a categorical attribute in a concept, read with its level above as reference.

    p(v) = (N(v) + w p_above(v)) / (C + w),   w = kappa * max(D, 1),

    with N the members of the concept with value v, C its members, D the values it has. A concept with
    few members leans on its level above; one whose members take many different values leans on it more.
    The same rule reads a category of a table and the token that follows a context of a text. With D = 1
    it is the rule with kappa pseudo-members (the reading while the concepts are searched).
    """
    weight = kappa*xp.maximum(distinct, 1)
    return (counts+weight*prior)/(members+weight)


def natural_em_step(c, u, q, batch_c, batch_u, batch_q, step, xp):
    """Single learning rule of the concept model on additive statistics: Fisher natural gradient = EM.

    For exponential-family concepts with posterior memberships T_tk, multiplying the gradient of the
    log-likelihood L = (1/n) sum_t log sum_k pi_k L_k(z_t; theta_k) by the inverse complete-data Fisher
    information matrix yields the expectation-space direction (batch_S - S). A step ``step`` in [0, 1] gives:

        S' = (1 - step) * S + step * batch_S.

    * ``step == 1`` (tabular EM on fixed rows): exact batch M-step in a single jump.
    * ``0 < step < 1`` (streaming pieces or deep layers whose inputs move): online natural EM.
    """
    if step >= 1.:
        return batch_c, batch_u, batch_q
    keep = 1. - step
    return keep*c + step*batch_c, keep*u + step*batch_u, keep*q + step*batch_q


def natural_em_dict(current, batch, step):
    """Apply the unified ``(1 - step) * S + step * batch_S`` update across a dictionary of additive statistics."""
    if current is None or step >= 1.:
        return batch
    keep = 1. - step
    return {key: keep*current[key] + step*batch[key] for key in batch}


def natural_em_parameters(log_c, g, log_s, batch_c, batch_u, batch_q, root, kappa, step, xp):
    """Apply ``natural_em_step`` directly to Gaussian concept parameters (log_c, g, log_s).

    Maps (log_c, g, log_s) to expectation statistics on the batch reference mass, takes a ``natural_em_step``
    of size ``step``, and projects back via ``centre_variance`` with prior ``root`` and strength ``kappa``.
    At ``step == 1`` it equals the tabular M-step; as ``step -> 0`` with ``kappa == 0`` its derivative equals
    the complete-data Fisher natural gradient F_c^{-1} grad L on every coordinate (log_c, g, log_s).
    """
    total = xp.clip(batch_c.sum(-1)[..., None], 1e-12, None)
    shifted = log_c - xp.amax(log_c, -1)[..., None]
    pi = xp.exp(shifted)
    pi = pi / xp.clip(pi.sum(-1)[..., None], 1e-30, None)
    old_c = total * pi
    old_u = old_c[..., None] * g
    old_q = old_c[..., None] * (g*g + xp.exp(2*log_s))
    new_c, new_u, new_q = natural_em_step(old_c, old_u, old_q, batch_c, batch_u, batch_q, step, xp)
    new_g, new_v = centre_variance(new_c[..., None], new_u, new_q, root, kappa, xp)
    new_pi = xp.clip(new_c, 1e-30, None) / xp.clip(new_c.sum(-1)[..., None], 1e-30, None)
    return xp.log(new_pi), new_g, .5*xp.log(xp.clip(new_v, 1e-12, None))

