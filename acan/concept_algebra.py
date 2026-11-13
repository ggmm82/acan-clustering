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

2. Generalization.  Two concepts are generalized by adding their statistics.
   ``description_length`` is the code length of a concept; the cost of a merge is
   the increase of the total length. Copies of one concept always have negative cost.

3. Judgment.  A concept is evaluated only on observations it has not absorbed:
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
