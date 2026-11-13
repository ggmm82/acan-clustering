"""Numerical evidence for docs/MATHEMATICAL-FOUNDATION.md.

Every check calls the production operator in acan/algebra.py (NumPy backend) or
states a closed-form identity. Nothing here trains a model or reads user data.
Run: python scripts/verify_math_foundation.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acan.algebra import concept_step  # noqa: E402


def softmax(scores):
    scores = scores - scores.max(axis=0, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=0, keepdims=True)


def cp_star(x, anchors, beta=64., masses=None, omega=None):
    """Current CP_* exactly as the tabular/LM adapters call it."""
    lo, hi, root = x.min(0), x.max(0), x.mean(0, keepdims=True)
    den = (.5*(hi-lo) + np.abs(root - .5*(lo+hi)))
    valid = den > 1e-8
    den = np.where(valid, den, 1.)
    omega = (np.ones_like(den) if omega is None else omega) * valid
    k = len(anchors)
    masses = np.full(k, 1/k) if masses is None else masses
    out = concept_step(x, anchors, root, den, omega, beta, slice(None), np.maximum, np.clip,
                       softmax, masses=masses, exponential=np.exp, distinct=1-np.eye(k))
    return float(out['normalized_cp']), out


def contraharmonic(r):
    r = np.asarray(r, float)
    return float((r*r).sum()/r.sum())


def explained_variance(x, t):
    """Proposed index: per-attribute correlation ratio eta^2_j (law of total variance)."""
    c = t.sum(1, keepdims=True)
    g = t @ x / c
    f = c / c.sum()
    mu = (f * g).sum(0)                     # equals the t-weighted global mean
    between = (f * (g-mu)**2).sum(0)
    total = (t.T[:, :, None] * (x[:, None, :]-mu)**2).sum((0, 1)) / c.sum()
    return between / np.maximum(total, 1e-300)


def adjusted(eta, n, k):
    """Bias-corrected explained variance (epsilon^2): zero in expectation without structure."""
    return 1 - (1-eta) * (n-1) / max(n-k, 1)


def hard(labels, k):
    return np.eye(k)[labels].T


def main():
    rng = np.random.default_rng(0)
    report = {}

    # 1. q = sum R^2 / sum R is not monotone and is maximal for a single extreme attribute.
    report['q_single_attribute'] = contraharmonic([1, 0, 0, 0])
    report['q_two_attributes_0.9_0.3'] = contraharmonic([.9, .3])
    report['q_one_attribute_0.9'] = contraharmonic([.9])
    report['q_all_four_attributes_0.8'] = contraharmonic([.8]*4)
    assert contraharmonic([.9, .3]) < contraharmonic([.9])           # more evidence, lower quality
    assert contraharmonic([1, 0, 0, 0]) > contraharmonic([.8]*4)     # one channel beats four

    # 2. With sharp assignments CP ignores within-concept dispersion: same centroids,
    #    very different spread. (Soft assignments react only by blurring the centroids.)
    centers = np.array([[.25, .25], [.75, .75]])
    labels = np.repeat([0, 1], 200)
    tight = centers[labels] + rng.normal(0, .01, (400, 2))
    loose = np.clip(centers[labels] + rng.normal(0, .20, (400, 2)), 0, 1)
    for name, data in (('tight', tight), ('loose', loose)):
        data = np.vstack([data, [[0, 0], [1, 1]]])      # same observed range in both sets
        report[f'cp_{name}'] = cp_star(data, centers, beta=4096.)[0]
        t = hard(cp_star(data, centers, beta=4096.)[1]['T'].argmax(0), 2)   # the same partition CP scores
        report[f'eta2_{name}'] = explained_variance(data, t).mean()
    assert abs(report['cp_tight']-report['cp_loose']) < .05
    assert report['eta2_tight'] - report['eta2_loose'] > .25

    # 3. Prototype inflation: same data, same hard partition, prototypes moved off the data.
    data = np.vstack([tight, [[0, 0], [1, 1]]])
    honest, out_h = cp_star(data, centers, beta=64.)
    inflated, out_i = cp_star(data, np.array([[0., 0.], [1., 1.]]), beta=64.)
    same_partition = bool((out_h['T'].argmax(0) == out_i['T'].argmax(0)).all())
    report.update(cp_prototypes_on_centroids=honest, cp_prototypes_off_data=inflated,
                  same_partition=same_partition)
    assert same_partition and inflated > honest + .1

    # 4. Duplicate penalty is evaded by sharpening beta: split ONE homogeneous blob in two.
    blob = np.vstack([.5 + rng.normal(0, .05, (400, 2)), [[0, 0], [1, 1]]])
    halves = np.array([[.47, .5], [.53, .5]])
    one = np.array([[.5, .5]])
    report['split_blob'] = {str(b): dict(one=cp_star(blob, one, beta=b)[0], two=cp_star(blob, halves, beta=b)[0],
                                         overlap=float(cp_star(blob, halves, beta=b)[1]['overlap'][0, 1]))
                            for b in (1., 64., 4096.)}
    assert report['split_blob']['4096.0']['overlap'] < 1e-3
    assert report['split_blob']['4096.0']['two'] > report['split_blob']['4096.0']['one']

    # 5. Range normalization: one outlier changes the relevance of every concept on that attribute.
    base = np.vstack([tight, [[0, 0], [1, 1]]])
    with_outlier = np.vstack([base, [[10., .5]]])
    r_clean = cp_star(base, centers)[1]['R'][0, 0]
    r_dirty = cp_star(with_outlier, centers)[1]['R'][0, 0]
    report.update(relevance_clean=float(r_clean), relevance_one_outlier=float(r_dirty))
    assert r_dirty < .2 * r_clean
    eta_clean = explained_variance(base, hard(np.r_[labels, 0, 1], 2))[1]
    eta_dirty = explained_variance(with_outlier, hard(np.r_[labels, 0, 1, 0], 2))[1]
    report.update(eta2_other_attribute_clean=float(eta_clean), eta2_other_attribute_outlier=float(eta_dirty))
    assert abs(eta_clean-eta_dirty) < .02   # per-attribute index: untouched attribute is unaffected

    # 6. Identity: on one-hot attributes the between-concept scatter is K times Category Utility.
    n, k = 600, 3
    z = rng.integers(0, k, n)
    cats = (z + (rng.random(n) < .3) * rng.integers(0, 4, n)) % 4
    onehot = np.eye(4)[cats]
    t = hard(z, k)
    f = t.sum(1)/n
    p_iv = t @ onehot / t.sum(1, keepdims=True)
    p_v = onehot.mean(0)
    scatter = (f[:, None]*(p_iv-p_v)**2).sum()
    category_utility = (f[:, None]*(p_iv**2 - p_v**2)).sum()/k
    report.update(between_scatter=float(scatter), k_times_category_utility=float(k*category_utility))
    assert abs(scatter - k*category_utility) < 1e-12

    # 7. Irrelevant attributes: explained variance has null mean (K-1)/(n-1), cancelled
    #    exactly by the adjustment (averaged over 100 independent draws).
    n, k, m_irr = 400, 4, 48
    relevant, irrelevant, corrected = [], [], []
    for _ in range(100):
        z = rng.integers(0, k, n)
        rel = np.eye(k)[z][:, :2]*.6 + rng.normal(0, .03, (n, 2))
        eta = explained_variance(np.hstack([rel, rng.random((n, m_irr))]), hard(z, k))
        relevant.append(eta[:2].mean())
        irrelevant.append(eta[2:].mean())
        corrected.append(np.mean([adjusted(e, n, k) for e in eta[2:]]))
    report.update(eta2_relevant=float(np.mean(relevant)), eta2_irrelevant=float(np.mean(irrelevant)),
                  eta2_irrelevant_adjusted=float(np.mean(corrected)), null_expectation=(k-1)/(n-1))
    assert abs(report['eta2_irrelevant']-report['null_expectation']) < 5e-4
    assert abs(report['eta2_irrelevant_adjusted']) < 5e-4 and report['eta2_relevant'] > .85

    # 8. The causal concept memory is a rank-k attention kernel (any T, any data).
    length, k = 256, 8
    t = softmax(rng.normal(0, 3, (k, length)))
    c = t.cumsum(1)                                            # C_{i,t}: prefix masses
    kernel = np.einsum('it,it,is->ts', t, 1/c, t) * np.tril(np.ones((length, length)))
    rank_full = np.linalg.matrix_rank(np.einsum('it,is->ts', t, t))
    values = rng.normal(size=(length, 5))
    recurrent = np.einsum('it,itd->td', t, (t[:, :, None]*values[None]).cumsum(1)/c[:, :, None])
    report.update(kernel_rank=int(rank_full), kernel_rows_sum_to_one=bool(np.allclose(kernel.sum(1), 1)),
                  kernel_equals_recurrent_state=bool(np.allclose(kernel @ values, recurrent)),
                  weight_of_a_token_seen_once=float(kernel[-1].max()))
    assert rank_full <= k and report['kernel_equals_recurrent_state']

    # 9. Dense softmax attention aggregated over key clusters. For keys within radius rho of
    #    their centroid: ||dense - clustered|| <= (exp(2*|q|*rho)-1)*max|v|. The multiplicity
    #    enters as log C_i with coefficient exactly one; the Gaussian term 0.5*q'Vq is the
    #    second-order correction (moment generating function of the keys of a slot).
    length, k, d = 2048, 16, 16
    slot = rng.integers(0, k, length)
    slot[: length//2] = 0                                   # unbalanced masses on purpose
    centroids_true = rng.normal(0, 1, (k, d))
    keys = centroids_true[slot] + rng.normal(0, .15, (length, d))
    values = rng.normal(0, 1, (k, 8))[slot] + rng.normal(0, .3, (length, 8))
    t = hard(slot, k)
    c = t.sum(1)
    g = t @ keys / c[:, None]
    var = t @ keys**2 / c[:, None] - g**2
    mean_value = t @ values / c[:, None]
    rho = np.linalg.norm(keys - g[slot], axis=1).max()
    errors = dict(log_mass=[], gaussian=[], no_mass=[])
    bound_holds = True
    for _ in range(200):
        q = rng.normal(0, 1, d)
        q *= rng.uniform(.5, 2.) / np.linalg.norm(q)
        dense = softmax((keys @ q)[:, None])[:, 0] @ values
        for name, logits in (('log_mass', g @ q + np.log(c)),
                             ('gaussian', g @ q + .5*(var @ q**2) + np.log(c)),
                             ('no_mass', g @ q)):
            errors[name].append(np.linalg.norm(softmax(logits[:, None])[:, 0] @ mean_value - dense))
        limit = (np.exp(2*np.linalg.norm(q)*rho)-1) * np.linalg.norm(values, axis=1).max()
        bound_holds &= errors['log_mass'][-1] <= limit
    report['clustered_attention'] = dict(bound_holds=bool(bound_holds),
        **{f'mean_error_{name}': float(np.mean(v)) for name, v in errors.items()},
        mean_output_norm=float(np.linalg.norm(values, axis=1).mean()),
        state_fraction_of_dense=k/length)
    assert bound_holds
    assert np.mean(errors['gaussian']) < np.mean(errors['log_mass']) < .5*np.mean(errors['no_mass'])

    # 10. Pair form of the concept criterion. With s_tu = 1 - mean_j d_tuj^2/(2*var_j) (unbiased
    #     variances) the criterion is the mean over ordered pairs of [same concept]*s_tu. It equals
    #     the form computed from the additive statistics, it is zero for one concept and for
    #     singletons, and along average linkage it rises while the merged concepts are more similar
    #     than the average pair, then falls: one interior maximum, no threshold.
    from scipy.cluster.hierarchy import fcluster, linkage
    n, m = 90, 3
    x = np.repeat(rng.normal(0, 1, (5, m)), n//5, 0)+rng.normal(0, .25, (n, m))
    y = x/np.sqrt(2*m*x.var(0, ddof=1))
    distance = ((y[:, None]-y[None])**2).sum(-1)
    similarity = 1-distance
    np.fill_diagonal(similarity, 0)
    assert abs(similarity.sum()) < 1e-8                      # no structure assumed: similarities are centred
    steps = linkage(y, 'average', metric='sqeuclidean')
    heights = steps[:, 2]
    assert (np.diff(heights) > -1e-12).all()                 # merges come in order of decreasing similarity

    def pair_criterion(labels):
        same = labels[:, None] == labels[None]
        return (same*similarity).sum()/(n*(n-1))

    def statistics_criterion(labels):
        value = 0.
        for label in np.unique(labels):
            members = x[labels == label]
            c = len(members)
            if c > 1:
                value += c*(c-1)*(1-(members.var(0, ddof=1)/x.var(0, ddof=1)).mean())
        return value/(n*(n-1))

    curve = []
    for k in range(1, n+1):
        labels = fcluster(steps, k, 'maxclust')
        curve.append(pair_criterion(labels))
        assert abs(curve[-1]-statistics_criterion(labels)) < 1e-10
    assert abs(curve[0]) < 1e-10 and abs(curve[-1]) < 1e-10
    best = int(np.argmax(curve))+1
    rising = np.diff(curve[best-1:])                         # more concepts than the optimum
    falling = np.diff(curve[:best])
    assert (rising < 1e-12).all() and (falling > -1e-12).all()
    # Resolution: two tight groups closer than the average pair are merged at one level.
    report['pair_criterion'] = dict(concepts_at_maximum=best, maximum=float(max(curve)), true_groups=5,
                                    merges_above_average=int((heights < 1).sum()))

    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
