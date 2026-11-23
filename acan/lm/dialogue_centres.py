"""Initialize token concepts from the joint counts of observed requests and responses."""
import numpy as np


def markers(tokenizer):
    start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if (None in (start, end) or start == end or tokenizer.unk_token_id in (start, end)
            or tokenizer.eos_token_id in (start, end)):
        return None
    user, assistant = [tuple(tokenizer.encode(f'<|im_start|>{role}\n', add_special_tokens=False))
                       for role in ('user', 'assistant')]
    if not user or not assistant or user[0] != start or assistant[0] != start or user == assistant:
        return None
    return start, end, user, assistant


def _pairs(ids, eos, headers):
    start, end, user, assistant = headers
    boundaries = np.flatnonzero((ids == start) | (ids == end) | (ids == eos))
    request = None
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if ids[left] == eos:
            request = None
        if ids[left] != start or ids[right] != end:
            continue
        turn = ids[left:right]
        if np.array_equal(turn[:len(user)], user):
            request = np.unique(turn[len(user):])
        elif np.array_equal(turn[:len(assistant)], assistant):
            if request is not None and request.size and len(turn) > len(assistant):
                yield request, np.unique(turn[len(assistant):])
            request = None


def paired_centres(ids, eos, headers, vocabulary, dimensions, budget, check):
    """Factor the positive log association of input and target tokens, with one vote per turn pair."""
    pairs = []
    for request, response in _pairs(ids, eos, headers):
        if len(pairs) % 256 == 0:
            check()
        pairs.append((request, response))
    if not pairs or vocabulary < 2:
        return None
    sizes = np.asarray([len(request) * len(response) for request, response in pairs], np.int64)
    # Bound sparse construction, coalescing and the temporary arrays used to normalize entries.
    capacity = budget // 64
    if sizes.sum() > capacity:
        selected = []
        for i in np.random.default_rng(0).permutation(len(pairs)):
            if sizes[i] <= capacity:
                selected.append(pairs[i])
                capacity -= int(sizes[i])
        pairs = selected
    size = sum(len(request) * len(response) for request, response in pairs)
    if not size:
        return None
    from scipy.sparse import coo_matrix
    from sklearn.decomposition import TruncatedSVD

    rows, columns = np.empty(size, np.int32), np.empty(size, np.int32)
    offset = 0
    for i, (request, response) in enumerate(pairs):
        if i % 256 == 0:
            check()
        shape = len(request), len(response)
        stop = offset + shape[0] * shape[1]
        rows[offset:stop].reshape(shape)[:] = request[:, None]
        columns[offset:stop].reshape(shape)[:] = response[None, :]
        offset = stop
    joint = coo_matrix((np.ones(size), (rows, columns)), shape=(vocabulary, vocabulary)).tocsr().tocoo()
    del rows, columns
    observed_input = np.asarray(joint.sum(axis=1)).ravel()
    observed_response = np.asarray(joint.sum(axis=0)).ravel()
    expected = observed_input[joint.row] * observed_response[joint.col]
    joint.data = np.maximum(np.log(joint.data * joint.data.sum() / expected), 0.)
    joint.eliminate_zeros()
    if not joint.nnz:
        return None
    check()
    factor = TruncatedSVD(n_components=min(dimensions, vocabulary - 1), random_state=0)
    left = factor.fit_transform(joint.tocsr())
    check()
    scale = np.sqrt(factor.singular_values_).clip(1e-9)
    return (left / scale + factor.components_.T * scale) * .5
