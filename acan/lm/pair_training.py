"""Fit native numeric pair concepts and judge categorical descendants on separate documents."""

from copy import copy

import numpy as np
import pandas as pd

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table
from acan.core.concepts import ConceptModel
from acan.core.conditional import calibrated_weights
from acan.concept_algebra import categorical_reading
from .pair_model import EncodedPrefix, PairEncoder, PairLanguageModel, shift, span_means, topic_codes


CONTEXT = 512                                                    # longest scale tried by default (concept_context sets more)


def enough_rows(fit, available, score, first=2048):
    """Fit on a drawn sample that doubles while it makes the next token more predictable.

    The concept model finds its concepts on ``refine_rows`` drawn rows and judges every candidate cut on
    ``judged_rows`` of them, so its time grows with these rows. The rows are not fixed: the sample
    doubles while the concepts it finds predict the next token of held-out text better, by more than
    one standard error of the paired difference (the same judgment the model uses to choose its cuts).
    ``score(model)`` gives the log-probability of every held-out next token.
    """
    rows = min(first, available)
    model = fit(rows)
    best = score(model)
    while rows < available:
        rows = min(2*rows, available)
        bigger = fit(rows)
        trial = score(bigger)
        gain = trial-best
        if gain.mean() <= gain.std(ddof=1)/np.sqrt(max(len(gain)-1, 1)):
            break
        model, best = bigger, trial
    model.report = None                                         # the page follows the learning, not the model
    return model


def reading_score(train_keys, train_y, tune_keys, tune_y, vocabulary):
    """Held-out log-probability of the next token read from one categorical column (best smoothing)."""
    from acan.concept_algebra import KAPPA_GRID
    table = Table.count_regions([(train_keys, train_y)], vocabulary)
    mass, counts, distinct = table.lookup(tune_keys, tune_y)
    prior = (np.bincount(train_y, minlength=vocabulary)+1.)[tune_y]
    prior /= len(train_y)+vocabulary
    return max((np.log(categorical_reading(counts, mass, distinct, prior, k, np)) for k in KAPPA_GRID),
               key=lambda logp: logp.mean())


def concept_model(rows, progress=None):
    model = ConceptModel(max_concepts=256, bins=24, judged_rows=max(rows//2, 1), refine_rows=rows,
                         covariance=False, seed=42)
    model.report = progress                                     # how far the search of the levels is
    return model


def semantic_vectors(ids, vocabulary, dimensions=32, top=512, seed=0, starts=None):
    """Token coordinates: the neighbours on the left and right of each token, compressed by SVD."""
    from sklearn.utils.extmath import randomized_svd
    ids=np.asarray(ids,dtype=np.int64)
    frequency=np.bincount(ids,minlength=vocabulary)
    top=min(top,vocabulary)
    common=np.argsort(-frequency,kind='stable')[:top]
    where=np.full(vocabulary,top,dtype=int);where[common]=np.arange(top)
    left,right=ids[:-1],ids[1:]
    if starts is not None:
        valid=np.ones(len(left),bool)
        valid[np.asarray(starts)[1:]-1]=False
        left,right=left[valid],right[valid]
    before=np.bincount(right*(top+1)+where[left],minlength=vocabulary*(top+1)).reshape(vocabulary,top+1)
    after=np.bincount(left*(top+1)+where[right],minlength=vocabulary*(top+1)).reshape(vocabulary,top+1)
    profile=np.hstack([np.sqrt(a/np.maximum(a.sum(1,keepdims=True),1)) for a in (before,after)])
    center=profile[frequency>0].mean(0)
    _,singular,directions=randomized_svd((profile if starts is None else profile[frequency>0])-center,n_components=min(dimensions,profile.shape[1]),n_iter=5,random_state=seed)
    return ((profile-center)@directions.T).astype(np.float32),singular


def topic_vectors(ids, starts, vocabulary, dimensions, seed=42):
    """Tokens described by the documents they appear in: counts token x document, log-scaled, weighted so
    that a token found in few documents says more, compressed by SVD and normalized (latent semantic
    analysis). Two tokens are close when they are used about the same things, wherever they stand."""
    from scipy import sparse
    from sklearn.utils.extmath import randomized_svd
    ids = np.asarray(ids, np.int64)
    position = np.arange(len(ids))
    document = np.searchsorted(np.asarray(starts), position, side="right") - 1
    counts = sparse.coo_matrix((np.ones(len(ids)), (ids, document)), shape=(vocabulary, document[-1] + 1)).tocsr()
    counts.data = np.log1p(counts.data)
    spread = np.asarray((counts > 0).sum(1)).ravel()
    counts = sparse.diags(np.log((document[-1] + 1) / (1 + spread))) @ counts
    rank = min(dimensions, min(counts.shape) - 1)
    u, singular, _ = randomized_svd(counts, n_components=max(rank, 1), n_iter=5, random_state=seed)
    vectors = u * singular
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def geometry(splits, vectors, width, progress=None):
    half = width // 2

    def keys(row):
        child = row.spans["e1" if width == 2 else f"c{half}"].astype(np.uint32, copy=False)
        older = np.zeros_like(child)
        older[half:] = child[:-half]
        older[row.age < width - 1] = 0
        return CategoryPairs.keys(older, child)

    def frame(key):
        x = np.column_stack((vectors[(key >> 32).astype(int)], vectors[(key & 0xFFFFFFFF).astype(int)]))
        return pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])

    packed = keys(splits["train"])
    types, counts = np.unique(packed[packed > 0], return_counts=True)
    x = frame(types)
    vocabulary = int(max(int(row.y.max()) for row in splits.values()))+1

    def assign(graph):
        found = {}
        for name, row in splits.items():
            packed = keys(row)
            typ, inv = np.unique(packed, return_inverse=True)
            valid = np.flatnonzero(typ > 0)
            codes = np.zeros(len(typ), np.uint32)
            for idx in np.array_split(valid, max(1, (len(valid) + 1023) // 1024)):
                if len(idx):
                    codes[idx] = graph.assignments(frame(typ[idx])).argmax(1) + 1
            found[name] = codes[inv], codes[valid]
        return found

    def score(graph):
        found = assign(graph)
        graph._found = found
        rows = {name: copy(row) for name, row in splits.items()}
        for name, row in rows.items():
            row.spans = dict(row.spans, **{f"c{width}": found[name][0]})
        return reading_score(rows["train"].feature(f"c{width}"), rows["train"].y,
                             rows["tune"].feature(f"c{width}"), rows["tune"].y, vocabulary)

    graph = enough_rows(lambda rows: concept_model(rows, progress).fit_counted_numeric(x, counts), int(counts.sum()), score)
    found = graph._found
    del graph._found
    for name, row in splits.items():
        row.spans[f"c{width}"] = found[name][0]
    train_codes = found["train"][1]
    mass = np.bincount(train_codes, weights=counts, minlength=graph.concepts + 1)
    means = np.zeros((len(mass), x.shape[1]))
    for j, col in enumerate(x.to_numpy().T):
        means[:, j] = np.bincount(train_codes, weights=counts * col, minlength=len(mass)) / np.maximum(
            mass, 1
        )
    if means.shape[1] > 32:
        centre = np.average(means, axis=0, weights=mass)
        _, basis = np.linalg.eigh(((means - centre) * mass[:, None]).T @ (means - centre))
        means = (means - centre) @ basis[:, -32:]
    return graph, means, dict(types=len(types), occurrences=int(counts.sum()), concepts=graph.concepts)


def finer_spans(splits, reading, width, size, routes=None):
    half = width // 2

    def halves(row, older, recent):
        left = np.zeros(len(row.y), np.uint32)
        left[half:] = row.spans[older][:-half]
        left[row.age < width - 1] = 0
        return left, row.spans[recent]

    tr = splits["train"]
    exact = f"e{half}"
    dictionary = CategoryPairs().fit(*halves(tr, exact, exact))
    for row in splits.values():
        row.spans[f"e{width}"] = dictionary.transform(*halves(row, exact, exact))
        row.spans[f"s{width}"] = row.spans[f"c{width}"].copy()
    judged = (np.cumsum(tr.age == 0) - 1) % 5 == 4
    reports = []
    pairs = [(f"c{half}", exact), (exact, f"s{width}")]
    if width >= 4:
        pairs.insert(0, (f"s{half}", f"s{half}"))
    for older, recent in pairs:
        l, r = halves(tr, older, recent)
        full = shift(r, 1, tr.age), shift(l, 1, tr.age), tr.y
        if not judged.any() or judged.all():
            break
        t, report = reading.subconcepts(
            tuple(a[~judged] for a in full), tuple(a[judged] for a in full), full
        )
        for row in splits.values():
            l, r = halves(row, older, recent)
            ix = t.find(CategoryPairs.keys(r, l))
            ok = ix >= 0
            row.spans[f"s{width}"][ok] = size + 1 + ix[ok]
        if routes is not None:
            enc = CategoryPairs()
            enc.observed = t.keys
            routes.append((older, "current" if recent == f"s{width}" else recent, size, enc))
        size += len(t.keys)
        reports.append(report)
    return reports


def stages(local=False):
    # The odd suffixes use a leaf and the dyadic suffix. Six uses two followed by four.
    for w, left, right in ((1, 0, 0), (2, 1, 1), (3, 1, 2), (4, 2, 2), (5, 1, 4), (6, 2, 4)):
        if w == 1:
            yield w, ["c1", "e1"]
            continue
        a, b = str(left), str(right)
        features = ["c" + str(w)] if w in (2, 4) else []
        features += ["c" + a + "|c" + b, "e" + a + "|c" + b, "c" + a + "|e" + b]
        if local:
            if w in (2, 4):
                features += ["s" + str(w)]
            for l in ("s" + a, "c" + a, "e" + a):
                for r in ("s" + b, "c" + b, "e" + b):
                    if "s" not in l + r or ("s1" in (l, r)):
                        continue
                    features.append(l + "|" + r)
        yield w, features + ["e" + str(w)]


def add_chunk(model, ids, starts, tail, tables, budget, pending=None):
    """Count a chunk of text into the tables of the selected readings.

    With ``pending`` the counts of the chunk are only queued: equal-sized parts are added two by two
    (a binary counter), and ``flush`` adds them to the large tables once. Adding a small part to a large
    table rewrites the large table, so doing it once per piece instead of once per chunk saves most of
    the work, with the same sums.
    """
    row = model.encoder.encode(np.r_[tail, ids], np.unique(np.r_[0, len(tail) + starts]).astype(int))
    begin = len(tail)
    strengths = getattr(model, "training_weights", model.weights)
    queue = {} if pending is None else pending
    for feature in strengths:
        key = row.feature(feature)[begin:]
        part = Table.count_regions([(key, ids.astype(np.int64, copy=False))], model.vocabulary)
        levels = queue.setdefault(feature, [])
        level = 0
        while level < len(levels) and levels[level] is not None:
            part = levels[level] + part
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(part)
        else:
            levels[level] = part
    if pending is None:
        flush(model, tables, queue, budget)
    age = int(row.age[-1]) + 1
    return row.y[-min(age, model.encoder.width) :].copy()


def flush(model, tables, pending, budget):
    """Add the queued counts to the tables and keep each table inside its share of the budget."""
    strengths = getattr(model, "training_weights", model.weights)
    topics = getattr(model, "topics", {})
    # Topic tables hold at most concepts x vocabulary cells: they are never pruned and take no share.
    total_weight = sum(k**-0.5 for f, k in strengths.items() if f not in topics)
    for feature, levels in pending.items():
        parts = [t for t in levels if t is not None]
        if not parts:
            continue
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        total = total if feature not in tables else tables[feature] + total
        # Each table inherits through its preceding readout when a rare key is removed.
        limit = np.inf if feature in topics else int(budget * strengths[feature] ** -0.5 / total_weight)
        if total.nbytes() > limit:
            sizes = 24 + 16 * total.D
            at = np.argsort(-total.C / (sizes + 0.0), kind="stable")
            keep = np.zeros(len(at), bool)
            keep[at[np.cumsum(sizes[at]) <= limit]] = True
            total = total.keep(keep)
        tables[feature] = total
    pending.clear()


def sample_documents(tokens, eos, limit=262144, block=4096, documents=None):
    """Sample contiguous training windows across the file, preserving all internal boundaries."""
    size = min(len(tokens), limit)
    chunks = min((size + block - 1) // block, max(1, len(tokens) // block))
    widths = np.diff(np.linspace(0, size, chunks + 1, dtype=int))
    starts = (
        np.r_[0, np.cumsum(widths)[:-1]]
        if size == len(tokens)
        else np.linspace(0, len(tokens) - widths[-1], chunks, dtype=int)
    )
    pieces = [np.asarray(tokens[i : i + w], np.int32) for i, w in zip(starts, widths)]
    ids = np.concatenate(pieces)
    boundaries = np.r_[0, np.cumsum(widths)[:-1], np.flatnonzero(ids[:-1] == eos) + 1]
    if documents is not None:
        for start, width, offset in zip(starts, widths, np.r_[0, np.cumsum(widths)[:-1]]):
            a, b = np.searchsorted(documents, [start, start + width])
            boundaries = np.r_[boundaries, documents[a:b] - start + offset]
    return ids, np.unique(boundaries)


def validation_rows(pieces, eos):
    ids = np.concatenate(pieces).astype(np.int32)
    return EncodedPrefix(
        ids,
        np.unique(
            np.r_[0, np.cumsum([len(p) for p in pieces])[:-1], np.flatnonzero(ids[:-1] == eos) + 1]
        ).astype(int),
    )


def learn(tokens, vocabulary, eos, tuning, check, documents=None, max_context=CONTEXT, report=None):
    """Discover the geometry on train and select predictors on held-out next-token likelihood.

    ``report(phase, width, seconds, **facts)`` is told when each scale starts and ends and, while the
    concepts of a scale are searched, how many are left (``concepts`` of ``start``), so the page can show
    how far the structure is before the model reads the text. ``step`` of ``steps`` counts the scales
    with concepts, the last one being the longest allowed: the validation may stop earlier.
    """
    import time
    say = report or (lambda *args, **facts: None)
    started = time.monotonic()
    planned = [1, *(1 << k for k in range(1, 21) if 1 << k <= max(max_context, 8))]
    where = dict(step=1, steps=len(planned) + 1)                # the scales, then the topics
    last, shown = [0.], [0.]

    def tell(phase, width, seconds, **facts):
        # The part done never goes back: the coordinates may double, the sample too, and a search start
        # again; the bar waits until the new search passes the point already shown.
        inside = facts.get('inside_permille', 1000 if phase in ('subconcepts', 'readings', 'done') else 0) / 1000
        shown[0] = max(shown[0], min((where['step'] - 1 + inside) / where['steps'], 1.))
        say(phase, width, seconds, **where, **facts, part_permille=int(1000 * shown[0]))

    def searching(width, phase=None, offset=0., span=1.):
        best = [0.]

        def progress(facts):
            now = time.monotonic()
            if facts.get('phase') == 'levels' and facts.get('start', 0) > 1:
                best[0] = max(best[0], 1-np.log(max(facts['concepts'], 1))/np.log(facts['start']))
                if now-last[0] >= 2:
                    last[0] = now
                    tell(phase or ('scale' if width > 1 else 'tokens'), width, now-started,
                         concepts=int(facts['concepts']), start=int(facts['start']),
                         inside_permille=int(1000*(offset + span*best[0])))
        return progress
    # A scale of w tokens is learned only from windows longer than w: the sample grows with the
    # longest scale allowed, and keeps its windows long enough to contain it.
    limit = max(int(__import__('os').environ.get('ACAN_PAIR_SAMPLE', 262144)), 4 * max_context)
    ids, starts = sample_documents(tokens, eos, limit=limit, block=max(4096, min(2 * max_context, limit)),
                                   documents=documents)
    tr = EncodedPrefix(ids, starts)
    va = validation_rows(tuning, eos)
    rows = {"train": tr, "tune": va}
    tell('tokens', 1, time.monotonic()-started)

    def token_space(dimensions):
        """Token coordinates with this many dimensions, and the concepts of the tokens found on them."""
        vectors, _ = semantic_vectors(ids, vocabulary, dimensions=dimensions, top=96, seed=42, starts=starts)
        names = [f"x{i}" for i in range(vectors.shape[1])]
        tokens = pd.DataFrame(vectors[np.unique(ids)], columns=names)
        every = pd.DataFrame(vectors, columns=names)

        def token_score(model):
            found = model.assignments(every).argmax(1) + 1
            previous = {name: shift(found[row.y].astype(np.uint32), 1, row.age) for name, row in (("train", tr), ("tune", va))}
            return reading_score(previous["train"], tr.y, previous["tune"], va.y, vocabulary)

        root = enough_rows(lambda rows: concept_model(rows, searching(1)).fit(tokens), len(tokens), token_score)
        return vectors, names, every, root, token_score(root)

    # How many numbers describe a token is not fixed: the coordinates double while the concepts of the
    # tokens found on them predict the next token of held-out text better, by more than one standard error
    # of the paired difference (the judgment used for the rows of every sample).
    dimensions = 8
    vectors, names, every, root, best = token_space(dimensions)
    while 2 * dimensions <= 2 * 97:                              # the profile has 2 x (96 neighbours + 1) columns
        trial = token_space(2 * dimensions)
        gain = trial[-1] - best
        if gain.mean() <= gain.std(ddof=1) / np.sqrt(max(len(gain) - 1, 1)):
            break
        dimensions *= 2
        vectors, names, every, root, best = trial
    classes = root.assignments(every).argmax(1) + 1
    padded = np.vstack([np.full(vectors.shape[1], np.nan), vectors])

    def batches(row):
        previous = row.feature("e1")
        for i in range(0, len(previous), 2048):
            check()
            yield pd.DataFrame(padded[previous[i : i + 2048]], columns=names), row.y[i : i + 2048]

    root.fit_observed_target("token", range(vocabulary), batches(tr), batches(va))
    enc = PairEncoder(vectors, classes, {}, {}, {})
    enc.open_exact = True
    model = PairLanguageModel(enc, root, {}, {}, {})
    for row in rows.values():
        row.spans["c1"] = classes[row.y]
    base = model.root_probability(va, np.arange(len(va.y)))
    reading = root.refine_category("token")
    old_vectors = np.nan_to_num(padded)
    misses = 0
    for width in (1, 2, 3, 4, 5, 6, *(1 << k for k in range(3, 21) if 1 << k <= max(max_context, 8))):
        check()
        if width > max(max_context, 6) or (width > 1 and tr.age.max() < width - 1):
            break
        if width > 1:
            right = 2 ** ((width - 1).bit_length() - 1)
            left = width - right
            # At 3,5,6 the more recent half is the longer one.
            if width & (width - 1):
                left, right = min(left, right), max(left, right)
            l, r = f"e{left}", f"e{right}"
            dictionary = CategoryPairs().fit(shift(tr.spans[l], right, tr.age), tr.spans[r])
            enc.exact[width] = left, right, dictionary
            for row in rows.values():
                row.exact_columns[f"e{width}"] = (l, r, right)
                row.spans[f"e{width}"] = dictionary.transform(
                    shift(row.spans[l], right, row.age), row.spans[r]
                )
            if width & (width - 1) == 0:
                where['step'] = planned.index(width) + 1
                tell('scale', width, time.monotonic()-started)
                graph, old_vectors, _ = geometry(rows, old_vectors, width, searching(width))
                enc.nodes[width] = graph, old_vectors
                enc.local[width] = []
                tell('subconcepts', width, time.monotonic()-started)
                finer_spans(rows, reading, width, graph.concepts, enc.local[width])
        if width <= 6:
            features = dict(stages(True))[width]
        else:
            h = width // 2
            features = (
                [f"c{width}", f"s{width}"]
                + [f"{a}{h}|{b}{h}" for a in "cse" for b in "cse" if (a, b) != ("e", "e")]
                + [f"e{width}"]
            )
        stats = []
        for f in features:
            t = Table.count_regions([(tr.feature(f), tr.y)], vocabulary)
            stats.append(t.lookup(va.feature(f), va.y))
        weights = calibrated_weights(stats, [1.0] * len(stats), base)
        gain = False
        for f, k, (m, n, d) in zip(features, weights, stats):
            if np.isfinite(k):
                base = categorical_reading(n, m, d, base, k, np)
                model.weights[f] = k
                gain = True
        tell('readings', width, time.monotonic()-started)
        if width >= 8:
            misses = 0 if gain else misses + 1
            if misses == 2:
                break
    # What the text talks about, at every scale the model learned: the concept of the mean topic vector of
    # the last W tokens, kept where its concepts make the next token of held-out text more predictable. It
    # is read by multiplying (topic / general)^beta, with beta chosen on validation once the counts are made
    # (PairText.judge). Every scale offers both readings, the ordered halves and the topic; the judgment
    # decides which ones help, scale by scale.
    def topic_score_for(graph, width, space):
        codes = {name: topic_codes(graph, space, row, width) for name, row in rows.items()}
        return reading_score(shift(codes["train"], 1, tr.age), tr.y, shift(codes["tune"], 1, va.age), va.y, vocabulary)

    def topic_concepts(width, space, label, offset=0., span=1.):
        # As for the spans: the topic of the last W tokens and the topic vector of the token after it, so
        # that two topics fall together only if what follows them is alike. Recognized without it.
        means = span_means(space, tr, width)
        draw = np.random.default_rng(42).permutation(np.flatnonzero(np.r_[tr.age[1:] > 0, False]))
        names_t = [f"x{i}" for i in range(space.shape[1])] + [f"f{i}" for i in range(space.shape[1])]
        rows_of = lambda n: pd.DataFrame(np.hstack([means[draw[:n]], space[tr.y[draw[:n] + 1]]]), columns=names_t)
        return enough_rows(lambda rows_: concept_model(rows_, searching(label, 'topics', offset, span)).fit(
            rows_of(rows_)), len(draw), lambda g: topic_score_for(g, width, space))

    # How many numbers describe the topic of a token: they double while the topic of all that the document
    # has said so far, as a concept, predicts the next token of held-out text better (the same judgment).
    where['step'] = where['steps']
    share = 1 / (len(enc.nodes) + 1)                            # the topic coordinates, then every scale
    tell('topics', 0, time.monotonic()-started)
    whole = int(tr.age.max()) + 1
    size = 16
    space = topic_vectors(tr.y, np.flatnonzero(tr.age == 0), vocabulary, size)
    best = topic_score_for(topic_concepts(whole, space, 0, 0., share), whole, space)
    while 2 * size <= 256:
        check()
        trial = topic_vectors(tr.y, np.flatnonzero(tr.age == 0), vocabulary, 2 * size)
        score = topic_score_for(topic_concepts(whole, trial, 0, 0., share), whole, trial)
        gain = score - best
        if gain.mean() <= gain.std(ddof=1) / np.sqrt(max(len(gain) - 1, 1)):
            break
        size, space, best = 2 * size, trial, score
    enc.topic_vectors = space
    # The scales of the topics are added from the shortest, as the scales of the context: a scale whose topic
    # does not read the next token better than the best one so far, by more than one standard error, is a
    # miss, and after two misses in a row the longer scales are not learned.
    best, misses = None, 0
    for i, width in enumerate(sorted(enc.nodes)):
        check()
        tell('topics', width, time.monotonic()-started, inside_permille=int(1000 * (i + 1) * share))
        graph = topic_concepts(width, space, width, (i + 1) * share, share)
        score = topic_score_for(graph, width, space)
        gain = score - best if best is not None else None
        if best is None or gain.mean() > gain.std(ddof=1) / np.sqrt(max(len(gain) - 1, 1)):
            best, misses = score, 0
        else:
            misses += 1
        if graph.concepts >= 2:
            enc.topics[width] = graph
            model.topics[f"t{width}"] = [1.0, 0.0]
        if misses == 2:
            break
    tell('done', max([6, *enc.nodes]), time.monotonic()-started)
    enc.width = max([6, *enc.nodes])                          # the topics keep their own tokens (PairCursor)
    # Every training target is subsequently counted exactly once by the streaming worker.
    model.tables = {f: Table.empty(vocabulary) for f in [*model.weights, *model.topics]}
    return model
