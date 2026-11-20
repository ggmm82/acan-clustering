"""Checkpointable text adapter of the tabular concept model for the long training queue."""

import json

import numpy as np
from scipy import sparse

from acan.core.category_pairs import CategoryPairs
from acan.core.category_table import Table
from acan.core.concepts import ConceptModel, Encoder
from acan.core.conditional import calibrated_weights
from acan.concept_algebra import categorical_reading
from .concept_lm import document_counts
from .pair_model import PairEncoder, PairLanguageModel, chain, merged_chain, network_targets
from .concept_net import ConceptNet
from .pair_training import add_chunk, flush, learn

PROFILE = "concept-pairs-v1"


def document_starts(ids, eos):
    return np.r_[0, np.flatnonzero(ids[:-1] == eos) + 1]


def document_stats(row, name):
    key = row.feature(name)
    docs = np.cumsum(row.age == 0)
    groups = np.unique(np.rec.fromarrays([docs, key]), return_inverse=True)[1]
    return document_counts(groups.astype(np.uint64), key != 0, row.y)


class PairText:
    # The concept network learned as a language model (acan/lm/concept_net.py): the start of the chain of readings.
    NETWORK = dict(dim=96, layers=3, heads=4, concepts=64, window=3)

    def __init__(self, vocabulary, eos, budget, max_context=512, network=False):
        self.vocabulary, self.eos, self.budget_bytes = int(vocabulary), eos, int(budget)
        self.network = network                                  # learn the concept network while reading
        self._learner = None
        self.max_context = int(max_context)                     # longest scale the validation may choose
        self.tokens = 0
        self.model = None
        self.target_mass = None
        self.target_kappa = 1.0
        self.tail = np.empty(0, np.int32)

    def add(self, ids, train, tune, check, documents=None, starts_in_piece=None):
        if self.model is None:
            self.model = learn(train, self.vocabulary, self.eos, tune, check, documents, self.max_context,
                               getattr(self, 'report', None))
            self.model.training_weights = {**self.model.weights, **{f: 1.0 for f in self.model.topics}}
            self.target_kappa = self.model.root.observed_targets["token"]["kappa"]
            self.target_mass = np.zeros_like(self.model.head)
        pending, bigrams = {}, None                             # counted per chunk, added to the tables once
        for i in range(0, len(ids), 200000):
            check()
            part = np.asarray(ids[i : i + 200000], dtype=np.int32)
            starts = np.flatnonzero(part[:-1] == self.eos) + 1
            if starts_in_piece is not None:
                a, b = np.searchsorted(starts_in_piece, [i, i + len(part)])
                starts = np.unique(np.r_[starts, starts_in_piece[a:b] - i])
            if not len(self.tail) or self.tail[-1] == self.eos:
                starts = np.r_[0, starts]
            # A sparse (target, previous token) table multiplies the same native
            # soft memberships as fit_observed_target, once per distinct bigram.
            previous = np.r_[self.tail[-1] + 1 if len(self.tail) else 0, part[:-1] + 1]
            previous[starts] = 0
            pairs = sparse.coo_matrix(
                (np.ones(len(part)), (part, previous)), shape=(self.vocabulary, self.vocabulary + 1)
            ).tocsr()
            bigrams = pairs if bigrams is None else bigrams + pairs
            self.tail = add_chunk(
                self.model, part, starts, self.tail, self.model.tables, self.budget_bytes, pending
            )
            self.tokens += len(part)
        if bigrams is not None:
            self.target_mass = self.target_mass + (bigrams @ self.model.leaf_weights).T
        flush(self.model, self.model.tables, pending, self.budget_bytes)
        counts = self.target_mass
        frequency = counts.sum(0) + 1
        self.model.head = categorical_reading(
            counts,
            counts.sum(1, keepdims=True),
            (counts > 1e-9).sum(1, keepdims=True),
            frequency / frequency.sum(),
            self.target_kappa,
            np,
        )
        self.model.root.observed_targets["token"]["probability"] = self.model.head
        self.model.unigram = frequency / frequency.sum()        # what a topic is compared with
        if self.network:
            self._learn_network(ids, check)

    def _learn_network(self, ids, check):
        """One pass of the concept network over the piece just counted."""
        from .concept_net_training import Learner, Network, output_groups
        if self._learner is None:
            m = self.model
            if getattr(m, "net", None) is not None:              # a queue that goes on: the network learned so far
                network = Network.restore(m.net)
            else:
                config = dict(self.NETWORK, vocabulary=self.vocabulary, **(self.network if isinstance(self.network, dict) else {}))
                frequency = m.unigram
                network = Network(config, centres=m.encoder.vectors, frequency=frequency,
                                  groups=output_groups(m.encoder.token_classes[:self.vocabulary], frequency))
            self._learner = Learner(network)
        self.network_loss = self._learner.learn(ids, self.eos, check)
        self.model.net = self._learner.network.export()

    def _scores(self, ids):
        """The start of the chain at every position: the concept network where it has read the document, the concepts
        of the previous token otherwise."""
        row = self.model.encoder.encode(ids, document_starts(ids, self.eos))
        p = self.model.root_probability(row, np.arange(len(ids)))
        net = network_targets(self.model, row)
        return row, np.where(np.isnan(net), p, net)

    def judge(self, pieces, final=False):
        if self.model is None:
            return
        m = self.model
        fields = list(m.tables)
        # The memory of the document is read as the corpus is: from the general to the particular,
        # every scale as concept, sub-concept and then exact identity, every exact suffix included.
        scales = sorted({1, *m.encoder.nodes, *m.encoder.exact})
        doc = ["root"]
        for w in scales:
            doc += [f"c{w}"] if w == 1 or w in m.encoder.nodes else []
            doc += [f"s{w}"] if w in m.encoder.nodes else []
            doc += [f"e{w}"]
            # By content: the positions of this document that talked about the same thing, at any distance.
            doc += [f"t{w}"] if w in getattr(m.encoder, "topics", {}) else []
        rows = [self._scores(np.asarray(p, np.int32)) for p in pieces if len(p)]
        base = np.concatenate([p for _, p in rows])

        # One chain from the general to the particular: at every distance, first what the corpus says, then
        # what the document says. A short context seen once in the document is corrected by the longer
        # contexts after it, as every reading is by the more particular ones.
        fields = [f for f in fields if f not in m.topics]
        order = merged_chain(fields, doc)
        stats = []
        for kind, f in order:
            parts = [m.tables[f].lookup(r.feature(f), r.y) if kind == "corpus" else document_stats(r, f)
                     for r, _ in rows]
            stats.append(tuple(np.concatenate(col) for col in zip(*parts)))
        old = {"corpus": m.weights, "memory": m.memory}
        weights = calibrated_weights(stats, [old[kind].get(f, 1.0) for kind, f in order], base)
        chosen = [(kind, f, float(k)) for (kind, f), k in zip(order, weights) if np.isfinite(k)]
        m.weights = {f: k for kind, f, k in chosen if kind == "corpus"}
        m.memory = {f: k for kind, f, k in chosen if kind == "memory"}
        m.order = [(kind, f) for kind, f, _ in chosen]
        if m.topics and (final or not getattr(self, "topics_judged", False)):
            self._judge_topics([row for row, _ in rows])
        gone = [f for f in m.memory if f.startswith("t") and f not in m.topics]   # topics not read any more
        for f in gone:
            del m.memory[f]
        if gone and getattr(m, "order", None) is not None:
            m.order = [(kind, f) for kind, f in m.order if not (kind == "memory" and f in gone)]

    def _judge_topics(self, rows):
        """Which scales the topic chain reads, and how far their concepts move the reading.

        The topics multiply the reading by q(chain) / q(general), Bayes' rule with exponent 1. Two things
        come from held-out text, as every choice of the model: the scales of the chain, added one at a time,
        the one that reads best first, while the improvement exceeds one standard error of the paired
        difference; and kappa, the weight of the level above against a concept's observations, the value
        that maximizes the held-out likelihood (a continuous search, no grid). The normalization splits into
        sums that do not depend on kappa or on the scales, so everything is judged from one pass. Scales left
        out are dropped, with their concepts and counts, and are not computed any more.
        """
        from .pair_model import walk
        m = self.model
        saved = {f: list(v) for f, v in m.topics.items()}
        order = sorted(m.topics, key=lambda name: int(name[1:]))
        for f in order:
            m.topics[f][1] = 0.0
        P, C, D, S, N = [], [], [], [], []                      # per position: p(y); per scale: C, D, s, n(y)
        for row in rows:
            keys = {f: row.feature(f) for f in order}
            for i, p in enumerate(walk(m, row)):
                y = int(row.y[i])
                P.append(p[y]); c, d, sv, ny = [], [], [], []
                for f in order:
                    table = m.tables.get(f)
                    ix = table.find(np.array([keys[f][i]], np.uint64))[0] if table is not None and keys[f][i] else -1
                    if ix < 0:
                        c.append(0.); d.append(1.); sv.append(0.); ny.append(0.)
                        continue
                    targets, counts = table.followers(ix)
                    share = counts / m.unigram[targets]
                    c.append(table.C[ix]); d.append(max(table.D[ix], 1)); sv.append(float(p[targets] @ share))
                    ny.append(float(share[targets == y].sum()))
                C.append(c); D.append(d); S.append(sv); N.append(ny)
        if not P:
            return saved
        P, C, D, S, N = map(np.asarray, (P, C, D, S, N))

        def losses(scales, kappa):
            ratio, norm = np.ones(len(P)), np.ones(len(P))      # q(y)/q0(y) and sum_v p(v) q(v)/q0(v)
            for j in scales:                                    # from the shortest span to the longest
                a = kappa * D[:, j]
                seen = C[:, j] > 0
                keep = np.where(seen, a / (C[:, j] + a), 1.)
                add = np.where(seen, 1. / (C[:, j] + a), 0.)
                ratio, norm = ratio * keep + add * N[:, j], norm * keep + add * S[:, j]
            return -np.log(P * ratio / norm)

        def fit(scales):
            """The kappa with the best held-out likelihood: golden-section search on log kappa."""
            lo, hi = np.log(1e-3), np.log(1e4)
            g = (np.sqrt(5) - 1) / 2
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            f1, f2 = losses(scales, np.exp(x1)).mean(), losses(scales, np.exp(x2)).mean()
            for _ in range(40):
                if f1 < f2:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - g * (hi - lo); f1 = losses(scales, np.exp(x1)).mean()
                else:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + g * (hi - lo); f2 = losses(scales, np.exp(x2)).mean()
            kappa = float(np.exp((lo + hi) / 2))
            return kappa, losses(scales, kappa)

        chosen, current, kappa = [], -np.log(P), 1.
        while True:
            trials = [(j, *fit(sorted(chosen + [j]))) for j in range(len(order)) if j not in chosen]
            if not trials:
                break
            j, k, loss = min(trials, key=lambda item: item[2].mean())
            gain = current - loss
            if gain.mean() <= gain.std(ddof=1) / np.sqrt(max(len(gain) - 1, 1)):
                break
            chosen, current, kappa = sorted(chosen + [j]), loss, k
        if __import__("os").environ.get("ACAN_PAIR_DEBUG"):
            print("catena degli argomenti: scale", [order[j] for j in chosen], f"κ {kappa:.3g}",
                  f"perdita {current.mean():.4f} senza {-np.log(P).mean():.4f}", flush=True)
        for j, f in enumerate(order):
            if j in chosen:
                m.topics[f] = [kappa, 1.0]
            else:                                               # not read: not computed any more
                del m.topics[f]
                m.encoder.topics.pop(int(f[1:]), None)
                m.tables.pop(f, None)
                m.training_weights.pop(f, None)
        self.topics_judged = True
        return saved

    def logp(self, ids, positions=None):
        ids = np.asarray(ids, np.int32)
        pos = np.arange(len(ids)) if positions is None else np.asarray(positions)
        if self.model is None:
            return np.full(len(pos), -np.log(self.vocabulary))
        if not len(ids):
            return np.empty(0)
        if any(active for _, active in getattr(self.model, "topics", {}).values()):
            from .pair_model import walk
            row = self.model.encoder.encode(ids, document_starts(ids, self.eos))
            values = np.fromiter((p[y] for p, y in zip(walk(self.model, row), ids)), float, len(ids))
            return np.log(values[pos])
        row, p = self._scores(ids)
        p = p[pos]
        for kind, f in chain(self.model):
            if kind == "corpus":
                mass, counts, d = self.model.tables[f].lookup(row.feature(f)[pos], ids[pos])
                k = self.model.weights[f]
            else:
                mass, counts, d = document_stats(row, f)[:, pos]
                k = self.model.memory[f]
            p = categorical_reading(counts, mass, d, p, k, np)
        return np.log(p)

    def prediction_cursor(self, ids):
        ids = np.asarray(ids, np.int32)
        start = document_starts(np.r_[ids, -1], self.eos)[-1]
        return self.model.cursor(ids[start:]) if self.model else None

    def distribution(self, ids):
        cursor = self.prediction_cursor(ids)
        return cursor.distribution() if cursor else np.full(self.vocabulary, 1 / self.vocabulary)

    def generate(self, ids, count, stop=(), temperature=0.0, seed=0, cursor=None):
        rng = np.random.default_rng(seed)
        cursor = cursor or self.prediction_cursor(ids)
        result = []
        for _ in range(count):
            p = cursor.distribution() if cursor else np.full(self.vocabulary, 1 / self.vocabulary)
            if temperature:
                logits = np.log(np.maximum(p, 1e-300))
                weights = np.exp((logits - logits.max()) / temperature)
                token = int(rng.choice(len(p), p=weights / weights.sum()))
            else:
                token = int(np.argmax(p))
            if token in stop:
                break
            result.append(token)
            if cursor:
                if token == self.eos:
                    cursor = self.model.cursor()
                else:
                    cursor.observe(token, remember=False)   # its own words do not reinforce the memory
        return result

    def concepts(self):
        if self.model is None:
            return []
        m = self.model
        return [m.root.concepts] + [
            graph.concepts + sum(len(d.observed) for _, _, _, d in m.encoder.local[w])
            for w, (graph, _) in m.encoder.nodes.items()
        ]

    def context_tokens(self):
        """Tokens of context the selected scales can read (chosen on validation)."""
        return int(self.model.encoder.width) if self.model is not None else 0

    def scales(self):
        """The span lengths whose concepts the model learned: 2, 4, 8, ..."""
        return sorted(int(w) for w in self.model.encoder.nodes) if self.model is not None else []

    def nbytes(self):
        return sum(a.nbytes for a in self.arrays().values())

    def arrays(self):
        data = {
            "pair_text": np.array(
                [
                    1,
                    self.vocabulary,
                    self.eos if self.eos is not None else -1,
                    self.budget_bytes,
                    self.tokens,
                ]
            ),
            "tail": self.tail,
        }
        if self.model is None:
            return data
        m = self.model
        if getattr(m, "net", None) is not None:
            data.update(m.net.arrays())
        meta = {
            "training_weights": list(m.training_weights.items()),
            "weights": list(m.weights.items()),
            "memory": list(m.memory.items()),
            "order": [list(item) for item in chain(m)],
            "nodes": [],
            "exact": [],
            "local": [],
            "tables": list(m.tables),
            "topics": [(int(f[1:]), float(k), float(b)) for f, (k, b) in getattr(m, "topics", {}).items()],
        }

        def put_graph(key, graph):
            data.update(
                {
                    key + k: v
                    for k, v in dict(
                        mean=graph.encoder.mean,
                        std=graph.encoder.std,
                        kappa=np.array([graph.kappa]),
                        resolution=graph.resolution,
                        root_p=graph.root_p,
                        root_g=graph.root_g,
                    ).items()
                }
            )
            for label, group in [("work_", graph.statistics), ("above_", graph.above or {})]:
                data.update({key + label + k: v for k, v in group.items()})

        put_graph("root_", m.root)
        data.update(
            vectors=m.encoder.vectors,
            classes=m.encoder.token_classes,
            head=m.head,
            target_mass=self.target_mass,
            target_kappa=np.array([self.target_kappa]),
        )
        for w, (g, centres) in m.encoder.nodes.items():
            meta["nodes"].append(w)
            put_graph(f"g{w}_", g)
            data[f"centres{w}"] = centres
        for w, graph in getattr(m.encoder, "topics", {}).items():
            put_graph(f"topic{w}_", graph)
        if getattr(m.encoder, "topic_vectors", None) is not None and m.encoder.topics:
            data["topic_vectors"] = m.encoder.topic_vectors
        if getattr(m, "unigram", None) is not None:
            data["unigram"] = m.unigram
        for w, (l, r, d) in m.encoder.exact.items():
            meta["exact"].append((w, l, r))
            data[f"exact{w}"] = d.observed
        for w, steps in m.encoder.local.items():
            for i, (l, r, offset, d) in enumerate(steps):
                meta["local"].append((w, i, l, r, offset))
                data[f"local{w}_{i}"] = d.observed
        for i, t in enumerate(m.tables.values()):
            for name in ("keys", "C", "cells", "counts"):
                data[f"t{i}_{name}"] = getattr(t, name)
        data["configuration"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        return data

    @classmethod
    def from_arrays(cls, data, budget=None):
        version, v, eos, limit, tokens = map(int, data["pair_text"])
        if version != 1:
            raise ValueError("Versione del checkpoint delle coppie sconosciuta.")
        text = cls(v, None if eos < 0 else eos, budget or limit)
        text.tokens, text.tail = tokens, data["tail"].copy()
        if "configuration" not in data:
            return text
        meta = json.loads(data["configuration"].tobytes())

        def graph(key):
            g = ConceptModel(kappa=float(data[key + "kappa"][0]), covariance=False)
            e = g.encoder = Encoder()
            e.mean, e.std = data[key + "mean"], data[key + "std"]
            e.numeric = [f"x{i}" for i in range(len(e.mean))]
            e.categorical, e.categories, e.offsets, e.width = [], {}, {}, 0
            g.labels = []
            g.resolution, g.root_p, g.root_g = (
                data[key + n] for n in ("resolution", "root_p", "root_g")
            )

            def group(name):
                p = key + name + "_"
                return {k[len(p) :]: v for k, v in data.items() if k.startswith(p)}

            g.statistics, g.above = group("work"), group("above") or None
            return g

        def dictionary(key):
            d = CategoryPairs()
            d.observed = data[key]
            return d

        text.target_mass = data["target_mass"]
        text.target_kappa = float(data["target_kappa"][0])
        root = graph("root_")
        root.observed_targets = {"token": {"probability": data["head"]}}
        nodes = {w: (graph(f"g{w}_"), data[f"centres{w}"]) for w in meta["nodes"]}
        exact = {w: (l, r, dictionary(f"exact{w}")) for w, l, r in meta["exact"]}
        local = {w: [] for w in nodes}
        for w, i, l, r, offset in meta["local"]:
            local[w].append((l, r, offset, dictionary(f"local{w}_{i}")))
        enc = PairEncoder(data["vectors"], data["classes"], nodes, exact, local)
        tables = {
            f: Table(*(data[f"t{i}_{field}"] for field in ("keys", "C", "cells", "counts")), v)
            for i, f in enumerate(meta["tables"])
        }
        enc.open_exact = True
        text.model = PairLanguageModel(enc, root, tables, dict(meta["weights"]), dict(meta["memory"]))
        text.model.training_weights = dict(meta["training_weights"])
        if "order" in meta:
            text.model.order = [tuple(item) for item in meta["order"]]
        enc.topic_vectors = data["topic_vectors"] if "topic_vectors" in data else None
        for w, k, b in meta.get("topics", []):
            enc.topics[w] = graph(f"topic{w}_")
            text.model.topics[f"t{w}"] = [k, b]
        text.model.unigram = data["unigram"] if "unigram" in data else None
        text.model.net = ConceptNet.from_arrays(data)
        text.network = text.model.net is not None
        return text
