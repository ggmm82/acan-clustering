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
from .pair_model import PairEncoder, PairLanguageModel
from .pair_training import add_chunk, learn

PROFILE = "concept-pairs-v1"


def document_starts(ids, eos):
    return np.r_[0, np.flatnonzero(ids[:-1] == eos) + 1]


def document_stats(row, name):
    key = row.feature(name)
    docs = np.cumsum(row.age == 0)
    groups = np.unique(np.rec.fromarrays([docs, key]), return_inverse=True)[1]
    return document_counts(groups.astype(np.uint64), key != 0, row.y)


class PairText:
    def __init__(self, vocabulary, eos, budget):
        self.vocabulary, self.eos, self.budget_bytes = int(vocabulary), eos, int(budget)
        self.tokens = 0
        self.model = None
        self.target_mass = None
        self.target_kappa = 1.0
        self.tail = np.empty(0, np.int32)

    def add(self, ids, train, tune, check, documents=None, starts_in_piece=None):
        if self.model is None:
            self.model = learn(train, self.vocabulary, self.eos, tune, check, documents)
            self.model.training_weights = dict(self.model.weights)
            self.target_kappa = self.model.root.observed_targets["token"]["kappa"]
            self.target_mass = np.zeros_like(self.model.head)
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
            self.target_mass = self.target_mass + (pairs @ self.model.leaf_weights).T
            self.tail = add_chunk(
                self.model, part, starts, self.tail, self.model.tables, self.budget_bytes
            )
            self.tokens += len(part)
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

    def _scores(self, ids):
        row = self.model.encoder.encode(ids, document_starts(ids, self.eos))
        p = self.model.root_probability(row, np.arange(len(ids)))
        return row, p

    def judge(self, pieces):
        if self.model is None:
            return
        m = self.model
        fields = list(m.tables)
        doc = ["root", "e1", "c1"] + [f"{s}{w}" for w in m.encoder.nodes for s in ("e", "c", "s")]
        rows = [self._scores(np.asarray(p, np.int32)) for p in pieces if len(p)]
        base = np.concatenate([p for _, p in rows])

        def fit(names, statistics, prior, initial):
            stats = [
                tuple(np.concatenate(col) for col in zip(*(statistics(row, f) for row, _ in rows)))
                for f in names
            ]
            weights = calibrated_weights(stats, [initial.get(f, 1.0) for f in names], prior)
            chosen = {}
            for f, k, (mass, counts, distinct) in zip(names, weights, stats):
                if np.isfinite(k):
                    prior = categorical_reading(counts, mass, distinct, prior, k, np)
                    chosen[f] = float(k)
            return chosen, prior

        m.weights, base = fit(
            fields, lambda r, f: m.tables[f].lookup(r.feature(f), r.y), base, m.weights
        )
        m.memory, _ = fit(doc, document_stats, base, m.memory)

    def logp(self, ids, positions=None):
        ids = np.asarray(ids, np.int32)
        pos = np.arange(len(ids)) if positions is None else np.asarray(positions)
        if self.model is None:
            return np.full(len(pos), -np.log(self.vocabulary))
        if not len(ids):
            return np.empty(0)
        row, p = self._scores(ids)
        p = p[pos]
        for f, k in self.model.weights.items():
            mass, counts, d = self.model.tables[f].lookup(row.feature(f)[pos], ids[pos])
            p = categorical_reading(counts, mass, d, p, k, np)
        for f, k in self.model.memory.items():
            mass, counts, d = document_stats(row, f)[:, pos]
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
                    cursor.observe(token)
        return result

    def concepts(self):
        if self.model is None:
            return []
        m = self.model
        return [m.root.concepts] + [
            graph.concepts + sum(len(d.observed) for _, _, _, d in m.encoder.local[w])
            for w, (graph, _) in m.encoder.nodes.items()
        ]

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
        meta = {
            "training_weights": list(m.training_weights.items()),
            "weights": list(m.weights.items()),
            "memory": list(m.memory.items()),
            "nodes": [],
            "exact": [],
            "local": [],
            "tables": list(m.tables),
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
        return text
