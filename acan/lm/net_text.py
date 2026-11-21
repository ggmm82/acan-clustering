"""The concept network alone as the model of a text: no counts and no concept model learned by EM.

Every object is the concept network's (acan/lm/concept_net.py): the tokens are concepts, the layers of concept
attention read the document through the sums of their concepts, the memory of the document is learned, and the next
token is the membership of the final state to the concepts of the tokens. Everything is learned along the gradient
of the log-likelihood of the next token. A piece of the queue is read again while held-out text is read better by
more than one standard error (concept_net_training.Grower, with its sizes fixed); every document starts after an
``eos``, so its first token is read as in training. The interface is the one of the text model of the queue
(add, judge, logp, prediction_cursor, generate, arrays), so training, generation, evaluation, the API and the export
read it as they read the other one.
"""
from types import SimpleNamespace

import numpy as np

from .concept_net import ConceptNet

VERSION = 1


class NetCursor:
    """Reads a document one token at a time with the network; ``remember=False`` keeps a token out of the memory of
    the document (the model's own words do not reinforce it)."""

    def __init__(self, net, eos, ids=()):
        self.cursor = net.cursor()
        self.cursor.observe(int(eos))
        if len(ids):
            self.cursor.prefill(np.asarray(ids, np.int64))

    def observe(self, token, remember=True):
        keys = len(self.cursor.keys)
        self.cursor.observe(int(token))
        if not remember and len(self.cursor.keys) > keys:      # the address left by the state before it goes away
            self.cursor.keys.pop(); self.cursor.after.pop()

    def distribution(self):
        return self.cursor.distribution()

    def copy(self):
        other = NetCursor.__new__(NetCursor)
        other.cursor = self.cursor.copy()
        return other


class NetText:
    NETWORK = dict(dim=96, layers=3, heads=4, concepts=64, window=3, grouped=False)

    def __init__(self, vocabulary, eos, budget, max_context=512, network=None):
        self.vocabulary, self.eos, self.budget_bytes = int(vocabulary), int(eos), int(budget)
        self.max_context = int(max_context)
        self.network = dict(self.NETWORK, **(network or {}))
        self.tokens, self.tail, self.report = 0, np.empty(0, np.int64), None
        self.model = None                                      # SimpleNamespace(net=ConceptNet, tables={})
        self._learner = None
        self.network_done, self.network_history = False, []

    # ---- learning ------------------------------------------------------------------------------------
    def _held(self, pieces):
        from .concept_net_training import documents
        out = []
        for piece in pieces:
            out += documents(np.asarray(piece, np.int64), self.eos, 1 << 30, begin=True)
        return out

    def add(self, ids, train, tune, check, documents=None, starts_in_piece=None):
        from .concept_net_training import Grower, Network, held_out
        ids = np.asarray(ids, np.int64)
        if self._learner is None:
            if self.model is not None:                          # a queue that goes on: the network learned so far
                network = Network.restore(self.model.net)
            else:
                frequency = np.bincount(ids, minlength=self.vocabulary) + 1.
                config = {k: v for k, v in self.network.items() if k != 'passes'}
                network = Network(dict(config, vocabulary=self.vocabulary), frequency=frequency / frequency.sum())
            self._learner = Grower(network, passes=self.network.get('passes', True))
            self._learner.KINDS = ()                            # the sizes are fixed
            self._learner.begin = True
        held = self._held(tune)
        score = (lambda net, tokens=None: held_out(net, held, tokens)) if held else (lambda net, tokens=None: np.zeros(2))
        best = self._learner.learn(ids, self.eos, score, check)
        if best is not None:
            self.model = SimpleNamespace(net=best, tables={})
        self.tokens += len(ids)
        self.tail = ids[-max(self.max_context, 1):]
        self.network_history, self.network_done = self._learner.history, self._learner.done
        if self._learner.done:
            self._learner = None

    def judge(self, pieces, final=False):
        """Nothing to choose: the network is its own reading."""

    # ---- reading ------------------------------------------------------------------------------------
    def logp(self, ids, positions=None):
        ids = np.asarray(ids, np.int64)
        pos = np.arange(len(ids)) if positions is None else np.asarray(positions)
        if self.model is None or not len(ids):
            return np.full(len(pos), -np.log(self.vocabulary))
        out = np.empty(len(ids))
        bounds = np.r_[0, np.flatnonzero(ids == self.eos) + 1]
        bounds = np.unique(np.r_[bounds, len(ids)])
        for a, b in zip(bounds[:-1], bounds[1:]):
            doc = np.r_[self.eos, ids[a:b]]                     # every document is read after an eos
            out[a:b] = np.log(np.maximum(self.model.net.targets(doc), 1e-300))
        return out[pos]

    def prediction_cursor(self, ids):
        if self.model is None:
            return None
        ids = np.asarray(ids, np.int64)
        ends = np.flatnonzero(ids == self.eos)
        start = int(ends[-1]) + 1 if len(ends) else 0
        return NetCursor(self.model.net, self.eos, ids[start:])

    def cursor(self, ids=()):
        return self.prediction_cursor(ids)

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
                    cursor = self.prediction_cursor(())
                else:
                    cursor.observe(token, remember=False)
        return result

    # ---- what the page shows -------------------------------------------------------------------------------
    def concepts(self):
        """The concepts of every layer of the network (groups x concepts)."""
        if self.model is None:
            return []
        net = self.model.net
        return [int(net.p[f'layers.{i}.g'].shape[0] * net.p[f'layers.{i}.g'].shape[1]) for i in range(net.layers)]

    def context_tokens(self):
        """The network reads the whole document: the context is the document itself."""
        return self.max_context if self.model is not None else 0

    def nbytes(self):
        return sum(a.nbytes for a in self.arrays().values())

    # ---- persistence -------------------------------------------------------------------------------------
    def arrays(self):
        data = {"net_text": np.array([VERSION, self.vocabulary, self.eos, self.budget_bytes, self.tokens, self.max_context,
                                      int(self.network_done)], np.int64),
                "tail": np.asarray(self.tail, np.int64)}
        if self.model is not None:
            data.update(self.model.net.arrays('net.'))
        return data

    @classmethod
    def from_arrays(cls, data, budget=None):
        version, vocabulary, eos, limit, tokens, context, done = (int(x) for x in data["net_text"])
        if version != VERSION:
            raise ValueError("Versione del checkpoint della rete sconosciuta.")
        text = cls(vocabulary, eos, budget or limit, context)
        text.tokens, text.tail, text.network_done = tokens, np.asarray(data["tail"], np.int64), bool(done)
        net = ConceptNet.from_arrays(data, 'net.')
        if net is not None:
            text.model = SimpleNamespace(net=net, tables={})
            text.network = {k: net.config[k] for k in ('dim', 'layers', 'heads', 'window') if k in net.config}
        return text
