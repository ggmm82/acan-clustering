"""The concept network alone as the model of a text: no counts and no concept model learned by EM.

Every object is the concept network's (acan/lm/concept_net.py): the tokens are concepts, the layers of concept
attention read the document through the sums of their concepts, the memory of the document is learned, and the next
token is the membership of the final state to the concepts of the tokens. Everything is learned along the gradient
of the log-likelihood of the next token. Every piece of the queue is read once, and the text goes on; with the last
piece the network reads the whole text of the queue again, pass after pass, while held-out text is read better by
more than one standard error (concept_net_training.Grower, with its sizes fixed): reading again is how it remembers
what it saw. Every document starts after an ``eos``, so its first token is read as in training. The interface is the one of the text model of the queue
(add, judge, logp, prediction_cursor, generate, arrays), so training, generation, evaluation, the API and the export
read it as they read the other one.
"""
import time
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
    NETWORK = dict(dim=96, layers=3, heads=4, concepts=64, window=3, grouped=True)
    # With growth the network starts small and the data choose how it grows (``_race``)
    GROWING = dict(dim=32, layers=1, heads=4, concepts=16, window=3, grouped=True)
    KINDS = ("com'è", 'larghezza', 'strato', 'concetti')

    def __init__(self, vocabulary, eos, budget, max_context=512, network=None, growth=False):
        self.vocabulary, self.eos, self.budget_bytes = int(vocabulary), int(eos), int(budget)
        self.max_context = int(max_context)
        self.growth = bool(growth)
        self.network = dict(self.GROWING if self.growth else self.NETWORK, **(network or {}))
        self.next_race, self.races = 0, []                     # with growth: when the next race is, and their results
        self.tokens, self.tail, self.report = 0, np.empty(0, np.int64), None
        self.model = None                                      # SimpleNamespace(net=ConceptNet, tables={})
        self._learner = None
        self.network_done, self.network_history = False, []
        self.read = 0                                          # tokens of the current text read so far
        self.trace = []                                        # [steps learned, validation NLL] whenever it is measured
        self.piece = 0

    # ---- learning ------------------------------------------------------------------------------------
    def _held(self, pieces):
        from .concept_net_training import documents
        out = []
        for piece in pieces:
            out += documents(np.asarray(piece, np.int64), self.eos, 1 << 30, begin=True)
        return out

    def add(self, ids, train, tune, check, documents=None, starts_in_piece=None):
        from .concept_net_training import Grower, Network, frequency_groups, held_out
        ids = np.asarray(ids, np.int64)
        if not len(self.tail):                                  # a new text of the queue begins
            self.read = 0
        if self._learner is None:
            if self.model is not None:                          # a queue that goes on: the network learned so far
                network = Network.restore(self.model.net)
            else:
                frequency = np.bincount(ids, minlength=self.vocabulary) + 1.
                config = {k: v for k, v in self.network.items() if k != 'passes'}
                network = Network(dict(config, vocabulary=self.vocabulary), frequency=frequency / frequency.sum(),
                                  groups=frequency_groups(frequency) if config.get('grouped') else None)
            self._learner = Grower(network, passes=self.network.get('passes', True))
            self._learner.KINDS = ()                            # the sizes are fixed
            self._learner.begin = True
        held = self._held(tune)
        self.piece = self.piece or len(ids)
        done_before = self.tokens - self._learner.learned           # what earlier learners of the queue learned

        def score(net, tokens=None):
            if not held:
                return np.zeros(2)
            values = held_out(net, held, tokens)
            if tokens is None:                                  # the whole validation: a point of the chart
                self._point(done_before + self._learner.learned, values)
            return values
        started, last = time.monotonic(), [0.]
        again = self.read + len(ids) >= len(train) and self.network.get('passes', True)

        def watched(measured=True):
            check()
            now = time.monotonic()
            if self.report is not None and now - last[0] >= 15:   # how far the network is, for the page
                last[0] = now
                if measured and held:                           # and its validation as it learns, for the chart
                    network = self._learner.learner.network
                    self._point(done_before + self._learner.learned, held_out(network, held))
                    network.train()
                best = self._learner.best_scores
                self.report('network', 0, now - started, learned=int(self._learner.learned), again=int(again),
                            held_milli=int(1000 * -best.mean()) if best is not None and len(best) else 0)
        self._learner.passes = False                            # every piece once ...
        best = self._learner.learn(ids, self.eos, score, watched)
        if again and not self._learner.done and not self.growth:   # ... and with the last one, the whole text again
            self._learner.passes = True
            best = self._learner.learn(np.asarray(train, np.int64), self.eos, score, watched)
        if best is not None:
            self.model = SimpleNamespace(net=best, tables={})
        self.read += len(ids)
        self.tokens += len(ids)
        self.tail = ids[-max(self.max_context, 1):]
        if self.growth and self._learner is not None and (self.read >= self.next_race or self.read >= len(train)):
            self._race(np.asarray(train[:self.read], np.int64), held, lambda: watched(False))
            self.next_race = 2 * self.read
        self.network_history, self.network_done = self._learner.history, self._learner.done
        if self._learner.done:
            self._learner = None

    def _point(self, tokens, values, kind=None):
        """A point of the chart: the validation after so many tokens learned, in steps of the queue (with the
        candidate, in a race of growth). A long queue keeps every other point of its older ones: the chart stays
        light and still shows the whole path."""
        self.trace.append([round(tokens / max(self.piece, 1), 3), float(-values.mean())] + ([kind] if kind else []))
        if len(self.trace) > 512:
            self.trace = self.trace[:-1:2] + self.trace[-1:]

    # ---- growth ------------------------------------------------------------------------------------------
    def _race(self, text, held, check):
        """Every time the text read doubles (and with the last piece) the network as it is and its grown copies
        (twice as wide, one more layer, every concept split in two: each reads exactly as the network) read the
        text read so far again and again, each until held-out text no longer improves by more than one standard
        error twice in a row, kept at its best; the smallest that the best does not beat by more than one standard
        error goes on. A wider copy starts from half the step: its coordinates move in pairs."""
        import copy
        import math
        from .concept_net_training import Learner, documents, held_out
        if not held:
            return
        docs = documents(text, self.eos, self._learner.learner.longest, begin=True)
        better = lambda a, b: len(a - b) > 1 and (a - b).mean() > (a - b).std(ddof=1) / math.sqrt(len(a) - 1)
        base = self._learner.learner
        results = []
        for number, kind in enumerate(self.KINDS):
            network = copy.deepcopy(base.network)
            if kind == 'larghezza':
                network = network.widened()
            elif kind == 'strato':
                network.add_layer()
            elif kind == 'concetti':
                for i in range(len(network.layers)):
                    network.split_concepts(i)
            learner = copy.deepcopy(base) if kind == "com'è" else Learner(network, tokens=base.tokens, longest=base.longest)
            if kind != "com'è":
                learner.steps, learner.warmup = base.steps, base.warmup
                learner.lr = base.lr / 2 if kind == 'larghezza' else base.lr
            best, best_s, misses, passes = copy.deepcopy(learner), held_out(learner.network, held), 0, 0
            while misses < 2:
                learner.learn_documents(docs, lambda: (check(), self._racing(number, passes, best_s)))
                passes += 1
                s = held_out(learner.network, held)
                self._point(self.tokens + passes * len(text), s, kind)
                if better(s, best_s):
                    best, best_s, misses = copy.deepcopy(learner), s, 0
                else:
                    misses += 1
            results.append((kind, best, best_s, sum(p.numel() for p in best.network.parameters())))
        top = max(results, key=lambda r: r[2].mean())
        kind, learner, scores, _ = min((r for r in results if not better(top[2], r[2])), key=lambda r: r[3])
        self._learner.learner = learner
        net = learner.network.export()
        self._learner.best, self._learner.best_scores = net, scores
        self.model = SimpleNamespace(net=net, tables={})
        c = net.config
        self.races.append(dict(tokens=int(self.read), choice=kind, dim=int(c['dim']), layers=int(c['layers']),
                               concepts=[int(k) for k in c['concepts']], held_out=float(-scores.mean())))

    def _racing(self, number, passes, best):
        """The page: which candidate of the race is reading, how many passes, the best validation so far."""
        now = time.monotonic()
        if self.report is not None and now - getattr(self, '_reported', 0.) >= 15:
            self._reported = now
            self.report('network', 0, 0., race=number + 1, passes=int(passes), learned=int(self.read),
                        held_milli=int(1000 * -best.mean()) if best is not None and len(best) else 0)

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
                                      int(self.network_done), int(self.growth), int(self.next_race)], np.int64),
                "tail": np.asarray(self.tail, np.int64)}
        if self.model is not None:
            data.update(self.model.net.arrays('net.'))
        return data

    @classmethod
    def from_arrays(cls, data, budget=None):
        values = [int(x) for x in data["net_text"]]
        version, vocabulary, eos, limit, tokens, context, done = values[:7]
        growth, next_race = (values[7], values[8]) if len(values) > 8 else (0, 0)
        if version != VERSION:
            raise ValueError("Versione del checkpoint della rete sconosciuta.")
        text = cls(vocabulary, eos, budget or limit, context, growth=bool(growth))
        text.next_race = next_race
        text.tokens, text.tail, text.network_done = tokens, np.asarray(data["tail"], np.int64), bool(done)
        net = ConceptNet.from_arrays(data, 'net.')
        if net is not None:
            text.model = SimpleNamespace(net=net, tables={})
            text.network = {k: net.config[k] for k in ('dim', 'layers', 'heads', 'window') if k in net.config}
        return text
