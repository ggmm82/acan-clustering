"""The concept network alone as the model of a text: no counts and no concept model learned by EM.

Every object is the concept network's (acan/lm/concept_net.py): the tokens are concepts, the layers of concept
attention read the document through the sums of their concepts, the memory of the document is learned, and the next
token is the membership of the final state to the concepts of the tokens. Everything is learned along the gradient
of the log-likelihood of the next token. Growth races reread the current piece until the structure is selected.
With the last piece the network reads the whole text again, pass after pass, while held-out text is read better by
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
    GROWING = dict(dim=32, layers=1, heads=1, concepts=1, window=3, grouped=True)
    KINDS = ("com'è", 'larghezza', 'strato', 'concetti', 'teste')

    def __init__(self, vocabulary, eos, budget, max_context=512, network=None, growth=False):
        self.vocabulary, self.eos, self.budget_bytes = int(vocabulary), int(eos), int(budget)
        self.max_context = int(max_context)
        self.growth = bool(growth)
        self.network = dict(self.GROWING if self.growth else self.NETWORK, **(network or {}))
        self.next_race, self.races = 0, []                     # with growth: when the next race is, and their results
        self.race_gap = 0                                      # text skipped after a race the network as it is won
        self.decide_on = None                                  # the text the decisions are judged on (the queue's validation)
        self._previous = None                                  # validation after the last piece: is the network still learning?
        self.tokens, self.tail, self.report = 0, np.empty(0, np.int64), None
        self.model = None                                      # SimpleNamespace(net=ConceptNet, tables={})
        self.device = 'cpu'
        self._learner = None
        self.network_done, self.network_history = False, []
        self.read = 0                                          # tokens of the current text read so far
        self.trace = []                                        # [steps learned, validation NLL] whenever it is measured
        self.watch = None                                      # the pieces the chart is measured on (the queue's validation)
        self.piece = 0

    # ---- learning ------------------------------------------------------------------------------------
    def _held(self, pieces, longest=1 << 30):
        """Held-out documents, each cut in stretches of at most ``longest`` tokens: the memory of the document compares
        every position with all the ones before, and a whole long document would not fit."""
        from .concept_net_training import documents
        out = []
        for piece in pieces:
            out += documents(np.asarray(piece, np.int64), self.eos, longest, begin=True)
        return out

    def add(self, ids, train, tune, check, documents=None, starts_in_piece=None):
        from copy import copy
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
            options = dict(tokens=256, longest=min(256, self.max_context)) if self.device == 'cuda' else {}
            self._learner = Grower(network.to(self.device), passes=self.network.get('passes', True), **options)
            self._learner.KINDS = ()                            # the sizes are fixed
            self._learner.begin = True
        if self.decide_on is not None:                          # all the validation of the queue: small gains are seen
            if getattr(self, '_decide_docs', None) is None:
                self._decide_docs = self._held([self.decide_on], self._learner.learner.longest)   # as long as in learning
            held = self._decide_docs
        else:
            held = self._held(tune, self._learner.learner.longest) if self.device == 'cuda' else self._held(tune)
        self.piece = self.piece or len(ids)
        begun = self.read, self._learner.learner.seen
        counted = dict(learner=self._learner.learner, seen=begun[1], total=0)

        def where():
            # tokens of the queue so far, rereadings included: it never goes back, also when a decision returns to an
            # earlier copy of the network (its count of tokens is lower: what it reads from there is added on)
            learner = self._learner.learner
            seen = getattr(learner, 'seen', 0)
            counted['total'] += max(seen - counted['seen'], 0)
            counted['learner'], counted['seen'] = learner, seen
            return begun[0] + counted['total']

        def score(net, tokens=None):
            if not held:
                return np.zeros(2)
            return held_out(net, held, tokens)
        watch_width = self._learner.learner.longest if self.device == 'cuda' else 1 << 30
        shown = self._held(self.watch, watch_width) if self.watch else held  # what the chart is measured on
        started, last = time.monotonic(), [0.]
        end = self.read + len(ids)
        again = end >= len(train) and self.network.get('passes', True)

        def watched():
            check()
            now = time.monotonic()
            if self.report is not None and now - last[0] >= 15:   # how far the network is, for the page
                last[0] = now
                network = self._learner.learner.network
                if shown:                                      # and its validation as it learns, for the chart
                    self._point(min(where(), end), held_out(network, shown))
                    network.train()
                best = self._learner.best_scores
                self.report('network', 0, now - started, learned=int(self._learner.learned), again=int(again),
                            held_milli=int(1000 * -best.mean()) if best is not None and len(best) else 0,
                            speed=int((where() - begun[0]) / max(now - started, 1e-9)), network=network.footprint())
        if shown and not self.trace:                            # where the queue starts
            self._point(where(), held_out(self._learner.learner.network, shown))
        if self.growth and held and (self.read >= self.next_race or end >= len(train)):
            self._learner = copy(self._learner)                 # an unfinished piece must not alter the preceding one
            first_race = len(self.races)
            self._race_began = time.monotonic()
            self._race_piece = len(ids)
            round_number = 1
            best, grew = self._race(ids, held, check, shown, round_number)
            while grew:
                round_number += 1
                best, grew = self._race(ids, held, check, shown, round_number)
            counted.update(seen=self._learner.learner.seen, total=len(ids))  # a rereading does not advance the queue
            self.race_gap = len(ids) if len(self.races) > first_race + 1 else max(2 * self.race_gap, len(ids))
            self.next_race = end + self.race_gap
            if again and len(self.races) > first_race:
                self._learner._keep(best, self._learner.best_scores)   # restore the stabilized winner if rereading does not help
            self._previous = self._learner.best_scores
        else:
            self._learner.passes = False                        # a piece without a race is read once
            best = self._learner.learn(ids, self.eos, score, watched)
            if self.growth and held and self._learner is not None and not self._learner.done:
                now = held_out(self._learner.learner.network, held)
                if self._previous is not None and not self._better(now, self._previous):
                    self.next_race = min(self.next_race, end)   # it no longer learns at this size: try to grow
                self._previous = now
        if again and not self._learner.done:                    # ... and with the last one, the whole text again
            self._learner.passes = True
            best = self._learner.learn(np.asarray(train, np.int64), self.eos, score, watched)
        if best is not None:
            self.model = SimpleNamespace(net=best, tables={})
        if shown and self._learner is not None:                 # where the step ends
            self._point(end, held_out(self._learner.learner.network, shown))
        self.read += len(ids)
        self.tokens += len(ids)
        self.tail = ids[-max(self.max_context, 1):]
        self.network_history, self.network_done = self._learner.history, self._learner.done
        if self._learner.done:
            self._learner = None

    @staticmethod
    def _better(a, b):
        """a reads held-out text better than b by more than one standard error of the paired difference."""
        import math
        d = np.asarray(a) - np.asarray(b)
        return len(d) > 1 and len(a) == len(b) and d.mean() > d.std(ddof=1) / math.sqrt(len(d) - 1)

    def _point(self, tokens, values, kind=None, round_number=None):
        """A point of the chart: the validation after so many tokens learned, in steps of the queue (with the
        candidate, in a race of growth). A long queue keeps every other point of its older ones: the chart stays
        light and still shows the whole path."""
        point = [round(tokens / max(self.piece, 1), 3), float(-values.mean())]
        if kind is not None:
            point.extend((kind, round_number))
        self.trace.append(point)
        if len(self.trace) > 512:
            self.trace = self.trace[:-1:2] + self.trace[-1:]

    # ---- growth ------------------------------------------------------------------------------------------
    def _race(self, ids, held, check, shown, round_number):
        """Compare the current network and its growth candidates on the same new piece.

        The smallest candidate within one standard error of the best validation score continues.
        The wider candidate learns at half the learning rate because its coordinates were duplicated.
        """
        import copy
        import math
        from .concept_net_training import Learner, documents, held_out
        grower, base = self._learner, self._learner.learner
        check()
        docs = documents(ids, self.eos, base.longest, begin=True)
        if not docs:
            return base.network.export(), False
        size = sum(map(len, docs))
        batch = min(grower.batch, max(1, math.isqrt(size)))      # as many batches as tokens in a batch
        warmup = base.warmup if grower.learned else min(base.warmup, max(1, -(-size // batch)))
        # Give each layer a chance without adding a full training run for every layer in every race.
        next_layer = len(self.races) % len(base.network.layers)
        selected = base.network.concepts_to_split(next_layer, docs)
        labels = [f'concetti · strato {next_layer + 1}' if kind == 'concetti' else kind for kind in self.KINDS]
        results, trials = [], []
        if round_number > 1:
            previous = grower.best_scores
        else:
            previous = held_out(base.network, held) if self.model is not None else None
        reference_nll = float(-previous.mean()) if previous is not None else None
        projection, candidate_shape = None, None
        current_shape = base.network.footprint()

        def report(number, read, evaluating=False):
            if self.report is None:
                return
            total = len(self.KINDS)
            elapsed = time.monotonic() - self._race_began
            spent = ((round_number - 1) * total + number) * size + read
            self.report('network', 0, elapsed, race=number + 1, race_total=total, race_round=round_number,
                        candidate=labels[number], learned=int(self.read), piece=int(size), done=int(read),
                        speed=int(spent / max(elapsed, 1e-9)), race_candidates=labels,
                        race_scores=[trial['held_out'] for trial in trials], race_evaluating=evaluating,
                        race_projection=projection, race_reference_nll=reference_nll,
                        network=current_shape, candidate_network=candidate_shape)

        report(0, 0)
        for number, (kind, label) in enumerate(zip(self.KINDS, labels)):
            layer = next_layer if kind == 'concetti' else None
            split = (layer, selected) if layer is not None else None
            if kind == "com'è":
                learner = copy.deepcopy(base)
            else:
                if kind == 'larghezza':
                    network = base.network.widened()
                elif kind == 'teste':
                    network = base.network.more_heads()
                else:
                    network = copy.deepcopy(base.network)
                    if kind == 'strato':
                        network.add_layer()
                    elif split is not None:
                        network.split_one(*split)
                learner = Learner(network, tokens=batch, longest=base.longest)
                # Start from the same optimizer, pace and order of training rows.
                learner.steps, learner.seen = base.steps, base.seen
                learner.rng = copy.deepcopy(base.rng)
                learner.lr = base.lr / 2 if kind == 'larghezza' else base.lr   # its coordinates move in pairs
                learner.carry(base, kind, split)
            learner.tokens, learner.warmup = batch, warmup
            candidate_shape = learner.network.footprint()
            start = learner.seen
            self._reported = float('-inf')
            learner.learn_documents(docs, lambda: (check(), self._racing(
                round_number, number, size, label, learner, start, shown, report)))
            check()
            report(number, size, evaluating=True)
            scores = held_out(learner.network, held)
            if shown:                                           # the chart can use a different sample
                chart_scores = scores if shown is held else held_out(learner.network, shown)
                self._point(self.read + len(ids), chart_scores, label, round_number)
            count = candidate_shape['parameters']
            results.append((kind, learner, scores, count, layer))
            trials.append(dict(choice=kind, layer=layer+1 if layer is not None else None,
                               heads=int(learner.network.config['heads']), concepts=list(learner.network.config['concepts']),
                               held_out=float(-scores.mean()), parameters=count))
            winner, retained = self._race_choice(results, previous)
            choice, _, choice_scores, _, _ = winner
            projection = dict(candidate=labels[self.KINDS.index(choice)], held_out=float(-choice_scores.mean()),
                              grows=choice != "com'è" and not retained, retained=retained)
            if retained:
                projection = dict(candidate="com'è", held_out=reference_nll, grows=False, retained=True)
            report(number, size)
        kind, learner, scores, _, layer = winner
        if retained:                                           # neither rereading nor growing improved the reference
            kind, learner, scores, layer = "com'è", base, previous, None
        else:
            grower.learned += size
        grower.learner = learner
        while grower.decision <= grower.learned:               # the race was the decision of these tokens
            grower.decision *= 2
        net = grower.best if retained and round_number > 1 else learner.network.export()
        grower.best, grower.best_scores = net, scores
        grower.best_learner = None
        c = net.config
        self.races.append(dict(tokens=int(self.tokens + len(ids)), choice=kind, round=round_number, retained=retained,
                               reference_nll=reference_nll,
                               layer=layer+1 if layer is not None else None,
                               dim=int(c['dim']), layers=int(c['layers']), heads=int(c['heads']),
                               concepts=[int(k) for k in c['concepts']], held_out=float(-scores.mean()),
                               candidates=trials))
        return net, kind != "com'è"

    def _race_choice(self, results, previous):
        eligible = results if previous is None else [result for result in results if self._better(result[2], previous)]
        if not eligible:
            return results[0], True
        top = max(eligible, key=lambda result: result[2].mean())
        winner = min((result for result in eligible if not self._better(top[2], result[2])), key=lambda result: result[3])
        return winner, False

    def _racing(self, round_number, number, size, kind, learner, start, shown, report):
        """The page: which candidate of the race is learning the new piece, how far it is, and its validation on
        the chart: every candidate a line over the same step."""
        from .concept_net_training import held_out
        now = time.monotonic()
        if self.report is not None and now - getattr(self, '_reported', 0.) >= 15:
            self._reported = now
            read = learner.seen - start
            if shown:
                self._point(self.read + self._race_piece * read / size, held_out(learner.network, shown), kind, round_number)
                learner.network.train()
            report(number, read)

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
                                      int(self.network_done), int(self.growth), int(self.next_race),
                                      int(self.race_gap)], np.int64),
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
        text.race_gap = values[9] if len(values) > 9 else 0
        text.tokens, text.tail, text.network_done = tokens, np.asarray(data["tail"], np.int64), bool(done)
        net = ConceptNet.from_arrays(data, 'net.')
        if net is not None:
            text.model = SimpleNamespace(net=net, tables={})
            text.network = {k: net.config[k] for k in ('dim', 'layers', 'heads', 'window') if k in net.config}
        return text
