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
    NETWORK = dict(dim=32, layers=3, heads=4, concepts=[16, 64, 256], window=1, view_window=16, cache=16,
                   grouped=True, tabular_hierarchy=True, linear_concepts=True, sentinel=True)
    # With growth: small synthetic vocabularies start minimal; full language vocabularies start with a multi-view pyramid
    GROWING = dict(dim=32, layers=1, heads=1, concepts=1, window=2,
                   grouped=True, tabular_hierarchy=True, linear_concepts=True, sentinel=True)
    LANGUAGE_GROWING = dict(dim=32, layers=2, heads=4, concepts=[16, 64], window=1, view_window=16, cache=16,
                            grouped=True, tabular_hierarchy=True, linear_concepts=True, sentinel=True)
    KINDS = ("com'è", 'larghezza', 'strato', 'concetti', 'teste')
    MAX_ROUNDS = 4

    @staticmethod
    def _clean_limits(limits):
        raw = limits or {}
        return {k: max(0, int(raw.get(k) or 0)) for k in ('max_dim', 'max_layers', 'max_heads', 'max_concepts')}

    def __init__(self, vocabulary, eos, budget, max_context=512, network=None, growth=False, weight=None, limits=None,
                 start_dim=0, start=None):
        self.vocabulary, self.eos, self.budget_bytes = int(vocabulary), int(eos), int(budget)
        self.max_context = int(max_context)
        self.growth = bool(growth)
        self.limits = self._clean_limits(limits)
        self.kinds_allowed = None                              # growths that may race (None: all); can change while it learns
        self.weight = None if weight is None else np.asarray(weight, np.float32)
        if self.growth and network is None and self.vocabulary >= 2048:
            base = dict(self.LANGUAGE_GROWING, concepts=list(self.LANGUAGE_GROWING['concepts']))
            # The network it starts from, where chosen (0: as above); growth may enlarge it up to the ceilings.
            start = dict(start or {})
            if start_dim and int(start_dim) > 0:
                start.setdefault('dim', int(start_dim))
            if int(start.get('dim') or 0) > 0:
                base['dim'] = int(start['dim'])
            if int(start.get('heads') or 0) > 0:
                base['heads'] = int(start['heads'])
            if int(start.get('layers') or 0) > 0 or int(start.get('concepts') or 0) > 0:
                layers = int(start.get('layers') or 0) or base['layers']
                first = int(start.get('concepts') or 0) or base['concepts'][0]
                base['layers'] = layers                          # the pyramid: the first layer, then four times as many
                base['concepts'] = [first] + [4 * first] * (layers - 1)
            if 0 < self.limits['max_dim'] < base['dim']:
                base['dim'] = self.limits['max_dim']
            if 0 < self.limits['max_heads'] < base['heads']:
                base['heads'] = self.limits['max_heads']
            if 0 < self.limits['max_layers'] < base['layers']:
                base['layers'] = self.limits['max_layers']
                base['concepts'] = base['concepts'][:base['layers']]
            if 0 < self.limits['max_concepts']:
                base['concepts'] = [min(c, self.limits['max_concepts']) for c in base['concepts']]
            self.network = base
        else:
            self.network = dict(self.GROWING if self.growth else self.NETWORK, **(network or {}))
        self.next_race, self.races = 0, []                     # with growth: when the next race is, and their results
        self.race_gap = 0                                      # text skipped after a race the network as it is won
        self.decide_on = None                                  # the text the decisions are judged on (the queue's validation)
        self._previous = None                                  # validation after the last piece: is the network still learning?
        self.tokens, self.tail, self.report = 0, np.empty(0, np.int64), None
        self.model = None                                      # SimpleNamespace(net=ConceptNet, tables={})
        self.device = 'cpu'
        self._learner = None
        self._saved_steps, self._saved_lr = 0, 3e-3
        self.network_done, self.network_history = False, []
        self.read = 0                                          # tokens of the current text read so far
        self.trace = []                                        # [steps learned, validation NLL] whenever it is measured
        self.watch = None                                      # the pieces the chart is measured on (the queue's validation)
        self.piece = 0

    def prepare_warm_start(self, growth=None, limits=None):
        """Realign queue-relative counters and optional growth ceilings when starting a new corpus from an earlier model."""
        if growth is not None:
            self.growth = bool(growth)
        if limits is not None:
            self.limits = self._clean_limits(limits)
        self.tail = np.empty(0, np.int64)
        self.read = 0
        self.next_race = 0
        self.race_gap = 0
        self.trace = []
        self.races = []
        self._previous = None
        self.network_done = False
        self._learner = None
        self._decide_docs = None

    def _allowed_kinds(self, network):
        """Growth candidates within the configured ceilings, prioritizing pyramidal sub-concept refinement."""
        limits = getattr(self, 'limits', None) or {}
        max_dim = int(limits.get('max_dim') or 0)
        capped = bool(max_dim or limits.get('max_layers') or limits.get('max_heads') or limits.get('max_concepts'))
        # Only the configured ceilings (0: none), no hidden ones.
        max_layers = int(limits.get('max_layers') or 0)
        max_heads = int(limits.get('max_heads') or 0)
        max_concepts = int(limits.get('max_concepts') or 0)
        dim = int(network.config['dim'])
        n_layers = len(network.layers)
        heads = int(network.config['heads'])
        concepts_now = network.config['concepts']
        eligible_layers = [i for i in range(n_layers) if not max_concepts or 2 * int(concepts_now[i]) <= max_concepts]

        def _layer_priority(i):
            if concepts_now[i] <= 1:
                return (0, float(i), 0)
            return (1, concepts_now[i] / (2 ** i), (i - len(self.races)) % n_layers)

        next_layer = min(eligible_layers, key=_layer_priority) if eligible_layers else None
        kinds = ["com'è"]
        if (not max_dim or 2 * dim <= max_dim) and (not capped or max(concepts_now) >= 4 or next_layer is None):
            kinds.append('larghezza')
        if (not max_layers or n_layers + 1 <= max_layers) and (not capped or min(concepts_now) >= 4 or next_layer is None):
            kinds.append('strato')
        if next_layer is not None:
            kinds.append('concetti')
        if not max_heads or 2 * heads <= max_heads:
            kinds.append('teste')
        allowed = getattr(self, 'kinds_allowed', None)
        if allowed is not None:                                 # the growths switched off do not race
            kinds = [kind for kind in kinds if kind == "com'è" or kind in allowed]
            if 'concetti' not in kinds:
                next_layer = None
        return tuple(kinds), next_layer

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
        import copy as _copy
        from copy import copy
        from .concept_net_training import Grower, Network, documents as cut_documents, frequency_groups, held_out, lexical_concepts
        ids = np.asarray(ids, np.int64)
        w = getattr(self, 'weight', None)
        if not len(self.tail):                                  # a new text of the queue begins
            self.read = 0
            if self.model is not None and self.tokens > 0 and self.next_race > len(train):
                self.next_race = 0
                self.race_gap = 0
        if self._learner is None:
            restoring = self.model is not None
            if restoring:                                       # a queue that goes on: the network learned so far
                network = Network.restore(self.model.net)
            else:
                sample = np.asarray(train[:min(len(train), 4_000_000)], np.int64) if len(train) else ids
                frequency = np.bincount(sample, minlength=self.vocabulary) + 1.
                config = {k: v for k, v in self.network.items() if k != 'passes'}
                centres, lex_groups = (lexical_concepts(sample, self.vocabulary, config['dim'], eos=self.eos)
                                       if config.get('grouped') else (None, None))
                if config.get('grouped') and lex_groups is None:
                    lex_groups = frequency_groups(frequency)
                network = Network(dict(config, vocabulary=self.vocabulary), centres=centres,
                                  frequency=frequency / frequency.sum(), groups=lex_groups)
            options = dict(tokens=256, longest=min(256, self.max_context)) if self.device == 'cuda' else {}
            if w is not None:
                options['weight'] = w
            self._learner = Grower(network.to(self.device), passes=self.network.get('passes', True), **options)
            self._learner.KINDS = ()                            # the sizes are fixed
            self._learner.begin = True
            if restoring and self.tokens > 0:
                batch_ref = max(self._learner.learner.tokens, 1)
                saved_steps = int(getattr(self, '_saved_steps', 0)) or max(1, self.tokens // batch_ref)
                saved_lr = float(getattr(self, '_saved_lr', 0.) or self._learner.learner.lr)
                self._learner.learned = int(self.tokens)
                self._learner.learner.seen = int(self.tokens)
                self._learner.learner.steps = saved_steps
                self._learner.learner.lr = saved_lr
                while self._learner.decision <= self._learner.learned:
                    self._learner.decision *= 2
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

        eval_held = (lambda net, d, *extra: held_out(net, d, *extra)) if w is None else (lambda net, d, tokens=None: held_out(net, d, tokens, weight=w))

        def score(net, tokens=None):
            if not held:
                return np.zeros(2)
            return eval_held(net, held, tokens)
        watch_width = self._learner.learner.longest if self.device == 'cuda' else 1 << 30
        shown = self._held(self.watch, watch_width) if self.watch else held  # what the chart is measured on
        started, last = time.monotonic(), [0.]
        end = self.read + len(ids)
        again = end >= len(train) and len(train) <= 2 * max(len(ids), 1) and self.network.get('passes', True)

        def watched():
            check()
            now = time.monotonic()
            if self.report is not None and now - last[0] >= 5:    # aligned with the 5s browser poll
                last[0] = now
                network = self._learner.learner.network
                if shown:                                      # and its validation as it learns, for the chart
                    self._point(min(where(), end), eval_held(network, shown))
                    network.train()
                best = self._learner.best_scores
                self.report('network', 0, now - started, learned=int(where()), again=int(again),
                            held_milli=int(1000 * -best.mean()) if best is not None and len(best) else 0,
                            speed=int((where() - begun[0]) / max(now - started, 1e-9)), network=network.footprint())
        if shown and not self.trace:                            # where the queue starts
            self._point(where(), eval_held(self._learner.learner.network, shown))
        can_grow = self.growth and len(self._allowed_kinds(self._learner.learner.network)[0]) > 1
        if can_grow and held and (self.read >= self.next_race or end >= len(train)):
            self._learner = copy(self._learner)                 # an unfinished piece must not alter the preceding one
            first_race = len(self.races)
            self._race_began = time.monotonic()
            all_docs = cut_documents(ids, self.eos, self._learner.learner.longest, begin=True)
            all_tokens = sum(map(len, all_docs))
            # The race is the whole new piece: a grown copy pays at first for what it changed (concepts split apart)
            # and earns it back only after enough text, so a short window would let the network as it is always win.
            trial_limit = all_tokens
            def take_trial(start_idx):
                picked, total_tok = [], 0
                for d in all_docs[start_idx:]:
                    if picked and total_tok >= trial_limit:
                        break
                    picked.append(d); total_tok += len(d)
                return picked, total_tok, start_idx + len(picked)
            trial_docs, trial_size, cursor = take_trial(0)
            has_rest = cursor < len(all_docs)
            max_rounds = 2 if has_rest else self.MAX_ROUNDS         # it races again on the piece while a growth wins
            self._race_offset = 0
            self._race_piece = len(ids) if not has_rest else max(1, int(len(ids) * trial_size / max(all_tokens, 1)))
            round_number = 1
            best, grew = self._race(ids, held, check, shown, round_number, docs=trial_docs)
            consumed = self._race_piece if has_rest else len(ids)
            while grew and round_number < max_rounds and len(self._allowed_kinds(self._learner.learner.network)[0]) > 1:
                round_number += 1
                if cursor < len(all_docs) and sum(map(len, all_docs[cursor:])) >= max(1024, trial_limit // 2):
                    next_docs, next_size, cursor = take_trial(cursor)
                    self._race_offset = consumed
                    self._race_piece = max(1, min(len(ids) - consumed, int(len(ids) * next_size / max(all_tokens, 1))))
                    best, grew = self._race(ids, held, check, shown, round_number, docs=next_docs)
                    if not self.races[-1]['retained']:
                        consumed += self._race_piece
                else:
                    best, grew = self._race(ids, held, check, shown, round_number, docs=trial_docs)
            rest_docs = all_docs[cursor:] if has_rest else []
            self._race_offset = 0
            if round_number == 1 and self.races[-1]['retained'] and not rest_docs:
                self._learner.learner.lr = max(1e-4, self._learner.learner.lr * .5)
            if rest_docs:
                saved_learner = _copy.deepcopy(self._learner.learner)
                saved_best, saved_scores = best, self._learner.best_scores
                counted.update(learner=self._learner.learner, seen=self._learner.learner.seen, total=consumed)
                rest_size = sum(map(len, rest_docs))
                self._learner.learner.tokens = min(self._learner.batch, max(1, rest_size // 64) if rest_size >= 16384 else max(1, int(rest_size ** .5)))
                self._learner.learner.learn_documents(rest_docs, watched)
                after = eval_held(self._learner.learner.network, held)
                if saved_scores is None or after.mean() > saved_scores.mean():
                    best = self._learner.learner.network.export()
                    self._learner._keep(best, after)
                    self._learner.learned += rest_size
                elif self._better(saved_scores, after):
                    self._learner.learner = saved_learner
                    best, self._learner.best_scores = saved_best, saved_scores
                    self._learner.best = best
                    self._learner.learner.lr = max(1e-4, self._learner.learner.lr * .5)
                else:
                    best = self._learner.learner.network.export()
                    self._learner.learned += rest_size
            counted.update(seen=self._learner.learner.seen, total=len(ids))  # a rereading does not advance the queue
            grew_any = any(not r['retained'] and r['choice'] != "com'è" for r in self.races[first_race:])
            # a growth won: the next piece races again; else the next race further away, 1, 2, 4... pieces, without
            # a ceiling: a network that stops improving races sooner anyway (below)
            self.race_gap = len(ids) if grew_any else max(2 * self.race_gap, len(ids))
            self.next_race = end if grew_any else end + self.race_gap   # it grew: the very next piece races again
            if again and len(self.races) > first_race:
                self._learner._keep(best, self._learner.best_scores)   # restore the stabilized winner if rereading does not help
            self._previous = self._learner.best_scores
        else:
            self._learner.passes = False                        # a piece without a race is read once
            best = self._learner.learn(ids, self.eos, score, watched)
            if self.growth and held and self._learner is not None and not self._learner.done:
                now = eval_held(self._learner.learner.network, held)
                if self._previous is not None and now.mean() <= self._previous.mean():
                    self.next_race = min(self.next_race, end)   # it no longer improves at this size: try to grow
                self._previous = now
        if again and not self._learner.done:                    # ... and with the last one, the whole text again
            self._learner.passes = True
            best = self._learner.learn(np.asarray(train, np.int64), self.eos, score, watched)
        if best is not None:
            self.model = SimpleNamespace(net=best, tables={})
        if shown and self._learner is not None:                 # where the step ends
            self._point(end, eval_held(self._learner.learner.network, shown))
        self.read += len(ids)
        self.tokens += len(ids)
        self.tail = ids[-max(self.max_context, 1):]
        self._saved_steps = int(self._learner.learner.steps)
        self._saved_lr = float(self._learner.learner.lr)
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
        candidate, in a race of growth). Older steps are thinned when the trace grows, while the active step stays intact."""
        import math
        point = [round(tokens / max(self.piece, 1), 3), float(-values.mean())]
        if kind is not None:
            point.extend((kind, round_number))
        self.trace.append(point)
        if len(self.trace) > 1024:
            active_floor = math.floor(self.trace[-1][0] - 1e-9)
            older = [p for p in self.trace if p[0] < active_floor]
            recent = [p for p in self.trace if p[0] >= active_floor]
            self.trace = (older[::2] + recent)[-1024:]

    # ---- growth ------------------------------------------------------------------------------------------
    def _race(self, ids, held, check, shown, round_number, docs=None):
        """Compare the current network and its growth candidates on the same new piece.

        The smallest candidate within one standard error of the best validation score continues.
        The wider candidate learns at half the learning rate because its coordinates were duplicated.
        """
        import copy
        import math
        from .concept_net_training import Learner, documents, held_out
        grower, base = self._learner, self._learner.learner
        check()
        docs = documents(ids, self.eos, base.longest, begin=True) if docs is None else docs
        if not docs:
            return base.network.export(), False
        kinds, next_layer = self._allowed_kinds(base.network)
        if len(kinds) <= 1:
            return base.network.export(), False
        size = sum(map(len, docs))
        if grower.learned and size >= 4 * base.tokens:
            batch = base.tokens
        else:
            batch = min(grower.batch, max(1, size // 64) if size >= 16384 else max(1, math.isqrt(size)))
        warmup = base.warmup if grower.learned else min(base.warmup, max(1, -(-size // batch)))
        if next_layer is not None and base.network.layers[next_layer].g.shape[1] == 1:
            base.network.concepts_to_split(next_layer, docs)
        labels = [f'concetti · strato {next_layer + 1}' if kind == 'concetti' else kind for kind in kinds]
        results, trials = [], []
        w = getattr(self, 'weight', None)
        eval_held = (lambda net, d: held_out(net, d)) if w is None else (lambda net, d: held_out(net, d, weight=w))
        if round_number > 1:
            previous = grower.best_scores
        else:
            previous = eval_held(base.network, held) if self.model is not None else None
        start_chart = (previous if (previous is not None and shown is held)
                       else eval_held(base.network, shown) if (shown and self.model is not None) else None)
        reference_nll = float(-previous.mean()) if previous is not None else None
        projection, candidate_shape = None, None
        current_shape = base.network.footprint()

        def report(number, read, evaluating=False):
            if self.report is None:
                return
            total = len(kinds)
            elapsed = time.monotonic() - self._race_began
            spent = ((round_number - 1) * total + number) * size + read
            self.report('network', 0, elapsed, race=number + 1, race_total=total, race_round=round_number,
                        candidate=labels[number], learned=int(self.read), piece=int(size), done=int(read),
                        speed=int(spent / max(elapsed, 1e-9)), race_candidates=labels,
                        race_scores=[trial['held_out'] for trial in trials], race_evaluating=evaluating,
                        race_projection=projection, race_reference_nll=reference_nll,
                        network=current_shape, candidate_network=candidate_shape)

        report(0, 0)
        offset = getattr(self, '_race_offset', 0)
        for number, (kind, label) in enumerate(zip(kinds, labels)):
            layer = next_layer if kind == 'concetti' else None
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
                    elif layer is not None:
                        network.split_concepts(layer)
                learner = Learner(network, tokens=batch, longest=base.longest, weight=w)
                # Start from the same optimizer, pace and order of training rows.
                learner.steps, learner.seen = base.steps, base.seen
                learner.rng = copy.deepcopy(base.rng)
                learner.lr = base.lr / 2 if kind == 'larghezza' else base.lr   # its coordinates move in pairs
                learner.carry(base, kind, layer)
            learner.anchor = None
            learner.tokens, learner.warmup = batch, warmup
            candidate_shape = learner.network.footprint()
            start = learner.seen
            start_pos = self.read + offset + max(1, self.piece // 1000)
            if start_chart is not None and start_pos < self.read + offset + self._race_piece:
                self._point(start_pos, start_chart, label, round_number)
            self._reported = time.monotonic()
            learner.learn_documents(docs, lambda: (check(), self._racing(
                round_number, number, size, label, learner, start, shown, report)))
            check()
            report(number, size, evaluating=True)
            scores = eval_held(learner.network, held)
            if shown:                                           # the chart can use a different sample
                chart_scores = scores if shown is held else eval_held(learner.network, shown)
                self._point(self.read + offset + self._race_piece, chart_scores, label, round_number)
            count = candidate_shape['parameters']
            results.append((kind, learner, scores, count, layer))
            trials.append(dict(choice=kind, layer=layer+1 if layer is not None else None,
                               heads=int(learner.network.config['heads']), concepts=list(learner.network.config['concepts']),
                               held_out=float(-scores.mean()), parameters=count))
            winner, retained = self._race_choice(results, previous, round_number)
            choice, _, choice_scores, _, _ = winner
            projection = dict(candidate=labels[kinds.index(choice)], held_out=float(-choice_scores.mean()),
                              grows=choice != "com'è" and not retained, retained=retained)
            if retained:
                projection = dict(candidate="com'è", held_out=reference_nll, grows=False, retained=True)
            report(number, size)
        kind, learner, scores, _, layer = winner
        if retained:                                           # neither rereading nor growing improved the reference
            kind, learner, scores, layer = "com'è", base, previous, None
        else:
            grower.learned += size
        learner.anchor = None
        grower.learner = learner
        while grower.decision <= grower.learned:               # the race was the decision of these tokens
            grower.decision *= 2
        net = grower.best if retained and round_number > 1 else learner.network.export()
        grower.best, grower.best_scores = net, scores
        grower.best_learner = copy.deepcopy(learner)
        c = net.config
        self.races.append(dict(tokens=int(self.tokens + len(ids)), choice=kind, round=round_number, retained=retained,
                               reference_nll=reference_nll,
                               layer=layer+1 if layer is not None else None,
                               dim=int(c['dim']), layers=int(c['layers']), heads=int(c['heads']),
                               concepts=[int(k) for k in c['concepts']], held_out=float(-scores.mean()),
                               candidates=trials))
        return net, kind != "com'è"

    def _race_choice(self, results, previous, round_number=1):
        if previous is None:
            eligible = list(results)
        else:
            as_is = results[0][2]
            eligible = []
            for idx, result in enumerate(results):
                scores = result[2]
                if idx == 0:
                    ok = scores.mean() > previous.mean() if round_number == 1 else self._better(scores, previous)
                else:
                    # Grown candidates share the exact mini-batch stream with "com'è", cancelling batch-sampling noise
                    # in their paired difference while never regressing below the starting reference.
                    ok = scores.mean() > previous.mean() and (
                        self._better(scores, previous) or self._better(scores, as_is)
                    )
                if ok:
                    eligible.append(result)
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
        if self.report is not None and now - getattr(self, '_reported', 0.) >= 4:
            self._reported = now
            read = learner.seen - start
            if shown:
                w = getattr(self, 'weight', None)
                vals = held_out(learner.network, shown) if w is None else held_out(learner.network, shown, weight=w)
                self._point(self.read + getattr(self, '_race_offset', 0) + self._race_piece * read / size,
                            vals, kind, round_number)
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

    def explain(self, prompt, produced, decode=None, top=3):
        """How the network wrote every token of an answer, from what verified measures can say (acan/MODEL.md):
        the memory of the document (its weight, and the earlier tokens it copied from), every level (how much the
        chosen token's log-probability falls when the level is closed), the earlier tokens every level's views looked
        at, and the concepts each level's position belonged to most. Positions are indices in prompt + produced."""
        from .concept_net import ConceptNet
        if self.model is None or not len(produced):
            return []
        net = self.model.net
        prompt, produced = [int(t) for t in prompt], [int(t) for t in produced]

        def follow(reader):
            """The chosen tokens' log-probabilities as generation reads them (own words out of the memory)."""
            cursor = NetCursor(reader, self.eos, prompt)
            out, memory = [], []
            for token in produced:
                p = cursor.distribution()
                out.append(float(np.log(max(p[token], 1e-300))))
                inner = cursor.cursor
                if reader is net:
                    if inner.keys:
                        q = net._addresses(inner.state)[0]
                        g, a = net._share(inner.state, np.array(inner.keys) @ q / np.sqrt(len(q)))
                        a = np.asarray(a, float)
                        sources = [(int(j), float(a[j])) for j in np.argsort(-a)[:top] if a[j] > .01]
                        memory.append(dict(weight=float(g), sources=sources))
                    else:
                        memory.append(dict(weight=0., sources=[]))
                cursor.observe(token, remember=False)
            return out, memory

        full, memory = follow(net)
        closed = []
        for i in range(net.layers):                             # every level closed in turn: what it did
            p = dict(net.p)
            key = f'layers.{i}.gate'
            if key not in p:
                closed.append(None); continue
            p[key] = np.zeros_like(p[key])
            closed.append(follow(ConceptNet(net.config, p))[0])
        sequence = np.asarray([self.eos] + prompt + produced, np.int64)
        member, after = [], []
        net._views = []
        finish = net._finish_tabular

        def kept(*args, **kwargs):                             # the state after every level, read with the token concepts
            state = finish(*args, **kwargs); after.append(state); return state
        net._finish_tabular = kept
        try:
            net.states(sequence[:-1], member)
            views = net._views
        finally:
            net._views = None
            net._finish_tabular = finish
        words = [decode([t]) if decode else str(t) for t in prompt + produced]
        steps = []
        for n, token in enumerate(produced):
            at = len(prompt) + n                                # the position of sequence[:-1] that predicts this token
            levels = []
            for i in range(net.layers):
                entry = dict(level=i + 1, effect=None if closed[i] is None else float(full[n] - closed[i][n]))
                if i < len(after):                              # what the network would say after this level
                    p = net.probabilities(after[i][at] @ net.p['final.weight'].T + net.p['final.bias'])[0]
                    entry['says'] = [(decode([int(t)]) if decode else str(int(t)), float(p[t])) for t in np.argsort(-p)[:5]]
                if i < len(views):                              # T x H x W: the last W positions, oldest first
                    w = views[i][at].mean(0); W = len(w)
                    looked = [(at - (W - 1 - k) - 1, float(w[k])) for k in np.argsort(-w)[:top] if at - (W - 1 - k) >= 1]
                    entry['looked'] = [(where, weight) for where, weight in looked if weight > .05]
                if i < len(member):
                    t = member[i][at]                           # H x K
                    best = np.dstack(np.unravel_index(np.argsort(-t, axis=None)[:2], t.shape))[0]
                    entry['concepts'] = [dict(id=f'{i + 1}.{h + 1}.{k + 1}', membership=float(t[h, k])) for h, k in best]
                    # every view's most held concept: what lights up in the graph, token after token
                    heads = []
                    for h in range(t.shape[0]):
                        k = int(np.argmax(t[h]))
                        column = member[i][:, h, k]             # the positions of this text that belong to it most
                        held = [(int(q) - 1, float(column[q])) for q in np.argsort(-column)[:top] if q >= 1]
                        looked_h = []
                        if i < len(views):
                            w = views[i][at][h]; W = len(w)
                            looked_h = [(at - (W - 1 - j) - 1, float(w[j])) for j in np.argsort(-w)[:top]
                                        if at - (W - 1 - j) >= 1 and w[j] > .05]
                        heads.append(dict(id=f'{i + 1}.{h + 1}.{k + 1}', membership=float(t[h, k]), held=held, looked=looked_h))
                    entry['lit'] = heads
                levels.append(entry)
            steps.append(dict(position=len(prompt) + n, token=words[len(prompt) + n], logp=full[n], memory=memory[n], levels=levels))
        return dict(words=words, prompt_length=len(prompt), steps=steps)

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
        learner = self._learner.learner if self._learner is not None else None
        steps = int(learner.steps) if learner is not None else int(getattr(self, '_saved_steps', 0))
        lr_scaled = int(round((learner.lr if learner is not None else getattr(self, '_saved_lr', 3e-3)) * 1e7))
        safe_next = min(max(0, int(self.next_race)), 1 << 60)
        safe_gap = min(max(0, int(self.race_gap)), 1 << 60)
        data = {"net_text": np.array([VERSION, self.vocabulary, self.eos, self.budget_bytes, self.tokens, self.max_context,
                                      int(self.network_done), int(self.growth), safe_next, safe_gap,
                                      max(0, steps), max(1, lr_scaled)], np.int64),
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
        text.next_race = min(max(0, next_race), 1 << 60)
        text.race_gap = min(max(0, values[9] if len(values) > 9 else 0), 1 << 60)
        text._saved_steps = values[10] if len(values) > 10 else max(0, tokens // 4096)
        text._saved_lr = (values[11] / 1e7) if len(values) > 11 and values[11] > 0 else 3e-3
        text.tokens, text.tail, text.network_done = tokens, np.asarray(data["tail"], np.int64), bool(done)
        net = ConceptNet.from_arrays(data, 'net.')
        if net is not None:
            text.model = SimpleNamespace(net=net, tables={})
            text.network = {k: net.config[k] for k in ('dim', 'layers', 'heads', 'concepts', 'window', 'view_window',
                                                       'grouped', 'tabular_hierarchy', 'pure_concepts', 'cache',
                                                       'sentinel', 'res_var_div')
                            if k in net.config}
        return text
