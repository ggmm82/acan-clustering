"""CPU character language prototype with competitive concept routing.

The routing follows softmax(gamma*log mass - beta*weighted squared distance).
Categorical prototypes, position relevances and emissions are optimized by
next-token cross entropy. This is an experimental extension, not ACAN v8's
tabular CP objective or evidence of large-language-model capability.
"""
import hashlib
import json
import math
import time

import numpy as np

from acan.core.learner import _NumpyAdam


def softmax(x, axis=-1):
    exp = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def demo_corpus():
    return '\n'.join(f'il {subject} {verb} {obj}.'
                     for subject in ('gatto', 'cane', 'bimbo', 'vento')
                     for verb in ('vede', 'segue', 'cerca', 'trova')
                     for obj in ('il sole', 'la luna', 'una casa', 'un fiore'))


class ConceptLanguageModel:
    FORMAT = 'acan-concept-character-lm-v1'
    SMOOTHING = 1e-4

    def __init__(self, context=6, slots=12, seed=42, beta=8.0, gamma=1.0):
        if not 1 <= int(context) <= 16 or not 2 <= int(slots) <= 64:
            raise ValueError('Context must be 1–16; concepts must be 2–64.')
        if not 0 <= int(seed) < 2**32:
            raise ValueError('Seed must be an unsigned 32-bit integer.')
        if not 0 < float(beta) <= 100 or not 0 < float(gamma) <= 4:
            raise ValueError('Invalid routing scale.')
        self.context, self.slots, self.seed = int(context), int(slots), int(seed)
        self.beta, self.gamma = float(beta), float(gamma)
        self.history, self.report = [], {}
        self.params = {}

    def _set_vocab(self, chars):
        if len(chars) > 126 or len(set(chars)) != len(chars) or any(len(c) != 1 for c in chars):
            raise ValueError('At most 126 distinct training characters are supported.')
        self.chars = list(chars)
        self.vocab = ['<UNK>', '<EOS>'] + self.chars
        self.lookup = {c: i + 2 for i, c in enumerate(self.chars)}
        self.v = len(self.vocab)
        self.bos = self.v

    def _windows(self, docs, limit=4096):
        contexts, targets = [], []
        for doc in docs:
            prefix = [self.bos] * self.context
            for char in list(doc) + [None]:
                token = 1 if char is None else self.lookup.get(char, 0)
                contexts.append(prefix[-self.context:].copy())
                targets.append(token)
                prefix.append(token)
        x, y = np.asarray(contexts, dtype=np.int64), np.asarray(targets, dtype=np.int64)
        if len(y) > limit:
            indices = np.random.default_rng(self.seed).choice(len(y), limit, replace=False)
            x, y = x[indices], y[indices]
        return x, y

    def _forward(self, x):
        onehot = np.eye(self.v + 1)[x]
        prototypes = softmax(self.params['concepts'])
        relevance = softmax(self.params['relevance'])
        prior = softmax(self.params['mass'])
        emissions = softmax(self.params['emissions'])
        dist = (prototypes**2).sum(-1)[None] + 1 - 2 * np.einsum('npv,kpv->nkp', onehot, prototypes)
        distance = np.einsum('nkp,kp->nk', dist, relevance)
        shifted_mass = self.params['mass'] - self.params['mass'].max()
        log_prior = shifted_mass - np.log(np.exp(shifted_mass).sum())
        routing = softmax(self.gamma * log_prior[None] - self.beta * distance)
        raw = routing @ emissions
        probabilities = (1 - self.SMOOTHING) * raw + self.SMOOTHING / self.v
        return probabilities, (onehot, prototypes, relevance, prior, emissions, dist, routing)

    def loss_and_grad(self, x, y):
        probabilities, cache = self._forward(x)
        onehot, concepts, relevance, prior, emissions, dist, routing = cache
        loss = float(-np.log(probabilities[np.arange(len(y)), y]).mean())
        dp = np.zeros_like(probabilities)
        dp[np.arange(len(y)), y] = -(1 - self.SMOOTHING) / (len(y) * probabilities[np.arange(len(y)), y])
        de = routing.T @ dp
        dr = dp @ emissions.T
        dl = routing * (dr - (routing * dr).sum(1, keepdims=True))
        dd = -self.beta * dl
        dconcepts = 2 * relevance[:, :, None] * (
            concepts * dd.sum(0)[:, None, None] - np.einsum('nk,npv->kpv', dd, onehot))
        drelevance = np.einsum('nk,nkp->kp', dd, dist)
        dmass = self.gamma * dl.sum(0)
        grads = {
            'concepts': concepts * (dconcepts - (dconcepts * concepts).sum(-1, keepdims=True)),
            'relevance': relevance * (drelevance - (drelevance * relevance).sum(-1, keepdims=True)),
            'mass': dmass - prior * dmass.sum(),
            'emissions': emissions * (de - (de * emissions).sum(-1, keepdims=True)),
        }
        return loss, grads

    def _score(self, x, y):
        nll, correct = 0., 0
        for start in range(0, len(y), 256):
            p, _ = self._forward(x[start:start + 256])
            target = y[start:start + 256]
            nll -= float(np.log(p[np.arange(len(target)), target]).sum())
            correct += int((p.argmax(1) == target).sum())
        nll /= len(y)
        return dict(nll=nll, bits_per_token=nll / math.log(2), perplexity=math.exp(nll),
                    accuracy=correct / len(y), tokens=len(y), unknown_tokens=int((y == 0).sum()))

    def fit(self, corpus, epochs=25, lr=.04, progress=None, cancelled=None):
        if not isinstance(corpus, str) or not 100 <= len(corpus) <= 40000 or not corpus.strip():
            raise ValueError('Provide 100–40,000 characters of UTF-8 text.')
        if not 1 <= int(epochs) <= 100 or not .0001 <= float(lr) <= .2:
            raise ValueError('Epochs must be 1–100; learning rate .0001–.2.')
        started = time.monotonic()
        # Group identical documents BEFORE splitting; never split overlapping windows.
        docs = [line.strip() for line in corpus.replace('\r\n', '\n').split('\n') if line.strip()]
        unique = sorted(set(docs))
        rng = np.random.default_rng(self.seed)
        if len(unique) >= 10:
            rng.shuffle(unique)
            cut1, cut2 = max(1, int(.7 * len(unique))), max(2, int(.85 * len(unique)))
            train, valid, test = unique[:cut1], unique[cut1:cut2], unique[cut2:]
            split = 'deduplicated documents, 70/15/15'
        else:
            cut1, cut2 = int(.7 * len(corpus)), int(.85 * len(corpus))
            train, valid, test = [corpus[:cut1]], [corpus[cut1:cut2]], [corpus[cut2:]]
            split = 'contiguous segments, contexts reset at each boundary'
        self._set_vocab(sorted(set(''.join(train))))
        x, y = self._windows(train)
        xv, yv = self._windows(valid)
        xt, yt = self._windows(test)
        if min(len(y), len(yv), len(yt)) < 5:
            raise ValueError('Each split needs at least five target tokens.')
        sample = x[rng.choice(len(x), self.slots, replace=len(x) < self.slots)]
        self.params = {
            'concepts': .1 * rng.normal(size=(self.slots, self.context, self.v + 1)) + 4 * np.eye(self.v + 1)[sample],
            'relevance': np.zeros((self.slots, self.context)),
            'mass': np.zeros(self.slots),
            'emissions': .1 * rng.normal(size=(self.slots, self.v)),
        }
        self.unigram = (np.bincount(y, minlength=self.v) + .5) / (len(y) + .5 * self.v)
        bigram = np.full((self.v + 1, self.v), .5)
        np.add.at(bigram, (x[:, -1], y), 1)
        self.bigram = bigram / bigram.sum(1, keepdims=True)
        optimizer = _NumpyAdam(self.params, lr=float(lr))
        best, best_loss, stale, best_epoch = None, float('inf'), 0, 0
        self.history = []
        for epoch in range(int(epochs)):
            order = rng.permutation(len(y))
            for start in range(0, len(y), 64):
                if cancelled and cancelled():
                    raise ValueError('Training cancelled.')
                ix = order[start:start + 64]
                loss, gradients = self.loss_and_grad(x[ix], y[ix])
                if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients.values()):
                    raise ValueError('Non-finite optimization result.')
                optimizer.step(gradients)
            validation = self._score(xv, yv)['nll']
            self.history.append(dict(epoch=epoch + 1, train_nll=self._score(x, y)['nll'], validation_nll=validation))
            if progress:
                progress(epoch + 1)
            if validation < best_loss - 1e-5:
                best = {key: value.copy() for key, value in self.params.items()}
                best_loss, stale, best_epoch = validation, 0, epoch + 1
            else:
                stale += 1
                if stale >= 8:
                    break
        self.params = best
        test_score = self._score(xt, yt)
        unigram_nll = float(-np.log(self.unigram[yt]).mean())
        bigram_nll = float(-np.log(self.bigram[xt[:, -1], yt]).mean())
        train_sequences = set(map(tuple, np.column_stack((x, y))))
        overlap = sum(tuple(row) in train_sequences for row in np.column_stack((xt, yt))) / len(yt)
        self.report = dict(model=self.FORMAT, objective='next-token cross entropy; not tabular CP',
                           split=split, train_tokens=len(y), validation_tokens=len(yv), test=test_score,
                           input_documents=len(docs), unique_documents=len(unique),
                           unigram_perplexity=math.exp(unigram_nll), bigram_perplexity=math.exp(bigram_nll),
                           selected_epoch=best_epoch,
                           train_test_window_overlap=overlap,
                           test_used_for_selection=False, seconds=time.monotonic() - started,
                           train_sha256=hashlib.sha256('\n'.join(train).encode()).hexdigest(),
                           validation_sha256=hashlib.sha256('\n'.join(valid).encode()).hexdigest(),
                           test_sha256=hashlib.sha256('\n'.join(test).encode()).hexdigest(),
                           parameters=sum(value.size for value in self.params.values()))
        return self

    def generate(self, prompt='', length=120, temperature=.8, top_k=0, top_p=.95, seed=7):
        if len(prompt) > 2000 or not 1 <= int(length) <= 512:
            raise ValueError('Prompt is limited to 2,000 characters; output to 512 tokens.')
        if not .05 <= float(temperature) <= 2 or not 0 < float(top_p) <= 1 or not 0 <= int(top_k) <= self.v:
            raise ValueError('Invalid sampling parameters.')
        rng = np.random.default_rng(int(seed))
        tokens = [self.bos] * self.context + [self.lookup.get(c, 0) for c in prompt]
        text, trace = prompt, []
        for _ in range(int(length)):
            probabilities, cache = self._forward(np.array([tokens[-self.context:]]))
            p = softmax(np.log(probabilities[0]) / float(temperature))
            indices = np.argsort(p)[::-1]
            if int(top_k):
                indices = indices[:int(top_k)]
            cumulative = np.cumsum(p[indices] / p[indices].sum())
            keep = max(1, int(np.searchsorted(cumulative, float(top_p), side='left')) + 1)
            indices = indices[:keep]
            weights = p[indices] / p[indices].sum()
            token = int(rng.choice(indices, p=weights))
            route = cache[-1][0]
            contribution = route * cache[4][:, token]
            winner = int(contribution.argmax())
            trace.append(dict(token=self.vocab[token], model_probability=float(probabilities[0, token]),
                              sampling_probability=float(weights[np.where(indices == token)[0][0]]),
                              routing_concept=int(route.argmax()), routing_weight=float(route.max()),
                              concept=winner, concept_contribution=float(contribution[winner] / max(contribution.sum(), 1e-300))))
            if token == 1:
                break
            text += self.chars[token - 2] if token >= 2 else '\ufffd'
            tokens.append(token)
        return text, trace

    def to_json(self):
        return json.dumps(dict(format=self.FORMAT, context=self.context, slots=self.slots,
                               seed=self.seed, beta=self.beta, gamma=self.gamma, chars=self.chars,
                               parameters={key: value.tolist() for key, value in self.params.items()},
                               history=self.history, report=self.report), allow_nan=False)

    @classmethod
    def from_json(cls, text):
        if len(text.encode()) > 8 * 1024 * 1024:
            raise ValueError('Model JSON exceeds 8 MiB.')
        obj = json.loads(text)
        if obj.get('format') != cls.FORMAT:
            raise ValueError('Not an experimental ACAN character model.')
        model = cls(obj['context'], obj['slots'], obj['seed'], obj['beta'], obj['gamma'])
        model._set_vocab(obj['chars'])
        shapes = dict(concepts=(model.slots, model.context, model.v + 1),
                      relevance=(model.slots, model.context), mass=(model.slots,), emissions=(model.slots, model.v))
        for key, shape in shapes.items():
            value = np.asarray(obj['parameters'][key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all() or np.abs(value).max() > 1e6:
                raise ValueError('Invalid model parameter array: ' + key)
            model.params[key] = value
        model.history, model.report = obj.get('history', []), obj.get('report', {})
        return model
