"""An OpenAI-compatible interface to the concept language models of a workspace (/v1/...).

Any client written for the OpenAI API (the ``openai`` library, LangChain, Open WebUI, curl) can use a concept
model by changing the base address and the key:

* ``GET  /v1/models``             the models of the workspace that have a saved checkpoint;
* ``POST /v1/chat/completions``   messages -> answer (with ``stream: true`` the tokens as server-sent events);
* ``POST /v1/completions``        text -> continuation.

Keys. ``Authorization: Bearer acan-...``: a key belongs to one workspace and sees only its projects. Only the
SHA-256 of a key is stored; the key itself is shown once, when it is created. The service token of the
installation never leaves the server.

A model is a project: its id is the project id (``<32 hex>``), optionally ``<id>:<checkpoint>``; without a
checkpoint the last saved one is read. The checkpoint is read from disk, so a project can answer while it keeps
learning. The model is the concept model (acan/lm/concept_lm.py): no GPU and no worker process; the last two
models read are kept in memory.

Sampling: ``temperature`` (0 = the most probable token), ``top_p`` (the smallest set of tokens holding that
probability), ``seed``, ``max_tokens``, ``stop`` (strings: the answer ends before the first one). The chat is
written in the format the conversation data uses (``<|im_start|>role ... <|im_end|>``) and the answer ends at the
end of the turn.
"""
import hashlib
import json
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
from flask import Response, jsonify, request, stream_with_context

PREFIX = 'acan-'
MAX_TOKENS = 4096
ROLES = ('system', 'user', 'assistant')


class APIError(Exception):
    def __init__(self, status, message, kind='invalid_request_error', code=None, param=None):
        super().__init__(message)
        self.status, self.kind, self.code, self.param = status, kind, code, param

    def body(self):
        return dict(error=dict(message=str(self), type=self.kind, param=self.param, code=self.code))


# ---- keys ------------------------------------------------------------------------------------
def _digest(key):
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(store, owner, name):
    name = (name or '').strip()[:80] or 'chiave'
    key = PREFIX+secrets.token_hex(24)
    identifier = uuid.uuid4().hex
    with store.db() as db:
        db.execute('INSERT INTO api_keys(id, owner, digest, name, created, revoked) VALUES(?,?,?,?,?,0)',
                   (identifier, owner, _digest(key), name, time.time()))
    return dict(id=identifier, name=name, key=key, hint=key[:9]+'…'+key[-4:])


def list_keys(store, owner):
    with store.db() as db:
        rows = db.execute('SELECT id, name, created, revoked, last_used FROM api_keys WHERE owner=? ORDER BY created',
                          (owner,)).fetchall()
    return [dict(row) for row in rows]


def revoke_key(store, owner, identifier):
    with store.db() as db:
        changed = db.execute('UPDATE api_keys SET revoked=1 WHERE id=? AND owner=?', (identifier, owner)).rowcount
    if not changed:
        raise KeyError(identifier)


def owner_of(store, header):
    """The workspace of the key in an ``Authorization: Bearer`` header."""
    match = re.fullmatch(r'Bearer\s+(\S+)', header or '')
    if not match or not match.group(1).startswith(PREFIX):
        raise APIError(401, 'Chiave API mancante o non valida: Authorization: Bearer acan-…', 'invalid_request_error',
                       'invalid_api_key')
    with store.db() as db:
        row = db.execute('SELECT id, owner FROM api_keys WHERE digest=? AND revoked=0', (_digest(match.group(1)),)).fetchone()
        if row is None:
            raise APIError(401, 'Chiave API non valida o revocata.', 'invalid_request_error', 'invalid_api_key')
        db.execute('UPDATE api_keys SET last_used=? WHERE id=?', (time.time(), row['id']))
    return row['owner']


# ---- models -------------------------------------------------------------------------------------
class Models:
    """The concept models read from their checkpoints, the last ``size`` kept in memory."""

    def __init__(self, store, size=2):
        self.store, self.size = store, size
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def available(self, owner):
        out = []
        for job in self.store.jobs(owner):
            if job['config'].get('architecture') == 'concepts' and job['checkpoint']:
                out.append(job)
        return out

    def resolve(self, owner, name):
        if not isinstance(name, str) or not name:
            raise APIError(400, 'Indica il modello («model»): l\'id di un progetto, da GET /v1/models.', param='model')
        identifier, _, checkpoint = name.partition(':')
        if not re.fullmatch('[0-9a-f]{32}', identifier):
            raise APIError(404, f'Modello «{name}» non trovato.', code='model_not_found', param='model')
        try:
            job = self.store.job(identifier, owner)
        except KeyError:
            raise APIError(404, f'Modello «{name}» non trovato.', code='model_not_found', param='model') from None
        if job['config'].get('architecture') != 'concepts':
            raise APIError(400, 'Solo i modelli a concetti rispondono dall\'API.', param='model')
        checkpoint = checkpoint or job['checkpoint']
        if not checkpoint or not re.fullmatch('[a-z0-9-]{1,128}', checkpoint):
            raise APIError(404, 'Il modello non ha ancora un checkpoint salvato.', code='model_not_found', param='model')
        path = self.store.directory('jobs', identifier)/'checkpoints'/checkpoint
        if not (path/'manifest.json').exists():
            raise APIError(404, f'Checkpoint «{checkpoint}» non trovato.', code='model_not_found', param='model')
        return job, path

    def load(self, owner, name):
        job, path = self.resolve(owner, name)
        key = str(path)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return job, self.cache[key]
        from transformers import AutoTokenizer
        from .concept_lm import load
        model, _ = load(path)
        tokenizer = AutoTokenizer.from_pretrained(str(self.store.directory('jobs', job['id'])/'tokenizer'),
                                                  local_files_only=True, trust_remote_code=False)
        with self.lock:
            self.cache[key] = (model, tokenizer)
            while len(self.cache) > self.size:
                self.cache.popitem(last=False)
        return job, (model, tokenizer)


# ---- requests --------------------------------------------------------------------------------
def chat_text(messages):
    """Messages in the chat format of the conversation data, ending where the assistant writes."""
    if not isinstance(messages, list) or not messages:
        raise APIError(400, '«messages» deve essere un elenco non vuoto.', param='messages')
    parts = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ROLES:
            raise APIError(400, 'Ogni messaggio ha «role» (system, user o assistant) e «content».', param='messages')
        content = message.get('content')
        if isinstance(content, list):                            # content as parts: only text is read
            content = ''.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        if not isinstance(content, str):
            raise APIError(400, 'Il contenuto di un messaggio deve essere testo.', param='messages')
        parts.append(f"<|im_start|>{message['role']}\n{content.strip()}<|im_end|>\n")
    return ''.join(parts)+'<|im_start|>assistant\n'


def options(body):
    if not isinstance(body, dict):
        raise APIError(400, 'Il corpo della richiesta deve essere un oggetto JSON.')
    if body.get('n', 1) != 1:
        raise APIError(400, 'Si può chiedere una sola risposta per volta (n = 1).', param='n')
    most = body.get('max_completion_tokens', body.get('max_tokens', 256))
    if isinstance(most, bool) or not isinstance(most, int) or not 1 <= most <= MAX_TOKENS:
        raise APIError(400, f'«max_tokens» da 1 a {MAX_TOKENS}.', param='max_tokens')
    temperature = body.get('temperature', 1.)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise APIError(400, '«temperature» da 0 a 2.', param='temperature')
    top_p = body.get('top_p', 1.)
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise APIError(400, '«top_p» fra 0 (escluso) e 1.', param='top_p')
    seed = body.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise APIError(400, '«seed» deve essere un intero.', param='seed')
    stop = body.get('stop') or []
    stop = [stop] if isinstance(stop, str) else stop
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(item, str) and item for item in stop):
        raise APIError(400, '«stop»: una stringa o fino a 4 stringhe.', param='stop')
    return dict(max_tokens=most, temperature=float(temperature), top_p=float(top_p), seed=seed, stop=stop,
                stream=bool(body.get('stream', False)))


def draw(p, temperature, top_p, rng):
    """The next token from the reading p: the most probable at temperature 0, else drawn from p^(1/T) within top_p."""
    if temperature == 0:
        return int(np.argmax(p))
    q = np.exp(np.log(np.maximum(p, 1e-300))/temperature)
    q /= q.sum()
    if top_p < 1:
        order = np.argsort(-q, kind='stable')
        keep = order[:int(np.searchsorted(np.cumsum(q[order]), top_p))+1]
        mask = np.zeros_like(q); mask[keep] = q[keep]
        q = mask/mask.sum()
    return int(rng.choice(len(q), p=q))


def write(model, tokenizer, prompt, settings):
    """Yields (new text, None) as the model writes, then ('', (finish reason, prompt tokens, answer tokens)).

    The text never contains a stop string nor what follows it: while a stop string may be starting, its first
    characters are held back.
    """
    from .probes import stop_tokens
    ids = list(tokenizer(prompt)['input_ids'])
    if not ids:
        raise APIError(400, 'Il testo da continuare è vuoto.')
    end = set(stop_tokens(tokenizer))
    rng = np.random.default_rng(settings['seed'])
    hold = max([len(item) for item in settings['stop']]+[1])-1
    produced, sent, reason, text = [], 0, 'length', ''
    for _ in range(settings['max_tokens']):
        token = draw(model.distribution(np.asarray(ids+produced, np.int64)), settings['temperature'], settings['top_p'], rng)
        if token in end:
            reason = 'stop'
            break
        produced.append(token)
        text = tokenizer.decode(produced, skip_special_tokens=True)
        found = [text.find(item, max(sent-hold, 0)) for item in settings['stop']]
        found = [at for at in found if at >= 0]
        if found:
            cut = min(found)
            if cut > sent:
                yield text[sent:cut], None
            yield '', ('stop', len(ids), len(produced))
            return
        ready = len(text)-hold
        if text.endswith('\ufffd'):                              # a character cut in the middle of its bytes
            ready = min(ready, len(text)-1)
        if ready > sent:
            yield text[sent:ready], None
            sent = ready
    if len(text) > sent:
        yield text[sent:].replace('\ufffd', ''), None
    yield '', (reason, len(ids), len(produced))


def register(app, store):
    """The /v1 routes on the service application (they authorize with their own keys)."""
    models = Models(store)
    app.openai_models = models

    @app.errorhandler(APIError)
    def api_error(exc):
        return jsonify(exc.body()), exc.status

    def owner():
        return owner_of(store, request.headers.get('Authorization', ''))

    @app.get('/v1/models')
    def list_models():
        workspace = owner()
        data = [dict(id=job['id'], object='model', owned_by='acan', created=int(job.get('updated') or 0),
                     name=job['config'].get('name', ''), checkpoint=job['checkpoint'])
                for job in models.available(workspace)]
        return jsonify(object='list', data=data)

    @app.get('/v1/models/<path:name>')
    def one_model(name):
        job, _ = models.resolve(owner(), name)
        return jsonify(id=name, object='model', owned_by='acan', created=int(job.get('updated') or 0),
                       name=job['config'].get('name', ''))

    def complete(kind):
        workspace = owner()
        body = request.get_json(silent=True)
        settings = options(body)
        if kind == 'chat':
            prompt = chat_text(body.get('messages'))
        else:
            prompt = body.get('prompt')
            if isinstance(prompt, list) and len(prompt) == 1:
                prompt = prompt[0]
            if not isinstance(prompt, str) or not prompt:
                raise APIError(400, '«prompt» deve essere un testo non vuoto.', param='prompt')
        if len(prompt) > 200_000:
            raise APIError(400, 'Testo troppo lungo (al massimo 200.000 caratteri).')
        name = body.get('model')
        _, (model, tokenizer) = models.load(workspace, name)
        identifier = ('chatcmpl-' if kind == 'chat' else 'cmpl-')+uuid.uuid4().hex
        created = int(time.time())
        steps = write(model, tokenizer, prompt, settings)
        if not settings['stream']:
            text, finish = '', None
            for piece, end in steps:
                text += piece
                finish = end or finish
            reason, prompt_tokens, completion_tokens = finish
            usage = dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                         total_tokens=prompt_tokens+completion_tokens)
            if kind == 'chat':
                choice = dict(index=0, message=dict(role='assistant', content=text), finish_reason=reason, logprobs=None)
                return jsonify(id=identifier, object='chat.completion', created=created, model=name,
                               choices=[choice], usage=usage)
            choice = dict(index=0, text=text, finish_reason=reason, logprobs=None)
            return jsonify(id=identifier, object='text_completion', created=created, model=name,
                           choices=[choice], usage=usage)

        def events():
            def event(payload):
                return 'data: '+json.dumps(payload, ensure_ascii=False)+'\n\n'
            if kind == 'chat':
                yield event(dict(id=identifier, object='chat.completion.chunk', created=created, model=name,
                                 choices=[dict(index=0, delta=dict(role='assistant', content=''), finish_reason=None)]))
            for piece, end in steps:
                if end:
                    reason, prompt_tokens, completion_tokens = end
                    usage = dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                 total_tokens=prompt_tokens+completion_tokens)
                    if kind == 'chat':
                        yield event(dict(id=identifier, object='chat.completion.chunk', created=created, model=name,
                                         choices=[dict(index=0, delta={}, finish_reason=reason)]))
                    else:
                        yield event(dict(id=identifier, object='text_completion', created=created, model=name,
                                         choices=[dict(index=0, text='', finish_reason=reason)]))
                    if (body.get('stream_options') or {}).get('include_usage'):
                        yield event(dict(id=identifier, object='chat.completion.chunk' if kind == 'chat' else 'text_completion',
                                         created=created, model=name, choices=[], usage=usage))
                    break
                if kind == 'chat':
                    yield event(dict(id=identifier, object='chat.completion.chunk', created=created, model=name,
                                     choices=[dict(index=0, delta=dict(content=piece), finish_reason=None)]))
                else:
                    yield event(dict(id=identifier, object='text_completion', created=created, model=name,
                                     choices=[dict(index=0, text=piece, finish_reason=None)]))
            yield 'data: [DONE]\n\n'
        response = Response(stream_with_context(events()), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response

    @app.post('/v1/chat/completions')
    def chat_completions():
        return complete('chat')

    @app.post('/v1/completions')
    def completions():
        return complete('text')
