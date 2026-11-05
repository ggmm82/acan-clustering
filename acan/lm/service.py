"""Internal LM service, separated from the lightweight Dash/Hostinger process."""
import hmac
import json
import os
import signal
import threading
import time
from flask import Flask, jsonify, request

from .manager import launch, request_generation, snapshots
from .store import ACTIVE, Store, owner_key
from .trainer import BASE_MODELS, validate_options


def create_app(root=None, token=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 2*1024*1024
    store = Store(root or os.environ.get('ACAN_LM_DATA', '/data'))
    token = token or os.environ.get('ACAN_LM_SERVICE_TOKEN', '')
    if len(token) < 32:
        raise ValueError('Configurare ACAN_LM_SERVICE_TOKEN (almeno 32 caratteri), uguale nel frontend e nel worker.')
    store.reconcile()
    app.store = store

    @app.before_request
    def authorize():
        if request.path == '/healthz':
            return None
        supplied = request.headers.get('Authorization', '')
        if not hmac.compare_digest(supplied, 'Bearer ' + token):
            return jsonify(error='Servizio non autorizzato.'), 401
        request.workspace = owner_key(request.headers.get('X-ACAN-Workspace', ''))

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(KeyError)
    def missing(exc):
        return jsonify(error='Progetto o corpus non trovato.'), 404

    @app.get('/healthz')
    def health():
        return jsonify(status='ok')

    @app.get('/state')
    def state():
        store.reconcile()
        jobs = store.jobs(request.workspace)
        for job in jobs:
            job['snapshots'] = snapshots(store, job)
            job['history'] = store.history(job['id'], request.workspace)
        return jsonify(jobs=jobs, corpora=store.corpora(request.workspace), busy=store.busy())

    @app.get('/capabilities')
    def capabilities():
        import torch
        return jsonify(cuda=torch.cuda.is_available(), torch=torch.__version__, base_models=BASE_MODELS,
                       profile='acan-lm-causal-prefix-v1', minicpm_supported=False)

    @app.get('/public-corpora')
    def public_catalog():
        from .public_corpus import catalog
        return jsonify(catalog())

    @app.post('/public-experiments')
    def public_experiment():
        from .public_corpus import experiment
        body = request.get_json()
        if not isinstance(body, dict) or set(body) != {'dataset', 'preset'}:
            raise ValueError('Specificare soltanto dataset e profilo della prova pubblica.')
        return jsonify(experiment(store, request.workspace, body['dataset'], body['preset']))

    @app.post('/corpora')
    def begin_upload():
        body = request.get_json()
        return jsonify(store.create_corpus(request.workspace, body['name'], body['size']))

    @app.post('/corpora/<identifier>/chunks')
    def chunk(identifier):
        return jsonify(store.append_corpus(identifier, request.workspace, int(request.headers['X-Offset']), request.get_data()))

    @app.post('/corpora/<identifier>/finish')
    def finish(identifier):
        return jsonify(store.finish_corpus(identifier, request.workspace))

    @app.post('/jobs')
    def create_job():
        config = validate_options(request.get_json())
        return jsonify(store.create_job(request.workspace, config))

    @app.post('/jobs/<identifier>/start')
    def start(identifier):
        job = store.job(identifier, request.workspace)
        if job['status'] in ACTIVE or job['status'] == 'completed':
            raise ValueError('Progetto già attivo o completato.')
        launch(store, job)
        return jsonify(store.job(identifier, request.workspace))

    @app.post('/jobs/<identifier>/pause')
    def pause(identifier):
        return jsonify(store.pause(identifier, request.workspace))

    @app.post('/jobs/<identifier>/generate')
    def generation(identifier):
        job = store.job(identifier, request.workspace)
        return jsonify(id=request_generation(store, job, request.workspace, request.get_json()))

    @app.get('/generations/<identifier>')
    def generation_state(identifier):
        with store.db() as db:
            row = db.execute('SELECT * FROM generations WHERE id=? AND owner=?', (identifier, request.workspace)).fetchone()
        if row is None:
            raise KeyError(identifier)
        result = dict(row)
        if result['result']:
            result['result'] = json.loads(result['result'])
        result.pop('request')
        return jsonify(result)

    return app


def main():
    from waitress import create_server
    app = create_app()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    server = create_server(app, host='0.0.0.0', port=int(os.environ.get('ACAN_LM_PORT','8000')), threads=8)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not stop.wait(1):
        pass
    with app.store.db() as db:
        db.execute("UPDATE jobs SET pause=1,status='pausing' WHERE status IN ('queued','preparing','running')")
    deadline = time.monotonic() + 100
    while app.store.busy() and time.monotonic() < deadline:
        time.sleep(.5)
    server.close()


if __name__ == '__main__':
    main()
