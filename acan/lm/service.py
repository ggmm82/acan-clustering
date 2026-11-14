"""Internal LM service, separated from the lightweight Dash/Hostinger process."""
import hmac
import json
import os
import signal
import threading
import time
from flask import Flask, jsonify, request, send_file

from acan import __version__
from .manager import launch, request_generation, snapshots
from .store import ACTIVE, Store, owner_key
from .trainer import BASE_MODELS, validate_options


CONTROL_NAME = 'Controllo denso · '


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
        return jsonify(status='ok',version=__version__,build_revision=os.environ.get('ACAN_BUILD_REVISION','development'))

    @app.get('/state')
    def state():
        store.reconcile()
        jobs = store.jobs(request.workspace)
        summaries=request.args.get('summary')=='1'
        selected=request.args.get('selected','')
        for job in jobs:
            detailed=not summaries or job['id']==selected
            if not detailed:
                job['progress']={key:job['progress'][key] for key in ('step','tokens_trained') if key in job['progress']}
            job['snapshots'] = snapshots(store, job) if detailed else []
            job['history'] = [{key:row[key] for key in ('step','training_nll','validation_nll') if key in row}
                              for row in store.history(job['id'],request.workspace,overview=True)] if detailed else []
            job['history_view'] = 'full_overview'
        result=dict(jobs=jobs, corpora=store.corpora(request.workspace), busy=store.busy())
        if summaries:result['detail_job']=selected
        return jsonify(result)

    @app.get('/jobs/<identifier>/history')
    def history(identifier):
        try:
            start=int(request.args.get('start','0'))
            limit=int(request.args.get('limit','500'))
            end=int(request.args['end']) if 'end' in request.args else None
        except ValueError:
            raise ValueError('Intervallo dello storico non valido.') from None
        rows=store.history(identifier,request.workspace,start=start,limit=limit,end=end)
        if request.args.get('plot')=='1':
            rows=[{key:row[key] for key in ('step','training_nll','validation_nll') if key in row} for row in rows]
        return jsonify(rows=rows,next_step=rows[-1]['step']+1 if rows else None)

    @app.get('/jobs/<identifier>/view')
    def job_view(identifier):
        job=store.job(identifier,request.workspace)
        job['snapshots']=snapshots(store,job)
        job['history']=[{key:row[key] for key in ('step','training_nll','validation_nll') if key in row}
                        for row in store.history(identifier,request.workspace,overview=True)]
        job['history_view']='full_overview'
        return jsonify(job)

    @app.get('/capabilities')
    def capabilities():
        import torch
        return jsonify(cuda=torch.cuda.is_available(), torch=torch.__version__, base_models=BASE_MODELS,
                       profile='acan-lm-causal-prefix-v1', minicpm_supported=False, controlled_training=True, distillation=True, best_checkpoint=True, model_export=True,attention_fractions=[.5,1.],build_revision=os.environ.get('ACAN_BUILD_REVISION','development'))

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

    @app.post('/public-corpora/<dataset>/import')
    def import_public_corpus(dataset):
        """A public text stored in the workspace, without starting a project."""
        from .public_corpus import import_corpus
        corpus, source = import_corpus(store, request.workspace, dataset)
        return jsonify(corpus=corpus, source=source)

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
        options=request.get_json()
        if not isinstance(options,dict):
            raise ValueError('Configurazione di training non valida.')
        if options.get('protocol')=='controlled-v1':
            from .research_options import validate_options as controlled_options
            config=controlled_options(options)
            from .resources import check_profile
            import torch
            check_profile(config,torch.cuda.is_available())
        else:
            config=validate_options(options)
        corpus=store.corpus(config['corpus_id'],request.workspace,ready=True)
        origin=store.directory('corpora',corpus['id'])/'public-source.json'
        if origin.exists() and json.loads(origin.read_text()).get('preprocessing')=='wikitext-articles-official-splits-v1' and config.get('protocol')!='controlled-v1':
            raise ValueError('Questo corpus contiene split ufficiali: usa il protocollo controllato per tenere il test fuori dal training.')
        return jsonify(store.create_job(request.workspace, config))

    @app.post('/jobs/<identifier>/rename')
    def rename(identifier):
        return jsonify(store.rename(identifier, request.workspace, (request.get_json() or {}).get('name')))

    @app.post('/jobs/<identifier>/delete')
    def delete(identifier):
        store.delete(identifier, request.workspace)
        return jsonify(deleted=identifier)

    @app.get('/jobs/<identifier>/evaluations')
    def job_evaluations(identifier):
        return jsonify(store.evaluations(identifier, request.workspace))

    @app.post('/series')
    def create_series():
        """Several projects on the same text, run one after the other by the worker."""
        import uuid as _uuid
        from .manager import series_configs
        from .research_options import validate_options as controlled_options
        from .resources import check_profile
        import torch
        body = request.get_json() or {}
        base = body.get('base')
        if not isinstance(base, dict) or base.get('protocol') != 'controlled-v1':
            raise ValueError('Una serie richiede il protocollo controllato.')
        prepared = []
        for item, label, options in series_configs(base, body.get('variants')):
            config = controlled_options(options)
            check_profile(config, torch.cuda.is_available())
            store.corpus(config['corpus_id'], request.workspace, ready=True)
            prepared.append((item, label, config))
        if store.busy():
            raise ValueError('Il worker è occupato: metti in pausa il progetto attivo prima di avviare una serie.')
        series = _uuid.uuid4().hex[:8]
        jobs = []
        for order, (item, label, config) in enumerate(prepared):
            config['series'] = dict(id=series, order=order, variant=item, label=label)
            jobs.append(store.create_job(request.workspace, config))
        reference = next((job for job in jobs if job['config']['series']['variant'] == 'chosen'), None)
        for job in jobs:                                      # the normal-attention run controls the chosen settings
            if reference and job['config']['series']['variant'] == 'dense':
                store.set_config(job['id'], dict(job['config'], control_of=reference['id'],
                                                 name=CONTROL_NAME + reference['config']['name']))
        launch(store, store.job(jobs[0]['id'], request.workspace))
        return jsonify(series=series, jobs=[store.job(job['id'], request.workspace) for job in jobs])

    @app.post('/small-tests')
    def create_small_test():
        from .manager import request_small_test
        return jsonify(id=request_small_test(store, request.workspace, request.get_json() or {}))

    @app.get('/small-tests')
    def list_small_tests():
        return jsonify(store.small_tests(request.workspace))

    @app.get('/small-tests/<identifier>')
    def small_test(identifier):
        return jsonify(store.small_test(identifier, request.workspace))

    @app.post('/small-tests/<identifier>/cancel')
    def cancel_small_test(identifier):
        test = store.small_test(identifier, request.workspace)
        if test['status'] == 'running':
            store.update_small_test(identifier, status='cancelling', phase='Arresto richiesto')
        return jsonify(store.small_test(identifier, request.workspace))

    @app.post('/jobs/<identifier>/start')
    def start(identifier):
        job = store.job(identifier, request.workspace)
        if job['status'] in ACTIVE or job['status'] == 'completed':
            raise ValueError('Progetto già attivo o completato.')
        if job['config'].get('protocol')=='controlled-v1':
            from .resources import check_profile
            import torch
            check_profile(job['config'],torch.cuda.is_available())
        launch(store, job)
        return jsonify(store.job(identifier, request.workspace))

    @app.post('/jobs/<identifier>/pause')
    def pause(identifier):
        return jsonify(store.pause(identifier, request.workspace))

    @app.post('/jobs/<identifier>/evaluate')
    def evaluate_model(identifier):
        from .manager import request_evaluation
        job=store.job(identifier,request.workspace)
        return jsonify(id=request_evaluation(store,job,request.workspace,request.get_json()))

    @app.post('/evaluations/<identifier>/cancel')
    def cancel_evaluation(identifier):
        with store.db() as db:
            row=db.execute('SELECT * FROM evaluations WHERE id=? AND owner=?',(identifier,request.workspace)).fetchone()
            if row is None:raise KeyError(identifier)
            if row['status']=='running':db.execute("UPDATE evaluations SET status='cancelling' WHERE id=?",(identifier,))
        return jsonify(status='cancellation requested')

    @app.get('/evaluations/<identifier>')
    def evaluation_result(identifier):
        with store.db() as db:
            row=db.execute('SELECT * FROM evaluations WHERE id=? AND owner=?',(identifier,request.workspace)).fetchone()
        if row is None:
            raise KeyError(identifier)
        value=dict(row)
        if value['result']:value['result']=json.loads(value['result'])
        value.pop('request')
        return jsonify(value)

    @app.post('/jobs/<identifier>/dense-control')
    def dense_control(identifier):
        job=store.job(identifier,request.workspace)
        if job['config'].get('protocol')!='controlled-v1':
            raise ValueError('Il controllo richiede un progetto con split train/validazione/test separati.')
        if store.busy():
            raise ValueError('Metti in pausa il progetto attivo prima di avviare il controllo.')
        from .research_options import validate_options as controlled_options
        config=controlled_options(dict(job['config'],architecture='dense-control',name=CONTROL_NAME+job['config']['name']))
        config['control_of']=job['id']                         # the interface draws both on one chart
        control=store.create_job(request.workspace,config)
        launch(store,control)
        return jsonify(control)

    @app.route('/jobs/<identifier>/layers', methods=['GET', 'POST'])
    def inspect_layers(identifier):
        from .inspection import queue, latest
        job = store.job(identifier, request.workspace)
        if request.method == 'POST':
            queue(store, job)
        return jsonify(latest(store, job))

    @app.get('/jobs/<identifier>/diagnostics')
    def diagnostics(identifier):
        from .diagnostics import report
        return jsonify(report(store,store.job(identifier,request.workspace),token))

    @app.get('/jobs/<identifier>/export')
    def export_model(identifier):
        from .exporting import export_bundle
        job = store.job(identifier, request.workspace)
        path = export_bundle(store, job, request.args.get('checkpoint', ''))
        response = send_file(path, mimetype='application/zip', as_attachment=True,
                             download_name='acan-model-'+path.name)
        response.headers['Cache-Control'] = 'no-store'
        return response

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
