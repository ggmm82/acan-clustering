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


def resume_after_restart(store):
    """After a restart of the server (scheduled maintenance), long work continues by itself.

    A long training with ``auto_resume`` that was interrupted, or an interrupted data preparation,
    is started again from its last saved point, one at a time. A pause chosen by the user is never
    undone, and a failure is not retried.
    """
    if store.busy():
        return
    from .manager import launch, resume_build
    with store.db() as db:
        build = db.execute("SELECT id, owner FROM builds WHERE status='interrupted' ORDER BY created LIMIT 1").fetchone()
        jobs = db.execute("SELECT id, owner FROM jobs WHERE status='interrupted' ORDER BY updated").fetchall()
    try:
        for row in jobs:
            job = store.job(row['id'], row['owner'])
            if job['config'].get('auto_resume'):
                # A process killed again and again at the same step (memory, a bad file) must not loop:
                # after three restarts without progress the project waits for the user.
                marker = store.directory('jobs', job['id'])/'auto-resume.json'
                seen = json.loads(marker.read_text()) if marker.exists() else {}
                step = job['progress'].get('step', 0)
                count = seen.get('count', 0)+1 if seen.get('step') == step else 1
                if count > 3:
                    continue
                marker.write_text(json.dumps(dict(step=step, count=count)))
                launch(store, job)
                return
        if build is not None:
            resume_build(store, build['id'], build['owner'])
    except ValueError:
        pass                                                    # the worker became busy meanwhile


def _ensure_concept_graph(store, job):
    """Populate network_graph and network_concepts from the latest saved checkpoint when missing on an idle job."""
    progress = job.get('progress') or {}
    if (job['config'].get('architecture') != 'concepts'
            or job['status'] in ACTIVE
            or not job.get('checkpoint')
            or (progress.get('network_graph') or {}).get('nodes')):
        return job
    directory = store.directory('jobs', job['id'])
    latest = directory / 'checkpoints' / job['checkpoint']
    if not latest.exists():
        return job
    try:
        from transformers import AutoTokenizer
        from .concept_lm import default_budget, load
        from .concept_trainer import judged_pieces
        from .native import resolve
        from .research_data import Blocks
        tok_path, _ = resolve(store, job, job['config'])
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=False, local_files_only=True)
        budget = int(job['config'].get('concept_memory_mb') or 0) * (1 << 20) or default_budget()
        loaded, _ = load(latest, budget_bytes=budget)
        net = getattr(getattr(loaded, 'model', None), 'net', None)
        if net is not None:
            blocks = Blocks(directory, job['config']['sequence_length'], job['config']['seed'],
                            job['config']['validation_batches'])
            tune, _ = judged_pieces(blocks, job['config']['validation_batches'], disjoint=True)
            docs = [p[:512] for p in tune[:8] if len(p) >= 2]
            if docs:
                progress = dict(progress,
                                network_concepts=net.describe(docs, tokenizer.decode),
                                network_graph=net.graph(docs, tokenizer.decode))
                store.update(job['id'], progress=progress)
                job['progress'] = progress
    except Exception:
        pass
    return job


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
        if request.path == '/healthz' or request.path.startswith('/v1/'):
            return None                                         # /v1 authorizes with the keys of the workspaces
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
        resume_after_restart(store)
        jobs = store.jobs(request.workspace)
        summaries=request.args.get('summary')=='1'
        selected=request.args.get('selected','')
        for job in jobs:
            detailed=not summaries or job['id']==selected
            if not detailed:
                job['progress']={key:job['progress'][key] for key in ('step','tokens_trained','validation_nll','best_validation_nll','tokens_per_second','graft') if key in job['progress']}
            else:                                               # polled every few seconds: the graph of concepts goes with the footprint
                job['progress']={key:value for key,value in job['progress'].items() if key!='network_graph'}
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
        job=_ensure_concept_graph(store, store.job(identifier,request.workspace))
        job['snapshots']=snapshots(store,job)
        job['history']=[{key:row[key] for key in ('step','training_nll','validation_nll') if key in row}
                        for row in store.history(identifier,request.workspace,overview=True)]
        job['history_view']='full_overview'
        return jsonify(job)

    @app.get('/capabilities')
    def capabilities():
        import torch
        from .resources import cuda_usable
        return jsonify(cuda=cuda_usable(), torch=torch.__version__, base_models=BASE_MODELS,
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

    @app.post('/queues')
    def create_queue():
        """A learning queue: one project per text, in order; each starts from the weights of the previous one."""
        import uuid as _uuid
        from .resources import check_profile
        import torch
        body = request.get_json() or {}
        base, corpora = body.get('base'), body.get('corpora')
        if not isinstance(base, dict) or base.get('protocol') != 'controlled-v1':
            raise ValueError('Una coda richiede il protocollo controllato.')
        if not isinstance(corpora, list) or not 1 <= len(corpora) <= 200:
            raise ValueError('Scegli da 1 a 200 testi per la coda.')
        queue = _uuid.uuid4().hex[:8]
        jobs = [queue_item(base, corpus, queue, order) for order, corpus in enumerate(corpora)]
        for config in jobs:
            check_profile(config, torch.cuda.is_available())
        created = [store.create_job(request.workspace, config) for config in jobs]
        message = 'Coda creata: il primo testo è partito.'
        try:
            launch(store, store.job(created[0]['id'], request.workspace))
        except ValueError as exc:
            message = f'Coda creata ma non avviata: {exc} Usa Riprendi sul primo testo quando il worker è libero.'
        return jsonify(queue=queue, jobs=[store.job(job['id'], request.workspace) for job in created], message=message)

    @app.post('/queues/<queue>/add')
    def add_to_queue(queue):
        """Another text at the end of an existing queue, with the same settings (plus optional growth/ceiling overrides)."""
        payload = request.get_json() or {}
        corpus = payload.get('corpus_id')
        members = [job for job in store.jobs(request.workspace) if (job['config'].get('series') or {}).get('id') == queue]
        if not members:
            raise KeyError(queue)
        last = max(members, key=lambda job: job['config']['series']['order'])
        base = {key: value for key, value in last['config'].items() if key not in ('series', 'corpus_id', 'name', 'warm_start')}
        for key in ('concept_network_growth', 'concept_max_dim', 'concept_max_layers', 'concept_max_heads', 'concept_max_concepts'):
            if key in payload and payload[key] is not None:
                base[key] = payload[key]
        base['name'] = last['config']['name'].split(' · ')[0]
        job = store.create_job(request.workspace, queue_item(base, corpus, queue, last['config']['series']['order']+1))
        if not any(other['status'] in ACTIVE or other['status'] == 'created' for other in members) and not store.busy():
            try:
                launch(store, store.job(job['id'], request.workspace))
            except ValueError:
                pass
        return jsonify(store.job(job['id'], request.workspace))

    def queue_item(base, corpus_id, queue, order):
        from .research_options import validate_options as controlled_options
        corpus = store.corpus(corpus_id, request.workspace, ready=True)
        config = controlled_options(dict(base, corpus_id=corpus['id'], warm_start=order > 0,
                                         name=f"{base.get('name', 'Coda')} · {order+1} · {corpus['name']}"))
        config['series'] = dict(id=queue, order=order, variant='queue', label=corpus['name'])
        return config

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

    @app.get('/jobs/<identifier>/footprint')
    def job_footprint(identifier):
        """The sizes of a project while it learns: parameters, speed, memory and disk."""
        import shutil
        job = _ensure_concept_graph(store, store.job(identifier, request.workspace))
        directory = store.directory('jobs', identifier)
        size = lambda path: sum(item.stat().st_size for item in path.rglob('*') if item.is_file()) if path.exists() else 0
        root = directory/'checkpoints'
        kinds = {kind: [path for path in root.glob(kind+'-*')] if root.exists() else [] for kind in ('step', 'milestone', 'best')}
        latest = root/job['checkpoint'] if job['checkpoint'] else None
        corpus = store.directory('corpora', job['config']['corpus_id'])/'corpus'
        disk = shutil.disk_usage(store.root)
        progress = job['progress'] if request.args.get('graph') == '1' else {   # the graph of concepts (megabytes) only when asked
            key: value for key, value in (job['progress'] or {}).items() if key != 'network_graph'}
        return jsonify(progress=progress, config=job['config'], status=job['status'],
                       checkpoint_bytes=size(latest) if latest and latest.exists() else 0,
                       checkpoints={kind: dict(count=len(paths), bytes=sum(size(path) for path in paths)) for kind, paths in kinds.items()},
                       token_bytes=sum(path.stat().st_size for path in directory.glob('*.bin')),
                       corpus_bytes=corpus.stat().st_size if corpus.exists() else 0,
                       disk_free=disk.free, disk_total=disk.total)

    @app.get('/jobs/<identifier>/answers')
    def job_answers(identifier):
        from .probes import read_answers
        store.job(identifier, request.workspace)
        return jsonify(read_answers(store.directory('jobs', identifier)))

    @app.get('/jobs/<identifier>/answer-paths')
    def job_answer_paths(identifier):
        store.job(identifier, request.workspace)
        path = store.directory('jobs', identifier)/'answer_paths.json'
        return jsonify(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {})

    @app.get('/dialogue-data')
    def dialogue_catalog():
        from .dialogue_data import catalog
        return jsonify(catalog())

    @app.post('/builds')
    def create_build():
        from .manager import request_build
        return jsonify(id=request_build(store, request.workspace, request.get_json() or {}))

    @app.get('/builds')
    def list_builds():
        return jsonify(store.builds(request.workspace))

    @app.get('/builds/<identifier>')
    def get_build(identifier):
        return jsonify(store.build(identifier, request.workspace))

    @app.post('/builds/<identifier>/pause')
    def pause_build(identifier):
        item = store.build(identifier, request.workspace)
        if item['status'] in ('queued', 'running'):
            store.update_build(identifier, status='pausing', phase='Pausa richiesta: salvo il punto raggiunto')
        return jsonify(store.build(identifier, request.workspace))

    @app.post('/builds/<identifier>/resume')
    def resume_build(identifier):
        from .manager import resume_build as resume
        return jsonify(resume(store, identifier, request.workspace))

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
        source={key:value for key,value in job['config'].items() if key not in ('series','warm_start','control_of')}
        if source.get('architecture')=='concepts':
            # The concept model is compared with a network born from zero on the same text, tokenizer and pieces.
            if not job['checkpoint']:
                raise ValueError('Avvia prima il progetto: il controllo usa il suo tokenizer.')
            source.update(architecture='native',routing_profile='concept-graph-v2',native_size='micro',passes=1)
            source['total_steps']=1000
        if source.get('architecture')=='native':
            # A native model is controlled by the same model born from zero with ordinary attention,
            # on the same text with the same tokenizer, steps and seed.
            if source.get('native_attention')=='dense':
                raise ValueError('Questo progetto è già un controllo con attenzione normale.')
            if not job['checkpoint']:
                raise ValueError('Avvia prima il progetto: il controllo usa il suo tokenizer.')
            source['passes']=0
            config=controlled_options(dict(source,native_attention='dense',name=CONTROL_NAME+source['name']))
        else:
            config=controlled_options(dict(source,architecture='dense-control',name=CONTROL_NAME+source['name']))
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

    @app.get('/jobs/<identifier>/project-archive')
    def project_archive(identifier):
        from .project_archive import export_project
        archive = export_project(store, request.workspace, identifier)
        return send_file(archive, as_attachment=True, download_name='acan-project.zip', conditional=False,
                         mimetype='application/zip')

    @app.post('/project-imports')
    def project_import_begin():
        from .project_archive import begin_import
        return jsonify(begin_import(store, request.workspace, (request.get_json() or {}).get('size')))

    @app.post('/project-imports/<identifier>/<action>')
    def project_import_action(identifier, action):
        from .project_archive import append_import, cancel_import, finish_import
        if action == 'chunk':
            offset = int(request.headers.get('X-Offset', '-1'))
            return jsonify(append_import(store, request.workspace, identifier, offset, request.get_data()))
        if action == 'finish':
            job = finish_import(store, request.workspace, identifier)
            return jsonify(id=job['id'], status=job['status'], name=job['config']['name'])
        if action == 'cancel':
            cancel_import(store, request.workspace, identifier)
            return jsonify(cancelled=True)
        raise ValueError('Operazione di trasferimento non valida.')

    @app.get('/jobs/<identifier>/export')
    def export_model(identifier):
        from .exporting import export_bundle
        job = store.job(identifier, request.workspace)
        path = export_bundle(store, job, request.args.get('checkpoint', ''))
        response = send_file(path, mimetype='application/zip', as_attachment=True,
                             download_name='acan-model-'+path.name)
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/api-keys')
    def api_keys():
        from .openai_api import list_keys
        return jsonify(keys=list_keys(store, request.workspace))

    @app.post('/api-keys')
    def new_api_key():
        from .openai_api import create_key
        return jsonify(create_key(store, request.workspace, (request.get_json() or {}).get('name')))

    @app.post('/api-keys/<identifier>/revoke')
    def revoke_api_key(identifier):
        from .openai_api import revoke_key
        revoke_key(store, request.workspace, identifier)
        return jsonify(ok=True)

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

    from .openai_api import register as openai_routes
    openai_routes(app, store)
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
