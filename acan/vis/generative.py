"""Lightweight Dash client for the separate, persistent generative worker."""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import request as flask_request, jsonify, Response, stream_with_context


MODELS = ['HuggingFaceTB/SmolLM2-135M', 'HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-1.7B']


def backend(workspace, path, method='GET', data=None, raw=False, headers=None, timeout=30):
    endpoint = os.environ.get('ACAN_LM_URL', '').rstrip('/')
    token = os.environ.get('ACAN_LM_SERVICE_TOKEN', '')
    if not endpoint or not token:
        raise ValueError('Worker generativo non configurato. Avvia il servizio dedicato e imposta ACAN_LM_URL e ACAN_LM_SERVICE_TOKEN sul frontend.')
    if not isinstance(workspace, str) or not re.fullmatch('[0-9a-f]{64}', workspace):
        raise ValueError('Workspace non ancora pronto: attendi il caricamento della pagina.')
    body = data if raw else json.dumps(data).encode() if data is not None else None
    fields = {'Authorization':'Bearer '+token, 'X-ACAN-Workspace':workspace,
              'Content-Type':'application/octet-stream' if raw else 'application/json'}
    fields.update(headers or {})
    try:
        with urlopen(Request(endpoint+path, data=body, headers=fields, method=method), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            message = json.load(exc).get('error', str(exc))
        except (ValueError, KeyError):
            message = str(exc)
        raise ValueError(message) from None
    except URLError:
        raise ValueError('Worker generativo non raggiungibile.') from None


layout = dbc.Container([
    html.H4('Laboratorio generativo · ACAN-LM Hybrid', className='mt-3'),
    html.P('Continued pre-training con LoRA e WSD su pesi SmolLM2. Nel profilo predefinito il 50% superiore usa il porting ACAN causale corretto; il profilo al 100% è sperimentale. Qualità linguistica e riduzione della cache sono da misurare: non sono garantite dal grafting.'),
    dcc.Store(id='lm-workspace', storage_type='local'), dcc.Store(id='lm-state'),
    dcc.Store(id='lm-active-job', storage_type='session'),
    dcc.Store(id='lm-upload-state'), dcc.Store(id='lm-generation-id'), dcc.Store(id='lm-browser-ready'), dcc.Store(id='lm-evaluation-id'), dcc.Store(id='lm-evaluation-report'),
    html.Div(id='lm-connection', role='status'),
    dbc.Button('Aggiorna progetti', id='lm-refresh', outline=True, className='my-2'),
    dbc.Button('Salva chiave di recupero', id='lm-export-key', color='link'),
    html.Div(id='lm-recovery-input'), dcc.Download(id='lm-key-download'),
    html.P('Il workspace resta nel browser dopo refresh e riavvii del servizio. Salva la chiave per recuperare i progetti da un altro browser; chi possiede la chiave può gestirli. Nessun account richiesto.', className='small text-muted'),
    dbc.Card(dbc.CardBody([
        html.H5('Avvio guidato · dati pubblici'),
        html.P('Non serve caricare file: scegli un testo pubblico e un budget. Scarichiamo e verifichiamo il corpus, prepariamo SmolLM2-135M con ACAN e avviamo il training. Prova guidata: ibrido al 50% senza teacher; per i profili avanzati usa la creazione manuale sotto. Non modifica il progetto già esistente.'),
        dbc.Row([
            dbc.Col([dbc.Label('Corpus pubblico'), dcc.Dropdown(id='lm-public-dataset', value='dante-it', clearable=False, options=[{'label':'Divina Commedia · italiano letterario', 'value':'dante-it'}, {'label':'Tiny Shakespeare · inglese', 'value':'tiny-shakespeare'}, {'label':'WikiText-2 · inglese · split ufficiali', 'value':'wikitext-2'}])], md=6),
            dbc.Col([dbc.Label('Budget di apprendimento'), dcc.Dropdown(id='lm-public-preset', value='check', clearable=False, options=[{'label':'Prova breve · 20 passi', 'value':'check'}, {'label':'Esercizio · 200 passi', 'value':'practice'}])], md=6),
        ]),
        html.Div(id='lm-public-info', className='my-2'),
        dbc.Button('Scarica e avvia la prova', id='lm-public-start', disabled=True),
        html.Div(id='lm-public-status', role='status', className='my-2'),
        html.P('Il primo avvio scarica anche i pesi del modello. La durata dipende dalla macchina: pochi passi non equivalgono a pochi secondi. Nessun ciclo illimitato, nessun benchmark tabellare automatico.', className='small text-muted'),
        html.Ol([
            html.Li('Attendi lo stato running: ogni passo legge testo e aggiorna i pesi addestrabili.'),
            html.Li('Osserva la validazione ai checkpoint: un valore più basso è migliore; confrontalo con il modello originale.'),
            html.Li('Premi Pausa e salva, attendi paused, seleziona l’ultimo checkpoint e prova il prompt suggerito.'),
            html.Li('Riprendi lo stesso progetto oppure attendi completed: il budget scelto termina automaticamente.'),
        ]),
    ]), className='my-3'),
    dbc.Row([
        dbc.Col([html.H5('1 · Corpus'), html.Div(id='lm-file-upload'),
                 html.Div(id='lm-upload-status', role='status'), html.P('TXT: paragrafi separati da righe vuote. JSONL: un documento text per record; split train/validation/test facoltativo.', className='small text-muted'), dcc.Dropdown(id='lm-corpus', placeholder='Seleziona corpus caricato')], md=6),
        dbc.Col([html.H5('2 · Modello di partenza'), dcc.Dropdown(id='lm-base', options=MODELS, value=MODELS[0], clearable=False),
                 dbc.Label('Nome progetto'), dbc.Input(id='lm-name', value='ACAN-LM experiment'),
                 dbc.Label('Passi di ottimizzazione totali'), dbc.Input(id='lm-steps', type='number', min=1, max=10000000, value=1000)], md=6),
    ]),
    dbc.Accordion([dbc.AccordionItem([
        html.P('Ogni passo aggiorna pesi/ottimizzatore dopo l’accumulo completo. Il budget WSD e questi parametri restano fissi durante la ripresa.'),
        dbc.Row([dbc.Col([dbc.Label(label), dbc.Input(id=id, type='number', value=value, min=lo, max=hi)], md=4, className='mb-2')
                 for label,id,value,lo,hi in [('Token per blocco','lm-length',128,16,2048),('Batch','lm-batch',1,1,8),
                     ('Accumulo gradienti','lm-accum',8,1,256),('Learning rate','lm-lr',.0001,.0000001,.01),
                     ('Slot','lm-slots',16,2,64),('Finestra locale','lm-window',16,1,128),
                     ('Rank LoRA','lm-rank',8,2,64),('Checkpoint ogni N passi','lm-save',100,1,10000)]]),
        dbc.Label('Attenzione ACAN nei nuovi progetti'),
        dcc.Dropdown(id='lm-fraction',value=.5,clearable=False,options=[{'label':'50% dei livelli · ibrido predefinito','value':.5},{'label':'100% dei livelli · profilo sperimentale, qualità da misurare','value':1.}]),
        dbc.Checkbox(id='lm-distill', value=False, label='Distillazione: imita anche il modello originale (più memoria e tempo, facoltativa su CPU)'),
        html.P('Nuovi progetti: documenti separati in training/validazione/test, duplicati esatti rimossi, miglior checkpoint conservato e pausa automatica dopo tre controlli senza miglioramenti sufficienti. Il progetto legacy conserva il metodo originale.', className='small text-muted'),
        dbc.Label('Dispositivo'), dcc.Dropdown(id='lm-device', options=['auto','cuda','cpu'], value='auto', clearable=False),
        html.P('Salvataggio anche ogni 5 minuti e su pausa. Un arresto improvviso può richiedere di rifare i passi dopo l’ultimo checkpoint; i checkpoint completi precedenti restano intatti. GPU non disponibile: solo il profilo CPU 135M per verifiche, molto più lento.', className='small text-muted'),
    ], title='Impostazioni avanzate')], start_collapsed=True, className='my-3'),
    dbc.Button('Crea e avvia training', id='lm-start', className='m-1'),
    html.Hr(), html.H5('3 · Progetti persistenti'),
    dcc.Dropdown(id='lm-job', placeholder='Scegli un progetto per controllarlo o riprenderlo'),
    dbc.Button('Pausa e salva', id='lm-pause', color='warning', className='m-1'),
    dbc.Button('Riprendi dal checkpoint', id='lm-resume', className='m-1'),
    html.Div(id='lm-action-status', role='status'), html.Div(id='lm-progress'), dcc.Graph(id='lm-loss'),
    dbc.Accordion([dbc.AccordionItem([html.Pre(id='lm-debug-view', style={'whiteSpace':'pre-wrap'}),
        dbc.Button('Scarica diagnostica', id='lm-debug-download'), dcc.Download(id='lm-debug-file'),
        dcc.Store(id='lm-debug-data')], title='Diagnostica del progetto e log')], start_collapsed=True),
    dcc.Interval(id='lm-poll', interval=5000, disabled=True),
    html.Hr(), html.H5('4 · Prova un checkpoint durante il percorso'),
    html.P('Sulla stessa GPU la generazione richiede il training in pausa. Il checkpoint è immutabile: la prova non modifica quanto appreso. Dopo la prova puoi riprendere il training.'),
    dcc.Dropdown(id='lm-checkpoint', placeholder='Checkpoint verificabili: migliore conservato e ultimi tre'),
    dbc.Textarea(id='lm-prompt', value='', placeholder='Scrivi un inizio di frase nella lingua del corpus', rows=3),
    dbc.Label('Nuovi token'), dbc.Input(id='lm-new-tokens', type='number', value=64, min=1, max=256),
    dbc.Button('Genera dal checkpoint', id='lm-generate', className='my-2'),
    html.Div(id='lm-generation-status', role='status'), html.Pre(id='lm-output', style={'whiteSpace':'pre-wrap'}),
    html.Div(id='lm-generation-metrics'),
    dbc.Button('Scarica modello ACAN/LoRA e struttura', id='lm-export-model', outline=True, className='my-2'),
    html.Div(id='lm-export-status', role='status'),
    html.Hr(),html.H5('5 · Verifica su test separato e confronto con il modello originale'),
    html.P('Su richiesta e con training in pausa: stessi token di test per entrambi i modelli, NLL/perplessità, accuratezza del prossimo token, tempi di prefill/decoding e cache totale misurata. Su CPU il confronto è piccolo: quattro blocchi, contesti 32 e 128, otto nuovi token.'),
    dbc.Label('Budget del confronto'),
    dcc.Dropdown(id='lm-eval-profile',value='quick',clearable=False,options=[{'label':'Rapido CPU: 4 blocchi test, contesti 32/128','value':'quick'},{'label':'Contesto lungo CPU: 1 blocco test, 128/1024 (lento)','value':'long'},{'label':'GPU: 64 blocchi test, 128/512/2048','value':'wide','disabled':True}]),
    dbc.Button('Confronta checkpoint e originale', id='lm-evaluate', className='m-1'),
    dbc.Button('Arresta confronto',id='lm-eval-stop',outline=True,color='warning',className='m-1'),
    dbc.Button('Scarica confronto', id='lm-eval-download', outline=True, className='m-1'),dcc.Download(id='lm-eval-file'),
    html.Div(id='lm-eval-status',role='status'),html.Div(id='lm-eval-view'),
    dbc.Accordion([dbc.AccordionItem([
        html.P('Il controllo addestra SmolLM2 con LoRA e attenzione densa, sugli stessi split e con lo stesso budget del progetto selezionato. Serve a distinguere problemi dei dati/ottimizzazione da effetti specifici di ACAN. Avvia un altro training: su CPU può durare a lungo.'),
        dbc.Button('Avvia controllo denso con stesso budget', id='lm-dense-control'),html.Div(id='lm-control-status',role='status')
    ],title='Controllo scientifico aggiuntivo (facoltativo)')],start_collapsed=True),
])


def register(app):
    @app.server.post('/generative-upload/<action>')
    def upload_proxy(action):
        try:
            workspace = flask_request.headers.get('X-ACAN-Workspace')
            if action == 'begin':
                result = backend(workspace, '/corpora', 'POST', flask_request.get_json())
            elif action in ('chunk','finish'):
                identifier = flask_request.headers.get('X-Corpus-ID', '')
                if not re.fullmatch('[0-9a-f]{32}', identifier):
                    raise ValueError('Identificatore corpus non valido.')
                if action == 'chunk':
                    if flask_request.content_length is None or flask_request.content_length > 2*1024*1024:
                        raise ValueError('Chunk troppo grande.')
                    result = backend(workspace, f'/corpora/{identifier}/chunks', 'POST', flask_request.get_data(), True,
                                     {'X-Offset':flask_request.headers.get('X-Offset','0')})
                else:
                    result = backend(workspace, f'/corpora/{identifier}/finish', 'POST')
            else:
                raise ValueError('Operazione non valida.')
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.server.get('/generative-export/<identifier>/<checkpoint>')
    def export_proxy(identifier, checkpoint):
        if not re.fullmatch('[0-9a-f]{32}', identifier) or not re.fullmatch('[a-z0-9-]{1,128}', checkpoint):
            return jsonify(error='Identificatore non valido.'), 400
        workspace = flask_request.headers.get('X-ACAN-Workspace', '')
        if not re.fullmatch('[0-9a-f]{64}', workspace):
            return jsonify(error='Workspace non pronto.'), 400
        endpoint = os.environ.get('ACAN_LM_URL', '').rstrip('/')
        token = os.environ.get('ACAN_LM_SERVICE_TOKEN', '')
        if not endpoint or not token:
            return jsonify(error='Worker non configurato.'), 400
        try:
            upstream = urlopen(Request(endpoint+f'/jobs/{identifier}/export?checkpoint={checkpoint}',
                headers={'Authorization':'Bearer '+token, 'X-ACAN-Workspace':workspace}), timeout=60)
        except (HTTPError, URLError):
            return jsonify(error='Export non disponibile: aggiorna il worker e verifica il checkpoint.'), 400
        def chunks():
            try:
                while True:
                    chunk = upstream.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()
        return Response(stream_with_context(chunks()), mimetype='application/zip',
                        headers={'Cache-Control':'no-store', 'Content-Disposition':'attachment; filename="acan-model.zip"'})

    app.clientside_callback("""async function(click, workspace, job, checkpoint) {
        if (!click) return window.dash_clientside.no_update;
        if (!workspace || !job || !checkpoint) return 'Scegli un progetto e un checkpoint.';
        try {
            const response = await fetch('/generative-export/'+encodeURIComponent(job)+'/'+encodeURIComponent(checkpoint),
                {headers:{'X-ACAN-Workspace':workspace}});
            if (!response.ok) throw new Error((await response.json()).error);
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a'); link.href=url; link.download='acan-model-'+checkpoint+'.zip';
            document.body.appendChild(link); link.click(); link.remove(); setTimeout(()=>URL.revokeObjectURL(url),60000);
            return 'Scaricato: pesi appresi, struttura e riferimento esatto alla base. I pesi originali saranno scaricati al primo caricamento.';
        } catch(error) {return 'Export non disponibile: '+error.message;}
    }""", Output('lm-export-status','children'), Input('lm-export-model','n_clicks'),
       State('lm-workspace','data'), State('lm-job','value'), State('lm-checkpoint','value'), prevent_initial_call=True)

    app.clientside_callback('''function(tab, workspace) {
        if (tab !== 'Generative') return [window.dash_clientside.no_update, true, window.dash_clientside.no_update];
        if (!workspace) {
            const bytes = new Uint8Array(32); crypto.getRandomValues(bytes);
            workspace = Array.from(bytes, x=>x.toString(16).padStart(2,'0')).join('');
        }
        window.acanLmWorkspace = workspace;
        return [workspace, false, Date.now()];
    }''', Output('lm-workspace','data'), Output('lm-poll','disabled'), Output('lm-browser-ready','data'),
       Input('tabs','value'), State('lm-workspace','data'))

    @app.callback(Output('lm-state','data'), Output('lm-connection','children'),
                  Input('lm-poll','n_intervals'), Input('lm-refresh','n_clicks'), Input('lm-workspace','data'), Input('lm-upload-state','data'), Input('lm-active-job','data'),
                  State('tabs','value'))
    def poll(interval, refresh, workspace, upload, active, tab):
        if tab != 'Generative' or not workspace:
            return dash.no_update, dash.no_update
        try:
            state = backend(workspace, '/state')
            try:
                state['public_catalog'] = backend(workspace, '/public-corpora')
            except ValueError:
                state['public_catalog'] = {}
            if dash.ctx.triggered_id == 'lm-active-job':
                state['select_job'] = active
            capabilities = backend(workspace, '/capabilities')
            state['capabilities']=capabilities
            hardware = 'CUDA disponibile' if capabilities['cuda'] else 'CPU · profilo di verifica 135M'
            return state, f'Worker collegato · {hardware} · persistenza attiva · una sola operazione alla volta.'
        except ValueError as exc:
            return None, dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-corpus','options'), Output('lm-job','options'),
                  Output('lm-corpus','value'), Output('lm-job','value'),
                  Input('lm-state','data'), State('lm-active-job','data'),
                  State('lm-upload-state','data'), State('lm-corpus','value'), State('lm-job','value'))
    def options(state, active, upload, corpus, selected):
        if not state:
            return [], [], None, None
        corpora = [item for item in state['corpora'] if item['status']=='ready']
        corpus_ids = {item['id'] for item in corpora}
        uploaded = (upload or {}).get('id')
        chosen_corpus = corpus if corpus in corpus_ids else uploaded if uploaded in corpus_ids else None
        job_ids = {item['id'] for item in state['jobs']}
        chosen_job = state.get('select_job')
        if chosen_job not in job_ids:
            chosen_job = selected if selected in job_ids else active if active in job_ids else None
        return ([dict(label=f"{item['name']} ({item['received']/1048576:.1f} MiB)", value=item['id']) for item in corpora],
                [dict(label=f"{item['config']['name']} · {item['status']} · passo {item['progress'].get('step',0)}",value=item['id']) for item in state['jobs']],
                chosen_corpus, chosen_job)

    @app.callback(Output('lm-base','options'),Output('lm-eval-profile','options'),Input('lm-state','data'))
    def available_models(state):
        cuda=(state or {}).get('capabilities',{}).get('cuda',False)
        return ([dict(label=name,value=name,disabled=not cuda and index>0) for index,name in enumerate(MODELS)],
                [dict(label='Rapido CPU: 4 blocchi test, contesti 32/128',value='quick'),dict(label='Contesto lungo CPU: 1 blocco test, 128/1024 (lento)',value='long'),dict(label='GPU: 64 blocchi test, 128/512/2048',value='wide',disabled=not cuda)])

    @app.callback(Output('lm-pause','disabled'),Output('lm-resume','disabled'),
                  Output('lm-evaluate','disabled'),Output('lm-dense-control','disabled'),
                  Input('lm-state','data'),Input('lm-job','value'))
    def allowed_actions(state,selected):
        job=next((item for item in (state or {}).get('jobs',[]) if item['id']==selected),None)
        if not job:return True,True,True,True
        active=job['status'] in ('queued','preparing','running','pausing')
        busy=(state or {}).get('busy',False)
        controlled=job['config'].get('protocol')=='controlled-v1'
        return (not active,active or job['status']=='completed' or busy,
                not controlled or active or busy or not job.get('snapshots'),
                not controlled or busy or job['config'].get('architecture')=='dense-control')

    @app.callback(Output('lm-public-dataset', 'options'), Output('lm-public-preset', 'options'),
                  Output('lm-public-info', 'children'), Output('lm-public-start', 'disabled'),
                  Input('lm-state', 'data'), Input('lm-public-dataset', 'value'))
    def public_info(state, selected):
        catalog = (state or {}).get('public_catalog', {})
        datasets, presets = catalog.get('datasets', []), catalog.get('presets', [])
        item = next((item for item in datasets if item['id'] == selected), None)
        description = 'Collega il worker aggiornato per usare i corpus pubblici.'
        if item:
            description = html.Div([html.P(item['note']),
                html.A('Fonte del testo', href=item['source'], target='_blank', rel='noopener noreferrer'),
                html.Span(' · '), html.A(item['license'], href=item['license_url'], target='_blank', rel='noopener noreferrer'),
                html.P(f"Download: {item['bytes']/1048576:.2f} MiB; integrità verificata con SHA-256. La prova breve verifica il percorso, non certifica la qualità linguistica."),
                html.P('Worker occupato: metti in pausa il progetto attivo prima di avviare questa prova.') if state.get('busy') else None])
        return ([dict(label=item['title'], value=item['id']) for item in datasets] if datasets else dash.no_update,
                [dict(label=item['title'], value=item['id']) for item in presets] if presets else dash.no_update,
                description, not item or bool((state or {}).get('busy')))

    @app.callback(Output('lm-active-job', 'data', allow_duplicate=True),
                  Output('lm-public-status', 'children'), Output('lm-prompt', 'value'),
                  Input('lm-public-start', 'n_clicks'), State('lm-workspace', 'data'),
                  State('lm-public-dataset', 'value'), State('lm-public-preset', 'value'),
                  prevent_initial_call=True)
    def public_start(clicks, workspace, dataset, preset):
        if not clicks:
            return dash.no_update, dash.no_update, dash.no_update
        try:
            result = backend(workspace, '/public-experiments', 'POST', dict(dataset=dataset, preset=preset), timeout=90)
            return result['job']['id'], result['message'], result['prompt']
        except ValueError as exc:
            return dash.no_update, dbc.Alert(str(exc), color='warning'), dash.no_update

    @app.callback(Output('lm-active-job','data'), Output('lm-action-status','children'),
                  Input('lm-start','n_clicks'), Input('lm-pause','n_clicks'), Input('lm-resume','n_clicks'),
                  State('lm-workspace','data'), State('lm-job','value'), State('lm-corpus','value'),
                  State('lm-base','value'), State('lm-name','value'), State('lm-steps','value'),
                  State('lm-length','value'), State('lm-batch','value'), State('lm-accum','value'),
                  State('lm-lr','value'), State('lm-slots','value'), State('lm-window','value'),
                  State('lm-rank','value'), State('lm-save','value'), State('lm-device','value'), State('lm-distill','value'), State('lm-fraction','value'), prevent_initial_call=True)
    def control(start, pause, resume, workspace, job, corpus, base, name, steps, length, batch, accumulation, lr, slots, window, rank, save_steps, device, distill, fraction):
        try:
            action = dash.ctx.triggered_id
            if action == 'lm-start':
                config = dict(corpus_id=corpus, base_model=base, name=name, total_steps=steps, sequence_length=length,
                              batch_size=batch, accumulation=accumulation, learning_rate=lr, slots=slots, window=window,
                              lora_rank=rank, save_steps=save_steps, device=device, protocol='controlled-v1', distillation_weight=.5 if distill else 0., graft_fraction=fraction)
                job = backend(workspace, '/jobs', 'POST', config)['id']
                backend(workspace, f'/jobs/{job}/start', 'POST')
                return job, 'Training avviato nel worker; puoi chiudere la pagina.'
            if not job:
                raise ValueError('Seleziona un progetto.')
            backend(workspace, f"/jobs/{job}/{'pause' if action=='lm-pause' else 'start'}", 'POST')
            return job, 'Pausa richiesta: salvataggio al confine del prossimo passo completo.' if action=='lm-pause' else 'Ripresa richiesta: verrà verificato il checkpoint prima di continuare.'
        except ValueError as exc:
            return dash.no_update, str(exc)

    @app.callback(Output('lm-progress','children'), Output('lm-loss','figure'),
                  Output('lm-checkpoint','options'), Output('lm-checkpoint','value'),
                  Input('lm-state','data'), Input('lm-job','value'), State('lm-checkpoint','value'))
    def progress(state, selected, checkpoint):
        job = next((j for j in (state or {}).get('jobs',[]) if j['id']==selected), None)
        if not job:
            return 'Seleziona un progetto.', go.Figure(), [], None
        metrics = job['progress']
        figure = go.Figure()
        for key, label in [('training_nll','Training NLL'),('validation_nll','Validation NLL')]:
            rows = [row for row in job['history'] if key in row]
            figure.add_scatter(x=[row['step'] for row in rows], y=[row[key] for row in rows], mode='lines', name=label)
        figure.update_layout(title='Errore di previsione · più basso è meglio', xaxis_title='Passi', yaxis_title='NLL')
        protected=next((item for item in job['snapshots'] if item.get('best')),None)
        if protected and protected.get('validation_nll') is not None:
            figure.add_hline(y=protected['validation_nll'],line_dash='dot',line_color='purple',annotation_text='Migliore conservato')
        baseline = metrics.get('baseline_nll')
        if baseline is not None:
            figure.add_hline(y=baseline, line_dash='dash', line_color='green', annotation_text='Modello originale sullo stesso corpus')
        text = f"{job['status']} · passo {metrics.get('step',0)}/{job['config']['total_steps']} · token elaborati {metrics.get('tokens_trained',0):,} · ultimo checkpoint {job['checkpoint'] or 'ancora nessuno'}"
        detail = [html.P(text), html.P(job['error'], className='text-danger') if job['error'] else None]
        for key,label in [('training_nll','NLL training'),('validation_nll','NLL validazione'),('validation_perplexity','Perplessità validazione'),
                          ('baseline_nll','NLL base prima del graft'),('tokens_per_second','Token/s misurati'),('trainable_parameters','Parametri trainabili'),('total_parameters','Parametri totali del modello innestato')]:
            if metrics.get(key) is not None:
                detail.append(html.P(f"{label}: {metrics[key]:,.5g}"))
        validation = metrics.get('validation_nll')
        if validation is not None and baseline is not None:
            meaning = 'ancora peggiore del modello originale' if validation > baseline else 'uguale o migliore del modello originale su questo campione'
            detail.append(dbc.Alert(f'Validazione {validation:.3f}; modello originale {baseline:.3f}: {meaning}. Non esiste una soglia universale di qualità. Controlla anche testi nuovi e generazioni.', color='info'))
        if metrics.get('early_stop_reason'):
            detail.append(dbc.Alert(metrics['early_stop_reason']+' Puoi provare il migliore conservato; riprendi solo per un esperimento consapevole.',color='warning'))
        if metrics.get('dataset'):
            data=metrics['dataset'];unique=data['tokens']['train']
            validation_tokens=min(data['tokens']['validation']//(job['config']['sequence_length']+1),job['config'].get('validation_batches',8))*job['config']['sequence_length']
            detail.append(html.P(f'Campione di validazione per controllo: {validation_tokens} token.'))
            detail.append(html.P(f"Documenti distinti: {data['documents']}; duplicati rimossi: {data['duplicates']}; passaggi equivalenti sul training: {metrics.get('tokens_trained',0)/max(1,unique):.2f}. Il test è escluso dalla scelta dei checkpoint."))
        for key,label in [('gradient_norm','Norma gradiente'),('acan_gradient_norm','Gradiente ACAN'),('lora_gradient_norm','Gradiente LoRA'),('teacher_kl','Distanza dal teacher')]:
            if metrics.get(key) is not None:detail.append(html.P(f'{label}: {metrics[key]:.5g}'))
        if metrics.get('graft'):
            graft=metrics['graft'];detail.append(html.P(f"Struttura: {len(graft['replaced_layers'])}/{graft['total_layers']} livelli con attenzione ACAN; slot {graft['slots']}; finestra locale {graft['window']}."))
        last_step = max((s['step'] for s in job['snapshots']), default=0)
        detail.append(html.P(f'La linea blu varia con i pezzi di testo. La rossa riporta l’ultima validazione disponibile (checkpoint passo {last_step}) e la ripete fino al prossimo salvataggio. I token elaborati possono includere ripetizioni del corpus.', className='small text-muted'))
        available = [dict(label=("Migliore conservato · " if s.get('best') else "")+f"Passo {s['step']}"+(f" · validazione {s['validation_nll']:.3f}" if s.get('validation_nll') is not None else ""), value=s['name']) for s in job['snapshots']]
        selected_snapshot = checkpoint if checkpoint in {s['value'] for s in available} else available[0]['value'] if available else None
        return detail, figure, available, selected_snapshot

    @app.callback(Output('lm-active-job','data',allow_duplicate=True),Output('lm-control-status','children'),
                  Input('lm-dense-control','n_clicks'),State('lm-workspace','data'),State('lm-job','value'),prevent_initial_call=True)
    def dense_control(clicks,workspace,job):
        try:
            if not job:raise ValueError('Seleziona un progetto controllato.')
            result=backend(workspace,f'/jobs/{job}/dense-control','POST',{})
            return result['id'],'Controllo denso avviato con gli stessi dati, seed e budget; progetto ACAN conservato.'
        except ValueError as exc:
            return dash.no_update,str(exc)

    @app.callback(Output('lm-evaluation-id','data'),Output('lm-eval-status','children'),
                  Input('lm-evaluate','n_clicks'),State('lm-workspace','data'),State('lm-job','value'),
                  State('lm-checkpoint','value'),State('lm-eval-profile','value'),prevent_initial_call=True)
    def start_evaluation(clicks,workspace,job,checkpoint,profile):
        try:
            if not job or not checkpoint:raise ValueError('Seleziona progetto e checkpoint.')
            budgets={'quick':dict(max_batches=4,context_lengths=[32,128],new_tokens=8),'long':dict(max_batches=1,context_lengths=[128,1024],new_tokens=4),'wide':dict(max_batches=64,context_lengths=[128,512,2048],new_tokens=16)}
            if profile not in budgets:raise ValueError('Budget del confronto non valido.')
            result=backend(workspace,f'/jobs/{job}/evaluate','POST',dict(checkpoint=checkpoint,**budgets[profile]))
            return result['id'],'Confronto avviato nel worker: nessun peso viene aggiornato.'
        except ValueError as exc:
            return None,str(exc)

    @app.callback(Output('lm-eval-status','children',allow_duplicate=True),Input('lm-eval-stop','n_clicks'),
                  State('lm-evaluation-id','data'),State('lm-workspace','data'),prevent_initial_call=True)
    def stop_evaluation(clicks,identifier,workspace):
        if not identifier:return 'Nessun confronto selezionato.'
        try:
            backend(workspace,f'/evaluations/{identifier}/cancel','POST',{})
            return 'Arresto richiesto: attendo la fine del calcolo corrente, senza modificare i pesi.'
        except ValueError as exc:return str(exc)

    @app.callback(Output('lm-eval-view','children'),Output('lm-evaluation-report','data'),
                  Input('lm-poll','n_intervals'),Input('lm-evaluation-id','data'),Input('lm-job','value'),
                  State('lm-workspace','data'))
    def evaluation_status(interval,identifier,job,workspace):
        if not identifier or not workspace:return '',None
        try:
            value=backend(workspace,'/evaluations/'+identifier)
            if value['job']!=job:return '',None
            if value['status']=='cancelled':return 'Confronto annullato.',None
            if value['status']=='failed':return dbc.Alert(value['error'],color='warning'),None
            if value['status']!='completed':return 'Confronto in corso…',None
            report=value['result'];rows=[]
            for key,label in [('original','Modello originale'),('selected_checkpoint','Checkpoint selezionato')]:
                q=report['records'][key]['quality']
                rows.append(html.Tr([html.Td(label),html.Td(f"{q['nll']:.4f}"),html.Td(f"{q['perplexity']:.2f}" if q['perplexity'] is not None else 'non finita'),
                    html.Td(f"{q['token_accuracy']:.1%}"),html.Td(q['tokens'])]))
            table=dbc.Table([html.Thead(html.Tr([html.Th(x) for x in ['Modello','NLL ↓','Perplessità ↓','Accuracy token ↑','Token test']])),html.Tbody(rows)],responsive=True)
            details=[]
            for item in report['comparison']:
                details.append(html.Li(f"Contesto {item['context']}: speedup prefill {item['prefill_speedup']:.2f}×; decoding {item['decode_speedup']:.2f}×; riduzione cache totale {item['total_cache_reduction']:.1%}. Valori negativi indicano più memoria."))
            return [table,html.Ul(details),html.P('Campione limitato e contesti brevi: non certifica la qualità generale né i risultati numerici del paper.')],report
        except ValueError as exc:return str(exc),None

    @app.callback(Output('lm-eval-file','data'),Input('lm-eval-download','n_clicks'),
                  State('lm-evaluation-report','data'),prevent_initial_call=True)
    def download_evaluation(clicks,report):
        return dict(content=json.dumps(report,indent=2),filename='acan-lm-comparison.json') if report else dash.no_update

    @app.callback(Output('lm-debug-view','children'),Output('lm-debug-data','data'),
                  Input('lm-state','data'),Input('lm-job','value'),State('lm-workspace','data'))
    def debug_project(state,job,workspace):
        if not job or not workspace:
            return 'Seleziona un progetto.',None
        try:
            report=backend(workspace,f'/jobs/{job}/diagnostics')
            return json.dumps(report,indent=2,ensure_ascii=False),report
        except ValueError as exc:
            return str(exc),None

    @app.callback(Output('lm-debug-file','data'),Input('lm-debug-download','n_clicks'),
                  State('lm-debug-data','data'),prevent_initial_call=True)
    def download_debug(clicks,report):
        return dict(content=json.dumps(report,indent=2,ensure_ascii=False),filename='acan-lm-diagnostics.json') if report else dash.no_update

    @app.callback(Output('lm-generation-id','data'), Output('lm-generation-status','children'),
                  Input('lm-generate','n_clicks'), State('lm-workspace','data'), State('lm-job','value'),
                  State('lm-checkpoint','value'), State('lm-prompt','value'), State('lm-new-tokens','value'), prevent_initial_call=True)
    def start_generation(click, workspace, job, checkpoint, prompt, count):
        try:
            if not job or not checkpoint:
                raise ValueError('Seleziona progetto e checkpoint, poi metti in pausa il training.')
            result = backend(workspace, f'/jobs/{job}/generate', 'POST', dict(checkpoint=checkpoint,prompt=prompt,max_new_tokens=count))
            return result['id'], 'Generazione avviata dal checkpoint selezionato.'
        except ValueError as exc:
            return None, str(exc)

    @app.callback(Output('lm-output','children'), Output('lm-generation-metrics','children'),
                  Input('lm-poll','n_intervals'), Input('lm-generation-id','data'), Input('lm-job','value'), State('lm-workspace','data'))
    def generation_status(interval, generation, selected, workspace):
        if not generation or not workspace:
            return '', []
        try:
            item = backend(workspace, '/generations/'+generation)
            if item['job'] != selected:
                return '', []
            if item['status'] == 'failed':
                return item['error'], []
            if item['status'] != 'completed':
                return 'Generazione in corso…', []
            result = item['result']
            description = f"Passo {result['step']} · {result['generated_tokens']} token · {result['tokens_per_second']:.2f} token/s · cache totale {result['cache_total_bytes']/1048576:.3f} MiB"
            reduction = result['cache_reduction_fraction']
            panels = [html.P(description), html.P(f"Cache layer densi: {result['cache_dense_layers_bytes']:,} byte; cache ACAN: {result['cache_acan_layers_bytes']:,} byte. Riduzione complessiva rispetto al riferimento GQA: {reduction:.1%}. Misura del cache residente, non del picco totale di memoria."),
                      html.P('Gli indici di canale e gli slot descrivono variabili latenti; non sono spiegazioni linguistiche certificate.'),
                      dbc.Accordion([dbc.AccordionItem(html.Pre(json.dumps(result['explanations'], indent=2)), title='Ispeziona slot e canali latenti')], start_collapsed=True)]
            return result['text'], panels
        except ValueError as exc:
            return str(exc), []

    @app.callback(Output('lm-key-download','data'), Input('lm-export-key','n_clicks'), State('lm-workspace','data'), prevent_initial_call=True)
    def export_key(click, workspace):
        if not workspace:
            return dash.no_update
        return dict(content=json.dumps({'acan_lm_workspace':workspace}), filename='acan-lm-recovery-key.json')
