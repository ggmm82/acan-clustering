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
from flask import request as flask_request, jsonify


MODELS = ['HuggingFaceTB/SmolLM2-135M', 'HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-1.7B']


def backend(workspace, path, method='GET', data=None, raw=False, headers=None):
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
        with urlopen(Request(endpoint+path, data=body, headers=fields, method=method), timeout=30) as response:
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
    html.P('Continued pre-training con LoRA e WSD su pesi SmolLM2. I layer superiori (50%) usano il porting ACAN causale corretto del nuovo paper. Qualità linguistica e riduzione della cache sono da misurare: non sono garantite dal grafting.'),
    dcc.Store(id='lm-workspace', storage_type='local'), dcc.Store(id='lm-state'),
    dcc.Store(id='lm-active-job', storage_type='session'),
    dcc.Store(id='lm-upload-state'), dcc.Store(id='lm-generation-id'), dcc.Store(id='lm-browser-ready'),
    html.Div(id='lm-connection', role='status'),
    dbc.Button('Aggiorna progetti', id='lm-refresh', outline=True, className='my-2'),
    dbc.Button('Salva chiave di recupero', id='lm-export-key', color='link'),
    html.Div(id='lm-recovery-input'), dcc.Download(id='lm-key-download'),
    html.P('Il workspace resta nel browser dopo refresh e riavvii del servizio. Salva la chiave per recuperare i progetti da un altro browser; chi possiede la chiave può gestirli. Nessun account richiesto.', className='small text-muted'),
    dbc.Row([
        dbc.Col([html.H5('1 · Corpus'), html.Div(id='lm-file-upload'),
                 html.Div(id='lm-upload-status', role='status'), dcc.Dropdown(id='lm-corpus', placeholder='Seleziona corpus caricato')], md=6),
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
        dbc.Label('Dispositivo'), dcc.Dropdown(id='lm-device', options=['auto','cuda','cpu'], value='auto', clearable=False),
        html.P('Salvataggio anche ogni 5 minuti e su pausa. Un arresto improvviso può richiedere di rifare i passi dopo l’ultimo checkpoint; i checkpoint completi precedenti restano intatti. GPU non disponibile: solo il profilo CPU 135M per verifiche, molto più lento.', className='small text-muted'),
    ], title='Impostazioni avanzate')], start_collapsed=True, className='my-3'),
    dbc.Button('Crea e avvia training', id='lm-start', className='m-1'),
    html.Hr(), html.H5('3 · Progetti persistenti'),
    dcc.Dropdown(id='lm-job', placeholder='Scegli un progetto per controllarlo o riprenderlo'),
    dbc.Button('Pausa e salva', id='lm-pause', color='warning', className='m-1'),
    dbc.Button('Riprendi dal checkpoint', id='lm-resume', className='m-1'),
    html.Div(id='lm-action-status', role='status'), html.Div(id='lm-progress'), dcc.Graph(id='lm-loss'),
    dcc.Interval(id='lm-poll', interval=5000, disabled=True),
    html.Hr(), html.H5('4 · Prova un checkpoint durante il percorso'),
    html.P('Sulla stessa GPU la generazione richiede il training in pausa. Il checkpoint è immutabile: la prova non modifica quanto appreso. Dopo la prova puoi riprendere il training.'),
    dcc.Dropdown(id='lm-checkpoint', placeholder='Ultimi checkpoint verificabili (conservati gli ultimi tre)'),
    dbc.Textarea(id='lm-prompt', value='Once upon a time', rows=3),
    dbc.Label('Nuovi token'), dbc.Input(id='lm-new-tokens', type='number', value=64, min=1, max=256),
    dbc.Button('Genera dal checkpoint', id='lm-generate', className='my-2'),
    html.Div(id='lm-generation-status', role='status'), html.Pre(id='lm-output', style={'whiteSpace':'pre-wrap'}),
    html.Div(id='lm-generation-metrics'),
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
                  Input('lm-poll','n_intervals'), Input('lm-refresh','n_clicks'), Input('lm-workspace','data'), Input('lm-upload-state','data'),
                  State('tabs','value'))
    def poll(interval, refresh, workspace, upload, tab):
        if tab != 'Generative' or not workspace:
            return dash.no_update, dash.no_update
        try:
            state = backend(workspace, '/state')
            capabilities = backend(workspace, '/capabilities')
            hardware = 'CUDA disponibile' if capabilities['cuda'] else 'CPU · profilo di verifica 135M'
            return state, f'Worker collegato · {hardware} · persistenza attiva · una sola operazione alla volta.'
        except ValueError as exc:
            return None, dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-corpus','options'), Output('lm-job','options'),
                  Output('lm-corpus','value'), Output('lm-job','value'),
                  Input('lm-state','data'), Input('lm-active-job','data'),
                  State('lm-upload-state','data'), State('lm-corpus','value'), State('lm-job','value'))
    def options(state, active, upload, corpus, selected):
        if not state:
            return [], [], None, None
        corpora = [item for item in state['corpora'] if item['status']=='ready']
        corpus_ids = {item['id'] for item in corpora}
        uploaded = (upload or {}).get('id')
        chosen_corpus = corpus if corpus in corpus_ids else uploaded if uploaded in corpus_ids else None
        job_ids = {item['id'] for item in state['jobs']}
        chosen_job = selected if selected in job_ids else active if active in job_ids else None
        return ([dict(label=f"{item['name']} ({item['received']/1048576:.1f} MiB)", value=item['id']) for item in corpora],
                [dict(label=f"{item['config']['name']} · {item['status']} · passo {item['progress'].get('step',0)}",value=item['id']) for item in state['jobs']],
                chosen_corpus, chosen_job)

    @app.callback(Output('lm-active-job','data'), Output('lm-action-status','children'),
                  Input('lm-start','n_clicks'), Input('lm-pause','n_clicks'), Input('lm-resume','n_clicks'),
                  State('lm-workspace','data'), State('lm-job','value'), State('lm-corpus','value'),
                  State('lm-base','value'), State('lm-name','value'), State('lm-steps','value'),
                  State('lm-length','value'), State('lm-batch','value'), State('lm-accum','value'),
                  State('lm-lr','value'), State('lm-slots','value'), State('lm-window','value'),
                  State('lm-rank','value'), State('lm-save','value'), State('lm-device','value'), prevent_initial_call=True)
    def control(start, pause, resume, workspace, job, corpus, base, name, steps, length, batch, accumulation, lr, slots, window, rank, save_steps, device):
        try:
            action = dash.ctx.triggered_id
            if action == 'lm-start':
                config = dict(corpus_id=corpus, base_model=base, name=name, total_steps=steps, sequence_length=length,
                              batch_size=batch, accumulation=accumulation, learning_rate=lr, slots=slots, window=window,
                              lora_rank=rank, save_steps=save_steps, device=device)
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
        figure.update_layout(title='Loss next-token misurata (non CP)', xaxis_title='Passi', yaxis_title='NLL')
        text = f"{job['status']} · passo {metrics.get('step',0)}/{job['config']['total_steps']} · token appresi {metrics.get('tokens_trained',0):,} · ultimo checkpoint {job['checkpoint'] or 'ancora nessuno'}"
        detail = [html.P(text), html.P(job['error'], className='text-danger') if job['error'] else None]
        for key,label in [('training_nll','NLL training'),('validation_nll','NLL validazione'),('validation_perplexity','Perplessità validazione'),
                          ('baseline_nll','NLL base prima del graft'),('tokens_per_second','Token/s misurati'),('trainable_parameters','Parametri trainabili')]:
            if metrics.get(key) is not None:
                detail.append(html.P(f"{label}: {metrics[key]:,.5g}"))
        available = [dict(label=f"Passo {s['step']}",value=s['name']) for s in job['snapshots']]
        selected_snapshot = checkpoint if checkpoint in {s['value'] for s in available} else available[0]['value'] if available else None
        return detail, figure, available, selected_snapshot

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
