"""The long training of the concept language model: data, learning queue, answers while it learns.

Every layer reads through concepts (graft fraction 1.0). The page drives the generative worker
through the same ``backend`` helper as the rest of the generative tab, and reads the projects from
``lm-state``, which that tab refreshes every five seconds.
"""
import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

MODELS = ['HuggingFaceTB/SmolLM2-135M', 'HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-1.7B']


def backend(*args, **kwargs):
    """The generative tab's connection to the worker (looked up at call time, so tests can replace it)."""
    from . import generative
    return generative.backend(*args, **kwargs)

STATUS = {'created': 'in attesa', 'queued': 'in partenza', 'preparing': 'prepara i token', 'running': 'sta imparando',
          'pausing': 'si sta fermando', 'paused': 'in pausa', 'completed': 'appreso', 'failed': 'errore',
          'interrupted': 'interrotto (riprende da solo se previsto)'}
QUEUE_NAME = 'LLM a concetti'


def profile(cuda, model=MODELS[0]):
    """The settings of a long training: every layer reads through concepts."""
    settings = dict(protocol='controlled-v1', routing_profile='concept-graph-v1', architecture='hybrid', graft_fraction=1.,
                    base_model=model, distillation_weight=0., early_stop_patience=0, validation_batches=16,
                    learning_rate=1e-4, save_seconds=300, device='auto', chat_probes=True, auto_resume=True)
    if cuda:
        settings.update(sequence_length=512, batch_size=8, accumulation=4, slots=64, window=64, lora_rank=64)
    else:   # the CPU profile with every layer on concepts: blocks of 64 tokens, 16 concepts, 16 exact tokens
        settings.update(sequence_length=64, batch_size=1, accumulation=8, slots=16, window=16, lora_rank=8)
    return settings


def queues(jobs):
    """Learning queues of the workspace: {id: items in order}."""
    found = {}
    for job in jobs:
        series = job['config'].get('series') or {}
        if series.get('variant') == 'queue':
            found.setdefault(series['id'], []).append(job)
    return {key: sorted(items, key=lambda job: job['config']['series']['order']) for key, items in found.items()}


def current_item(items):
    """The item that is learning now, or the next one to learn, or the last learned."""
    for wanted in (('queued', 'preparing', 'running', 'pausing'), ('paused', 'interrupted', 'failed'), ('created',)):
        found = [job for job in items if job['status'] in wanted]
        if found:
            return found[0]
    return items[-1] if items else None


def queue_table(items):
    rows = []
    for job in items:
        progress = job['progress'] or {}
        total = job['config']['total_steps'] if job['config'].get('passes') is None or job['checkpoint'] else None
        step = progress.get('step', 0)
        if job['status'] == 'completed':
            share = 100
        elif job['status'] == 'created' or not job['checkpoint']:
            share = 0
        else:
            share = 100*step/max(job['config']['total_steps'], 1)
        label = '—' if job['status'] == 'created' else f"passo {step:,} di {job['config']['total_steps']:,}" if total else f'passo {step:,}'
        best = progress.get('best_validation_nll')
        error = progress.get('validation_nll')
        rows.append(html.Tr([
            html.Td(job['config']['series']['order']+1), html.Td(job['config']['series'].get('label', '')),
            html.Td(STATUS.get(job['status'], job['status']), className='text-danger' if job['status'] == 'failed' else ''),
            html.Td([dbc.Progress(value=share, label=f'{share:.0f}%', style={'height': '18px'}, className='mb-1'), html.Small(label)]),
            html.Td('—' if error is None else f"{error:.3f}" + ('' if best is None else f" (migliore {best:.3f})")),
        ]))
    return dbc.Table([html.Thead(html.Tr([html.Th('#'), html.Th('Testo'), html.Th('Stato'), html.Th('Avanzamento'),
                                          html.Th('Errore su testo mai visto ↓')])), html.Tbody(rows)],
                     bordered=True, size='sm', responsive=True)


def answers_view(entries):
    if not entries:
        return html.P('Ancora nessuna risposta: compaiono al primo salvataggio (ogni 100 passi o 5 minuti).', className='small text-muted')
    first, last = entries[0], entries[-1]
    header = (f"Ultimo salvataggio: passo {last['step']:,}"
              + ('' if last.get('validation_nll') is None else f" · errore sul riferimento {last['validation_nll']:.3f}")
              + (f" · confronto con il passo {first['step']:,}" if len(entries) > 1 else ''))
    rows = []
    for index, item in enumerate(last['answers']):
        before = first['answers'][index]['answer'] if len(entries) > 1 and index < len(first['answers']) else None
        rows.append(html.Tr([html.Td(item['question']),
                             html.Td(html.Pre(before or '—', className='small mb-0', style={'whiteSpace': 'pre-wrap'})),
                             html.Td(html.Pre(item['answer'] or '(nessuna risposta)', className='small mb-0', style={'whiteSpace': 'pre-wrap'}))]))
    history = [html.Li(f"passo {entry['step']:,}" + ('' if entry.get('validation_nll') is None else f": {entry['validation_nll']:.3f}"))
               for entry in entries[-10:]]
    return html.Div([html.P(header, className='small'),
                     dbc.Table([html.Thead(html.Tr([html.Th('Domanda'), html.Th('Prima'), html.Th('Adesso')])), html.Tbody(rows)],
                               bordered=True, size='sm', responsive=True),
                     html.Details([html.Summary('Errore sul riferimento agli ultimi salvataggi', className='small'),
                                   html.Ul(history, className='small')])])


def _bytes(value):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(value) < 1024 or unit == 'TB':
            return f'{value:,.1f} {unit}' if unit != 'B' else f'{value:,.0f} B'
        value /= 1024


def footprint_view(item):
    """The main sizes of the model while it learns, grouped by what they help to tune."""
    progress, config = item['progress'] or {}, item['config']
    step, total = progress.get('step', 0), config['total_steps']
    speed = progress.get('tokens_per_second')
    per_step = config['batch_size']*config['accumulation']*config['sequence_length']
    remaining = (total-step)*per_step/speed if speed else None
    trainable, overall = progress.get('trainable_parameters'), progress.get('total_parameters')
    graft = progress.get('graft') or {}
    layers, total_layers = graft.get('replaced_layers') or [], graft.get('total_layers')
    kept = config['window']+config['slots']
    rows = [
        ('Modello', [
            ('Parametri totali', f'{overall:,}' if overall else '—'),
            ('Parametri che imparano (LoRA)', f"{trainable:,} ({100*trainable/overall:.2f}%)" if trainable and overall else '—'),
            ('Layer a concetti', f"{len(layers)} su {total_layers}" if layers and total_layers else ('tutti' if config.get('graft_fraction') == 1. else 'metà')),
            ('Memoria di lettura per layer', f"{config['window']} token esatti + {config['slots']} concetti = {kept} vettori, a qualunque lunghezza"),
            ('Testo letto per pezzo', f"{config['sequence_length']} token")]),
        ('Avanzamento e velocità', [
            ('Passo', f'{step:,} di {total:,}'),
            ('Token elaborati', f"{progress.get('tokens_trained', 0):,}"),
            ('Velocità', f'{speed:,.1f} token al secondo' if speed else '—'),
            ('Tempo che manca, stima', '—' if remaining is None else f'{remaining/86400:,.1f} giorni' if remaining > 2*86400 else f'{remaining/3600:,.1f} ore'),
            ('Token per passo', f'{per_step:,} ({config["batch_size"]} × {config["accumulation"]} pezzi × {config["sequence_length"]} token)')]),
        ('Qualità', [
            ('Errore sul testo di addestramento', '—' if progress.get('training_nll') is None else f"{progress['training_nll']:.3f}"),
            ('Errore su testo mai visto', '—' if progress.get('validation_nll') is None else f"{progress['validation_nll']:.3f}"),
            ('Migliore finora', '—' if progress.get('best_validation_nll') is None else f"{progress['best_validation_nll']:.3f}"),
            ('Modello di partenza sullo stesso testo', '—' if progress.get('baseline_nll') is None else f"{progress['baseline_nll']:.3f}")]),
        ('Ottimizzazione', [
            ('Passo di apprendimento attuale', '—' if progress.get('learning_rate') is None else f"{progress['learning_rate']:.2e}"),
            ('Ampiezza della correzione (norma del gradiente)', '—' if progress.get('gradient_norm') is None else
             f"{progress['gradient_norm']:.3f}" + (' · tagliata a 1' if progress['gradient_norm'] > 1 else '')),
            ('Memoria del processo (picco)', '—' if progress.get('process_peak_rss_kib') is None else _bytes(progress['process_peak_rss_kib']*1024)),
            ('Memoria della GPU (picco)', '—' if not progress.get('gpu_peak_bytes') else _bytes(progress['gpu_peak_bytes']))]),
        ('Spazio su disco', [
            ('Un checkpoint', _bytes(item['checkpoint_bytes'])),
            ('Checkpoint tenuti', ', '.join(f"{value['count']} {name}" for name, value in
                                            zip(('recenti', 'permanenti', 'migliore'), (item['checkpoints'][key] for key in ('step', 'milestone', 'best'))))
             + f" · {_bytes(sum(value['bytes'] for value in item['checkpoints'].values()))}"),
            ('Testo e token', f"{_bytes(item['corpus_bytes'])} di testo, {_bytes(item['token_bytes'])} di token"),
            ('Spazio libero sul server', f"{_bytes(item['disk_free'])} di {_bytes(item['disk_total'])}")]),
    ]
    advice = []
    if speed and remaining and remaining > 30*86400:
        advice.append('A questa velocità mancano più di 30 giorni: con una GPU lo stesso lavoro richiede giorni.')
    if progress.get('gradient_norm') and progress['gradient_norm'] > 5:
        advice.append('Correzioni molto ampie: se l\'errore oscilla, abbassare il passo di apprendimento.')
    if item['disk_free'] < 5*1024**3:
        advice.append('Meno di 5 GB liberi: ridurre i checkpoint permanenti o liberare spazio.')
    tables = [html.Div([html.H6(title, className='mt-2'),
                        dbc.Table(html.Tbody([html.Tr([html.Td(name), html.Td(value)]) for name, value in values]), size='sm', bordered=True)])
              for title, values in rows]
    return html.Div([dbc.Row([dbc.Col(table, md=6) for table in tables])] +
                    [dbc.Alert(text, color='info', className='small') for text in advice])


def build_view(build):
    if not build:
        return html.P('Nessuna preparazione ancora.', className='small text-muted')
    result = build.get('result') or {}
    progress = result.get('progress') or {}
    share = 100*progress.get('written_mb', 0)/max(progress.get('target_mb', 1), 1)
    status = {'queued': 'in partenza', 'running': 'in corso', 'pausing': 'si sta fermando', 'paused': 'in pausa',
              'interrupted': 'interrotta da un riavvio (riparte da sola)', 'completed': 'pronta', 'failed': 'errore'}.get(build['status'], build['status'])
    lines = [html.P(f"Miscela «{build['request']['mixture']}» · {status} · {build.get('phase') or ''}", className='small mb-1'),
             dbc.Progress(value=min(share, 100), label=f"{progress.get('written_mb', 0):,.0f} di circa {progress.get('target_mb', 0):,} MB",
                          style={'height': '20px'}, className='mb-1')]
    if progress:
        records = progress.get('records', {})
        lines.append(html.Small(f"File letti {progress.get('files_done', 0)} di {progress.get('files', 0)} · documenti: "
                                f"{records.get('train', 0):,} per imparare, {records.get('validation', 0):,} di validazione, "
                                f"{records.get('test', 0):,} di test", className='text-muted'))
    if build.get('error'):
        lines.append(dbc.Alert(build['error'], color='danger', className='mt-2'))
    if result.get('corpus_name'):
        lines.append(html.P(f"Testo pronto: {result['corpus_name']}. Ora puoi metterlo in coda.", className='small text-success mt-1'))
    return html.Div(lines)


layout = dbc.Card(dbc.CardBody([
    html.H4('Apprendimento lungo · LLM a concetti'),
    html.P("Tutti i layer del modello leggono il testo attraverso i concetti ACAN: gli ultimi 16 token esatti più 16 "
           "concetti per tutto il resto, a qualunque lunghezza. Si parte dai pesi di SmolLM2 (che conosce già la lingua) e "
           "si addestrano le matrici LoRA che gli insegnano a leggere così e a conversare. L'addestramento salva il "
           "lavoro ogni 100 passi e ogni 5 minuti, riprende da solo dopo un riavvio del server, e a ogni salvataggio "
           "risponde alle domande di prova: le vedi qui sotto mentre impara.", className='small'),
    html.H5('1 · I dati: italiano e inglese per conversare', className='mt-3'),
    html.P("Il worker scarica le fonti una alla volta da versioni fissate, verifica ogni file, tiene solo il testo e "
           "cancella il resto. La preparazione si mette in pausa e riprende senza perdere nulla.", className='small'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-mixture', placeholder='Scegli la quantità di dati', clearable=False), md=6),
             dbc.Col([dbc.Button('Prepara i dati', id='lm-long-build', color='primary', className='me-2'),
                      dbc.Button('Pausa', id='lm-long-build-pause', outline=True, color='warning', className='me-2'),
                      dbc.Button('Riprendi', id='lm-long-build-resume', outline=True, color='primary')], md=6)], className='mb-2'),
    html.Div(id='lm-long-mixture-info', className='small text-muted'),
    html.Div(id='lm-long-build-status', role='status', className='small'), html.Div(id='lm-long-build-view'),
    dbc.Accordion([dbc.AccordionItem(html.Div(id='lm-long-sources'), title='Fonti, licenze e riferimento fisso')],
                  start_collapsed=True, className='my-2'),
    html.H5('2 · La coda di apprendimento', className='mt-3'),
    html.P("I testi vengono imparati uno dopo l'altro: ognuno riparte dai pesi lasciati dal precedente. Puoi mettere in "
           "coda la miscela preparata e i tuoi file (caricati negli strumenti avanzati: testo, Markdown, Python, ZIP, EPUB, "
           "DOCX, PDF), e aggiungerne altri mentre impara.", className='small'),
    dbc.Row([dbc.Col([dbc.Label('Testi da imparare, in ordine'), dcc.Dropdown(id='lm-long-corpora', multi=True, placeholder='Scegli uno o più testi')], md=6),
             dbc.Col([dbc.Label('Modello di partenza'), dcc.Dropdown(id='lm-long-model', value=MODELS[0], clearable=False,
                                                                       options=[dict(label=MODELS[0].split('/')[-1], value=MODELS[0])])], md=6)], className='mb-2'),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupText('Giri su ogni testo'), dbc.Input(id='lm-long-passes', type='number', value=1, min=.01, max=100, step=.01)]), md=4),
             dbc.Col(dbc.InputGroup([dbc.InputGroupText('Salva ogni'), dbc.Input(id='lm-long-save', type='number', value=100, min=10, max=10000), dbc.InputGroupText('passi')]), md=4),
             dbc.Col(dbc.InputGroup([dbc.InputGroupText('Tieni per sempre ogni'), dbc.Input(id='lm-long-milestone', type='number', value=1000, min=0, max=10000000), dbc.InputGroupText('passi')]), md=4)],
            className='mb-2'),
    dbc.Label('Domande di prova (una per riga, al massimo 6; vuoto = domande predefinite in italiano e inglese)'),
    dbc.Textarea(id='lm-long-questions', rows=3, placeholder='Ciao! Chi sei?\nQual è la capitale della Francia?'),
    dbc.Button('Crea la coda e avvia', id='lm-long-start', color='primary', className='mt-2'),
    html.Div(id='lm-long-start-status', role='status', className='small mt-1'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-queue', placeholder='Coda da seguire', clearable=False), md=6),
             dbc.Col([dbc.Button('Pausa e salva', id='lm-long-pause', color='warning', className='me-2'),
                      dbc.Button('Riprendi', id='lm-long-resume', color='primary', className='me-2'),
                      dbc.Button('Mostra nel grafico', id='lm-long-chart', outline=True, color='primary')], md=6)], className='mt-3 mb-2'),
    html.Div(id='lm-long-queue-view'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-add-corpus', placeholder='Aggiungi un testo in fondo alla coda'), md=8),
             dbc.Col(dbc.Button('Aggiungi', id='lm-long-add', outline=True, color='primary'), md=4)], className='mb-1'),
    html.Div(id='lm-long-queue-status', role='status', className='small'),
    html.H5('Grandezze del modello mentre impara', className='mt-3'),
    html.Div(id='lm-long-footprint'),
    html.H5('3 · Cosa risponde adesso', className='mt-3'),
    html.P("Le stesse domande a ogni salvataggio, con risposta deterministica: le differenze vengono solo da ciò che il modello "
           "ha imparato. Per fargli domande libere: Pausa e salva, poi «Fai scrivere un modello» negli strumenti avanzati "
           "con la casella «Conversazione», poi Riprendi. Scaricare e ispezionare il modello si fa da «Segui i progetti».",
           className='small'),
    html.Div(id='lm-long-answers'),
    dcc.Store(id='lm-long-catalog'), dcc.Store(id='lm-long-builds'), dcc.Interval(id='lm-long-poll', interval=15000),
]), className='my-3', color='primary', outline=True)


def register(app):
    @app.callback(Output('lm-long-catalog', 'data'), Output('lm-long-builds', 'data'),
                  Input('lm-long-poll', 'n_intervals'), Input('lm-workspace', 'data'), State('tabs', 'value'),
                  State('lm-long-catalog', 'data'))
    def refresh(interval, workspace, tab, catalog):
        if tab != 'Generative' or not workspace:
            return dash.no_update, dash.no_update
        try:
            return catalog or backend(workspace, '/dialogue-data'), backend(workspace, '/builds')
        except ValueError:
            return dash.no_update, dash.no_update

    @app.callback(Output('lm-long-mixture', 'options'), Output('lm-long-mixture', 'value'), Output('lm-long-sources', 'children'),
                  Input('lm-long-catalog', 'data'), State('lm-long-mixture', 'value'))
    def mixtures(catalog, current):
        if not catalog:
            return [], None, 'Il servizio di addestramento non è collegato o va aggiornato.'
        options = [dict(label=item['title'], value=item['id']) for item in catalog['mixtures']]
        sources = html.Div([html.Ul([html.Li([html.A(item['title'], href=item['page'], target='_blank', rel='noopener noreferrer'),
                                              f" · {item['license']} · lingue: {', '.join(item['languages'])}"]) for item in catalog['sources']],
                                    className='small'),
                            html.P(catalog['reference'], className='small')])
        return options, current or options[0]['value'], sources

    @app.callback(Output('lm-long-mixture-info', 'children'), Input('lm-long-mixture', 'value'), State('lm-long-catalog', 'data'))
    def mixture_info(mixture, catalog):
        item = next((item for item in (catalog or {}).get('mixtures', []) if item['id'] == mixture), None)
        if not item:
            return None
        return (f"Testo finale circa {item['corpus_mb']:,} MB (italiano {item['languages']['it']:,} MB, inglese {item['languages']['en']:,} MB); "
                f"si scaricano circa {item['download_mb']/1024:,.1f} GB, un file alla volta. Le conversazioni in italiano sono la parte "
                "più scarsa: l'italiano viene soprattutto da Wikipedia e FineWeb-2.")

    @app.callback(Output('lm-long-build-view', 'children'), Input('lm-long-builds', 'data'))
    def builds(items):
        return build_view((items or [None])[0])

    @app.callback(Output('lm-long-build-status', 'children'),
                  Input('lm-long-build', 'n_clicks'), Input('lm-long-build-pause', 'n_clicks'), Input('lm-long-build-resume', 'n_clicks'),
                  State('lm-workspace', 'data'), State('lm-long-mixture', 'value'), State('lm-long-builds', 'data'), prevent_initial_call=True)
    def build_action(start, pause, resume, workspace, mixture, items):
        latest = (items or [None])[0]
        try:
            if dash.ctx.triggered_id == 'lm-long-build':
                backend(workspace, '/builds', 'POST', dict(mixture=mixture))
                return 'Preparazione avviata: il primo file è in download.'
            if not latest:
                raise ValueError('Nessuna preparazione da mettere in pausa o riprendere.')
            action = 'pause' if dash.ctx.triggered_id == 'lm-long-build-pause' else 'resume'
            backend(workspace, f"/builds/{latest['id']}/{action}", 'POST', {})
            return 'Pausa richiesta: il punto raggiunto viene salvato.' if action == 'pause' else 'Preparazione ripresa.'
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-long-corpora', 'options'), Output('lm-long-add-corpus', 'options'), Output('lm-long-model', 'options'),
                  Output('lm-long-queue', 'options'), Output('lm-long-queue', 'value'),
                  Input('lm-state', 'data'), State('lm-long-queue', 'value'))
    def queue_options(state, current):
        if not state:
            return [], [], [], [], None
        corpora = [dict(label=f"{item['name']} ({item['received']/1048576:,.1f} MB)", value=item['id'])
                   for item in state.get('corpora', []) if item['status'] == 'ready']
        cuda = (state.get('capabilities') or {}).get('cuda', False)
        models = [dict(label=name.split('/')[-1] + ('' if index == 0 or cuda else ' · richiede una GPU'), value=name, disabled=index > 0 and not cuda)
                  for index, name in enumerate(MODELS)]
        found = queues(state.get('jobs', []))
        options = [dict(label=f"{items[0]['config']['name'].split(' · ')[0]} · {len(items)} testi · #{key[:4]}", value=key) for key, items in found.items()]
        value = current if current in found else (options[0]['value'] if options else None)
        return corpora, corpora, models, options, value

    @app.callback(Output('lm-long-queue-view', 'children'), Input('lm-state', 'data'), Input('lm-long-queue', 'value'))
    def queue_view(state, queue):
        items = queues((state or {}).get('jobs', [])).get(queue)
        if not items:
            return html.P('Nessuna coda ancora.', className='small text-muted')
        job = current_item(items)
        note = None
        if job and job.get('error'):
            note = dbc.Alert(f"Testo {job['config']['series']['order']+1}: {job['error']}", color='danger')
        return html.Div([queue_table(items), note])

    @app.callback(Output('lm-long-start-status', 'children'), Output('lm-long-queue', 'value', allow_duplicate=True),
                  Input('lm-long-start', 'n_clicks'), State('lm-workspace', 'data'), State('lm-long-corpora', 'value'),
                  State('lm-long-model', 'value'), State('lm-long-passes', 'value'), State('lm-long-save', 'value'),
                  State('lm-long-milestone', 'value'), State('lm-long-questions', 'value'), State('lm-state', 'data'),
                  prevent_initial_call=True)
    def start(clicks, workspace, corpora, model, passes, save, milestone, questions, state):
        try:
            if not corpora:
                raise ValueError('Scegli almeno un testo da imparare.')
            cuda = ((state or {}).get('capabilities') or {}).get('cuda', False)
            base = dict(profile(cuda, model or MODELS[0]), name=QUEUE_NAME, passes=float(passes or 1), save_steps=int(save or 100),
                        milestone_steps=int(milestone or 0),
                        probe_prompts=[line.strip() for line in (questions or '').splitlines() if line.strip()][:6])
            result = backend(workspace, '/queues', 'POST', dict(base=base, corpora=list(corpora)), timeout=120)
            return result.get('message', 'Coda creata.'), result['queue']
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update

    @app.callback(Output('lm-long-queue-status', 'children'), Output('lm-active-job', 'data', allow_duplicate=True),
                  Input('lm-long-add', 'n_clicks'), Input('lm-long-pause', 'n_clicks'), Input('lm-long-resume', 'n_clicks'),
                  State('lm-workspace', 'data'), State('lm-long-queue', 'value'), State('lm-long-add-corpus', 'value'),
                  State('lm-state', 'data'), prevent_initial_call=True)
    def queue_action(add, pause, resume, workspace, queue, corpus, state):
        items = queues((state or {}).get('jobs', [])).get(queue)
        try:
            if not items:
                raise ValueError('Crea o scegli prima una coda.')
            if dash.ctx.triggered_id == 'lm-long-add':
                if not corpus:
                    raise ValueError('Scegli il testo da aggiungere.')
                job = backend(workspace, f'/queues/{queue}/add', 'POST', dict(corpus_id=corpus))
                return f"Aggiunto come testo {job['config']['series']['order']+1}.", dash.no_update
            job = current_item(items)
            if dash.ctx.triggered_id == 'lm-long-pause':
                backend(workspace, f"/jobs/{job['id']}/pause", 'POST')
                return 'Pausa richiesta: salva il punto raggiunto e si ferma. Puoi aggiornare il server tranquillamente.', dash.no_update
            backend(workspace, f"/jobs/{job['id']}/start", 'POST')
            return 'Ripreso dall\'ultimo salvataggio.', job['id']
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update

    @app.callback(Output('lm-job', 'value', allow_duplicate=True), Input('lm-long-chart', 'n_clicks'),
                  State('lm-long-queue', 'value'), State('lm-state', 'data'), prevent_initial_call=True)
    def chart(clicks, queue, state):
        job = current_item(queues((state or {}).get('jobs', [])).get(queue) or [])
        return job['id'] if job else dash.no_update

    @app.callback(Output('lm-long-footprint', 'children'), Input('lm-state', 'data'), Input('lm-long-queue', 'value'),
                  State('lm-workspace', 'data'))
    def footprint(state, queue, workspace):
        items = [job for job in queues((state or {}).get('jobs', [])).get(queue) or [] if job['status'] != 'created']
        if not items:
            return html.P('Compaiono quando il primo testo comincia a imparare.', className='small text-muted')
        try:
            return footprint_view(backend(workspace, f"/jobs/{current_item(items)['id']}/footprint"))
        except ValueError as exc:
            return html.P(str(exc), className='small text-muted')

    @app.callback(Output('lm-long-answers', 'children'), Input('lm-state', 'data'), Input('lm-long-queue', 'value'),
                  State('lm-workspace', 'data'))
    def answers(state, queue, workspace):
        items = queues((state or {}).get('jobs', [])).get(queue) or []
        learned = [job for job in items if job['status'] != 'created']
        if not learned:
            return answers_view([])
        job = current_item(learned)
        try:
            return answers_view(backend(workspace, f"/jobs/{job['id']}/answers"))
        except ValueError as exc:
            return html.P(str(exc), className='small text-muted')
