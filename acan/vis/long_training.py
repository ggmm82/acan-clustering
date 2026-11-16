"""The long training of the concept language model: data, learning queue, answers while it learns.

Every layer reads through concepts (graft fraction 1.0). The page drives the generative worker
through the same ``backend`` helper as the rest of the generative tab, and reads the projects from
``lm-state``, which that tab refreshes every five seconds.
"""
import dash
from dash import Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from acan.lm.native import SIZES, parameters


def backend(*args, **kwargs):
    """The generative tab's connection to the worker (looked up at call time, so tests can replace it)."""
    from . import generative
    return generative.backend(*args, **kwargs)

STATUS = {'created': 'in attesa', 'queued': 'in partenza', 'preparing': 'prepara i token', 'running': 'sta imparando',
          'pausing': 'si sta fermando', 'paused': 'in pausa', 'completed': 'appreso', 'failed': 'errore',
          'interrupted': 'interrotto (riprende da solo se previsto)'}
QUEUE_NAME = 'Modello a concetti'


def profile(cuda, size=None):
    """The settings of the concept language model: the training text is a table of concepts, no neural network.

    There is no size to choose: the model grows with what it reads (its concepts), up to its memory budget,
    a quarter of the server's memory unless set. ``size`` is accepted for earlier callers and ignored.
    """
    return dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-lm-v1', vocabulary=32000,
                distillation_weight=0., early_stop_patience=0, validation_batches=16, save_seconds=300, device='cpu',
                chat_probes=True, auto_resume=True, sequence_length=256, concept_depth=6, concept_document=4,
                concept_memory_mb=0, concept_piece=1_000_000, concept_paths=False)


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
        return html.P('Ancora nessuna risposta: compaiono al primo salvataggio (ogni 10 passi, ogni 5 minuti e alla fine).', className='small text-muted')
    first, last = entries[0], entries[-1]
    header = (f"Ultimo salvataggio: passo {last['step']:,}"
              + ('' if last.get('validation_nll') is None else f" · errore sul riferimento {last['validation_nll']:.3f}")
              + (f" · confronto con il passo {first['step']:,}" if len(entries) > 1 else ''))
    rows = []
    for index, item in enumerate(last['answers']):
        before = first['answers'][index]['answer'] if len(entries) > 1 and index < len(first['answers']) else None
        from .generative import media_view
        rows.append(html.Tr([html.Td(item['question']),
                             html.Td(media_view(before or '—', small=True)),
                             html.Td(media_view(item['answer'] or '(nessuna risposta)', small=True))]))
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


def _direct(weights, progress):
    """The weights of the direct path (the first reading): one per context level."""
    depth = progress.get('context_depth')
    return weights[:depth+1] if isinstance(depth, int) else weights


def _document(progress):
    depth = progress.get('document_depth')
    return depth if isinstance(depth, int) else len(progress.get('weights_document', []))-1


def concept_footprint_view(item):
    """The quantities of the concept language model while it learns."""
    progress, config = item['progress'] or {}, item['config']
    step, total = progress.get('step', 0), config['total_steps']
    speed = progress.get('tokens_per_second')
    piece = config.get('concept_piece', 1_000_000)
    remaining = (total-step)*piece/speed if speed else None
    levels = progress.get('concepts_per_level') or []
    budget = progress.get('budget_bytes')
    number = lambda value, digits=3: '—' if value is None else f'{value:.{digits}f}'
    rows = [
        ('Modello', [
            ('Tipo', 'modello a concetti: nessuna rete neurale, nessun peso addestrato'),
            ('Vocabolario', f"{(progress.get('graft') or {}).get('vocabulary', 0):,} token" if progress.get('graft') else '—'),
            ('Livelli di contesto', f"da 0 a {progress.get('context_depth', config.get('concept_depth'))} token precedenti"),
            ('Livelli del testo in corso', f"da 1 a {progress.get('document_depth', config.get('concept_document'))} token precedenti, a qualunque distanza"),
            ('Concetti per livello', ' · '.join(f'{count:,}' for count in levels) if levels else '—'),
            ('Concetti in tutto', f"{progress['concepts_total']:,}" if progress.get('concepts_total') else '—'),
            ('Concetti generalizzati nel livello sopra', f"{progress.get('generalized_concepts', 0):,} (per stare nella memoria)"),
            ('Memoria dei concetti', (_bytes(progress['statistics_bytes']) if progress.get('statistics_bytes') else '—')
             + (f' di {_bytes(budget)} disponibili' if budget else '')),
            ('Concetti dei token', ' · '.join(f'{count:,}' for count in progress['token_concepts'])
             + ' (trovati dal modello stesso, dal generale al particolare)' if progress.get('token_concepts') else '—'),
            ('Percorsi verso il prossimo token', (f"{progress['paths']}: diretto"
             + (f" e {progress['paths']-1} attraverso i concetti dei token (contesti fino a {progress.get('path_depth')} token)" if progress['paths'] > 1 else '')
             + ' · pesi ' + ', '.join(f'{value:.2f}' for value in progress.get('path_weights', [])))
             if progress.get('paths') else '1: diretto'),
            ('Pesi scelti dal giudizio', ('contesto ' + ', '.join(f'{value:g}' for value in _direct(progress['weights_global'], progress))
                                          + ' · testo in corso ' + ', '.join(f'{value:g}' for value in progress.get('weights_document', [])[1:_document(progress)+1]))
             if progress.get('weights_global') else '—')]),
        ('Avanzamento e velocità', [
            ('Passo', f'{step:,} di {total:,}'),
            ('Token letti', f"{progress.get('tokens_trained', 0):,}"),
            ('Velocità', f'{speed:,.0f} token al secondo' if speed else '—'),
            ('Tempo che manca, stima', '—' if remaining is None else f'{remaining/3600:,.1f} ore' if remaining > 3600 else f'{remaining/60:,.0f} minuti'),
            ('Token per passo', f'{piece:,} (un pezzo del testo, contato e aggiunto)')]),
        ('Qualità', [
            ('Errore sul testo che sta per leggere', number(progress.get('training_nll'))),
            ('Errore su testo mai visto', number(progress.get('validation_nll'))),
            ('Migliore finora', number(progress.get('best_validation_nll'))),
            ('Prima di leggere questo testo', number(progress.get('baseline_nll')))]),
        ('Memoria e disco', [
            ('Memoria del processo (picco)', _bytes(progress['memory_peak_bytes']) if progress.get('memory_peak_bytes') else '—'),
            ('Un checkpoint', _bytes(item['checkpoint_bytes'])),
            ('Checkpoint tenuti', ', '.join(f"{value['count']} {name}" for name, value in
                                            zip(('recenti', 'permanenti', 'migliore'), (item['checkpoints'][key] for key in ('step', 'milestone', 'best'))))
             + f" · {_bytes(sum(value['bytes'] for value in item['checkpoints'].values()))}"),
            ('Testo e token', f"{_bytes(item['corpus_bytes'])} di testo, {_bytes(item['token_bytes'])} di token"),
            ('Spazio libero sul server', f"{_bytes(item['disk_free'])} di {_bytes(item['disk_total'])}")]),
    ]
    tables = [html.Div([html.H6(title, className='mt-2'),
                        dbc.Table(html.Tbody([html.Tr([html.Td(name), html.Td(value)]) for name, value in values]), size='sm', bordered=True)])
              for title, values in rows]
    note = html.P("Gli errori sono in nat per token: e elevato all'errore è fra quanti token, in media, il modello è incerto. "
                  "Si misurano solo su testo che il modello non ha contato.", className='small text-muted')
    return html.Div([dbc.Row([dbc.Col(table, md=6) for table in tables]), note])


def footprint_view(item):
    """The main sizes of the model while it learns, grouped by what they help to tune."""
    if item['config'].get('architecture') == 'concepts':
        return concept_footprint_view(item)
    progress, config = item['progress'] or {}, item['config']
    step, total = progress.get('step', 0), config['total_steps']
    speed = progress.get('tokens_per_second')
    per_step = config['batch_size']*config['accumulation']*config['sequence_length']
    remaining = (total-step)*per_step/speed if speed else None
    trainable, overall = progress.get('trainable_parameters'), progress.get('total_parameters')
    graft = progress.get('graft') or {}
    layers, total_layers = graft.get('replaced_layers') or [], graft.get('total_layers')
    graph = config.get('routing_profile') == 'concept-graph-v2'
    kept = config['window']+config['slots']
    levels = max(config['sequence_length']-config['window'], 1).bit_length()
    rows = [
        ('Modello', [
            ('Parametri totali', f'{overall:,}' if overall else '—'),
            ('Parametri che imparano', f"{trainable:,} ({100*trainable/overall:.1f}%)" if trainable and overall else '—'),
            ('Origine', 'nato da zero (pesi casuali)' if config.get('architecture') == 'native' else 'adattato da un modello preaddestrato'),
            ('Vocabolario', f"{graft['vocabulary']:,} token" if graft.get('vocabulary') else '—'),
            ('Layer a concetti', 'nessuno: attenzione normale (è il controllo)' if config.get('native_attention') == 'dense' else
             f"{len(layers)} su {total_layers}" if layers and total_layers else ('tutti' if config.get('graft_fraction') == 1. else 'metà')),
            ('Memoria di lettura per layer', 'un vettore per ogni token letto (cresce con il testo)' if config.get('native_attention') == 'dense' else
             f"{config['window']} token esatti + {config['slots']} concetti per livello: al massimo {config['window']+config['slots']*levels} vettori "
             f"su un pezzo di {config['sequence_length']} token (un livello in più a ogni raddoppio)" if graph else
             f"{config['window']} token esatti + {config['slots']} concetti = {kept} vettori, a qualunque lunghezza"),
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
            ('Al passo 0 (pesi casuali)' if config.get('architecture') == 'native' else 'Modello di partenza sullo stesso testo',
             '—' if progress.get('baseline_nll') is None else f"{progress['baseline_nll']:.3f}")]),
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
    extras = build['request'].get('extras') or []
    lines = [html.P(f"Miscela «{build['request']['mixture']}»" + (f" + {', '.join(extras)}" if extras else '') + f" · {status} · {build.get('phase') or ''}",
                    className='small mb-1'),
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
    html.H4('Apprendimento lungo · il modello a concetti'),
    html.P("Lo stesso modello della scheda Modello, applicato al testo: nessuna rete neurale e nessun peso addestrato. "
           "Il testo è una tabella: ogni posizione è una riga, i token che la precedono sono i suoi attributi, il token "
           "che segue è il target. Imparare è contare: ogni passo legge un pezzo di testo e aggiunge le sue somme ai "
           "concetti. Il tokenizer nasce dai tuoi testi, in italiano e inglese insieme. L'apprendimento salva il lavoro "
           "ogni 10 passi (dieci milioni di token) e ogni 5 minuti, riprende da solo dopo un riavvio del server e a ogni salvataggio risponde "
           "alle domande di prova. Come funziona: schede Matematica e Informatica.",
           className='small'),
    html.H5('1 · I dati: italiano e inglese, più codice, comandi, musica, immagini, suoni e video a scelta', className='mt-3'),
    html.P("Il worker scarica le fonti una alla volta da versioni fissate, verifica ogni file, tiene solo il testo e "
           "cancella il resto. La preparazione si mette in pausa e riprende senza perdere nulla.", className='small'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-mixture', placeholder='Scegli la quantità di dati', clearable=False), md=6),
             dbc.Col([dbc.Button('Prepara i dati', id='lm-long-build', color='primary', className='me-2'),
                      dbc.Button('Pausa', id='lm-long-build-pause', outline=True, color='warning', className='me-2'),
                      dbc.Button('Riprendi', id='lm-long-build-resume', outline=True, color='primary')], md=6)], className='mb-2'),
    html.Div(id='lm-long-mixture-info', className='small text-muted'),
    dbc.Label('Altri dati da aggiungere al testo (facoltativi)', className='mt-2 mb-0'),
    dbc.Checklist(id='lm-long-extras', value=[], inline=True, className='small'),
    html.Div(id='lm-long-extras-info', className='small text-muted mb-2'),
    html.Div(id='lm-long-build-status', role='status', className='small'), html.Div(id='lm-long-build-view'),
    dbc.Accordion([dbc.AccordionItem(html.Div(id='lm-long-sources'), title='Fonti, licenze e riferimento fisso')],
                  start_collapsed=True, className='my-2'),
    html.H5('2 · La coda di apprendimento', className='mt-3'),
    html.P("I testi vengono imparati uno dopo l'altro: ognuno parte dai concetti del precedente e aggiunge i suoi, "
           "senza dimenticare nulla. Puoi mettere in coda la miscela preparata e i tuoi file (testo, Markdown, Python, ZIP, "
           "EPUB, DOCX, PDF), e aggiungerne altri mentre impara con «Carica e aggiungi alla coda», sotto la coda.", className='small'),
    dbc.Label('Testi da imparare, in ordine'),
    dcc.Dropdown(id='lm-long-corpora', multi=True, placeholder='Scegli uno o più testi', className='mb-2'),
    html.Div(id='lm-long-size-info', className='small text-muted mb-2'),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupText('Salva ogni'), dbc.Input(id='lm-long-save', type='number', value=10, min=1, max=10000), dbc.InputGroupText('passi')]), md=6),
             dbc.Col(dbc.InputGroup([dbc.InputGroupText('Tieni per sempre ogni'), dbc.Input(id='lm-long-milestone', type='number', value=100, min=0, max=10000000), dbc.InputGroupText('passi')]), md=6)],
            className='mb-1'),
    html.Small("Un passo conta un milione di token, quindi i passi sono i milioni di token del testo (un testo di 300 MB ne ha circa 40): "
               "il testo si legge una volta sola. Si salva anche ogni 5 minuti e alla fine; «0» non tiene salvataggi permanenti.",
               className='text-muted d-block mb-2'),
    dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupText('Memoria massima dei concetti'), dbc.Input(id='lm-long-memory', type='number', value=0, min=0, max=262144, step=256), dbc.InputGroupText('MB')]), md=6),
             dbc.Col(html.Small("0 = automatica: un decimo della memoria del server (o del suo contenitore), al massimo 2 GB; vale per tutte le somme, e mentre somma il processo ne usa al massimo circa il triplo. Oltre questo limite i concetti che costano meno a chi "
                                "legge vengono generalizzati nel livello sopra.", className='text-muted'), md=6)], className='mb-2'),
    dbc.Row([dbc.Col(dbc.Checkbox(id='lm-long-paths', value=False, label='Concetti dei token e percorsi'), md=6),
             dbc.Col(html.Small("Il modello trova da sé i concetti dei token («gatto» e «cane» si somigliano) e legge il prossimo token anche dal "
                                "concetto generale al particolare. Su wikitext-2 (2 milioni di token) l'errore scende da 4,59 a 4,45; su testi "
                                "grandi, con il limite di memoria, peggiora (69 milioni di token, 2 GB: da 3,91 a 3,96), perché le somme dei "
                                "percorsi tolgono memoria ai contesti lunghi.", className='text-muted'), md=6)], className='mb-2'),
    # Earlier controls kept for the callbacks they feed: a concept model has no size, no passes and no piece length to choose.
    html.Div([dcc.Dropdown(id='lm-long-model', value='micro', options=[dict(label='micro', value='micro')]),
              dbc.Input(id='lm-long-passes', type='number', value=1), dbc.Input(id='lm-long-length', type='number', value=256)],
             style={'display': 'none'}),
    dbc.Label('Domande di prova (una per riga, al massimo 6; vuoto = domande predefinite in italiano e inglese)'),
    dbc.Textarea(id='lm-long-questions', rows=3, placeholder='Ciao! Chi sei?\nQual è la capitale della Francia?'),
    dbc.Button('Crea la coda e avvia', id='lm-long-start', color='primary', className='mt-2'),
    html.Div(id='lm-long-start-status', role='status', className='small mt-1'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-queue', placeholder='Coda da seguire', clearable=False), md=6),
             dbc.Col([dbc.Button('Pausa e salva', id='lm-long-pause', color='warning', className='me-2'),
                      dbc.Button('Riprendi', id='lm-long-resume', color='primary', className='me-2'),
                      dbc.Button('Mostra nel grafico', id='lm-long-chart', outline=True, color='primary', className='me-2'),
                      html.Div(dbc.Button('Avvia il controllo', id='lm-long-control'), style={'display': 'none'})], md=6)], className='mt-3 mb-2'),
    html.Div(id='lm-long-queue-view'),
    dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-add-corpus', placeholder='Aggiungi un testo in fondo alla coda'), md=8),
             dbc.Col(dbc.Button('Aggiungi', id='lm-long-add', outline=True, color='primary'), md=4)], className='mb-1'),
    html.Div([html.Small('Oppure carica un documento dal computer (testo, Markdown, Python, ZIP, EPUB, DOCX, PDF): '
                         'viene convertito in testo e aggiunto in fondo alla coda scelta sopra.', className='text-muted d-block'),
              html.Div(id='lm-long-file-upload'), html.Div(id='lm-long-upload-status', role='status', className='small'),
              dcc.Store(id='lm-long-upload-state')], className='mb-1'),
    html.Div(id='lm-long-queue-status', role='status', className='small'),
    html.H5('Grandezze del modello mentre impara', className='mt-3'),
    html.Div(id='lm-long-footprint'),
    html.H5('3 · Cosa risponde adesso', className='mt-3'),
    html.P("Le stesse domande a ogni salvataggio, con la stessa estrazione casuale (seme fisso): le differenze vengono solo da ciò che il modello "
           "ha imparato. Per fargli domande libere: Pausa e salva, poi «Fai scrivere un modello» negli strumenti avanzati "
           "con la casella «Conversazione», poi Riprendi. Scaricare e ispezionare il modello si fa da «Segui i progetti».",
           className='small'),
    html.Div(id='lm-long-answers'),
    dcc.Store(id='lm-long-catalog'), dcc.Store(id='lm-long-builds'), dcc.Interval(id='lm-long-poll', interval=5000),
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

    @app.callback(Output('lm-long-extras', 'options'), Input('lm-long-catalog', 'data'))
    def extras(catalog):
        return [dict(label=item['title'], value=item['id']) for item in (catalog or {}).get('domains', [])]

    @app.callback(Output('lm-long-extras-info', 'children'), Input('lm-long-extras', 'value'), State('lm-long-catalog', 'data'))
    def extras_info(chosen, catalog):
        items = [item for item in (catalog or {}).get('domains', []) if item['id'] in (chosen or [])]
        if not items:
            return ("Il testo italiano e inglese è il punto di partenza. Si possono aggiungere codice, comandi bash, musica (ABC), "
                    "immagini, suoni e video: il tokenizer impara da tutto insieme.")
        lines = [html.Li(f"{item['title']}: circa {item['corpus_mb']:,} MB di dati, {item['download_mb']:,} MB da scaricare · {'; '.join(item['sources'])}")
                 for item in items]
        hint = []
        if any(item['id'] in ('images', 'audio', 'video') for item in items):
            hint.append(html.Li("Per imparare immagini, suoni o video interi alza «Token letti per pezzo» (un'immagine circa 1.060 token, "
                                "una cifra pronunciata e un video di cifre meno di 2.048: con «Token letti per pezzo» a 2.048 entrano interi) e metti fra le domande di prova, per esempio, "
                                "«Disegna la cifra 7.»."))
        return html.Ul(lines + hint)

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
                  State('lm-workspace', 'data'), State('lm-long-mixture', 'value'), State('lm-long-builds', 'data'),
                  State('lm-long-extras', 'value'), prevent_initial_call=True)
    def build_action(start, pause, resume, workspace, mixture, items, extras=None):
        latest = (items or [None])[0]
        try:
            if dash.ctx.triggered_id == 'lm-long-build':
                backend(workspace, '/builds', 'POST', dict(mixture=mixture, extras=list(extras or [])))
                return 'Preparazione avviata: il primo file è in download.'
            if not latest:
                raise ValueError('Nessuna preparazione da mettere in pausa o riprendere.')
            action = 'pause' if dash.ctx.triggered_id == 'lm-long-build-pause' else 'resume'
            backend(workspace, f"/builds/{latest['id']}/{action}", 'POST', {})
            return 'Pausa richiesta: il punto raggiunto viene salvato.' if action == 'pause' else 'Preparazione ripresa.'
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-long-corpora', 'options'), Output('lm-long-add-corpus', 'options'), Output('lm-long-model', 'options'),
                  Output('lm-long-queue', 'options'), Output('lm-long-queue', 'value'), Output('lm-long-model', 'value'),
                  Input('lm-state', 'data'), State('lm-long-queue', 'value'), State('lm-long-model', 'value'))
    def queue_options(state, current, size=None):
        if not state:
            return [], [], [], [], None, size or 'micro'
        corpora = [dict(label=f"{item['name']} ({item['received']/1048576:,.1f} MB)", value=item['id'])
                   for item in state.get('corpora', []) if item['status'] == 'ready']
        cuda = (state.get('capabilities') or {}).get('cuda', False)
        models = [dict(label=item['title'] + ('' if key != 'base' or cuda else ' (non disponibile senza GPU)'), value=key,
                       disabled=key == 'base' and not cuda) for key, item in SIZES.items()]
        found = queues(state.get('jobs', []))
        options = [dict(label=f"{items[0]['config']['name'].split(' · ')[0]} · {len(items)} testi · #{key[:4]}", value=key) for key, items in found.items()]
        value = current if current in found else (options[0]['value'] if options else None)
        # The size stays chosen when the options are refreshed (an empty box meant no size was shown).
        allowed = [item['value'] for item in models if not item['disabled']]
        return corpora, corpora, models, options, value, size if size in allowed else 'micro'

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
                  State('lm-long-length', 'value'), State('lm-long-memory', 'value'), State('lm-long-paths', 'value'),
                  prevent_initial_call=True)
    def start(clicks, workspace, corpora, model, passes, save, milestone, questions, state, length=256, memory=0, paths=False):
        try:
            if not corpora:
                raise ValueError('Scegli almeno un testo da imparare.')
            cuda = ((state or {}).get('capabilities') or {}).get('cuda', False)
            base = dict(profile(cuda), name=QUEUE_NAME, passes=1., save_steps=int(save or 10),
                        milestone_steps=int(milestone or 0), concept_memory_mb=int(memory or 0), concept_paths=bool(paths),
                        probe_prompts=[line.strip() for line in (questions or '').splitlines() if line.strip()][:6])
            result = backend(workspace, '/queues', 'POST', dict(base=base, corpora=list(corpora)), timeout=120)
            return result.get('message', 'Coda creata.'), result['queue']
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update

    @app.callback(Output('lm-long-queue-status', 'children'), Output('lm-active-job', 'data', allow_duplicate=True),
                  Input('lm-long-add', 'n_clicks'), Input('lm-long-pause', 'n_clicks'), Input('lm-long-resume', 'n_clicks'),
                  Input('lm-long-control', 'n_clicks'),
                  State('lm-workspace', 'data'), State('lm-long-queue', 'value'), State('lm-long-add-corpus', 'value'),
                  State('lm-state', 'data'), prevent_initial_call=True)
    def queue_action(add, pause, resume, control, workspace, queue, corpus, state):
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
            if dash.ctx.triggered_id == 'lm-long-control':
                learned = [item for item in items if item['checkpoint']]
                if not learned:
                    raise ValueError('Il controllo si avvia dopo il primo salvataggio del primo testo.')
                result = backend(workspace, f"/jobs/{learned[0]['id']}/dense-control", 'POST', {})
                return ('Controllo avviato sul primo testo della coda, con lo stesso tokenizer, gli stessi passi e lo stesso seme. '
                        'Lo trovi in «Segui i progetti».'), result['id']
            if dash.ctx.triggered_id == 'lm-long-pause':
                backend(workspace, f"/jobs/{job['id']}/pause", 'POST')
                return 'Pausa richiesta: salva il punto raggiunto e si ferma. Puoi aggiornare il server tranquillamente.', dash.no_update
            backend(workspace, f"/jobs/{job['id']}/start", 'POST')
            return 'Ripreso dall\'ultimo salvataggio.', job['id']
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update

    @app.callback(Output('lm-long-queue-status', 'children', allow_duplicate=True), Input('lm-long-upload-state', 'data'),
                  State('lm-workspace', 'data'), State('lm-long-queue', 'value'), State('lm-state', 'data'), prevent_initial_call=True)
    def add_uploaded(uploaded, workspace, queue, state):
        """A document just uploaded from the queue section goes to the end of the chosen queue."""
        try:
            if not uploaded or not uploaded.get('id'):
                raise ValueError('Caricamento non riuscito.')
            if not queues((state or {}).get('jobs', [])).get(queue):
                raise ValueError(f"«{uploaded.get('name', 'Il documento')}» è caricato: scegli una coda e aggiungilo dal menu qui sopra.")
            job = backend(workspace, f'/queues/{queue}/add', 'POST', dict(corpus_id=uploaded['id']))
            return f"«{uploaded.get('name', 'Documento')}» aggiunto come testo {job['config']['series']['order']+1} della coda."
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-job', 'value', allow_duplicate=True), Input('lm-long-chart', 'n_clicks'),
                  State('lm-long-queue', 'value'), State('lm-state', 'data'), prevent_initial_call=True)
    def chart(clicks, queue, state):
        job = current_item(queues((state or {}).get('jobs', [])).get(queue) or [])
        return job['id'] if job else dash.no_update

    @app.callback(Output('lm-long-size-info', 'children'), Input('lm-long-model', 'value'))
    def size_info(size):
        return ("Non c'è una dimensione da scegliere: il modello è fatto dei concetti del testo che legge e cresce con esso, "
                "fino alla memoria massima. Contesti fino a 6 token precedenti, più i concetti del testo in corso a qualunque distanza.")

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
