"""The long training of the concept language model: data, learning queue, answers while it learns.

Every layer reads through concepts (graft fraction 1.0). The page drives the generative worker
through the same ``backend`` helper as the rest of the generative tab, and reads the projects from
``lm-state``, which that tab refreshes every five seconds.
"""
import dash
import numpy as np
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
    """The settings of the concept language model: the concept network, its concepts learned along the gradient.
    ``size`` is accepted for earlier callers and ignored."""
    return dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-pairs-v1', vocabulary=32000,
                distillation_weight=0., early_stop_patience=0, validation_batches=16, save_seconds=300, device='auto',
                chat_probes=True, auto_resume=True, sequence_length=256, concept_memory_mb=0, concept_piece=1_000_000)


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


def folded(title, children, key, class_name='mt-2'):
    """A fold the page redraws every few seconds: it stays open or closed as the reader left it."""
    return dbc.Accordion([dbc.AccordionItem(children, title=title, item_id=key+'-item')], id=key, start_collapsed=True,
                         persistence=True, persistence_type='session', className=class_name)


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
                     folded('Errore sul riferimento agli ultimi salvataggi', html.Ul(history, className='small'),
                            'lm-long-answers-history')])


def _bytes(value):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(value) < 1024 or unit == 'TB':
            return f'{value:,.1f} {unit}' if unit != 'B' else f'{value:,.0f} B'
        value /= 1024


def _tile(title, value, note=None):
    return dbc.Col(dbc.Card(dbc.CardBody([html.Div(title, className='small text-muted'), html.Div(value, className='fs-5 fw-semibold'),
                                          html.Div(note, className='small text-muted') if note else None], className='py-2')),
                   xs=6, md=4, className='mb-2')


def structure_progress(learning):
    """How far the structure is (0 to 1) and the seconds left, estimated from the time so far.

    The scales are the steps; inside a scale the concepts are joined from ``start`` down to one, each
    round joining a share of them, so the part done is the share of the way on a logarithmic scale.
    """
    if 'part_permille' in learning:                      # computed where it is learned, never going back
        part = min(max(learning['part_permille']/1000, 0.), 1.)
        seconds = learning.get('seconds') or 0.
        return part, (seconds*(1-part)/part if part > .02 else None)
    steps, done = max(learning.get('steps') or 1, 1), max((learning.get('step') or 1)-1, 0)
    inside = 0.
    if learning.get('phase') in ('subconcepts', 'readings'):
        inside = 1.
    elif 'inside_permille' in learning:
        inside = learning['inside_permille']/1000
    elif learning.get('start', 0) > 1:
        inside = 1-np.log(max(learning.get('concepts', 1), 1))/np.log(learning['start'])
    part = min(max((done+inside)/steps, 0.), 1.)
    seconds = learning.get('seconds') or 0.
    return part, (seconds*(1-part)/part if part > .02 else None)


def concept_summary(progress, config, checkpoint_bytes=None, status=None):
    """The few numbers that say how the concept model is doing, as tiles: progress, time, speed, error, size, concepts."""
    step, total = progress.get('step', 0), config.get('total_steps') or 0
    ready = bool(progress.get('graft'))
    speed = progress.get('tokens_per_second')
    piece = config.get('concept_piece', 1_000_000)
    remaining = (total-step)*piece/speed if speed and ready and status not in ('completed',) else None
    shape = progress.get('network')                             # the size of the network: layers, state, concepts
    share = 100*step/max(total, 1) if ready else 0
    error, before = progress.get('validation_nll'), progress.get('baseline_nll')
    perplexity = progress.get('validation_perplexity')
    size, budget = progress.get('statistics_bytes'), progress.get('budget_bytes')
    when = lambda seconds: '—' if seconds is None else f'{seconds/3600:,.1f} ore' if seconds > 3600 else f'{max(seconds, 0)/60:,.0f} minuti'
    learning = progress.get('structure') if ready and step == 0 and status not in ('completed',) else None
    network = progress.get('structure') if (progress.get('structure') or {}).get('phase') == 'network' and status != 'completed' else None
    if learning and learning.get('phase') not in ('done', 'network'):
        part, left = structure_progress(learning)
        doing = {'tokens': 'concetti dei token',
                 'scale': f"concetti dei tratti di {learning['width']:,} token",
                 'subconcepts': f"sotto-concetti dei tratti di {learning['width']:,} token",
                 'readings': f"quali letture usare fino a {learning['width']:,} token",
                 'topics': (f"di cosa parlano gli ultimi {learning['width']:,} token" if learning['width']
                            else "di cosa parla il documento")}.get(learning['phase'], 'struttura')
        if learning.get('start') and learning['phase'] in ('tokens', 'scale', 'topics'):
            doing += f" ({learning['concepts']:,} concetti da unire, partiti da {learning['start']:,})"
        where = f"scala {learning['step']} di {learning['steps']} al massimo · " if learning.get('steps') else ''
        first = _tile('Avanzamento', f'struttura {100*part:.0f}%',
                      f"prima di leggere il testo: {where}{doing} · da {learning['seconds']/60:,.0f} minuti")
        tiles = [first, _tile('Tempo che manca', when(left), 'per la struttura, al massimo: la validazione può fermarsi prima')]
        bar = dbc.Progress(value=100*part, className='mb-2', style={'height': '8px'}, color='info')
        return html.Div([bar, dbc.Row(tiles)])
    else:
        inside = ''
        if network:                                             # the network inside the step: how far it is
            kinds = ("la rete com'è", 'il doppio della larghezza', 'uno strato in più', 'concetti divisi in due')
            part = min(network.get('done', 0) / max(network.get('piece', 1), 1), 1.)
            inside = (f" · gara di crescita al {100 * (network['race'] - 1 + part) / 4:.0f}%: quattro candidati imparano lo stesso "
                      f"pezzo uno dopo l'altro, ognuno da 0 a 100% (una linea ciascuno nel grafico «Dentro il passo»); ora il "
                      f"{network['race']}° di 4, {kinds[network['race'] - 1] if network['race'] <= 4 else ''}, al {100 * part:.0f}% del pezzo"
                      if network.get('race')
                      else f" · la rete rilegge tutto il testo: {network.get('learned', 0):,} token imparati" if network.get('again')
                      else f" · la rete impara il pezzo: {network.get('learned', 0):,} token imparati")
            if network.get('held_milli'):
                inside += f", validazione migliore {network['held_milli'] / 1000:.3f}"
            if network.get('seconds'):
                inside += f", da {network['seconds'] / 60:,.0f} minuti"
        first = _tile('Avanzamento', f'{share:.0f}%' if ready else 'prepara i token',
                      (f'passo {step:,} di {total:,} · {progress.get("tokens_trained", 0):,} token letti' + inside) if ready else 'il totale si conosce quando i token sono pronti')
    tiles = [
        first,
        _tile('Tempo che manca', 'finito' if status == 'completed' else when(remaining), 'stima dalla velocità finora'),
        _tile('Velocità', f'{speed:,.0f} token/s' if speed else f"{network['speed']:,} token/s" if network and network.get('speed') else '—',
              f'un passo è {piece:,} token' + ('' if speed else ' · nel passo in corso' if network and network.get('speed')
                                                  else ' · si misura mentre la rete impara')),
        _tile('Errore su testo mai visto', '—' if error is None else f'{error:.3f}',
              (f'esita fra circa {perplexity:,.0f} token' if perplexity else '') + (f' · prima di leggere {before:.3f}' if before is not None else '')),
        _tile('Dimensione del modello', _bytes(size) if size else '—',
              'i parametri della rete' + (f' · file scaricabile {_bytes(checkpoint_bytes)}' if checkpoint_bytes else '')),
        _tile('Rete', f"{shape['layers']} {'strato' if shape['layers'] == 1 else 'strati'}" if shape else '—',
              f"{shape['dim']} numeri di stato · " + ', '.join(f'{g}×{k}' for g, k in shape['groups']) + ' concetti per strato · '
              f"{shape['parameters']:,} parametri" if shape else 'si conosce alla fine del primo passo'),
    ]
    bar = dbc.Progress(value=share, className='mb-2', style={'height': '8px'}) if ready else None
    return html.Div([bar, dbc.Row(tiles)])


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
            ('Tipo', 'rete a concetti: i concetti dei token e di ogni strato imparati col gradiente prevedendo il token che segue'),
            ('Vocabolario', f"{(progress.get('graft') or {}).get('vocabulary', 0):,} token" if progress.get('graft') else
             (f"{config.get('vocabulary', 32000):,} token, lo stesso del testo precedente della coda (si stanno preparando i token di questo testo)"
              if config.get('warm_start') else f"{config.get('vocabulary', 32000):,} token (si sta imparando dai testi della coda)")),
            ('Contesto letto', 'tutto il documento, attraverso le somme dei concetti di ogni strato'),
            ('Concetti per strato', ' · '.join(f'{count:,}' for count in levels) if levels else '— (dopo il primo pezzo di testo)'),
            ('Concetti in tutto', f"{progress['concepts_total']:,}" if progress.get('concepts_total') else '—'),
            ('Parametri della rete', _bytes(progress['statistics_bytes']) if progress.get('statistics_bytes') else '—')]),
        ('Avanzamento e velocità', [
            ('Dispositivo di addestramento', (progress.get('device') or '—').upper()),
            ('Passo', f'{step:,} di {total:,}' if progress.get('graft') else '— (il totale si conosce quando i token del testo sono pronti)'),
            ('Token letti', f"{progress.get('tokens_trained', 0):,}"),
            ('Velocità', f'{speed:,.0f} token al secondo' if speed else '—'),
            ('Tempo che manca, stima', '—' if remaining is None else f'{remaining/3600:,.1f} ore' if remaining > 3600 else f'{remaining/60:,.0f} minuti'),
            ('Token per passo', f'{piece:,} (un pezzo del testo, letto e riletto finché la validazione migliora)')]),
        ('Qualità', [
            ('Errore sul testo che sta per leggere', number(progress.get('training_nll'))),
            ('Errore su testo mai visto', number(progress.get('validation_nll'))),
            ('Migliore finora', number(progress.get('best_validation_nll'))),
            ('Prima di leggere questo testo', number(progress.get('baseline_nll')))]),
        *([('Rete a concetti', [
            ('Apprendimento', 'si è fermata: la validazione non migliorava più' if progress.get('network_stopped') else 'impara a ogni pezzo'),
            *[(f"Crescita a {race['tokens']:,} token", f"{race['choice']}: larghezza {race['dim']}, {race['layers']} strati, "
               f"concetti {race['concepts']}, validazione {race['held_out']:.3f}") for race in progress.get('network_races') or []],
            *[(f"Strato {layer['layer']}",
               f"{' · '.join(f'{a:.0f}' for a in layer['active'])} concetti attivi per gruppo; "
               + (f"in un concetto il token più frequente è il {layer['tied']:.0%}. " if layer['tied'] is not None else '')
               + ' '.join(f"Concetto {c['concept']} ({c['mass']:.0%}): " + ' | '.join(c['examples']) for c in layer['concepts']))
              for layer in progress['network_concepts']]])] if progress.get('network_concepts') else []),
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
                  "Si misurano solo su testo che il modello non ha letto.", className='small text-muted')
    graph = network_graph_view(progress.get('network_graph'))
    details = folded('Tutti i dettagli: vocabolario, concetti, contesto, memoria e disco',
                     [dbc.Row([dbc.Col(table, md=6) for table in tables]), note] + ([graph] if graph is not None else []),
                     'lm-long-details')
    return html.Div([concept_summary(progress, config, item.get('checkpoint_bytes'), item.get('status')), details])


def network_graph_view(graph):
    """The graph of the network, to explore: a column per layer, a node per concept that works (its size is its mass,
    its name the tokens that enter it and the tokens that follow it), solid edges from a layer to the next (where the
    positions of a concept go), dotted edges inside a layer (which concepts a concept reads). A click on a concept shows
    all its edges and lists them; a second click lets it go."""
    if not graph or not graph.get('nodes'):
        return None
    note = html.P("Un nodo è un concetto che lavora: il nome dice quali token vi entrano → quali token vengono dopo; la grandezza, "
                  "la sua parte del gruppo. Linee continue: il concetto dello strato dopo dove vanno le posizioni del concetto, "
                  "più di quanto ci andrebbero comunque. Linee punteggiate: quali concetti dello stesso strato il concetto legge. "
                  "Si mostrano i concetti e gli archi che pesano più di una parte uguale; senza scelta, fra gli strati solo l'arco "
                  "più caratteristico di ogni concetto. Clic su un concetto: tutti i suoi archi, e l'elenco qui sotto.",
                  className='small text-muted')
    return html.Div([html.H6('Il grafo della rete', className='mt-3'),
                     dcc.Graph(id='lm-network-graph', figure=network_graph_figure(graph), config={'scrollZoom': True}),
                     dcc.Store(id='lm-network-data', data=graph), html.Div(id='lm-network-info'), note])


def _concept_name(n):
    return f"{' '.join(n['enters'])} → {' '.join(n['follows'])}"


def network_graph_figure(graph, selected=None):
    import plotly.graph_objects as go
    nodes = sorted(graph['nodes'], key=lambda n: (n['layer'], n['group'], n['concept']))
    rows, place = {}, {}
    for n in nodes:
        rows[n['layer']] = rows.get(n['layer'], -1) + 1
        place[n['id']] = (n['layer'], rows[n['layer']])
    if selected in place:                                       # a chosen concept: all its edges
        shown = [e for e in graph['edges'] if selected in (e['source'], e['target'])]
        near = {selected} | {e['source'] for e in shown} | {e['target'] for e in shown}
    else:
        strongest = {}                                          # between layers, each concept's most telling edge
        for e in graph['edges']:
            if e['kind'] == 'porta a' and e.get('lift', 0) > strongest.get(e['source'], {}).get('lift', 0):
                strongest[e['source']] = e
        shown, near = [e for e in graph['edges'] if e['kind'] == 'legge'] + list(strongest.values()), None
    figure = go.Figure()
    for e in shown:
        if e['source'] not in place or e['target'] not in place:
            continue
        (x0, y0), (x1, y1) = place[e['source']], place[e['target']]
        inside = e['kind'] == 'legge'
        xs = [x0, x0 - .25, x1] if inside else [x0, (x0 + x1) / 2, x1]
        ys = [y0, (y0 + y1) / 2, y1]
        figure.add_scatter(x=xs, y=ys, mode='lines', showlegend=False, hoverinfo='text' if selected in place else 'skip',
                           hovertext=f"{e['source']} {e['kind']} {e['target']} · peso {e['weight']:.2f}",
                           line=dict(width=1 + 4 * e['weight'], dash='dot' if inside else 'solid',
                                     color='rgba(220,90,40,.6)' if inside else 'rgba(60,110,200,.45)'))
    faded = lambda n: near is not None and n['id'] not in near
    figure.add_scatter(x=[place[n['id']][0] for n in nodes], y=[place[n['id']][1] for n in nodes], mode='markers+text',
                       text=[_concept_name(n) for n in nodes], textposition='middle right', showlegend=False,
                       customdata=[n['id'] for n in nodes],
                       textfont=dict(color=['#bbb' if faded(n) else '#222' for n in nodes]),
                       marker=dict(size=[8 + 40 * n['mass'] for n in nodes],
                                   color=['#ddd' if faded(n) else ('#d9480f' if n['id'] == selected else '#2c6fbb') for n in nodes]),
                       hovertext=[f"concetto {n['id']} · {n['mass']:.0%} del gruppo<br>entrano: {', '.join(n['enters'])}<br>"
                                  f"poi viene: {', '.join(n['follows'])}" for n in nodes], hoverinfo='text')
    layers = sorted(rows)
    figure.update_layout(height=max(320, 26 * max(rows.values()) + 120), margin=dict(l=20, r=20, t=40, b=30),
                         title='Il grafo della rete: concetti e relazioni, misurati sul testo di validazione',
                         uirevision='network-graph', clickmode='event', hovermode='closest',
                         xaxis=dict(tickvals=layers, ticktext=[f'strato {i}' for i in layers], range=[min(layers) - .35, max(layers) + .9]),
                         yaxis=dict(visible=False, autorange='reversed'))
    return figure


def network_graph_info(graph, selected):
    """What a chosen concept is: its tokens, the concepts it reads and that read it, where its positions go and come from."""
    by = {n['id']: n for n in graph['nodes']}
    if selected not in by:
        return None
    n = by[selected]
    out = sorted([e for e in graph['edges'] if e['source'] == selected], key=lambda e: -e['weight'])
    into = sorted([e for e in graph['edges'] if e['target'] == selected], key=lambda e: -e['weight'])
    words = {'legge': ('legge', 'è letto da'), 'porta a': ('porta a', 'viene da')}
    rows = [html.Tr([html.Td(words[e['kind']][0]), html.Td(f"{e['target']} · {_concept_name(by[e['target']])}"), html.Td(f"{e['weight']:.2f}")])
            for e in out] + \
           [html.Tr([html.Td(words[e['kind']][1]), html.Td(f"{e['source']} · {_concept_name(by[e['source']])}"), html.Td(f"{e['weight']:.2f}")])
            for e in into]
    return html.Div([html.P([html.B(f"Concetto {n['id']}"), f" (strato {n['layer']}, gruppo {n['group']}, {n['mass']:.0%} del gruppo) · "
                             f"entrano: {', '.join(n['enters'])} · poi viene: {', '.join(n['follows'])}"], className='small mb-1'),
                     dbc.Table(html.Tbody(rows), size='sm', bordered=True, className='small') if rows else
                     html.P('Nessun arco che pesi più di una parte uguale.', className='small text-muted')])


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
            ('Dispositivo di addestramento', (progress.get('device') or '—').upper()),
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


FORMATS = 'testo, Markdown, Python, ZIP, EPUB, DOCX, PDF'


def step(number, title, *children):
    return dbc.Card(dbc.CardBody([html.H4(f'{number} · {title}'), *children]), className='my-3', color='primary', outline=True)


layout = html.Div([
    dbc.Alert(id='lm-guide', color='info', className='my-3'),
    step(1, 'Prepara i testi',
         html.P("Il modello impara dai testi che prepari qui. Puoi usare una miscela pubblica (italiano e inglese, più codice, comandi, "
                "musica, immagini, suoni e video a scelta) oppure i tuoi documenti, o entrambi.", className='small'),
         html.H6('Una miscela pubblica'),
         html.P("Il server scarica le fonti una alla volta da versioni fissate, verifica ogni file, tiene solo il testo e cancella il resto. "
                "Si mette in pausa e riprende senza perdere nulla.", className='small text-muted mb-1'),
         dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-mixture', placeholder='Scegli la quantità di dati', clearable=False), md=6),
                  dbc.Col([dbc.Button('Prepara i dati', id='lm-long-build', color='primary', className='me-2'),
                           dbc.Button('Pausa', id='lm-long-build-pause', outline=True, color='warning', className='me-2'),
                           dbc.Button('Riprendi', id='lm-long-build-resume', outline=True, color='primary')], md=6)], className='mb-2'),
         html.Div(id='lm-long-mixture-info', className='small text-muted'),
         dbc.Label('Altri dati da aggiungere (facoltativi)', className='mt-2 mb-0'),
         dbc.Checklist(id='lm-long-extras', value=[], inline=True, className='small'),
         html.Div(id='lm-long-extras-info', className='small text-muted mb-2'),
         html.Div(id='lm-long-build-status', role='status', className='small'), html.Div(id='lm-long-build-view'),
         dbc.Accordion([dbc.AccordionItem(html.Div(id='lm-long-sources'), title='Fonti, licenze e riferimento fisso')],
                       start_collapsed=True, className='my-2'),
         html.H6('I tuoi documenti', className='mt-3'),
         html.P(f"Scegli un file ({FORMATS}, oppure .jsonl con un campo «text» per riga) e premi il pulsante: viene caricato a pezzi e "
                "convertito in testo. Quando è pronto compare fra i testi del passo 2.", className='small text-muted mb-1'),
         html.Div(id='lm-file-upload'), html.Div(id='lm-upload-status', role='status', className='small')),
    step(2, 'Crea la coda e avvia',
         html.P("La coda impara i testi uno dopo l'altro: ognuno parte da ciò che hanno imparato i precedenti e aggiunge il suo, senza "
                "dimenticare nulla. Il vocabolario (32.000 token) si impara una volta, all'avvio, dai testi che sono in coda in quel momento: "
                "metti subito tutti quelli che vuoi.", className='small'),
         dbc.Label('Testi da imparare, in ordine'),
         dcc.Dropdown(id='lm-long-corpora', multi=True, placeholder='Scegli uno o più testi preparati al passo 1', className='mb-2'),
         html.Div(id='lm-long-size-info', className='small text-muted mb-2'),
         dbc.Label('Domande di prova (una per riga, al massimo 6; vuoto = domande predefinite in italiano e inglese)'),
         dbc.Textarea(id='lm-long-questions', rows=3, placeholder='Ciao! Chi sei?\nQual è la capitale della Francia?'),
         dbc.Accordion([dbc.AccordionItem([
             dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupText('Salva ogni'), dbc.Input(id='lm-long-save', type='number', value=10, min=1, max=10000), dbc.InputGroupText('passi')]), md=6),
                      dbc.Col(dbc.InputGroup([dbc.InputGroupText('Tieni per sempre ogni'), dbc.Input(id='lm-long-milestone', type='number', value=100, min=0, max=10000000), dbc.InputGroupText('passi')]), md=6)],
                     className='mb-1'),
             html.Small("Un passo è un pezzo di un milione di token: la rete lo legge e lo rilegge finché il testo tenuto da parte migliora. "
                        "Si salva anche ogni 5 minuti e alla fine; «0» non tiene salvataggi permanenti.", className='text-muted d-block mb-2'),
             dbc.Checkbox(id='lm-long-growth', value=True, label='La rete cresce con i dati'),
             html.Small("La rete parte piccola. Ogni volta che il testo letto raddoppia, la rete com'è e tre copie cresciute "
                        "(più larga, uno strato in più, concetti divisi) rileggono il testo letto finché la validazione "
                        "migliora; resta la più semplice fra quelle che la migliore non batte di più di un errore standard.",
                        className='text-muted d-block mb-2'),
         ], title='Impostazioni (facoltative: i valori proposti vanno bene)')], start_collapsed=True, className='my-2'),
         # Earlier controls kept for the callbacks they feed: a concept model has no size, no passes and no piece length to choose.
         html.Div([dcc.Dropdown(id='lm-long-model', value='micro', options=[dict(label='micro', value='micro')]),
                   dbc.Input(id='lm-long-passes', type='number', value=1), dbc.Input(id='lm-long-length', type='number', value=256)],
                  style={'display': 'none'}),
         dbc.Button('Crea la coda e avvia', id='lm-long-start', color='primary', className='mt-2'),
         html.Div(id='lm-long-start-status', role='status', className='small mt-1')),
    step(3, 'Segui la coda',
         html.P("Qui vedi a che punto è ogni testo, come cresce il modello e cosa risponde. Il server fa una cosa alla volta: mentre la coda "
                "impara, la preparazione dei dati e la scrittura aspettano.", className='small'),
         dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-queue', placeholder='Scegli la coda', clearable=False), md=6),
                  dbc.Col([dbc.Button('Pausa e salva', id='lm-long-pause', color='warning', className='me-2'),
                           dbc.Button('Riprendi', id='lm-long-resume', color='primary', className='me-2'),
                           dbc.Button('Apri nel passo 4', id='lm-long-chart', outline=True, color='primary', className='me-2'),
                           html.Div(dbc.Button('Avvia il controllo', id='lm-long-control'), style={'display': 'none'})], md=6)], className='mb-2'),
         html.Div(id='lm-long-queue-view'),
         html.Div(id='lm-long-queue-status', role='status', className='small'),
         dbc.Accordion([dbc.AccordionItem([
             dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-add-corpus', placeholder='Un testo già preparato'), md=8),
                      dbc.Col(dbc.Button('Aggiungi in fondo alla coda', id='lm-long-add', outline=True, color='primary'), md=4)], className='mb-2'),
             html.Small(f'Oppure un documento dal computer ({FORMATS}): viene caricato, convertito in testo e aggiunto in fondo alla coda scelta. '
                        'Usa il vocabolario della coda, imparato all\'avvio.', className='text-muted d-block'),
             html.Div(id='lm-long-file-upload'), html.Div(id='lm-long-upload-status', role='status', className='small'),
             dcc.Store(id='lm-long-upload-state'),
         ], title='Aggiungi un testo alla coda')], start_collapsed=True, className='my-2'),
         html.H5('Il modello mentre impara', className='mt-3'),
         html.Div(id='lm-long-footprint'),
         html.H5('Cosa risponde', className='mt-3'),
         html.P("Le stesse domande a ogni salvataggio, con la stessa estrazione casuale: le differenze vengono solo da ciò che il modello ha "
                "imparato. Per fargli domande libere e scaricarlo: passo 4.", className='small'),
         html.Div(id='lm-long-answers')),
    dcc.Store(id='lm-long-catalog'), dcc.Store(id='lm-long-builds'), dcc.Store(id='lm-network-selected', storage_type='session'), dcc.Interval(id='lm-long-poll', interval=5000),
])


def guide(state, builds):
    """What to do now, from the state of the workspace: one sentence and the step to go to."""
    if not state:
        return 'Collegamento al server in corso…'
    running = next((item for item in builds or [] if item['status'] in ('queued', 'running', 'pausing')), None)
    paused = next((item for item in builds or [] if item['status'] in ('paused', 'interrupted')), None)
    corpora = [item for item in state.get('corpora', []) if item['status'] == 'ready']
    found = queues(state.get('jobs', []))
    active = [items for items in found.values() if any(job['status'] in ('queued', 'preparing', 'running', 'pausing') for job in items)]
    if active:
        items = active[0]
        job = current_item(items)
        order = (job['config'].get('series') or {}).get('order', 0)+1
        progress = job['progress'] or {}
        if job['status'] in ('queued', 'preparing') or not progress.get('graft'):
            now = f"La coda sta preparando i token del testo {order} di {len(items)} ({job['config']['series'].get('label', '')}): poi comincia a imparare da sola."
        else:
            now = f"La coda sta imparando il testo {order} di {len(items)}: passo {progress.get('step', 0):,} di {job['config']['total_steps']:,}."
        learned = [item for item in items if item['status'] == 'completed']
        tail = (' Puoi seguirla al passo 3; un testo che ha finito si prova e si scarica al passo 4.' if learned else
                ' Seguila al passo 3. Per provarlo e scaricarlo aspetta che un testo finisca (passo 4).')
        return [html.B('Adesso: '), now, tail]
    if running:
        return [html.B('Adesso: '), 'si stanno preparando i dati (passo 1). Quando sono pronti, crea la coda al passo 2.']
    if any(job['status'] in ('paused', 'interrupted') for items in found.values() for job in items):
        return [html.B('Adesso: '), 'la coda è in pausa. Per continuare premi «Riprendi» al passo 3.']
    if paused:
        return [html.B('Adesso: '), 'la preparazione dei dati è in pausa: premi «Riprendi» al passo 1.']
    if found and all(job['status'] == 'completed' for items in found.values() for job in items):
        return [html.B('Adesso: '), "la coda ha finito. Prova il modello e scaricalo al passo 4 (scegli l'ultimo testo), "
                "aggiungi altri testi al passo 3, oppure crea una coda nuova al passo 2."]
    if not corpora:
        return [html.B('Per cominciare: '), 'prepara una miscela pubblica o carica i tuoi documenti al passo 1.']
    return [html.B('Prossimo passo: '), 'scegli i testi al passo 2 e premi «Crea la coda e avvia».']


def register(app):
    @app.callback(Output('lm-network-graph', 'figure'), Output('lm-network-info', 'children'), Output('lm-network-selected', 'data'),
                  Input('lm-network-graph', 'clickData'), Input('lm-network-data', 'data'), State('lm-network-selected', 'data'),
                  prevent_initial_call=True)
    def explore(click, graph, selected):
        # A click chooses a concept (a second click on it lets it go); a new graph keeps the concept chosen.
        if not graph:
            return dash.no_update, None, selected
        if dash.ctx.triggered_id == 'lm-network-graph' and click:
            chosen = (click.get('points') or [{}])[0].get('customdata')
            if chosen:
                selected = None if chosen == selected else chosen
        return network_graph_figure(graph, selected), network_graph_info(graph, selected), selected

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
                  State('lm-long-length', 'value'), State('lm-long-growth', 'value'), prevent_initial_call=True)
    def start(clicks, workspace, corpora, model, passes, save, milestone, questions, state, length=256, growth=False):
        try:
            if not corpora:
                raise ValueError('Scegli almeno un testo da imparare.')
            cuda = ((state or {}).get('capabilities') or {}).get('cuda', False)
            base = dict(profile(cuda), name=QUEUE_NAME, passes=1., save_steps=int(save or 10),
                        milestone_steps=int(milestone or 0), concept_network_growth=bool(growth), probe_prompts=[line.strip() for line in (questions or '').splitlines() if line.strip()][:6])
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
                        'Lo trovi al passo 4, in «Progetto».'), result['id']
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
        return ("Non c'è una dimensione da scegliere: la rete a concetti parte piccola e cresce con il testo che legge, "
                "quando la validazione dice che una rete più grande prevede meglio. Legge il documento attraverso i suoi concetti, a qualunque distanza.")

    @app.callback(Output('lm-guide', 'children'), Input('lm-state', 'data'), Input('lm-long-builds', 'data'))
    def show_guide(state, builds):
        return guide(state, builds)

    @app.callback(Output('lm-job', 'value', allow_duplicate=True), Input('lm-long-queue', 'value'), State('lm-job', 'value'),
                  State('lm-state', 'data'), prevent_initial_call=True)
    def follow_queue(queue, job, state):
        """Step 4 starts on the last text of the chosen queue that has a saved model."""
        items = queues((state or {}).get('jobs', [])).get(queue) or []
        if job and any(item['id'] == job for item in items):
            return dash.no_update
        saved = [item for item in items if item['checkpoint']]
        return saved[-1]['id'] if saved else dash.no_update

    @app.callback(Output('lm-long-footprint', 'children'), Input('lm-state', 'data'), Input('lm-long-queue', 'value'),
                  State('lm-workspace', 'data'))
    def footprint(state, queue, workspace):
        items = [job for job in queues((state or {}).get('jobs', [])).get(queue) or [] if job['status'] != 'created']
        if not items:
            return html.P('Compaiono quando il primo testo comincia a imparare.', className='small text-muted')
        try:
            current = current_item(items)
            learned = [job for job in items if job['checkpoint'] and (job['progress'] or {}).get('graft')]
            if not (current['progress'] or {}).get('graft') and learned:
                # The next text is still preparing its tokens: the model is the one learned so far.
                previous = learned[-1]
                order = lambda job: (job['config'].get('series') or {}).get('order', 0)+1
                note = dbc.Alert(f"Il testo {order(current)} sta preparando i token: qui sotto il modello appreso fino al testo "
                                 f"{order(previous)}, da cui il testo {order(current)} ripartirà.", color='info', className='small py-2')
                return html.Div([note, footprint_view(backend(workspace, f"/jobs/{previous['id']}/footprint"))])
            return footprint_view(backend(workspace, f"/jobs/{current['id']}/footprint"))
        except ValueError as exc:
            return html.P(str(exc), className='small text-muted')

    @app.callback(Output('lm-long-answers', 'children'), Input('lm-state', 'data'), Input('lm-long-queue', 'value'),
                  State('lm-workspace', 'data'))
    def answers(state, queue, workspace):
        items = queues((state or {}).get('jobs', [])).get(queue) or []
        learned = [job for job in items if job['status'] != 'created']
        if not learned:
            return answers_view([])
        current = current_item(learned)
        # The text learning now first, then the earlier ones: the answers are those of the last save that has them.
        candidates = [current]+[job for job in reversed(learned) if job is not current and job['checkpoint']]
        try:
            for job in candidates:
                found = backend(workspace, f"/jobs/{job['id']}/answers")
                if found:
                    if job is not current:
                        order = lambda item: (item['config'].get('series') or {}).get('order', 0)+1
                        return html.Div([dbc.Alert(f"Il testo {order(current)} non ha ancora salvato: qui sotto le risposte del modello "
                                                   f"appreso fino al testo {order(job)}.", color='info', className='small py-2'),
                                         answers_view(found)])
                    return answers_view(found)
            return answers_view([])
        except ValueError as exc:
            return html.P(str(exc), className='small text-muted')
