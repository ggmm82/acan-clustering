"""The long training of the concept language model: data, learning queue, answers while it learns.

Every layer reads through concepts (graft fraction 1.0). The page drives the generative worker
through the same ``backend`` helper as the rest of the generative tab, and reads the projects from
``lm-state``, which that tab refreshes every five seconds.
"""
import math

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
    """The settings of the concept language model: a hierarchical graph of concepts, abstractions and sub-concepts.
    ``size`` is accepted for earlier callers and ignored."""
    return dict(protocol='controlled-v1', architecture='concepts', routing_profile='concept-pairs-v1', vocabulary=32000,
                distillation_weight=0., early_stop_patience=0, validation_batches=16, save_seconds=300, device='auto',
                chat_probes=True, auto_resume=True, sequence_length=256, concept_memory_mb=0, concept_piece=1_000_000,
                concept_max_dim=64, concept_max_layers=0, concept_max_heads=0, concept_max_concepts=0, concept_start_dim=0,
                concept_start_layers=0, concept_start_heads=0, concept_start_concepts=0)


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


def answers_view(entries, job=None):
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
        how = (dbc.Button('Come l\'ha scritta', id={'type': 'lm-answer-open', 'index': f'{job}|{index}'}, size='sm',
                          outline=True, color='primary') if job else None)
        rows.append(html.Tr([html.Td(item['question']),
                             html.Td(media_view(before or '—', small=True)),
                             html.Td([media_view(item['answer'] or '(nessuna risposta)', small=True), how])]))
    history = [html.Li(f"passo {entry['step']:,}" + ('' if entry.get('validation_nll') is None else f": {entry['validation_nll']:.3f}"))
               for entry in entries[-10:]]
    return html.Div([html.P(header, className='small'),
                     dbc.Table([html.Thead(html.Tr([html.Th('Domanda'), html.Th('Prima'), html.Th('Adesso')])), html.Tbody(rows)],
                               bordered=True, size='sm', responsive=True),
                     folded('Errore sul riferimento agli ultimi salvataggi', html.Ul(history, className='small'),
                            'lm-long-answers-history')])


def _word(words, j):
    return (words[j].replace('\n', '⏎') if 0 <= j < len(words) else '?').strip() or '·'


def answer_node(path, key):
    """A node of the walk through how an answer was written: its title, what was measured there and the nodes it
    leads to, each (key, label, weight 0-1). Keys: token:p, level:p:l, view:p:l:h, concept:id, word:j, memory:p."""
    words, steps = path['words'], {step['position']: step for step in path['steps']}
    names = path.get('names') or {}
    kind, _, rest = key.partition(':')
    parts = rest.split(':')
    if kind == 'token':
        step = steps[int(parts[0])]; pos = step['position']
        near = [(f"level:{pos}:{level['level']}", f"strato {level['level']}: direbbe «{(level.get('says') or [('?', 0)])[0][0].strip()}»",
                 min(1., abs(level.get('effect') or 0) / 5)) for level in step['levels']]
        near.append((f'memory:{pos}', 'memoria del documento', step['memory']['weight']))
        text = (f"«{_word(words, pos)}», scritto con probabilità {math.exp(step['logp']):.0%}. Viene dagli strati della rete e dalla "
                f"memoria del documento: il peso accanto a ogni strato è quanto la probabilità scende chiudendolo.")
        return f"«{_word(words, pos)}»", text, near
    if kind == 'level':
        pos, l = int(parts[0]), int(parts[1]); level = steps[pos]['levels'][l - 1]
        says = ', '.join(f"«{t.strip() or t}» {q:.0%}" for t, q in level.get('says', []))
        effect = level.get('effect')
        near = [(f"view:{pos}:{l}:{n + 1}", f"vista {n + 1} → concetto {view['id']}", view['membership'])
                for n, view in enumerate(level.get('lit', []))]
        text = (f"Dopo lo strato {l} la rete direbbe: {says}. "
                + ('' if effect is None else f"Chiudendolo, la probabilità del token {'scende' if effect >= 0 else 'sale'} di "
                   f"{abs(effect):.2f} in log. ")
                + 'Ogni vista dello strato cade in un concetto e guarda alcune parole.')
        return f'Strato {l}', text, near
    if kind == 'view':
        pos, l, h = int(parts[0]), int(parts[1]), int(parts[2]); view = steps[pos]['levels'][l - 1]['lit'][h - 1]
        near = [(f"concept:{view['id']}", f"concetto {view['id']}", view['membership'])]
        near += [(f'word:{j}', f'«{_word(words, j)}»', weight) for j, weight in view.get('looked', [])]
        text = (f"La posizione cade nel concetto {view['id']} ({view['membership']:.0%}) e guarda le parole: "
                + (', '.join(f"«{_word(words, j)}» {w:.0%}" for j, w in view.get('looked', [])) or 'nessuna in particolare') + '.')
        return f'Vista {h} dello strato {l}', text, near
    if kind == 'concept':
        cid = rest
        held = next(([j for j, _ in view.get('held', [])] for step in path['steps'] for level in step['levels']
                     for view in level.get('lit', []) if view['id'] == cid), [])
        lit = [step['position'] for step in path['steps'] if any(view['id'] == cid for level in step['levels']
                                                                 for view in level.get('lit', []))]
        near = [(f'token:{p}', f'token «{_word(words, p)}»', 1.) for p in lit if p in steps]
        name = names.get(cid)
        text = ((f'Nel grafo dei concetti: {name}. ' if name else '')
                + (f"Le parole di questo testo che gli appartengono di più: {', '.join(f'«{_word(words, j)}»' for j in held)}. " if held else '')
                + 'Qui accanto i token della risposta in cui si accende.')
        return f'Concetto {cid}', text, near
    if kind == 'word':
        j = int(parts[0]); near = []
        for step in path['steps']:
            used = max([w for level in step['levels'] for view in level.get('lit', []) for jj, w in view.get('looked', []) if jj == j]
                       + [w for jj, w in step['memory']['sources'] if jj == j] + [0])
            if used > .05:
                near.append((f"token:{step['position']}", f"token «{_word(words, step['position'])}»", used))
        where = 'della domanda' if j < path['prompt_length'] else 'della risposta'
        return f'Parola «{_word(words, j)}»', f'Una parola {where}. Qui accanto i token che l\'hanno letta o copiata.', near
    if kind == 'memory':
        pos = int(parts[0]); memory = steps[pos]['memory']
        near = [(f'word:{j}', f'«{_word(words, j)}»', w) for j, w in memory['sources']]
        text = (f"La memoria del documento pesa {memory['weight']:.0%} su questo token: copia la parola che seguiva i punti del "
                'testo più simili a quello attuale. Qui accanto le parole da cui copia.')
        return 'Memoria', text, near
    raise KeyError(key)


def token_graph(path, pos, current=None):
    """The way to one written token, left to right: the words read (of the question and of the answer so far), the
    concept each view of every layer falls into, and the token; the document memory beside. A click on a node opens it."""
    import plotly.graph_objects as go
    words, step = path['words'], next(s for s in path['steps'] if s['position'] == pos)
    levels = step['levels']
    read = {}
    for level in levels:
        for view in level.get('lit', []):
            for j, w in view.get('looked', []):
                read[j] = max(read.get(j, 0.), w)
    for j, w in step['memory']['sources']:
        read[j] = max(read.get(j, 0.), w)
    shown = sorted(sorted(read, key=lambda j: -read[j])[:12])            # in the order of the text
    nodes, edges = {}, []                                                 # key -> (x, y, label, hover, kind)
    for n, j in enumerate(shown):
        where = 'domanda' if j < path['prompt_length'] else 'risposta'
        nodes[f'word:{j}'] = (0, 1 - 2 * (n + .5) / max(len(shown), 1), f'«{_word(words, j)}»', f'parola della {where}', 'word')
    names = path.get('names') or {}
    for l, level in enumerate(levels, 1):
        lit = level.get('lit', [])
        for h, view in enumerate(lit, 1):
            key = f'view:{pos}:{l}:{h}'
            nodes[key] = (l, 1 - 2 * (h - .5) / max(len(lit), 1), view['id'],
                          f"strato {l}, vista {h}: concetto {view['id']} ({view['membership']:.0%})"
                          + (f"<br>{names[view['id']]}" if view['id'] in names else ''), 'concept')
            for j, w in view.get('looked', []):
                if f'word:{j}' in nodes:
                    edges.append((f'word:{j}', key, w, 'read'))
            if l > 1 and h <= len(levels[l - 2].get('lit', [])):
                edges.append((f'view:{pos}:{l - 1}:{h}', key, view['membership'], 'down'))
    last = len(levels) + 1
    nodes[f'token:{pos}'] = (last, 0., f'«{_word(words, pos)}»', f"token scritto, probabilità {math.exp(step['logp']):.0%}", 'token')
    nodes[f'memory:{pos}'] = (last - .5, -.9, 'memoria', f"memoria del documento: pesa {step['memory']['weight']:.0%}", 'memory')
    for h, view in enumerate(levels[-1].get('lit', []) if levels else [], 1):
        edges.append((f'view:{pos}:{len(levels)}:{h}', f'token:{pos}', view['membership'] * (1 - step['memory']['weight']), 'out'))
    edges.append((f'memory:{pos}', f'token:{pos}', step['memory']['weight'], 'out'))
    for j, w in step['memory']['sources']:
        if f'word:{j}' in nodes:
            edges.append((f'word:{j}', f'memory:{pos}', w, 'copy'))
    colors = dict(read='rgba(44,111,187,.45)', down='rgba(120,120,120,.45)', out='rgba(231,111,81,.6)', copy='rgba(42,157,143,.6)')
    figure = go.Figure()
    for a, b, w, kind in edges:
        (xa, ya), (xb, yb) = nodes[a][:2], nodes[b][:2]
        figure.add_scatter(x=[xa, xb], y=[ya, yb], mode='lines', hoverinfo='skip', showlegend=False,
                           line=dict(width=.5 + 6 * max(0., min(1., w)), color=colors[kind]))
    fill = dict(word='#2c6fbb', concept='#6c757d', token='#e76f51', memory='#2a9d8f')
    keys = list(nodes)
    figure.add_scatter(x=[nodes[k][0] for k in keys], y=[nodes[k][1] for k in keys], mode='markers+text', showlegend=False,
                       text=[nodes[k][2] for k in keys], hovertext=[nodes[k][3] for k in keys], hoverinfo='text', customdata=keys,
                       textposition=['middle left' if nodes[k][4] == 'word' else 'middle right' if nodes[k][4] == 'token'
                                     else 'top center' for k in keys],
                       marker=dict(size=[22 if k == current else 14 for k in keys],
                                   color=[fill[nodes[k][4]] for k in keys],
                                   line=dict(width=[3 if k == current else 0 for k in keys], color='#000')))
    ticks = ['parole lette'] + [f'strato {l}' for l in range(1, len(levels) + 1)] + ['token scritto']
    figure.update_layout(height=460, margin=dict(l=90, r=90, t=30, b=10), clickmode='event', plot_bgcolor='white',
                         xaxis=dict(tickvals=list(range(len(ticks))), ticktext=ticks, side='top', showgrid=False, zeroline=False,
                                    range=[-.6, len(ticks) - .4]),
                         yaxis=dict(visible=False, range=[-1.1, 1.1]), uirevision=f'{pos}')
    return figure


def answer_walk_view(data, walk):
    """How an answer was written, as a walk: (the answer's tokens above, the graph from the question to the token
    or None, the trail and the open node below)."""
    if not data or not walk:
        return None, None, None
    paths = data.get('paths') or []
    index = walk.get('question', 0)
    if index >= len(paths) or not paths[index]:
        return dbc.Alert('Questo modello non ha il percorso delle risposte: compare al prossimo salvataggio.', color='info',
                         className='small py-2'), None, None
    path, trail = paths[index], walk.get('trail') or []
    words, start = path['words'], path['prompt_length']
    current = trail[-1] if trail else None
    tokens = [dbc.Button(_word(words, step['position']), id={'type': 'lm-walk-go', 'index': f"token:{step['position']}"},
                         size='sm', className='me-1 mb-1', color='primary',
                         outline=current != f"token:{step['position']}") for step in path['steps']]
    body = [html.Div([html.B(f"Domanda: {data['questions'][index]}"), html.Span(f" · salvataggio del passo {data['step']:,}", className='text-muted'),
                      dbc.Button('Chiudi', id={'type': 'lm-walk-go', 'index': 'close'}, size='sm', color='link', className='ms-2')]),
            html.Div('Risposta, token per token: clicca un token per vedere da dove viene.', className='small text-muted mt-1'),
            html.Div(tokens, className='mb-2')]
    figure, detail = None, None
    if current:
        crumbs = []
        for n, key in enumerate(trail):
            crumbs.append(dbc.Button(answer_node(path, key)[0], id={'type': 'lm-walk-back', 'index': n}, size='sm', color='secondary',
                                     outline=n != len(trail) - 1, className='me-1 mb-1'))
            if n < len(trail) - 1:
                crumbs.append(html.Span('›', className='me-1'))
        title, text, near = answer_node(path, current)
        links = [dbc.Button(f'{label} · {weight:.0%}', id={'type': 'lm-walk-go', 'index': key}, size='sm', outline=True,
                            color='secondary', className='me-1 mb-1') for key, label, weight in near]
        pos = int(trail[0].split(':')[1])
        body.append(html.Div('Dalla domanda al token: le parole lette (blu), il concetto in cui cade ogni vista di ogni strato '
                             '(grigio), la memoria del documento (verde) e il token scritto (arancio). Lo spessore è quanto ha '
                             'pesato; un clic su un nodo lo apre qui sotto.', className='small text-muted'))
        figure = token_graph(path, pos, current)
        detail = html.Div([html.Div(crumbs), dbc.Card(dbc.CardBody([html.H6(title), html.P(text, className='small mb-2'),
                                                                    html.Div(links or html.Span('Nessun passo oltre.', className='small text-muted'))]),
                                                      className='mb-2')])
    return html.Div(body, className='mt-2'), figure, detail


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


def _candidate_label(kind):
    return {'teste': 'il doppio delle teste', 'larghezza': 'il doppio della larghezza',
            'strato': 'uno strato in più', "com'è": "la rete com'è"}.get(kind, kind)


def _concepts_used_tile(used, shape):
    """How many concepts each level really uses (effective number, mean over its heads) out of those it has:
    near the total, the level could use more; far below, it already has enough."""
    if not used:
        return _tile('Concetti usati', '—', 'si misura sul testo di validazione')
    have = [concepts for _, concepts in (shape or {}).get('groups', [])]
    parts = [f'{u:.0f} di {have[i]}' if i < len(have) else f'{u:.0f}' for i, u in enumerate(used)]
    return _tile('Concetti usati', ' · '.join(parts),
                 'per testa in ogni strato: vicino al totale ne servirebbero di più, molto sotto ne ha già abbastanza')


def _network_details(shape):
    groups = ', '.join(f'{heads}×{concepts}' for heads, concepts in shape['groups'])
    return f"{shape['dim']} numeri di stato · {groups} (teste × concetti per testa, per strato) · {shape['parameters']:,} parametri"


def _race_live(network):
    candidates = network['race_candidates']
    scores = network['race_scores']
    best = min(scores, default=None)
    current = network['race'] - 1
    part = min(1., network['done'] / max(network['piece'], 1))
    rows = []
    for index, candidate in enumerate(candidates):
        if index < len(scores):
            score = f'{scores[index]:.3f}'
            state = 'In testa per NLL' if scores[index] == best else 'Valutato'
        elif index == current:
            score = '—'
            state = 'In validazione' if network.get('race_evaluating') else f'In prova · {part:.0%}'
        else:
            score, state = '—', 'In attesa'
        rows.append(html.Tr([html.Th(_candidate_label(candidate), scope='row', className='fw-normal'),
                             html.Td(score, className='text-end'), html.Td(state)],
                            className='table-info' if index == current and index >= len(scores) else None))
    table = dbc.Table([html.Thead(html.Tr([html.Th('Candidato', scope='col'), html.Th('NLL', scope='col', className='text-end'),
                                         html.Th('Stato', scope='col')])), html.Tbody(rows)], size='sm', responsive=True,
                      className='small mb-1')
    heading = f"Gara di crescita · tornata {network['race_round']}"
    reference_nll = network.get('race_reference_nll')
    reference = html.P(f'NLL della rete di partenza {reference_nll:.3f}', className='small mb-1') if reference_nll is not None else None
    criteria = ('La scelta favorisce la rete più semplice entro un errore standard dalla migliore.' if reference_nll is None else
                'La scelta favorisce la rete più semplice tra quelle che battono la partenza e restano entro un errore standard dalla migliore.')
    shape = network.get('candidate_network')
    candidate_details = (html.P([html.Strong(_candidate_label(candidates[current])),
                                 f" · {_network_details(shape)} · {_bytes(shape['bytes'])}"], className='small mb-1')
                         if shape and current >= len(scores) else None)
    projection = network.get('race_projection')
    outcome = None
    if projection:
        lead = 'Scelta della tornata' if len(scores) == len(candidates) else 'Se i punteggi confermati restano decisivi'
        candidate = 'si conserva la rete di partenza' if projection['retained'] else _candidate_label(projection['candidate'])
        following = 'Nuova tornata sullo stesso pezzo.' if projection['grows'] else 'La gara per questo pezzo si chiude.'
        outcome = html.Div([html.Strong(lead), f" · {candidate} ({projection['held_out']:.3f}). {following}"], className='small')
    return dbc.Card(dbc.CardBody([html.H6(heading), reference, candidate_details, table, outcome,
                                 html.Small(criteria, className='text-muted')], className='py-2'), className='mb-2')


def concept_summary(progress, config, checkpoint_bytes=None, status=None):
    """The few numbers that say how the concept model is doing, as tiles: progress, time, speed, error, size, concepts."""
    step, total = progress.get('step', 0), config.get('total_steps') or 0
    ready = bool(progress.get('graft'))
    speed = progress.get('tokens_per_second')
    piece = config.get('concept_piece', 1_000_000)
    remaining = (total-step)*piece/speed if speed and ready and status not in ('completed',) else None
    share = 100*step/max(total, 1) if ready else 0
    error, before = progress.get('validation_nll'), progress.get('baseline_nll')
    perplexity = progress.get('validation_perplexity')
    when = lambda seconds: '—' if seconds is None else f'{seconds/3600:,.1f} ore' if seconds > 3600 else f'{max(seconds, 0)/60:,.0f} minuti'
    structure = (progress.get('structure') or {}) if status in ('running', 'pausing') else {}
    learning = structure if ready and step == 0 else None
    network = structure if structure.get('phase') == 'network' else None
    shape = (network or {}).get('network') or progress.get('network')
    size = shape.get('bytes', progress.get('statistics_bytes')) if shape else None
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
            kinds = ("la rete com'è", 'il doppio della larghezza', 'uno strato in più', 'concetti divisi in due',
                     'il doppio delle teste')
            candidates = network.get('race_total', 4)  # progress written by an earlier worker records four
            race = network.get('race', 0)
            label = network.get('candidate') or (kinds[race-1] if 1 <= race <= len(kinds) else '')
            label = _candidate_label(label)
            part = min(network.get('done', 0) / max(network.get('piece', 1), 1), 1.)
            round_label = f" · tornata {network['race_round']}" if network.get('race_round') else ''
            if race:
                inside = (f" · gara di crescita al {100 * (race - 1 + part) / candidates:.0f}%{round_label} · "
                          f"{race}° di {candidates}, {label}")
            else:
                reading = 'rilegge tutto il testo' if network.get('again') else 'impara il pezzo'
                inside = f" · la rete {reading}: {network.get('learned', 0):,} token imparati"
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
              'NLL di validazione' + (f' · prima di leggere {before:.3f}' if before is not None else '')),
        _tile('Perplessità', '—' if not perplexity else f'{perplexity:,.0f}',
              'fra quanti token il modello esita in media, sul testo mai visto'
              + (f' · prima di leggere {math.exp(before):,.0f}' if before is not None and before < 700 else '')),
        _concepts_used_tile(progress.get('concepts_used') or [sum(layer['active']) / max(len(layer['active']), 1)
                                                             for layer in progress.get('network_concepts') or []], shape),
        _tile('Dimensione del modello', _bytes(size) if size else '—',
              'i parametri della rete' + (f' · file scaricabile {_bytes(checkpoint_bytes)}' if checkpoint_bytes else '')),
        _tile('Rete corrente', f"{shape['layers']} {'strato' if shape['layers'] == 1 else 'strati'}" if shape else '—',
              _network_details(shape) if shape else 'in attesa della configurazione del worker'),
    ]
    bar = dbc.Progress(value=share, className='mb-2', style={'height': '8px'}) if ready else None
    race_card = _race_live(network) if network and network.get('race_candidates') else None
    return html.Div([bar, dbc.Row(tiles), race_card])


def _race_summary(race):
    heads = f"{race['heads']} teste, " if 'heads' in race else ''
    layer = f" nello strato {race['layer']}" if race.get('layer') else ''
    winner = (f"{race['choice']}{layer} · larghezza {race['dim']}, {race['layers']} strati, "
              f"{heads}concetti {race['concepts']}, NLL {race['held_out']:.3f}")
    if race.get('retained'):
        winner += ' · conservata la rete di partenza'
    if race.get('reference_nll') is not None:
        winner += f" · NLL di partenza {race['reference_nll']:.3f}"
    tested = race.get('candidates')
    if not tested:
        return winner
    return html.Div([html.Div(winner), html.Small(' · '.join(
        f"{choice['choice']}{' ' + str(choice['layer']) if choice.get('layer') else ''} {choice['held_out']:.3f}"
        for choice in tested))])


def concept_footprint_view(item):
    """The quantities of the concept language model while it learns."""
    progress, config = item['progress'] or {}, item['config']
    step, total = progress.get('step', 0), config['total_steps']
    speed = progress.get('tokens_per_second')
    piece = config.get('concept_piece', 1_000_000)
    remaining = (total-step)*piece/speed if speed else None
    live = (progress.get('structure') or {}) if item.get('status') in ('running', 'pausing') else {}
    shape = live.get('network') or progress.get('network')
    levels = [heads * concepts for heads, concepts in shape['groups']] if shape else progress.get('concepts_per_level') or []
    size = shape.get('bytes', progress.get('statistics_bytes')) if shape else None
    number = lambda value, digits=3: '—' if value is None else f'{value:.{digits}f}'
    kinds = config.get('concept_growth_kinds')
    cap_parts = [f"{label} iniziale {int(config[key])}" for label, key in (
        ('larghezza', 'concept_start_dim'), ('strati', 'concept_start_layers'), ('teste', 'concept_start_heads'),
        ('concetti', 'concept_start_concepts')) if config.get(key)] + (
        [f"crescite ammesse: {', '.join(kinds) or 'nessuna'}"] if kinds is not None and len(kinds) < 4 else []) + [
        f"{label} {int(config.get(key) or 0) or 'illimitato'}" for label, key in (
        ('larghezza', 'concept_max_dim'), ('livelli', 'concept_max_layers'),
        ('gruppi', 'concept_max_heads'), ('concetti/gruppo', 'concept_max_concepts'))]
    layer_p = (shape or {}).get('layer_parameters') or 0
    concept_p = (shape or {}).get('layer_concept_parameters') or 0
    concept_share = (f"{100 * concept_p / layer_p:.0f}% ({concept_p:,} su {layer_p:,} parametri degli strati)"
                     if layer_p > 0 else None)
    rows = [
        ('Modello', [
            ('Tipo', 'grafo gerarchico di concetti, astrazioni e sotto-concetti particolari collegati da matrici esplicite'),
            ('Vocabolario', f"{(progress.get('graft') or {}).get('vocabulary', 0):,} token" if progress.get('graft') else
             (f"{config.get('vocabulary', 32000):,} token, lo stesso del testo precedente della coda (si stanno preparando i token di questo testo)"
              if config.get('warm_start') else f"{config.get('vocabulary', 32000):,} token (raggruppati in concetti lessicali semantici)")),
            ('Contesto letto', 'tutto il documento, esclusivamente attraverso le somme dei concetti e le loro relazioni'),
            ('Concetti per livello', ' → '.join(f'{count:,}' for count in levels) if levels else '—'),
            ('Concetti in tutto', f'{sum(levels):,}' if levels else '—'),
            *([('Quota concetti negli strati', concept_share)] if concept_share else []),
            *( [('Tetti di crescita', ' · '.join(cap_parts))] if config.get('concept_network_growth', True) else [('Taglia fissa', 'crescita disattivata')] ),
            ('Dimensione dei parametri', _bytes(size) if size else '—')]),
        ('Avanzamento e velocità', [
            ('Dispositivo di addestramento', (progress.get('device') or '—').upper()),
            ('Passo', f'{step:,} di {total:,}' if progress.get('graft') else '— (il totale si conosce quando i token del testo sono pronti)'),
            ('Token letti', f"{progress.get('tokens_trained', 0):,}"),
            ('Velocità', f'{speed:,.0f} token al secondo' if speed else '—'),
            ('Tempo che manca, stima', '—' if remaining is None else f'{remaining/3600:,.1f} ore' if remaining > 3600 else f'{remaining/60:,.0f} minuti'),
            ('Token per passo', f'{piece:,} nuovi al massimo. Durante la crescita, ogni tornata rilegge lo stesso pezzo.')]),
        ('Qualità', [
            ('Errore sul testo che sta per leggere', number(progress.get('training_nll'))),
            ('Errore su testo mai visto', number(progress.get('validation_nll'))),
            ('Migliore finora', number(progress.get('best_validation_nll'))),
            ('Prima di leggere questo testo', number(progress.get('baseline_nll')))]),
        *([('Grafo dei concetti', [
            ('Apprendimento', 'si è fermata: la validazione non migliorava più' if progress.get('network_stopped') else 'raffina concetti e sotto-concetti a ogni pezzo'),
            *[(f"Crescita a {race['tokens']:,} token" + (f" · tornata {race['round']}" if race.get('round') else ''), _race_summary(race))
              for race in progress.get('network_races') or []],
            *[(f"Livello {layer['layer']}",
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
    details = folded('Tutti i dettagli: vocabolario, concetti, contesto, memoria e disco',
                     [dbc.Row([dbc.Col(table, md=6) for table in tables]), note],
                     'lm-long-details')
    return html.Div([concept_summary(progress, config, item.get('checkpoint_bytes'), item.get('status')), details])


def graph_section(item):
    """The inspectable graph of the concept network, or what it is waiting for."""
    progress = item['progress'] or {}
    graph = network_graph_view(progress.get('network_graph'))
    if graph is None:
        waiting = ('Il grafo ispezionabile di concetti, astrazioni e sotto-concetti compare appena i token sono pronti e la rete legge il primo brano.'
                   if item.get('status') in ('queued', 'preparing') or not progress.get('graft') else
                   'Il grafo ispezionabile di concetti, astrazioni e sotto-concetti si sta popolando sul testo di validazione…')
        return html.Div([html.H6('Il grafo ispezionabile di concetti, astrazioni e sotto-concetti', className='mt-3'),
                         html.P(waiting, className='small text-muted')])
    return graph


MAX_GRAPH_CONCEPTS_PER_LEVEL = 8


def _level_label(layer, total):
    if total <= 1:
        return f'Livello {layer} · Concetti'
    if layer == 1:
        return 'Livello 1 · Astrazioni generali'
    if layer == total:
        return f'Livello {layer} · Sotto-concetti particolari'
    return f'Livello {layer} · Concetti intermedi'


def _short_word(words):
    for w in (words or []):
        s = str(w).strip()
        if s:
            return s[:14] + ('…' if len(s) > 14 else '')
    return '—'


def _concept_name(n):
    return f"{', '.join((n.get('enters') or [])[:2])} → {', '.join((n.get('follows') or [])[:2])}"


def _node_badge(n, show_head=False):
    prefix = f"V{n['group']}·C{n['concept']}" if show_head else f"C{n['concept']}"
    return f"{prefix} ({n['mass']:.0%})<br>{_short_word(n.get('enters'))} → {_short_word(n.get('follows'))}"


def network_graph_view(graph):
    """The inspectable top-down graph of concepts, abstractions and multi-parent sub-concepts."""
    if not graph or not graph.get('nodes'):
        return None
    heads = sorted({int(n.get('group', 1)) for n in graph['nodes']})
    default_head = heads[0] if heads else 1
    head_options = ([dict(label=f'Punto di vista {h}', value=h) for h in heads]
                    + ([dict(label='Tutte le viste (principali)', value=0)] if len(heads) > 1 else []))
    selector = (dbc.Row([
        dbc.Col(html.Small('Testa di attenzione:', className='text-muted fw-semibold'), width='auto'),
        dbc.Col(dbc.RadioItems(id='lm-network-head', options=head_options, value=default_head, inline=True,
                               persistence=True, persistence_type='session', className='small')),
    ], className='align-items-center mb-1') if len(heads) > 1 else
        html.Div(dbc.RadioItems(id='lm-network-head', options=[dict(label='Vista 1', value=default_head)],
                                value=default_head, style={'display': 'none'})))
    note = html.P("Dall'alto verso il basso: dal livello delle astrazioni generali fino ai sotto-concetti particolari (come nel grafo tabulare). "
                  "Ogni nodo indica «contesto riconosciuto → token predetto» e la sua quota nella vista. "
                  "Le linee blu mostrano l'ereditarietà padre → sotto-concetto (anche da più concetti padre contemporaneamente). "
                  "Clicca su un nodo per vedere tutti i suoi concetti padre (% padre), i sotto-concetti figli e i legami contestuali.",
                  className='small text-muted mt-1')
    return html.Div([html.H6('Il grafo ispezionabile di concetti, astrazioni e sotto-concetti', className='mt-3'),
                     selector,
                     dcc.Graph(id='lm-network-graph', figure=network_graph_figure(graph, head=default_head),
                               config={'displayModeBar': False}),
                     dcc.Store(id='lm-network-data', data=graph), html.Div(id='lm-network-info'), note])


def network_graph_figure(graph, selected=None, head=1):
    import plotly.graph_objects as go
    all_nodes = graph.get('nodes') or []
    if not all_nodes:
        return go.Figure()
    heads = sorted({int(n.get('group', 1)) for n in all_nodes})
    chosen_head = int(head) if head is not None else (heads[0] if heads else 1)
    if chosen_head > 0 and chosen_head not in heads:
        chosen_head = heads[0]
    by_id = {n['id']: n for n in all_nodes}
    if selected not in by_id or (chosen_head > 0 and by_id[selected].get('group') != chosen_head):
        selected = None
    pool = [n for n in all_nodes if chosen_head == 0 or n.get('group') == chosen_head]
    layers = sorted({n['layer'] for n in pool})
    total_layers = max(layers) if layers else 1

    # Keep any neighbor of the selected concept visible even if outside top-8 mass.
    pinned = set()
    if selected:
        pinned.add(selected)
        for e in graph.get('edges') or ():
            if e['source'] == selected and e['target'] in by_id:
                if chosen_head == 0 or by_id[e['target']].get('group') == chosen_head:
                    pinned.add(e['target'])
            elif e['target'] == selected and e['source'] in by_id:
                if chosen_head == 0 or by_id[e['source']].get('group') == chosen_head:
                    pinned.add(e['source'])

    visible_nodes, omitted = [], 0
    place = {}
    for lyr in layers:
        in_layer = sorted([n for n in pool if n['layer'] == lyr], key=lambda n: (-n['mass'], n['group'], n['concept']))
        top = in_layer[:MAX_GRAPH_CONCEPTS_PER_LEVEL]
        top_ids = {n['id'] for n in top}
        extra = [n for n in in_layer[MAX_GRAPH_CONCEPTS_PER_LEVEL:] if n['id'] in pinned and n['id'] not in top_ids]
        row_nodes = sorted(top + extra, key=lambda n: (n['group'], -n['mass'], n['concept']))
        omitted += max(0, len(in_layer) - len(row_nodes))
        count = len(row_nodes)
        for idx, n in enumerate(row_nodes):
            x_pos = (idx - (count - 1) / 2.) * 1.35
            place[n['id']] = (x_pos, float(lyr))
            visible_nodes.append(n)

    vis_ids = set(place)
    edges_pool = [e for e in (graph.get('edges') or ()) if e['source'] in vis_ids and e['target'] in vis_ids]
    if selected in place:
        shown = [e for e in edges_pool if selected in (e['source'], e['target'])]
        near = {selected} | {e['source'] for e in shown} | {e['target'] for e in shown}
    else:
        strongest, by_child = {}, {}
        for e in edges_pool:
            if e['kind'] == 'porta a':
                if e.get('lift', 0) > strongest.get(e['source'], {}).get('lift', 0):
                    strongest[e['source']] = e
                by_child.setdefault(e['target'], []).append(e)
        multi_parents = []
        for child_edges in by_child.values():
            ranked = sorted(child_edges, key=lambda e: -(e.get('parent_share', e['weight'])))
            multi_parents.extend(ranked[:2])
        seen_pairs, shown = set(), []
        for e in list(strongest.values()) + multi_parents:
            pair = (e['source'], e['target'])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                shown.append(e)
        near = None

    figure = go.Figure()
    for e in shown:
        (x0, y0), (x1, y1) = place[e['source']], place[e['target']]
        inside = e['kind'] == 'legge'
        xs = [x0, (x0 + x1) / 2., x1]
        ys = [y0, y0 - .22, y1] if inside else [y0, (y0 + y1) / 2., y1]
        rel = 'legge nel contesto' if inside else 'specializza in'
        pshare = f" · quota padre {e['parent_share']:.0%}" if not inside and 'parent_share' in e else ''
        strength = e.get('parent_share', e['weight']) if not inside else e['weight']
        figure.add_scatter(x=xs, y=ys, mode='lines', showlegend=False, hoverinfo='text',
                           hovertext=f"{e['source']} → {e['target']} ({rel}) · peso {e['weight']:.2f}{pshare}",
                           line=dict(width=1.3 + 4. * min(float(strength), 1.),
                                     dash='dot' if inside else 'solid',
                                     color='rgba(217,72,15,.65)' if inside else 'rgba(44,111,187,.52)'))

    faded = lambda n: near is not None and n['id'] not in near
    level_palette = {1: '#1c7ed6', total_layers: '#0ca678'}
    node_colors = [
        '#dee2e6' if faded(n) else ('#d9480f' if n['id'] == selected else level_palette.get(n['layer'], '#5c7cfa'))
        for n in visible_nodes
    ]
    show_head = chosen_head == 0 and len(heads) > 1
    figure.add_scatter(
        x=[place[n['id']][0] for n in visible_nodes],
        y=[place[n['id']][1] for n in visible_nodes],
        mode='markers+text',
        text=[_node_badge(n, show_head=show_head) for n in visible_nodes],
        textposition='bottom center',
        showlegend=False,
        customdata=[n['id'] for n in visible_nodes],
        textfont=dict(size=11, color=['#adb5bd' if faded(n) else '#212529' for n in visible_nodes]),
        marker=dict(
            size=[18 + 28 * min(float(n['mass']), 1.) for n in visible_nodes],
            color=node_colors,
            line=dict(width=2, color='#ffffff'),
        ),
        hovertext=[
            f"<b>{n.get('role', 'concetto').capitalize()} {n['id']}</b> (vista {n['group']} · {n['mass']:.1%})<br>"
            f"Contesto riconosciuto: {', '.join(n.get('enters') or ())}<br>"
            f"Token predetti: {', '.join(n.get('follows') or ())}"
            for n in visible_nodes
        ],
        hoverinfo='text',
    )
    max_cols = max(1, max(sum(1 for n in visible_nodes if n['layer'] == lyr) for lyr in layers))
    half_span = max(2.2, (max_cols * 1.35) / 2. + .7)
    view_title = f'Punto di vista {chosen_head}' if chosen_head > 0 else 'Tutte le viste'
    extra_note = f' · principali su +{omitted} concetti minori' if omitted > 0 else ''
    figure.update_layout(
        height=max(260, 155 * len(layers) + 85),
        margin=dict(l=180, r=30, t=45, b=45),
        plot_bgcolor='#f8f9fa',
        title=f'Gerarchia dei concetti ({view_title}{extra_note})',
        uirevision=f'network-graph:{chosen_head}',
        clickmode='event',
        hovermode='closest',
        xaxis=dict(visible=False, range=[-half_span, half_span]),
        yaxis=dict(
            tickvals=[float(lyr) for lyr in layers],
            ticktext=[_level_label(lyr, total_layers) for lyr in layers],
            range=[max(layers) + .55, min(layers) - .4],
            gridcolor='rgba(0,0,0,0.06)',
            zeroline=False,
        ),
    )
    return figure


def network_graph_info(graph, selected):
    """What a chosen concept is: its abstraction/sub-concept role, multiple parents, specializations, and predicted tokens."""
    by = {n['id']: n for n in graph['nodes']}
    if selected not in by:
        return None
    n = by[selected]
    out = sorted([e for e in graph['edges'] if e['source'] == selected], key=lambda e: -e['weight'])
    into = sorted([e for e in graph['edges'] if e['target'] == selected],
                  key=lambda e: -(e.get('parent_share', e['weight'])))
    out_label = lambda e: 'specializza in sotto-concetto' if e['kind'] == 'porta a' else 'legge nel contesto'
    in_label = lambda e: 'concetto padre' if e['kind'] == 'porta a' else 'è letto da'
    in_metric = lambda e: (f"{e['parent_share']:.0%} padre" if e['kind'] == 'porta a' and 'parent_share' in e
                           else f"{e['weight']:.2f}")
    rows = [html.Tr([html.Td(in_label(e)), html.Td(f"{e['source']} · {_concept_name(by[e['source']])}"), html.Td(in_metric(e))])
            for e in into if e['source'] in by] + \
           [html.Tr([html.Td(out_label(e)), html.Td(f"{e['target']} · {_concept_name(by[e['target']])}"), html.Td(f"{e['weight']:.2f}")])
            for e in out if e['target'] in by]
    role_title = n.get('role', 'concetto').capitalize()
    return html.Div([html.P([html.B(f"{role_title} {n['id']}"),
                             f" (livello {n['layer']}, punto di vista {n['group']}, {n['mass']:.1%} della vista) · "
                             f"contesto riconosciuto: {', '.join(n['enters'])} · token predetti: {', '.join(n['follows'])}"],
                            className='small mb-1'),
                     dbc.Table(html.Tbody(rows), size='sm', bordered=True, className='small') if rows else
                     html.P('Nessun arco significativo sopra la media uniforme.', className='small text-muted')])


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


def step(number, title, *children, folding=False):
    """A step of the page. A folding step (preparing texts, creating a queue) closes by itself once a queue exists:
    it is needed rarely then, and its heading stays, with the button that opens it."""
    if not folding:
        return dbc.Card(dbc.CardBody([html.H4(f'{number} · {title}'), *children]), className='my-3', color='primary', outline=True)
    head = html.Div([html.H4(f'{number} · {title}', className='mb-0 me-3'),
                     dbc.Button('Apri', id=f'lm-step{number}-toggle', size='sm', outline=True, color='primary')],
                    className='d-flex align-items-center')
    return dbc.Card(dbc.CardBody([head, dbc.Collapse(html.Div(list(children), className='mt-2'), id=f'lm-step{number}-body',
                                                     is_open=True)]), className='my-3', color='primary', outline=True)


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
         html.Div(id='lm-file-upload'), html.Div(id='lm-upload-status', role='status', className='small'), folding=True),
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
             html.Small("Un passo è un pezzo di al massimo un milione di token nuovi. Alla fine, la rete rilegge tutto il testo finché la validazione migliora. "
                        "Si salva anche ogni 5 minuti e alla fine; «0» non tiene salvataggi permanenti.", className='text-muted d-block mb-2'),
             dbc.Checkbox(id='lm-long-growth', value=True, label='Il grafo di concetti cresce con i dati'),
             html.Small("Come nel modello tabulare, non c'è alcun numero di concetti da scegliere: «0» lascia che concetti, punti di vista "
                        "e livelli di sotto-concetti con più concetti padre siano scelti automaticamente dalla validazione, "
                        "mantenendo il modello decine di volte più leggero di un LLM (fino a 64 coordinate, senza matrici quadratiche).",
                        className='text-muted d-block mb-2'),
             dbc.Row([
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Larghezza iniziale'), dbc.Input(id='lm-long-start-dim', type='number', value=0, min=0, max=2048, step=8)]), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Strati iniziali'), dbc.Input(id='lm-long-start-layers', type='number', value=0, min=0, max=32, step=1)]), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Teste iniziali'), dbc.Input(id='lm-long-start-heads', type='number', value=0, min=0, max=64, step=1)]), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Concetti iniziali'), dbc.Input(id='lm-long-start-concepts', type='number', value=0, min=0, max=1024, step=1)]), md=3, className='mb-1'),
             ], className='mb-1'),
             dbc.Row([
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Larghezza max'), dbc.Input(id='lm-long-max-dim', type='number', value=64, min=0, max=2048, step=16)]), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Strati max'), dbc.Input(id='lm-long-max-layers', type='number', value=0, min=0, max=32, step=1)]), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Teste max'), dbc.Input(id='lm-long-max-heads', type='number', value=0, min=0, max=64, step=1)]), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Concetti/testa max'), dbc.Input(id='lm-long-max-concepts', type='number', value=0, min=0, max=1024, step=1)]), md=3, className='mb-1'),
             ], className='mb-1'),
             html.Small("Valori iniziali: la rete da cui parte il primo testo della coda (i successivi ripartono dalla rete appresa). "
                        "«0» = automatico: larghezza 32, 2 strati, 4 teste, 16 concetti per testa nel primo strato. La larghezza è un "
                        "multiplo di 8 divisibile per le teste; ogni strato dopo il primo parte con 4 volte i concetti del primo. "
                        "Da lì la rete cresce se vince le gare, fino ai massimi. "
                        "Tetti facoltativi: «0» = automatico dai dati senza tetto utente; un valore > 0 impone un massimale esplicito "
                        "(o fissa direttamente la piramide iniziale se si disattiva la crescita).",
                        className='text-muted d-block mb-2'),
         ], title='Impostazioni (facoltative: i valori proposti vanno bene)')], start_collapsed=True, className='my-2'),
         # Earlier controls kept for the callbacks they feed: a concept model has no size, no passes and no piece length to choose.
         html.Div([dcc.Dropdown(id='lm-long-model', value='micro', options=[dict(label='micro', value='micro')]),
                   dbc.Input(id='lm-long-passes', type='number', value=1), dbc.Input(id='lm-long-length', type='number', value=256)],
                  style={'display': 'none'}),
         dbc.Button('Crea la coda e avvia', id='lm-long-start', color='primary', className='mt-2'),
         html.Div(id='lm-long-start-status', role='status', className='small mt-1'), folding=True),
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
             html.Small("Si possono cambiare mentre la coda impara: valgono dal prossimo pezzo del testo in corso e per i testi "
                        "che restano. Una crescita spenta non partecipa più alle gare; i massimi «0» non hanno tetto.",
                        className='text-muted d-block mb-2'),
             dbc.Checkbox(id='lm-growth-on', value=True, label='Il grafo di concetti cresce con i dati', className='mb-1'),
             dbc.Checklist(id='lm-growth-kinds', inline=True, className='mb-2', value=['larghezza', 'strato', 'concetti', 'teste'],
                           options=[dict(label='larghezza', value='larghezza'), dict(label='strato in più', value='strato'),
                                    dict(label='concetti divisi', value='concetti'), dict(label='teste raddoppiate', value='teste')]),
             dbc.Row([
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Larghezza max'), dbc.Input(id='lm-growth-max-dim', type='number', min=0, max=2048, step=16)], size='sm'), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Strati max'), dbc.Input(id='lm-growth-max-layers', type='number', min=0, max=32, step=1)], size='sm'), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Teste max'), dbc.Input(id='lm-growth-max-heads', type='number', min=0, max=64, step=1)], size='sm'), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Concetti/testa max'), dbc.Input(id='lm-growth-max-concepts', type='number', min=0, max=1024, step=1)], size='sm'), md=3, className='mb-1'),
             ], className='mb-2'),
             dbc.Button('Applica alla coda', id='lm-growth-apply', size='sm', color='primary'),
             html.Div(id='lm-growth-status', role='status', className='small mt-1'), dcc.Store(id='lm-growth-queue'),
         ], title='Crescita della coda (si cambia mentre impara)')], start_collapsed=True, className='my-2'),
         dbc.Accordion([dbc.AccordionItem([
             dbc.Row([dbc.Col(dcc.Dropdown(id='lm-long-add-corpus', placeholder='Un testo già preparato'), md=8),
                      dbc.Col(dbc.Button('Aggiungi in fondo alla coda', id='lm-long-add', outline=True, color='primary'), md=4)], className='mb-2'),
             dbc.Checkbox(id='lm-long-add-growth', value=True, label='Continua a far crescere il grafo di concetti sul nuovo testo', className='small mb-1'),
             dbc.Row([
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Larghezza max'), dbc.Input(id='lm-long-add-max-dim', type='number', value=64, min=0, max=2048, step=16)], size='sm'), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Strati max'), dbc.Input(id='lm-long-add-max-layers', type='number', value=0, min=0, max=32, step=1)], size='sm'), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Teste max'), dbc.Input(id='lm-long-add-max-heads', type='number', value=0, min=0, max=64, step=1)], size='sm'), md=3, className='mb-1'),
                 dbc.Col(dbc.InputGroup([dbc.InputGroupText('Concetti/testa max'), dbc.Input(id='lm-long-add-max-concepts', type='number', value=0, min=0, max=1024, step=1)], size='sm'), md=3, className='mb-1'),
             ], className='mb-2'),
             html.Small(f'Oppure un documento dal computer ({FORMATS}): viene caricato, convertito in testo e aggiunto in fondo alla coda scelta. '
                        'Usa il vocabolario della coda, imparato all\'avvio («0» = automatico dai dati senza tetto utente).', className='text-muted d-block'),
             html.Div(id='lm-long-file-upload'), html.Div(id='lm-long-upload-status', role='status', className='small'),
             dcc.Store(id='lm-long-upload-state'),
         ], title='Aggiungi un testo alla coda')], start_collapsed=True, className='my-2'),
         html.H5('Il modello mentre impara', className='mt-3'),
         html.Div(id='lm-long-footprint'), html.Div(id='lm-long-graph'), dcc.Store(id='lm-long-graph-key'),
         html.H5('Cosa risponde', className='mt-3'),
         html.P("Le stesse domande a ogni salvataggio, con la stessa estrazione casuale: le differenze vengono solo da ciò che il modello ha "
                "imparato. Per fargli domande libere e scaricarlo: passo 4.", className='small'),
         html.Div(id='lm-long-answers'),
         html.Div(id='lm-answer-walk'),
         html.Div(dcc.Graph(id='lm-answer-graph', config={'displaylogo': False}), id='lm-answer-graph-box', style={'display': 'none'}),
         html.Div(id='lm-answer-walk-detail'),
         dcc.Store(id='lm-answer-paths', storage_type='session'), dcc.Store(id='lm-answer-walk-state', storage_type='session')),
    dcc.Store(id='lm-long-catalog'), dcc.Store(id='lm-long-builds'), dcc.Store(id='lm-steps-folded'), dcc.Store(id='lm-network-selected', storage_type='session'), dcc.Interval(id='lm-long-poll', interval=5000),
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
    waiting = [job for items in found.values() for job in items if job['status'] in ('paused', 'interrupted')]
    if waiting:                                                 # which queue is waiting, among the others
        job = max(waiting, key=lambda job: job.get('updated') or 0)
        label = (job['config'].get('series') or {}).get('label') or job['config'].get('name', '')
        where = f"passo {(job['progress'] or {}).get('step', 0):,} di {job['config'].get('total_steps') or 0:,}"
        return [html.B('Adesso: '), f"la coda «{label}» (#{job['id'][:4]}, {where}) è in pausa. Per continuare "
                "sceglila e premi «Riprendi» al passo 3."]
    if paused:
        return [html.B('Adesso: '), 'la preparazione dei dati è in pausa: premi «Riprendi» al passo 1.']
    if found and all(job['status'] == 'completed' for items in found.values() for job in items):
        return [html.B('Adesso: '), "la coda ha finito. Prova il modello e scaricalo al passo 4 (scegli l'ultimo testo), "
                "aggiungi altri testi al passo 3, oppure crea una coda nuova al passo 2."]
    if not corpora:
        return [html.B('Per cominciare: '), 'prepara una miscela pubblica o carica i tuoi documenti al passo 1.']
    return [html.B('Prossimo passo: '), 'scegli i testi al passo 2 e premi «Crea la coda e avvia».']


def register(app):
    @app.callback(Output('lm-step1-body', 'is_open'), Output('lm-step2-body', 'is_open'),
                  Output('lm-step1-toggle', 'children'), Output('lm-step2-toggle', 'children'), Output('lm-steps-folded', 'data'),
                  Input('lm-step1-toggle', 'n_clicks'), Input('lm-step2-toggle', 'n_clicks'), Input('lm-state', 'data'),
                  State('lm-step1-body', 'is_open'), State('lm-step2-body', 'is_open'), State('lm-steps-folded', 'data'))
    def fold_steps(first, second, state, open1, open2, decided):
        # Once, when the projects arrive: steps 1 and 2 close if a queue exists. Then only the buttons change them.
        trigger = dash.ctx.triggered_id
        if trigger == 'lm-step1-toggle':
            open1 = not open1
        elif trigger == 'lm-step2-toggle':
            open2 = not open2
        elif state and not decided:
            open1 = open2 = not (state.get('jobs') or [])
            decided = True
        else:
            return (dash.no_update,) * 5
        label = lambda shown: 'Chiudi' if shown else 'Apri'
        return open1, open2, label(open1), label(open2), decided

    @app.callback(Output('lm-network-graph', 'figure'), Output('lm-network-info', 'children'), Output('lm-network-selected', 'data'),
                  Input('lm-network-graph', 'clickData'), Input('lm-network-data', 'data'), Input('lm-network-head', 'value'),
                  State('lm-network-selected', 'data'), prevent_initial_call=True)
    def explore(click, graph, head, selected):
        # A click chooses a concept (a second click on it lets it go); switching head clears an out-of-view selection.
        if not graph:
            return dash.no_update, None, selected
        if dash.ctx.triggered_id == 'lm-network-head' and head and selected:
            by_id = {n['id']: n for n in graph.get('nodes') or ()}
            if selected in by_id and by_id[selected].get('group') != int(head):
                selected = None
        elif dash.ctx.triggered_id == 'lm-network-graph' and click:
            chosen = (click.get('points') or [{}])[0].get('customdata')
            if chosen:
                selected = None if chosen == selected else chosen
        return network_graph_figure(graph, selected, head=head), network_graph_info(graph, selected), selected

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
                  Output('lm-active-job', 'data', allow_duplicate=True),
                  Input('lm-long-start', 'n_clicks'), State('lm-workspace', 'data'), State('lm-long-corpora', 'value'),
                  State('lm-long-model', 'value'), State('lm-long-passes', 'value'), State('lm-long-save', 'value'),
                  State('lm-long-milestone', 'value'), State('lm-long-questions', 'value'), State('lm-state', 'data'),
                  State('lm-long-length', 'value'), State('lm-long-growth', 'value'),
                  State('lm-long-max-dim', 'value'), State('lm-long-max-layers', 'value'),
                  State('lm-long-max-heads', 'value'), State('lm-long-max-concepts', 'value'), State('lm-long-start-dim', 'value'),
                  State('lm-long-start-layers', 'value'), State('lm-long-start-heads', 'value'), State('lm-long-start-concepts', 'value'),
                  prevent_initial_call=True)
    def start(clicks, workspace, corpora, model, passes, save, milestone, questions, state, length=256, growth=False,
              max_dim=0, max_layers=0, max_heads=0, max_concepts=0, start_dim=0, start_layers=0, start_heads=0, start_concepts=0):
        try:
            if not corpora:
                raise ValueError('Scegli almeno un testo da imparare.')
            cuda = ((state or {}).get('capabilities') or {}).get('cuda', False)
            base = dict(profile(cuda), name=QUEUE_NAME, passes=1., save_steps=int(save or 10),
                        milestone_steps=int(milestone or 0), concept_network_growth=bool(growth),
                        concept_max_dim=max(0, int(max_dim or 0)),
                        concept_max_layers=max(0, int(max_layers or 0)),
                        concept_max_heads=max(0, int(max_heads or 0)),
                        concept_max_concepts=max(0, int(max_concepts or 0)),
                        concept_start_dim=max(0, int(start_dim or 0)),
                        concept_start_layers=max(0, int(start_layers or 0)),
                        concept_start_heads=max(0, int(start_heads or 0)),
                        concept_start_concepts=max(0, int(start_concepts or 0)),
                        probe_prompts=[line.strip() for line in (questions or '').splitlines() if line.strip()][:6])
            result = backend(workspace, '/queues', 'POST', dict(base=base, corpora=list(corpora)), timeout=120)
            first_job = (result.get('jobs') or [{}])[0].get('id', dash.no_update)
            return result.get('message', 'Coda creata.'), result['queue'], first_job
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update, dash.no_update

    @app.callback(Output('lm-long-queue-status', 'children'), Output('lm-active-job', 'data', allow_duplicate=True),
                  Input('lm-long-add', 'n_clicks'), Input('lm-long-pause', 'n_clicks'), Input('lm-long-resume', 'n_clicks'),
                  Input('lm-long-control', 'n_clicks'),
                  State('lm-workspace', 'data'), State('lm-long-queue', 'value'), State('lm-long-add-corpus', 'value'),
                  State('lm-state', 'data'), State('lm-long-add-growth', 'value'),
                  State('lm-long-add-max-dim', 'value'), State('lm-long-add-max-layers', 'value'),
                  State('lm-long-add-max-heads', 'value'), State('lm-long-add-max-concepts', 'value'),
                  prevent_initial_call=True)
    def queue_action(add, pause, resume, control, workspace, queue, corpus, state,
                     add_growth=None, max_dim=None, max_layers=None, max_heads=None, max_concepts=None):
        items = queues((state or {}).get('jobs', [])).get(queue)
        try:
            if not items:
                raise ValueError('Crea o scegli prima una coda.')
            if dash.ctx.triggered_id == 'lm-long-add':
                if not corpus:
                    raise ValueError('Scegli il testo da aggiungere.')
                payload = dict(corpus_id=corpus)
                if add_growth is not None:
                    payload['concept_network_growth'] = bool(add_growth)
                for key, val in (('concept_max_dim', max_dim), ('concept_max_layers', max_layers),
                                 ('concept_max_heads', max_heads), ('concept_max_concepts', max_concepts)):
                    if val is not None:
                        payload[key] = max(0, int(val or 0))
                job = backend(workspace, f'/queues/{queue}/add', 'POST', payload)
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

    @app.callback(Output('lm-growth-on', 'value'), Output('lm-growth-kinds', 'value'), Output('lm-growth-max-dim', 'value'),
                  Output('lm-growth-max-layers', 'value'), Output('lm-growth-max-heads', 'value'), Output('lm-growth-max-concepts', 'value'),
                  Output('lm-growth-queue', 'data'),
                  Input('lm-long-queue', 'value'), Input('lm-growth-status', 'children'), State('lm-state', 'data'),
                  State('lm-growth-queue', 'data'))
    def growth_options(queue, applied, state, filled):
        # The options as the queue has them, when another queue is chosen or after applying: not at every refresh of
        # the page, which would undo what is being edited.
        if queue == filled and dash.ctx.triggered_id == 'lm-long-queue':
            return (dash.no_update,) * 7
        items = queues((state or {}).get('jobs', [])).get(queue) or []
        if not items:
            return (dash.no_update,) * 7
        config = current_item(items)['config']
        if dash.ctx.triggered_id == 'lm-growth-status':          # just applied: the state of the page may be a refresh behind
            return (dash.no_update,) * 6 + (queue,)
        return (bool(config.get('concept_network_growth', True)),
                list(config.get('concept_growth_kinds') or ['larghezza', 'strato', 'concetti', 'teste']),
                *[int(config.get(key) or 0) for key in ('concept_max_dim', 'concept_max_layers', 'concept_max_heads', 'concept_max_concepts')],
                queue)

    @app.callback(Output('lm-growth-status', 'children'), Input('lm-growth-apply', 'n_clicks'),
                  State('lm-workspace', 'data'), State('lm-long-queue', 'value'), State('lm-growth-on', 'value'),
                  State('lm-growth-kinds', 'value'), State('lm-growth-max-dim', 'value'), State('lm-growth-max-layers', 'value'),
                  State('lm-growth-max-heads', 'value'), State('lm-growth-max-concepts', 'value'), prevent_initial_call=True)
    def apply_growth(clicks, workspace, queue, grows, kinds, max_dim, max_layers, max_heads, max_concepts):
        try:
            if not queue:
                raise ValueError('Scegli prima una coda.')
            payload = dict(concept_network_growth=bool(grows), concept_growth_kinds=list(kinds or []),
                           concept_max_dim=max(0, int(max_dim or 0)), concept_max_layers=max(0, int(max_layers or 0)),
                           concept_max_heads=max(0, int(max_heads or 0)), concept_max_concepts=max(0, int(max_concepts or 0)))
            result = backend(workspace, f'/queues/{queue}/growth', 'POST', payload)
            texts = '1 testo' if result['changed'] == 1 else f"{result['changed']} testi"
            return (f"Applicato a {texts} della coda: vale dal prossimo pezzo." if result['changed']
                    else 'La coda ha già finito tutti i testi: niente da cambiare.')
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning', className='small py-2')

    @app.callback(Output('lm-long-queue-status', 'children', allow_duplicate=True), Input('lm-long-upload-state', 'data'),
                  State('lm-workspace', 'data'), State('lm-long-queue', 'value'), State('lm-state', 'data'),
                  State('lm-long-add-growth', 'value'),
                  State('lm-long-add-max-dim', 'value'), State('lm-long-add-max-layers', 'value'),
                  State('lm-long-add-max-heads', 'value'), State('lm-long-add-max-concepts', 'value'),
                  prevent_initial_call=True)
    def add_uploaded(uploaded, workspace, queue, state,
                     add_growth=None, max_dim=None, max_layers=None, max_heads=None, max_concepts=None):
        """A document just uploaded from the queue section goes to the end of the chosen queue."""
        try:
            if not uploaded or not uploaded.get('id'):
                raise ValueError('Caricamento non riuscito.')
            if not queues((state or {}).get('jobs', [])).get(queue):
                raise ValueError(f"«{uploaded.get('name', 'Il documento')}» è caricato: scegli una coda e aggiungilo dal menu qui sopra.")
            payload = dict(corpus_id=uploaded['id'])
            if add_growth is not None:
                payload['concept_network_growth'] = bool(add_growth)
            for key, val in (('concept_max_dim', max_dim), ('concept_max_layers', max_layers),
                             ('concept_max_heads', max_heads), ('concept_max_concepts', max_concepts)):
                if val is not None:
                    payload[key] = max(0, int(val or 0))
            job = backend(workspace, f'/queues/{queue}/add', 'POST', payload)
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
        """Step 4 starts on the active or last saved text of the chosen queue."""
        items = queues((state or {}).get('jobs', [])).get(queue) or []
        if not items or (job and any(item['id'] == job for item in items)):
            return dash.no_update
        active = next((item for item in items if item.get('status') in ('running', 'preparing')), None)
        if active is not None:
            return active['id']
        saved = [item for item in items if item['checkpoint']]
        return (saved[-1] if saved else current_item(items))['id']

    def shown_item(items, workspace):
        # The model of the queue to show: the one learning, or the last learned while the next prepares its tokens.
        current = current_item(items)
        learned = [job for job in items if job['checkpoint'] and (job['progress'] or {}).get('graft')]
        if not (current['progress'] or {}).get('graft') and learned:
            return current, learned[-1], backend(workspace, f"/jobs/{learned[-1]['id']}/footprint?graph=1")
        return current, None, backend(workspace, f"/jobs/{current['id']}/footprint?graph=1")

    @app.callback(Output('lm-long-graph', 'children'), Output('lm-long-graph-key', 'data'),
                  Input('lm-state', 'data'), Input('lm-long-queue', 'value'),
                  State('lm-workspace', 'data'), State('lm-long-graph-key', 'data'))
    def network_graph(state, queue, workspace, shown):
        # The graph of concepts weighs megabytes: drawn again only when the model it describes changes.
        items = [job for job in queues((state or {}).get('jobs', [])).get(queue) or [] if job['status'] != 'created']
        if not items:
            return None, None
        current = current_item(items)
        progress = current['progress'] or {}
        key = [current['id'], current['status'], current.get('checkpoint'), progress.get('step'), bool(progress.get('graft'))]
        if key == shown:
            return dash.no_update, dash.no_update
        try:
            return graph_section(shown_item(items, workspace)[2]), key
        except ValueError as exc:
            return html.P(str(exc), className='small text-muted'), None

    @app.callback(Output('lm-long-footprint', 'children'), Input('lm-state', 'data'), Input('lm-long-queue', 'value'),
                  State('lm-workspace', 'data'))
    def footprint(state, queue, workspace):
        items = [job for job in queues((state or {}).get('jobs', [])).get(queue) or [] if job['status'] != 'created']
        if not items:
            return html.P('Compaiono quando il primo testo comincia a imparare.', className='small text-muted')
        current = current_item(items)
        try:
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
                                         answers_view(found, job['id'])])
                    return answers_view(found, job['id'])
            return answers_view([])
        except ValueError as exc:
            return html.P(str(exc), className='small text-muted')

    @app.callback(Output('lm-answer-paths', 'data'), Output('lm-answer-walk-state', 'data'),
                  Input({'type': 'lm-answer-open', 'index': dash.ALL}, 'n_clicks'), State('lm-workspace', 'data'),
                  prevent_initial_call=True)
    def open_answer_walk(clicks, workspace):
        if not isinstance(dash.ctx.triggered_id, dict) or not dash.ctx.triggered[0].get('value'):
            return dash.no_update, dash.no_update             # the table redrawn, not a click
        job, index = dash.ctx.triggered_id['index'].rsplit('|', 1)
        try:
            data = backend(workspace, f'/jobs/{job}/answer-paths') or {}
        except ValueError:
            data = {}
        return dict(data, job=job), dict(question=int(index), trail=[])

    @app.callback(Output('lm-answer-walk-state', 'data', allow_duplicate=True),
                  Input({'type': 'lm-walk-go', 'index': dash.ALL}, 'n_clicks'),
                  Input({'type': 'lm-walk-back', 'index': dash.ALL}, 'n_clicks'), Input('lm-answer-graph', 'clickData'),
                  State('lm-answer-walk-state', 'data'), prevent_initial_call=True)
    def walk_answer(go, back, click, walk):
        if not walk:
            return dash.no_update
        trail = list(walk.get('trail') or [])
        if dash.ctx.triggered_id == 'lm-answer-graph':      # a node of the graph: opened below it, on the same token
            key = ((click or {}).get('points') or [{}])[0].get('customdata')
            if not key or not trail or trail[-1] == key:
                return dash.no_update
            return dict(walk, trail=trail + [key]) if not key.startswith('token:') else dash.no_update
        if not isinstance(dash.ctx.triggered_id, dict) or not dash.ctx.triggered[0].get('value'):
            return dash.no_update
        if dash.ctx.triggered_id['type'] == 'lm-walk-back':
            return dict(walk, trail=trail[:dash.ctx.triggered_id['index'] + 1])
        key = dash.ctx.triggered_id['index']
        if key == 'close':
            return None
        if key.startswith('token:'):                         # a token of the answer: the walk starts again from it
            return dict(walk, trail=[key])
        return dict(walk, trail=trail + [key]) if not trail or trail[-1] != key else dash.no_update

    @app.callback(Output('lm-answer-walk', 'children'), Output('lm-answer-graph', 'figure'), Output('lm-answer-graph-box', 'style'),
                  Output('lm-answer-walk-detail', 'children'), Input('lm-answer-paths', 'data'), Input('lm-answer-walk-state', 'data'))
    def show_answer_walk(data, walk):
        try:
            head, figure, detail = answer_walk_view(data, walk)
        except (KeyError, IndexError, ValueError, TypeError, StopIteration):
            head, figure, detail = None, None, None
        hidden = {'display': 'none'}
        return head, figure if figure is not None else {}, hidden if figure is None else {}, detail
