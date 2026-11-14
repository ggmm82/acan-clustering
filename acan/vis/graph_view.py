"""Bounded graph windows; rendering never changes the trained model."""
import copy
import math
import numpy as np

PAGE_SIZE = 12


def group_indices(node, include_empty=False):
    if include_empty:
        return list(range(node.k()))
    return [i for i, count in enumerate(np.asarray(node.Counts).reshape(-1)) if count >= 1]


def resolve_focus(graph, focus):
    try:
        focus = int(focus)
    except (TypeError, ValueError):
        focus = graph.rootId
    return focus if focus in graph.clusterModel else graph.rootId


def navigation(graph, focus=None, page=1, include_empty=False, action=None):
    focus = resolve_focus(graph, focus)
    parents = {child:(node.id, group) for node in graph.clusterModel.values() for group,child in node.children.items()}
    if action == 'root':
        focus = graph.rootId
    elif action == 'up' and focus in parents:
        focus = parents[focus][0]
    options = []
    def visit(node_id, path):
        node = graph.clusterModel[node_id]
        options.append(dict(label=f'{path} · {node.n} righe', value=node_id))
        for group,child in sorted(node.children.items()):
            visit(child, f'{path} › G{group} › {child}')
    visit(graph.rootId, f'Radice {graph.rootId}')
    node = graph.clusterModel[focus]
    groups = group_indices(node, include_empty)
    pages = max(1,math.ceil(len(groups)/PAGE_SIZE))
    page = min(max(int(page or 1),1),pages)
    page_options = [dict(label=f'Pagina {i} di {pages}',value=i) for i in range(1,pages+1)]
    active = len(group_indices(node))
    shown = len(groups[(page-1)*PAGE_SIZE:page*PAGE_SIZE])
    path = next(option['label'] for option in options if option['value']==focus)
    summary = level_summary(node)+(f'Albero: {len(graph.clusterModel)} nodi, {sum(n.k() for n in graph.clusterModel.values())} prototipi locali complessivi (non classi globali). {path}. {active} gruppi con record assegnati, {node.k()-active} prototipi senza assegnazioni hard. '
               f'Pagina {page}/{pages}: {shown} gruppi mostrati su {len(groups)}. '
               'I prototipi senza assegnazioni hard possono comunque contribuire alle previsioni soft.')
    return options,focus,page_options,page,summary


def elements(graph, focus=None, page=1, include_empty=False):
    from .render import to_elements
    focus = resolve_focus(graph,focus)
    node = graph.clusterModel[focus]
    _,_,_,page,_ = navigation(graph,focus,page,include_empty)
    indices = group_indices(node,include_empty)[(page-1)*PAGE_SIZE:page*PAGE_SIZE]
    # Reuse native IDs, colors and inspection metadata for this one concept only.
    local = copy.copy(graph)
    local.rootId = focus
    local.clusterModel = {focus:node}
    all_elements = to_elements(local)
    root = all_elements[str(focus)]
    root['data']['label'] = f'Concetto {focus}\n{node.n} righe · {node.k()} prototipi'
    root['classes'] = 'local-root'
    root['position'] = dict(x=175,y=20)
    result = [root]
    for position,index in enumerate(indices):
        group_id = f'{focus}_G{index}'
        group = all_elements[group_id]
        count = int(np.asarray(node.Counts).reshape(-1)[index])
        labels = node.fullLabels(graph.labels)
        detail = ''
        if labels:
            column = max(labels,key=lambda col:node.Groups[index,node.cols.index(col)])
            detail = column
            for target in graph.labels:
                if column.startswith(str(target)+'_'):
                    detail = f'{target}: {column[len(str(target))+1:]}'
                    break
        group['data']['label'] = f'G{index} · {count} righe' + (f'\n{detail}' if detail else '')
        group['classes'] = 'local-group'
        if index in node.children:
            child = node.children[index]
            group['data']['child_id'] = child
            group['data']['label'] += f'\nApri concetto {child}'
            group['classes'] += ' has-child'
        group['position'] = dict(x=(position%3)*175,y=140+(position//3)*115)
        result.extend([group,all_elements[f'N{focus}G{index}']])
    result = with_levels(result,node,focus,indices)
    # Edges of the concept graph: concepts between which some observation hesitates.
    shown = set(indices)
    links = [edge for edge in (getattr(node,'routing_parameters',None) or {}).get('edges') or []
             if edge['first'] in shown and edge['second'] in shown]
    strongest = max((edge['weight'] for edge in links),default=1.) or 1.
    for edge in links:
        result.append(dict(data=dict(id=f"E{focus}_{edge['first']}_{edge['second']}",source=f"{focus}_G{edge['first']}",
                                     target=f"{focus}_G{edge['second']}",strength=1+4*edge['weight']/strongest),
                           classes='concept-edge'))
    return result


LEVEL_NODES = 12                                                 # a level larger than this is summarized, not drawn


def level_summary(node):
    """One line per level of the concept graph: root, general concepts, particular behaviours."""
    levels = (getattr(node,'routing_parameters',None) or {}).get('levels') or []
    if not levels:
        return ''
    parts = ['radice']+[f"livello {level['depth']}: {level['concepts']} concetti (CP a coppie {level['pair_criterion']:.3f})" for level in levels]
    return 'Grafo dei concetti: '+' → '.join(parts)+'. '


def with_levels(result, node, focus, indices):
    """Draw the levels above the working level: the root, the general concepts and the dense links between levels.

    A link joins a concept to every concept of the level above it descends from (weight above 10%),
    so a concept reached from several general concepts is one node with several parents. Levels too
    large to draw are skipped: the links then go through them (product of the matrices).
    """
    levels = (getattr(node,'routing_parameters',None) or {}).get('levels') or []
    upper = []
    for level in levels[:-1]:
        if level['concepts'] > LEVEL_NODES:
            break
        upper.append(level)
    if not upper:
        return result
    step, shift = 110, 110*len(upper)
    # Cytoscape edge endpoints are immutable and the React diff keeps an element whose id is
    # unchanged: when another model arrives in the same page, an edge with the same id would
    # not be drawn again once its nodes are replaced. The ids carry a fingerprint of the model.
    import json, zlib
    stamp = format(zlib.crc32(json.dumps([level.get('parents') for level in levels]).encode()), 'x')
    for element in result:                                       # make room under the root
        if element.get('classes','').startswith('local-group'):
            element['position'] = dict(x=element['position']['x'],y=element['position']['y']+shift)
    result = [element for element in result if not str(element['data'].get('id','')).startswith(f'N{focus}G')]
    identifier = lambda depth,index: f'{focus}_L{depth}_{index}'
    previous = None
    for row,level in enumerate(upper):
        for index in range(level['concepts']):
            reads = level['reads'][index] if level.get('reads') else ''
            label = f"L{level['depth']}·{index} · {100*level['mass'][index]:.0f}%"+(f'\n{reads}' if reads else '')
            result.append(dict(data=dict(id=identifier(level['depth'],index),label=label,level=level['depth']),
                               classes='level-concept',position=dict(x=175+(index-(level['concepts']-1)/2)*128,y=20+step*(row+1))))
            if previous is None:
                result.append(dict(data=dict(id=f'R{focus}_{stamp}_{index}',source=str(focus),target=identifier(level['depth'],index),strength=2),
                                   classes='level-edge'))
        if previous is not None:
            result.extend(level_links(level['parents'],lambda i,d=previous['depth']: identifier(d,i),
                                      lambda j,d=level['depth']: identifier(d,j),range(level['concepts']),f'P{focus}_{stamp}_{level["depth"]}'))
        previous = level
    descent = np.eye(previous['concepts'])                       # from the last drawn level down to the working level
    for level in levels[len(upper):]:
        descent = descent@np.asarray(level['parents'])
    result.extend(level_links(descent,lambda i,d=previous['depth']: identifier(d,i),
                              lambda j: f'{focus}_G{j}',indices,f'P{focus}_{stamp}_G'))
    return result


def level_links(parents, source, target, children, prefix):
    links = []
    for child in children:
        for parent,row in enumerate(parents):
            if child < len(row) and row[child] > .1:
                links.append(dict(data=dict(id=f'{prefix}_{parent}_{child}',source=source(parent),target=target(child),
                                            strength=1+3*float(row[child])),classes='level-edge'))
    return links


# ---- the whole model, readable ---------------------------------------------------
def _prototype(graph, node, groups, relevance, index, top=8):
    """Typical value and relevance of the most relevant attributes of one concept, in real units."""
    import pandas as pd
    values = graph.denormalizeData(pd.DataFrame(np.asarray(groups)[[index]], columns=node.cols)).iloc[0]
    weight = np.asarray(relevance)[index]
    order = np.argsort(-weight)[:top]
    return [dict(attribute=str(node.cols[j]), value=round(float(values.iloc[j]), 4), relevance=round(float(weight[j]), 3))
            for j in order]


def model_structure(graph):
    """Everything a person needs to read the concept graph, without the internal matrices."""
    node = graph.clusterModel[graph.rootId]
    state = getattr(node, 'routing_parameters', None) or {}
    counts = np.asarray(node.Counts).reshape(-1)
    levels = state.get('levels') or []
    out = dict(profile=getattr(node, 'routing_profile', 'legacy-v8'), target=list(graph.labels), rows=int(node.n),
               attributes=[str(c) for c in node.cols], selection=state.get('selection'),
               training_score=state.get('training_score'), levels=[], groups=[], edges=state.get('edges') or [])
    for level in levels[:-1]:
        concepts = []
        for i in range(level['concepts']):
            item = dict(name=f"L{level['depth']}·{i}", share=round(float(level['mass'][i]), 4),
                        reads=level['reads'][i] if level.get('reads') else None)
            if level.get('parents') is not None:
                item['parents'] = {f"L{level['depth']-1}·{p}": round(float(row[i]), 3)
                                   for p, row in enumerate(level['parents']) if row[i] > .01}
            if level.get('groups') is not None:
                item['prototype'] = _prototype(graph, node, level['groups'], level['relevance'], i)
            concepts.append(item)
        out['levels'].append(dict(depth=level['depth'], concepts=concepts, pair_criterion=level['pair_criterion']))
    last = levels[-1] if levels else None
    for k in range(node.k()):
        item = dict(name=f'G{k}', rows=int(counts[k]) if k < len(counts) else 0)
        if last is not None:
            item['share'] = round(float(last['mass'][k]), 4)
            if last.get('reads'):
                item['reads'] = last['reads'][k]
            if last.get('parents') is not None:
                item['parents'] = {f"L{last['depth']-1}·{p}": round(float(row[k]), 3)
                                   for p, row in enumerate(last['parents']) if row[k] > .01}
        item['prototype'] = _prototype(graph, node, node.Groups, node.Relevance, k)
        out['groups'].append(item)
    return out


def model_overview(structure):
    """The same structure as a page: summary, levels, groups, edges."""
    from dash import html
    import dash_bootstrap_components as dbc

    def table(rows, columns):
        return dbc.Table([html.Thead(html.Tr([html.Th(c) for c in columns])),
                          html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])],
                         size='sm', bordered=True, responsive=True, className='small')

    def traits(item):
        return '; '.join(f"{p['attribute']} {p['value']:g} ({100*p['relevance']:.0f}%)" for p in item.get('prototype', [])[:4])

    def parents(item):
        return ', '.join(f'{name} {100*w:.0f}%' for name, w in (item.get('parents') or {}).items())

    selection = structure.get('selection') or {}
    parts = [html.P([html.B('Profilo: '), structure['profile'], ' · ', html.B('righe: '), str(structure['rows']), ' · ',
                     html.B('attributi: '), str(len(structure['attributes'])), ' · ', html.B('target: '),
                     ', '.join(structure['target']) or 'nessuno (scoperta senza target)']),
             html.P(['Numero di concetti scelto dal giudizio su righe non viste: ',
                     html.B(str(selection.get('chosen_k', '—'))),
                     f" (il migliore era {selection.get('best_k', '—')}; si tiene il più generale entro un errore standard)."],
                    className='small')]
    for level in structure['levels']:
        parts.append(html.H6(f"Livello {level['depth']} · {len(level['concepts'])} concetti · CP a coppie {level['pair_criterion']:.3f}", className='mt-3'))
        parts.append(table([[c['name'], f"{100*c['share']:.0f}%", c.get('reads') or '', parents(c), traits(c)] for c in level['concepts']],
                           ['Concetto', 'Righe', 'Classe', 'Discende da', 'Attributi più rilevanti (valore, rilevanza)']))
    parts.append(html.H6(f"Gruppi che fanno le previsioni · {len(structure['groups'])}", className='mt-3'))
    parts.append(table([[g['name'], g['rows'], g.get('reads') or '', parents(g), traits(g)] for g in structure['groups']],
                       ['Gruppo', 'Righe', 'Classe', 'Discende da', 'Attributi più rilevanti (valore, rilevanza)']))
    if structure['edges']:
        strongest = sorted(structure['edges'], key=lambda e: -e['weight'])[:15]
        parts.append(html.H6('Confini incerti fra gruppi (i più forti)', className='mt-3'))
        parts.append(table([[f"G{e['first']}", f"G{e['second']}", f"{100*e['weight']:.2f}%"] for e in strongest],
                           ['Gruppo', 'Gruppo', 'Righe che esitano']))
    return parts


def legend():
    """What every mark of the graph means."""
    from dash import html
    swatch = lambda style: html.Span(style={**dict(display='inline-block', width='26px', height='14px', marginRight='6px',
                                                   verticalAlign='middle'), **style})
    items = [
        (swatch(dict(background='#ef0000', borderRadius='3px')), 'Radice: tutta la tabella.'),
        (swatch(dict(background='#e0e7ff', border='2px solid #4338ca', borderRadius='3px')),
         'Concetto di un livello (L1 i concetti generali, L2… i più particolari): percentuale di righe e classe prevalente. Un clic mostra il suo prototipo.'),
        (swatch(dict(background='#f472b6', borderRadius='3px')),
         'Gruppo che fa le previsioni (G): righe e classe prevalente; il colore segue la classe. Un clic mostra il prototipo, un doppio clic lo approfondisce a mano.'),
        (swatch(dict(border='3px solid #2563eb', borderRadius='3px')), 'Bordo blu: il gruppo ha già un approfondimento; doppio clic per aprirlo.'),
        (swatch(dict(borderTop='3px solid #6366f1', height='0', marginTop='6px')),
         'Freccia blu: «discende da». Più è spessa, più righe del concetto sotto stanno nel concetto sopra. Più frecce in entrata: un comportamento comune a più concetti.'),
        (swatch(dict(borderTop='3px dashed #d97706', height='0', marginTop='6px')),
         'Linea tratteggiata arancione: confine incerto, righe che esitano fra due gruppi. Più è spessa, più righe.'),
    ]
    return html.Div([html.Div([mark, html.Span(text)], className='small mb-1') for mark, text in items],
                    className='border rounded p-2 mt-2', id='graph-legend')
