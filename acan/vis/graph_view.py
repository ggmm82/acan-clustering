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


LEVEL_NODES = 24                                                 # a level larger than this is summarized, not drawn


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
    so a concept reached from several general concepts is one node with several parents.
    """
    levels = (getattr(node,'routing_parameters',None) or {}).get('levels') or []
    upper = [level for level in levels[:-1] if level['concepts'] <= LEVEL_NODES]
    if not upper or len(upper) != len(levels)-1:
        return result
    step, shift = 110, 110*len(upper)
    for element in result:                                       # make room under the root
        if element.get('classes','').startswith('local-group'):
            element['position'] = dict(x=element['position']['x'],y=element['position']['y']+shift)
    result = [element for element in result if not str(element['data'].get('id','')).startswith(f'N{focus}G')]
    identifier = lambda depth,index: f'{focus}_L{depth}_{index}'
    previous = None
    for row,level in enumerate(upper):
        width = max(level['concepts']-1,1)
        for index in range(level['concepts']):
            reads = level['reads'][index] if level.get('reads') else ''
            label = f"L{level['depth']}·{index} · {100*level['mass'][index]:.0f}%"+(f'\n{reads}' if reads else '')
            result.append(dict(data=dict(id=identifier(level['depth'],index),label=label,level=level['depth']),
                               classes='level-concept',position=dict(x=175+(index-(level['concepts']-1)/2)*min(175,700/width),y=20+step*(row+1))))
            if previous is None:
                result.append(dict(data=dict(id=f'R{focus}_{index}',source=str(focus),target=identifier(level['depth'],index),strength=2),
                                   classes='level-edge'))
        if previous is not None:
            result.extend(level_links(level['parents'],lambda i,d=previous['depth']: identifier(d,i),
                                      lambda j,d=level['depth']: identifier(d,j),range(level['concepts']),f'P{focus}_{level["depth"]}'))
        previous = level
    result.extend(level_links(levels[-1]['parents'],lambda i,d=previous['depth']: identifier(d,i),
                              lambda j: f'{focus}_G{j}',indices,f'P{focus}_G'))
    return result


def level_links(parents, source, target, children, prefix):
    links = []
    for child in children:
        for parent,row in enumerate(parents):
            if child < len(row) and row[child] > .1:
                links.append(dict(data=dict(id=f'{prefix}_{parent}_{child}',source=source(parent),target=target(child),
                                            strength=1+3*row[child]),classes='level-edge'))
    return links
