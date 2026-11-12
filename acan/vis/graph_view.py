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
    summary = (f'Albero: {len(graph.clusterModel)} nodi, {sum(n.k() for n in graph.clusterModel.values())} prototipi locali complessivi (non classi globali). {path}. {active} gruppi con record assegnati, {node.k()-active} prototipi senza assegnazioni hard. '
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
    return result
