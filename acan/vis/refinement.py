"""Requested tree edits built from the native split and single-node learner APIs."""
import hashlib
from acan import AlgebraicClusterLearner, GraphClusterModel


def root_signature(model_json):
    graph = GraphClusterModel.loadModel(model_json)
    return hashlib.sha256(graph.clusterModel[graph.rootId].to_json().encode()).hexdigest()


def parse_group(node_id):
    if not isinstance(node_id, str) or '_G' not in node_id:
        raise ValueError('Doppio clic su un gruppo (non sulla radice) per approfondirlo.')
    node, group = node_id.split('_G')
    return int(node), int(group)


def training_subset(graph, training, node_id):
    encoded = GraphClusterModel.full_dummies(training)
    cols = graph.clusterModel[graph.rootId].cols
    normalized = graph.alignData(encoded, cols).reindex(columns=cols, fill_value=0)
    def find(current, data):
        if current == node_id:
            return data
        node = graph.clusterModel[current]
        for group, child in node.children.items():
            result = find(child, node.splitDataset(data)[group][0])
            if result is not None:
                return result
        return None
    result = find(graph.rootId, normalized)
    if result is None:
        raise ValueError('Nodo non raggiungibile dalla radice.')
    return result


def refine(model_json, training, node_id, options):
    graph = GraphClusterModel.loadModel(model_json)
    parent_id, group = parse_group(node_id)
    parent = graph.clusterModel[parent_id]
    if (getattr(parent,'routing_profile','legacy-v8')=='cp-joint-v1' and
            parent.routing_parameters.get('cp_definition')!='exclusive-cp-v1'):
        raise ValueError('Modello CP precedente: riaddestrare prima di approfondire con il nuovo nucleo, evitando di mescolare formule diverse.')
    if group in parent.children:
        raise ValueError('Il gruppo ha già un approfondimento: scegli un suo gruppo figlio.')
    if len(graph.clusterModel) >= 64:
        raise ValueError('Limite interattivo di 64 nodi raggiunto; rimuovere un ramo prima di continuare.')
    parent_data = training_subset(graph, training, parent_id)
    subset = parent.splitDataset(parent_data)[group][0]
    if len(subset) < options['min_samples_split'] or len(subset) >= len(parent_data):
        raise ValueError('Campioni insufficienti o gruppo identico al padre: nessuno split utile.')
    # Reuse global normalization: renormalizing the child would change its metric.
    _, child = AlgebraicClusterLearner.gLearn(
        subset, labels=graph.labels, gcm=GraphClusterModel(), max_depth=1,
        max_epochs=options['max_epochs'], lr=options['lr'], tau=options['tau'],
        alpha=options['alpha'], device=options['device'],
        routing_profile=getattr(parent,'routing_profile','legacy-v8'))
    child.id = max(graph.clusterModel) + 1
    graph.clusterModel[child.id] = child
    parent.children[group] = child.id
    return graph


def remove_branch(model_json, selected_id):
    graph = GraphClusterModel.loadModel(model_json)
    if '_G' in str(selected_id):
        node_id, group = parse_group(selected_id)
        parent = graph.clusterModel[node_id]
        if group in parent.children:
            child = parent.children.pop(group)
        elif node_id != graph.rootId:
            child = node_id
            for ancestor in graph.clusterModel.values():
                for index, target in list(ancestor.children.items()):
                    if target == child:
                        del ancestor.children[index]
        else:
            raise ValueError('Questo gruppo non ha un approfondimento da rimuovere.')
    else:
        child = int(selected_id)
        if child == graph.rootId:
            raise ValueError('La radice non può essere rimossa.')
        for parent in graph.clusterModel.values():
            for group, target in list(parent.children.items()):
                if target == child:
                    del parent.children[group]
    def discard(node_id):
        node = graph.clusterModel.pop(node_id)
        for target in node.children.values():
            discard(target)
    discard(child)
    return graph
