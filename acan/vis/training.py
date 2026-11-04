"""UI orchestration of native ACAN APIs; no alternative learning objective."""
import multiprocessing as mp
import threading
import time

from acan import AlgebraicClusterLearner, GraphClusterModel


def train_native(data, labels, options):
    normalized, minmax = GraphClusterModel.normalizeData(GraphClusterModel.full_dummies(data))
    graph = GraphClusterModel()
    graph.minMax, graph.labels = minmax, list(labels)
    common = {k: options[k] for k in ('lr', 'tau', 'alpha', 'device')}
    if options['max_depth'] == 1:
        learner = AlgebraicClusterLearner(normalized, labels=labels,
                                         batch_size=options['batch_size'], **common)
        root = learner.nLearn(max_epochs=options['max_epochs'])
        root.history = learner.history
        graph.clusterModel[root.id] = root
    else:
        graph, root = AlgebraicClusterLearner.gLearn(
            normalized, labels=labels, gcm=graph,
            max_depth=options['max_depth'], max_epochs=options['max_epochs'],
            min_samples_split=options['min_samples_split'],
            similarity_threshold=options['similarity_threshold'], **common)
    graph.rootId = root.id
    return graph


def _worker(sender, data, labels, options):
    try:
        sender.send(('ok', train_native(data, labels, options).to_json()))
    except Exception as exc:
        sender.send(('error', str(exc)))
    finally:
        sender.close()


class NativeTraining:
    """An owned subprocess makes native recursive training cancellable per tab."""
    iteration = 'training nativo in corso'

    def __init__(self, data, labels, options):
        self.data, self.labels, self.options = data, labels, options
        self._stop = threading.Event()
        self.graph = None

    def request_stop(self):
        self._stop.set()

    def nLearn(self, **unused):
        context = mp.get_context('spawn')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(sender, self.data, self.labels, self.options), daemon=True)
        started = False
        try:
            if self._stop.is_set():
                raise ValueError('Training annullato; modello precedente conservato.')
            process.start()
            started = True
            sender.close()
            deadline = time.monotonic() + 900
            while not receiver.poll(0.1):
                if self._stop.is_set():
                    raise ValueError('Training annullato; modello precedente conservato.')
                if time.monotonic() > deadline:
                    raise ValueError('Limite di training di 15 minuti raggiunto.')
                if not process.is_alive():
                    raise ValueError('Processo di training terminato senza risultato.')
            status, payload = receiver.recv()
            if self._stop.is_set():
                raise ValueError('Training annullato; modello precedente conservato.')
            if status != 'ok':
                raise ValueError(payload)
            self.graph = GraphClusterModel.loadModel(payload)
        finally:
            sender.close()
            receiver.close()
            if started:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()


def graph_stats(graph):
    def depth(node_id):
        return 1 + max((depth(child) for child in graph.clusterModel[node_id].children.values()), default=0)
    return dict(nodes=len(graph.clusterModel), depth=depth(graph.rootId),
                groups=sum(node.k() for node in graph.clusterModel.values()))
