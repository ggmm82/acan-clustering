"""UI orchestration of native ACAN APIs; no alternative learning objective."""
import multiprocessing as mp
import threading
import time

from acan import AlgebraicClusterLearner, GraphClusterModel


def train_native(data, labels, options, report=None):
    normalized, minmax = GraphClusterModel.normalizeData(GraphClusterModel.full_dummies(data))
    graph = GraphClusterModel()
    graph.minMax, graph.labels = minmax, list(labels)
    from acan.preprocessing import fit_features
    graph.preprocessing = fit_features(data, labels)
    common = {k: options[k] for k in ('lr', 'tau', 'alpha', 'device')}
    common['routing_profile'] = options.get('routing_profile','legacy-v8')
    if options['max_depth'] == 1 or common['routing_profile'] in ('cp-v9', 'concept-graph-v1'):
        learner = AlgebraicClusterLearner(normalized, labels=labels,
                                         batch_size=options['batch_size'], **common)
        # One-hot blocks are single categorical attributes for the concept core.
        from pandas.api.types import is_numeric_dtype, is_bool_dtype
        learner.categorical_names = [str(name) for name in data.columns if name not in labels
                                     and not is_numeric_dtype(data[name]) and not is_bool_dtype(data[name])]
        learner.report = report                                  # phases and quantities while learning, if followed
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


def _worker(sender, data, labels, options, base_model=None, branch=None):
    try:
        if branch is not None:
            from .refinement import refine
            graph = refine(base_model, data, branch, options)
        else:
            started = time.monotonic()

            def report(facts):
                """What the model is doing now, sent to the page: the phase, its quantities, time and memory."""
                import resource
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                score = facts.get('score')
                sender.send(('progress', dict(facts, score=None if score is None else float(score),
                                              seconds=time.monotonic()-started, memory_mb=peak)))
            graph = train_native(data, labels, options, report)
        sender.send(('ok', graph.to_json()))
    except Exception as exc:
        sender.send(('error', str(exc)))
    finally:
        sender.close()


class NativeTraining:
    """An owned subprocess makes native recursive training cancellable per tab."""
    iteration = 'training nativo in corso'

    def __init__(self, data, labels, options, base_model=None, branch=None):
        self.data, self.labels, self.options = data, labels, options
        self.base_model, self.branch = base_model, branch
        self._stop = threading.Event()
        self.graph = None
        self.status = None                                       # the last news from the training process
        self.rows = len(data)

    def request_stop(self):
        self._stop.set()

    def nLearn(self, **unused):
        payload = run_isolated(_worker, (self.data, self.labels, self.options, self.base_model, self.branch), self._stop,
                               on_progress=lambda facts: setattr(self, 'status', facts))
        self.graph = GraphClusterModel.loadModel(payload)


def run_isolated(target, arguments, stop, timeout=900, on_progress=None):
    """Run ``target`` in its own process. ``timeout`` seconds without news from it end the run:
    a process that reports its progress may take as long as it needs, one that is silent may not."""
    context = mp.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *arguments), daemon=True)
    started = False
    try:
        if stop.is_set():
            raise ValueError('Operazione annullata; modello precedente conservato.')
        process.start()
        started = True
        sender.close()
        deadline = time.monotonic() + timeout
        while True:
            while not receiver.poll(.1):
                if stop.is_set():
                    raise ValueError('Operazione annullata; modello precedente conservato.')
                if time.monotonic() > deadline:
                    raise ValueError('Limite di 15 minuti senza avanzamento raggiunto.' if on_progress else 'Limite di 15 minuti raggiunto.')
                if not process.is_alive():
                    raise ValueError('Processo terminato senza risultato.')
            try:
                status, payload = receiver.recv()
            except EOFError:
                raise ValueError('Processo terminato senza risultato.') from None
            if status != 'progress':
                break
            if on_progress is not None:
                on_progress(payload)
                deadline = time.monotonic() + timeout
        if stop.is_set():
            raise ValueError('Operazione annullata; modello precedente conservato.')
        if status != 'ok':
            raise ValueError(payload)
        return payload
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


PHASES = {'scale': 'Lettura 1 di 3 · scala degli attributi ed estrazione delle righe',
          'scan': 'Lettura 2 di 3 · concetti specifici a blocchi',
          'levels': 'Ricerca dei livelli',
          'finish': 'Lettura 3 di 3 · ogni riga aggiunta ai concetti finali'}


def progress_view(status, rows):
    """Where the training is, as a share of the whole and as the quantities to show while it runs.

    A table read in chunks goes through four phases (two readings, the search of the levels, a last
    reading); a table held in memory only searches the levels. The search goes from the specific
    concepts to one concept adding together a twentieth at a time, twice (general concepts, then
    what several of them share): its share is how far the number of concepts has come, in logarithm.
    """
    import math
    if not status:
        return dict(share=0., title='Preparazione dei dati', facts=[('Righe di addestramento', f'{rows:,}')])
    phase = status['phase']
    chunked = rows > 20000
    seconds = status.get('seconds', 0.)
    part = 0.
    if phase in ('scale', 'scan', 'finish'):
        total = status.get('total') or rows
        part = min(status.get('rows', 0)/max(total, 1), 1.)
    elif status.get('start', 0) > 1:
        within = 1-math.log(max(status['concepts'], 1))/math.log(status['start'])
        part = (status.get('search', 1)-1+within)/2
    spans = {'scale': (0., .1), 'scan': (.1, .6), 'levels': (.6, .9), 'finish': (.9, 1.)} if chunked else {'levels': (0., 1.)}
    low, high = spans.get(phase, (0., 1.))
    share = low+(high-low)*part
    facts = [('Righe di addestramento', f'{rows:,}')]
    if phase in ('scale', 'scan', 'finish'):
        done = status.get('rows', 0)
        facts.append(('Righe lette in questa lettura', f"{done:,} di {status.get('total') or rows:,}"))
    if phase == 'levels':
        facts.append(('Ricerca', 'concetti generali (1 di 2)' if status.get('search', 1) == 1 else 'comportamenti condivisi (2 di 2)'))
        facts.append(('Concetti a questo passo', f"{status['concepts']:,} (partiti da {status.get('start', status['concepts']):,})"))
        if status.get('score') is not None:
            facts.append(('Errore sulle righe non viste (meno è meglio)', f"{-status['score']:.4f}"))
    elif status.get('concepts') is not None:
        facts.append(('Concetti nei blocchi' if phase == 'scan' else 'Concetti del livello di lavoro', f"{status['concepts']:,}"))
    facts.append(('Tempo trascorso', duration(seconds)))
    if share > .02:
        facts.append(('Tempo che manca, stima', duration(seconds*(1-share)/share)))
    if status.get('memory_mb'):
        facts.append(('Memoria del processo (picco)', f"{status['memory_mb']/1024:.2f} GB"))
    return dict(share=share, title=PHASES.get(phase, phase), facts=facts)


def duration(seconds):
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f'{seconds} s'
    if seconds < 3600:
        return f'{seconds//60} min {seconds % 60} s'
    return f'{seconds//3600} h {seconds % 3600//60} min'
