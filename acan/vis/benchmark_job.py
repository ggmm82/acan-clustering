"""Owned CPU process for the comparison on the application's current dataset."""
import hashlib
import threading

from .training import run_isolated


def signature(model, data):
    return hashlib.sha256((model + data.to_csv(index=False)).encode()).hexdigest()


def _worker(sender, model, test, train, before):
    try:
        import os
        if hasattr(os, 'nice'):
            os.nice(10)
        from threadpoolctl import threadpool_limits
        from .evaluation import evaluate
        with threadpool_limits(limits=1):
            report = evaluate(model, test, train, before, baselines=True)
        report['signature'] = signature(model, test)
        sender.send(('ok', report))
    except Exception as exc:
        sender.send(('error', str(exc)))
    finally:
        sender.close()


class BenchmarkJob:
    iteration = 'Confronto CPU sui dati della sessione in corso'

    def __init__(self, model, test, train, before=None):
        self.arguments = model, test, train, before
        self.report = None
        self._stop = threading.Event()

    def nLearn(self, **unused):
        self.report = run_isolated(_worker, self.arguments, self._stop)

    def request_stop(self):
        self._stop.set()
