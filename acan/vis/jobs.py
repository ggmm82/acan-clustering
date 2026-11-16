"""In-process training jobs, isolated by the server's signed browser session.

Use one server process: jobs are intentionally not shared across workers.
Completed jobs expire after one hour; running jobs are never evicted.
"""

import threading
import time


class TrainingJobs:
    def __init__(self, capacity=32, ttl=3600, max_running=None):
        self._jobs = {}
        self._lock = threading.Lock()
        self.capacity = capacity
        self.ttl = ttl
        self.max_running = max_running

    def start(self, session_id, learner, minmax, labels):
        with self._lock:
            now = time.monotonic()
            for key, job in list(self._jobs.items()):
                if job['done'].is_set() and now - job['finished'] > self.ttl:
                    del self._jobs[key]
            old = self._jobs.get(session_id)
            if old and not old['done'].is_set():
                raise ValueError('Un training è già in corso in questa sessione.')
            if self.max_running is not None and sum(not job['done'].is_set() for job in self._jobs.values()) >= self.max_running:
                raise ValueError('Un benchmark è già in corso sul server: attendere o fermare il proprio confronto.')
            if old is None and len(self._jobs) >= self.capacity:
                raise ValueError('Server occupato: riprovare più tardi.')
            job = dict(learner=learner, minmax=minmax, labels=list(labels),
                       done=threading.Event(), error=None, finished=None)
            self._jobs[session_id] = job

            def run():
                try:
                    learner.nLearn(max_epochs=500, patience=80)
                except Exception as exc:
                    job['error'] = str(exc)
                finally:
                    job['finished'] = time.monotonic()
                    job['done'].set()

            threading.Thread(target=run, daemon=True).start()
            return job

    def get(self, session_id):
        with self._lock:
            return self._jobs.get(session_id)

    def stop(self, session_id):
        job = self.get(session_id)
        if job and not job['done'].is_set():
            job['learner'].request_stop()
            return True
        return False


class FullTables:
    """Uploaded tables too large for the page, kept on the server for training (one per browser tab).

    The page works on a sample; training reads the whole table from here. Tables expire after
    ``ttl`` seconds without use and only the most recent ``capacity`` are kept.
    """

    def __init__(self, capacity=4, ttl=6*3600):
        self._tables, self._lock = {}, threading.Lock()
        self.capacity, self.ttl = capacity, ttl

    def put(self, key, frame, name):
        with self._lock:
            now = time.monotonic()
            for other, item in list(self._tables.items()):
                if now-item['used'] > self.ttl:
                    del self._tables[other]
            self._tables[key] = dict(frame=frame, name=name, used=now)
            while len(self._tables) > self.capacity:
                del self._tables[min(self._tables, key=lambda other: self._tables[other]['used'])]

    def drop(self, key):
        with self._lock:
            self._tables.pop(key, None)

    def get(self, key):
        with self._lock:
            item = self._tables.get(key)
            if item is None:
                return None
            item['used'] = time.monotonic()
            return item['frame']
