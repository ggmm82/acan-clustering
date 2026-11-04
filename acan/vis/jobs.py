"""In-process training jobs, isolated by the server's signed browser session.

Use one server process: jobs are intentionally not shared across workers.
Completed jobs expire after one hour; running jobs are never evicted.
"""

import threading
import time


class TrainingJobs:
    def __init__(self, capacity=32, ttl=3600):
        self._jobs = {}
        self._lock = threading.Lock()
        self.capacity = capacity
        self.ttl = ttl

    def start(self, session_id, learner, minmax, labels):
        with self._lock:
            now = time.monotonic()
            for key, job in list(self._jobs.items()):
                if job['done'].is_set() and now - job['finished'] > self.ttl:
                    del self._jobs[key]
            old = self._jobs.get(session_id)
            if old and not old['done'].is_set():
                raise ValueError('Un training è già in corso in questa sessione.')
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
