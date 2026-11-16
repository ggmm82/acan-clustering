"""Training on a table larger than the page's sample: the whole table, with progress and quantities."""
import threading
import time
import unittest

import numpy as np
import pandas as pd

from acan.vis.jobs import FullTables
from acan.vis.training import progress_view, run_isolated, train_native


def table(n, seed=1):
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 6, n)
    frame = pd.DataFrame(rng.normal(0, 1, (n, 3))+group[:, None]*3., columns=list('abc'))
    frame['colour'] = pd.Series(group % 3).map(lambda v: 'rgb'[v])
    frame['kind'] = pd.Series(np.where(rng.random(n) < .9, group % 4, rng.integers(0, 4, n))).map(lambda v: f'k{v}')
    return frame


def talkative(sender, seconds):
    """A process that reports while it works."""
    for step in range(int(seconds/.2)):
        time.sleep(.2)
        sender.send(('progress', dict(step=step)))
    sender.send(('ok', 'done'))
    sender.close()


def silent(sender, seconds):
    time.sleep(seconds)
    sender.send(('ok', 'done'))
    sender.close()


class LargeTrainingTests(unittest.TestCase):
    def test_whole_tables_are_kept_per_tab_and_the_oldest_gives_way(self):
        tables = FullTables(capacity=2, ttl=3600)
        for name in 'abc':
            tables.put(name, pd.DataFrame(dict(x=[1])), name)
            time.sleep(.01)
        self.assertIsNone(tables.get('a'))
        self.assertIsNotNone(tables.get('b'))
        tables.drop('b')
        self.assertIsNone(tables.get('b'))
        short = FullTables(ttl=0)
        short.put('x', pd.DataFrame(dict(x=[1])), 'x')
        time.sleep(.01)
        short.put('y', pd.DataFrame(dict(x=[1])), 'y')              # the expired table leaves when another arrives
        self.assertIsNone(short.get('x'))

    def test_progress_is_a_share_of_the_whole_with_the_quantities_of_the_phase(self):
        self.assertEqual(progress_view(None, 500)['share'], 0.)
        scan = progress_view(dict(phase='scan', rows=50000, total=100000, concepts=300, seconds=20., memory_mb=512.), 100000)
        self.assertAlmostEqual(scan['share'], .1+.5*.5)
        self.assertIn(('Concetti nei blocchi', '300'), scan['facts'])
        self.assertIn(('Righe lette in questa lettura', '50,000 di 100,000'), scan['facts'])
        small = progress_view(dict(phase='levels', concepts=10, start=100, search=1, score=-.5, seconds=3.), 800)
        self.assertAlmostEqual(small['share'], .25)                 # half of the first of two searches, in logarithm
        self.assertIn(('Errore sulle righe non viste (meno è meglio)', '0.5000'), small['facts'])
        shares = [progress_view(dict(phase=phase, rows=rows, total=10, concepts=5, start=50, search=search), 30000)['share']
                  for phase, rows, search in (('scale', 5, 1), ('scan', 5, 1), ('levels', 0, 1), ('levels', 0, 2), ('finish', 5, 1))]
        self.assertEqual(shares, sorted(shares))                    # the phases follow one another

    def test_a_process_that_reports_may_take_longer_than_one_that_is_silent(self):
        seen = []
        self.assertEqual(run_isolated(talkative, (9.,), threading.Event(), timeout=6, on_progress=seen.append), 'done')
        self.assertGreater(len(seen), 3)
        with self.assertRaisesRegex(ValueError, '15 minuti'):
            run_isolated(silent, (20.,), threading.Event(), timeout=6, on_progress=seen.append)

    def test_a_large_table_is_trained_whole_and_tells_where_it_is(self):
        frame = table(26000)
        options = dict(routing_profile='concept-graph-v1', max_depth=1, max_epochs=100, min_samples_split=12, tau=1., alpha=.5,
                       similarity_threshold=.92, lr=.05, batch_size=1000, device='numpy')
        seen = []
        graph = train_native(frame, ['kind'], options, seen.append)
        phases = [item['phase'] for item in seen]
        self.assertEqual(sorted(set(phases), key=phases.index), ['scale', 'scan', 'levels', 'finish'])
        self.assertEqual(max(item['rows'] for item in seen if item['phase'] == 'scan'), 26000)   # every row, once
        root = graph.clusterModel[graph.rootId]
        self.assertEqual(root.n, 26000)
        self.assertGreater(root.k(), 1)


if __name__ == '__main__':
    unittest.main()
