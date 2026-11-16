"""The box at the top of the generative page says what to do now (acan/vis/long_training.py, guide)."""
import unittest

from acan.vis.long_training import guide


def text(parts):
    return ''.join(part if isinstance(part, str) else str(getattr(part, 'children', '')) for part in (parts if isinstance(parts, list) else [parts]))


def job(order, status, step=0, total=40, graft=True, checkpoint='c'):
    return dict(id=f'j{order}', status=status, checkpoint=checkpoint if status != 'created' else None,
                progress=dict(step=step, graft=dict(vocabulary=32000)) if graft else {},
                config=dict(total_steps=total, series=dict(id='q', order=order, variant='queue', label=f'testo{order}')))


class GuideTests(unittest.TestCase):
    def test_every_state_names_the_step_to_go_to(self):
        self.assertIn('passo 1', text(guide(dict(jobs=[], corpora=[]), [])))
        self.assertIn('passo 2', text(guide(dict(jobs=[], corpora=[dict(status='ready')]), [])))
        self.assertIn('passo 1', text(guide(dict(jobs=[], corpora=[]), [dict(status='running')])))
        self.assertIn('Riprendi', text(guide(dict(jobs=[], corpora=[]), [dict(status='paused')])))
        learning = text(guide(dict(jobs=[job(0, 'completed'), job(1, 'running', step=12)], corpora=[]), []))
        self.assertIn('testo 2 di 2', learning)
        self.assertIn('passo 12', learning)
        self.assertIn('passo 4', learning)
        preparing = text(guide(dict(jobs=[job(0, 'completed'), job(1, 'preparing', graft=False)], corpora=[]), []))
        self.assertIn('preparando i token del testo 2', preparing)
        self.assertIn('Riprendi', text(guide(dict(jobs=[job(0, 'paused')], corpora=[]), [])))
        self.assertIn('ha finito', text(guide(dict(jobs=[job(0, 'completed')], corpora=[]), [])))


if __name__ == '__main__':
    unittest.main()
