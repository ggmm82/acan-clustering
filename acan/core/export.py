"""A learned concept graph as a ready-to-use Python package (inference and further training).

The package depends only on numpy and pandas and holds only what it needs: a copy of the
core (``concepts.py``, ``concept_algebra.py``), the application's encoding, the learned
state, the command line (``__main__.py``), ``requirements.txt`` and a README.
"""
import inspect
import io
import json
import zipfile
from pathlib import Path

import numpy as np

PACKAGE = 'modello_concetti'
TEMPLATE = Path(__file__).resolve().parent.parent/'export_template'
OPTIONS = ('max_concepts', 'rows_per_concept', 'kappa', 'seed', 'judged_rows', 'merge_fraction', 'em_steps',
           'restarts', 'restart_rows', 'covariance', 'covariance_attributes', 'shared_start')


def _lists(value):
    if isinstance(value, dict):
        return {str(key): _lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_lists(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def core_state(model):
    """Everything the concept model learned, as JSON-ready lists: sums, levels, encoding of rows."""
    levels = [dict(depth=level['depth'], concepts=level['concepts'], pair_criterion=level['pair_criterion'],
                   score=level['score'], mass=level['mass'], parents=level['parents'],
                   statistics=level['statistics'], table=level['table']) for level in model.levels]
    return _lists(dict(
        options={name: getattr(model, name) for name in OPTIONS},
        labels=model.labels, root_p=model.root_p, root_g=model.root_g, resolution=model.resolution,
        train_weights=list(model.train_weights), statistics=model.statistics, cross=model.cross, cov=model.cov,
        above=model.above, levels=levels, edges=model.edges, selection=model.selection))


def encoder_state(encoder):
    keys = ('columns', 'numeric_index', 'numeric', 'categorical', 'blocks', 'mean', 'std', 'constant',
            'offsets', 'width', 'marks')
    return _lists({key: getattr(encoder, key) for key in keys})


def _requirements():
    """The libraries the package needs, with the same minimum versions as the ACAN library."""
    return ('numpy>=1.24\n'
            'pandas>=2.0\n'
            '# facoltativo, solo per leggere file Excel\n'
            '# openpyxl>=3.1\n')


def _source(module_path):
    return Path(module_path).read_text(encoding='utf-8')


def build_package(graph):
    """Zip bytes of the package for the root of a saved application model."""
    from acan import concept_algebra
    from acan.core import concept_adapter, concepts
    from acan import preprocessing
    node = graph.clusterModel[graph.rootId]
    state = getattr(node, 'routing_parameters', None) or {}
    if 'core' not in state:
        raise ValueError('Questo modello è stato salvato prima che il modello completo fosse conservato: '
                         'riaddestralo per scaricare il pacchetto Python.')
    # Only what inference and further training need: the levels keep their shares and matrices,
    # not the statistics used to draw them.
    core = dict(state['core'])
    core['levels'] = [{key: level[key] for key in ('depth', 'concepts', 'mass', 'parents')} for level in core['levels']]
    for key in ('edges', 'selection'):
        core.pop(key, None)
    full = dict(labels=list(graph.labels), columns=[str(column) for column in node.cols],
                min_max={str(key): list(value) for key, value in graph.minMax.items() if key in node.cols},
                preprocessing=graph.preprocessing, encoder=state['encoder_state'], core=core)
    encoding = ('"""The application\'s encoding of rows, copied from the ACAN library."""\n'
                'import numpy as np\nimport pandas as pd\n\n\n'
                + inspect.getsource(concept_adapter.MatrixEncoder) + '\n\n'
                + inspect.getsource(preprocessing.transform_features))
    core_source = _source(concepts.__file__).replace('from acan.concept_algebra import', 'from .concept_algebra import')
    files = {
        'requirements.txt': _requirements(),
        'README.md': (TEMPLATE/'README.md.tmpl').read_text(encoding='utf-8').replace(
            '{target}', ', '.join(graph.labels) or 'nessuno').replace('{rows}', str(node.n)),
        '__main__.py': (TEMPLATE/'__main__.py.tmpl').read_text(encoding='utf-8'),
        f'{PACKAGE}/__init__.py': (TEMPLATE/'__init__.py.tmpl').read_text(encoding='utf-8'),
        f'{PACKAGE}/model.py': (TEMPLATE/'model.py.tmpl').read_text(encoding='utf-8'),
        f'{PACKAGE}/concepts.py': core_source,
        f'{PACKAGE}/concept_algebra.py': _source(concept_algebra.__file__),
        f'{PACKAGE}/encoding.py': encoding,
        f'{PACKAGE}/state.json': json.dumps(_lists(full)),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            # The package sits at the root of the zip: ``python file.zip`` runs __main__.py,
            # Python can import it from the zip, and pip can install the zip.
            archive.writestr(name, content)
    return buffer.getvalue()
