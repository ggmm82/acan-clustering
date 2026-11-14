"""Tabular adapter of the concept graph (acan/core/concepts.py).

The interactive graph keeps its encoded, min-max normalized matrix. Numeric
columns are Gaussian attributes; each one-hot block is one categorical
attribute. Targets are read from the concepts and never used for routing.
"""
import numpy as np

from .concepts import LOG_2PI, ConceptModel, _softmax

PROFILE = 'concept-graph-v1'


class MatrixEncoder:
    """Encoder interface of concepts.py over an already encoded matrix."""

    def __init__(self, columns, blocks, matrix):
        member = {index for indices in blocks.values() for index in indices}
        self.columns = list(columns)
        self.numeric_index = [i for i in range(len(columns)) if i not in member]
        self.numeric = [columns[i] for i in self.numeric_index]
        self.categorical = list(blocks)
        self.blocks = {name: list(indices) for name, indices in blocks.items()}
        # The interface fills an empty numeric cell with the mean and marks it in a column
        # ``<name>_NaN``. The mark stays an attribute; the filled value is not an observation.
        place = {str(name): index for index, name in enumerate(self.columns)}
        self.marks = [place.get(str(name)+'_NaN', -1) for name in self.numeric]
        observed = self._observed(matrix)
        count = np.maximum(observed.sum(0), 1)
        values = matrix[:, self.numeric_index]*observed
        self.mean = values.sum(0)/count
        std = np.sqrt(np.maximum((values*values).sum(0)/count-self.mean**2, 0))
        self.constant = std <= 1e-12
        self.std = np.where(self.constant, 1., std)
        self.offsets, start = {}, 0
        for name, indices in self.blocks.items():
            self.offsets[name] = start
            start += len(indices)
        self.width = start

    def block(self, name):
        return slice(self.offsets[name], self.offsets[name]+len(self.blocks[name]))

    def _observed(self, matrix):
        matrix = np.asarray(matrix, float)
        observed = np.ones((len(matrix), len(self.numeric_index)))
        for j, mark in enumerate(self.marks):
            if mark >= 0:
                observed[:, j] = matrix[:, mark] <= .5
        return observed

    def transform(self, matrix):
        matrix = np.asarray(matrix, float)
        observed = self._observed(matrix)
        x = (matrix[:, self.numeric_index]-self.mean)/self.std*observed
        onehot = (np.concatenate([matrix[:, self.blocks[name]] for name in self.categorical], axis=1)
                  if self.categorical else np.zeros((len(matrix), 0)))
        # An encoded row may carry no level of a block (unseen category): it is unobserved.
        return x, observed, (onehot > .5).astype(float)


def attribute_blocks(learner):
    """One-hot blocks: targets from the learner, features from the declared categorical names."""
    blocks = {}
    for label, indices in zip([l for l in learner.labels if l not in learner.cols], learner._supervised_groups):
        blocks[str(label)] = indices
    taken = {i for indices in blocks.values() for i in indices}
    for name in getattr(learner, 'categorical_names', None) or []:
        indices = [i for i, col in enumerate(learner.cols) if str(col).startswith(str(name)+'_') and i not in taken]
        if len(indices) > 1 and str(name) not in blocks:
            blocks[str(name)] = indices
            taken.update(indices)
    return blocks


def train(learner, epochs):
    """Fit the concept graph on the encoded table and fill one node of the application's graph."""
    x = learner.full_dataset.to_numpy(dtype=np.float64)
    blocks = attribute_blocks(learner)
    encoder = MatrixEncoder(learner.cols, blocks, x)
    labels = [str(label) for label in learner.labels]
    learner.history = []

    def progress(step):
        learner.iteration = len(learner.history)
        learner.history.append(dict(epoch=len(learner.history), loss=-step['score'], nll=-step['score'], mse=0.,
            prediction_loss=-step['score'], normalized_cp=float('nan'), cp=float('nan'), cp_weight=0.,
            k_active=step['k']))
        return not learner._should_stop

    model = ConceptModel(seed=learner.random_state).fit(x, labels, encoder=encoder, progress=progress)
    s = model.statistics
    k, m = len(s['C']), len(learner.cols)
    groups, relevance = _prototypes(model, encoder, s, m)
    g, v, log_p = model._parameters(s)
    data = encoder.transform(x)
    full_t = model._route(s, model._seen(data)[0], model.train_weights)[0]
    view_t = model._route(s, model._seen(encoder.transform(learner.NormDataset))[0], model.train_weights)[0].T
    # Prototypes are shown in the units of the table, whatever level the concepts are seen through.
    table = model._statistics(full_t, data)
    groups[:, encoder.numeric_index] = table['U']/np.maximum(table['Cm'], 1e-12)*encoder.std+encoder.mean
    explained = model.explained_variance(x)
    learner.k = k
    learner.Groups, learner.Relevance = groups, relevance
    learner.Transformation = view_t
    hard = np.zeros_like(view_t)
    hard[view_t.argmax(0), np.arange(view_t.shape[1])] = 1
    learner.Transformation_hard = hard
    learner.Counts = np.bincount(full_t.argmax(1), minlength=k)[:, None]
    learner.Freqs = (s['C']/s['C'].sum())[None]
    learner.cp = explained['CP']*np.sqrt(m)
    learner.k_active = int((learner.Counts > 0).sum())
    for row in learner.history:
        row['normalized_cp'], row['cp'] = explained['CP'], learner.cp
    feature_numeric = [j for j, name in enumerate(encoder.numeric) if name not in labels]
    feature_blocks = [name for name in encoder.categorical if name not in labels]
    learner.variational_state = dict(
        objective='predictive-concepts-v1', cp_definition='explained-variance-v1', generalization='reader-loss-v1',
        hierarchy_readout='conditional-mixture-v1', selection=model.selection,
        training_rows=len(x), training_score=model.final_score,
        numeric_targets=[learner.cols.index(label) for label in learner.labels if label in learner.cols],
        numeric_readout='weighted-training-mean',
        edges=[dict(first=i, second=j, weight=w) for i, j, w in model.edges],
        levels=[_level(model, encoder, level) for level in model.levels],
        explained={name: value for name, value in explained.items() if name != 'CP'},
        # The level through which the concepts read a row (None when they read the table directly).
        context=None if model.above is None else _recognition(model, encoder, model.above, feature_numeric, feature_blocks),
        **_recognition(model, encoder, s, feature_numeric, feature_blocks))
    return learner.setModel()


def _recognition(model, encoder, s, feature_numeric, feature_blocks):
    """What is needed to recognize the concepts of a level from the observed features of a row."""
    g, v, log_p = model._parameters(s)
    shift = s['U']/np.maximum(s['Cm'], 1e-12)                      # the centre this level takes away from its rows
    block = None
    if model.cross is not None and s is model.statistics:        # features recognized together
        together = set(model.cov.tolist())
        feature_numeric = [j for j in feature_numeric if j not in together]
        centre, covariance = model._covariance()
        block = dict(columns=[int(encoder.numeric_index[j]) for j in model.cov], mean=encoder.mean[model.cov].tolist(),
                     std=encoder.std[model.cov].tolist(), centre=centre.tolist(), covariance=covariance.tolist())
    return dict(
        correlated=block,
        log_prior=np.log(s['C']/s['C'].sum()).tolist(),
        numeric=[dict(column=encoder.numeric_index[j], mean=float(encoder.mean[j]), std=float(encoder.std[j]),
                      mark=int(encoder.marks[j]),
                      centre=g[:, j].tolist(), variance=v[:, j].tolist(), shift=shift[:, j].tolist()) for j in feature_numeric],
        blocks=[dict(name=name, columns=encoder.blocks[name], log_probability=log_p[:, encoder.block(name)].tolist())
                for name in feature_blocks])


def _prototypes(model, encoder, s, m):
    """Typical value and relevance of every attribute for each concept, in the units of the app's matrix."""
    g, v, log_p = model._parameters(s)
    k = len(s['C'])
    groups, relevance = np.zeros((k, m)), np.zeros((k, m))
    groups[:, encoder.numeric_index] = g*encoder.std+encoder.mean
    relevance[:, encoder.numeric_index] = np.where(encoder.constant, 0., np.clip(1-v, 0, 1))
    for name in encoder.categorical:
        block, indices = encoder.block(name), encoder.blocks[name]
        p, root = np.exp(log_p[:, block]), model.root_p[block]
        groups[:, indices] = p
        impurity = 1-(root*root).sum()
        gain = ((p-root)**2).sum(1)/impurity if impurity > 1e-12 else np.zeros(k)
        relevance[:, indices] = np.clip(gain, 0, 1)[:, None]
    return groups, relevance


def _level(model, encoder, level):
    """One level of the graph as it is saved: its concepts and the dense matrix towards the level above."""
    _, _, log_p = model._parameters(level['statistics'])
    reads = []
    for name in encoder.categorical:
        if name in model.labels:
            columns = [str(encoder.columns[index]) for index in encoder.blocks[name]]
            values = [column[len(name)+1:] if column.startswith(name+'_') else column for column in columns]
            reads.append([f'{name}: {values[place]}' for place in log_p[:, encoder.block(name)].argmax(1)])
    groups, relevance = _prototypes(model, encoder, level['table'], len(encoder.columns))
    return dict(depth=level['depth'], concepts=level['concepts'], pair_criterion=float(level['pair_criterion']),
                groups=np.round(groups, 6).tolist(), relevance=np.round(relevance, 4).tolist(),
                mass=np.round(level['mass'], 6).tolist(),
                reads=['\n'.join(parts) for parts in zip(*reads)] if reads else [],
                # parents[i][j] = P(concept i of the level above | concept j): every column sums to one
                parents=None if level['parents'] is None else np.round(level['parents'], 4).tolist())


def predict_assignments(model, rows, cix):
    """Posterior of the concepts given only the observed feature columns (k x n)."""
    state = model.routing_parameters
    rows = np.asarray(rows, float)
    position = {index: place for place, index in enumerate(cix)}
    shift = None
    if state.get('context'):                                     # recognize the level above, take its centre away
        above = _posterior(state['context'], rows, position)
        shift = {attribute['column']: above@np.asarray(attribute['shift']) for attribute in state['context']['numeric']}
    return _posterior(state, rows, position, shift).T


def _posterior(state, rows, position, shift=None):
    logits = np.repeat(np.asarray(state['log_prior'])[None], len(rows), axis=0)
    for attribute in state['numeric']:
        if attribute['column'] not in position:
            continue
        z = (rows[:, position[attribute['column']]]-attribute['mean'])/attribute['std']
        if shift is not None:
            z = z-shift[attribute['column']]
        centre, variance = np.asarray(attribute['centre']), np.asarray(attribute['variance'])
        term = -.5*((z[:, None]-centre)**2/variance+np.log(variance)+LOG_2PI)
        mark = attribute.get('mark', -1)
        if mark in position:                                     # a filled cell is not an observation
            term = term*(rows[:, position[mark]] <= .5)[:, None]
        logits = logits+term
    block = state.get('correlated')
    if block:                                                    # the observed part of the features read together
        seen = [place for place, column in enumerate(block['columns']) if column in position]
        if seen:
            z = rows[:, [position[block['columns'][place]] for place in seen]]
            z = (z-np.asarray(block['mean'])[seen])/np.asarray(block['std'])[seen]
            centre = np.asarray(block['centre'])[:, seen]
            covariance = np.asarray(block['covariance'])[:, seen][:, :, seen]
            inverse, volume = np.linalg.inv(covariance), np.linalg.slogdet(covariance)[1]
            d = z[:, None]-centre[None]
            logits = logits-.5*(np.einsum('nkp,kpq,nkq->nk', d, inverse, d)+volume+len(seen)*LOG_2PI)
    for block in state['blocks']:
        # A single completed row is encoded with only its own level: absent
        # levels are zeros, and a block without any level is unobserved.
        onehot = np.zeros((len(rows), len(block['columns'])))
        for place, column in enumerate(block['columns']):
            if column in position:
                onehot[:, place] = rows[:, position[column]] > .5
        logits = logits+onehot@np.asarray(block['log_probability']).T
    return _softmax(logits)[0]


def predict(model, row, cix):
    weights = predict_assignments(model, np.asarray(row).reshape(1, -1), cix)[:, 0]
    best = int(weights.argmax())
    return weights, float(weights[best]), best
