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
        values = matrix[:, self.numeric_index]
        self.mean = values.mean(0) if len(values) else np.zeros(len(self.numeric_index))
        std = values.std(0) if len(values) else np.ones(len(self.numeric_index))
        self.constant = std <= 1e-12
        self.std = np.where(self.constant, 1., std)
        self.offsets, start = {}, 0
        for name, indices in self.blocks.items():
            self.offsets[name] = start
            start += len(indices)
        self.width = start

    def block(self, name):
        return slice(self.offsets[name], self.offsets[name]+len(self.blocks[name]))

    def transform(self, matrix):
        matrix = np.asarray(matrix, float)
        x = (matrix[:, self.numeric_index]-self.mean)/self.std
        onehot = (np.concatenate([matrix[:, self.blocks[name]] for name in self.categorical], axis=1)
                  if self.categorical else np.zeros((len(matrix), 0)))
        # An encoded row may carry no level of a block (unseen category): it is unobserved.
        return x, np.ones_like(x), (onehot > .5).astype(float)


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
    from .graph import ConceptGraph
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
    g, v, log_p = model._parameters(s)
    k, m = len(s['C']), len(learner.cols)
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
    data = encoder.transform(x)
    full_t = model._route(s, data, model.train_weights)[0]
    view_t = model._route(s, encoder.transform(learner.NormDataset), model.train_weights)[0].T
    explained = model.explained_variance(x)
    graph = ConceptGraph(model, s, model.train_weights) if k > 1 else None
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
        pair_levels=[dict(depth=level['depth'], concepts=level['concepts'], pair_criterion=level['pair_criterion'],
                          members=[list(map(int, graph.members(node))) for node in level['nodes']])
                     for level in graph.levels()] if graph else [],
        explained={name: value for name, value in explained.items() if name != 'CP'},
        log_prior=np.log(s['C']/s['C'].sum()).tolist(),
        numeric=[dict(column=encoder.numeric_index[j], mean=float(encoder.mean[j]), std=float(encoder.std[j]),
                      centre=g[:, j].tolist(), variance=v[:, j].tolist()) for j in feature_numeric],
        blocks=[dict(name=name, columns=encoder.blocks[name], log_probability=log_p[:, encoder.block(name)].tolist())
                for name in feature_blocks])
    return learner.setModel()


def predict_assignments(model, rows, cix):
    """Posterior of the concepts given only the observed feature columns (k x n)."""
    state = model.routing_parameters
    rows = np.asarray(rows, float)
    position = {index: place for place, index in enumerate(cix)}
    logits = np.repeat(np.asarray(state['log_prior'])[None], len(rows), axis=0)
    for attribute in state['numeric']:
        if attribute['column'] not in position:
            continue
        z = (rows[:, position[attribute['column']]]-attribute['mean'])/attribute['std']
        centre, variance = np.asarray(attribute['centre']), np.asarray(attribute['variance'])
        logits = logits-.5*((z[:, None]-centre)**2/variance+np.log(variance)+LOG_2PI)
    for block in state['blocks']:
        # A single completed row is encoded with only its own level: absent
        # levels are zeros, and a block without any level is unobserved.
        onehot = np.zeros((len(rows), len(block['columns'])))
        for place, column in enumerate(block['columns']):
            if column in position:
                onehot[:, place] = rows[:, position[column]] > .5
        logits = logits+onehot@np.asarray(block['log_probability']).T
    return _softmax(logits)[0].T


def predict(model, row, cix):
    weights = predict_assignments(model, np.asarray(row).reshape(1, -1), cix)[:, 0]
    best = int(weights.argmax())
    return weights, float(weights[best]), best
