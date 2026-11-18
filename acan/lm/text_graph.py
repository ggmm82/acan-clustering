"""Text is a stream of numeric rows learned by the tabular ConceptModel."""
import numpy as np
from scipy import sparse

from acan.core.concepts import ConceptModel, Encoder
from .text_rows import TextRows, semantic_vectors
from .text_reading import LevelReading, TextCursor

PROFILE = 'concept-table-v1'
BOOTSTRAP = 200_008
CHUNK = 512


class TextGraph:
    """One learned partition. Subsequent training rows accumulate continuation counts."""
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.tokens = 0
        self.tail = np.empty(0, np.int64)
        self.model = None
        self.parents = []
        self.reading = None

    def _learn_graph(self, ids, check):
        vectors, _ = semantic_vectors(ids, self.vocabulary, dimensions=min(32, self.vocabulary))
        self.rows = TextRows(vectors, context=8)
        self.model = ConceptModel(max_concepts=256, judged_rows=512, refine_rows=20000,
                                  chunk_rows=2048, covariance=False, workers=1)
        def frames():
            for frame in self.rows.chunks(ids, len(ids)-8, 2048):
                check()
                yield frame
        self.model.report = lambda info: check()
        self.model.fit_chunks(frames, labels=self.rows.targets)
        self.model.report = None
        self.parents = [level['parents'] for level in self.model.levels[1:]]
        self.counts = np.zeros((self.model.concepts, self.vocabulary))
        self._count(ids, 8, len(ids), self.rows.context, None, check)
        head = self._reader().probability(np.arange(self.vocabulary))
        self.counts.fill(0)
        self._count(ids, 8, len(ids), 1, head, check)

    def _reader(self):
        if self.reading is None:
            self.reading = LevelReading(sparse.csr_matrix(self.counts), self.parents, kappa=.5)
        return self.reading

    def _count(self, ids, begin, end, width, head, check):
        for offset in range(begin, end, CHUNK):
            check()
            pos = np.arange(offset, min(offset+CHUNK, end))
            weights = self.model.assignments(self.rows.frame(ids, pos, observed_context=width))
            targets = ids[pos]
            if head is not None:
                weights *= head[:, targets].T
                weights /= weights.sum(1, keepdims=True)
            np.add.at(self.counts.T, targets, weights)
        self.reading = None

    def add(self, tokens, check=lambda: None):
        ids = np.concatenate((self.tail, np.asarray(tokens, np.int64)))
        begin = len(self.tail)
        if self.model is None and len(ids) > 8:
            begin = min(len(ids), BOOTSTRAP)
            self._learn_graph(ids[:begin], check)
        if self.model is not None and begin < len(ids):
            head = self._reader().probability(np.arange(self.vocabulary))
            self._count(ids, begin, len(ids), 1, head, check)
        self.tail = ids[-8:].copy()
        self.tokens += len(tokens)

    def concepts(self):
        return [] if self.model is None else [len(p) for p in self.parents]+[self.model.concepts]

    def cursor(self, ids):
        return TextCursor(self, self._reader(), observed_context=1).prefill(ids)

    def distribution(self, ids):
        if self.model is None:
            return np.full(self.vocabulary, 1/self.vocabulary)
        return self.cursor(ids).distribution()

    def generate(self, ids, count, stop=(), temperature=0., seed=0):
        rng = np.random.default_rng(seed)
        cursor = self.cursor(ids) if self.model is not None else None
        result = []
        for _ in range(count):
            p = cursor.distribution() if cursor is not None else self.distribution([])
            if temperature:
                logits = np.log(np.maximum(p, 1e-300))
                weight = np.exp((logits-logits.max())/temperature)
                token = int(rng.choice(len(p), p=weight/weight.sum()))
            else:
                token = int(np.argmax(p))
            if token in stop:
                break
            result.append(token)
            if cursor is not None:
                cursor.observe(token)
        return result

    def logp(self, ids, positions=None):
        ids = np.asarray(ids, np.int64)
        if self.model is None:
            result = np.full(len(ids), -np.log(self.vocabulary))
        elif len(ids) <= 4096:
            frame = self.rows.frame(ids, np.arange(len(ids)), observed_context=1)
            result = np.log(self._reader().sequence_probability(self.model.assignments(frame), ids))
        else:
            state = self._reader().document()
            result = np.empty(len(ids))
            for start in range(0, len(ids), CHUNK):
                pos = np.arange(start, min(start+CHUNK, len(ids)))
                weights = self.model.assignments(self.rows.frame(ids, pos, observed_context=1))
                for t, w in zip(pos, weights):
                    result[t] = np.log(state.probability(w, [ids[t]])[0])
                    state.observe(w, ids[t])
        return result if positions is None else result[positions]

    def arrays(self):
        data = dict(text_graph=np.array([1, self.vocabulary, self.tokens]), tail=self.tail)
        if self.model is None:
            return data
        m = self.model
        data.update(vectors=self.rows.vectors, counts=self.counts, mean=m.encoder.mean,
                    std=m.encoder.std, resolution=m.resolution, kappa=np.array([m.kappa]),
                    root_p=m.root_p, root_g=m.root_g)
        for prefix, stats in [('work', m.statistics), ('above', m.above or {})]:
            data.update({prefix+'_'+key: value for key,value in stats.items()})
        data.update({f'parent_{i}':p for i,p in enumerate(self.parents)})
        return data

    @classmethod
    def from_arrays(cls, data):
        version, vocabulary, tokens = data['text_graph']
        if version != 1:
            raise ValueError('Formato del grafo di testo non supportato.')
        text = cls(int(vocabulary))
        text.tokens, text.tail = int(tokens), data['tail'].copy()
        if 'vectors' not in data:
            return text
        text.rows = TextRows(data['vectors'], 8)
        m = text.model = ConceptModel(kappa=float(data['kappa'][0]), covariance=False)
        encoder = m.encoder = Encoder()
        encoder.numeric = text.rows.features+text.rows.targets
        encoder.categorical, encoder.categories, encoder.offsets, encoder.width = [], {}, {}, 0
        encoder.mean, encoder.std = data['mean'], data['std']
        m.labels = text.rows.targets
        m.resolution, m.root_p, m.root_g = data['resolution'], data['root_p'], data['root_g']
        def stats(prefix):
            return {key[len(prefix):]: value for key,value in data.items() if key.startswith(prefix)}
        m.statistics, m.above = stats('work_'), stats('above_') or None
        text.counts = data['counts'].copy()
        text.parents = [data[f'parent_{i}'] for i in range(sum(k.startswith('parent_') for k in data))]
        return text

    def nbytes(self):
        return sum(a.nbytes for a in self.arrays().values())
