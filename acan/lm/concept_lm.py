"""The language model that is the concept model: no neural network, no gradient.

The training text is a table. Every position t is a row: the tokens before it are its attributes,
the token at t is its target. The concepts are those of the concept graph (MODEL.md): nodes that keep
only sums of their members, read with mass times similarity, generalized where the reader loses least,
judged on rows they never saw.

Concepts by context. At level j a concept holds the positions whose last j tokens are equal (level 0
holds every position). It keeps

    C = members,   N(y) = members followed by y,   D = distinct tokens that follow.

Reading. A concept is read with its level above (the context one token shorter) as prior, as a concept
of a table is read with the scale of its level above:

    p_j(y) = (N_j(y) + w_j p_{j-1}(y)) / (C_j + w_j),      w_j = a_j * D_j,

so a concept followed by many different tokens trusts its parent more. The level above of level 0 is
the uniform distribution. A context never seen leaves the reading as it is (the concept was generalized
into its parent). The weights a_j are chosen by the judgment on unseen text.

Concepts of the text being read. The same concepts built only from the earlier positions of the text in
hand, read after the global ones with the global reading as prior (weights b_j): what the text said
before can be said again, at any distance.

Learning is counting: a piece of text is a table of sums, and two tables are added together. Pieces are
added like the dyadic blocks of the other concept layers: equal tables are merged two by two, so every
position is merged about log2(pieces) times. Above the memory budget the concepts the reader loses least
on are generalized into their parent (their sums are already in it): the loss is the log-likelihood of
the members read through the parent instead of the concept.

Contexts are written as integers: key_j = key_{j-1} * (V + 1) + token + 1 in 64 bits. Up to
(V + 1)^j < 2^64 (four tokens with 32,000 tokens of vocabulary) the key is exact; beyond, it wraps
around as a hash, and two contexts may share a key with probability about n^2 / 2^64.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
GRID = tuple(2.**k for k in range(-5, 9))                       # weights tried by the judgment


def context_keys(ids, depth, vocabulary):
    """Keys of the contexts of every position for levels 0..depth (uint64); positions with fewer tokens get 0 and are invalid."""
    ids = np.asarray(ids, np.int64)
    base = np.uint64(vocabulary+1)
    keys = [np.zeros(len(ids), np.uint64)]
    valid = [np.ones(len(ids), bool)]
    key = np.zeros(len(ids), np.uint64)
    with np.errstate(over='ignore'):
        for j in range(1, depth+1):
            shifted = np.zeros(len(ids), np.uint64)
            shifted[j:] = (ids[:-j]+1).astype(np.uint64) if j < len(ids) else shifted[j:]
            key = key*base+shifted
            keys.append(key.copy())
            ok = np.zeros(len(ids), bool); ok[j:] = True
            valid.append(ok)
    return keys, valid


class Table:
    """The concepts of one level: sorted keys with C, D and the cells (concept, token) with their counts."""

    def __init__(self, keys, C, cells, counts, vocabulary):
        self.V = vocabulary
        self.keys, self.C, self.cells, self.counts = keys, C, cells, counts
        self.D = np.bincount((cells//(vocabulary+1)).astype(np.int64), minlength=len(keys)) if len(keys) else np.zeros(0, np.int64)

    @classmethod
    def count(cls, keys, targets, vocabulary):
        """The sums of one level from the context keys and the targets of some positions."""
        concepts, which, C = np.unique(keys, return_inverse=True, return_counts=True)
        cells, counts = np.unique(which.astype(np.int64)*(vocabulary+1)+targets, return_counts=True)
        return cls(concepts, C.astype(np.int64), cells, counts.astype(np.int64), vocabulary)

    @classmethod
    def empty(cls, vocabulary):
        return cls(np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), vocabulary)

    def pairs(self):
        """Every cell as (context key, token, count)."""
        concept = (self.cells//(self.V+1)).astype(np.int64)
        return self.keys[concept], self.cells % (self.V+1), self.counts

    def __add__(self, other):
        """Two tables of the same level added together: sums are additive."""
        if not len(other.keys):
            return self
        if not len(self.keys):
            return other
        keys = np.concatenate((self.keys, other.keys))
        concepts, which = np.unique(keys, return_inverse=True)
        C = np.bincount(which, weights=np.concatenate((self.C, other.C)), minlength=len(concepts)).astype(np.int64)
        k1, y1, n1 = self.pairs()
        k2, y2, n2 = other.pairs()
        index = np.searchsorted(concepts, np.concatenate((k1, k2))).astype(np.int64)
        cells, where = np.unique(index*(self.V+1)+np.concatenate((y1, y2)), return_inverse=True)
        counts = np.bincount(where, weights=np.concatenate((n1, n2)), minlength=len(cells)).astype(np.int64)
        return Table(concepts, C, cells, counts, self.V)

    def keep(self, alive):
        """Only the concepts marked alive, with their cells."""
        concept = (self.cells//(self.V+1)).astype(np.int64)
        kept = alive[concept]
        renumber = np.cumsum(alive)-1
        cells = renumber[concept[kept]]*(self.V+1)+self.cells[kept] % (self.V+1)
        return Table(self.keys[alive], self.C[alive], cells, self.counts[kept], self.V)

    def find(self, keys):
        """Index of the concept of every key, -1 where there is none."""
        if not len(self.keys):
            return np.full(len(keys), -1, np.int64)
        place = np.minimum(np.searchsorted(self.keys, keys), len(self.keys)-1)
        return np.where(self.keys[place] == keys, place, -1)

    def lookup(self, keys, targets):
        """C, N(target) and D of the concept of every key (zeros where there is none)."""
        index = self.find(keys)
        seen = index >= 0
        safe = np.where(seen, index, 0)
        cell = safe*(self.V+1)+targets
        N = np.zeros(len(keys))
        if len(self.cells):
            at = np.minimum(np.searchsorted(self.cells, cell), len(self.cells)-1)
            N = np.where(seen & (self.cells[at] == cell), self.counts[at], 0).astype(float)
        if not len(self.keys):
            z = np.zeros(len(keys)); return z, z, z
        return np.where(seen, self.C[safe], 0).astype(float), N, np.where(seen, self.D[safe], 0).astype(float)

    def followers(self, index):
        """The tokens that follow concept ``index`` and their counts."""
        start = np.searchsorted(self.cells, index*(self.V+1))
        end = np.searchsorted(self.cells, (index+1)*(self.V+1))
        return self.cells[start:end] % (self.V+1), self.counts[start:end]

    @property
    def size(self):
        return len(self.keys)+len(self.cells)

    def nbytes(self):
        return sum(array.nbytes for array in (self.keys, self.C, self.D, self.cells, self.counts))


class ConceptLM:
    """Concepts of a text by context, global (the training text) and of the text being read."""

    def __init__(self, vocabulary, depth=6, document=4, budget_bytes=None):
        self.V, self.depth, self.document = int(vocabulary), int(depth), int(document)
        self.budget_bytes = budget_bytes
        self.levels = [Table.empty(self.V) for _ in range(self.depth+1)]
        self.a = np.ones(self.depth+1)
        self.b = np.ones(self.document+1)
        self.counter = []                                       # tables of 2^i pieces each, not yet merged into ``levels``
        self.tokens = 0
        self.generalized = 0                                    # concepts generalized into their parent to stay in the budget

    # ---- learning: counting and adding ------------------------------------------------
    def piece(self, ids):
        """The tables of a piece of text (positions after its first token, contexts inside the piece)."""
        ids = np.asarray(ids, np.int64)
        keys, valid = context_keys(ids, self.depth, self.V)
        targets = ids
        return [Table.count(keys[j][valid[j]], targets[valid[j]], self.V) for j in range(self.depth+1)]

    def add(self, ids):
        """Learn a piece of text: count it and add it to the binary counter of pieces."""
        tables = self.piece(ids)
        self.tokens += len(ids)
        level = 0
        while level < len(self.counter) and self.counter[level] is not None:
            tables = [x+y for x, y in zip(self.counter[level], tables)]
            self.counter[level] = None
            level += 1
            tables = self._within_budget(tables)
        if level == len(self.counter):
            self.counter.append(None)
        self.counter[level] = tables

    def settle(self):
        """Add every table of the counter into the levels (before reading or saving)."""
        for tables in self.counter:
            if tables is not None:
                self.levels = [x+y for x, y in zip(self.levels, tables)]
        self.counter = []
        self.levels = self._within_budget(self.levels)
        return self

    def merge(self, other):
        """Add another model learned on other text (a learning queue): the sums of both."""
        self.settle(); other.settle()
        self.levels = self._within_budget([x+y for x, y in zip(self.levels, other.levels)])
        self.tokens += other.tokens
        return self

    def nbytes(self):
        tables = list(self.levels)+[t for tables in self.counter if tables is not None for t in tables]
        return sum(table.nbytes() for table in tables)

    def concepts(self):
        return [len(table.keys) for table in self.levels]

    def _within_budget(self, tables):
        """Generalize into their parent the concepts the reader loses least on, until the tables fit the budget.

        The loss of a concept is the log-likelihood its members lose when they are read through the
        parent instead of the concept (in nats): sum_y N(y) log(p_concept(y) / p_parent(y)), with
        p_concept = N/C and p_parent the parent's frequencies (1/V for a token it never saw). Its sums are
        already in the parent, so nothing is lost but the distinction. Level 0 is never generalized.
        """
        if not self.budget_bytes:
            return tables
        total = sum(table.nbytes() for table in tables)
        if total <= self.budget_bytes:
            return tables
        losses = []
        for j in range(1, len(tables)):
            losses.append(self._losses(tables, j))
        every = np.concatenate([loss for loss in losses if len(loss)])
        if not len(every):
            return tables
        # Bytes per concept: its key, C, D and its cells. Generalize from the cheapest until under budget.
        sizes = np.concatenate([self._bytes_per_concept(tables[j]) for j in range(1, len(tables)) if len(losses[j-1])])
        order = np.argsort(every/np.maximum(sizes, 1), kind='stable')            # least loss per byte first
        freed = np.cumsum(sizes[order])
        cut = int(np.searchsorted(freed, total-self.budget_bytes*.9))+1         # a margin, so merges do not prune at every step
        drop = np.zeros(len(every), bool); drop[order[:cut]] = True
        start, out = 0, [tables[0]]
        for j in range(1, len(tables)):
            n = len(tables[j].keys)
            alive = ~drop[start:start+n]
            start += n
            self.generalized += int((~alive).sum())
            out.append(tables[j].keep(alive))
        return out

    def _bytes_per_concept(self, table):
        cells = np.bincount((table.cells//(self.V+1)).astype(np.int64), minlength=len(table.keys))
        return 8*3+cells*16

    def _losses(self, tables, j):
        """Loss of reading every concept of level j through its parent (level j-1)."""
        table, parent = tables[j], tables[j-1]
        if not len(table.keys):
            return np.zeros(0)
        concept = (table.cells//(self.V+1)).astype(np.int64)
        token = table.cells % (self.V+1)
        # The parent key of a concept is not written in it: it is the key of the same positions one token shorter.
        # It is recovered from the key where the key is exact, and otherwise the parent is read as the whole level above.
        base = np.uint64(self.V+1)
        exact = (self.V+1)**j < 2**64
        if exact:
            parent_keys = table.keys//base
            C_p, N_p, _ = parent.lookup(parent_keys[concept], token)
        else:
            C_p, N_p = np.full(len(concept), parent.C.sum(), float), np.zeros(len(concept))
        p_concept = table.counts/table.C[concept]
        p_parent = np.where(C_p > 0, (N_p+1.)/(C_p+self.V), 1./self.V)
        per_cell = table.counts*np.log(np.maximum(p_concept, 1e-300)/p_parent)
        return np.bincount(concept, weights=per_cell, minlength=len(table.keys))

    # ---- reading ----------------------------------------------------------------------------
    def logp(self, ids, positions=None, a=None, b=None):
        """log p (nats) of the token at every position of a text, read causally.

        The global levels read the context of each position; the levels of the text read only the earlier
        positions of ``ids`` itself.
        """
        ids = np.asarray(ids, np.int64)
        positions = np.arange(len(ids)) if positions is None else np.asarray(positions)
        a = self.a if a is None else a
        b = self.b if b is None else b
        keys, valid = context_keys(ids, max(self.depth, self.document), self.V)
        y = ids[positions]
        p = np.full(len(positions), 1./self.V)
        for j, table in enumerate(self.levels):
            C, N, D = table.lookup(keys[j][positions], y)
            ok = valid[j][positions]
            C, N, D = C*ok, N*ok, D*ok
            weight = a[j]*np.maximum(D, 1)
            p = (N+weight*p)/(C+weight)
        for j in range(1, self.document+1):
            C, N, D = document_counts(keys[j], valid[j], ids)
            C, N, D = C[positions], N[positions], D[positions]
            weight = b[j]*np.maximum(D, 1)
            p = (N+weight*p)/(C+weight)
        return np.log(p)

    def judge(self, texts, positions=None):
        """Choose the weights a_j, b_j: the values that read best the given unseen texts (two rounds per weight)."""
        positions = positions or [np.arange(1, len(text)) for text in texts]

        def score(a, b):
            return np.mean(np.concatenate([self.logp(text, where, a, b) for text, where in zip(texts, positions)]))
        for j in list(range(self.depth+1))*2:
            values = [score(np.r_[self.a[:j], value, self.a[j+1:]], self.b) for value in GRID]
            self.a[j] = GRID[int(np.argmax(values))]
        for j in list(range(1, self.document+1))*2:
            values = [score(self.a, np.r_[self.b[:j], value, self.b[j+1:]]) for value in GRID]
            self.b[j] = GRID[int(np.argmax(values))]
        return -score(self.a, self.b)

    def distribution(self, ids):
        """Probabilities of the next token after ``ids`` (the whole vocabulary)."""
        ids = np.asarray(ids, np.int64)
        tail = np.r_[ids[-max(self.depth, self.document):], 0]           # a position after the text
        keys, valid = context_keys(tail, max(self.depth, self.document), self.V)
        last = len(tail)-1
        p = np.full(self.V+1, 1./self.V); p[self.V] = 0.
        for j, table in enumerate(self.levels):
            if not valid[j][last]:
                break
            index = table.find(keys[j][last:last+1])[0]
            if index < 0:
                continue
            tokens, counts = table.followers(index)
            weight = self.a[j]*max(table.D[index], 1)
            p *= weight/(table.C[index]+weight)
            p[tokens] += counts/(table.C[index]+weight)
        if self.document:
            full_keys, full_valid = context_keys(np.r_[ids, 0], self.document, self.V)
            here = len(ids)
            for j in range(1, self.document+1):
                if not full_valid[j][here]:
                    break
                earlier = np.flatnonzero(full_valid[j][:here] & (full_keys[j][:here] == full_keys[j][here]))
                if not len(earlier):
                    continue
                tokens, counts = np.unique(ids[earlier], return_counts=True)
                weight = self.b[j]*len(tokens)
                p *= weight/(len(earlier)+weight)
                p[tokens] += counts/(len(earlier)+weight)
        return p[:self.V]

    def generate(self, ids, new_tokens, stop=(), temperature=0., seed=0):
        """Continue a text: the most probable token (temperature 0) or one drawn from the reading."""
        rng = np.random.default_rng(seed)
        ids = list(np.asarray(ids, np.int64))
        produced = []
        for _ in range(new_tokens):
            p = self.distribution(np.asarray(ids))
            if temperature > 0:
                q = np.exp(np.log(np.maximum(p, 1e-300))/temperature)
                token = int(rng.choice(len(q), p=q/q.sum()))
            else:
                token = int(np.argmax(p))
            if token in stop:
                break
            ids.append(token); produced.append(token)
        return produced

    # ---- saving ---------------------------------------------------------------------------
    def arrays(self):
        self.settle()
        out = dict(a=self.a, b=self.b, shape=np.array([self.V, self.depth, self.document, self.tokens, self.generalized], np.int64))
        for j, table in enumerate(self.levels):
            out.update({f'level{j}.keys': table.keys, f'level{j}.C': table.C, f'level{j}.cells': table.cells,
                        f'level{j}.counts': table.counts})
        return out

    @classmethod
    def from_arrays(cls, arrays, budget_bytes=None):
        V, depth, document, tokens, generalized = (int(x) for x in arrays['shape'])
        model = cls(V, depth, document, budget_bytes)
        model.a, model.b, model.tokens, model.generalized = np.array(arrays['a']), np.array(arrays['b']), tokens, generalized
        model.levels = [Table(arrays[f'level{j}.keys'], arrays[f'level{j}.C'], arrays[f'level{j}.cells'],
                              arrays[f'level{j}.counts'], V) for j in range(depth+1)]
        return model


def document_counts(keys, valid, ids):
    """For every position: C, N(own token), D of its context among the EARLIER positions of the same text only."""
    n = len(ids)
    position = np.arange(n)

    def earlier(group):
        order = np.lexsort((position, group))
        g = group[order]
        start = np.r_[True, g[1:] != g[:-1]]
        first = np.flatnonzero(start)[np.cumsum(start)-1]
        out = np.empty(n, np.int64)
        out[order] = np.arange(n)-first
        return out
    with np.errstate(over='ignore'):
        pair = keys*np.uint64(1000003)+(ids+1).astype(np.uint64)  # (context, token) as one key
    C = earlier(keys)
    N = earlier(pair)
    novel = (N == 0).astype(np.int64)
    order = np.lexsort((position, keys))
    g, f = keys[order], novel[order]
    start = np.r_[True, g[1:] != g[:-1]]
    running = np.cumsum(f)
    before = np.r_[0, running][np.flatnonzero(start)][np.cumsum(start)-1]
    D = np.empty(n, np.int64)
    D[order] = running-f-before
    out = np.stack((C, N, D)).astype(float)
    out[:, ~valid] = 0
    return out


def default_budget():
    """Bytes for the statistics: a quarter of the memory of the machine, at most 4 GB."""
    try:
        import os
        total = os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        total = 4 << 30
    return int(min(total//4, 4 << 30))


def save(directory, model, state, identity):
    """A checkpoint in the format every project uses (manifest, sums, state), so pause, resume, best and
    milestones work as for any other project. The sums go in ``trainables.safetensors``; there is no optimizer."""
    import os
    import shutil
    import uuid
    from safetensors.numpy import save_file
    from . import checkpoints
    from .store import atomic_json, digest_file, sync_directory
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['step']:012d}-{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    temporary, final = root/('.'+name+'.tmp'), root/name
    temporary.mkdir()
    try:
        save_file({key: np.ascontiguousarray(value) for key, value in model.arrays().items()}, str(temporary/'trainables.safetensors'))
        (temporary/'optimizer.pt').write_bytes(b'concept-lm: no optimizer\n')
        atomic_json(temporary/'state.json', state)
        (temporary/'rng.pt').write_bytes(b'concept-lm: no random state\n')
        files = ['trainables.safetensors', 'optimizer.pt', 'state.json', 'rng.pt']
        for filename in files:
            with (temporary/filename).open('rb') as stream:
                os.fsync(stream.fileno())
        manifest = dict(format=1, contract=checkpoints.contract(), identity=identity, step=state['step'],
                        sha256={filename: digest_file(temporary/filename) for filename in files})
        atomic_json(temporary/'manifest.json', manifest)
        sync_directory(temporary)
        os.replace(temporary, final)
        sync_directory(root)
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load(directory, expected=None, budget_bytes=None):
    """The model and the state of a checkpoint (verified as every checkpoint)."""
    from safetensors.numpy import load_file
    from . import checkpoints
    checkpoints.verify(directory, expected)
    arrays = load_file(str(Path(directory)/'trainables.safetensors'))
    state = json.loads((Path(directory)/'state.json').read_text())
    return ConceptLM.from_arrays(arrays, budget_bytes), state
