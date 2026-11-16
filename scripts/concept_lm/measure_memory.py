"""The memory tasks of the small test, with the concept language model (same sequences, same measure)."""
import sys, time, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from model import ConceptLM
from acan.lm import small_test as S
pairs, values, questions = 8, 16, 4
keys = max(values, 2*pairs)
V = keys+values+1+8
def batch(n, filler, generator, varied=False):
    rows, targets = [], []
    for _ in range(n):
        gap = int(torch.randint(0, filler+1, (1,), generator=generator)) if varied else filler
        i, t = S.recall_batch(1, pairs, keys, values, gap, questions, generator)
        rows.append(i[0].numpy()); targets.append(t[0].numpy())
    return rows, targets
for filler, varied, label in ((48, False, 'memoria (risposta a 49 token)'), (176, True, 'memoria lontana (fino a 177 token)')):
    g = torch.Generator().manual_seed(0)
    train_rows, _ = batch(2000, filler, g, varied)                  # as many sequences as the neural test sees in 60 steps
    stream = np.concatenate([np.r_[r, V-1] for r in train_rows])    # sequences one after the other, a separator between
    t = time.time()
    model = ConceptLM(V, depth=4, doc_depth=2, least=1).fit(stream, judged=20000)
    # the weights of the sequence's own concepts, judged on training sequences
    grid = tuple(2.**k for k in range(-5, 9))
    tune_rows, tune_t = batch(200, filler, torch.Generator().manual_seed(1), varied)
    for j in list(range(1, model.doc_depth+1))*2:
        scores = []
        for value in grid:
            trial = model.b.copy(); trial[j] = value
            scores.append(np.mean([model.logp(np.r_[r[:a+1], tt[a]], np.array([a+1]), trial)[0]
                                   for r, tt in zip(tune_rows, tune_t) for a in np.flatnonzero(tt != -100)]))
        model.b[j] = grid[int(np.argmax(scores))]
    seconds = time.time()-t
    test_rows, test_t = batch(512, filler, torch.Generator().manual_seed(10000), False)
    right = []
    for r, tt in zip(test_rows, test_t):
        asked = np.flatnonzero(tt != -100)
        for a in asked:                                             # the symbol most probable after position a
            scores = [model.logp(np.r_[r[:a+1], v], np.array([a+1]))[0] for v in range(keys, keys+values)]
            right.append(keys+int(np.argmax(scores)) == tt[a])
    print(f'{label}: {100*np.mean(right):.1f}% giuste | {seconds:.1f} s | b {model.b.round(2).tolist()}', flush=True)
