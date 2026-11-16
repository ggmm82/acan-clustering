"""Same protocol as the neural small test: pieces of 256 characters of unseen text, positions 128..255 measured,
the document concepts built only from the piece itself."""
import sys, time, numpy as np, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from model import ConceptLM
from acan.lm import small_test as S
text = open('.data/wikitext2-test.txt', encoding='utf-8', errors='ignore').read()[:1_500_000]
common = sorted(set(text), key=lambda c: -text.count(c))[:200]; index = {c: i for i, c in enumerate(common)}
V = len(common)+1
data = np.array([index.get(c, len(common)) for c in text], np.int64)
split = int(.9*len(data)); train, held = data[:split], data[split:]
depth, doc, block = int(sys.argv[1]), int(sys.argv[2]), 256
t = time.time()
model = ConceptLM(V, depth=depth, doc_depth=doc, least=2).fit(train[:-100000])
if doc:   # the weights of the piece's own concepts, judged on pieces of unseen training text
    tune = train[-100000:]
    pieces = [tune[s:s+block] for s in range(0, len(tune)-block, block)][:200]
    grid = tuple(2.**k for k in range(-5, 9))
    for j in list(range(1, doc+1))*2:
        scores = []
        for value in grid:
            trial = model.b.copy(); trial[j] = value
            scores.append(np.mean([model.logp(p, np.arange(block//2, block), trial).mean() for p in pieces]))
        model.b[j] = grid[int(np.argmax(scores))]
seconds = time.time()-t
# exactly the pieces the neural test measures: 8 batches of 16 pieces drawn with seed 20000
g = torch.Generator().manual_seed(20000)
total = []
for _ in range(8):
    ids, targets = S.text_batch(torch.tensor(held), 16, block, g)
    for row_ids, row_t in zip(ids, targets):
        piece = np.r_[row_ids.numpy(), row_t.numpy()[-1:]]           # the piece and the symbol after it
        total.append(model.logp(piece, np.arange(block//2+1, block+1)))
bits = -np.concatenate(total).mean()/np.log(2)
print(f'contesto {depth} | livelli del pezzo {doc} | concetti {model.concepts():,} | bit/carattere (protocollo della rete) {bits:.3f} | {seconds:.1f} s', flush=True)
