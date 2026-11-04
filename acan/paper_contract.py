"""Reviewed paper/code correspondence. A passing software audit is not a theorem proof."""
PAPER_SHA256 = '314e7cb016b4de2c986b591f2e601ac4e844ee836ff2db621a74efe180320671'
CONFORMANCE = 'partial'
GAPS = [
    {'id': 'backend-initialization', 'status': 'qualified',
     'detail': 'CPU forward/gradient parity is checked with identical parameters; native initializers differ. GPU execution and bitwise cross-backend training equivalence are not certified.'},
    {'id': 'routing', 'status': 'different',
     'detail': 'Tabular code uses mass*blended_similarity/tau; paper Eq. v8_routing uses gamma*log(mass)-beta*distance_squared.'},
    {'id': 'omega', 'status': 'different',
     'detail': 'omega is fixed correlation-based preprocessing; no optimized omega parameter or explicit Phi(X) feature expansion.'},
    {'id': 'masking', 'status': 'different',
     'detail': 'Runtime arbitrary-column completion exists; tabular training optimizes CP, not an explicit masked reconstruction loss.'},
    {'id': 'pruning', 'status': 'partial',
     'detail': 'Empty hard-assignment rows are removed from the exported model, not physically pruned from the training parameter arrays.'},
    {'id': 'complexity', 'status': 'qualified',
     'detail': 'Routing is O(n*k*m) for fixed k/m; coupling preprocessing includes O(n*m^2) work.'},
    {'id': 'theorems', 'status': 'unverified',
     'detail': 'Early stopping ensures bounded iterations, not global Adam convergence. Noise and collapse theorems require independent mathematical review.'},
    {'id': 'benchmark-provenance', 'status': 'different',
     'detail': 'Historical Iris/Zoo/Wine/Wisconsin-style benchmarks generate synthetic samples; they do not load the real UCI datasets.'},
    {'id': 'published-percentages', 'status': 'unverified',
     'detail': 'Selected contract fixtures are regressions, not certification of every published percentage, dataset size or baseline. See docs/PAPER-AUDIT.md.'},
    {'id': 'modalities', 'status': 'out-of-scope',
     'detail': 'No LLM, audio, video or multimodal training is demonstrated by the paper. The character language model is a separately evaluated extension.'},
]
