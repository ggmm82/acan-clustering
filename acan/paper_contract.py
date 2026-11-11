"""Reviewed paper/code correspondence. A passing software audit is not a theorem proof."""
PAPER_SHA256 = '314e7cb016b4de2c986b591f2e601ac4e844ee836ff2db621a74efe180320671'
PDF_SHA256 = '1fdd1a5f6e235d291c29364cae2dd1bd138df74db42256bb9ec36dd528f40396'
CONFORMANCE = 'partial'
GAPS = [
    {'id': 'joint-cp-core', 'status': 'qualified',
     'detail': 'cp-joint-v1 shares bounded root-shift relevance, weighted routing, summaries and normalized CP/NLL objective between tabular and LM adapters. This explicit supervised extension omits the paper-specific dual relevance and mass/bilinear routing; backend values/gradients, causality and resume are tested, not general language quality.'},
    {'id': 'backend-initialization', 'status': 'qualified',
     'detail': 'CPU forward/gradient parity is checked with identical parameters; native initializers differ. GPU execution and bitwise cross-backend training equivalence are not certified.'},
    {'id': 'routing', 'status': 'qualified',
     'detail': 'Current legacy-v8 retains mass*blended_similarity/tau. Experimental paper-routing-v1 implements squared distance and two-stage log-mass routing through common NumPy/Torch operators; gradients and the unchanged LM equations are checked. This is not full-model conformance.'},
    {'id': 'objective-and-relevance', 'status': 'different',
     'detail': 'The experimental tabular routing still uses supplied legacy CP/root-shift relevance. The tabular paper also describes dual relevance, an entropy term with unspecified coefficient and feature expansion without a complete selection protocol. These are not invented or claimed as implemented.'},
    {'id': 'omega', 'status': 'different',
     'detail': 'omega is fixed correlation-based preprocessing; no optimized omega parameter or explicit Phi(X) feature expansion.'},
    {'id': 'masking', 'status': 'different',
     'detail': 'Runtime arbitrary-column completion exists; legacy tabular training optimizes CP; cp-joint-v1 adds categorical NLL with targets excluded from routing. Neither is an arbitrary masked reconstruction objective.'},
    {'id': 'pruning', 'status': 'partial',
     'detail': 'Legacy empty hard-assignment rows are removed on export; cp-joint-v1 retains its bounded prototype set. Neither physically prunes training parameter arrays.'},
    {'id': 'complexity', 'status': 'qualified',
     'detail': 'Routing is O(n*k*m) for fixed k/m; coupling preprocessing includes O(n*m^2) work.'},
    {'id': 'theorems', 'status': 'unverified',
     'detail': 'Early stopping ensures bounded iterations, not global Adam convergence. Noise and collapse theorems require independent mathematical review.'},
    {'id': 'benchmark-provenance', 'status': 'different',
     'detail': 'Historical Iris/Zoo/Wine/Wisconsin-style benchmarks generate synthetic samples; they do not load the real UCI datasets.'},
    {'id': 'published-percentages', 'status': 'unverified',
     'detail': 'Selected contract fixtures are regressions, not certification of every published percentage, dataset size or baseline. See docs/PAPER-AUDIT.md.'},
    {'id': 'modalities', 'status': 'out-of-scope',
     'detail': 'No LLM, audio, video or multimodal training is demonstrated by the paper. The separate newly supplied LM paper has an explicitly corrected causal PyTorch port and its own audit; this does not certify that paper’s generalization claims.'},
]
