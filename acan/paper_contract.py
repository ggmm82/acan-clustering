"""Reviewed paper/code correspondence. A passing software audit is not a theorem proof."""
PAPER_SHA256 = '314e7cb016b4de2c986b591f2e601ac4e844ee836ff2db621a74efe180320671'
PDF_SHA256 = '1fdd1a5f6e235d291c29364cae2dd1bd138df74db42256bb9ec36dd528f40396'
CONFORMANCE = 'partial'
GAPS = [
    {'id': 'predictive-concepts-core', 'status': 'different',
     'detail': 'concept-graph-v1 (one model for new tabular models and new generative projects, docs/MODEL.md) replaces the paper CP as objective: concepts are additive sufficient statistics, generalized bottom-up where the reader loses least (reads received times radius) and, on tables, selected by leave-one-out predictive log-probability; the reported CP is explained variance. It is derived in docs/MATHEMATICAL-FOUNDATION.md and measured in docs/CP-V9-RESULTS.md, not taken from the paper. The same functions (acan/concept_algebra.py) define the attention layer, exact without compression. Cost is linear in the attributes: no covariance. Measures and remaining gaps in docs/MODEL-RESULTS.md.'},
    {'id': 'joint-cp-core', 'status': 'qualified',
     'detail': 'cp-joint-v1 shares bounded root-shift relevance, learned simplex masses, routing, summaries and pair-supported-cp-v1/NLL between tabular and LM adapters. The extension adds explicit overlap exclusion, distinct-observation support and probability affinity; it is not a proof of the paper theorems or general language quality. It omits paper-specific dual relevance and mass/bilinear routing. Values, derivatives, causality and resume are tested.'},
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
     'detail': 'The new joint model learns simplex masses. Exact zero support is removed from saved tabular arrays and masked in LM slots; no similarity-based merge is used. Occupied support is not a guarantee of semantic concept count or globally optimal cardinality.'},
    {'id': 'complexity', 'status': 'qualified',
     'detail': 'Routing is O(n*k*m); coupling preprocessing includes O(n*m^2). Exclusive CP adds O(k^2*m) per training evaluation and line search overhead. LM inference skips this pairwise training cost and retains fixed-prefix-state memory. No universal throughput improvement is claimed.'},
    {'id': 'theorems', 'status': 'unverified',
     'detail': 'Early stopping ensures bounded iterations, not global Adam convergence. The review in docs/MATHEMATICAL-FOUNDATION.md finds the noise proof incorrect and the collapse statement false at gamma=1; see docs/ERRATA.md.'},
    {'id': 'benchmark-provenance', 'status': 'different',
     'detail': 'Historical Iris/Zoo/Wine/Wisconsin-style benchmarks generate synthetic samples; they do not load the real UCI datasets.'},
    {'id': 'published-percentages', 'status': 'unverified',
     'detail': 'Selected contract fixtures are regressions, not certification of every published percentage, dataset size or baseline. See docs/PAPER-AUDIT.md.'},
    {'id': 'modalities', 'status': 'out-of-scope',
     'detail': 'No LLM, audio, video or multimodal training is demonstrated by the paper. The separate newly supplied LM paper has an explicitly corrected causal PyTorch port and its own audit; this does not certify that paper’s generalization claims.'},
]
