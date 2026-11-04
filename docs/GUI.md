# Native ACAN concept workspace

The GUI invokes the existing `AlgebraicClusterLearner` APIs without adding a new
objective, routing rule or generative architecture. This exposes the current
implementation; correspondence to the paper remains **partial**, as described in
[PAPER-AUDIT.md](PAPER-AUDIT.md).

## Workflow

1. Upload Data. Choose categorical columns, excluded columns and target Labels in
   Model; clear Labels for unsupervised discovery.
2. Configure maximum depth, root epochs, learning rate, tau, alpha, minimum split
   samples and supervised purity threshold. Depth 1 uses `nLearn`; greater depths
   use recursive `gLearn`. The latter keeps its native batch of 300. A depth limit
   does not force splitting: purity, variance and sample criteria decide it.
   Child epochs follow native `max(40, parent_epochs // 2)` semantics.
3. Explore the full graph or just the concept hierarchy. All node IDs and child
   links survive download/reload. Select nodes, groups or child links for details.
   Choose a tree, force or concentric layout.
4. Open the concept laboratory. Choose any node to inspect its relevance matrix,
   native centroid rules and support counts. Export all rules as JSON. Relevance
   and dispersion are model statistics, not calibrated probabilities of accuracy.
5. Evaluation uses the held-out 20%, reports accuracy independently per categorical
   target, and MAE/RMSE in original units independently per numeric target.
6. Prediction calls native `predictDataset`, follows child nodes and exports
   `ACAN_path` plus mean native target confidence (`ACAN_confidence`, not calibrated).
   Masked completion calls the existing root-only `predict_masked`; it does not
   claim recursive completion or text/audio/video generation.

Training runs in one owned subprocess per page job. Stop terminates that job and
preserves the previous model. The existing signed cookie plus random page token
isolates browsers and tabs without authentication. A job is limited to 15 minutes;
interactive inputs are limited to 10,000 training rows, 1,000 encoded attributes
and 2 million encoded cells. No durable jobs or multi-worker registry is added.
Native core validation/preprocessing rules still apply; this is a research GUI.

## Regression gates

`tests/test_hierarchy_ui.py` trains an actual multilevel native model, checks JSON
round trips, every rendered edge endpoint in both views, child-edge inspection,
hierarchical predictions and cancellation of the isolated worker. These tests run
in the existing PR/Python and installed Docker suites. `scripts/smoke.py` exercises
HTTP training, reload, prediction and completion. The paper hash, numerical audit,
CPU backend parity and synthetic benchmark receipts remain mandatory release gates.
