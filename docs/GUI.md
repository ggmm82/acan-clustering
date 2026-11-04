# Dataset benchmarks and on-demand concept discovery

The UI uses the dataset uploaded once in **Data**. No separate CSV is requested
for evaluation. It exposes native ACAN APIs without changing its objective or
routing. Paper correspondence remains partial; see [PAPER-AUDIT.md](PAPER-AUDIT.md).

1. In Model, select targets (clear for unsupervised discovery) and click
   **Scopri concetti**. The default initial depth is one. Parameters are collapsed
   under **Impostazioni avanzate**. Automatic target typing treats text and
   integer codes with 2–20 distinct values as classification; choose an explicit
   classification/regression mode there when that suggestion does not match
   the task. Suggested target and excluded identifier/high-cardinality columns
   remain editable. Unknown `?`/blank targets are excluded and counted.
2. Double-click a group to learn one child node on its training samples. The
   model retains one original root; the full graph connects the new groups
   directly below their parent group without rendering an intermediate root.
   A button provides an accessible alternative to double-clicking. Global
   normalization and the parent parameters are retained; only this new node is
   trained with native `gLearn(max_depth=1)`. Existing descendants are retained.
3. Evaluation automatically recomputes ACAN metrics and the before/after delta
   on the identical 20% validation holdout (seed 42). Select a branch and use
   **Rimuovi approfondimento selezionato** to delete it and its descendants.
   Removing a terminal child group removes its level. The global root cannot
   be deleted. Predictions and evaluation update after deletion.
4. **Confronta con gli altri modelli** explicitly starts the benchmark. It never
   starts when training finishes or a graph is clicked. It fits majority/mean,
   KNN, CART, Random Forest, Extra Trees, HistGradientBoosting, XGBoost CPU and
   CatBoost on the same training rows and evaluates the same holdout targets.
   It runs in a cancellable, low-priority owned process, one thread per model;
   at most one benchmark runs on the server, with a 15-minute limit. Sessions
   retain separate datasets/results. Stop terminates only the requesting page's
   benchmark. Model changes clear stale benchmark rankings.
5. Classification reports accuracy, balanced accuracy, F1 macro and confusion
   matrices. Regression reports MAE, RMSE, R² and actual/predicted plots per
   target in original units. Models use fixed CPU budgets, not hyperparameter
   tuning: k≤5, CART depth≤5, forests 200 trees, boosting 100 iterations; boosting
   trees have depth 6 where configured. No best-in-class claim is implied.
   Imputation, one-hot schema and scales are fitted on training rows only for
   baselines. Failed estimators are reported, never silently replaced.
6. Without targets, Evaluation measures root-only masked completion on up to
   100 validation rows, seed 42, at least one known cell per row. Adding children
   cannot change root-only completion. Supervised comparisons require a target.
7. The concept laboratory exposes relevance matrices and native `extractRules`.
   Model, predictions, metrics, protocol/library versions and rules can be saved.
   Prediction paths use native `predictDataset`; confidence is not calibrated.

The holdout becomes a validation set when used to select branches; its reused
scores are not an independent estimate of generalization. CP is a training
objective, not accuracy. Timing tables distinguish previously trained ACAN
inference from newly fitted baseline training; a missing duration is not zero.

Browser cookies plus random page tokens isolate jobs without authentication.
Data stay in memory; reloading starts a new page. A saved model JSON contains
no training records, so resuming training-data benchmarks/refinement requires
training in the current page. Existing interactive input/time limits still apply.

`test_evaluation_refinement.py` checks real modern classifiers and regressors,
training-only transforms, identical comparison rows, root preservation, branch
undo and masked scoring. Hierarchy/session tests and Docker HTTP smoke checks
remain part of CI, along with the frozen paper/PDF audit and synthetic historical
scientific regression suite. Historical benchmarks are separate CI fixtures,
not replacement datasets for the user's benchmark.

The supervised split is stratified when class counts permit. Saved GUI models
include a training-fitted feature schema: absent categorical levels become zero
and numerical missing values use training means and trained missing indicators.
Retrain older JSON models to acquire this schema. Resource checks run before
one-hot allocation and report which columns cause expansion. See the measured
[five-dataset verification](DATASET-EVALUATION.md).
