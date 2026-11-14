# Scheda Modello e Valutazione

L'interfaccia usa il dataset caricato una sola volta in **Dati**; la valutazione
non chiede un altro CSV. Il modello è il grafo dei concetti
([MODEL.md](MODEL.md)).

1. In **Modello** scegliere la colonna da prevedere (vuota per scoprire gruppi
   senza target) e avviare la scoperta dei concetti. Il tipo di target è
   suggerito automaticamente: testo e codici interi con 2–20 valori diventano
   classi; se il suggerimento è sbagliato si sceglie classificazione o
   regressione nelle **Impostazioni avanzate**. Target e colonne escluse
   (identificativi, testo con troppi valori) restano modificabili; i target
   ignoti (`?` o vuoti) sono esclusi e contati.
2. Il grafo mostra la radice, i livelli di concetti generali (righe azzurre) e
   i gruppi che fanno le previsioni. Le frecce blu sono le celle della matrice
   fra due livelli: una freccia da un concetto generale a un gruppo significa
   che più del 10% delle righe del gruppo appartiene a quel concetto; un
   gruppo con più frecce in entrata è un comportamento comune a più concetti
   generali. Le linee tratteggiate arancioni collegano gruppi fra cui le righe
   esitano. I livelli con più di 12 concetti sono riassunti nel testo sopra il
   grafo. I gruppi sono mostrati 12 per pagina. Un clic su un concetto di
   livello, come su un gruppo, mostra il suo prototipo: valori tipici e
   rilevanza di ogni attributo, e da quali concetti del livello sopra discende.
   Nel modello salvato (JSON) i livelli sono in
   `clusterModel → <nodo> → routing_parameters → levels`, con gli stessi
   prototipi (`groups`, `relevance`), la massa e la matrice `parents`.
3. Un doppio clic su un gruppo lo approfondisce a mano: si addestra un nodo
   figlio sulle sole righe del gruppo, senza riaddestrare il resto (c'è anche
   un pulsante, in alternativa al doppio clic). **Valutazione** ricalcola le
   metriche e la differenza prima/dopo sullo stesso 20% di righe tenute da
   parte (seme 42); **Rimuovi approfondimento selezionato** elimina il ramo e i
   suoi discendenti. La radice non si può eliminare.
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

The rest of this page keeps the original notes in English.

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
and numerical missing values are filled with training means and marked; the
concept graph treats a marked cell as missing, not as observed.
Retrain older JSON models to acquire this schema. Resource checks run before
one-hot allocation and report which columns cause expansion. Current measurements: [MODEL-RESULTS.md](MODEL-RESULTS.md).

## Filter state regression

The browser must send the same exclusions displayed in Model. Exclusions now
use explicit checkboxes instead of the stale multi-select control. Data upload now
sends filter options and their selected values together; model JSON loading
keeps filter options based on the uploaded table rather than its target-only
dropdown. Use **Ripristina filtri suggeriti** to restore defaults after clearing.
The CI gate includes real Chromium tests at mobile and desktop widths, including
upload, model export/reload and restore, and asserts the actual training request
contains all displayed exclusions. The original Titanic failure was reproduced
on the deployed app: four visually selected filters were sent as an empty list.
A public reference with identical table values was used; uploaded user data was
not sent to the public app. Corrected local browser training completes with all
four exclusions in the request.
