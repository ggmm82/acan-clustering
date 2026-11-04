# ACAN clustering

ACAN (Algebraic Concept Attention Network) provides concept discovery and
tabular autocompletion with a NumPy backend and an optional PyTorch backend.
Requires Python 3.9 or newer.

## Development setup

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Install the optional PyTorch backend with `python -m pip install -e '.[torch]'`.

## Interactive application

```sh
python -m pip install -e '.[ui]'
python -m acan.vis.render
```

Open `http://localhost:4000` in your browser.

1. Upload a dataset in **Data**, select targets in **Model**, and click **Learn Model**.
2. Start with one concept level. Double-click a group to learn one additional
   level on demand; remove branches that do not help. Advanced parameters are
   collapsed. **Evaluation** benchmarks your own data against baselines and the
   model before the change. See [GUI workflow](docs/GUI.md).
3. Use **Download Model** to save a JSON model. Later, use **Load Model JSON**
   in **Model** to restore it, including its target labels.
4. In **Inference**, upload a new UTF-8 CSV and choose **Predici Labels** or
   **Completa campi vuoti**. Click **Esegui inferenza**, inspect the preview,
   and download the complete CSV. No training occurs in this tab.

Prediction ignores any supplied target values and predicts the saved model's
labels by traversing the native hierarchy, with path and confidence columns.
Completion uses the native root-only masked predictor and fills empty cells in columns present in the trained schema,
preserving observed values; include the columns you want completed in the CSV.
Each incomplete row needs at least one known attribute. A model without target
labels can still be used in completion mode.

Training jobs are isolated by signed browser session and a distinct random page
identifier. Tabs in the same browser can train on different datasets independently;
a second concurrent training in the same page is rejected. Use one server process: jobs are kept
in memory, are lost on restart, and completed jobs expire after one hour when
new jobs start. Separate browser profiles have separate sessions; tabs sharing
cookies still have distinct page identifiers. Refreshing starts a new page scope
and does not reattach to an earlier job. This is a local research application, without accounts
or durable job storage. Inference uses the model held in the current page.

The default development command enables Dash debugging. For local runs without
the debugger/reloader, use:

```sh
python -c "from acan.vis.render import app; app.run(debug=False, port=4000)"
```

## Contents

The [paper audit](docs/PAPER-AUDIT.md) records partial correspondence, corrected
numerical gradients and outstanding scientific claims. Every release now requires
the audit, full synthetic benchmark receipts and Docker HTTP checks.

For Docker deployment and the PR → CI → prerelease → stable release process,
see [Controlled Docker releases](docs/RELEASING.md) and [Contributing](CONTRIBUTING.md).
For the VPS deployment, use the sealed `compose.hostinger.yaml` release asset and
follow [Hostinger deployment](docs/HOSTINGER.md).

- `acan/`: library source
- `tests/`: unit tests
- `benchmarks/`: experimental benchmark script
- `experiments/`: archived results and figures
- `paper/`: paper source

Run the historical synthetic suite without changing the archived paper outputs:

```sh
OPENBLAS_NUM_THREADS=1 python benchmarks/run_benchmarks.py --output-dir .runs/paper-benchmark
```

The UCI-like cases in that script are synthetic, not the real UCI datasets.

## Come leggere ACAN: matematica, algoritmo e limiti verificati

Il riferimento scientifico è il [PDF fornito](paper/Algebraic_Concept_Attention_Scientific_Paper.pdf),
in particolare §3 e Tabella 1. Le formule del paper e quelle oggi eseguite dal
codice sono distinte qui sotto: la GUI espone il core esistente e non certifica
una corrispondenza completa. L'[audit](docs/PAPER-AUDIT.md) documenta le differenze.

### Dalle righe ai concetti

Dopo codifica one-hot delle categorie e normalizzazione min–max, il training usa
una matrice $X\in[0,1]^{n\times m}$. Ogni riga è un campione; ogni colonna è un
attributo codificato. I concetti sono prototipi $G\in[0,1]^{k\times m}$.
La matrice $T\in[0,1]^{k\times n}$ descrive l'appartenenza morbida dei campioni:

$$
\sum_{i=1}^k T_{it}=1,\qquad
C_i=\sum_{t=1}^n T_{it},\qquad
F_i=C_i/n,\qquad
G^{post}_{ij}=\frac{\sum_t T_{it}X_{tj}}{C_i+\varepsilon}.
$$

La softmax compete **tra concetti per ogni campione**. $C_i$ è la massa morbida,
$F_i$ la frequenza; i supporti interi esportati usano invece l'assegnamento hard.
Il centroide descrive i valori tipici del gruppo. La rilevanza $R_{ij}$ indica
quanto l'attributo $j$ caratterizza il concetto $i$: non è una probabilità di
predizione corretta e non sostituisce la validazione.

### Routing del paper e routing attualmente implementato

Il paper, Eq. (4), propone distanza quadratica pesata e prior logaritmico di massa:

$$
D^2(i,t)=\frac1{m'}\sum_j\widetilde R_{ij}(X'_{tj}-G_{ij})^2,\qquad
T_{it}=\frac{\exp(\gamma\log(C_i+\varepsilon)-\beta D^2(i,t))}
{\sum_\ell\exp(\gamma\log(C_\ell+\varepsilon)-\beta D^2(\ell,t))}.
$$

Qui $X'$ indica lo spazio relazionale espanso descritto dal paper e
$\widetilde R$ le rilevanze affilate. Tabella 1 descrive
$R=0.55R_{shift}+0.45R_{tight}$ e
$\mathcal L=-CP(T,R,F)+\lambda_H H(T)$.

**Il core corrente non esegue integralmente queste formule.** In
[`network.py`](acan/core/network.py), per `alpha > 0`, costruisce invece
una similarità mista tra prodotto scalare pesato e distanza:

$$
S_{it}=(1-\alpha)\frac{\langle G_i\odot\omega,X_t\rangle}
{\max(\sum_j\omega_j,1)}
+\alpha\left(1-\sqrt{\sum_j w_{ij}(X_{tj}-G_{ij})^2+\varepsilon}\right),
\qquad T_{it}=\operatorname{softmax}_i(C_i^{prior}S_{it}/\tau).
$$

I pesi $w$ sono normalizzati a partire dalle rilevanze elevate alla potenza
`gamma`, con un piccolo termine in $\omega$. Nel codice `alpha` miscela le due
similarità; non va confuso con l'esponente della notazione del paper. Con
`alpha=0` si usa il ramo dot-product senza questa miscela. $\omega$ è calcolato
dai dati e rimane fisso: non è un parametro ottimizzato e non realizza da solo
l'espansione esplicita $\Phi(X)$ del paper.

### Obiettivo realmente ottimizzato

Il forward combina il centroide da attenzione e il prototipo parametrico:
$G^{eff}=0.5G^{post}+0.5\sigma(\theta_G)$. La rilevanza implementata confronta
$G^{eff}$ con la media globale `Root`, normalizzando per l'intervallo
`RootMin`–`RootMax` (e moltiplicando per $\omega$ nel ramo metrico).
Posto $s_i=\sum_jR_{ij}$, il codice massimizza, con protezioni numeriche:

$$
CP=\sqrt{\sum_i\frac{F_i}{s_i/m}\sum_j R_{ij}^2},\qquad
\mathcal L_{codice}=-CP.
$$

Adam aggiorna i parametri; l'early stopping conserva lo stato con CP migliore.
**CP non è accuracy**, non è una loss di ricostruzione mascherata e non prova
che un ulteriore livello migliori le predizioni. Non viene aggiunto il termine
entropico della Tabella 1. L'audit verifica gradienti numerici e parità
NumPy/PyTorch a parametri uguali; non dimostra i teoremi del paper.

### Algoritmo e approfondimento su richiesta

1. Il training iniziale apprende un singolo nodo con `nLearn`: assegnamenti,
   centroidi, rilevanze, ottimizzazione di CP e rimozione degli slot vuoti dal
   modello esportato. Le matrici di training non vengono fisicamente ridotte.
2. Un doppio clic su **un gruppo** seleziona i suoi soli campioni di training
   tramite `splitDataset`. Un nuovo nodo è appreso con `gLearn(max_depth=1)`,
   mantenendo la normalizzazione globale. Si aggiunge il collegamento al gruppo:
   non si duplica la radice e non si riaddestra il resto del grafo.
3. La predizione supervisionata usa `predictDataset` e attraversa i figli con
   il routing nativo. Evaluation ricalcola metriche e delta prima/dopo sullo
   stesso holdout; si può rimuovere il ramo, inclusi i suoi discendenti.
4. Il completamento mascherato usa `predict_masked`, che oggi lavora sulla
   radice. Aggiungere figli non ne modifica i risultati: la GUI lo dichiara.

La complessità del routing è $O(nkm)$ per un nodo. Questo diventa lineare in
$n$ solo mantenendo $k,m$ fissati; il preprocessing di $\omega$ comprende
operazioni $O(nm^2)$. Un ramo aggiuntivo aumenta capacità, memoria e costo di
inferenza; il beneficio va misurato, non dedotto dalla profondità.

### Benchmark sui propri dati

Il benchmark parte **solo su richiesta**, premendo **Confronta con gli altri
modelli**. Il training e i clic sul grafo non lo avviano. Usa un processo a bassa
priorità, un solo thread per modello, un benchmark alla volta sul server,
arresto disponibile e limite di 15 minuti. Le foreste usano 200 alberi e i
boosting 100 iterazioni: budget fissi, senza tuning automatico.

Evaluation usa **il CSV dell'utente**: holdout 20% con seed 42, ricavato automaticamente dai dati già caricati
in Dati, senza altri CSV. Confronta ACAN con maggioranza/media del training, KNN, CART, Random Forest, Extra Trees,
HistGradientBoosting, XGBoost e CatBoost, usando
le stesse righe osservate per ogni target. Imputazione, schema categorico e scale
delle baseline sono appresi solo sul training. Per classificazione mostra
accuracy, balanced accuracy, F1 macro e matrice di confusione; per regressione MAE, RMSE, $R^2$ e
reale/predetto. Per modelli senza target nasconde celle note del proprio holdout
(seed 42, massimo 100 righe) e misura la ricostruzione sulle sole celle nascoste.

Dopo aver scelto i rami guardando l'holdout, esso è **validazione**, non un test
indipendente. Il confronto dei rami è uno strumento di selezione, non una
certificazione della generalizzazione. I benchmark
sintetici storici restano controlli scientifici della CI e non sostituiscono la
valutazione sui propri dati; le percentuali del paper non sono risultati attesi
per un dataset diverso.

## Generative workbench (PR #5)

The **Generative** tab supports the newly supplied ACAN-LM paper through a
separate persistent worker: pretrained SmolLM2 upper-half attention grafting,
LoRA/WSD continued pre-training, streamed text corpora, explicit start,
checkpointed pause/resume and generation from saved snapshots. The original
package had future-token leakage; the user-authorized PyTorch prefix-causal
correction is documented and tested, with the reference source preserved.

See [Generative operation and persistence](docs/GENERATIVE.md) and the
[LM paper/package audit](docs/LM-AUDIT.md). Remaining dense layers retain their
KV cache: 98% whole-model compression and unchanged language quality are not
assumed. The regular web image needs a configured LM worker; it does not launch
training or download weights merely by opening the tab. A sudden interruption
can lose steps since the latest complete checkpoint, not earlier saved learning.

## License

MIT; see [LICENSE](LICENSE).
