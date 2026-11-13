# ACAN clustering

ACAN (Algebraic Concept Attention Network) provides concept discovery and
tabular autocompletion with a NumPy backend and an optional PyTorch backend.
Requires Python 3.9 or newer.

## Il modello

Tabella e linguaggio usano **un solo modello**, il grafo dei concetti (profilo
`concept-graph-v1`): un concetto è un nodo che riassume con somme le
osservazioni che gli appartengono; riconoscere è massa per somiglianza attesa;
generalizzare unisce i due concetti la cui unione costa meno a chi legge;
giudicare usa solo osservazioni che il concetto non contiene. Il costo è
lineare nel numero di attributi.

I concetti sono disposti in **livelli**: sotto la radice i concetti generali,
poi i comportamenti particolari. Ogni livello toglie alle osservazioni il
centro del proprio concetto e dà loro la sua scala, come la radice fa con media
e deviazione della tabella; due livelli consecutivi sono uniti da una matrice
densa, quindi un comportamento che compare in più concetti generali è un solo
nodo con più genitori, non un frammento per ramo.

- Riferimento unico, con proposizioni e limiti: [docs/MODEL.md](docs/MODEL.md)
- Misure, compresi gli esiti negativi: [docs/MODEL-RESULTS.md](docs/MODEL-RESULTS.md)
- Affermazioni dei paper da non citare: [docs/ERRATA.md](docs/ERRATA.md)

Gli altri documenti in `docs/` sono la storia di come ci si è arrivati. I
profili precedenti (`cp-v9`, `cp-joint-v1` e più vecchi) restano caricabili.
Gli attributi che variano insieme sono descritti insieme solo nel livello di
lavoro, a ricerca finita e se giudicato migliore: Letter 97,9%. Limiti aperti: nel
linguaggio lo strato compresso non eguaglia l'attenzione densa senza
addestramento ed è più lento, e i livelli con comportamenti condivisi, misurati
sulle chiavi di un modello preaddestrato, non migliorano la lettura.

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
Il nuovo profilo GUI `cp-joint-v1` unifica tabulare e generativo: le formule
complete e le misure della proposta sono in [CP-SELECTION.md](docs/CP-SELECTION.md) e nella sezione
[Unico nucleo CP](#unico-nucleo-cp-per-modello-interattivo-e-generativo). Le
equazioni legacy seguenti descrivono il profilo conservato per compatibilità.

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

### Routing del paper e routing legacy

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

**Il profilo `legacy-v8` non esegue integralmente queste formule.** In
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

### Obiettivo del profilo legacy

Il forward combina il centroide da attenzione e il prototipo parametrico:
$G^{eff}=0.5G^{post}+0.5\sigma(\theta_G)$. La rilevanza implementata confronta
$G^{eff}$ con la media globale `Root`, normalizzando per l'intervallo
`RootMin`–`RootMax` (e moltiplicando per $\omega$ nel ramo metrico).
Posto $s_i=\sum_jR_{ij}$, il profilo legacy massimizza, con protezioni numeriche:

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

1. Il training iniziale apprende un nodo con `nLearn`: assegnamenti, prototipi,
   rilevanze e obiettivo del profilo selezionato. `cp-joint-v1` ottimizza NLL + CP
   estesa e conserva soltanto il supporto positivo appreso; il legacy ottimizza CP e rimuove gli
   slot vuoti dall’export. Le matrici di training non vengono fisicamente ridotte.
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

Evaluation usa **il dataset dell'utente (anche Excel)**: holdout 20% con seed 42, stratificato quando possibile, ricavato automaticamente dai dati già caricati
in Dati, senza altri CSV. Confronta ACAN con maggioranza/media del training, KNN, CART, Random Forest, Extra Trees,
HistGradientBoosting, XGBoost e CatBoost, usando
le stesse righe osservate per ogni target. Imputazione, schema categorico e scale
delle baseline sono appresi solo sul training. Per classificazione mostra
accuracy, balanced accuracy, F1 macro e matrice di confusione; per regressione MAE, RMSE, $R^2$ e
reale/predetto. Per modelli senza target nasconde celle note del proprio holdout
(seed 42, massimo 100 righe) e misura la ricostruzione sulle sole celle nascoste.

I nuovi modelli GUI conservano nel JSON lo schema delle categorie e l’imputazione
appresi sul training, usati anche in inferenza. Il target suggerito e le esclusioni
delle colonne restano modificabili; la modalità classificazione/regressione può
essere scelta nelle impostazioni avanzate. Verifica misurata sui cinque dataset
forniti: [risultati e limiti](docs/DATASET-EVALUATION.md).

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

The Generative tab also provides a **guided public-corpus experiment**: choose
Dante (Italian) or Tiny Shakespeare (English) and a 20/200-step budget. One
explicit action downloads and verifies the source, prepares a separate persistent
project and starts training; pause/resume and checkpoint generation remain
available. These are bounded mechanics experiments, not certified assistant
training. See [source provenance, budgets and limitations](docs/GENERATIVE.md#guided-public-corpus-experiments).

The PR #6 controlled laboratory adds best-checkpoint preservation and early
stopping for new projects, optional frozen-teacher distillation, disjoint
document splits, WikiText-2 official splits, a matched dense-LoRA control and
explicit quality/cache/speed comparisons. Model deltas and architecture can be
downloaded; own-project diagnostics and redacted logs remain available in the
web UI. The 100% ACAN profile is experimental and CPU contract-tested, not a
claim of general-language quality. [Step-by-step workflow and limits](docs/CONTROLLED-LM.md).


La verifica dei due paper, gli operatori comuni e il confronto sui dataset
caricati sono descritti in [SCIENTIFIC-ALIGNMENT.md](docs/SCIENTIFIC-ALIGNMENT.md).
Il routing sperimentale del paper non sostituisce automaticamente il profilo
attuale: sui dati reali non emerge un vincitore generale. I risultati completi,
compresi regressioni e baseline, sono conservati senza pubblicare le righe dei
file utente.

### Unico nucleo CP per modello interattivo e generativo

I nuovi training web usano un’unica proposta `cp-joint-v1`, identificata nei checkpoint da
`pair-supported-cp-v1`. Tabulare e generativo condividono routing, masse sul simplesso,
statistiche, CP estesa e obiettivo NLL + lambda × (1 − CP*). Le masse esattamente zero
definiscono gli slot inattivi; non vengono fusi prototipi né imposto un numero di gruppi.
La scheda **Matematica** spiega prototipi, rilevanza, appartenenza, CP e gerarchia con
esempi e limiti. Senza target si ottimizza la sola CP; con target la NLL introduce
supervisione. Questi due compiti non devono essere confusi.
I vecchi checkpoint e le API storiche rimangono compatibili; nuovi rami non mescolano formule diverse.

**La CP ora esclude l'auto-sostegno dei singoli esempi.** La frazione di coppie
provenienti da osservazioni diverse pesa la qualità del concetto, senza K imposto
né regole per dataset. La stessa formula è usata nel tabulare e nei prefissi del
linguaggio. Gioco completo, **101 righe e nessun target: 4 gruppi**, compresa la
riga con valore mancante. Il clustering web usa l'intera popolazione caricata;
lo split 80/20 resta per la classificazione.

Letter gerarchico: **92,325%** completo, **91,05%** nel campione web, con 3.346 e
1.387 prototipi. La versione precedente misurava 92,70% e 89,85%: migliorano
il campione web e la dimensione del modello, mentre il completo perde 0,375 punti.
Il 95,78% storico resta non recuperato. I confronti sugli altri file restano misti.
La qualità linguistica su LLM preaddestrati non è ancora dimostrata.

Dettagli della correzione e ricevute: [CP-PAIR-SUPPORT.md](docs/CP-PAIR-SUPPORT.md).
I modelli precedenti restano utilizzabili in inferenza; per addestrare nuovi rami
con la nuova CP occorre un nuovo training coerente.

Formule, precisazioni dei teoremi originali, risultati completi, limiti e istruzioni per
provare la PR dal web: [CP-SELECTION.md](docs/CP-SELECTION.md).
Il [piano di semplificazione dell’interfaccia](docs/NEXT-UI-PR.md) è destinato alla
prossima PR, in un run successivo; comprende classi numeriche e storico generativo
navigabile oltre i 500 punti attualmente restituiti dal server.
I protocolli e i risultati della versione precedente sono conservati in
[JOINT-CP.md](docs/JOINT-CP.md) e [LETTER-DIAGNOSIS.md](docs/LETTER-DIAGNOSIS.md).


### Grafo leggibile e navigazione dei concetti

La vista **Concetti e gruppi** mostra un concetto alla volta, con **12 gruppi per
pagina** e schede con etichette leggibili. Il selettore del concetto e i pulsanti
**Radice / Sali al padre** permettono di percorrere la gerarchia. Un doppio clic
su un gruppo con bordo blu apre il figlio già addestrato; sugli altri gruppi
richiede un approfondimento. La vista **Solo gerarchia** conserva la panoramica.
Sono disponibili zoom, centratura e schermo intero.

Il conteggio globale indica i **prototipi complessivi**, non certifica altrettanti
concetti distinti. Per ogni nodo sono distinti i gruppi con record assegnati e i
prototipi senza assegnazioni hard, consultabili con l'apposita opzione. Questi
ultimi possono ancora contribuire alle previsioni soft. Il rendering non elimina
o accorpa i prototipi del modello. Nel nuovo training la selezione avviene tramite le masse
del nucleo condiviso, come descritto in [CP-SELECTION.md](docs/CP-SELECTION.md); il conteggio
non certifica un numero semanticamente corretto di concetti su qualunque dataset.

### Usare il modello e leggere il training

Nella scheda Model, lasciare il target vuoto per scoprire gruppi oppure scegliere
una colonna da prevedere. La scelta visibile distingue classi (anche codici numerici)
e quantità continue: per un prezzo si usa regressione con media pesata nei prototipi,
per i codici selezionare Classi. Automatico è un suggerimento da verificare.
Cambiare il compito richiede riaddestramento, non converte i JSON esistenti. I controlli
inattivi sono nascosti o disabilitati; i parametri effettivi hanno spiegazioni.
Il riepilogo del grafo distingue i prototipi dell'albero dai gruppi del nodo.

Nel generativo, **Intero percorso** mostra la panoramica dall'inizio alla fine;
**Esamina una finestra di 500 passi** consente di leggere i punti esatti, inserire
un passo iniziale e navigare avanti/indietro. Il polling conserva lo zoom. Per
serie lunghe la panoramica è campionata e non garantisce di mostrare ogni picco:
la finestra contiene le misure esatte dell'intervallo. Frontend e worker devono
includere questo aggiornamento; i progetti esistenti non vengono riaddestrati.

La lettura numerica continua usa le appartenenze del training per stimare una
media per prototipo. Su autoPrice (127 training,32 test,seed42), la verifica
misura MAE1797,48,RMSE2522,45 e R²0,6424; baseline della media training RMSE4315,01.
È un controllo di sviluppo su uno split, non una garanzia generale.

Il cambio fra progetti generativi carica soltanto la vista selezionata. Le
panoramiche immutate sono riutilizzate e il grafico riceve solo i campi necessari;
lo zoom è conservato per progetto. Per questa ottimizzazione aggiornare anche il
worker, oltre al frontend. I training in corso non vengono riavviati.
