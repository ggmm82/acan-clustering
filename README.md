# ACAN clustering

ACAN (Algebraic Concept Attention Network) scopre concetti in una tabella, li
usa per prevedere e completare valori mancanti, e studia lo stesso meccanismo
come strato di attenzione di un modello linguistico. Backend NumPy, PyTorch
opzionale. Richiede Python 3.9 o successivo.

## Il modello

Tabella e linguaggio usano **un solo modello**, il grafo dei concetti (profilo
`concept-graph-v1`): un concetto è un nodo che riassume con somme le
osservazioni che gli appartengono; riconoscere è massa per somiglianza attesa;
generalizzare unisce i due concetti la cui unione costa meno a chi legge;
giudicare usa solo osservazioni che il concetto non contiene. La ricerca dei
concetti è lineare nel numero di attributi.

I concetti sono disposti in **livelli**: sotto la radice i concetti generali,
poi i comportamenti particolari. Ogni livello toglie alle osservazioni il
centro del proprio concetto e dà loro la sua scala, come la radice fa con media
e deviazione della tabella; due livelli consecutivi sono uniti da una matrice
densa, quindi un comportamento che compare in più concetti generali è un solo
nodo con più genitori, non un frammento per ramo. A ricerca finita, il livello
che fa le previsioni può descrivere insieme gli attributi che variano insieme,
se così legge meglio righe non viste.

- Riferimento unico, con proposizioni e limiti: [docs/MODEL.md](docs/MODEL.md)
- Misure, compresi gli esiti negativi: [docs/MODEL-RESULTS.md](docs/MODEL-RESULTS.md)
- Affermazioni dei paper da non citare: [docs/ERRATA.md](docs/ERRATA.md)

Stato, in breve:

- **Tabelle.** Letter 97,9%, Sonar 85,5%, Titanic 81,7% (venti split per gli
  ultimi due), confrontati con KNN, Random Forest e HistGradientBoosting nello
  stesso protocollo. I numeri e i limiti sono in MODEL-RESULTS.
- **Linguaggio.** Con SmolLM2-135M a pesi invariati, 64 token esatti più 64
  concetti danno NLL 3,01 contro 3,23 del controllo a pari memoria e 2,76
  dell'attenzione densa. Lo strato è più lento della densa ai contesti provati.
  Che un modello addestrato con lo strato a concetti eguagli la densa non è
  ancora dimostrato: il laboratorio generativo serve a misurarlo.

## Installazione per lo sviluppo

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Il backend PyTorch si installa con `python -m pip install -e '.[torch]'`.

## Applicazione interattiva

```sh
python -m pip install -e '.[ui]'
python -m acan.vis.render
```

Aprire `http://localhost:4000`.

1. **Dati**: caricare un file (CSV, Excel o ARFF).
2. **Modello**: scegliere la colonna da prevedere, oppure lasciarla vuota per
   scoprire gruppi senza target, e avviare la scoperta dei concetti. Il grafo
   mostra la radice, i livelli di concetti generali e i gruppi che fanno le
   previsioni; le frecce dicono da quali concetti generali discende ciascun
   gruppo, le linee tratteggiate dove le righe esitano fra due gruppi. Un clic
   su un concetto o su un gruppo mostra il suo prototipo; un doppio clic su un
   gruppo lo approfondisce a mano. Il modello si salva e si
   ricarica come JSON. Da «Ispeziona il modello» si scarica anche il modello
   come progetto Python (`modello_concetti.zip`, solo numpy e pandas): si usa
   senza installare, `python modello_concetti.zip prevedi righe.csv`, oppure da
   Python per previsioni, completamento, aggiornamento con righe nuove e nuova
   scoperta dei concetti.
3. **Valutazione**: misura il modello sul 20% di righe tenute da parte e, su
   richiesta, lo confronta con altri metodi (KNN, alberi, foreste, boosting)
   sulle stesse righe. Se si scelgono i rami guardando queste righe, esse
   diventano validazione, non un test indipendente.
4. **Previsione**: caricare un CSV nuovo e scegliere **Predici Labels** oppure
   **Completa campi vuoti**; nessun addestramento avviene in questa scheda.
5. **Matematica**: le formule del modello, gli esempi e i limiti misurati.

Le celle numeriche vuote sono trattate come mancanti; ogni riga da completare
deve avere almeno un attributo noto. I lavori di addestramento sono separati
per sessione del browser e restano in memoria: un riavvio del server li perde.
È un'applicazione di ricerca locale, senza account.

Per avviarla senza il debugger di Dash:

```sh
python -c "from acan.vis.render import app; app.run(debug=False, port=4000)"
```

Guida dettagliata della scheda Modello e della valutazione: [docs/GUI.md](docs/GUI.md).

## Laboratorio generativo

La scheda **Generativo** innesta lo strato a concetti in SmolLM2 (135M su CPU;
360M e 1.7B richiedono una GPU), lo adatta con LoRA su un corpus scelto, con
pausa e ripresa da checkpoint, e genera testo dai checkpoint salvati. Per
sapere se i concetti funzionano, il confronto valido è con il **controllo
denso**: lo stesso addestramento con attenzione completa, avviato dalla
sezione 5 della scheda; compare sullo stesso grafico con il verdetto.
Scendere sotto il modello di partenza non basta, perché qualunque adattamento al
corpus lo fa. La scheda permette anche di eseguire esperimenti in serie, di
misurare memoria e velocità su testi lunghi, di rinominare ed eliminare i
progetti.

- Percorso passo per passo e cosa misurare: [docs/CONTROLLED-LM.md](docs/CONTROLLED-LM.md)
- Funzionamento e persistenza: [docs/GENERATIVE.md](docs/GENERATIVE.md)
- Differenze dal paper e dal pacchetto di riferimento: [docs/LM-AUDIT.md](docs/LM-AUDIT.md)

## Rilascio e installazione sul server

Docker, CI e rilasci: [docs/RELEASING.md](docs/RELEASING.md) e
[CONTRIBUTING.md](CONTRIBUTING.md). Per il VPS si usa il file
`compose.hostinger.yaml` allegato a un rilascio verificato:
[docs/HOSTINGER.md](docs/HOSTINGER.md). Ogni rilascio richiede l'audit del
paper, le ricevute dei benchmark sintetici e le prove HTTP dell'immagine.

## Contenuto del repository

- `acan/`: libreria (`acan/core/concepts.py` nucleo tabulare,
  `acan/lm/concept_attention.py` strato di attenzione,
  `acan/concept_algebra.py` funzioni comuni)
- `tests/`: test unitari e, in `tests/browser/`, test nel browser
- `scripts/`: valutazioni e sonde che rigenerano le misure dei documenti
- `benchmarks/`, `experiments/`: suite sintetica storica e risultati archiviati
- `paper/`: sorgenti dei paper

La suite sintetica storica si esegue senza toccare i risultati archiviati:

```sh
OPENBLAS_NUM_THREADS=1 python benchmarks/run_benchmarks.py --output-dir .runs/paper-benchmark
```

I casi «tipo UCI» di quella suite sono sintetici, non i dataset UCI reali.

## Storia

I documenti seguenti descrivono versioni precedenti e restano per memoria; dove
differiscono da [MODEL.md](docs/MODEL.md), vale MODEL.md. I modelli salvati con
i profili precedenti (`cp-v9`, `cp-joint-v1`, `legacy-v8`) restano caricabili.

- Paper originale e corrispondenza con il codice: [PAPER-AUDIT.md](docs/PAPER-AUDIT.md),
  [SCIENTIFIC-ALIGNMENT.md](docs/SCIENTIFIC-ALIGNMENT.md)
- CP congiunta e selezione: [JOINT-CP.md](docs/JOINT-CP.md),
  [CP-SELECTION.md](docs/CP-SELECTION.md), [CP-PAIR-SUPPORT.md](docs/CP-PAIR-SUPPORT.md),
  [LETTER-DIAGNOSIS.md](docs/LETTER-DIAGNOSIS.md)
- Fondamento e revisioni verso il modello unico:
  [MATHEMATICAL-FOUNDATION.md](docs/MATHEMATICAL-FOUNDATION.md),
  [CONCEPT-GRAPH.md](docs/CONCEPT-GRAPH.md), [CP-V9-RESULTS.md](docs/CP-V9-RESULTS.md),
  [CP-GRAPH-RESULTS.md](docs/CP-GRAPH-RESULTS.md), [REVISION-PLAN.md](docs/REVISION-PLAN.md),
  [NEXT-PR-PLAN.md](docs/NEXT-PR-PLAN.md)
- Prime verifiche sui dataset: [DATASET-EVALUATION.md](docs/DATASET-EVALUATION.md),
  [HISTORICAL-VERSION-COMPARISON.md](docs/HISTORICAL-VERSION-COMPARISON.md)

## Licenza

MIT; vedere [LICENSE](LICENSE).
