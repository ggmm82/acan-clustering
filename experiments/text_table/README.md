# Misure su WikiText-2

Questa directory conserva i protocolli e le ricevute delle prove sul grafo
tabulare. L'implementazione usata dall'app risiede in `acan/lm`: la scoperta
dei concetti è affidata a `acan/core/concepts.py`. La spiegazione di
apprendimento e previsione si trova nella [guida unica](../../acan/MODEL.md).

## Coppie di concetti

Per ogni posizione il riconoscitore tabulare apprende la coppia ordinata di token
precedenti e poi la coppia di concetti che copre quattro token. Per arrivare a sei,
combina la coppia più vecchia con le quattro posizioni recenti. Le letture esatte
degli stessi suffissi permettono di misurare separatamente quanto aggiungono i
concetti. Il test comprende 32.704 previsioni nei 64 blocchi del protocollo
WikiText-2 raw v1. Ogni orizzonte usa soltanto le colonne che gli competono.
I pesi sono scelti sui primi 32 blocchi di validazione, separatamente con e
senza memoria. La memoria locale di entrambi usa al massimo quattro token come
chiave, sempre entro l'orizzonte della riga. L’ablazione delle sole identità
esatte mantiene il lettore nativo del token singolo e toglie le colonne dei
concetti composti. Gli altri 32 blocchi servono da controllo.

Il primo passaggio usa fino a 256 concetti geometrici per scala. Il successivo
scopre figli locali dentro i padri. Per la lettura categorica, un articolo su
cinque del train fornisce il giudizio su 577.414 righe, mentre gli altri articoli
forniscono i conteggi candidati. Dopo la selezione si ricalcolano le frequenze
sull'intero train. Un secondo esperimento ripete la CP numerica dentro ciascun
padre della coppia di token su tutte le coppie distinte osservate. Nel confronto
fra queste due risoluzioni si usa la NLL senza memoria a sei token sulla metà di
controllo della validazione. Le quattro righe riportate per la soluzione scelta
usano la stessa procedura.

| Token visibili | Concetti senza memoria | Contatore senza memoria | Concetti con memoria | Contatore con memoria |
|---:|---:|---:|---:|---:|
| 1 | 5,2676 | 5,3004 | 4,8088 | 4,8313 |
| 2 | 4,9028 | 4,9562 | 4,5035 | 4,5426 |
| 4 | 4,8541 | 4,9202 | 4,4600 | 4,5056 |
| 6 | 4,8453 | 4,9141 | 4,4537 | 4,5000 |

[Ricevuta con selezione, ablazioni, costi, risultati tabulari e differenze per blocco](results/native-pair-scales.json).

A sei token, conservare solo le letture delle identità esatte sopra il lettore
nativo a un token dà 4,8849 senza memoria. Con le coppie apprese e le letture
esatte si arriva a 4,8453. Togliendo solo le letture dei suffissi interamente
esatti più lunghi di un token si ottiene 4,9024. Sul controllo di validazione
separato il modello scelto ottiene 4,9137 e il contatore 4,9742.

La ricerca dentro i padri usa 27.878 codici categorici distinti sul train
alla scala due e 15.461 alla scala quattro. Sullo stesso corpus il riesame
geometrico di 561.791 tipi di coppia con CP divide 159 padri e occupa 6.971
codici geometrici. Il giudizio geometrico considera 545.153 righe locali.
Questa ricerca richiede 704 secondi e un picco di circa 851 MiB nel processo
di misura. Il suo contributo aggiuntivo sul test a sei token è di 0,00075
nat/token rispetto ai soli figli categorici. Le due ricerche locali hanno
perciò costi e contributi diversi.

Le dieci prove tabulari standard riproducono le predizioni di riferimento. In
una prova separata con il 20% del train riservato al giudizio locale, su
breast-cancer la NLL passa da 0,5314 a 0,5293 e le classificazioni corrette da
45 a 44 su 58. Su Vote e Mushroom le predizioni conservano la stessa NLL e
accuratezza. Su Iris è stata verificata l'assegnazione dei dati di test ai
sottoconcetti numerici.

Per riprodurre, preparare i parquet e il tokenizer BPE 8.000 indicati sotto.
Le colonne intermedie e i parametri appresi vengono scritti in `RIGHE`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.text_table.pair_layers --data DIR --workdir RIGHE
python -m experiments.text_table.pair_counter --data DIR --output contatori.json
python -m experiments.text_table.pair_reading --workdir RIGHE --output scale.json
python -m experiments.text_table.pair_subconcepts --workdir RIGHE
python -m experiments.text_table.pair_reading --workdir RIGHE --local --output figli.json
python -m experiments.text_table.pair_local_geometry --workdir RIGHE
python -m experiments.text_table.pair_reading --workdir RIGHE --local --native-local --output figli_cp.json
python -m experiments.text_table.dyadic_tabular DIR_ALLEGATI tabulare.json
python -m experiments.text_table.pair_tabular DIR_ALLEGATI rami_tabulari.json
```

## Identità esatta del token precedente

Il punto 1 mantiene il prefisso numerico della prova nativa e aggiunge una
codifica univoca del token immediatamente precedente. Tredici bit distinguono
gli 8.000 token del vocabolario e il valore mancante al confine di articolo
o blocco. Il riconoscitore tabulare apprende su tutte le 2.729.601 posizioni.
Con il limite precedente di 256 concetti, il livello finale ne usa 99.
Nella lettura categorica, ACAN può anche consultare le frequenze del token
successivo per il valore esatto della colonna. Il peso con cui queste frequenze
si combinano con la previsione dei concetti viene scelto su 16.352 posizioni di
validazione. Il test usa 32.704 token e la memoria del documento è spenta.

| Lettura | NLL sul test |
|---|---:|
| Modello nativo con la precedente codifica del contesto | 6,8272 |
| Modello nativo con i 13 bit aggiunti ai predittori | 6,6677 |
| Modello nativo con i 13 bit e la lettura della categoria esatta | 5,3158 |
| Sola categoria esatta, con le frequenze globali come riferimento | 5,3052 |
| Contatore a un token | 5,3004 |
| Contatore a sei token | 4,9141 |

La lettura della categoria esatta recupera gran parte del divario. Su questo
test la previsione dei concetti aggiunta alle frequenze delle coppie porta la
NLL da 5,3052 a 5,3158. Nella lettura della sola categoria, le 27.752 coppie
già viste hanno NLL 4,4231. Le altre 4.952 posizioni, compresi cinque inizi
d’articolo, hanno NLL 10,2490. L’identità indica quali conteggi consultare;
quando manca la coppia, la probabilità dipende dalla distribuzione di riferimento.

Il valore storico di 4,8204 nella prova di troncamento conservava i conteggi
locali del documento. Per il confronto attuale il contatore viene addestrato
con profondità uno e memoria zero. Per la fase successiva il riferimento corretto a un token senza memoria è 5,3004. Le misure aggiornate delle coppie sono riportate sopra. [Ricevuta](results/exact-identity-gate.json).

Il comando usa le matrici prodotte dallo stadio `prepare` descritto sotto.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m experiments.text_table.exact_identity --data DIR --workdir RIGHE --output identita.json
```

## Verifica del riconoscitore nativo

WikiText-2 fornisce 2.729.601 posizioni di training. In ogni riga, il vettore
dell’ultimo token e le medie su fasce successive del prefisso descrivono il
contesto. Le fasce si fermano al confine dell’articolo. I vettori a 8 dimensioni
derivano dalle co-occorrenze del train tramite SVD. Una proiezione comune a 16
dimensioni riassume le fasce e i loro indicatori di disponibilità. Entrambi i
riconoscitori ricevono le stesse 24 colonne, lo stesso limite di 256 concetti
e la stessa regola per leggere le frequenze dei token.

Il giudizio sceglie lo smoothing sui primi 32 blocchi di validazione, poi confronta
i riconoscitori sugli altri 32. Il modello nativo trova 114 concetti e ottiene
6,868 nat/token. L’albero ne usa 256 e ottiene 6,950. Per la verifica finale si
usa quindi `ConceptModel.fit_chunks`. Dopo aver appreso la geometria, ogni riga
contribuisce al conteggio del token osservato in ciascun concetto con il proprio
grado di appartenenza. In previsione `predict_proba` combina le distribuzioni
categoriche dei concetti con gli stessi pesi. Con la memoria attiva si aggiornano
anche i conteggi locali dopo aver osservato ciascun token.

Sul test il contatore ottiene una NLL inferiore in entrambe le condizioni,
anche ripetendo il confronto sul 10% degli articoli di train. Per la ripetizione
si riapprende anche il tokenizer. Le NLL vanno confrontate entro ciascun budget.
Il risultato arresta la prova prima di WikiText-103 e dell’integrazione nel sito.
[Misure, impronte e confronto tabulare](results/native-recognition-gate.json).

| Train WikiText-2 | Memoria del documento | Modello nativo | Contatore |
|---|---|---:|---:|
| Completo | Attiva | 6,4696 | 4,5000 |
| Completo | Spenta | 6,8272 | 4,9141 |
| 63 articoli su 629 | Attiva | 6,7193 | 5,2915 |
| 63 articoli su 629 | Spenta | 7,0989 | 5,9034 |

Per riprodurre, usare i parquet descritti sotto e il BPE 8.000 del protocollo.
`RIGHE` è una directory di lavoro per le matrici su disco.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
for stage in prepare native tree; do
  python -m experiments.text_table.recognition_comparison --data DIR --workdir RIGHE --stage "$stage" --output "$stage.json"
done
python -m experiments.text_table.native_gate --workdir RIGHE --output nativo.json
python -m experiments.text_table.memory_control --data DIR --output contatore.json
python -m experiments.text_table.train_fraction --data DIR --destination DIR_10
python -m experiments.text_table.recognition_comparison --stage prepare --data DIR_10 --workdir RIGHE_10 --output righe_10.json
python -m experiments.text_table.native_gate --workdir RIGHE_10 --output nativo_10.json
python -m experiments.text_table.memory_control --data DIR_10 --output contatore_10.json
python -m experiments.text_table.dyadic_tabular DIR_ALLEGATI tabelle.json
```

Il lettore `text_graph` e le sue dipendenze servono ancora a caricare e
riprendere i checkpoint salvati con il profilo `concept-table-v1`.

## Protocolli precedenti

Le ricevute in `results/` distinguono i blocchi di test e gli abbinamenti usati
per le singole prove. Nei confronti di sviluppo i vettori erano ricavati dal
train completo. Il precedente grafo a righe causali li ricavava dal primo tratto. La lettura CP nei suffissi ricalcola i profili dai bigrammi conservati.

Per riprodurre le misure storiche, preparare i parquet train, validation e test
di WikiText-2-raw-v1, revisione `b08601e04326c79dfdd32d625aee71d232d685c3`.
Il comando `python -m experiments.text_table.run --data DIR --output prima.json`
prepara il BPE sul train. Le ricevute storiche rimangono in `results/`.

Per confrontare i gruppi CP nei suffissi ordinati usare i medesimi parquet e
tokenizer del protocollo precedente. Il giudizio sceglie i pesi sui primi 32
blocchi di validazione. Le ricevute riportano i 64 nuovi offset del test,
l'errore standard delle differenze per blocco e il controllo con gruppi permutati.

```sh
python -m experiments.text_table.refinement --data DIR --budget-mib 256 --output confronto.json
python -m experiments.text_table.refinement --data DIR --budget-mib 1024 --output confronto-1024.json
python -m experiments.text_table.refinement --data DIR --budget-mib 256 --continuous --output continuo.json
python -m experiments.text_table.suffix_recall --data DIR --output richiamo.json
python -m experiments.text_table.load_test --data DIR --output carico.json
```

Il confronto storico separa i blocchi di conteggio ogni 32.768 token. Con
`--continuous` entrambe le letture conservano il suffisso al confine come la
nuova coda. La prova di carico ripete e ruota gli ID del train per generare
nuove chiavi. Misura tempi e memoria al crescere del flusso. Il numero di
byte è quello degli identificatori a 32 bit. La prova di qualità usa invece
il train originale e un test separato.

Per misurare l'effetto di una quantità maggiore di testo naturale, il protocollo
`natural_stream` legge entrambi gli shard originali di train di WikiText-103 raw
alla stessa revisione. WikiText-103 e WikiText-2 condividono esattamente i
parquet di validazione e test, verificabili tramite SHA-256. Si conserva il BPE
8.000 già imparato sul train piccolo per confrontare le NLL nella stessa unità.
Il programma tokenizza in streaming, verifica la validazione alla fine di ciascuno
shard e legge i 64 blocchi di test dopo aver completato entrambi.

```sh
python -m experiments.text_table.natural_stream --data DIR \
  --shards train-00000-of-00002.parquet train-00001-of-00002.parquet --output naturale.json
```

Per provare il profilo della coda web, compresi il tokenizer da 32.000, la
preparazione dei documenti, il worker e il caricamento del checkpoint, eseguire
`python -m experiments.text_table.worker_queue --data DIR --output worker.json`.
I dati hanno split espliciti e i duplicati vengono rimossi dal worker.
Le NLL con vocabolari diversi sono espresse in unità di token diverse.

## Contesto a scale diadiche

L'esperimento usa 20 scale sovrapposte, una tabella di composizione per ciascuna
scala con esempi disponibili e un `ConceptModel` con il token come target
categorico. Il contatore con identità esatte e la precedente lettura dei suffissi
con gruppi CP sono i controlli. Le prove di associazione, copia e riformulazione
usano soltanto la tabella diadica.

```sh
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m experiments.text_table.dyadic_run \
  --data DIR --output diadico.json --checkpoint-output selezionato.pkl
python -m experiments.text_table.dyadic_control --data DIR --output controlli.json
python -m experiments.text_table.dyadic_tabular DIR_ALLEGATI tabelle.json
OPENBLAS_NUM_THREADS=1 python -m experiments.text_table.dyadic_judgment > giudizio.json
```

Il tokenizer e i parquet in DIR sono quelli del protocollo WikiText-2 descritto
sopra. Le co-occorrenze e le finestre rispettano gli inizi di articolo. Il test
usa gli stessi 64 offset del confronto con i suffissi. La ricevuta dà sia il
risultato partendo dal blocco vuoto, sia quello con il prefisso dell'articolo.
La prova lessicale cambia «doctor» in «physician» e così via per sedici chiavi
inglesi con risposte assegnate a caso. Gli ID delle risposte sono esclusi dal
riempimento. Il file pickle opzionale serve alla ripetizione locale della prova.

Per completare un prompt con il checkpoint locale appena prodotto:

```python
import pickle
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("DIR/bpe-8000.json")
with open("selezionato.pkl", "rb") as file:
    modello = pickle.load(file)
prefisso = tokenizer.encode("The dog ran across the").ids
print(tokenizer.decode(modello.generate(prefisso, 8)))
```

## Lettura condizionale delle regioni CP

Usare i parquet e il BPE della prova diadica descritta sopra. I due candidati
partono dallo stesso checkpoint locale di `dyadic_run`. Omettendo `--base`
il programma lo ricostruisce dal training. I file di checkpoint contengono
oggetti Python e i comandi caricano il checkpoint prodotto localmente.

```sh
python -m experiments.text_table.dyadic_run --data DIR --output base.json --checkpoint-output base.pkl
python -m experiments.text_table.conditional_run --data DIR --base base.pkl --mode svd --neighbors 96 --output svd.json --checkpoint svd.pkl
python -m experiments.text_table.conditional_run --data DIR --base base.pkl --mode cp_log --neighbors 96 --output cp.json --checkpoint cp.pkl
python -m experiments.text_table.word_refinement --data DIR --base svd.json --output parole.json
python -m experiments.text_table.conditional_eval --data DIR --candidates parole.json svd.json cp.json --output test.json
python -m experiments.text_table.conditional_generate --checkpoint parole.pkl --tokenizer DIR/bpe-8000.json --tokens 16 'The first train'
python -m experiments.text_table.conditional_control --data DIR --checkpoint parole.pkl --output confronto.json
python -m experiments.text_table.conditional_load --data DIR --checkpoint svd.pkl --output carico.json
```

La NLL sui primi 32 blocchi di validazione sceglie il checkpoint. Gli altri 32
servono da controllo. Il test usa gli offset pubblicati, compresi i confini di
articolo dentro ogni blocco. La prova di carico verifica i totali di ciascuna
categoria dopo aver sommato un GiB di righe ripetute in un livello del modello.
[Qualità, ablazioni e richiamo lessicale](results/wikitext2-word-regions.json) ·
[Confronto appaiato](results/word-regions-paired.json) ·
[Somme in streaming](results/conditional-load.json).
