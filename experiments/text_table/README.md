# Prima verifica: testo come righe vettoriali di ConceptModel

Esperimento separato dal prodotto. Usa **lo stesso ConceptModel tabulare**:
scansione, generalizzazione, livelli, appartenenze parziali e statistiche additive.
Non usa le chiavi di contesti esatti per riconoscere le righe. Il contatore viene
rieseguito soltanto come controllo, sullo stesso tokenizer e sugli stessi token.

## Che cosa è stato realizzato

- Tokenizer BPE8000 imparato solo sul train ufficiale di WikiText-2.
- Vettori32 ottenuti dalle radici delle frequenze dei token precedenti/successivi
  sul train, rispetto ai512 token più frequenti più una casella per gli altri,
  e SVD con seme fissato. Nessun gradiente o peso di rete.
- Ogni posizione ha256 colonne note: otto vettori in ordine di distanza. Le32
  colonne target contengono il vettore del token da prevedere. Non partecipano
  al riconoscimento; non si legge la posizione corrente o il futuro.
- `fit_chunks`:200000 righe generate a blocchi di2048; livelli rifiniti su20000
  righe e giudizio su512. Parametri dichiarati prima del test, nessuna scelta
  di questi parametri sul test. Il budget configurato256 non impone il K.
- Seconda passata sulle stesse righe: somme del vettore target e conteggi sparsi
  delle coppie concetto/token, pesati dalle appartenenze. Non si creano8000
  colonne target nella tabella. La lettura usa letteralmente
  `categorical_reading`, con un prior normalizzato dalla distanza del vettore.
- La varianza del target è avvicinata alla varianza della radice con la stessa
  `centre_variance` del nucleo, con peso della radice1. Il kappa della lettura
  categorica è distinto e scelto sulla validation. La lettura assoluta del target viene riaggregata
  sulle appartenenze finali anche quando i livelli lavorano su residui.

La densità delle coppie memorizzate è87,35%: il formato sparso non rende rare
le appartenenze morbide. Il prior vettoriale è una matrice concetti×vocabolario;
non viene materializzato un tensore posizioni×concetti×vocabolario.

## Risultato misurato

64 blocchi disgiunti da512 token, seme42. Il token iniziale è dato, quindi si
misurano32704 predizioni. Validation: primi32 blocchi per scegliere kappa,
altri32 per controllo; tutti i64 blocchi test rimangono separati.

| Lettura | Test, nat/token | Tempo di apprendimento e scelta pesi |
|---|---:|---:|
| Contatore,200008 token | 5,4075 | 1,13s |
| Contatore, tutto il train (2729601 token) | 4,5199 | 8,51s |
| ConceptModel,200000 righe di vettori | **6,9207** | **335,26s**, esclusa la successiva scelta di kappa |

Per il prototipo: **83 concetti, livelli2→14→83**. Non si ripete il collasso a
uno solo, ma la lettura del testo peggiora. Kappa scelto sulla validation=0,03125,
il minimo della griglia; la validation di controllo misura7,0138. Vettori e
tokenizer hanno visto tutto il train: il primo controllo non ha quindi la stessa
esposizione complessiva al testo. Il controllo su tutto il train è incluso per
rendere visibile questa differenza.

Bande del test del prototipo: 0–32:6,9836;32–128:6,9509;128–256:6,9636;
256–512:6,8804. Il picco dell'intero processo è2751592KiB, circa2,62GiB:
comprende anche il contatore completo conservato in memoria e non è una misura
isolata del prototipo. I tempi sono di questa macchina, non del server finale.

Le vecchie misure4,403 e4,011 della rete non sono state rieseguite. Mancano il
tokenizer e gli indici dei blocchi di quelle esecuzioni: non si presenta questa
prova come un confronto identico con quei numeri. Gli hash e gli indici di
questa esecuzione sono invece conservati nella [ricevuta](results/wikitext2-200k.json).

## Diagnosi e limite che impedisce l'integrazione

Il nucleo corrente tratta le32 dimensioni target come32 risposte numeriche.
Ogni risposta riceve peso pari alle256 colonne note: il peso totale del target
è8192 contro256 della parte nota. Non è il peso di un solo vettore target.
Inoltre il giudizio riguarda la densità gaussiana del vettore, non la NLL del
token esatto; quest'ultima sceglie qui solo kappa, dopo la ricerca. Il costo
usato per unire concetti considera la parte riconosciuta, non il target.

Questi scarti sono verificabili nel codice invariato. Non costituiscono una
prova che correggerne uno basti a migliorare: serve un'ablazione. Il numero di
righe maggiore e la scala standardizzata non bastano a ottenere una lettura
utile del testo. Non è stato necessario sostituire ConceptModel per misurarlo.

**Esito: fermarsi prima dell'integrazione**, come richiesto. Vettori affinati,
memoria del documento,300000 token, più teste e secondo layer non sono stati
implementati o dichiarati verificati. Il prossimo passo da concordare è il
trattamento del vettore come un unico blocco target e il giudizio sul token
esatto dentro la ricerca condivisa, preservando i controlli tabulari.

## Riprodurre

Occorrono le dipendenze scientifiche e il tokenizer degli extra `ui,llm`.
Preparare fuori dal repository i tre parquet ufficiali di WikiText-2-raw-v1
alla revisione `b08601e04326c79dfdd32d625aee71d232d685c3`, nominandoli
train.parquet,validation.parquet,test.parquet. URL e hash sono nel catalogo
`acan/lm/public_corpus.py`; la ricevuta registra gli hash effettivamente usati.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m experiments.text_table.run \
  --data /percorso/parquet --output reports/text-table-200k.json
python -m pytest tests/test_text_table_experiment.py -q
```

La directory dati conserva il tokenizer e la sua identità train. Il prodotto,
i suoi checkpoint e i file di contratto non vengono modificati dall'esperimento.

Verifica completata:277 test,13 subtest,nessuno skip; cinque nuovi controlli
passano anche in un ambiente senza Torch. Tutti i test browser del repository
sono passati. I dataset per i controlli opzionali Titanic e credit-g sono stati
forniti tramite ACAN_DATA, e sono stati installati i client opzionali mancanti: i
cinque skip iniziali non sono stati lasciati come sostituto delle verifiche.
