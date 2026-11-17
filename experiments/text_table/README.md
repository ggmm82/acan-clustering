# Il testo letto dai livelli del modello tabulare

Prova fuori dal prodotto: una sola istanza del **ConceptModel tabulare invariato**
impara le righe di contesto. Non vengono introdotti un altro algoritmo di
clustering, una rete o un contatore di contesti esatti. La PR resta in bozza:
il risultato migliora la prima prova, ma la qualità non raggiunge ancora il
contatore dell'app. Le spiegazioni di riferimento dell'app restano invariate
finché il percorso non è pronto per l'integrazione.

## Percorso verificato

1. BPE8000 e vettori32 da radici delle co-occorrenze e SVD, imparati sul solo
   train ufficiale di WikiText-2. Il tokenizer e i vettori vedono tutto il train.
2. Una posizione diventa una riga: gli otto vettori precedenti, in ordine, sono
   le256 colonne note; il vettore del token successivo è il target numerico.
   `fit_chunks` impara200000 righe, a blocchi2048; i livelli vengono rifiniti
   su20000 righe e giudicati su512, come nella prima prova.
3. Il riconoscimento senza target assegna ogni riga ai concetti appresi. Le sue
   appartenenze pesano i conteggi sparsi concetto/token. Le matrici dei genitori
   del modello propagano questi conteggi verso i livelli generali.
4. La lettura scende dagli stessi concetti generali a quelli particolari. A ogni
   livello usa `categorical_reading`: i conteggi del concetto sono sostenuti
   dalla distribuzione dei genitori. La radice ha un prior uniforme proprio,
   quindi anche un token mai contato ha probabilità positiva.
5. Il documento conserva altre somme negli **stessi** concetti. A ogni livello,
   i conteggi globali leggono usando il genitore del documento come prior; i
   conteggi del documento leggono usando quella previsione globale come prior.
   Sono due applicazioni della stessa formula, non due modelli appresi.
6. `TextCursor` riconosce il contesto, restituisce le probabilità e solo quando
   riceve il token aggiorna le somme. Conserva una finestra ordinata di8 token;
   tutto il passato più lontano rimane nelle statistiche dei livelli.

Questa estensione usa i livelli già appresi. Non aggiunge una seconda rete o un
secondo modello addestrato sulle uscite del primo. La ricerca dei concetti resta
quella numerica corrente: il token esatto non giudica ancora il percorso delle
fusioni. Non sono dimostrati recupero di informazioni lontane o induction.

## Misure

Stesso tokenizer e stessi64 blocchi test disgiunti da512 token della prima
ricevuta:32704 predizioni. I primi32 blocchi validation scelgono il peso del
prior, gli altri32 controllano; il test non sceglie pesi. Tutte le letture sotto
usano la medesima tabella di appartenenze e le stesse righe di apprendimento.

| Lettura | Errore test, nat/token (meno è meglio) |
|---|---:|
| Prima prova: prior dal vettore target | 6,9207 |
| Livelli appresi, senza memoria del documento | 6,5794 |
| Livelli appresi e documento causale | **6,1896** |
| Controllo: soli conteggi della radice e documento | 6,8499 |
| Contatore dell'app, primi200008 token | 5,4075 |
| Contatore dell'app, train completo | 4,5199 |

Il controllo della sola radice accorpa i conteggi dei concetti: serve a verificare
che il vantaggio non venga soltanto dalle frequenze del documento. Con il modello
invariato rimangono **83 concetti**, nei livelli **2→14→83**. Il peso della lettura
è0,5, scelto sulla validation; validation di controllo6,2411. La prima prova e i
contatori sono registrati in [wikitext2-200k.json](results/wikitext2-200k.json),
questa prova in [wikitext2-layers-200k.json](results/wikitext2-layers-200k.json).
I vecchi numeri delle reti non sono confronti rieseguiti con questo tokenizer.

Una diagnosi isolata ha diviso per32 il peso del target durante l'apprendimento:
l'errore è peggiorato a8,1157. La modifica è stata scartata e non è nel nucleo.

## Costo e garanzie

A modello fissato, il numero delle posizioni passate non è una dimensione delle
matrici di lettura. Con K concetti complessivi e V token, i conteggi del documento
occupano al massimo K×V valori, più masse e cardinalità. La probabilità di un token
legge una colonna dei conteggi; quella di tutto il vocabolario legge V colonne.
Le matrici dei genitori dipendono dai concetti, non dalla lunghezza del documento.
Questo esclude un confronto quadratico fra posizioni; non dimostra che ogni
informazione del passato rimanga recuperabile.

Misurati riconoscimento a blocchi, probabilità del token vero e aggiornamento:
0,317ms/token fino a1000;0,353 fra1000 e10000;0,318 fra10000 e100000;
0,364 fra100000 e300000. Questi tempi **non sono velocità di generazione**
sull'intero vocabolario. A300000 token gli array dei conteggi del documento
occupano5711904 byte: sono esclusi dizionari, modello globale e blocchi di query.

Verificati: conservazione della massa fra livelli, formula letterale per la
lettura globale e per quella del documento, normalizzazione, probabilità dei
token non osservati, causalità, reset e finestra limitata. La suite completa
prima dell'ultima rifinitura della lettura del documento ha passato282 test
senza skip; tutti gli11 test mirati passano sulla versione finale. Il nucleo
resta identico a main. Il test Docker monta anche gli esperimenti, continuando
a importare il pacchetto installato.

## Riprodurre

Preparare fuori dal repository i parquet train/validation/test di
WikiText-2-raw-v1, revisione `b08601e04326c79dfdd32d625aee71d232d685c3`.
Gli hash della prima ricevuta identificano i file. Occorrono le dipendenze
scientifiche e il tokenizer degli extra `ui,llm`.

```sh
# Prima prova e tokenizer train-only (una volta).
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m experiments.text_table.run \
  --data /percorso/parquet --output reports/text-table-200k.json
# Gli stessi concetti, lettura gerarchica e misura del costo fino a300000 token.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m experiments.text_table.layers \
  --data /percorso/parquet --output reports/text-layers.json --measure-cost
python -m pytest tests/test_text_table_experiment.py tests/test_level_reading.py -q
```
