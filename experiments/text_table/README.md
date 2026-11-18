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

## Dal testo alla previsione con il modello addestrato

Ogni posizione del testo di addestramento produce una riga. Le otto colonne
vettoriali contengono i token già letti, dal più recente al più lontano.
Il vettore del token successivo è il bersaglio. Il modello tabulare impara
quali righe possono condividere un concetto e come i concetti si collegano
fra livelli. Per ogni riga si distribuisce un conteggio del token successivo
tra i concetti in proporzione alle appartenenze. Per esempio, una riga con
appartenenze 0,8 e 0,2 aggiunge quei pesi alla colonna del suo token.

In predizione il riconoscimento può osservare anche solo alcune colonne. Per
ciascun concetto si parte dalla sua massa e si valuta quanto le colonne visibili
somiglino alle sue osservazioni. Normalizzando questi valori si ottengono le
appartenenze del prefisso. La posizione del prossimo token resta sconosciuta.
Se il concetto ha visto spesso una certa continuazione, il suo conteggio aumenta
la probabilità di quel token. Quando le osservazioni sono poche, il concetto
riceve più informazione dal livello superiore. Con conteggio $n_i(y)$, massa
$C_i$, numero di valori distinti $D_i$ e previsione superiore $q_i(y)$, la
lettura vale $(n_i(y)+a_i q_i(y))/(C_i+a_i)$ con
$a_i=\kappa\max(1,D_i)$. All'inizio il prior della radice è uniforme.
La stessa operazione combina i conteggi di addestramento con quelli già letti
nel documento. La media delle previsioni dei concetti, pesata con le appartenenze,
dà la distribuzione del prossimo token. Dopo la scelta si aggiorna il documento
e la nuova posizione diventa prevedibile. Non servono confronti fra coppie di
posizioni del testo.

Abbiamo confrontato 1, 2, 4 e 8 posizioni visibili durante il riconoscimento,
lasciando al documento tutte le osservazioni precedenti. La prima metà della
validazione sceglie una posizione. A parità di concetti e memoria, su altri 64
blocchi di test da 512 token la NLL passa da 6,1916 a 6,0876 (differenza media
−0,1040, errore standard tra differenze appaiate per blocco 0,0078).
Il contatore sui primi 200.008 token ottiene 5,4117 sugli stessi blocchi.
Gli offset e i risultati sono in [wikitext2-context.json](results/wikitext2-context.json).

Per verificare il ricordo di singoli fatti abbiamo presentato una sola volta
16 associazioni con una frase comune e una chiave diversa, e 16 sequenze casuali
da completare. Ogni domanda ripete il suo indizio in una copia dello stesso
prefisso. Le risposte e le chiavi finali sono escluse dal testo di riempimento.
Con il riconoscimento a otto posizioni, dopo 300.000 token la risposta è prima
tra 16 candidati in 6 associazioni e 7 completamenti. Sul vocabolario intero
nessuna delle 32 risposte arriva prima, anche alla distanza più breve di 1.000
token. È una prova sintetica su un solo insieme di chiavi.
[Risultati per distanza](results/wikitext2-recall.json).

Per provare il completamento, aggiungere `--save-table /tmp/concetti.pkl` al
comando `layers` sopra. Il checkpoint registra l'identità del tokenizer.
Il comando stampa il testo completato scegliendo il token più probabile a ogni passo.

```sh
python -m experiments.text_table.complete --table /tmp/concetti.pkl \
  --tokenizer /percorso/parquet/bpe-8000.json --tokens 4 'The history of'
python -m experiments.text_table.context --table /tmp/concetti.pkl \
  --data /percorso/parquet --output /tmp/contesto.json
python -m experiments.text_table.recall --table /tmp/concetti.pkl \
  --data /percorso/parquet --output /tmp/ricordo.json
```
