"""What the software does, step by step, when the concept model learns a table or a text."""
from dash import dcc, html

TEXT = r"""
## Che cosa fa il programma, passo per passo

Questa pagina segue ciò che succede quando premi i pulsanti, dal file caricato al
modello che risponde. Il modello è uno solo: nella scheda Modello impara una
tabella, nella scheda Generativo impara un testo, che per lui è una tabella in
cui ogni posizione è una riga e il token in quella posizione è la colonna da
prevedere. Ogni passo qui sotto vale per entrambe; dove il programma fa cose
diverse lo dice. Le formule sono nella scheda Matematica; qui c'è il meccanismo.

Imparare non vuol dire correggere pesi con un gradiente: vuol dire **sommare**.
Un concetto è fatto di somme sui suoi membri, e due concetti si uniscono
sommandole.

---

### Passo 1 · Portare i dati

**Una tabella** si carica nella scheda Modello. Il file viene letto e ripulito.
Se ha più di 10.000 righe la pagina lavora su un **campione** di 10.000 (grafici,
filtri, anteprima), mentre la **tabella intera** resta sul server per
l'addestramento, fino a un milione di righe.

**Un testo** si prepara nella scheda Generativo. «Prepara i dati» scarica le
fonti una alla volta da versioni fissate, controlla ogni file, tiene solo il
testo e cancella il resto; si mette in pausa e riprende. I tuoi documenti (testo,
Markdown, Python, ZIP, EPUB, DOCX, PDF) si caricano dal computer al passo 1, «I
tuoi documenti», oppure al passo 3, «Aggiungi un testo alla coda»: il file viene
caricato a pezzi, convertito in testo e, nel secondo caso, messo in fondo alla
coda scelta.

In entrambi i casi una parte dei dati non viene mai usata per imparare: per la
tabella il 20% delle righe, per il testo i documenti di **validazione** (10%,
servono a giudicare) e di **test** (10%, tenuti da parte fino alla verifica
finale).

### Passo 2 · I dati diventano righe

**Una tabella.** Il programma propone un target e le colonne da escludere
(identificativi, testo libero); senza target cerca solo i gruppi. I numeri sono
portati sulla stessa scala, le categorie diventano caselle, i valori mancanti
restano non osservati.

**Un testo.** Un **tokenizer** lo divide in token: parole intere, pezzi di
parola, segni. Lo si costruisce dai testi stessi, all'inizio: guarda quali
sequenze di caratteri sono frequenti e ne fa un vocabolario di 32.000 pezzi.
Così l'italiano non paga i pezzi in più di un vocabolario fatto per l'inglese, e
qualunque carattere si può scrivere. I token vengono scritti su disco una volta
sola (`train.bin`, `validation.bin`, `test.bin`). Ogni posizione è una riga: i
token che la precedono sono le colonne note.

**Facoltativo, solo per il testo: i concetti dei token.** Con l'opzione
«Concetti dei token e percorsi» (spenta se non la si accende: aiuta sui testi
piccoli, peggiora su quelli grandi con la memoria limitata), prima di contare il
programma costruisce una **tabella dei token**: una riga per ognuno degli 8.000
token più frequenti dei primi 20 milioni, con quanto spesso ciascuno dei 256
token più comuni lo precede e lo segue, ridotta alle sue direzioni principali.
La dà allo **stesso modello** che impara le tabelle (passo 3), che ne trova i
concetti e i loro livelli: token che compaiono negli stessi posti finiscono
nello stesso concetto. Ci vogliono alcuni minuti; i concetti restano fissi e sono
salvati nel checkpoint. Con loro il prossimo token si legge anche lungo un
**percorso**: prima il concetto generale, poi quello particolare, poi il token.

### Passo 3 · Imparare: sommare a blocchi

Le righe entrano una volta sola, a pezzi, e ogni pezzo diventa le sue somme. Le
somme entrano in un **contatore in base due**: quando ci sono due blocchi grandi
uguali si uniscono in uno grande il doppio.

**Una tabella** piccola cerca subito i livelli. Oltre le 20.000 righe viene letta
tre volte, un pezzo alla volta, e la barra di avanzamento dice in quale lettura
si trova:

1. **scala degli attributi**: media e deviazione di ogni colonna, e 20.000 righe
   estratte a caso;
2. **concetti specifici a blocchi**: ogni riga entra una volta in un blocco; due
   blocchi uguali si uniscono e si generalizzano fino alla dimensione di uno;
3. **ricerca dei livelli** dai concetti dei blocchi a un solo concetto, unendo un
   ventesimo dei concetti alla volta e giudicando ogni passo sulle righe non
   viste; due volte: prima i concetti generali, poi ciò che più concetti generali
   hanno in comune;
4. **ogni riga aggiunta ai concetti finali**, così le loro somme sono
   sull'intera tabella.

**Un testo** è letto a passi di **1.000.000 di token**. Per ogni posizione e per
ogni lunghezza di contesto da 0 a 6 il programma calcola un numero che identifica
il contesto (gli ultimi token); per i percorsi il numero comprende anche il nodo
del percorso e, a ogni distanza, il concetto del token. Ordinando questi numeri e
contando i doppioni (`numpy.unique`) escono, per ogni contesto, quante volte
compare e da quali token è seguito: le somme del pezzo. Sommare due blocchi vuol
dire unire due elenchi ordinati di contesti e sommare i conteggi dei contesti
uguali. Sono operazioni su vettori, senza cicli sulle posizioni; il passo è
finito quando il pezzo è contato.

Un testo successivo della coda parte dal checkpoint del precedente e aggiunge le
sue somme: nulla viene dimenticato.

### Passo 4 · Restare nella memoria

**Una tabella** grande resta in memoria per pezzi: un pezzo di righe più i
blocchi; su due milioni di righe il processo è rimasto intorno a 1 GB.

**Un testo** fa crescere i contesti diversi con la sua lunghezza. La **memoria
massima dei concetti** vale per tutte le somme insieme, quelle già unite e quelle
ancora nel contatore (automatica: un quarto della memoria che il processo può
usare, del server o del suo contenitore, al massimo 2 GB; si può impostare). Dopo
ogni pezzo, se le somme la superano, il programma unisce tutto il contatore,
calcola per ogni concetto quanto peggio sarebbero letti i suoi membri dal
contesto più corto, ciascuno senza sé stesso come nel giudizio, e toglie i
concetti che costano meno per byte finché le somme scendono al 60% del limite.
Un contesto visto una volta sola non costa nulla e va per primo; le somme dei
concetti tolti restano nel livello sopra, si perde solo la distinzione. La soglia
si trova dimezzando l'intervallo, senza ordinare tutti i concetti, e le somme si
uniscono un livello alla volta liberando subito le vecchie: mentre somma il
processo usa al massimo circa tre volte il limite (con 794 MB di limite, su 40
milioni di token di prova, un picco di circa 2,3 GB).

### Passo 5 · Giudicare e salvare

Ogni scelta è giudicata su dati che il modello non ha visto.

**Una tabella.** A ogni passo della ricerca dei livelli si misura l'errore sulle
righe non viste (ogni riga letta senza sé stessa); il livello di lavoro è il più
generale vicino al migliore. Il risultato si salva come modello della scheda.

**Un testo.** Ogni 10 passi (impostabile), ogni 5 minuti e alla fine:

1. il contatore viene sommato tutto (le somme dell'intero testo letto finora);
2. il **giudizio** sceglie i pesi della lettura: per ciascun livello di ogni
   lettura prova quattordici valori, da 1/32 a 256, e tiene quello che legge
   meglio metà dei pezzi di validazione; due giri; con i percorsi sceglie anche
   quanto pesa ciascuno nella miscela;
3. l'**errore su testo mai visto** si misura sull'altra metà dei pezzi di
   validazione; l'**errore sul testo che sta per leggere** sui 4.096 token
   successivi, non ancora contati;
4. si scrive un **checkpoint**: le somme in un file `safetensors`, lo stato in
   `state.json`, un manifesto con le impronte SHA-256 di ogni file. È lo stesso
   formato di ogni progetto, quindi pausa, ripresa dopo un riavvio, «migliore»,
   salvataggi permanenti e coda funzionano allo stesso modo;
5. il modello risponde alle **domande di prova**, con temperatura 0,7 e un seme
   fisso: le risposte sono ripetibili, e cambiano solo perché il modello ha
   imparato.

### Passo 6 · Usare il modello

**Una tabella.** Il grafo mostra i livelli (i concetti generali in alto, quelli
particolari sotto), le frecce fra livelli (da quali concetti generali discende
ciascuno) e le linee tratteggiate fra concetti vicini. La scheda Valutazione
misura le previsioni sulle righe messe da parte; la scheda Previsione completa
una riga nuova.

**Un testo.** Per scrivere il token successivo il programma cerca il concetto del
contesto attuale per ogni lunghezza, dalla più corta alla più lunga (ricerca
binaria nell'elenco ordinato; nei percorsi per tutti i nodi insieme), combina le
distribuzioni dei token che seguono, ciascuna con il livello sopra come
riferimento, e aggiunge i concetti del testo scritto finora, compresa la frequenza
di ogni token nel testo (la domanda inclusa): così la risposta segue l'argomento
della domanda e non solo i suoi ultimi token, che in una chat sono sempre i segnali
del formato. Con i percorsi le
probabilità dei passi si moltiplicano e i percorsi si mescolano. Ne esce la
probabilità di ogni token del vocabolario, come l'uscita di una rete; si sceglie
il più probabile (temperatura 0) o se ne estrae uno (temperatura più alta, testo
più vario). In una conversazione la domanda è scritta con i segnali
`<|im_start|>user … <|im_end|>` e il modello scrive fino al fine turno.

**Da altri programmi: l'API compatibile OpenAI.** Nella sezione «API compatibile
OpenAI» della scheda Generativo si crea una chiave (`acan-…`, mostrata una volta
sola). Con quella, ogni programma scritto per OpenAI usa il modello cambiando solo
indirizzo (`https://<questo sito>/v1`) e chiave: `GET /v1/models` elenca i
progetti con un salvataggio, `POST /v1/chat/completions` risponde a una
conversazione (anche un pezzo alla volta, con `stream`), `POST /v1/completions`
continua un testo. Il servizio legge il salvataggio dal disco e tiene in memoria
gli ultimi due modelli usati: un progetto risponde anche mentre continua a
imparare.

### Come leggere le grandezze delle pagine

| Voce | Dove | Che cosa dice |
|---|---|---|
| Righe lette in questa lettura | tabella | A che punto è la lettura in corso. |
| Concetti nei blocchi | tabella | Quanti concetti specifici ci sono finora. |
| Concetti a questo passo | tabella | Quanti concetti restano nella ricerca dei livelli, e da quanti è partita. |
| Errore sulle righe non viste | tabella | Il giudizio del passo attuale: scende finché unire concetti non toglie informazione, poi risale. |
| Concetti per livello | testo | Quanti contesti diversi conosce per ogni lunghezza, da 0 a 6 token. |
| Concetti dei token | testo | Quanti concetti di token ha trovato il modello, dal più generale al più particolare. |
| Percorsi verso il prossimo token | testo | Il percorso diretto e quelli attraverso i concetti dei token, con il peso scelto dal giudizio. |
| Concetti generalizzati | testo | Quanti contesti ha tolto per stare nella memoria, lasciandoli al contesto più corto. |
| Memoria dei concetti | testo | Quanto occupano le somme, e quanto è disponibile. |
| Pesi scelti dal giudizio | testo | Per ogni livello, quanto si fida del livello sopra (più alto = più prudente). |
| Passo, token letti | testo | Pezzi contati sul totale; un passo è un milione di token. |
| Errore sul testo che sta per leggere | testo | Misurato prima di contarlo: è un testo mai visto. |
| Errore su testo mai visto | testo | Sulla validazione: è quello che conta. |
| Prima di leggere questo testo | testo | L'errore all'inizio del testo: per il primo è quello di un modello che non sa nulla. |
| Tempo trascorso / che manca, velocità | entrambi | La stima usa la parte già fatta. |
| Memoria del processo | entrambi | Il picco di memoria del processo che impara. |

L'errore del testo è in nat per token: $e^{\text{errore}}$ è fra quanti token, in
media, il modello è incerto. Errore 5 vuol dire circa 150 token possibili; errore
3, circa 20. Il limite di 15 minuti di una tabella vale solo per un addestramento
che non dà più notizie: finché avanza, continua.

### Quanto è veloce

Su un processore di sviluppo:

| | Modello a concetti | Rete micro con concetti (versione precedente) |
|---|---:|---:|
| Imparare un testo (vocabolario di 32.000, un thread) | circa 414.000 token al secondo | circa 765 token al secondo |
| Scrivere | circa 790 token al secondo (circa 70 con i percorsi) | circa 120 token al secondo |
| Imparare una tabella sintetica di due milioni di righe | circa 6 minuti, due processi | — |

Dopo 8 milioni di token le somme occupano circa 920 MB (21,5 milioni di
concetti): con la memoria massima automatica il modello generalizza i concetti
meno utili prima di arrivare lì. Su wikitext-103 (69 milioni di token, 2 GB) il
modello a concetti ha imparato in 7 minuti e sbaglia meno di una rete Llama a 4
strati addestrata 90 minuti sullo stesso processore (3,905 contro 4,011). Sul tuo
server la velocità reale è quella mostrata dalla pagina.

### Limiti, in chiaro

- **Valori simili.** I concetti dei token sono fissati all'inizio, dai primi 20
  milioni di token, e ogni token ne ha uno per livello; una rete neurale dà a ogni
  token un vettore che continua a cambiare. Nella tabella due valori diversi di
  una categoria restano diversi.
- **Memoria.** I contesti di un testo crescono con il testo: oltre la memoria
  massima i meno utili vengono generalizzati, e qualcosa si perde.
- **Contesti oltre 4 token.** Con 32.000 token di vocabolario un contesto di 5 o
  più token è scritto come un numero di 64 bit non più esatto: due contesti
  diversi possono condividerlo, con probabilità di circa $n^2/2^{64}$.
- **Ripetizioni.** Un modello che impara contando ripete volentieri le frasi
  frequenti: le risposte migliorano con più testo e più conversazioni.
- **Tabelle molto grandi.** Oltre un milione di righe il file va addestrato dalla
  libreria (`ConceptModel.fit_chunks`), non dalla pagina; la pagina usa un solo
  processo, la libreria può usarne di più. Un miliardo di righe non è stato
  misurato.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth': '960px', 'margin': 'auto'})
