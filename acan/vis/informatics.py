"""What the software does, step by step, when a concept language model or a table is trained."""
from dash import dcc, html

TEXT = r"""
## Che cosa fa il programma, passo per passo

Questa pagina segue ciò che succede quando premi i pulsanti: prima
l'apprendimento lungo del modello che scrive (scheda Generativo), poi
l'addestramento di una tabella (scheda Modello). È lo stesso modello a concetti
nei due casi. Le formule sono nella scheda
Matematica; qui c'è il meccanismo.

---

## Parte A · Il modello a concetti che scrive

**In breve.** Un modello linguistico sceglie a ogni passo il pezzo di testo
successivo. Qui non c'è una rete neurale: il testo è trattato come una tabella e
il modello è quello della scheda Modello. Ogni posizione del testo è una riga; i
token che la precedono sono le colonne note; il token in quella posizione è la
colonna da prevedere. Imparare vuol dire **contare**: per ogni contesto (gli
ultimi 1, 2, … 6 token) quante volte è comparso e da quali token è stato seguito.

### Passo 1 · Preparare i dati

Premendo «Prepara i dati» il programma scarica le fonti una alla volta da
versioni fissate, controlla ogni file, tiene solo il testo e cancella il resto.
Divide i documenti in tre parti: quelli **per imparare** (80%), quelli di
**validazione** (10%, mai contati: servono a giudicare) e quelli di **test**
(10%, tenuti da parte fino alla verifica finale). La preparazione si può mettere
in pausa e riprendere.

### Passo 2 · Il testo diventa token

Un **tokenizer** divide il testo in token: parole intere, pezzi di parola,
segni. Lo si costruisce dai testi stessi, all'inizio: guarda quali sequenze di
caratteri sono frequenti e ne fa un vocabolario di 32.000 pezzi. Così
l'italiano non paga i pezzi in più di un vocabolario fatto per l'inglese, e
qualunque carattere si può scrivere. I token del testo da imparare vengono
scritti su disco una volta sola (file `train.bin`, `validation.bin`, `test.bin`).

### Passo 3 · Un passo di apprendimento: contare un pezzo

1. Si leggono dal disco i prossimi **1.000.000 di token** del testo da imparare.
2. Per ogni posizione e per ogni lunghezza di contesto $j$ da 0 a 6 si calcola
   un numero che identifica il contesto (gli ultimi $j$ token). Sono operazioni
   su vettori, senza cicli sulle posizioni.
3. Si ordinano questi numeri e si contano i doppioni (`numpy.unique`): ne escono,
   per ogni contesto, quante volte compare e quante volte è seguito da ciascun
   token. Sono le **somme dei concetti** di quel pezzo.
4. Le somme del pezzo entrano in un **contatore in base due**: quando ci sono
   due tabelle grandi uguali si sommano in una tabella grande il doppio, come i
   blocchi della scheda Matematica. Sommare due tabelle vuol dire unire due
   elenchi ordinati di contesti e sommare i conteggi dei contesti uguali.

Non ci sono pesi da correggere né gradienti da calcolare: il passo è finito
quando il pezzo è contato.

### Passo 4 · Restare nella memoria

I contesti diversi crescono con il testo. Se le somme superano la **memoria
massima dei concetti** (automatica: un decimo della memoria del server, al
massimo 2 GB, perché mentre somma due tabelle il processo ne usa circa il
triplo; si può impostare), il programma calcola per ogni concetto quanto
peggio sarebbero letti i suoi membri dal contesto più corto, e toglie i concetti
che costano meno finché le somme rientrano con un margine del 10%. Le loro somme
restano nel livello sopra: si perde solo la distinzione. Il numero di concetti
così tolti è fra le grandezze della pagina.

### Passo 5 · Salvare e giudicare

Ogni 100 passi e ogni 5 minuti:

1. il contatore viene sommato tutto (le somme dell'intero testo letto finora);
2. il **giudizio** sceglie i pesi della lettura: per ciascun livello prova
   quattordici valori, da 1/32 a 256, e tiene quello che legge meglio metà dei
   pezzi di validazione; due giri;
3. l'**errore su testo mai visto** si misura sull'altra metà dei pezzi di
   validazione; l'**errore sul testo che sta per leggere** sui 4.096 token
   successivi, che non sono ancora stati contati;
4. si scrive un **checkpoint**: le somme di tutti i livelli in un file
   `safetensors`, lo stato in `state.json`, un manifesto con le impronte SHA-256
   di ogni file. È lo stesso formato dei progetti precedenti, quindi pausa,
   ripresa dopo un riavvio, «migliore», salvataggi permanenti e coda funzionano
   allo stesso modo;
5. il modello risponde alle **domande di prova**.

Un testo successivo della coda parte dal checkpoint del precedente e aggiunge le
sue somme.

### Passo 6 · Come scrive

Per scrivere il token successivo:

1. per ogni lunghezza di contesto, dalla più corta alla più lunga, si cerca il
   concetto del contesto attuale (ricerca binaria nell'elenco ordinato);
2. si combinano le distribuzioni dei token che seguono, ciascuna con il livello
   sopra come riferimento, con i pesi scelti dal giudizio;
3. si aggiungono i concetti del testo in corso: gli stessi contesti contati solo
   sul testo scritto finora, così il modello può riprendere ciò che ha già detto;
4. si sceglie un token: il più probabile (temperatura 0) oppure estratto a caso
   secondo le probabilità (temperatura più alta, testo più vario).

Le domande di prova usano temperatura 0,7 con un seme fisso: le risposte sono
ripetibili, e con temperatura 0 un modello che conta ripete la sua frase più
frequente. In una conversazione la domanda è scritta con i segnali
`<|im_start|>user … <|im_end|>` e il modello scrive fino al segnale di fine turno.

### Come leggere le grandezze della pagina

| Voce | Che cosa dice |
|---|---|
| Concetti per livello | Quanti contesti diversi conosce per ogni lunghezza, da 0 a 6 token. |
| Concetti generalizzati | Quanti contesti ha tolto per stare nella memoria, lasciandoli al contesto più corto. |
| Memoria dei concetti | Quanto occupano le somme, e quanto è disponibile. |
| Pesi scelti dal giudizio | Per ogni livello, quanto si fida del livello sopra (più alto = più prudente). |
| Passo, token letti | Pezzi contati sul totale del testo; un passo è un milione di token. |
| Velocità, tempo che manca | Token contati al secondo e la stima del resto. |
| Errore sul testo che sta per leggere | Misurato prima di contarlo: è un testo mai visto. |
| Errore su testo mai visto | Sulla validazione: è quello che conta. |
| Prima di leggere questo testo | L'errore all'inizio del testo: per il primo è il livello di un modello che non sa nulla. |

L'errore è in nat per token: $e^{\text{errore}}$ è fra quanti token, in media, il
modello è incerto. Errore 5 vuol dire circa 150 token possibili; errore 3,
circa 20.

### Quanto è veloce

Misurato su un processore di sviluppo, un thread, vocabolario di 32.000 token,
pezzi da un milione di token di un testo artificiale con frequenze come quelle di
una lingua e frasi che si ripetono:

| | Modello a concetti | Rete micro con concetti (versione precedente) |
|---|---:|---:|
| Imparare | circa 414.000 token al secondo | circa 765 token al secondo |
| Scrivere | circa 790 token al secondo | circa 120 token al secondo |

(La rete, stesso processore: pezzi di 256 token; scrittura di 128 token dopo un
prompt di 64. Sul tuo server la rete faceva circa 390 token al secondo.)

Dopo 8 milioni di token le somme occupano circa 920 MB (21,5 milioni di
concetti) e il processo arriva a 3 GB mentre somma. Con la memoria massima
automatica il modello generalizza i concetti meno utili prima di arrivare lì.
Sul tuo server la velocità reale è quella mostrata dalla pagina.

### Limiti, in chiaro

- **Valori simili.** Come nella tabella due città diverse sono semplicemente
  diverse, così nel testo «gatto» e «cane»: due contesti diversi si aiutano solo
  attraverso il contesto più corto che hanno in comune. Una rete neurale impara
  che certi token si somigliano; con molti dati potrebbe tornare in vantaggio
  (non misurato).
- **Memoria.** I contesti diversi crescono con il testo: oltre la memoria
  massima i meno utili vengono generalizzati, e qualcosa si perde.
- **Contesti lunghi oltre 4 token.** Con 32.000 token di vocabolario un contesto
  di 5 o più token è scritto come un numero di 64 bit che non è più esatto: due
  contesti diversi possono condividerlo, con probabilità di circa
  $n^2/2^{64}$ (trascurabile per miliardi di contesti, non nulla).
- Un modello che impara contando ripete volentieri le frasi frequenti: le
  risposte migliorano con più testo e più conversazioni.
## Parte B · Una tabella

### Passo 1 · Caricare il file

Il file viene letto e ripulito. Se ha più di 10.000 righe la pagina lavora su
un **campione** di 10.000 (grafici, filtri, anteprima), mentre la **tabella
intera** resta sul server per l'addestramento, fino a un milione di righe.

### Passo 2 · Scegliere che cosa prevedere

Il programma propone un target e le colonne da escludere (identificativi, testo
libero). Senza target cerca solo i gruppi. Una parte delle righe (il 20%) è
messa da parte per la valutazione.

### Passo 3 · Addestrare

Con una tabella piccola il programma cerca subito i livelli. Con una tabella
oltre le 20.000 righe la legge tre volte, un pezzo alla volta, e la barra di
avanzamento dice in quale lettura si trova:

1. **Lettura 1 · scala degli attributi.** Media e deviazione di ogni colonna,
   e 20.000 righe estratte a caso.
2. **Lettura 2 · concetti specifici a blocchi.** Ogni riga entra una volta in
   un blocco; i blocchi uguali si uniscono a due a due. È lo stesso meccanismo
   del contatore del passo 3 qui sopra.
3. **Ricerca dei livelli.** Dai concetti dei blocchi a un solo concetto,
   unendo un ventesimo dei concetti alla volta e giudicando ogni passo sulle
   righe non viste. Si fa due volte: prima i concetti generali, poi ciò che più
   concetti generali hanno in comune.
4. **Lettura 3 · ogni riga aggiunta ai concetti finali**, così le loro somme
   sono sull'intera tabella.

Durante l'addestramento la pagina mostra:

| Voce | Che cosa dice |
|---|---|
| Righe lette in questa lettura | A che punto è la lettura in corso. |
| Concetti nei blocchi | Quanti concetti specifici ci sono finora. |
| Concetti a questo passo | Quanti concetti restano nella ricerca dei livelli, e da quanti è partita. |
| Errore sulle righe non viste | Il giudizio del passo attuale: meno è meglio. Scende finché unire concetti non toglie informazione, poi risale. |
| Tempo trascorso / che manca | La stima usa la parte già fatta; nella ricerca dei livelli è approssimata. |
| Memoria del processo | Il picco di memoria del processo che addestra. |

Il limite di 15 minuti vale solo per un addestramento che non dà più notizie:
finché avanza, continua.

### Passo 4 · Leggere il risultato

Il grafo mostra i livelli (i concetti generali in alto, quelli particolari
sotto), le frecce fra livelli (da quali concetti generali discende ciascuno) e
le linee tratteggiate fra concetti vicini. La scheda Valutazione misura le
previsioni sulle righe messe da parte; la scheda Previsione completa una riga
nuova.

### Limiti, in chiaro

- Oltre un milione di righe il file va addestrato dalla libreria
  (`ConceptModel.fit_chunks`), non dalla pagina.
- La pagina addestra con un solo processo; la libreria può usarne di più.
- Su due milioni di righe sintetiche l'addestramento ha richiesto circa sei
  minuti con due processi. Un miliardo di righe non è stato misurato.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth': '960px', 'margin': 'auto'})
