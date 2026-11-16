"""What the software does, step by step, when a concept language model or a table is trained."""
from dash import dcc, html

TEXT = r"""
## Che cosa fa il programma, passo per passo

Questa pagina segue ciò che succede quando premi i pulsanti: prima
l'apprendimento lungo del modello linguistico (scheda Generativo), poi
l'addestramento di una tabella (scheda Modello). Le formule sono nella scheda
Matematica; qui c'è il meccanismo.

---

## Parte A · Il modello linguistico a concetti

**In breve.** Un modello linguistico scrive un pezzo alla volta: a ogni passo
guarda ciò che c'è prima e sceglie il pezzo successivo. Nei modelli
tradizionali ogni pezzo nuovo viene confrontato con *tutti* i precedenti
(l'attenzione): più il testo è lungo, più lavoro e più memoria servono. Qui
ogni pezzo legge gli ultimi 16 esatti e, per tutto il resto, pochi **concetti**:
precisi per il passato vicino, più generali per quello lontano.

### Passo 1 · Preparare i dati

Premendo «Prepara i dati» il programma scarica le fonti una alla volta da
versioni fissate, controlla ogni file, tiene solo il testo e cancella il resto.
Divide i documenti in tre parti: quelli **per imparare**, circa l'1% di
**validazione** (mai usati per imparare, servono a misurare) e una parte di
**test** tenuta da parte. La preparazione si può mettere in pausa e riprendere.

### Passo 2 · Il testo diventa numeri

Un **tokenizer** divide il testo in *token*: parole intere, pezzi di parola,
segni. Lo si costruisce dai testi stessi, all'inizio: guarda quali sequenze di
caratteri sono frequenti e ne fa un vocabolario di 32.000 pezzi. Così
l'italiano non paga i pezzi in più di un vocabolario fatto per l'inglese, e
qualunque carattere si può scrivere.

Tutto ciò che diventa una sequenza di token si può insegnare: il codice è
testo; un'immagine diventa 32 righe di 32 punti; un suono migliaia di numeri
al secondo; un'animazione una serie di piccole immagini.

### Passo 3 · Il modello nasce

Il modello è una pila di strati (4 nella misura «micro», 8 nella «piccolo», 12
nella «base»). Tutti i pesi partono casuali, dal seme del progetto: nessun
modello già addestrato, nessun adattatore. All'inizio l'errore è circa 10,4:
il logaritmo di 32.000, cioè tirare a caso fra tutti i token.

### Passo 4 · Un passo di addestramento

1. Si prende un **pezzo** di testo di 256 token.
2. Ogni token diventa un vettore e attraversa gli strati. In ogni strato deve
   leggere i token precedenti; lo fa così:
   - gli ultimi 16 (la **finestra**) li legge uno per uno;
   - i token più vecchi sono raccolti in **blocchi** di 16, 32, 64, 128…
     token, come le cifre di un contachilometri in base due. Un blocco ha al
     massimo 16 concetti: finché i token sono pochi restano esatti, poi si
     uniscono a due a due quelli che costano meno a chi legge. Due token uguali
     si uniscono gratis; un token che è stato letto molto resta da solo;
   - il token legge insieme la finestra e i blocchi del suo passato.
3. Dopo l'ultimo strato il modello dà, per ogni posizione, una probabilità a
   ogni token del vocabolario. L'**errore** è quanto poca ne ha data al token
   vero che seguiva.
4. L'errore dice come cambiare ogni peso per sbagliare meno. Si accumulano 8
   pezzi, poi si corregge: questo è **un passo** (2.048 token letti).

I blocchi dipendono solo dalla posizione, non dal testo: per questo tutti i
token di un pezzo vengono letti con poche operazioni fatte insieme, senza un
ciclo che avanzi un token alla volta.

### Passo 5 · Salvare, riprendere, mettere in coda

Ogni 100 passi e ogni 5 minuti il programma salva un **checkpoint** (pesi,
stato della correzione, punto raggiunto nei dati) e misura l'errore sul testo
di validazione. Dopo una pausa o un riavvio del server riprende da lì, da solo.
Tiene gli ultimi tre, il migliore e, se richiesto, uno ogni N passi per sempre.

I testi della **coda** vengono imparati uno dopo l'altro: ognuno parte dai pesi
lasciati dal precedente.

### Passo 6 · Le domande di prova

A ogni salvataggio il modello risponde alle stesse domande, sempre scegliendo
il token più probabile: le differenze fra «prima» e «adesso» vengono solo da
ciò che ha imparato. All'inizio le risposte sono ripetizioni; poi compaiono
frasi corrette nella forma e sbagliate nei fatti; i fatti arrivano per ultimi.

### Passo 7 · Come scrive

Il modello dà una probabilità a ogni token, se ne sceglie uno, lo si aggiunge e
si ripete. Con temperatura 0 si prende sempre il più probabile; più alta, il
testo è più vario e meno preciso. In una conversazione la domanda è scritta con
i segnali `<|im_start|>user … <|im_end|>` e il modello scrive fino al segnale
di fine turno.

Mentre scrive, ogni strato tiene la finestra e i blocchi, e li aggiorna a ogni
token: dopo 1.000 token sono al massimo 16 + 16 × 10 vettori, dopo un milione
16 + 16 × 20, contro 1.000 e un milione dell'attenzione normale.

### Come leggere le grandezze della pagina

| Voce | Che cosa dice |
|---|---|
| Parametri totali / che imparano | Quanti pesi ha il modello; nel modello nato da zero imparano quasi tutti. |
| Layer a concetti | Quanti strati leggono attraverso i concetti: tutti. |
| Memoria di lettura per layer | Finestra + concetti per livello: i vettori che uno strato tiene per leggere un pezzo. |
| Passo | Correzioni fatte sul totale previsto per questo testo. |
| Token elaborati | Token letti finora (passi × 2.048). |
| Velocità | Token letti al secondo, misurati sul server. |
| Tempo che manca | Token rimasti diviso la velocità attuale. |
| Errore sul testo di addestramento | L'errore sui pezzi appena letti: oscilla, perché ogni passo legge testo diverso. |
| Errore su testo mai visto | L'errore sulla validazione: è quello che conta. Se risale mentre l'altro scende, il modello impara a memoria. |
| Al passo 0 | L'errore prima di imparare: il punto di partenza. |
| Passo di apprendimento | Quanto forte è ogni correzione; sale all'inizio e poi cala. |
| Ampiezza della correzione | Norma del gradiente: valori stabili sono un buon segno, salti grandi no. |
| Checkpoint tenuti / spazio | Quanti salvataggi ci sono e quanto disco occupano. |

L'errore è un logaritmo: $e^{\text{errore}}$ è fra quanti token, in media, il
modello è incerto. Errore 5 vuol dire circa 150 token possibili; errore 3,
circa 20.

Il **controllo** è lo stesso modello, sugli stessi dati, con l'attenzione
normale: se le due linee del grafico restano vicine, i concetti non costano
qualità.

### Quanto è veloce

Token al secondo in addestramento, modello micro intero, un processore di
sviluppo con un thread:

| Token per pezzo | Attenzione normale | Concetti |
|---:|---:|---:|
| 256 | 886 | 765 |
| 1.024 | 907 | 812 |
| 2.048 | 581 | 792 |

Solo gli strati che leggono, a contesti più lunghi:

| Token per pezzo | Attenzione normale | Concetti |
|---:|---:|---:|
| 2.048 | 956 | 2.025 |
| 4.096 | 476 | 1.892 |
| 8.192 | 250 | 1.845 |

L'attenzione normale rallenta con la lunghezza; i concetti quasi no. Sul
server la velocità reale è quella mostrata dalla pagina: dipende dal
processore.

### Limiti, in chiaro

- Sui pezzi di 256 token i concetti sono un po' più lenti dell'attenzione
  normale; conviene sui contesti lunghi.
- Un modello nato da zero deve imparare tutto: la lingua, i fatti, il modo di
  conversare. Il modello micro impara a scrivere in modo scorrevole molto prima
  di sapere i fatti; un chatbot che risponde bene richiede miliardi di token e
  modelli più grandi, quindi una GPU.
- I concetti non riducono i dati necessari: cambiano come si legge il
  contesto, non dove si conserva la conoscenza, che resta nei pesi.
- La memoria di lettura non è fissa: cresce di un livello a ogni raddoppio del
  testo.

---

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
   dei token del passo 4 qui sopra.
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
