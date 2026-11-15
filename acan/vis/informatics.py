"""How the language model of the project works, in plain words: goal, learning, generation."""
from dash import dcc, html

TEXT = r"""
## Informatica: come funziona il modello linguistico a concetti

### L'obiettivo del progetto

Un modello linguistico (LLM) scrive un testo un pezzo alla volta: a ogni passo
guarda tutto quello che c'è prima e sceglie il pezzo successivo. Nei modelli
tradizionali questo «guardare prima» si chiama **attenzione**: ogni nuovo pezzo
viene confrontato con **tutti** i pezzi precedenti, uno per uno.

Il problema: più il testo è lungo, più confronti servono. Il costo di un passo
cresce con la lunghezza del testo, e quello di tutto il testo cresce con il suo
quadrato (testo lungo il doppio, lavoro quattro volte); anche la memoria cresce,
perché si conserva un vettore per ogni pezzo letto.

ACAN sostituisce l'attenzione con i **concetti**: gli ultimi pezzi restano
esatti, quelli più vecchi vengono riassunti in pochi concetti: precisi per il
passato vicino, più generali per quello lontano. Ogni nuovo pezzo legge pochi
pezzi esatti più i concetti. Obiettivo: un modello che conversa come un LLM
tradizionale, con memoria e costo per pezzo che crescono appena con il testo
(con il suo logaritmo: raddoppiare il testo aggiunge un solo livello).

### Token: i pezzi del testo

Il testo viene diviso in **token**: parole intere, pezzi di parola, segni di
punteggiatura. Il modello lavora solo con numeri: ogni token ha un numero, e il
modello impara a prevedere il numero del token successivo. Il tokenizer del modello
nativo impara dai testi dell'addestramento quali pezzi sono frequenti (BPE sui
byte, 32.000 token): così l'italiano non paga i token in più di un vocabolario
costruito per l'inglese, e qualunque carattere si può scrivere.

### Non solo parole: codice, immagini, suoni

Tutto ciò che diventa una sequenza di token si può insegnare. Il codice è
testo: basta che il tokenizer lo veda all'inizio, così impara pezzi come
`    return`. Un'immagine diventa 32 righe di 32 pixel, ognuno uno di 256 colori;
un suono 8.000 numeri al secondo; un'animazione una serie di piccole immagini.
Il modello li legge e li scrive come parole, e la pagina li mostra come
immagini, suoni e animazioni. Sono sequenze lunghe: è qui che la memoria piccola
dei concetti conviene di più.

### Come legge un layer a concetti

Ogni token produce tre vettori: una **domanda** (che cosa cerco), una **chiave**
(come mi riconoscono) e un **valore** (che cosa porto). Nell'attenzione normale
la domanda del token nuovo si confronta con le chiavi di tutti i token
precedenti, e il risultato è una media dei loro valori pesata dalla somiglianza.

In un layer a concetti:

1. gli ultimi 16 token (la **finestra**) restano esatti;
2. quando un token esce dalla finestra, diventa un **concetto** di un solo
   membro: un riassunto che conserva solo somme (quanti token, la somma delle
   chiavi, dei valori, delle posizioni, e quanto è stato letto finché era nella
   finestra), da cui si ricavano il centro e il valore medio;
3. i concetti sono organizzati a **livelli**, come un contachilometri in base
   due: un blocco copre 1, 2, 4, 8… token consecutivi e contiene al massimo 16
   concetti. Due blocchi uguali, appena sono completi, diventano un blocco
   grande il doppio;
4. finché i due blocchi hanno insieme al massimo 16 concetti restano esatti;
   se ne hanno di più, si uniscono a due a due i concetti che costano meno a chi
   li legge, finché ne restano 16. È la stessa regola con cui il modello
   tabulare generalizza le righe di una tabella: due copie dello stesso
   concetto si uniscono gratis, un token che è stato letto molto resta da solo;
5. la domanda del token nuovo legge insieme la finestra esatta e i blocchi del
   suo passato: quelli vicini sono piccoli e precisi, quelli lontani grandi e
   generali. Un concetto pesa quanto i token che riassume.

I livelli crescono con il logaritmo della lunghezza: dopo 1.000 token un layer
tiene al massimo 16 + 16 × 10 vettori, dopo un milione 16 + 16 × 20, contro
1.000 e un milione dell'attenzione normale. Nell'apprendimento lungo di questo
laboratorio **tutti i layer** leggono così.

### Perché non c'è più il ciclo lento

Nella versione precedente (`concept-graph-v1`) c'era un solo gruppo di concetti
per tutto il passato: ogni 16 token nuovi un ciclo univa una coppia alla volta,
e ogni unione dipendeva dalla precedente. Per questo un pezzo di testo andava
letto a tratti, in sequenza, e il calcolo non si poteva fare tutto insieme.

Ora (`concept-graph-v2`) **quando** si unisce e **fra quali** concetti è deciso
dalla posizione, non da una ricerca: i blocchi sono sempre gli stessi per ogni
testo della stessa lunghezza. Quindi:

- durante l'addestramento tutti i blocchi di un livello si calcolano insieme,
  con poche operazioni fra matrici, e tutti i token del pezzo leggono in un
  colpo solo; i passi sono tanti quanti i livelli (circa il logaritmo della
  lunghezza), come i layer di una rete;
- dentro un blocco le unioni sono un numero fisso (da 32 concetti a 16: sedici
  unioni) e si fanno per tutti i blocchi contemporaneamente; dopo ogni unione
  si ricalcola solo la riga del concetto nuovo;
- mentre scrive, il modello tiene gli stessi blocchi e li aggiorna un token alla
  volta: il risultato è quello dell'addestramento (identico a dieci cifre in
  doppia precisione; nei modelli veri, in precisione singola, due unioni di
  costo quasi uguale possono scambiarsi e la lettura cambia di poco), anche
  se riceve un prompt intero e poi continua.

Che cosa **non** è cambiato: che cos'è un concetto, come lo si legge e il costo
con cui si decide chi unire. Con abbastanza concetti il layer **è** l'attenzione
normale (verificato). I progetti creati prima continuano con il loro strato.

### Come impara

- **Obiettivo.** Il modello vede un pezzo di testo e, per ogni posizione, deve
  dare la probabilità del token vero successivo. L'errore è il logaritmo
  negativo di quella probabilità, mediato: **NLL**. Più basso è meglio; 0
  vorrebbe dire certezza assoluta. La **perplessità** ($e^{\text{NLL}}$) è fra
  quanti token in media il modello è incerto.
- **Correzione.** A ogni passo l'errore indica come cambiare i pesi per
  sbagliare meno (discesa del gradiente, ottimizzatore AdamW). Un passo legge
  8 pezzi di testo (accumulo) prima di correggere.
- **Che cosa si addestra.** Il modello nasce da zero: tutti i pesi partono
  casuali (dal seme del progetto) e tutti imparano, senza un modello
  preaddestrato e senza adattatori. Anche il **tokenizer** nasce dai testi
  dell'addestramento, italiano e inglese insieme. Dimensioni: micro (circa 12
  milioni di parametri, 4 layer), piccolo (circa 50 milioni, 8 layer), base
  (circa 138 milioni, 12 layer, solo con GPU). I concetti non hanno pesi
  propri: si formano dalle chiavi del testo.
- **Validazione.** Una parte dei testi (circa l'1%) non viene mai usata per
  imparare. A ogni salvataggio il modello viene misurato lì: se l'errore su
  quel testo smette di scendere mentre quello di addestramento scende ancora,
  il modello sta imparando a memoria invece di generalizzare.
- **Salvataggi.** Ogni 100 passi e ogni 5 minuti si salva un checkpoint
  (pesi, ottimizzatore, posizione nei dati): dopo una pausa o un riavvio
  l'addestramento riprende esattamente da lì. Si tengono gli ultimi tre, il
  migliore sulla validazione e, se lo si chiede, uno ogni N passi per sempre.
- **Coda di apprendimento.** I testi vengono imparati uno dopo l'altro; ognuno
  riparte dai pesi lasciati dal precedente, quindi niente di quanto imparato
  viene buttato. Ogni testo ha il suo avanzamento.

### Come genera e come conversa

- **Un token alla volta.** Il modello dà una probabilità a ogni token del
  vocabolario; se ne sceglie uno, lo si aggiunge al testo e si ripete.
- **Scelta.** Con *temperatura 0* si prende sempre il più probabile (risposte
  ripetibili: le usano le domande di prova); con temperatura più alta si
  pesca secondo le probabilità, e il testo è più vario e meno preciso.
- **Conversazione.** I dati di dialogo sono scritti con dei segnali che dicono
  chi parla: `<|im_start|>user` … `<|im_end|>` per la persona,
  `<|im_start|>assistant` … `<|im_end|>` per il modello. Quando gli si fa una
  domanda, il modello riceve la domanda in questo formato e scrive fino al
  segnale di fine turno.
- **Memoria mentre scrive.** Con l'attenzione normale ogni token scritto
  aggiunge un vettore alla memoria di ogni layer. Con i concetti la memoria di
  un layer è la finestra più 16 concetti per livello: cresce con il logaritmo
  del testo.

### Come si capisce se funziona

1. **Prova in piccolo** (laboratorio): tre modellini identici imparano da zero;
   cambia solo come leggono. Si misura se i concetti ritrovano informazione
   lontana come l'attenzione normale (memoria) e se non peggiorano il testo.
2. **Errore sul riferimento fisso** durante l'addestramento lungo: deve
   scendere nel tempo.
3. **Risposte di prova**: le stesse domande a ogni salvataggio, per leggere i
   miglioramenti con i propri occhi.

### Quanto è veloce (misurato)

Sul processore di sviluppo, un thread, modello micro (4 layer, 256 dimensioni),
token al secondo in addestramento:

| Token per pezzo | Attenzione normale | Concetti di prima (v1) | Concetti di adesso (v2) |
|---:|---:|---:|---:|
| 256 | 3.534 | 542 | 1.543 |
| 1.024 | 2.297 | 494 | 1.880 |
| 2.048 | 956 | 437 | 2.025 |
| 4.096 | 476 | — | 1.892 |
| 8.192 | 250 | — | 1.845 |

Sono i soli layer che leggono. Con tutto il resto del modello nativo micro
(vocabolario di 32.000 token, strati interni), lo stesso confronto dà 886, 403 e
765 token al secondo a 256 token, e 581, 337 e 792 a 2.048.

Mentre scrive, per layer, dopo N token di contesto:

| Contesto | Attenzione normale | v1 | v2 |
|---:|---:|---:|---:|
| 256 | 3.644 | 678 | 1.682 |
| 1.024 | 1.924 | 892 | 1.531 |
| 4.096 | 324 | 1.047 | 819 |
| 16.384 | 67 | 917 | 1.254 |

Dopo 16.384 token l'attenzione normale tiene 16.384 vettori per layer, i
concetti 64. (Il valore a 4.096 cade nel momento in cui molti blocchi si
uniscono insieme: è il caso peggiore, non la media.)

L'attenzione normale rallenta con la lunghezza; i concetti no. Sui pezzi corti
l'attenzione normale resta la più veloce, perché le sue moltiplicazioni fra
matrici sono le più ottimizzate che esistano.

### Limiti attuali, in chiaro

- Sui pezzi di 256 token lo strato a concetti è ancora più lento
  dell'attenzione normale; la supera sui contesti lunghi (tabella sopra).
- Le prove di qualità sono su modellini e su pochi semi: dicono che il
  meccanismo non perde rispetto a quello di prima, non che un modello grande
  converserà bene.
- La memoria non è più fissa ma cresce con il logaritmo della lunghezza: è il
  prezzo per poter calcolare tutto insieme.
- Un modello che nasce da zero deve imparare tutto: la lingua, i fatti, il
  modo di conversare. Un chatbot che risponde bene richiede miliardi di token
  e modelli più grandi, quindi una GPU.
- Un modello piccolo resta limitato con o senza concetti: impara a scrivere in
  modo scorrevole prima di sapere i fatti.
- I concetti non riducono i dati necessari: lo strato cambia come si legge il
  contesto, non come si conserva la conoscenza, che resta nei pesi.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth': '960px', 'margin': 'auto'})
