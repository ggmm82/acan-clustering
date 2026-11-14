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
esatti, quelli più vecchi vengono riassunti in un numero fisso di concetti. Ogni
nuovo pezzo legge pochi pezzi esatti più i concetti, qualunque sia la lunghezza
del testo. Obiettivo: un modello che conversa come un LLM tradizionale, con
memoria fissa e costo per pezzo che non cresce con il testo.

### Token: i pezzi del testo

Il testo viene diviso in **token**: parole intere, pezzi di parola, segni di
punteggiatura. Il modello lavora solo con numeri: ogni token ha un numero, e il
modello impara a prevedere il numero del token successivo. Il tokenizer del modello
nativo impara dai testi dell'addestramento quali pezzi sono frequenti (BPE sui
byte, 32.000 token): così l'italiano non paga i token in più di un vocabolario
costruito per l'inglese, e qualunque carattere si può scrivere.

### Come legge un layer a concetti

Ogni token produce tre vettori: una **domanda** (che cosa cerco), una **chiave**
(come mi riconoscono) e un **valore** (che cosa porto). Nell'attenzione normale
la domanda del token nuovo si confronta con le chiavi di tutti i token
precedenti, e il risultato è una media dei loro valori pesata dalla somiglianza.

In un layer a concetti:

1. gli ultimi 16 token (la **finestra**) restano esatti;
2. quando un token esce dalla finestra, entra in un **concetto**: un riassunto
   che conserva solo somme (quanti token, la somma delle chiavi, dei valori,
   delle posizioni) da cui si ricavano il centro e il valore medio;
3. se i concetti sono più del massimo (16 sulla CPU), i due che si possono unire
   costando meno a chi li legge diventano uno solo;
4. la domanda del token nuovo legge insieme la finestra esatta e i concetti; un
   concetto pesa quanto i token che riassume.

Il numero di vettori letti resta finestra + concetti, a qualunque lunghezza.
Nell'apprendimento lungo di questo laboratorio **tutti i layer** leggono così.

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
  un layer resta finestra + concetti.

### Come si capisce se funziona

1. **Prova in piccolo** (laboratorio): tre modellini identici imparano da zero;
   cambia solo come leggono. Si misura se i concetti ritrovano informazione
   lontana come l'attenzione normale (memoria) e se non peggiorano il testo.
2. **Errore sul riferimento fisso** durante l'addestramento lungo: deve
   scendere nel tempo.
3. **Risposte di prova**: le stesse domande a ogni salvataggio, per leggere i
   miglioramenti con i propri occhi.

### Limiti attuali, in chiaro

- Lo strato a concetti unisce i concetti con un ciclo in sequenza: sui testi
  corti è da 10 a 25 volte più lento dell'attenzione normale; diventa più
  veloce intorno ai 5.000 token di contesto.
- Un modello che nasce da zero deve imparare tutto: la lingua, i fatti, il
  modo di conversare. Sulla CPU il modello micro elabora qualche centinaio di
  token al secondo (circa un miliardo in qualche mese); un chatbot che risponde
  bene richiede miliardi di token e modelli più grandi, quindi una GPU.
- Un modello piccolo resta limitato con o senza concetti: impara a scrivere in
  modo scorrevole prima di sapere i fatti.
- I concetti non riducono i dati necessari: lo strato cambia come si legge il
  contesto, non come si conserva la conoscenza, che resta nei pesi.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth': '960px', 'margin': 'auto'})
