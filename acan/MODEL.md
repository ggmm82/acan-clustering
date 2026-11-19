# Il modello a concetti

## Matematica

### L'idea in una frase

Il modello impara da una **tabella** e completa le colonne che mancano in una riga
nuova. Lo stesso modello serve per due lavori:

- **tabelle**: date alcune colonne (peso, cilindrata) completa la colonna target (prezzo);
- **testo**: dati i token già scritti completa la colonna «prossimo token», poi la
  riga successiva, e così via. Così scrive.

Non c'è una rete neurale e non c'è un gradiente: il modello conserva solo **somme**
delle righe che ha visto, raggruppate in **concetti**.

### Passo 1 · Una tabella piccola

Seguiamo una tabella di quattro automobili (numeri inventati, per l'esempio):

| riga | peso (t) | cilindrata (cc) | prezzo (k€) |
|---:|---:|---:|---:|
| 1 | 1,0 | 1000 | 12 |
| 2 | 1,1 | 1200 | 14 |
| 3 | 1,9 | 2800 | 40 |
| 4 | 2,0 | 3000 | 44 |

A occhio ci sono due gruppi: le utilitarie (righe 1 e 2) e le auto grandi (righe 3
e 4). Il modello li trova da solo, con i passi seguenti.

### Passo 2 · Che cosa conserva un concetto

Un concetto è un gruppo di righe. Non conserva le righe, ma solo tre somme per ogni
colonna numerica: quante righe ci sono, la somma dei valori e la somma dei quadrati.
Per una colonna di categorie conserva quante volte compare ciascun valore.

Una riga può appartenere un po' a più concetti. Se $T_{ri}$ è la quota della riga
$r$ nel concetto $i$, la massa e la media del concetto sono

$$
C_i=\sum_r T_{ri},\qquad
\mu_{ij}=\frac{\sum_{r}T_{ri}\,x_{rj}}{\sum_{r}T_{ri}} .
$$

Dalla somma dei quadrati si ricava la dispersione, cioè quanto i membri si
allontanano dalla media. Nell'esempio il concetto «utilitarie» ha massa 2, peso medio
1,05 t, cilindrata media 1100 cc e prezzo medio 13 k€. Unire due concetti vuol dire
**sommare** le loro somme. Un valore mancante non entra nelle somme della sua colonna.

### Passo 3 · A quale concetto appartiene una riga

Una riga nuova appartiene di più al concetto che la «spiega» meglio: un valore
vicino al centro di un concetto stretto è molto probabile; un concetto disperso
spiega valori più lontani ma ciascuno con meno probabilità. Se $L_i(x)$ è questa
probabilità per le colonne note della riga $x$, la sua quota nel concetto $i$ è

$$
T_i(x)=\frac{C_i\,L_i(x)}{\sum_h C_h\,L_h(x)} .
$$

Conta anche la massa $C_i$: a parità di somiglianza pesa di più il concetto che ha
visto più righe. Le quote di una riga sommano a 1.

Esempio: un'auto di 1,2 t e 1300 cc è molto vicina alle utilitarie e lontanissima
dalle auto grandi: $T_{\rm utilitarie}\approx 0{,}98$, $T_{\rm grandi}\approx 0{,}02$.

### Passo 4 · Quanti concetti e quanti livelli: li sceglie il modello

Il modello parte da tanti piccoli gruppi e li unisce a due a due, cominciando dalle
coppie che costano meno. Il **raggio** $\rho$ di un gruppo è la sua dispersione media
(la radice), misurata rispetto al livello sopra. Il costo di unire i gruppi $i$ e
$j$ è quanto cresce il raggio pesato per la massa:

$$
\Delta_{ij}=(C_i+C_j)\,\rho_{i\cup j}-C_i\,\rho_i-C_j\,\rho_j .
$$

Unire le righe 1 e 2 costa poco (si somigliano); unire le utilitarie con le auto
grandi costa molto. Ogni momento di questa sequenza di unioni è un possibile modo di
tagliare la tabella in gruppi. Per scegliere il taglio il modello usa la **CP a
coppie**: prende due membri diversi dello stesso gruppo e chiede se si somigliano più
di due righe qualunque del livello sopra. Con $r_{ij}$ la dispersione dell'attributo
$j$ nel gruppo $i$ divisa per quella di tutta la tabella, il punteggio di un taglio è

$$
\mathrm{CP}^{\circ}=
\frac{\sum_i C_i\max(C_i-1,0)\,\frac1m\sum_j [1-\min(2,r_{ij})]}
     {\max(n(n-1),\epsilon)} .
$$

In parole: ogni gruppo porta tante coppie di membri quante ne contiene, e ogni coppia
vale di più se il gruppo è stretto rispetto alla tabella. Un gruppo piccolo porta
poche coppie; un gruppo grande disperso come tutta la tabella non aggiunge
somiglianza. Per una colonna di categorie la dispersione è la probabilità che due
membri presi a caso abbiano valori diversi.

Il modello poi **giudica** la scelta su righe che non ha usato: legge ogni riga
togliendone il contributo dalle somme. Fra i tagli che stanno entro un errore
standard dal migliore sceglie il più semplice. Dentro ogni gruppo può ripetere la
ricerca e trovare un livello più specifico, se ce n'è la prova. Anche un solo gruppo è
un risultato possibile. Nell'esempio il risultato è: due concetti.

### Passo 5 · Predire la colonna che manca

Per un numero (il prezzo) si leggono le medie dei concetti, pesate con le quote della
riga (in k€):

$$
\hat y = \sum_i T_i(x)\,\mu_{i,\rm prezzo}\approx 0{,}98\cdot 13+0{,}02\cdot 42\approx 13{,}6 .
$$

Per una categoria (come il prossimo token) ogni concetto dà la sua distribuzione dei
valori. Con $N_i(y)$ la massa del valore $y$ nel concetto, $M_i$ la massa totale e
$D_i$ il numero di valori diversi osservati, il modello legge

$$
q_i(y)=\frac{N_i(y)+a_i\,q^{\rm sopra}(y)}{M_i+a_i},
\qquad a_i=\kappa\max(1,D_i),\qquad
p(y\mid x)=\sum_i T_i(x)\,q_i(y).
$$

$q^{\rm sopra}$ è la stima del livello più generale: quando un concetto ha visto poco,
la sua lettura resta vicina a quella stima. Il numero $\kappa$ si sceglie su righe di
validazione. Esempio: un concetto ha visto dopo di sé «.», «.» ed «e» (tre
osservazioni, due valori diversi); con $\kappa=1$ e una stima generale del punto pari a
$1/4$, il punto riceve $(2+2\cdot\tfrac14)/(3+2)=\tfrac12$.

### Passo 6 · Il testo diventa una tabella

Seguiamo la frase **«il gatto nero dorme sul tappeto del salotto .»** e supponiamo,
per semplicità, che ogni parola sia un token. Ogni posizione del testo è una riga. Per
la posizione del punto finale la riga è:

| colonna nota | che cosa descrive |
|---|---|
| scala 1 | il token «salotto» |
| scala 2 | il concetto di «del salotto» |
| scala 4 | il concetto di «sul tappeto del salotto» |
| scala 8 | il concetto di «il gatto nero dorme sul tappeto del salotto» |
| **da completare** | **il prossimo token: «.»** |

Le colonne guardano sempre **indietro** dal token da predire: la colonna di scala
$2^k$ sono gli ultimi $2^k$ token. Così la posizione è relativa per costruzione, e il
token da predire non è mai fra le colonne note. Con venti scale la riga copre circa
mezzo milione di token, con venti colonne.

### Passo 7 · Dai token ai concetti dei tratti

**Un token diventa numeri.** Per ogni token il modello conta quali token compaiono
subito prima e subito dopo, e comprime questi conteggi in otto numeri con la SVD.
Token usati allo stesso modo hanno numeri vicini: «salotto» e «cucina» stanno vicini,
«corre» lontano. Il modello tiene anche l'identità esatta del token.

**Una coppia diventa una riga.** «del salotto» è una riga di sedici numeri: gli otto
di «del» seguiti dagli otto di «salotto». L'ordine conta: «salotto del» sarebbe
un'altra riga. Su tutte le coppie del testo il modello fa i passi 2–4 e trova i
concetti delle coppie: «del salotto», «della cucina», «nel bagno» finiscono nello
stesso concetto «preposizione + stanza».

**Due concetti formano un tratto più lungo.** Il concetto di una coppia è descritto
dal suo **centro**, la media delle righe che contiene. «sul tappeto del salotto» è una
riga fatta del centro di «sul tappeto» seguito dal centro di «del salotto»; il modello
ne trova i concetti, e così via raddoppiando: due tratti di 4 formano un tratto di 8.
Quando un centro supera 32 numeri si tengono le 32 direzioni in cui i centri
differiscono di più. Per ogni tratto si usa il concetto a cui appartiene di più.

### Passo 8 · Dal contesto alla probabilità del prossimo token

La previsione si costruisce **dal generale al particolare**, con la formula del
passo 5 applicata più volte: ogni lettura usa come $q^{\rm sopra}$ la stima della
lettura precedente.

1. Si parte dai concetti dell'ultimo token: che cosa segue di solito i token come
   «salotto».
2. Si legge l'identità esatta «salotto».
3. Si legge il concetto della coppia, poi le combinazioni: concetto + identità
   («del» esatto + «stanza»), identità + concetto, e infine la coppia esatta «del
   salotto».
4. Lo stesso per le scale 4, 8, 16…: concetto del tratto, metà come concetti o come
   identità, tratto esatto.

Esempio: la stima del punto dopo i passi precedenti è 0,10. La coppia esatta «del
salotto» è stata vista 20 volte, seguita 12 volte da «.», 5 da «,» e 3 da «e» (tre
valori diversi). Con $\kappa=1$: $q(.)=(12+3\cdot0{,}10)/(20+3)\approx0{,}53$. Una
combinazione mai vista lascia la stima com'è: 0,10. Una vista una volta sola, seguita
da «e», la sposta poco: $q(.)=(0+1\cdot0{,}10)/(1+1)=0{,}05$.

Così, quando il testo esatto non è mai stato visto, il modello non accorcia il
contesto: lo **astrae**, sostituendo i pezzi rari con il loro concetto.

**Sotto-concetti.** Dentro un concetto il modello prova le combinazioni più precise
(per esempio «stanza» seguita proprio da «salotto»). Le misura su documenti del
training tenuti da parte: per ogni osservazione calcola il guadagno

$$
g=\log p_{\rm figlio}(y)-\log p_{\rm padre}(y)
$$

e tiene il figlio solo se è stato letto almeno due volte e il guadagno medio supera il
suo errore standard. Il numero di sotto-concetti lo decidono i dati.

**Quali scale tenere.** Il modello aggiunge una scala alla volta (8, 16, 32, …) e
la tiene se migliora la previsione su testo di validazione. Si ferma dopo due scale di
fila senza guadagno, o quando i documenti sono più corti della scala.

### Passo 9 · La memoria del testo in corso

Mentre legge un documento il modello applica la stessa formula ai **conteggi del
documento stesso**: $N_i(y)$ conta quante volte, **prima** della posizione corrente, lo
stesso contesto (identità esatte o concetti) è stato seguito da $y$. La stima di
riferimento è la previsione ottenuta dal corpus. Così una frase detta molte pagine
prima può tornare, a qualunque distanza. Il token corrente entra nei conteggi solo
dopo che ha ricevuto la sua probabilità.

### Come si misura la qualità sul testo

Con la **NLL**: la media, sui token veri di un testo mai visto, di $-\log p$(token
vero). Più è bassa, meglio è; $e^{\rm NLL}$ dice fra quanti token, in media, il modello
è incerto.

## Informatica

Il codice del modello è uno solo: `acan/core/concepts.py` (`ConceptModel`). La
scheda tabulare lo usa direttamente; il generativo lo usa attraverso
`acan/lm/pair_text.py`, che trasforma il testo in righe.

### Una tabella: dall'apprendimento alla predizione

1. **Caricamento.** Si carica un file (CSV, JSON, XLSX, Parquet, ARFF o MAT) e si sceglie la colonna da
   predire. Le colonne numeriche restano numeri; le altre diventano categorie.
2. **Codifica.** Il modello ricava dalla tabella la scala di ogni colonna numerica
   e l'elenco dei valori di ogni categoria.
3. **Apprendimento.** Assegna le righe ai concetti, unisce quelli che costano meno,
   sceglie il taglio con la CP e lo giudica su righe tenute da parte (Matematica,
   passi 2–4). Una tabella grande si legge a pezzi: le somme di pezzi diversi si
   sommano, quindi il risultato non dipende da come la tabella è stata divisa.
4. **Salvataggio.** Il modello salvato contiene solo le somme dei concetti, i loro
   livelli e la codifica delle colonne: nessuna riga della tabella.
5. **Predizione.** Per una riga nuova calcola le quote $T_i$ dalle colonne note e
   legge la colonna mancante dai concetti (passo 5). La pagina mostra anche a quali
   concetti appartiene la riga.

### Un testo: dalla coda al checkpoint

1. **La coda.** Si caricano uno o più file di testo (anche PDF, DOCX, EPUB, ZIP), che
   il modello legge uno dopo l'altro.
2. **I token.** Un tokenizer BPE impara dai testi della coda a dividere le parole in
   pezzi (32.000 di default). Ogni documento va per intero in training, validazione o
   test: il modello non viene mai misurato su testo che ha letto.
3. **Il primo pezzo: la struttura.** Su un campione del primo file (almeno 262.144
   token, in finestre lunghe almeno il doppio della scala più lunga ammessa) il
   modello impara:
   - gli otto numeri di ogni token (SVD) e i concetti dei token;
   - scala dopo scala, i concetti dei tratti (passo 7) e i sotto-concetti (passo 8);
   - quali letture usare e con quale peso, scegliendo sulla validazione (passo 8).
   Il contesto massimo ammesso è un'impostazione (`concept_context`, da 8 a
   1.048.576 token, di default 512): la validazione sceglie fin dove salire.
   Ogni scala costa da qualche decina di secondi a un paio di minuti; un contesto
   massimo più alto allarga anche il campione, quindi la coda parte più tardi.
4. **Tutti i pezzi: i conteggi.** Il modello legge il testo a pezzi di un milione di
   token. Per ogni posizione aggiunge il token vero ai conteggi delle chiavi lette
   (identità e concetti delle scale scelte). I conteggi stanno in tabelle sparse,
   dentro un budget di memoria (un quarto della RAM del server, al massimo 2 GB, se
   non impostato): quando una tabella supera la sua quota tiene le chiavi con più
   osservazioni per byte; le altre leggono il concetto più generale.
5. **Misure e salvataggi.** Circa venti volte per file il modello riceglie i pesi
   delle letture su una parte della validazione e misura la NLL sull'altra. Salva un
   checkpoint ogni N passi e ogni cinque minuti, e risponde alle domande di prova.
6. **Il checkpoint.** Contiene i concetti (somme e centri), i dizionari delle
   identità, i conteggi, i pesi delle letture e gli ultimi token letti, così da
   riprendere esattamente da dove si era fermato. Il tokenizer è nella cartella del
   progetto.
7. **Il file successivo.** Riparte dal checkpoint del precedente: tiene concetti e
   conteggi e aggiunge quelli nuovi.

### Una domanda: dal prompt al prossimo token

1. **Il prompt diventa token.** Una domanda in chat viene avvolta nel formato della
   conversazione, poi il tokenizer la trasforma in identificatori.
2. **La lettura del prompt.** Un cursore legge i token uno dopo l'altro: per ogni
   scala ricorda la metà più vecchia del tratto e calcola il concetto del tratto
   nuovo componendo due concetti già noti. Intanto aggiorna la memoria del
   documento.
3. **La probabilità.** Per la posizione successiva applica le letture scelte
   (passo 8) e poi quelle della memoria (passo 9). Il risultato è una probabilità per
   ogni token del vocabolario, anche per quelli mai visti dopo questo contesto.
4. **La scelta.** A temperatura 0 prende il token più probabile; con temperatura
   positiva pesca a caso secondo le probabilità. Si ferma a un token di fine o al
   numero massimo di token.
5. **Avanti.** Il token scelto entra nel prefisso e nella memoria, e si torna al
   punto 3.

**Costo.** Per ogni token il cursore fa una composizione per scala (circa venti al
massimo) e qualche lettura di tabella: nessun confronto con tutti i token
precedenti, come fa invece l'attenzione. Tiene in memoria gli ultimi token fino alla
scala più lunga e i conteggi del documento.

### Le misure

WikiText-2, tokenizer BPE da 8.000 token, 32.704 token di test, NLL in nat/token,
memoria del documento spenta. Il confronto è con un contatore di suffissi esatti,
il modello generativo usato prima dal sito: oggi non fa più parte del prodotto e resta
solo come riferimento negli esperimenti.

| Scala provata | Coppie sul controllo | Coppie sul test | Contatore sul test |
|---:|---:|---:|---:|
| 6 | 4,91266 | 4,84475 | 4,91300 |
| 8 | 4,91098 | 4,84207 | 4,91300 |
| 16 | 4,90990 | 4,84097 | 4,91300 |
| 32 | 4,90987 | 4,84090 | 4,91300 |
| 64 | 4,90987 | 4,84090 | 4,91300 |
| 128 | 4,90913 | 4,84216 | 4,91300 |
| 256 | 4,90913 | 4,84216 | 4,91300 |
| 512 | 4,90913 | 4,84216 | 4,91300 |

- Il modello a concetti è sotto il contatore a ogni scala. Quasi tutto il vantaggio
  c'è già a sei token; da 6 a 32 si guadagnano 0,004 nat/token e oltre non cambia.
  Gli articoli di WikiText-2 sono brevi: scale più lunghe vanno provate su testi lunghi.
- Con la memoria del documento, a 32 token: 4,08543 contro 4,11082 del contatore.
  La memoria delle identità esatte fa quasi tutto (4,08816 da sola); quella dei
  concetti da sola dà 4,30614.
- Richiamo: un fatto ripetuto con le stesse parole torna fino a 300.000 token di
  distanza; con un sinonimo la risposta è fra i candidati 2 volte su 16 e al primo
  posto 0 volte su 16.
- Sui dieci dataset tabulari di prova le predizioni sono le stesse di prima.

[Risultati per blocco, scala e compito](../experiments/text_table/results/native-long-context.json) ·
[Protocolli e comandi per ripetere le prove](../experiments/text_table/README.md).
