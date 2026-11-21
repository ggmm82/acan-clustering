# Il modello a concetti

## Matematica

### 1 · Introduzione

Presentiamo un modello che impara da una **tabella** e completa le colonne che
mancano in una riga nuova. Lo stesso modello risolve due problemi:

- **predizione tabellare**: date alcune colonne di una riga (peso, cilindrata) si
  stima una colonna target (prezzo);
- **generazione di testo**: dati i token già scritti si stima la colonna «prossimo
  token»; il token scelto diventa parte della riga successiva, e così via.

Sulle tabelle il modello conserva **somme** delle righe osservate, raggruppate in
**concetti** organizzati in livelli dal generale al particolare: i concetti si trovano
dalle somme, senza discesa del gradiente (sezioni 3–6). Sul testo il modello è una **rete a
concetti**: gli stessi oggetti, con le stesse formule, i cui centri, dispersioni e masse si
imparano come in un modello di linguaggio, lungo il gradiente della verosimiglianza del
token che segue (sezione 7). Ogni scelta (quanti concetti, quanti livelli, quante righe
servono, quando rileggere il testo, quando smettere di imparare) è presa dai dati con un
unico criterio, descritto nella sezione 2.

Lungo le sezioni sulle tabelle seguiamo un esempio: una tabella di quattro automobili.

### 2 · Il criterio di scelta

Ogni volta che il modello deve decidere fra un'alternativa più semplice e una più
ricca, le confronta su dati che non ha usato per costruirle. Per ogni osservazione $t$
di validazione calcola il guadagno di log-verosimiglianza

$$
g_t=\log p_{\rm ricca}(y_t)-\log p_{\rm semplice}(y_t)
$$

e sceglie l'alternativa più ricca solo se il guadagno medio supera il suo errore
standard:

$$
\bar g>\frac{s_g}{\sqrt{n-1}} ,
$$

con $s_g$ la deviazione standard dei guadagni e $n$ il numero di osservazioni. Il
confronto è appaiato: le due alternative si misurano sulle stesse osservazioni, quindi
la variabilità comune non conta. Non ci sono soglie fissate a mano: la grandezza del
guadagno richiesto dipende solo da quanto è rumorosa la misura.

Lo stesso criterio decide quante righe servono per imparare una struttura (il campione
raddoppia finché il guadagno è significativo), quanti concetti e quanti livelli tenere e,
sul testo, quante volte rileggere un pezzo e quando la rete smette di imparare. Quando
una quantità è continua, come la forza $\kappa$ della sezione 6, si sceglie invece il
valore che massimizza la verosimiglianza di validazione.

### 3 · Concetti: statistiche sufficienti

Consideriamo la tabella (numeri inventati, per l'esempio):

| riga | peso (t) | cilindrata (cc) | prezzo (k€) |
|---:|---:|---:|---:|
| 1 | 1,0 | 1000 | 12 |
| 2 | 1,1 | 1200 | 14 |
| 3 | 1,9 | 2800 | 40 |
| 4 | 2,0 | 3000 | 44 |

Un **concetto** è un gruppo di righe, rappresentato da statistiche sufficienti: per
ogni colonna numerica la massa, la somma dei valori e la somma dei quadrati; per ogni
colonna di categorie la massa di ciascun valore. Una riga può appartenere in parte a più
concetti. Indicando con $T_{ri}\in[0,1]$ la quota della riga $r$ nel concetto $i$,
massa e media sono

$$
C_i=\sum_r T_{ri},\qquad
\mu_{ij}=\frac{\sum_{r}T_{ri}\,x_{rj}}{\sum_{r}T_{ri}} ,
$$

e dalla somma dei quadrati si ricava la dispersione $\sigma_{ij}^2$. Il concetto
«utilitarie» (righe 1 e 2) ha massa 2, peso medio 1,05 t, cilindrata media 1100 cc,
prezzo medio 13 k€.

Le statistiche sono **additive**: unire due concetti significa sommare le loro somme, e
imparare su una tabella divisa in pezzi dà lo stesso risultato che impararla intera. Un
valore mancante non entra nelle somme della sua colonna.

### 4 · Appartenenza

Sia $L_i(x)$ la densità del concetto $i$ sulle colonne note della riga $x$: gaussiana
con media $\mu_{ij}$ e dispersione $\sigma_{ij}$ per i numeri, frequenza relativa per
le categorie, con le colonne trattate come indipendenti dentro il concetto. La quota di
$x$ nel concetto $i$ è la probabilità a posteriori

$$
T_i(x)=\frac{C_i\,L_i(x)}{\sum_h C_h\,L_h(x)} .
$$

Un valore vicino al centro di un concetto stretto è molto probabile; un concetto
disperso spiega valori più lontani, ciascuno con meno probabilità. La massa $C_i$ fa da
probabilità a priori. Un'auto di 1,2 t e 1300 cc ha
$T_{\rm utilitarie}\approx 0{,}98$ e $T_{\rm grandi}\approx 0{,}02$.

### 5 · Gerarchia dei concetti

**Unioni.** Si parte da molti piccoli gruppi e si uniscono a due a due, prima le
coppie che costano meno. Con $\rho$ il raggio di un gruppo (la radice della sua
dispersione media, relativa al livello superiore), il costo di unire $i$ e $j$ è
l'aumento del raggio pesato per la massa:

$$
\Delta_{ij}=(C_i+C_j)\,\rho_{i\cup j}-C_i\,\rho_i-C_j\,\rho_j .
$$

**Taglio.** Ogni istante della sequenza di unioni è una partizione candidata. Il
punteggio di una partizione è la **coesione a coppie**: due membri distinti dello
stesso gruppo si somigliano più di due righe qualunque? Con $r_{ij}$ la dispersione
dell'attributo $j$ nel gruppo $i$ divisa per quella dell'intera tabella,

$$
\mathrm{CP}^{\circ}=
\frac{\sum_i C_i\max(C_i-1,0)\,\frac1m\sum_j [1-\min(2,r_{ij})]}
     {\max(n(n-1),\epsilon)} .
$$

Ogni gruppo contribuisce con le sue coppie di membri, pesate da quanto il gruppo è
stretto. Per una categoria la dispersione è la probabilità che due membri presi a caso
abbiano valori diversi.

**Giudizio.** Le partizioni migliori vengono confrontate su righe non usate per
costruirle, lette togliendo dalle somme il contributo della riga stessa (stima
*leave-one-out*). Fra le partizioni entro un errore standard dalla migliore si sceglie
la più semplice. Dentro ogni gruppo la ricerca si ripete e produce un livello più
specifico, se i dati lo giustificano; anche un solo gruppo è un esito possibile. Sulla
tabella dell'esempio il risultato è di due concetti.

### 6 · Predizione

**Colonna numerica.** Si pesano le medie dei concetti con le quote della riga:

$$
\hat y = \sum_i T_i(x)\,\mu_{i,\rm prezzo}\approx 0{,}98\cdot 13+0{,}02\cdot 42\approx 13{,}6\ \text{k€}.
$$

**Colonna di categorie: la lettura gerarchica.** Siano $N_i(y)$ la massa del valore
$y$ nel concetto, $M_i$ la massa totale e $D_i$ il numero di valori distinti osservati.
Data una stima più generale $q^{\rm sopra}$, il concetto la aggiorna così:

$$
q_i(y)=\frac{N_i(y)+a_i\,q^{\rm sopra}(y)}{M_i+a_i},
\qquad a_i=\kappa\max(1,D_i),\qquad
p(y\mid x)=\sum_i T_i(x)\,q_i(y).
$$

È la media a posteriori di una distribuzione di Dirichlet centrata sulla stima più
generale. Un concetto che ha visto poco resta vicino a $q^{\rm sopra}$; uno che ha
visto molto la sostituisce. Il termine $D_i$ rende la fiducia più bassa dove i seguiti
osservati sono molto vari. La forza $\kappa$ si sceglie sulla validazione.

Esempio: un concetto ha visto «.», «.» ed «e» (tre osservazioni, due valori). Con
$\kappa=1$ e $q^{\rm sopra}(.)=1/4$ si ottiene $q(.)=(2+2\cdot\tfrac14)/(3+2)=\tfrac12$.

Più letture si applicano **in catena**: ognuna usa come $q^{\rm sopra}$ il risultato
della precedente, dalla più generale alla più particolare.

### 7 · Il testo: la rete a concetti

Ogni posizione di un testo è una riga, e la colonna da completare è il token che segue.
La rete a concetti descrive ogni posizione con uno **stato continuo**, fatto con gli
stessi oggetti delle tabelle e imparato per prevedere il token che segue.

**Token come concetti.** Ogni token $y$ è un concetto con centro $e_y$ e massa $C_y$. Lo
stato di una posizione parte dal centro del suo token, e la previsione del token che
segue è l'appartenenza dello stato finale $h$ ai concetti dei token (sezione 4):

$$
p_{\rm rete}(y\mid h)=\frac{C_y\,\mathcal N(h;\,e_y,\sigma)}{\sum_{y'}C_{y'}\,\mathcal N(h;\,e_{y'},\sigma)} .
$$

**Strati di attenzione fra concetti.** In ogni strato, ciascun gruppo di concetti ha
$K$ concetti con centro $g_k$, dispersione $s_k$ e massa $C_k$. Una posizione con stato
$h_i$ vi appartiene con $T_k(h_i)$, la formula della sezione 4. Lungo il documento ogni
concetto accumula le sue somme (sezione 3),

$$
M_k(i)=\sum_{j\le i}T_k(h_j),\qquad S_k(i)=\sum_{j\le i}T_k(h_j)\,v_j ,
$$

e la posizione $i$ legge i concetti che le servono:

$$
r_i=\sum_k T_k(h_i)\sum_{k'}A_{kk'}\,\frac{S_{k'}(i)}{M_{k'}(i)} ,
$$

con $A_{kk'}$ l'attenzione dal concetto $k$ al concetto $k'$. Un token visto 10 o 10.000
posizioni prima entra nelle stesse somme: è un'attenzione per contenuto, a qualunque
distanza, il cui costo per token non dipende dalla lunghezza del documento. L'ordine
delle parole vicine entra con gli stati delle ultime $W$ posizioni, una trasformazione
per distanza. Ogni strato ha un'apertura $a$: il suo stato è
$h+a\,(\mathrm{strato}(h)-h)$.

**Memoria del documento.** Ogni posizione letta lascia un indirizzo, una proiezione del
suo stato, insieme al token che l'ha seguita. La posizione corrente confronta il suo
indirizzo con quelli precedenti e legge i token che seguivano gli indirizzi più simili:

$$
p(y\mid h_i)=(1-\lambda_i)\,p_{\rm rete}(y\mid h_i)+\lambda_i\sum_{j<i}\alpha_{ij}\,[y_{j+1}=y],
$$

con $\alpha_{ij}$ la somiglianza normalizzata degli indirizzi e $\lambda_i$ una valvola
calcolata dallo stato. Indirizzi e valvola si imparano con il resto della rete.

**Come si imparano i concetti.** Centri, dispersioni, masse, attenzioni, aperture,
memoria e centri dei token si muovono lungo il gradiente della log-verosimiglianza del
token che segue, la misura della sezione 9. Ogni concetto di ogni strato si piega così a
servire la previsione finale. Un pezzo di testo non si legge una volta sola: dopo la
prima lettura lo si rilegge dopo uno, due, quattro passaggi, finché il testo tenuto da
parte migliora più di un errore standard (sezione 2), e il passaggio che non aiuta si
annulla. È rileggendo che la rete **ricorda** ciò che ha visto, oltre a generalizzarlo.
Quando due decisioni di fila sul testo nuovo non migliorano, la rete smette di imparare.

**Le dimensioni della rete.** La rete ha 96 numeri di stato, 3 strati, 4 gruppi da 64
concetti per strato e 3 distanze vicine; i centri dei token partono a caso. Ogni documento
comincia dopo il token di fine, così anche il suo primo token si legge come in
addestramento.

### 8 · Generazione

Il modello sceglie un token dalla distribuzione $p$ della rete: il più probabile a
temperatura 0, altrimenti un'estrazione da $p^{1/\tau}$ normalizzata. Il token scelto entra
nello stato della rete, ma **non** nella memoria del documento: il testo che il modello
scrive non è una prova di che cosa dice il documento, e lasciarlo entrare farebbe
rinforzare al modello le proprie ripetizioni.

### 9 · Misura della qualità

Sul testo si usa la log-verosimiglianza negativa media dei token veri di testo mai
visto, in nat per token:

$$
\mathrm{NLL}=-\frac1n\sum_{t=1}^{n}\log p(y_t\mid y_{<t}) .
$$

Più è bassa, meglio è; $e^{\rm NLL}$ (la perplessità) dice fra quanti token, in media,
il modello è incerto. Sulle tabelle si usano l'accuratezza e l'errore quadratico.

### 10 · Limiti noti

- **Velocità di apprendimento della rete.** Per ogni token la rete calcola l'appartenenza
  ai concetti di tutto il vocabolario, avanti e indietro lungo il gradiente, e rilegge il
  testo più volte: impara lentamente su un processore senza GPU.
- **Interpretabilità.** I concetti delle tabelle si leggono come somme dei loro membri;
  quelli della rete sono parametri ottimizzati, che si ispezionano attraverso le
  posizioni che vi appartengono.

## Informatica

### 1 · Panoramica

Le stesse formule servono la scheda tabulare e quella generativa. Nel caso tabulare le
righe sono quelle del file e i concetti si trovano dalle somme. Nel caso generativo ogni
posizione del testo diventa una riga e il modello è la rete a concetti (Matematica,
sezione 7). Questa scheda descrive gli algoritmi, il loro costo e
le misure.

### 2 · Apprendimento e predizione su una tabella

1. **Lettura.** Si carica un file (CSV, JSON, XLSX, Parquet, ARFF o MAT) e si sceglie la
   colonna da predire. Le colonne numeriche restano numeri; le altre diventano
   categorie.
2. **Codifica.** Si ricavano la scala di ogni colonna numerica e l'elenco dei valori
   di ogni categoria.
3. **Apprendimento.** Le righe vengono assegnate ai concetti, i concetti uniti per costo
   crescente, il taglio scelto con la coesione a coppie e giudicato con la stima
   *leave-one-out* (Matematica, sezioni 3–5). Una tabella grande si legge a pezzi: per
   l'additività delle statistiche il risultato non dipende dalla divisione.
4. **Modello salvato.** Contiene solo le somme dei concetti, i loro livelli e la
   codifica delle colonne: nessuna riga della tabella.
5. **Predizione.** Per una riga nuova si calcolano le quote $T_i$ dalle colonne note e si
   legge la colonna mancante (sezione 6). La pagina mostra anche a quali concetti la
   riga appartiene.

### 3 · Addestramento su un testo

1. **La coda.** Uno o più file di testo (anche PDF, DOCX, EPUB, ZIP) letti uno dopo
   l'altro.
2. **Tokenizzazione.** Un tokenizzatore BPE impara dai testi della coda a dividere le
   parole in pezzi (32.000 di default). Ogni documento va per intero in addestramento,
   validazione o test: il modello non viene mai misurato su testo che ha letto.
3. **Apprendimento a pezzi.** Il testo si legge a pezzi di un milione di token. Ogni pezzo
   insegna alla rete con AdamW, a lotti di documenti interi: tanti token per lotto quanti
   lotti in un passaggio, al più 2.048, con un avvio del passo che dura al più un
   passaggio. Finito il pezzo, lo si rilegge dopo uno, due, quattro passaggi finché la
   validazione migliora più di un errore standard; il passaggio che non aiuta si annulla
   (Matematica, sezione 7). Quando due decisioni di fila sul testo nuovo non migliorano,
   la rete smette di imparare.
4. **Calcolo.** Le somme dei concetti si leggono a blocchi di 64 posizioni: dentro il
   blocco con un prodotto di matrici, fra un blocco e l'altro con le somme correnti. Le
   appartenenze sviluppano il quadrato in prodotti di matrici. La validazione si legge a
   lotti, senza gradiente. Per imparare si usa PyTorch; per leggere, numpy e i parametri
   salvati.
5. **Checkpoint.** Contiene i parametri della rete e gli ultimi token letti, così che
   l'addestramento riprenda dove si era fermato. Il file successivo della coda riparte
   dalla rete del checkpoint.

### 4 · Inferenza

1. **Prompt.** Una domanda in chat viene avvolta nel formato della conversazione e
   trasformata in token.
2. **Lettura del prompt.** Un cursore legge il prompt e costruisce lo stato: per ogni
   strato le somme dei concetti e gli ultimi stati, più gli indirizzi della memoria del
   documento. Un prompt lungo si legge in blocco.
3. **Distribuzione.** Per la posizione successiva la rete dà la probabilità di ogni token
   del vocabolario, con la sua memoria del documento.
4. **Scelta e avanzamento.** Si sceglie il token (Matematica, sezione 8), lo si aggiunge
   allo stato senza aggiungerlo alla memoria del documento, e si ripete fino a un token di
   fine o al numero massimo di token.

La lettura token per token e la lettura di un testo intero producono la stessa
distribuzione in ogni posizione: è la stessa funzione applicata allo stesso stato.

### 5 · Costo

- **Per token, in inferenza:** in ogni strato le appartenenze ai concetti,
  l'aggiornamento delle somme e la lettura, più la lettura finale sui token
  (vocabolario × larghezza dello stato) e il confronto con gli indirizzi della memoria
  del documento.
- **Memoria in inferenza:** le somme dei concetti e gli ultimi stati di ogni strato, fissi
  a qualunque lunghezza del documento, più un indirizzo per token del documento.
- **Addestramento:** lineare nei token per ogni passaggio, con più passaggi per pezzo.

### 6 · Misure

Dialoghi in italiano e inglese (SYNTH): tokenizzatore da 32.000 token, 2.000 dialoghi di
addestramento, dialoghi di test mai visti. La rete ha 96 numeri di stato, 3 strati e 4
gruppi da 64 concetti (3,4 milioni di parametri).

| Passaggi sul testo | Testo già letto | Validazione | Test |
|---:|---:|---:|---:|
| 1 | 5,066 | 5,258 | 5,352 |
| 2 | 4,423 | 4,875 | 4,966 |
| 3 | 4,086 | 4,767 | **4,877** |

NLL in nat per token. A ogni passaggio la rete legge meglio sia il testo che ha visto sia
quello che non ha visto: ricorda e generalizza. La misura sui 12.000 dialoghi del SYNTH
completo è in corso.

[Protocolli sperimentali](../experiments/text_table/README.md).
