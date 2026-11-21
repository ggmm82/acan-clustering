# Il modello a concetti

## Matematica

### 1 · Introduzione

Presentiamo un modello che impara da una **tabella** e completa le colonne che
mancano in una riga nuova. Lo stesso modello risolve due problemi:

- **predizione tabellare**: date alcune colonne di una riga (peso, cilindrata) si
  stima una colonna target (prezzo);
- **generazione di testo**: dati i token già scritti si stima la colonna «prossimo
  token»; il token scelto diventa parte della riga successiva, e così via.

Il modello conserva **somme** delle righe osservate, raggruppate in **concetti**
organizzati in livelli dal generale al particolare. Sulle tabelle è tutto qui: i
concetti si trovano dalle somme, senza discesa del gradiente. Sul testo i concetti
esistono in due forme. I concetti contati costruiscono la catena di letture (sezioni
7–10). I concetti di una **rete a concetti** hanno le stesse formule, ma i loro centri,
dispersioni e masse si imparano come in un modello di linguaggio, cioè lungo il
gradiente della verosimiglianza del token che segue (sezione 11). La rete fornisce la
stima da cui parte la catena. Ogni scelta strutturale (quanti concetti, quanti livelli,
quante coordinate, quali letture, con quale forza, quando smettere di imparare) è presa
dai dati con un unico criterio, descritto nella sezione 2.

Lungo tutto il testo seguiamo due esempi: una tabella di quattro automobili e la frase
«il gatto nero dorme sul tappeto del salotto .».

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
raddoppia finché il guadagno è significativo), quante coordinate descrivono un token,
quanti sotto-concetti tenere, quali scale di contesto, quali letture e quali argomenti
usare. Quando
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

### 7 · Il testo come tabella

Ogni posizione di un testo è una riga; la colonna da completare è il token successivo,
e le colonne note descrivono il contesto **all'indietro**, a scale diadiche. Per la
posizione del punto finale di «il gatto nero dorme sul tappeto del salotto .»:

| colonna nota | che cosa descrive |
|---|---|
| scala 1 | il token «salotto» |
| scala 2 | il concetto di «del salotto» |
| scala 4 | il concetto di «sul tappeto del salotto» |
| scala 8 | il concetto di «il gatto nero dorme sul tappeto del salotto» |
| argomento | di che cosa parlano gli ultimi token (sezione 10) |
| **da completare** | **il prossimo token: «.»** |

La colonna di scala $2^k$ copre gli ultimi $2^k$ token. La posizione è quindi relativa
per costruzione e il token da predire non compare mai fra le colonne note. Con $k$
scale la riga copre $2^k$ token con $k$ colonne: venti scale coprono circa un milione
di token.

### 8 · Rappresentazione dei token e dei tratti

**Coordinate dei token.** Per ogni token si misura la distribuzione dei token che lo
precedono e lo seguono (fra i più frequenti), se ne prende la radice quadrata (distanza
di Hellinger) e si comprime il profilo con la decomposizione ai valori singolari. Token
usati allo stesso modo ricevono coordinate vicine: «salotto» e «cucina» stanno vicini,
«corre» lontano. Il numero di coordinate non è fissato: parte da 8 e raddoppia finché i
concetti dei token, trovati su quelle coordinate, prevedono meglio il token successivo
secondo il criterio della sezione 2. Accanto alle coordinate il modello conserva
l'identità esatta del token.

**Coppie ordinate.** «del salotto» è la riga formata dalle coordinate di «del»
seguite da quelle di «salotto»; l'ordine conta. Su tutte le coppie del campione si
applicano le sezioni 3–5 e si ottengono i concetti delle coppie: «del salotto», «della
cucina», «nel bagno» cadono nello stesso concetto «preposizione + stanza».

**Composizione ricorsiva.** Un concetto è descritto dal suo **centro**, la media delle
sue righe. Un tratto di 4 token è la riga formata dai centri delle sue due metà; due
tratti di 4 formano un tratto di 8, e così via. Oltre 32 numeri si tengono le 32
direzioni in cui i centri differiscono di più. Ogni tratto viene assegnato al concetto
a cui appartiene di più.

**Sotto-concetti.** Dentro un concetto si provano combinazioni più precise (per
esempio «stanza» seguita proprio da «salotto»); ciascuna resta solo se supera il
criterio della sezione 2 su documenti tenuti da parte ed è stata letta almeno due
volte.

### 9 · La catena di letture: corpus e documento

Per ogni posizione il modello dispone di due fonti di conteggi:

- il **corpus**: quante volte, in tutto l'addestramento, un contesto (un'identità
  esatta, un concetto o una loro combinazione) è stato seguito da ciascun token;
- il **documento in corso**: gli stessi conteggi, limitati alla parte del documento
  già letta. Una frase detta molte pagine prima può così tornare, a qualunque distanza.
  Il documento si legge anche **per contenuto**: con l'argomento come contesto, la memoria
  ritrova le posizioni del documento che parlavano della stessa cosa, ovunque siano, e
  ciò che le seguiva.

Entrambe si leggono con la formula della sezione 6, in **un'unica catena ordinata per
distanza**, dal contesto più corto al più lungo; a parità di distanza viene prima il
corpus, poi il documento:

$$
\underbrace{q_{\rm rete}}_{\text{rete a concetti (sez. 11)}}
\;\to\;\text{corpus}(1)\;\to\;\text{documento}(1)
\;\to\;\text{corpus}(2)\;\to\;\text{documento}(2)\;\to\;\cdots
$$

La stima di partenza è la previsione della rete a concetti; al primo token di un
documento, che la rete non ha ancora letto, e senza rete è la lettura dei concetti
dell'ultimo token. Così ogni contesto corrisponde a uno stato di conoscenza più
specifico del precedente,
e una prova raccolta su un contesto corto viene corretta dai contesti più lunghi che la
seguono. Le forze $\kappa$ di tutte le letture si scelgono **congiuntamente** sulla
validazione: ogni lettura è affine nella stima che riceve, quindi l'effetto di una
modifica su tutta la catena si calcola in forma chiusa. Una lettura resta nella catena
solo se, con le altre come sono, il suo guadagno sulla validazione supera l'errore
standard (sezione 2); le altre non si leggono e non si calcolano.

*Esempio.* All'inizio di una risposta, in un dialogo, il contesto è «`assistant` a
capo». Il documento contiene un solo «a capo» già seguito da una parola: quello dopo
«`user`», seguito dalla prima parola della domanda. La lettura del documento sul
contesto di un token sposta su quella parola circa il 10% della probabilità. La lettura
successiva della catena è il contesto di due token «`assistant` a capo», visto migliaia
di volte nel corpus: con $a/(M+a)\approx0{,}16$ conserva solo il 16% della stima
ricevuta e porta il resto sui seguiti osservati di quel contesto. Una prova su un
contesto corto pesa quindi solo per ciò che i contesti più lunghi non dicono già.

**Quali scale.** Il modello aggiunge una scala alla volta (8, 16, 32, …) e la conserva
se migliora la previsione; si ferma dopo due scale di fila senza guadagno o quando i
documenti sono più corti della scala.

### 10 · Gli argomenti

I concetti dei tratti descrivono l'**ordine** dei token. Per descrivere **di che
cosa** parla il testo, il modello usa una seconda rappresentazione dei token.

**Vettori di argomento.** Si costruisce la matrice dei conteggi token × documento, si
applica $\log(1+n)$, si pesa ogni token con $\log\frac{\text{documenti}}{1+\text{documenti che lo contengono}}$
(un token presente ovunque dice poco) e la si comprime con la decomposizione ai valori
singolari, normalizzando ogni vettore. Due token sono vicini quando compaiono negli
stessi documenti, in qualunque posizione: è l'analisi semantica latente. Il numero di
dimensioni parte da 16 e raddoppia secondo il criterio della sezione 2.

**Argomento di una scala.** Per la scala $W$ si fa la media dei vettori di argomento
degli ultimi $W$ token e la si assegna a un concetto, appreso con le sezioni 3–5 sulle
medie del campione. Il concetto si ricalcola ogni $\max(1,W/8)$ token, quindi
l'argomento cambia lentamente, come il testo.

**Lettura per la regola di Bayes.** L'argomento non sostituisce la previsione locale,
la **modula**. Siano $p_{\rm loc}$ la previsione della catena della sezione 9, $q_0$ la
frequenza generale dei token e $q_A$ la lettura dei seguiti osservati per gli argomenti
$A_{W_1},\dots,A_{W_k}$ delle scale scelte, letti in catena dalla scala più corta alla
più lunga con la formula della sezione 6 (partendo da $q_0$). Allora

$$
p(y)\;\propto\;p_{\rm loc}(y)\,\frac{q_A(y)}{q_0(y)} .
$$

Il rapporto vale 1 per i token che l'argomento non rende né più né meno probabili,
più di 1 per quelli tipici dell'argomento. È la regola di Bayes se si assume che,
noto il prossimo token, contesto locale e argomento siano indipendenti: il contesto
locale dice quale token viene adesso, l'argomento quali sono plausibili, e le due
prove si moltiplicano.

**Quali argomenti e con quale forza.** Le scale di argomento si aggiungono una alla
volta, la più utile prima, finché il criterio della sezione 2 lo consente. Tutte
condividono una sola forza $\kappa$, il valore che massimizza la verosimiglianza di
validazione (ricerca a sezione aurea su $\log\kappa$). Per ogni posizione la
verosimiglianza in funzione di $\kappa$ si calcola da poche somme precalcolate,
quindi la scelta richiede una sola lettura della validazione.

### 11 · La rete a concetti

I concetti contati descrivono il contesto con chiavi discrete: un contesto mai visto
ricade su concetti grossolani. La rete a concetti descrive invece ogni posizione con uno
**stato continuo**, fatto con gli stessi oggetti del modello e imparato per prevedere il
token che segue.

**Token come concetti.** Ogni token $y$ è un concetto con centro $e_y$ e massa $C_y$. Lo
stato di una posizione parte dal centro del suo token, e la previsione del token che
segue è l'appartenenza dello stato finale $h$ ai concetti dei token (sezione 4):

$$
p_{\rm rete}(y\mid h)=\frac{C_y\,\mathcal N(h;\,e_y,\sigma)}{\sum_{y'}C_{y'}\,\mathcal N(h;\,e_{y'},\sigma)} .
$$

I centri partono dalle coordinate dei token della sezione 8 e le masse dalle loro
frequenze.

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
delle parole vicine entra con gli stati delle ultime tre posizioni, una trasformazione
per distanza. Lo stato dello strato successivo combina lo stato, ciò che ha letto e
l'ordine vicino.

**Come si imparano i concetti.** Centri, dispersioni, masse, attenzioni e centri dei
token si muovono lungo il gradiente della log-verosimiglianza del token che segue, la
misura della sezione 13. È così che un modello di linguaggio impara le sue
rappresentazioni: ogni concetto di ogni strato si piega a servire la previsione finale,
quindi i concetti di uno strato si formano al servizio di quelli dello strato sopra. Il
testo si legge una volta, nell'ordine della coda, un pezzo alla volta. Dopo ogni pezzo
la rete si misura sulla validazione: se non migliora più di un errore standard (sezione
2), si tiene la rete del pezzo prima e la rete smette di imparare, mentre i conteggi
proseguono.

**Perché la catena resta.** La rete generalizza dove i dati sono pochi; i conteggi
ricordano con esattezza ciò che il corpus e il documento hanno detto. Nella catena la
rete fornisce la stima generale e i contesti esatti la correggono con la formula della
sezione 6, tanto più quanto più li hanno visti. Il peso fra le due fonti non è fissato:
lo stabiliscono, posizione per posizione, le osservazioni di ciascun contesto.

### 12 · Generazione

Il modello sceglie un token dalla distribuzione $p$ della catena: il più probabile a temperatura 0,
altrimenti un'estrazione da $p^{1/\tau}$ normalizzata. Il token scelto entra nel
contesto e negli argomenti, ma **non** nei conteggi del documento: il testo che il
modello scrive non è una prova di che cosa dice il documento, e lasciarlo entrare
farebbe rinforzare al modello le proprie ripetizioni.

### 13 · Misura della qualità

Sul testo si usa la log-verosimiglianza negativa media dei token veri di testo mai
visto, in nat per token:

$$
\mathrm{NLL}=-\frac1n\sum_{t=1}^{n}\log p(y_t\mid y_{<t}) .
$$

Più è bassa, meglio è; $e^{\rm NLL}$ (la perplessità) dice fra quanti token, in media,
il modello è incerto. Sulle tabelle si usano l'accuratezza e l'errore quadratico.
Le appartenenze, le somme dei concetti e la lettura per categorie della rete sono le
stesse formule delle sezioni 3, 4 e 6; cambia solo il modo in cui se ne trovano i
parametri.

### 14 · Limiti noti

- **La risposta dipende poco dalla domanda.** All'inizio di una risposta i contesti
  esatti vedono solo l'intestazione del turno, e gli argomenti predicono come continua
  il testo di quell'argomento, non come comincia una risposta. Anche la rete a concetti,
  che pure legge la domanda attraverso le somme dei concetti, all'inizio della risposta
  non migliora la catena: il suo guadagno viene dal resto del testo.
- **Velocità di apprendimento della rete.** La rete impara circa trenta volte più
  lentamente di quanto i conteggi leggano; per questo smette di imparare quando non
  migliora più.
- **Interpretabilità.** I concetti contati si leggono come somme dei loro membri; quelli
  della rete sono parametri ottimizzati, che si ispezionano attraverso le posizioni che
  vi appartengono.

## Informatica

### 1 · Panoramica

Le stesse formule servono la scheda tabulare e quella generativa. Nel caso tabulare le
righe sono quelle del file e i concetti si trovano dalle somme. Nel caso generativo ogni
posizione del testo diventa una riga (Matematica, sezione 7); i concetti contati formano
la catena di letture, e la rete a concetti, se la coda la impara, ne fornisce la stima di
partenza (Matematica, sezione 11). Questa scheda descrive gli algoritmi, il loro costo e
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
3. **Struttura.** Su un campione del primo file, in finestre lunghe almeno il doppio
   della scala più lunga ammessa, il modello impara in quest'ordine:
   1. le coordinate dei token e il loro numero, poi i concetti dei token;
   2. scala dopo scala, i concetti dei tratti, i sotto-concetti e le letture da usare;
   3. i vettori di argomento e il loro numero, poi i concetti di argomento di ogni
      scala.

   La dimensione del campione di ogni fase raddoppia finché serve (Matematica,
   sezione 2). La pagina mostra la fase in corso e la parte svolta, che non torna mai
   indietro anche quando una ricerca riparte con più righe o più coordinate.
4. **Conteggi in flusso.** Il testo si legge a pezzi di un milione di token. Per ogni
   posizione si calcolano le chiavi di contesto (identità, concetti, argomenti) e si
   aggiunge il token vero ai conteggi. I conteggi di più pezzi si fondono a livelli,
   come un contatore binario: ogni conteggio viene fuso $O(\log n)$ volte. Stanno in
   tabelle sparse entro un budget di memoria; quando una tabella supera la sua quota
   conserva le chiavi con più osservazioni per byte, e le altre leggono il concetto
   più generale. Gli argomenti non vengono potati.
5. **Rete a concetti (opzione della coda, attiva per default nella pagina).** Dopo i
   conteggi, ogni pezzo insegna alla rete in un solo passaggio. Si procede a lotti di
   circa 2.048 token di documenti interi, con AdamW e un breve riscaldamento del passo.
   Le somme dei concetti si leggono a blocchi di 64 posizioni: dentro il blocco con un
   prodotto di matrici, fra un blocco e l'altro con le somme correnti. Le appartenenze
   sviluppano il quadrato in prodotti di matrici. Dopo il pezzo la rete si misura sulla
   validazione e, se non migliora più di un errore standard, smette di imparare. Per
   imparare usa PyTorch; per leggere, numpy e i parametri salvati.
6. **Giudizio periodico.** Una ventina di volte per file si scelgono di nuovo le forze
   $\kappa$ della catena su metà della validazione e si misura la NLL sull'altra metà. Le
   scale di argomento e la loro forza si scelgono la prima volta e alla fine
   dell'addestramento. Le scale non scelte vengono tolte e non si calcolano più. Con la
   rete, le forze si scelgono con la rete come stima di partenza.
7. **Checkpoint.** Contiene concetti (somme e centri), dizionari delle identità,
   conteggi, forze delle letture, ordine della catena, vettori e concetti di argomento,
   i parametri della rete e gli ultimi token letti, così che l'addestramento riprenda esattamente dove si era
   fermato. Il file successivo della coda riparte dal checkpoint e aggiunge i suoi
   conteggi.

### 4 · Inferenza

1. **Prompt.** Una domanda in chat viene avvolta nel formato della conversazione e
   trasformata in token.
2. **Lettura del prompt.** Un cursore legge il prompt e costruisce lo stato: gli ultimi
   token fino alla scala più lunga, la metà più vecchia di ogni tratto, l'argomento di
   ogni scala, i conteggi del documento e, per la rete, le somme dei concetti di ogni
   strato con gli ultimi tre stati. Un prompt lungo si legge a blocchi, con conteggi
   fusi come nell'addestramento.
3. **Distribuzione.** Per la posizione successiva la rete dà la stima di partenza
   (Matematica, sezione 11), poi si applicano la catena di corpus e documento (sezione 9)
   e il rapporto degli argomenti (sezione 10).
   Il risultato è una probabilità per ogni token del vocabolario.
4. **Scelta e avanzamento.** Si sceglie il token (sezione 12), lo si aggiunge allo stato
   senza aggiungerlo ai conteggi del documento, e si ripete fino a un token di fine o
   al numero massimo di token.

La lettura token per token e la lettura di un testo intero producono la stessa
distribuzione in ogni posizione: è la stessa funzione applicata allo stesso stato.

### 5 · Costo

- **Per token, in inferenza:** una composizione di concetti per scala (al più una
  ventina), qualche ricerca in tabella per lettura e un aggiornamento dell'argomento
  ogni $W/8$ token per scala. Non c'è confronto con tutti i token precedenti: il costo
  per token non cresce con la lunghezza del contesto.
- **Memoria in inferenza:** gli ultimi token fino alla scala più lunga e i conteggi del
  documento, che crescono con il documento ma solo per i contesti effettivamente visti.
- **Rete, per token:** in ogni strato le appartenenze ai concetti, l'aggiornamento delle
  somme e la lettura (alcune decine di migliaia di operazioni), più la lettura finale
  sui token (vocabolario × larghezza dello stato). La memoria della rete è fissa: le somme
  dei concetti e gli ultimi stati, a qualunque lunghezza del documento.
- **Addestramento:** lineare nel numero di token per i conteggi; la struttura si impara
  una volta, su un campione la cui dimensione la stabiliscono i dati. La rete è lineare
  nei token che legge, ma circa trenta volte più lenta dei conteggi.

### 6 · Misure

Dialoghi in italiano e inglese (SYNTH): 12.000 dialoghi di addestramento, tokenizzatore
da 32.000 token, 200 dialoghi di test mai visti.

| Misura | Solo conteggi | Con la rete a concetti |
|---|---:|---:|
| NLL sul test | 4,3658 nat/token (perplessità ~79) | **4,1486** (perplessità ~63) |
| Struttura e primo pezzo | 805 s | 1.471 s |
| Lettura | 61.800 token/s | 1.961 token/s |
| Memoria di picco | 2,4 GB | 4,8 GB |

La rete ha 3 strati, ciascuno con 4 gruppi di 64 concetti, e uno stato di 96 numeri.
Dove il modello migliora la catena, su 2.000 dialoghi:

| Posizioni | Quota | Guadagno |
|---|---:|---:|
| ultimi 2 token mai visti nel corpus | 32% | 0,21 nat/token |
| visti da 1 a 9 volte | 35% | 0,29 |
| visti 100 volte o più | 13% | 0,12 |
| primi 5 token di una risposta | 1,5% | 0,00 |
| token rari | 29% | 0,28 |

Il guadagno viene dalla generalizzazione dove i contesti sono stati visti poco e dalla
vicinanza dei token simili in uscita. All'inizio di una risposta non c'è guadagno: le
risposte di prova restano poco legate alla domanda (Matematica, sezione 14).

[Protocolli sperimentali](../experiments/text_table/README.md).
