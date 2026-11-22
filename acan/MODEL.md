# Il modello a concetti

## Matematica

### 1 · Motivazioni

Un modello che impara dai dati deve fare tre cose: **riassumere** ciò che ha visto,
**riconoscere** a che cosa somiglia un caso nuovo e **predire** ciò che manca. Questo
progetto le fa tutte e tre con un solo oggetto, il **concetto**: un gruppo di
osservazioni descritto dalle sue somme (quante sono, la somma dei valori, la somma dei
quadrati, quante volte compare ogni categoria). Le domande che guidano il progetto sono:

- si può costruire un modello che si legge? Un concetto è una media e una dispersione,
  e si può guardare: «le auto leggere costano in media 13 k€»;
- si può imparare senza scegliere a mano le dimensioni? Quanti concetti, quanti livelli,
  quante righe servono, quanto grande è la rete: tutto si decide sui dati, con un solo
  criterio (sezione 2);
- si può usare lo stesso oggetto per una tabella e per un testo? Una posizione di un testo
  è una riga i cui attributi sono i token precedenti e la cui colonna da completare è il
  token che segue.

Il progetto ha quindi due modelli costruiti con gli stessi concetti e le stesse formule:

| | Tabelle | Testo |
|---|---|---|
| Che cosa impara | concetti delle righe, in livelli dal generale al particolare | una rete di concetti che legge il documento |
| Come impara | per somme e assegnazioni alternate (EM), senza gradiente | lungo il gradiente della verosimiglianza del token che segue |
| Che cosa salva | le somme dei concetti, nessuna riga | i parametri della rete |
| Che cosa predice | qualunque colonna mancante | il token che segue, un token alla volta |

La differenza nasce dai dati. In una tabella ogni riga è un punto in uno spazio fisso di
colonne: i concetti si trovano raggruppando i punti, e le loro somme sono già il modello.
In un testo la «riga» di una posizione contiene tutti i token che la precedono, una
quantità che cresce senza limite e che non si può raggruppare direttamente: serve una
funzione che riassuma il passato in uno stato di dimensione fissa, e quella funzione si
impara meglio lungo il gradiente.

Seguiamo due esempi: una tabella di quattro automobili per le sezioni 3–6, e la frase
«il gatto dorme sul» per le sezioni 7–9.

### 2 · Il criterio di scelta

Ogni volta che il modello deve decidere fra un'alternativa più semplice e una più
ricca, le confronta su dati che non ha usato per costruirle. Per ogni osservazione $t$
tenuta da parte calcola il guadagno di log-verosimiglianza

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
guadagno richiesto dipende solo da quanto è rumorosa la misura. Fra alternative
equivalenti entro un errore standard vince la più semplice.

Lo stesso criterio decide, sulle tabelle, quante righe servono, quanti concetti e quanti
livelli tenere; sul testo, quando la rete cresce, quante volte rileggere e quando smettere
di imparare. Una quantità continua, come la forza $\kappa$ della sezione 6, si sceglie
invece come il valore che massimizza la verosimiglianza di validazione.

### 3 · Concetti: statistiche sufficienti

Consideriamo la tabella (numeri inventati, per l'esempio):

| riga | peso (t) | cilindrata (cc) | prezzo (k€) |
|---:|---:|---:|---:|
| 1 | 1,0 | 1000 | 12 |
| 2 | 1,1 | 1200 | 14 |
| 3 | 1,9 | 2800 | 40 |
| 4 | 2,0 | 3000 | 44 |

Un **concetto** è un gruppo di righe rappresentato da statistiche sufficienti: per ogni
colonna numerica la massa, la somma dei valori e la somma dei quadrati; per ogni colonna
di categorie la massa di ciascun valore. Una riga può appartenere in parte a più
concetti. Con $T_{ri}\in[0,1]$ la quota della riga $r$ nel concetto $i$, massa e media
sono

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
con media $\mu_{ij}$ e dispersione $\sigma_{ij}$ per i numeri, frequenza relativa per le
categorie, con le colonne indipendenti dentro il concetto. La quota di $x$ nel concetto
$i$ è la probabilità a posteriori

$$
T_i(x)=\frac{C_i\,L_i(x)}{\sum_h C_h\,L_h(x)} .
$$

Un valore vicino al centro di un concetto stretto è molto probabile; un concetto
disperso spiega valori più lontani, ciascuno con meno probabilità. La massa $C_i$ fa da
probabilità a priori. Un'auto di 1,2 t e 1300 cc ha $T_{\rm utilitarie}\approx 0{,}98$ e
$T_{\rm grandi}\approx 0{,}02$.

Questa formula è il cuore di entrambi i modelli: sulle tabelle dice a quali concetti
appartiene una riga, nella rete dice a quali concetti appartiene lo stato di una
posizione e, alla fine, quale token segue.

### 5 · Apprendimento sulle tabelle

L'apprendimento alterna due operazioni sulle stesse somme, e poi sceglie il livello di
dettaglio.

**Assegnare e riassumere (EM).** Dati i concetti, ogni riga si distribuisce fra loro con
le quote $T_{ri}$ della sezione 4 (passo E, *expectation*); date le quote, le somme di
ogni concetto si ricalcolano (passo M, *maximization*). Ogni ripetizione non diminuisce
la verosimiglianza delle righe, e dopo poche ripetizioni i concetti si stabilizzano. Nella
riga 2 dell'esempio $T\approx 1$ per le utilitarie: le sue somme vanno quasi tutte lì.

**Partenza dal particolare.** Si parte da molti concetti piccoli: blocchi di righe
consecutive, uniti a due a due dove si perde meno, fino a circa una manciata di righe per
concetto. Le colonne da predire pesano quanto tutte le altre insieme, così i concetti
separano soprattutto ciò che serve a predire.

**Unioni.** Si uniscono poi i concetti a due a due, prima le coppie che costano meno, e
dopo ogni giro di unioni si ripete l'EM. Con $\rho$ il raggio di un concetto (la radice
della sua dispersione media, relativa al livello superiore), il costo di unire $i$ e $j$
è l'aumento del raggio pesato per la massa:

$$
\Delta_{ij}=(C_i+C_j)\,\rho_{i\cup j}-C_i\,\rho_i-C_j\,\rho_j .
$$

Unendo, le somme si sommano: nessuna riga va riletta.

**Giudizio.** Ogni momento della sequenza di unioni è una partizione candidata, dalla più
particolare (molti concetti) alla più generale (uno solo). Ogni candidata si giudica
sulla log-probabilità delle colonne da predire di righe che i concetti non hanno visto:
per ogni riga si tolgono dalle somme i suoi stessi contributi (stima *leave-one-out*),
cosa che le statistiche additive rendono esatta e immediata. Il **livello di lavoro** è il
più generale entro un errore standard dal migliore (sezione 2). Sulla tabella
dell'esempio sono due concetti.

**Livelli.** I livelli fra il più generale e quello di lavoro si scelgono con la
**coesione a coppie**: due membri distinti dello stesso concetto si somigliano più di due
righe qualunque? Con $r_{ij}$ la dispersione dell'attributo $j$ nel concetto $i$ divisa per
quella dell'intera tabella,

$$
\mathrm{CP}^{\circ}=
\frac{\sum_i C_i\max(C_i-1,0)\,\frac1m\sum_j [1-\min(2,r_{ij})]}
     {\max(n(n-1),\epsilon)} .
$$

Ogni concetto contribuisce con le sue coppie di membri, pesate da quanto è stretto; per
una categoria la dispersione è la probabilità che due membri presi a caso abbiano valori
diversi. Dentro i concetti generali la ricerca si ripete sulla tabella vista con la scala
di ciascun concetto: ciò che è distinto solo dentro un concetto generale diventa visibile,
e ciò che si ripete in più concetti generali diventa un solo concetto.

### 6 · Predizione sulle tabelle

Per una riga nuova si calcolano le quote $T_i(x)$ sulle sole colonne note.

**Colonna numerica.** Si pesano le medie dei concetti con le quote della riga:

$$
\hat y = \sum_i T_i(x)\,\mu_{i,\rm prezzo}\approx 0{,}98\cdot 13+0{,}02\cdot 42\approx 13{,}6\ \text{k€}.
$$

**Colonna di categorie: la lettura gerarchica.** Siano $N_i(y)$ la massa del valore $y$
nel concetto, $M_i$ la massa totale e $D_i$ il numero di valori distinti osservati. Data
una stima più generale $q^{\rm sopra}$, il concetto la aggiorna così:

$$
q_i(y)=\frac{N_i(y)+a_i\,q^{\rm sopra}(y)}{M_i+a_i},
\qquad a_i=\kappa\max(1,D_i),\qquad
p(y\mid x)=\sum_i T_i(x)\,q_i(y).
$$

È la media a posteriori di una distribuzione di Dirichlet centrata sulla stima più
generale. Un concetto che ha visto poco resta vicino a $q^{\rm sopra}$; uno che ha visto
molto la sostituisce. Il termine $D_i$ abbassa la fiducia dove i valori osservati sono
molto vari. La forza $\kappa$ si sceglie sulla validazione. Più livelli si applicano **in
catena**, dal più generale al più particolare: ognuno usa come $q^{\rm sopra}$ il risultato
del precedente.

Esempio: un concetto ha visto «.», «.» ed «e» (tre osservazioni, due valori). Con
$\kappa=1$ e $q^{\rm sopra}(.)=1/4$ si ottiene $q(.)=(2+2\cdot\tfrac14)/(3+2)=\tfrac12$.

Poiché il modello descrive tutte le colonne insieme, la stessa tabella imparata una volta
predice **qualunque** colonna a partire dalle altre, e dice anche a quali concetti la riga
appartiene.

### 7 · Il testo: la rete a concetti

Nel testo ogni posizione è una riga e la colonna da completare è il token che segue.
Dopo «il gatto dorme sul» la colonna vale, per esempio, «divano». La rete descrive ogni
posizione con uno **stato** $h$, un vettore di $D$ numeri, e lo costruisce con gli stessi
oggetti delle tabelle.

**Token come concetti.** Ogni token $y$ del vocabolario (di $V$ token) è un concetto con
centro $e_y$ e massa $C_y$, con una dispersione comune $\sigma$. Lo stato di una posizione
parte dal centro del suo token, e la previsione del token che segue è l'appartenenza dello
stato finale ai concetti dei token (sezione 4):

$$
p_{\rm rete}(y\mid h)=\frac{C_y\,\mathcal N(h;\,e_y,\sigma)}{\sum_{y'}C_{y'}\,\mathcal N(h;\,e_{y'},\sigma)}
=\mathrm{softmax}_y\!\Big(\frac{h\cdot e_y-\tfrac12\lVert e_y\rVert^2}{\sigma^2}+\log C_y\Big).
$$

Per non confrontare lo stato con tutto il vocabolario, i token sono riuniti in gruppi per
frequenza, dal più frequente in giù, di circa $\sqrt V$ token ciascuno. Ogni gruppo $G$ è a
sua volta un concetto con centro e massa, e la previsione si legge in due passi:

$$
p_{\rm rete}(y\mid h)=p(G_y\mid h)\;p(y\mid h,G_y),
$$

entrambi con la formula di appartenenza: lo stato si confronta con circa $2\sqrt V$
concetti invece che con $V$.

**Strati di attenzione fra concetti.** Lo stato passa per $L$ strati. In ogni strato
ciascun gruppo di concetti (una «testa») ha $K$ concetti con centro $g_k$, dispersione
$s_k$ e massa $C_k$. Una posizione $i$ proietta il suo stato in un'interrogazione $q_i$ e
in un valore $v_i$, e appartiene ai concetti con $T_k(q_i)$, la formula della sezione 4.
Lungo il documento ogni concetto accumula le sue somme, come un concetto delle tabelle
accumula le righe:

$$
M_k(i)=\sum_{j\le i}T_k(q_j),\qquad S_k(i)=\sum_{j\le i}T_k(q_j)\,v_j ,
$$

e la posizione $i$ legge le medie dei concetti che le servono:

$$
r_i=\sum_k T_k(q_i)\sum_{k'}A_{kk'}\,\frac{S_{k'}(i)}{M_{k'}(i)} ,
$$

con $A_{kk'}$ (una distribuzione su $k'$) l'attenzione dal concetto $k$ al concetto $k'$.
In «il gatto dorme sul», la posizione di «sul» appartiene a un concetto «luogo dopo un
verbo di stato»; la sua lettura porta la media dei valori delle posizioni passate di quel
concetto, e con essa l'informazione che serve per predire un luogo. Un token visto 10 o
10.000 posizioni prima entra nelle stesse somme: è un'attenzione per contenuto, a qualunque
distanza, e il suo costo per token dipende da $K$, non dalla lunghezza del documento.

L'ordine delle parole vicine entra con gli stati delle ultime $W$ posizioni, ciascuno
trasformato da una matrice propria della sua distanza. Lettura e vicini si combinano con
lo stato, si normalizzano, e lo strato ha un'**apertura** $a$ appresa: il suo stato è
$h+a\,(\mathrm{strato}(h)-h)$, così uno strato chiuso ($a=0$) lascia lo stato com'è.

**Memoria del documento.** Gli strati riassumono il documento in medie: ricordano di che
cosa si parla, non le parole esatte. Per ripetere un nome proprio o una frase già letta
c'è una memoria: ogni posizione letta lascia un indirizzo (una proiezione del suo stato)
insieme al token che l'ha seguita. La posizione corrente confronta il suo indirizzo con
quelli precedenti e legge i token che seguivano gli indirizzi più simili. Fra gli
indirizzi ce n'è uno in più, «nessuno somiglia al mio», che sta per la lettura della rete:

$$
p(y\mid h_i)=\alpha_{i0}\,p_{\rm rete}(y\mid h_i)+\sum_{j<i}\alpha_{ij}\,[y_{j+1}=y],
\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{e^{s_{i0}}+\sum_{j'<i}e^{s_{ij'}}},
$$

con $s_{ij}$ la somiglianza degli indirizzi delle posizioni $i$ e $j$ e $s_{i0}$ la
somiglianza dell'indirizzo in più, calcolata dallo stato. È la stessa forma della lettura
gerarchica della sezione 6: gli indirizzi sono le osservazioni (pesate da quanto
somigliano), la rete è la stima più generale e $e^{s_{i0}}$ è la forza $\kappa$. Pochi
indirizzi poco simili, come una domanda breve all'inizio di una risposta, lasciano la
parola alla rete; molti indirizzi molto simili la sostituiscono.

### 8 · Apprendimento della rete

**L'obiettivo.** Tutti i parametri (centri, dispersioni e masse dei token e dei concetti,
proiezioni, attenzioni $A$, trasformazioni dei vicini, aperture, indirizzi della memoria)
si imparano insieme massimizzando la log-verosimiglianza del token che segue su ogni
posizione del testo di addestramento:

$$
\mathcal L=\frac1n\sum_{t}\log p(y_{t+1}\mid y_{\le t}) .
$$

Il gradiente di $\mathcal L$ rispetto a ogni parametro si calcola all'indietro lungo le
formule della sezione 7 (retropropagazione), e i parametri si muovono di un piccolo passo
in quella direzione, con l'ottimizzatore AdamW. Ogni concetto di ogni strato si piega così
a servire la previsione finale: non raggruppa le posizioni perché si somigliano, ma perché
raggrupparle aiuta a predire.

**Perché il gradiente e non l'EM.** Sulle tabelle l'EM funziona perché le righe sono
fisse: il passo M ha una soluzione esatta, le medie. Nella rete le «righe» degli strati
interni sono gli stati, che dipendono da tutti i parametri degli strati precedenti: non
c'è una soluzione esatta da calcolare, e il modo diretto di sapere come cambiare un
parametro è il gradiente dell'obiettivo finale. Il passo E resta, dentro la rete: è la
formula di appartenenza che ogni strato applica.

**A pezzi, con un calendario.** Il testo si legge a pezzi di un milione di token, in lotti
di documenti interi (ogni documento comincia dopo il token di fine, così anche il suo
primo token è predetto). Un lotto ha tanti token quanti lotti ci sono in un passaggio sul
pezzo, al più 2.048; il passo cresce linearmente all'inizio per al più un passaggio. Le
decisioni (smettere, crescere) cadono su un calendario che raddoppia con i token
imparati: dopo un lotto, due, quattro… Quando due decisioni di fila non migliorano il
testo tenuto da parte, la rete smette di imparare.

**Dimensioni dai dati.** La rete parte piccola (32 numeri di stato, uno strato, 4 gruppi
da 16 concetti, 3 distanze vicine; i centri dei token partono a caso). Con il primo
pezzo, ogni volta che il testo letto raddoppia e con l'ultimo pezzo, la rete com'è e tre
copie cresciute imparano il pezzo nuovo, una volta ciascuna:

- **il doppio della larghezza:** ogni coordinata scritta due volte, con dispersioni e
  masse dei concetti corrette perché appartenenze e letture restino identiche;
- **uno strato in più,** con l'apertura chiusa: lascia lo stato com'è;
- **ogni concetto diviso in due,** con centri quasi uguali e metà della massa.

Ogni copia, all'inizio, legge **esattamente** come la rete: la crescita non costa niente
finché i dati non la usano. La copia più larga impara a metà passo, perché le sue
coordinate si muovono a coppie. Prosegue, con il pezzo imparato, la più semplice che la
migliore non batte di più di un errore standard sul testo tenuto da parte (sezione 2). La
gara usa testo nuovo e non rilegge quello già letto: a parità di token il testo nuovo
insegna di più (Informatica, sezione 6).

**Ricordare rileggendo.** Un passo di gradiente sposta i parametri di poco: una frase vista
una volta lascia una traccia piccola. Finito il testo, la rete lo rilegge tutto, e dopo
ogni passaggio misura il testo tenuto da parte; continua finché migliora di più di un
errore standard e annulla il passaggio che non aiuta. È rileggendo che la rete ricorda ciò
che ha visto, oltre a generalizzarlo; quando il testo è abbondante, le riletture servono
poco e si fermano presto.

### 9 · Predizione e generazione

Per predire, la rete legge il prompt una posizione alla volta (o tutto insieme: il
risultato è identico) e tiene uno **stato di lettura**: per ogni strato le somme
$M_k, S_k$ dei concetti e gli ultimi $W$ stati, più gli indirizzi della memoria. Per la
posizione successiva dà la distribuzione $p(y\mid h)$ della sezione 7.

Si sceglie un token: il più probabile a temperatura 0, altrimenti un'estrazione da
$p^{1/\tau}$ normalizzata. Il token scelto entra nello stato della rete, ma **non** nella
memoria del documento: il testo che il modello scrive non è una prova di che cosa dice il
documento, e lasciarlo entrare farebbe rinforzare al modello le proprie ripetizioni. Si
ripete fino al token di fine turno o al numero massimo di token.

Dopo «il gatto dorme sul», lo stato di «sul» appartiene ai concetti dei luoghi; la lettura
dei token dà alta probabilità a «divano», «letto», «tetto»; se poco prima il documento
diceva «il divano nuovo», l'indirizzo di quella posizione somiglia a quello corrente e la
memoria aggiunge probabilità a «divano».

### 10 · Confronti concettuali

**Le tabelle e il machine learning classico.**

- *Miscele gaussiane e naive Bayes.* L'appartenenza della sezione 4 è quella di una miscela
  di gaussiane con colonne indipendenti dentro ogni componente, imparata con l'EM. La
  differenza è nel resto: i concetti sono organizzati in livelli, il numero di concetti
  non si fissa ma si sceglie con la stima leave-one-out, e la predizione legge la
  gerarchia in catena (sezione 6).
- *Clustering gerarchico.* Le unioni per costo crescente ricordano il clustering
  agglomerativo (il costo è vicino a quello di Ward), ma qui ogni unione mantiene un
  modello probabilistico completo, giudicato sulla predizione e non solo sulla distanza.
- *Alberi, foreste, gradient boosting, reti neurali.* Sono modelli discriminativi: imparano
  una colonna da predire scelta prima, ottimizzando una perdita, con iperparametri
  (profondità, numero di alberi, passo) scelti spesso per tentativi. Il modello a
  concetti è **generativo**: descrive tutte le colonne insieme, predice qualunque colonna
  dalle altre, gestisce i valori mancanti senza imputazione, si aggiorna sommando nuove
  righe e non ha iperparametri da provare. Il prezzo è l'ipotesi di colonne indipendenti
  dentro un concetto: una relazione fra colonne si coglie solo attraverso più concetti,
  mentre un albero o una rete la possono rappresentare direttamente.

**La rete a concetti e i modelli di linguaggio con attenzione (Transformer).**

- *Cosa hanno in comune.* Lo stesso compito e lo stesso apprendimento: predire il token che
  segue, con la stessa perdita (la log-verosimiglianza), lungo il gradiente, con AdamW, a
  lotti di documenti. I centri dei token sono le *embedding*; la lettura finale è una
  softmax su prodotti scalari con un termine di massa; le aperture degli strati e la
  normalizzazione hanno il ruolo delle connessioni residue.
- *L'attenzione.* In un Transformer ogni posizione confronta la sua interrogazione con le
  chiavi di **tutte** le posizioni precedenti: il costo per token cresce con il contesto,
  e la memoria in inferenza (le chiavi e i valori salvati) cresce con il documento. Nella
  rete a concetti ogni posizione legge le somme di $K$ concetti per testa: le posizioni
  passate sono già riassunte nelle somme dei concetti a cui appartenevano. Costo e memoria
  per token sono fissi a qualunque lunghezza, come nelle attenzioni lineari e nei modelli a
  stato; in cambio una posizione non può distinguere due posizioni passate dello stesso
  concetto. È per questo che c'è la memoria del documento, che recupera il token esatto
  quando serve.
- *L'ordine.* Un Transformer riceve la posizione di ogni token con codifiche posizionali.
  Qui l'ordine entra in due modi: le somme sono causali (contengono solo il passato) e gli
  ultimi $W$ stati entrano con una trasformazione per distanza.
- *La memoria del documento.* Corrisponde ai meccanismi di copia dei modelli di linguaggio
  (puntatori con «sentinella», cache del documento): una probabilità che viene dai token
  già letti, mescolata a quella della rete secondo quanto il contesto somiglia.
- *Le dimensioni.* In un Transformer larghezza, profondità e numero di teste si fissano
  prima di imparare, di solito da leggi di scala misurate su modelli più piccoli. Qui la
  rete cresce mentre impara, con trasformazioni che conservano esattamente ciò che sa, e
  solo quando il testo tenuto da parte lo giustifica.
- *La scala.* I modelli di linguaggio moderni imparano da migliaia di miliardi di token su
  acceleratori grafici, con miliardi di parametri. Questa rete impara su un processore,
  da milioni di token, con milioni di parametri: le sue risposte sono molto meno coerenti,
  e la differenza dipende soprattutto dalla quantità di testo e di calcolo, non dalla
  formula.

**I due modelli del progetto fra loro.** Sulle tabelle i concetti si trovano: le somme
sono il modello, e ogni numero ha un significato diretto. Sul testo i concetti si
imparano: sono parametri che l'ottimizzazione piega a servire la previsione, e si
ispezionano attraverso le posizioni che vi appartengono. Le formule (appartenenza,
somme, lettura gerarchica) sono le stesse; cambia chi decide dove stanno i concetti: le
righe stesse nelle tabelle, il gradiente nel testo.

### 11 · Misura della qualità

Sul testo si usa la log-verosimiglianza negativa media dei token veri di testo mai
visto, in nat per token:

$$
\mathrm{NLL}=-\frac1n\sum_{t=1}^{n}\log p(y_t\mid y_{<t}) .
$$

Più è bassa, meglio è; $e^{\rm NLL}$ (la perplessità) dice fra quanti token, in media, il
modello è incerto. Prima di leggere, con un vocabolario di 32.000 token,
$\mathrm{NLL}=\log 32000\approx 10{,}37$. Sulle tabelle si usano l'accuratezza e l'errore
quadratico.

### 12 · Limiti noti

- **Velocità di apprendimento della rete.** Per ogni token la rete calcola, avanti e
  indietro lungo il gradiente, le appartenenze di ogni strato e quelle dei gruppi e dei
  token: su un processore senza GPU impara molto più lentamente di quanto il modello delle
  tabelle impari per somme.
- **Colonne indipendenti dentro un concetto** (tabelle): le relazioni fra colonne si
  colgono solo attraverso più concetti.
- **Interpretabilità della rete.** I concetti delle tabelle si leggono come somme dei loro
  membri; quelli della rete sono parametri ottimizzati, che si ispezionano attraverso le
  posizioni che vi appartengono.

## Informatica

### 1 · Panoramica

Il sistema ha due schede di lavoro che usano le stesse formule (Matematica, sezioni 3–4):

- la scheda **tabulare** impara i concetti di una tabella con l'EM e predice qualunque
  colonna;
- la scheda **generativa** impara la rete a concetti da una coda di testi, lungo il
  gradiente, e la fa scrivere; un'interfaccia compatibile con quella di OpenAI permette di
  usarla da altri programmi.

L'apprendimento della rete gira in un processo separato (il *worker*), che fa una cosa
alla volta, salva a ogni passo e riprende dall'ultimo salvataggio.

### 2 · Apprendimento e predizione su una tabella

1. **Lettura.** Si carica un file (CSV, JSON, XLSX, Parquet, ARFF o MAT) e si sceglie la
   colonna da predire. Le colonne numeriche restano numeri; le altre diventano categorie.
2. **Codifica.** Si ricavano la scala di ogni colonna numerica e l'elenco dei valori di
   ogni categoria; i valori mancanti restano fuori dalle somme.
3. **Concetti particolari.** Una passata sulle righe forma blocchi e li unisce a due a due
   dove si perde meno, fino a circa quattro righe per concetto (al più 1.024 concetti).
4. **EM e unioni.** Si alternano assegnazioni e somme (Matematica, sezione 5); poi si
   uniscono le coppie meno costose e si ripete l'EM, fino a un solo concetto. Ogni momento
   del percorso è una partizione candidata.
5. **Giudizio.** Su un campione di righe si misura la log-probabilità delle colonne da
   predire con la stima leave-one-out, togliendo dalle somme i contributi della riga. Il
   livello di lavoro è il più generale entro un errore standard dal migliore; i livelli
   intermedi si scelgono con la coesione a coppie, e dentro i concetti generali la ricerca
   si ripete sulla tabella vista con la loro scala.
6. **Tabelle grandi.** Si leggono a pezzi: per l'additività delle somme il risultato non
   dipende dalla divisione.
7. **Modello salvato.** Contiene solo le somme dei concetti, i loro livelli e la codifica
   delle colonne: nessuna riga della tabella.
8. **Predizione.** Per una riga nuova si calcolano le quote $T_i$ sulle colonne note e si
   legge la colonna mancante: media pesata per i numeri, lettura gerarchica per le
   categorie. La pagina mostra anche a quali concetti la riga appartiene.

**Costo.** Ogni passo di EM è lineare nelle righe, nei concetti e nelle colonne; il
giudizio leave-one-out non rilegge la tabella, perché usa le somme.

### 3 · Addestramento su un testo

1. **La coda.** Uno o più file di testo (anche PDF, DOCX, EPUB, ZIP) letti uno dopo
   l'altro; ogni testo riparte dalla rete imparata sui precedenti.
2. **Tokenizzazione.** Un tokenizzatore BPE impara dai testi della coda a dividere le
   parole in pezzi (32.000 di default). Ogni documento va per intero in addestramento,
   validazione o test: il modello non viene mai misurato su testo che ha letto.
3. **Pezzi e lotti.** Il testo si legge a pezzi di un milione di token. Un pezzo si divide
   in documenti (al più 512 token per tratto, ognuno dopo il token di fine), e i documenti
   in lotti di circa $\sqrt{\text{token del pezzo}}$ token, al più 2.048. Per ogni lotto:
   lettura in avanti, perdita (Matematica, sezione 8), gradiente all'indietro, norma del
   gradiente limitata a 1, passo di AdamW (passo $3\cdot10^{-3}$, avvio lineare per al più
   200 passi e al più un passaggio), un lieve decadimento dei parametri.
4. **Crescita.** Con il primo pezzo, dopo 1, 2, 4… pezzi e con l'ultimo, quattro
   candidati (la rete com'è, più larga, con uno strato in più, con i concetti divisi)
   imparano il pezzo nuovo, uno dopo l'altro; prosegue il più semplice che il migliore non
   batte di più di un errore standard. Una gara costa quattro letture di un pezzo, e le
   gare sono sempre più rade. Con la crescita spenta (opzione della coda) la rete ha
   dimensioni fisse: 96 numeri di stato, 3 strati, 4 gruppi da 64 concetti.
5. **Riletture finali.** Con l'ultimo pezzo la rete rilegge tutto il testo, passaggio dopo
   passaggio, finché la validazione migliora di più di un errore standard; il passaggio che
   non aiuta si annulla.
6. **Calcolo.** Le somme dei concetti si leggono a blocchi di 64 posizioni: dentro il
   blocco con un prodotto di matrici, fra un blocco e l'altro con le somme correnti. Le
   appartenenze sviluppano il quadrato in prodotti di matrici. La lettura finale usa i
   gruppi di token: per ogni posizione la probabilità del gruppo e quella del token dentro
   il gruppo. Per imparare si usa PyTorch; per leggere bastano numpy e i parametri salvati.
7. **Avanzamento.** La pagina mostra il passo, la velocità, la dimensione della rete
   (strati, numeri di stato, concetti, parametri), la gara in corso e due grafici: l'errore
   su testo mai visto passo dopo passo, e il passo in corso aperto, con la validazione
   mentre la rete legge il pezzo e una linea per ogni candidato della gara.
8. **Checkpoint.** A ogni passo si salvano i parametri della rete e gli ultimi token letti;
   l'addestramento riprende dall'ultimo passo completato.

### 4 · Inferenza

1. **Prompt.** Una domanda in chat viene avvolta nel formato della conversazione e
   trasformata in token.
2. **Lettura del prompt.** Un cursore legge il prompt e costruisce lo stato di lettura:
   per ogni strato le somme dei concetti e gli ultimi stati, più gli indirizzi della
   memoria del documento. Un prompt lungo si legge in blocco.
3. **Distribuzione.** Per la posizione successiva la rete dà la probabilità di ogni token
   del vocabolario, mescolata con la memoria del documento.
4. **Scelta e avanzamento.** Si sceglie il token (Matematica, sezione 9), lo si aggiunge
   allo stato senza aggiungerlo alla memoria, e si ripete fino al token di fine turno o al
   numero massimo di token.

La lettura token per token e la lettura di un testo intero producono la stessa
distribuzione in ogni posizione: è la stessa funzione applicata allo stesso stato.

### 5 · Costo

- **Per token, in inferenza:** in ogni strato le appartenenze a $H\times K$ concetti,
  l'aggiornamento delle somme e la lettura; la lettura finale su circa $2\sqrt V$ concetti
  di token; il confronto con gli indirizzi della memoria.
- **Memoria in inferenza:** le somme dei concetti e gli ultimi stati di ogni strato, fissi
  a qualunque lunghezza del documento, più un indirizzo per token del documento.
- **Addestramento:** lineare nei token per ogni passaggio; una gara costa quattro passaggi
  su un pezzo; le riletture finali, un passaggio ciascuna sul testo intero.

### 6 · Misure

Dialoghi in italiano e inglese (SYNTH), tokenizzatore da 32.000 token, dialoghi di
validazione e di test mai visti; NLL in nat per token (più basso è meglio). Rete fissa
con 96 numeri di stato, 3 strati e 4 gruppi da 64 concetti (3,4 milioni di parametri).

**Rileggere.** 2.000 dialoghi di addestramento, uscita su tutto il vocabolario:

| Passaggi sul testo | Testo già letto | Validazione | Test |
|---:|---:|---:|---:|
| 1 | 5,066 | 5,258 | 5,352 |
| 2 | 4,423 | 4,875 | 4,966 |
| 3 | 4,086 | 4,767 | **4,877** |

A ogni passaggio la rete legge meglio sia il testo che ha visto sia quello che non ha
visto: ricorda e generalizza.

**Testo nuovo o riletto.** Agli stessi token letti, testo nuovo letto una volta batte la
rilettura dello stesso testo:

| Token letti | Rilettura di 2.000 dialoghi: validazione | Dialoghi nuovi letti una volta: validazione | Test |
|---|---:|---:|---:|
| come 2 passaggi | 4,858 | 4,832 (4.000 dialoghi) | 4,917 |
| come 3 passaggi | 4,750 | 4,639 (6.000 dialoghi) | **4,716** |

**Memoria del documento.** Senza memoria, dopo due passaggi la validazione è 5,651 contro
4,858 con la memoria. Con l'indirizzo «nessuno somiglia al mio», addestrando fino a
convergenza, il test passa da 4,832 a **4,805**; e sulle risposte a 30 domande mai viste
(temperatura 0,7, al più 120 token):

| | Valvola dallo stato | Nessuno somiglia al mio | Risposte vere |
|---|---:|---:|---:|
| sotto i 15 token | 10% | **3%** | 0% |
| lunghezza media (token) | 80 | **101** | — |
| token presi dalla domanda | 47% | **32%** | 21% |
| trigrammi ripetuti | 5% | 6% | 2% |

**Uscita in due livelli.** Con i gruppi di token la rete impara circa il doppio più
veloce (2.007 contro 1.015 token al secondo su due processori), con test 5,359 contro
5,337 dopo un passaggio e 5,009 contro 4,948 dopo due.

[Protocolli sperimentali](../experiments/text_table/README.md).
