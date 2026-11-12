"""A self-contained explanation of the current concept model."""
from dash import dcc, html

TEXT = r"""
## Come il modello cerca concetti nei dati

Un concetto è una descrizione riutilizzabile di più osservazioni: per esempio
«piove e il luogo è un palazzetto». Il modello cerca descrizioni rappresentative,
stima quanto ogni osservazione appartenga a ciascuna e favorisce concetti
caratteristici che non siano semplici copie fra loro.

**La modalità dipende da ciò che chiediamo di apprendere.** Senza target il
criterio è soltanto la CP: è clustering non supervisionato. Indicando una classe
da prevedere, aggiungiamo un errore di previsione e le etichette influenzano
l'apprendimento: diventa un modello supervisionato basato su concetti.
Nel linguaggio il testo fornisce da sé l'obiettivo, cioè il token successivo:
si parla di apprendimento auto-supervisionato. Queste modalità condividono
rilevanza, appartenenza e CP; non hanno lo stesso problema statistico.

### 1. Dalle osservazioni ai prototipi

Ogni riga viene rappresentata da un vettore $x$ di attributi fra zero e uno.
Le grandezze numeriche sono normalizzate; una categoria viene rappresentata
con indicatori distinti. «Rosso» e «blu» non diventano numeri arbitrariamente
ordinati. Valori mancanti e normalizzazione usano statistiche del solo training.

Un **prototipo** $a_i$ è il rappresentante apprendibile del concetto i. Non deve
coincidere con una riga: può descrivere una combinazione condivisa da molte righe.
A ogni concetto associamo una massa $\pi_i$, non negativa, con somma uno.
È il suo peso iniziale nella competizione; la frequenza effettiva $F_i$ verrà
invece misurata dalle osservazioni che gli vengono assegnate.

Un cluster non è una classe. Due concetti possono prevedere la stessa classe:
nel gioco, quattro combinazioni di tempo e luogo possono condurre a due esiti,
«SI» e «NO». Cercare gruppi in un file che contiene anche l'esito come attributo
osservato è diverso dal cercarli senza quella colonna o dal prevedere un esito nascosto.

### 2. Rilevanza: quali differenze caratterizzano un concetto?

Confrontiamo il prototipo con la media generale $\mu$. Una caratteristica comune
a quasi tutti distingue poco un gruppo; una differenza sistematica può renderlo
riconoscibile. Per confrontare attributi con scale diverse, misuriamo lo
scostamento rispetto alla massima distanza dalla media nell'intervallo osservato:

$$
\mu_j=\frac1n\sum_t x_{tj},\qquad
\Delta_j=\max(\mu_j-\min_j,\max_j-\mu_j).
$$

La rilevanza usata per confrontare una riga con il concetto è:

$$
R^{route}_{ij}=\operatorname{clip}_{[0,1]}
\left(\omega_j\frac{|a_{ij}-\mu_j|}{\Delta_j}\right).
$$

Se l'attributo è costante, la sua rilevanza è zero. Il fattore $\omega_j$ pesa
l'utilità dell'attributo: nel tabulare è stimato dalle dipendenze osservate,
comprese alcune interazioni fra coppie di attributi; con un target considera
anche le associazioni con le classi. Nel linguaggio è apprendibile.

Per esempio, media 0,5, intervallo [0,1], prototipo 0,8 e peso uno danno rilevanza
0,6. Il concetto si distingue su quell'attributo. **Questo non dimostra da solo
che sia un gruppo compatto:** anche un valore anomalo si allontana dalla media.
Un gruppo coerente ma vicino alla media può invece risultare poco caratteristico.
La rilevanza misura distinzione rispetto al riferimento, non certezza o causalità.

### 3. Appartenenza: vicinanza e competizione fra concetti

Ogni concetto pesa maggiormente gli attributi che lo caratterizzano. Elevare
la rilevanza al quadrato rafforza questa preferenza; un piccolo termine positivo
impedisce di ignorare completamente tutti gli attributi:

$$
W_{ij}=\frac{(R^{route}_{ij})^2+\varepsilon}
{\sum_{q\in O}[(R^{route}_{iq})^2+\varepsilon]},\qquad \varepsilon=10^{-4}.
$$

O contiene gli attributi disponibili per decidere l'appartenenza. La distanza
è una media delle differenze quadratiche, con questi pesi:

$$
D_i(x_t)=\sum_{j\in O}W_{ij}(x_{tj}-a_{ij})^2.
$$

Da distanze e masse otteniamo appartenenze che sommano a uno per ogni riga:

$$
T_{it}=\frac{\pi_i\exp[-\beta D_i(x_t)]}
{\sum_q\pi_q\exp[-\beta D_q(x_t)]}.
$$

A parità di massa, vince in probabilità il prototipo più vicino. Una beta grande
rende la scelta più netta; una beta piccola distribuisce la riga fra più concetti.
La beta viene appresa. La massa evita che la sola vicinanza determini tutto:
un concetto poco sostenuto parte con un peso minore.

Le appartenenze sono morbide: una riga può contribuire a più concetti.
Quando un gruppo di righe somiglianti riceve appartenenze simili, il suo riepilogo
si avvicina a una descrizione comune. Distanze, riepiloghi e obiettivo vengono
aggiornati insieme: è così che il modello può stabilizzare un cluster.
Non è però garantito che ogni raggruppamento semanticamente utile sia anche
quello favorito dalla distanza scelta.

Per prevedere una classe, **il suo valore non entra nella distanza**. Questo
impedisce di usare una risposta già nota in inferenza; non rende non supervisionato
un training che ha usato le etichette per apprendere prototipi e pesi.
T è una probabilità di appartenenza secondo il modello, non una probabilità
calibrata che la risposta finale sia corretta.

### 4. CP: qualità dei concetti, frequenza e duplicazione

Le appartenenze definiscono un conteggio morbido C e una media g per concetto:

$$
C_i=\sum_tT_{it},\qquad U_i=\sum_tT_{it}x_t,\qquad
 g_i=U_i/C_i,\qquad F_i=C_i/\sum_q C_q.
$$

Per valutare il concetto consideriamo sia il prototipo appreso sia ciò che gli
viene effettivamente assegnato. La loro media evita di valutare soltanto uno
dei due; il peso un mezzo è una scelta del modello, non un teorema universale:

$$
a_i^{eff}=(a_i+g_i)/2,\qquad
R_{ij}=\operatorname{clip}_{[0,1]}
\left(\omega_j\frac{|a^{eff}_{ij}-\mu_j|}{\Delta_j}\right).
$$

Riassumiamo le rilevanze con una media che dà più peso a quelle alte:

$$
q_i=\frac{\sum_jR_{ij}^2}{\sum_jR_{ij}}.
$$

Se tutte le rilevanze sono zero, poniamo anche q pari a zero. Rilevanze 0,9 e
0,1 danno q=0,82; due rilevanze 0,5 danno q=0,5. La formula preferisce una
descrizione con attributi fortemente caratterizzanti. Non misura direttamente
la dispersione interna delle righe.

La CP normalizzata di base combina qualità e frequenza:

$$
\overline{CP}=\sqrt{\sum_i F_iq_i}.
$$

Un concetto caratteristico e frequente contribuisce di più di uno raro.
La media pesata mantiene il punteggio fra zero e uno senza farlo crescere
automaticamente con il numero di attributi. La radice è crescente: massimizzarla
favorisce gli stessi valori della quantità interna, pur cambiando i gradienti.

Prima di valutare un concetto, escludiamo il sostegno che ogni riga dà a sé
stessa. Pescando due assegnazioni nel concetto, la frazione che proviene da
osservazioni diverse è:

$$
h_i=\frac{C_i^2-\sum_tT_{it}^2}{C_i^2},\qquad
\widetilde q_i=h_iq_i.
$$

Se il concetto contiene una sola riga, h vale zero: non c'è evidenza condivisa.
Con assegnazioni nette e C righe, h=(C−1)/C. Un gruppo sostenuto da molte righe
riceve una correzione piccola. Non si sceglie un numero minimo di righe né un K:
il fattore deriva dalle coppie di osservazioni diverse. Non è una dimostrazione
di indipendenza statistica delle righe.

Questa CP non distingue un concetto dalle sue copie. Per scoraggiare la
duplicazione misuriamo la sovrapposizione fra prototipi con la stessa distanza
usata per le righe, resa simmetrica:

$$
S_{ii}=0,\qquad
S_{ij}=\exp\left[-\frac\beta2(D_i(a_j)+D_j(a_i))\right]\quad(i\ne j).
$$

S è vicino a uno per due descrizioni quasi coincidenti, vicino a zero per
prototipi molto separati. Un secondo fattore confronta il peso iniziale dei
concetti con il loro uso effettivo:

$$
B(\pi,F)=\sum_i\sqrt{\pi_iF_i}.
$$

B vale uno quando masse e frequenze coincidono e diminuisce quando divergono.
È l'affinità di Bhattacharyya fra due distribuzioni. La CP usata attualmente è:

$$
z_i=F_i\sqrt{\widetilde q_i},\qquad
CP_* = B(\pi,F)\sqrt{\sum_iF_i\widetilde q_i-z^TSz}.
$$

Il termine sottratto evita di contare più volte evidenza condivisa da concetti
sovrapposti. Per r copie identiche **con frequenze uguali** e masse uguali alle
frequenze, la qualità sostenuta viene penalizzata così:

$$
CP_*^{copie}=\sqrt{\widetilde q/r},\qquad CP_*^{unico}=\sqrt{\widetilde q}.
$$

Per esempio, con qualità sostenuta 0,81, quattro copie danno 0,45; un solo rappresentante dà
0,9. È un motivo matematico concreto per preferire una descrizione unica alle
sue copie, senza imporre il numero di cluster.

Il punteggio resta limitato: 0≤B≤1 e 0≤S≤1; poiché le frequenze sommano a uno,
Cauchy–Schwarz garantisce che il radicando sia non negativo e non superi
la qualità media. Piccole protezioni numeriche evitano divisioni per zero.
**Il nome CP indica qui un indice di organizzazione, non la probabilità che
il raggruppamento sia quello vero.**

### 5. Che cosa viene ottimizzato nelle tre modalità?

**Clustering non supervisionato.** Tutte le colonne selezionate sono osservabili,
nessuna è indicata come risposta. Si ottimizza solo l'organizzazione dei concetti:

$$
\mathcal L_{cluster}=\lambda(1-CP_*).
$$

Non viene aggiunta una NLL delle classi né una probabilità generativa dei dati.
La qualità dipende quindi da quanto rilevanza e distanza descrivono una struttura
utile nei dati. Non è dimostrata l'equivalenza fra massima CP e cluster veri.

**Classificazione supervisionata.** Ogni concetto ha anche una distribuzione
sulle classi. La previsione è la miscela delle distribuzioni dei concetti:

$$
p(y\mid x)=\sum_iT_i(x)\,p_i(y),\qquad
\mathcal L_{class}= -\frac1n\sum_t\log p(y_t\mid x_t)+\lambda(1-CP_*).
$$

La NLL penalizza soprattutto una risposta sbagliata alla quale è stata assegnata
probabilità molto alta. La CP favorisce l'organizzazione mentre la NLL favorisce
la previsione: possono aiutarsi, ma anche entrare in conflitto. Con più target
si fa la media delle rispettive NLL. Nel tabulare lambda vale 0,1.
Le teste categoriche usano softmax e vengono inizializzate con il logaritmo delle
probabilità, così non si introduce una sicurezza artificiale nei valori iniziali.

Una classe può essere scritta come numero: 0/1 oppure codici 1…26 restano
categorie se il numero è un nome. Nell'app ogni target indicato viene trattato
come classe, anche quando contiene numeri decimali o più di 20 valori distinti.
Per scoprire gruppi senza supervisione basta svuotare il target. Stimare un
valore continuo è invece regressione, non supportata da questa testa categorica.
Il formato numerico da solo non determina il compito.

**Linguaggio auto-supervisionato.** Si apprende a continuare un testo; ogni parola
successiva fornisce il target. Ogni layer mantiene concetti del contesto già letto:

$$
\mathcal L_{LM}=-\frac1N\sum_t\log p(w_{t+1}\mid w_{\le t})
+\lambda\,\operatorname{media}_{\ell,t}(1-CP_{*,\ell,t}).
$$

Anche la somma delle assegnazioni al quadrato viene accumulata solo sul prefisso.
Media, intervalli e riepiloghi sono aggiornati solo sul prefisso: il futuro non
entra nella CP del token corrente. I conteggi e le somme precedenti vengono
accumulati; le vecchie assegnazioni non sono ricalcolate con token futuri.
La media della penalità evita che il suo peso cresca aggiungendo layer o token.
Si considerano solo posizioni che hanno un token successivo da prevedere.

Il nucleo dei concetti è lo stesso; la testa linguistica, la finestra locale e
le rappresentazioni dei token adattano quel nucleo al testo. Il laboratorio
serve a provare l'adattamento di un modello linguistico a un corpus. Diminuire
la NLL non basta a costruire un buon assistente: occorrono un test separato e
prove di continuazione del testo. Il peso CP è configurabile; zero permette
il confronto con sola NLL. L'eventuale distillazione aggiunge l'imitazione del
modello di partenza e deve essere dichiarata nel confronto.

### 6. Perché i prototipi possono proliferare? Quali parametri contano?

L'ottimizzazione modifica prototipi, beta e masse. Le masse restano non negative
e con somma uno; vengono provati anche trasferimenti fra concetti, accettati
quando l'obiettivo di training non aumenta. Una massa esattamente zero disattiva
il concetto: non si decide di cancellarlo con una soglia di similarità.

$$
K_{attivo}=\left|\{i:\pi_i>0\}\right|.
$$

**Questo non garantisce un numero minimo o semanticamente corretto.** Se i
prototipi sono molto separati, S è quasi zero. Con qualità simili e masse coerenti,
la CP può rimanere alta anche con molti gruppi. In classificazione, piccoli gruppi
possono ridurre l'errore di training memorizzando casi particolari. Un training
più lungo o una CP alta non sono quindi prove di generalizzazione.

La gerarchia aggiunge un altro fattore: ciascun nodo può avere fino a 256
prototipi, e il totale mostrato somma i prototipi di tutti i nodi. Non rappresenta
altrettante classi globali, ma resta un costo reale e può indicare frammentazione.
Ogni nodo viene ottimizzato localmente: non c'è una penalità globale sul numero
di nodi dell'intero albero. Anche l'espansione usa regole operative: profondità,
numero minimo di righe e purezza delle classi; senza target usa la variabilità
degli attributi. **Non è la sola CP a decidere tutta la struttura gerarchica.**

In previsione, un figlio sostituisce la distribuzione del suo concetto padre,
conservando il peso probabilistico di quel ramo:

$$
p_v(y\mid x)=\sum_iT_{v,i}(x)\,p_{v,i}(y\mid x).
$$

Se un figlio descrive la stessa distribuzione del padre, la previsione resta
invariata. Tutti i rami contribuiscono; il percorso mostrato è quello con
maggiore probabilità congiunta, non l'unica spiegazione della previsione.

| Impostazione | Che cosa controlla | Come interpretarla |
|---|---|---|
| Epoche | Quante volte si aggiorna il modello sui dati | Più epoche danno più opportunità di apprendere e di sovradattarsi; confrontare la validazione. |
| Learning rate | Ampiezza dei passi numerici | Se il training è instabile ridurlo; non è una misura della qualità dei cluster. |
| Batch | Righe per aggiornamento | Nel nucleo tabulare è limitato a 300; il campo attuale riguarda il training iniziale a un livello, non tutta la gerarchia. |
| Livelli iniziali | Profondità massima dell'albero automatico | Partire da un livello rende più facile vedere se l'approfondimento serve; più livelli possono moltiplicare i prototipi. |
| Campioni minimi del ramo | Quando ci sono abbastanza righe per espandere | Aumentarlo evita espansioni molto piccole, ma può perdere strutture rare. |
| Soglia di purezza | Quando approfondire un ramo con classi mescolate | Più vicina a uno richiede rami più puri e può inseguire il rumore nelle etichette. |
| Peso CP nel generativo | Compromesso fra previsione e organizzazione | Non esiste un valore ottimale universale; confrontare a parità di dati e budget. |

Sono limiti di capacità e scelte di ottimizzazione, non numeri di cluster
imposti. Alcuni controlli dell'interfaccia hanno un ambito ristretto: una loro
semplificazione deve rendere espliciti i valori effettivi e nascondere quelli
che non agiscono sul training scelto. I parametri di vecchie formule non servono
al nucleo attuale.

### 7. Cosa dimostrano le verifiche

Le verifiche numeriche controllano formule, gradienti, causalità del linguaggio,
conservazione del modello e coerenza della gerarchia. Non dimostrano da sole
che il modello generalizzi né che il numero dei concetti sia quello desiderato.
Un test usato ripetutamente per scegliere miglioramenti diventa uno strumento
di sviluppo: serve poi una verifica indipendente.

La CP corretta trova **4 gruppi sulle 101 righe originali di Gioco, senza target**,
includendo la riga con valore mancante. La NLL è zero. L'app conserva tutte le
righe per il clustering; il 20% separato resta per la previsione supervisionata.
Sulle cinque prove di classificazione controllate, Gioco mantiene quattro concetti
e 100%. Il numero atteso non viene fornito all'ottimizzatore.

Letter gerarchico misura **92,325%** sulle 4.000 righe di test completo e
**91,05%** nel campione web di 2.000 righe. I prototipi sono rispettivamente
3.346 e 1.387. Rispetto alla versione immediatamente precedente, il test completo
perde 0,375 punti, mentre il campione web guadagna 1,20 punti e usa meno prototipi.
Il **95,78%** storico non è stato recuperato. Sugli altri file i risultati sono
misti: la correzione non garantisce un miglioramento su ogni distribuzione.

Questi risultati misurano classificazione tabulare. Non provano un miglioramento
della qualità linguistica. Nel generativo bisogna confrontare lo stesso test,
lo stesso budget e le continuazioni prodotte, oltre a memoria e tempi.
Le formule spiegano il comportamento e i suoi limiti; i risultati sui dati
stabiliscono quanto quel comportamento sia utile nel problema concreto.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
