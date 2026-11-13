# Il grafo dei concetti a livelli: studio della funzione e fondamento rivisto

> **Documento storico.** Il modello attuale è uno solo ed è descritto in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md). Dove questa pagina differisce, vale MODEL.md: in particolare la covarianza nel concetto, i livelli automatici e la fusione per lunghezza di descrizione sono stati tolti.

Questo documento rivede il fondamento di
[MATHEMATICAL-FOUNDATION.md](MATHEMATICAL-FOUNDATION.md) dopo lo studio della
CP proposta in «Riportare la Cluster Probability al centro dell'apprendimento»
e dopo le prove sul multilivello. Risponde a tre domande: che funzione è quella
CP; perché a un livello aggrega troppo e a più livelli no; come lo stesso
oggetto si applica a un modello linguistico con molti layer.
Piano della revisione successiva: [NEXT-PR-PLAN.md](NEXT-PR-PLAN.md).

Legenda: **[D]** dimostrato qui, **[V]** verificato numericamente
(`scripts/verify_math_foundation.py`, controllo 10), **[S]** misurato,
**[C]** congettura da verificare.

## 1. Studio della funzione

Per una partizione delle $n$ righe in concetti di numerosità $C_i$, con
$\hat s_{ij}^2$ e $\hat\sigma_j^2$ varianze non distorte entro il concetto e
nella radice, la funzione proposta è (a meno della radice quadrata esterna)

$$
CP^\circ=\frac1m\sum_j\sum_i\frac{C_i(C_i-1)}{n(n-1)}\left(1-\frac{\hat s_{ij}^2}{\hat\sigma_j^2}\right).
$$

### 1.1 Forma a coppie: la CP è una funzione su un grafo

**Proposizione G1 [D][V].** Sia, per ogni coppia di righe distinte,

$$
s_{tu}=1-\frac1m\sum_j\frac{(x_{tj}-x_{uj})^2}{2\hat\sigma_j^2}.
$$

Allora $\sum_{t\ne u}s_{tu}=0$ e

$$
\boxed{\;CP^\circ=\frac1{n(n-1)}\sum_{t\ne u}[z_t=z_u]\;s_{tu}\;}
$$

dove $z_t$ è il concetto della riga $t$.

*Dimostrazione.* In un insieme di $C$ valori
$\sum_{t\ne u}(x_t-x_u)^2=2C(C-1)\hat s^2$. Applicata a tutte le righe dà
$\sum_{t\ne u}(x_t-x_u)^2/(2\hat\sigma^2)=n(n-1)$, quindi le $s_{tu}$ hanno
somma nulla. Applicata a un concetto dà
$\sum_{t\ne u\in i}s_{tu}=C_i(C_i-1)(1-\hat s_i^2/\hat\sigma^2)$. ∎

$s_{tu}$ è positiva quando due righe sono più simili della coppia media e
negativa quando lo sono meno. $CP^\circ$ è dunque la somma dei pesi interni ai
concetti in un **grafo completo di somiglianze centrate**: è la modularità di
quel grafo, con «la coppia media» come modello nullo. Per un attributo
categorico la stessa formula vale con $[x_t\ne x_u]$ al posto della distanza e
l'impurità di Gini al posto di $2\hat\sigma^2$; coincide con la riduzione di
Gini normalizzata.

Conseguenze immediate **[D][V]**:

- un solo concetto vale 0 (somma di tutti i pesi) e i singoletti valgono 0
  (nessuna coppia interna);
- il massimo è strettamente positivo appena esiste una coppia più simile della
  media: l'ottimo non è mai banale e non richiede soglie.

### 1.2 Regola di fusione e massimo interno

**Proposizione G2 [D].** Unire due concetti cambia la funzione di

$$
\Delta_{il}=\frac{2C_iC_l}{n(n-1)}\,\bar s_{il},\qquad
\bar s_{il}=1-\frac1m\sum_j\frac{(g_{ij}-g_{lj})^2+s_{ij}^2+s_{lj}^2}{2\hat\sigma_j^2},
$$

con $s^2$ varianze di popolazione. Conviene unirli **se e solo se la loro
distanza quadratica media incrociata è minore di quella della coppia media
della radice**. La regola dipende solo dalle statistiche additive.

**Proposizione G3 (massimo interno) [D][V].** Lungo l'agglomerazione a legame
medio le fusioni avvengono in ordine di $\bar s_{il}$ non crescente. Quindi
$CP^\circ$ cresce finché $\bar s_{il}>0$ e poi decresce: ha **un solo massimo**,
nel taglio dell'albero all'altezza della coppia media. È la versione corretta
del «Teorema 1» del documento esaminato, che dimostrava soltanto i valori agli
estremi. Il massimo globale su tutte le partizioni è un problema di
*correlation clustering* ed è difficile in generale; il percorso a legame medio
ne dà una soluzione con la garanzia di unimodalità.

L'albero di generalizzazione della tesi è quindi l'albero a legame medio, e la
CP ne sceglie il taglio.

### 1.3 Limite di risoluzione

**Proposizione G4 [D][V][S].** Per G2, due gruppi compatti vengono fusi appena
i loro centri distano meno di $\sqrt2\,\hat\sigma$ in media sugli attributi.
Un solo livello non può quindi distinguere strutture più fini della scala
della radice, **qualunque sia la loro compattezza**. È il limite di risoluzione
noto per la modularità.

- Verifica: 5 gruppi compatti, massimo di $CP^\circ$ a 3 concetti.
- Misura: su Letter (26 classi) la funzione sceglie 3 concetti e l'accuratezza
  è 8,65%; è lo stesso comportamento della versione storica a un livello.

Non è un difetto di implementazione: è una proprietà della funzione.

### 1.4 Perché il multilivello funziona

Dentro un concetto $v$ la stessa funzione, calcolata con la varianza del
concetto al posto di quella della radice, usa come riferimento la coppia media
**di $v$**. Ogni livello spiega la somiglianza che eccede la media del livello
superiore.

**Proposizione G5 (composizione dei livelli) [D].** Lungo un cammino
radice → foglia, con $r_\ell=1-s_\ell^2/s_{\ell-1}^2$ la rilevanza guadagnata al
livello $\ell$,

$$
1-\frac{s_{foglia}^2}{\sigma_{radice}^2}=1-\prod_\ell(1-r_\ell),\qquad
\log\frac{\sigma_{radice}^2}{s_{foglia}^2}=\sum_\ell\log\frac1{1-r_\ell}.
$$

Le rilevanze dei livelli si compongono moltiplicando le varianze residue; in
logaritmo si sommano. La somma è la riduzione di lunghezza di descrizione
(il logaritmo del rapporto di cardinalità della tesi) usata da cp-v9. I due
criteri non sono alternativi: **la CP è il guadagno di un livello, la lunghezza
di descrizione è la somma dei guadagni di tutti i livelli.**

Misure, split dell'app, soglie di approfondimento dell'app
(`scripts/probe_hierarchy.py`):

| Dataset | $CP^\circ$ un livello | $CP^\circ$ a più livelli | concetti | cp-v9 un livello | concetti |
|---|---:|---:|---:|---:|---:|
| Letter | 8,65% | 43,9% (3) → 85,4% (6) → 89,7% (12) | 1.317 | 93,3% | 412–476 |
| Sonar | 73,8% | 84,1% | 13–23 | 86,5% | 13–18 |
| Titanic | 76,4% | 78,6% | 49–63 | 77,7% | 32–37 |
| Adult minimale | 76,3% | 72,7% | 27–32 | 75,3% | 4 |

### 1.5 Ciò che la funzione non può fare

**Proposizione G6 [D].** In ogni nodo con almeno una coppia più simile della
media il massimo di $CP^\circ$ è positivo, anche su dati senza struttura.
La funzione **non decide quando fermarsi**: dividere vale sempre più di non
dividere. Nelle prove l'arresto è stato deciso da purezza, righe minime e
profondità, e su Adult la profondità peggiora il risultato (76,3% → 72,7%).
L'arresto deve venire da un altro operatore.

## 2. Fondamento rivisto: un grafo di concetti a livelli

Il modello resta quello degli operatori di
[`acan/concept_algebra.py`](../acan/concept_algebra.py). Cambia la struttura:
da una sola partizione a una **catena di partizioni annidate**.

**Oggetto.** Un grafo a livelli $\mathcal G=(P_0,P_1,\dots,P_D)$: $P_0$ sono le
osservazioni, ogni concetto di $P_\ell$ è unione di concetti di $P_{\ell-1}$,
$P_D$ è la radice. Ogni nodo porta le statistiche additive $(C,U,Q,Y)$; quelle
di un nodo sono la **somma** di quelle dei figli. Un albero ricorsivo e una
pila di livelli sono lo stesso oggetto: l'albero è il grafo visto per rami,
il multilivello nativo è il grafo visto per tagli.

**Operatore 1 — appartenenza, a ogni livello.** Invariato:
$T_i(o)\propto C_i\,\mathbb E_{z\sim i}[\kappa(o,z)]$.

**Operatore 2 — generalizzazione, che costruisce i livelli.** Si uniscono i
concetti con $\bar s_{il}$ massimo (G2). Il livello $\ell$ è il taglio in cui
$\bar s$ scende sotto la media del livello superiore (G3).

**Operatore 3 — giudizio, che decide la profondità.** Invariato: un livello
più fine è usato solo dove prevede meglio le osservazioni non viste.

**Novità — ogni livello è il prior del successivo.** In cp-v9 la varianza e le
frequenze di un concetto sono ritirate verso la radice con peso $\kappa$.
Nel grafo a livelli il riferimento è il **genitore**:

$$
v_{child}=\frac{C\,s_{child}^2+\kappa\,v_{parent}}{C+\kappa},\qquad
p_{child}=\frac{N_{child}+\kappa\,p_{parent}}{C+\kappa},\qquad
\bar y_{child}=\frac{Y_{child}+\kappa\,\bar y_{parent}}{C+\kappa}.
$$

**Proposizione G7 [D].** Un concetto senza sostegno proprio ($C\to0$) coincide
con il genitore; con molto sostegno coincide con le proprie statistiche. Un
livello fine privo di evidenza non cambia la previsione: l'approfondimento
smette di essere una decisione netta a soglia e diventa una correzione
graduale. cp-v9 è il caso $D=1$.

Questo risponde al limite di G6 senza soglie di purezza, e spiega perché su
Adult la profondità a soglia peggiorava: concetti di poche righe venivano
letti come certi. Che la correzione basti va misurato **[C]**.

**Perché i livelli aiutano anche la geometria [S].** Ogni livello riscala gli
attributi con la varianza del proprio concetto: l'insieme dei livelli
approssima a tratti la covarianza che un concetto ad attributi indipendenti
non vede. La covarianza piena resta comunque utile: su Letter miscele per
classe a covarianza piena danno 96,6% con 104 componenti.

## 3. Il caso generativo: lo stesso grafo, letto da molti layer

### 3.1 I layer sono attributi dei token

Nel tabulare una riga ha $m$ attributi e un concetto conserva statistiche per
ciascuno. In un modello linguistico un token del prefisso ha, **per ogni layer
e ogni testa**, una chiave e un valore. Sono attributi della stessa riga.

> Il contesto è una tabella: righe i token, colonne i blocchi
> (layer, testa). Il grafo dei concetti è **uno solo** per tutto il contesto;
> ogni layer lo legge sul proprio blocco di colonne.

È la stessa definizione del tabulare, non un'analogia: la somiglianza $s_{tu}$
di G1 è la media sugli attributi, cioè su layer e teste. La causalità è
rispettata perché un token entra nel grafo dopo essere uscito dalla finestra,
quando tutti i suoi attributi esistono già.

Misura (SmolLM2-135M, tre segmenti da 1.024 token, pesi invariati,
[`scripts/probe_concept_tree.py`](../scripts/probe_concept_tree.py)): con una
partizione unica costruita sulle chiavi di cinque layer insieme, la varianza
delle chiavi spiegata in ciascun layer è 0,44–0,53, contro 0,49–0,59 della
partizione costruita sul solo layer; l'errore dell'attenzione è praticamente
lo stesso (tabella sotto, colonne «condiviso»). **Una struttura sola serve
tutti i layer [S].**

### 3.2 Leggere il grafo: la risoluzione dipende dalla query

L'attenzione densa è una somma su tutti i token passati pesata da un nucleo.
Riassumere un gruppo lontano con il suo centro e aprire quelli che contano è
il principio dei codici ad albero per le somme di nuclei. Il criterio di
apertura è già nel fondamento: per la Proposizione U4 l'errore di un nodo è
limitato da massa × dispersione,

$$
\epsilon_v(q)\;\lesssim\;C_v\,e^{s\,q^\top g_v}\cdot s\,\lVert q\rVert\,\rho_v .
$$

**Lettura adattiva.** Si parte dalla radice e si apre ripetutamente il nodo con
$\epsilon_v(q)$ massimo, fino al numero di vettori disponibile. Aprendo tutto
si ottiene l'attenzione densa **[D]**: l'esattezza nel limite resta.

Errore dell'uscita dell'attenzione rispetto alla densa (somma delle norme
dell'errore su somma delle norme dell'uscita), 64 token recenti esatti più 64
vettori letti, salvo dove indicato:

| Layer | soli recenti | primi 4 + recenti | recenti ×2 | primi 4 + recenti ×2 | concetti piatti | **grafo, lettura adattiva** | piatti, condiviso | adattiva, condiviso |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2,64 | 1,48 | 1,53 | 1,07 | 0,78 | **0,21** | 0,55 | 0,19 |
| 9 | 1,65 | 1,19 | 1,09 | 0,92 | 0,62 | **0,20** | 0,61 | 0,21 |
| 15 | 1,11 | 0,51 | 0,76 | 0,25 | 0,31 | **0,21** | 0,34 | 0,23 |
| 21 | 1,16 | 0,45 | 0,82 | 0,24 | 0,23 | **0,14** | 0,25 | 0,16 |
| 27 | 1,05 | 0,17 | 0,90 | 0,08 | 0,14 | 0,10 | 0,14 | 0,10 |

Lettura:

- **La lettura adattiva è l'unica uniformemente buona** (0,10–0,21 in tutti i
  layer). Nei layer bassi riduce l'errore di tre-quattro volte rispetto ai
  concetti piatti e di cinque-sette rispetto a qualunque finestra.
- **I primi token contano.** I modelli preaddestrati concentrano attenzione
  all'inizio del contesto: tenerli esatti spiega gran parte del guadagno nei
  layer alti (layer 27: 0,17, e 0,08 con finestra doppia, meglio del grafo).
  Le prove precedenti a finestra non avevano questo controllo: il confronto
  «5,51 contro 5,26» di CP-V9-RESULTS.md va riletto con questa riserva.
- **La risoluzione necessaria diminuisce con la profondità.** I concetti
  piatti sbagliano 0,78 al layer 3 e 0,14 al layer 27.

Limiti di questa misura: è l'errore di un singolo strato con ingressi esatti,
non la NLL del modello con tutti gli strati sostituiti; l'albero completo è
conservato (memoria lineare) e solo la **lettura** è limitata; le chiavi sono
riassunte prima della rotazione posizionale.

### 3.3 Multilivello nativo: i livelli del grafo e i layer del modello

Le due osservazioni precedenti indicano come rendere il multilivello nativo
invece che un albero per testa:

1. **Un grafo, livelli annidati.** I livelli $P_1\subset\dots\subset P_D$ sono
   tagli dello stesso albero a legame medio: le statistiche di un livello sono
   la somma di quelle del livello più fine. La struttura si calcola una volta
   per il contesto, non per layer.
2. **Ogni layer legge il livello che gli basta.** Per G4 un livello risolve le
   strutture alla propria scala; per la misura, i layer alti tollerano livelli
   grossolani e quelli bassi no. La scelta non va fissata a priori: il
   criterio $\epsilon_v(q)$ la determina per ogni query, e il livello medio
   aperto da ciascun layer diventa una grandezza osservabile.
3. **Lo stato è limitato potando i livelli fini.** Il livello $P_0$ (token
   esatti) si conserva solo per la finestra recente e per i nodi che vengono
   aperti spesso; i rami mai aperti si riducono al genitore per somma. Così la
   memoria torna limitata, e la perdita è quella controllata da U4.
4. **Ogni livello è il prior del successivo** anche qui: un nodo potato è
   letto con le statistiche del genitore, senza discontinuità.

In questa forma tabella e linguaggio hanno la stessa struttura, gli stessi
operatori e la stessa regola di lettura: nel tabulare la «query» è la riga da
completare e i livelli fini si usano dove hanno sostegno; nel linguaggio è la
query di attenzione e si aprono dove l'errore stimato è maggiore.

### 3.4 Che cosa resta da dimostrare o misurare

- **[S]** La NLL del modello completo con lettura adattiva in tutti i layer,
  con il controllo «primi token + finestra».
- **[S]** Quanto si può potare a stato fissato prima che l'errore salga.
- **[C]** Che la potatura guidata dalle aperture sia vicina all'ottimo.
- **[C]** Il costo: la lettura adattiva richiede una coda di priorità per
  query; va resa vettoriale (apertura per livelli, non per nodo).
- **[C]** L'effetto dell'addestramento. Nulla qui è addestrato.
- La Proposizione U6 resta: a stato fisso la perdita non si annulla.

## 4. Che cosa ha mostrato la realizzazione

Misure in [CP-GRAPH-RESULTS.md](CP-GRAPH-RESULTS.md). Quattro punti correggono
o precisano le sezioni precedenti.

1. **Il prior di livello si realizza dall'alto, non dal basso.** Costruire i
   livelli su un solo adattamento eredita il prior di radice: un concetto di
   poche righe ha varianza vicina a quella della tabella e i gruppi fini si
   fondono. I livelli vengono quindi costruiti riesaminando ogni concetto sulle
   proprie righe e sulla propria scala; così G7 vale per costruzione.
2. **Accettare un livello richiede righe fresche.** Scegliere il miglior taglio
   fra molti e giudicarlo sulle stesse righe lo fa sembrare utile anche quando
   non lo è. Il figlio sceglie con metà delle righe e viene giudicato
   sull'altra metà, contro un solo concetto e contro il genitore.
3. **La covarianza è una statistica additiva come le altre.** La somma dei
   prodotti incrociati rende il concetto un'ellisse orientata; togliere un
   membro è una modifica di rango uno e il giudizio resta in forma chiusa.
   È ciò che porta Letter oltre il 95,78% storico.
4. **Nell'attenzione il costo di una fusione è la perdita del lettore.** La
   quantità che limita l'errore (U4), attenzione ricevuta × dispersione, usata
   come costo di fusione dà il risultato migliore a stato limitato, e conserva
   da sola i primi token. È l'operatore 2 con la perdita di chi legge al posto
   della lunghezza di descrizione; la lettura adattiva del §3.2 ne è il limite
   a memoria illimitata.

Non realizzati: il grafo unico condiviso da tutti i layer (§3.1) e la lettura
adattiva per query nello strato (§3.2). Restano misure fuori dallo strato.

## 5. Sintesi

| Domanda | Risposta |
|---|---|
| Che cos'è la CP proposta? | La modularità del grafo delle somiglianze centrate (G1). |
| Ha un massimo interno? | Sì, unico lungo il legame medio (G3). |
| Perché a un livello aggrega troppo? | Limite di risoluzione: fonde ciò che dista meno di $\sqrt2\sigma$ (G4). |
| Perché il multilivello funziona? | Ogni livello usa come riferimento la media del livello superiore; i guadagni si compongono (G5). |
| Sa fermarsi? | No (G6). Serve il giudizio, reso graduale dal prior di livello (G7). |
| Albero o multilivello nativo? | Lo stesso grafo: per rami o per tagli. Il nativo condivide la struttura e somma le statistiche. |
| E nei layer di un modello linguistico? | I layer sono attributi dei token: un grafo solo, letto da ogni layer alla risoluzione che la query richiede. |
