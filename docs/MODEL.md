# Il modello: il grafo dei concetti

Questo è il riferimento unico. Vale per l'app tabulare e per il laboratorio
generativo, con le stesse definizioni e le stesse funzioni
([`acan/concept_algebra.py`](../acan/concept_algebra.py)). I documenti
precedenti (MATHEMATICAL-FOUNDATION, CONCEPT-GRAPH, CP-V9-RESULTS,
CP-GRAPH-RESULTS) restano come storia di come ci si è arrivati; dove
differiscono da questa pagina, vale questa pagina.

Profilo: `concept-graph-v1`, in entrambi i casi.
Codice: nucleo tabulare [`acan/core/concepts.py`](../acan/core/concepts.py),
strato di attenzione [`acan/lm/concept_attention.py`](../acan/lm/concept_attention.py).
Misure: [MODEL-RESULTS.md](MODEL-RESULTS.md).

## 1. Oggetti

Un'**osservazione** è una coppia $(z,y)$: la parte con cui la si riconosce e la
parte che si legge.

| | Tabella | Strato di attenzione |
|---|---|---|
| osservazione | una riga | un token del prefisso |
| $z$, riconosciuta | attributi noti | chiave |
| $y$, letta | target, o attributi mancanti | valore |
| chi legge | la riga da completare | la query |
| somiglianza $\kappa(o,z)$ | identità: la riga è osservata esattamente | $e^{s\,q^\top k}$ |

Un **concetto** è un nodo che conserva somme sui propri membri:

$$
C_i=\sum_tT_{it},\quad U_i=\sum_tT_{it}z_t,\quad Q_i=\sum_tT_{it}z_t^2,\quad
Y_i=\sum_tT_{it}\,\phi(y_t),\quad A_i=\text{letture ricevute},
$$

da cui centro $g_i=U_i/C_i$, dispersione $s_i^2=Q_i/C_i-g_i^2$, lettura
$\bar y_i=Y_i/C_i$. Per un attributo categorico le somme sono conteggi.

**Costo.** Ogni somma ha la dimensione di un'osservazione: durante la ricerca
dei concetti memoria e tempo sono lineari nel numero di attributi e nel numero
di osservazioni, e quadratici soltanto nel numero di concetti.

**Attributi che variano insieme.** Solo il livello di lavoro, a ricerca
finita, può conservare in più la somma dei prodotti fra gli attributi
riconosciuti, $S_i=\sum_tT_{it}z_tz_t^\top$ (ancora una somma sui membri), e
riconoscere con la covarianza $\Sigma_i=(S_i-C_ig_ig_i^\top+\kappa_0I)/(C_i+\kappa_0)$
al posto delle varianze. È adottata solo se le osservazioni non viste sono
lette meglio per più di un errore standard. Costa una passata sui dati,
quadratica negli attributi, e una fattorizzazione per concetto: la ricerca
non cambia.

Un **arco** collega due concetti dello stesso livello fra cui qualche
osservazione esita: $E_{il}=\frac1n\sum_tT_{it}T_{lt}$ sulle due appartenenze
maggiori di ogni osservazione.

I concetti sono disposti in **livelli**. La radice è la tabella intera. Il
livello 1 contiene i concetti generali; ogni livello successivo contiene i
comportamenti particolari, cioè ciò che è distinto solo dopo aver tolto quello
che il livello sopra già spiega. Due livelli consecutivi sono uniti da una
**matrice densa**

$$
P^{(\ell)}_{pi}=\frac{\sum_tT^{(\ell-1)}_{pt}\,T^{(\ell)}_{it}}{\sum_tT^{(\ell)}_{it}}
=\Pr(\text{concetto }p\text{ del livello sopra}\mid\text{concetto }i),
$$

quindi un concetto può discendere da più concetti del livello sopra: il grafo
non è un albero. Livelli, matrici e archi sono salvati con il modello e
mostrati nell'app.

## 2. Tre operazioni

**Riconoscere.**

$$
T_i(o)=\frac{C_i\;\mathbb E_{z\sim i}[\kappa(o,z)]}{\sum_lC_l\;\mathbb E_{z\sim l}[\kappa(o,z)]},
\qquad \hat y(o)=\sum_iT_i(o)\,\bar y_i .
$$

**Generalizzare.** Due concetti si sommano. Fra tutte le coppie si sceglie
quella che costa meno a chi legge:

$$
\rho_i=\sqrt{\frac1m\sum_j\frac{s_{ij}^2}{\sigma_j^2}},\qquad
\Delta_{il}=(a_i+a_l)\,\rho_{i\cup l}-a_i\,\rho_i-a_l\,\rho_l ,
$$

con $a_i$ le letture ricevute dal concetto e $\sigma_j$ la scala della radice.

**Giudicare.** Un concetto è valutato solo su osservazioni che non contiene:
$J=\frac1n\sum_t\log p^{(-t)}(y_t\mid z_t)$, con $p^{(-t)}$ ottenuta sottraendo
il contributo di $t$ dalle somme. Senza target ogni attributo è letto a turno
dagli altri, come quando si completa una tabella.

### Livelli: ogni livello è il contesto del successivo

La radice standardizza la tabella: toglie la media e divide per la deviazione.
Un livello fa lo stesso per il livello sotto, con il centro e la dispersione
del concetto di ogni osservazione:

$$
z'_t=z_t-\sum_pT^{(\ell)}_{pt}\,g_p,\qquad
\sigma'^2_{tj}=\sum_pT^{(\ell)}_{pt}\,s^2_{pj}.
$$

Sul livello sotto operano le stesse tre operazioni, sulla tabella $z'$ e con
scala $\sigma'$ al posto della scala della radice. Le somme in più che un
concetto conserva sono le somme della scala dei propri membri: restano
additive e lineari negli attributi. Per gli attributi categorici il contesto
dà frequenze e impurità del concetto sopra (non c'è un centro da togliere).

Quanto un livello aggiunge al livello sopra si misura con il **criterio a
coppie** (la CP quadratica):

$$
CP^\circ=\frac1{n(n-1)}\sum_iC_i(C_i-1)\Bigl(1-\frac1m\sum_j\frac{\hat s^2_{ij}}{\bar\sigma'^2_{ij}}\Bigr),
$$

la media, su tutte le coppie di osservazioni, di «stesso concetto» per
somiglianza centrata sulla coppia media del livello sopra ($\hat s^2$ è la
dispersione stimata sulle coppie di membri distinti, $\bar\sigma'^2_{ij}$ la
scala media dei membri; la somiglianza è limitata fra $-1$ e $1$).

Come si scelgono i livelli:

1. **Livello 1.** Lungo la generalizzazione, il taglio con $CP^\circ$ massimo
   sulla scala della radice; fra tagli a meno del 5% dal massimo, quello con
   meno concetti. Se nessun taglio è positivo, c'è un solo concetto.
2. **Seconda ricerca.** La stessa generalizzazione sulla tabella vista
   attraverso il livello 1 ($z'$, $\sigma'$).
3. **Livello di lavoro.** Fra tutti i passi delle due ricerche che leggono le
   osservazioni non viste entro un errore standard dal migliore, quello con
   meno nodi nel grafo.
4. **Livelli intermedi.** Lungo la generalizzazione che ha prodotto il livello
   di lavoro, il massimo di $CP^\circ$ contro il livello corrente, poi contro
   quello appena scelto, e così via. Ci si ferma quando il massimo è sotto il
   5% di quello del livello 1, o quando il livello, come descrizione della
   tabella, non legge le osservazioni non viste meglio del livello sopra.
5. **Pari merito.** Un livello intermedio le cui coppie non sono più simili
   nel livello di lavoro (meno del 5%) non è un livello distinto. Senza target
   nient'altro giustifica il livello di lavoro: se è in pari con il livello
   sopra, il livello sopra prende il suo posto.

Le due tolleranze (5% per i pari merito, $10^{-6}$ di varianza standardizzata
per «attributo costante») sono le uniche costanti della costruzione.

## 3. Proposizioni

**M1 (tabella).** Con $\kappa$ identità e membri gaussiani di varianza
$v_{ij}=(C_is_{ij}^2+\kappa_0)/(C_i+\kappa_0)$ sulla scala della radice,
$\mathbb E_{z\sim i}[\kappa(o,z)]=\prod_{j\in O}\mathcal N(o_j;g_{ij},v_{ij})$:
riconoscere è la posterior di una miscela. Omettere un attributo dal prodotto
lo marginalizza esattamente, quindi lo stesso modello completa ogni
sottoinsieme di attributi.

**M2 (attenzione).** Con $\kappa(q,k)=e^{s\,q^\top k}$ e un concetto per token,
riconoscere è **esattamente** l'attenzione softmax densa; il termine $\log C_i$
è la massa dell'operazione, non un'ipotesi. *Verifica:* lo strato coincide con
l'attenzione della libreria entro $2\cdot10^{-5}$ quando la finestra o il
numero di concetti copre il prefisso.

**M3 (errore di lettura).** Se ogni membro dista al più $\rho$ dal centro del
proprio concetto,
$\lVert\hat y_{denso}(q)-\hat y(q)\rVert\le(e^{2s\lVert q\rVert\rho}-1)\max\lVert y\rVert$.
*Dimostrazione.* Sostituire ogni membro con il proprio centro cambia ciascun
logit di al più $s\lVert q\rVert\rho$; numeratore e normalizzatore cambiano
ciascuno al più di un fattore $e^{\pm s\lVert q\rVert\rho}$, quindi i pesi
differiscono in norma uno di al più $e^{2s\lVert q\rVert\rho}-1$. ∎
Al primo ordine l'errore attribuibile a un concetto è la massa che riceve per
il suo raggio: è il costo $\Delta$.

**M4 (il costo non è mai negativo).** $\Delta_{il}\ge0$, con uguaglianza solo
fra due copie dello stesso concetto. *Dimostrazione.* La dispersione
dell'unione è la media pesata delle due dispersioni più la distanza fra i
centri, quindi non è minore della media pesata; la radice è concava. ∎
Unire copie è sempre gratuito: è la proprietà aggregante, senza un termine
dedicato. *Verifica:* stessi valori, alla precisione di macchina, dalla
funzione del nucleo tabulare e da quella dello strato di attenzione
(`tests/test_lm_concepts.py`).

**M5 (giudizio e causalità).** Poiché le somme sono additive, $p^{(-t)}$ si
ottiene per sottrazione. In un prefisso ordinato le somme al tempo $t$
contengono solo $\tau<t$: il giudizio è la verosimiglianza del token
successivo. Escludere la riga nel tabulare ed essere causali nel linguaggio
sono la stessa condizione.

**M6 (rilevanza).** Per ogni assegnazione
$\sigma_j^2=\sum_iF_is_{ij}^2+\sum_iF_i(g_{ij}-\mu_j)^2$. La rilevanza
$r_{ij}=1-s_{ij}^2/\sigma_j^2$ ha media $\eta_j^2\in[0,1]$, la frazione di
variabilità spiegata; $CP=\sqrt{\frac1m\sum_j\eta_j^2}$ è riportata, non
ottimizzata. Sugli attributi categorici coincide con la Category Utility.

**M8 (un livello non aggiunge nulla a sé stesso).** Misurato contro il
contesto dato da sé stesso, un livello ha $CP^\circ=0$; così un solo concetto
e i concetti di una sola osservazione. *Dimostrazione.* Tolto il proprio
centro, la dispersione di un concetto è la sua dispersione, che è la scala che
esso stesso dà ai propri membri: ogni rapporto vale uno. Con un membro non ci
sono coppie. ∎ Dividere a caso un concetto dà valore atteso nullo, perché
$\hat s^2$ è stimata sulle coppie. *Verifica:* `tests/test_concept_graph.py`.
Il criterio ha un solo massimo interno lungo una generalizzazione e non si
ferma da solo ([CONCEPT-GRAPH.md](CONCEPT-GRAPH.md), G3 e G6): per questo
l'arresto è affidato al giudizio.

**M9 (un comportamento condiviso è un solo nodo).** Se le osservazioni sono
$z=g_p+\delta_b+\varepsilon$, con $K_1$ concetti generali e $K_2$ scostamenti
che si ripetono in più concetti generali, nella tabella $z'$ le osservazioni
con lo stesso scostamento hanno lo stesso centro qualunque sia $p$. Il livello
sotto ha quindi $K_2$ concetti, ciascuno con più genitori, e il grafo ha
$K_1+K_2$ nodi dove un albero ne ha $K_1K_2$. *Verifica:* due famiglie con gli
stessi tre gruppi interni danno i livelli 2 e 3 (con un livello intermedio di
2), e la matrice verso le famiglie vale $\tfrac12$ in ogni cella; un albero
darebbe sei frammenti.

**M10 (le matrici conservano la massa).** Ogni colonna di $P^{(\ell)}$ somma a
uno e $P^{(\ell)}m^{(\ell)}=m^{(\ell-1)}$, con $m$ le masse dei concetti. La
discendenza attraverso più livelli è il prodotto delle matrici.

**M11 (giudizio con covarianza).** Togliere un'osservazione di peso $w$ da un
concetto è un aggiornamento di rango uno di $\Sigma_i$: con
$q=\delta^\top\Sigma_i^{-1}\delta/(C_i+\kappa_0)$, $\delta=z-g_i$ e
$C'=C_i-w$, la distanza diventa
$(C_i/C')^2(C'+\kappa_0)\,q/(1-wC_iq/C')$ e il determinante si moltiplica per
$1-wC_iq/C'$. Il giudizio su osservazioni non viste resta quindi in forma
chiusa. *Verifica:* coincide con il ricalcolo senza la riga
(`tests/test_concepts.py`).

**M7 (limite).** Uno stato di $P$ numeri a $b$ bit non può restituire
esattamente più di $Pb$ bit del passato. Un numero fisso di concetti è una
compressione con perdita; la perdita è quella limitata da M3.

## 4. Dove tabella e attenzione differiscono, e perché

Le definizioni sono le stesse. Cambiano quattro quantità che dipendono dal
problema, non dal modello.

| | Tabella | Attenzione | Motivo |
|---|---|---|---|
| letture ricevute $a_i$ | la massa $C_i$ | attenzione ricevuta dalle query passate | in una tabella ogni riga legge i concetti a cui appartiene |
| quanti concetti | scelti dal giudizio, con la regola dell'errore standard | fissati dalla memoria disponibile | nel linguaggio la memoria è la risorsa da limitare |
| riassegnazione dei membri | sì | no | il passato di una sequenza non si rilegge |
| livelli | scelti dal criterio a coppie | due: i concetti del prefisso e, sotto, la finestra esatta in cui ogni token è un concetto | a pesi invariati, due livelli di concetti con comportamenti condivisi leggono peggio di uno solo ([misura](MODEL-RESULTS.md)) |
| varianza nel riconoscere | predittiva (M1) | ordine zero | i membri di un concetto di attenzione sono dati memorizzati, non un campione |

Nell'attenzione i concetti riassumono le chiavi prima della rotazione di
posizione e sono riportati alla posizione media dei membri: è
un'approssimazione, non coperta da M3.

## 5. Che cosa è stato tolto, e perché

| Tolto | Motivo |
|---|---|
| Covarianza dentro la ricerca dei concetti | costo cubico a ogni unione. La covarianza è ora solo nel livello di lavoro, a ricerca finita: su Letter dà 98,1% in pochi secondi in più |
| Livelli costruiti dall'alto, un sottomodello per concetto | frammentavano: lo stesso comportamento veniva riscoperto sotto ogni concetto, con meno dati ogni volta. I livelli ora sono dello stesso modello (sezione 2) e un comportamento condiviso è un solo nodo (M9) |
| Albero di tagli sopra i concetti (`acan/core/graph.py`) | era un albero: un concetto aveva un solo genitore. Sostituito dai livelli con matrici dense |
| Punteggi normali | toglievano i concetti spurii con rumore uniforme e cancellavano i vuoti fra gruppi separati |
| Fusione per lunghezza di descrizione | sostituita dall'unico costo del lettore, con risultati equivalenti sul tabulare e migliori nell'attenzione |
| Termine di varianza e smorzamento di fase nell'attenzione | misurati: il primo peggiora, il secondo non cambia nulla |

## 6. Ciò che il modello non fa

- I concetti sono cercati senza relazioni fra attributi; la covarianza entra
  solo dopo, nel livello di lavoro, e solo con attributi numerici tutti
  osservati (al più 128) e livello di lavoro letto direttamente sulla tabella.
- Il criterio a coppie preferisce concetti grandi: gruppi adiacenti lungo una
  direzione compaiono prima uniti e sono separati al livello successivo.
- Il contesto di un livello toglie il centro solo agli attributi numerici; per
  i categorici dà frequenze e impurità.
- I livelli sono scelti, non garantiti: con poche osservazioni il loro numero
  cambia con il seme.
- Nell'attenzione i concetti sono per testa e a un solo livello sopra la
  finestra esatta. Due livelli con comportamenti condivisi, misurati sulle
  chiavi di un modello preaddestrato, leggono peggio: M9 richiede una struttura
  che quelle chiavi non hanno. Archi fra teste e layer non sono realizzati.
- Con compressione, e senza addestramento, non eguaglia l'attenzione densa.
