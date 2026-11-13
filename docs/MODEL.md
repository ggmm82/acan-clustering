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

**Costo.** Ogni somma ha la dimensione di un'osservazione. Il modello non
conserva prodotti fra attributi diversi: memoria e tempo sono lineari nel
numero di attributi e nel numero di osservazioni, e quadratici soltanto nel
numero di concetti durante la generalizzazione.

Un **arco** collega due concetti fra cui qualche osservazione esita:
$E_{il}=\frac1n\sum_tT_{it}T_{lt}$ sulle due appartenenze maggiori di ogni
osservazione. Gli archi sono la struttura ispezionabile di ciò che è stato
appreso; sono salvati con il modello e mostrati nel grafo dell'app.

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
il contributo di $t$ dalle somme.

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
| varianza nel riconoscere | predittiva (M1) | ordine zero | i membri di un concetto di attenzione sono dati memorizzati, non un campione |

Nell'attenzione i concetti riassumono le chiavi prima della rotazione di
posizione e sono riportati alla posizione media dei membri: è
un'approssimazione, non coperta da M3.

## 5. Che cosa è stato tolto, e perché

| Tolto | Motivo |
|---|---|
| Covarianza fra attributi dentro il concetto | costo cubico nel numero di attributi. Su Letter dava 97,8% contro 93,6%: la qualità persa è nota e accettata |
| Livelli automatici costruiti dall'alto | una seconda struttura, con un secondo adattamento per ogni concetto; sui dati reali provati non veniva quasi mai adottata |
| Punteggi normali | toglievano i concetti spurii con rumore uniforme e cancellavano i vuoti fra gruppi separati |
| Fusione per lunghezza di descrizione | sostituita dall'unico costo del lettore, con risultati equivalenti sul tabulare e migliori nell'attenzione |
| Termine di varianza e smorzamento di fase nell'attenzione | misurati: il primo peggiora, il secondo non cambia nulla |

## 6. Ciò che il modello non fa

- Non rappresenta relazioni fra attributi dentro un concetto: gruppi allungati
  in diagonale richiedono più concetti.
- A un solo livello non separa gruppi molto vicini rispetto alla scala della
  tabella; l'approfondimento resta manuale.
- Senza target non garantisce il numero di concetti quando il rumore non è
  gaussiano.
- Nell'attenzione i concetti sono per testa: gli archi fra concetti non
  attraversano teste e layer.
- Con compressione, e senza addestramento, non eguaglia l'attenzione densa.
