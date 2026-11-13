# Fondazione matematica di ACAN: validazione, correzioni ed estensione

> **Documento storico.** Il modello attuale è uno solo ed è descritto in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md). Dove questa pagina differisce, vale MODEL.md: in particolare la covarianza nel concetto, i livelli automatici e la fusione per lunghezza di descrizione sono stati tolti.

Questo documento esamina il principio matematico esposto nella scheda
**Matematica** dell'app ([`acan/vis/mathematics.py`](../acan/vis/mathematics.py)),
nel nucleo [`acan/algebra.py`](../acan/algebra.py), nei layer linguistici
[`acan/lm/`](../acan/lm) e nei due paper in [`paper/`](../paper). Ha tre parti:

- **A. Validazione**: che cosa è corretto, che cosa è falso o non dimostrato.
- **B. Fondazione proposta**: un unico principio da cui routing, rilevanza, CP,
  massa e selezione del numero di concetti si *derivano* invece di essere postulati.
- **C. Attenzione**: che cosa il layer attuale calcola davvero, il limite che
  nessuna memoria costante può superare, e la derivazione di uno strato a
  concetti come approssimazione *con errore limitato* dell'attenzione densa.

Ogni affermazione numerica è riproducibile sul nucleo di produzione:

```bash
python scripts/verify_math_foundation.py
```

## Il modello unico

Questa sezione è l'enunciato definitivo. Le parti A, B e C che seguono ne sono
la motivazione e la derivazione. Codice: un solo file di operatori,
[`acan/concept_algebra.py`](../acan/concept_algebra.py), chiamato dal nucleo
tabulare ([`acan/core/concepts.py`](../acan/core/concepts.py), NumPy) e dallo
strato di attenzione ([`acan/lm/concept_attention.py`](../acan/lm/concept_attention.py),
PyTorch).

**Oggetti.** Un'*osservazione* è una coppia $(z,y)$: una parte $z$ con cui la si
riconosce e una parte $y$ che si vuole leggere. Un *concetto* è la somma delle
statistiche dei suoi membri,

$$
C_i=\sum_{t\in i}w_t,\qquad U_i=\sum_{t\in i}w_tz_t,\qquad Q_i=\sum_{t\in i}w_tz_t^2,\qquad
Y_i=\sum_{t\in i}w_t\,\phi(y_t),
$$

con centro $g_i=U_i/C_i$, varianza $s_i^2=Q_i/C_i-g_i^2$ e lettura
$\bar y_i=Y_i/C_i$.

| | Tabella | Attenzione |
|---|---|---|
| osservazione | una riga | un token del prefisso |
| $z$ | attributi osservati | chiave |
| $y$ | target (o gli attributi mancanti) | valore |
| interrogazione | riga con $y$ ignoto | query |
| somiglianza $\kappa$ | identità: la riga è osservata esattamente | $\exp(s\,q^\top k)$ |

**Operatore 1 — appartenenza.** Il peso di un concetto per un'interrogazione
$o$ è la sua massa per la somiglianza attesa con i suoi membri:

$$
\boxed{\;T_i(o)=\frac{C_i\;\mathbb E_{z\sim i}[\kappa(o,z)]}{\sum_lC_l\;\mathbb E_{z\sim l}[\kappa(o,z)]},
\qquad \hat y(o)=\sum_iT_i(o)\,\bar y_i\;}
$$

**Operatore 2 — generalizzazione.** Due concetti si generalizzano sommando le
statistiche. Fra tutte le coppie si sceglie quella con il minimo aumento della
lunghezza di descrizione $\ell(i)=\tfrac12C_i\sum_j\log v_{ij}-C_i\log C_i$
(più il termine categorico nel tabulare):
$\Delta_{il}=\ell(i\cup l)-\ell(i)-\ell(l)$.

**Operatore 3 — giudizio.** Un concetto è valutato solo su osservazioni che non
contiene. Il criterio è la log-probabilità di ciò che va letto:
$J=\frac1n\sum_t\log p^{(-t)}(y_t\mid z_t)$.

**Proposizione U1 (tabella) [D][V].** Con $\kappa$ identità e membri gaussiani
di varianza predittiva $v_{ij}=(C_is_{ij}^2+\kappa_0\sigma_j^2)/(C_i+\kappa_0)$,
$\mathbb E_{z\sim i}[\kappa(o,z)]=\prod_j\mathcal N(o_j;g_{ij},v_{ij})$:
l'appartenenza è la posterior di una miscela e $\hat y$ la sua media
condizionata. Omettere un attributo dal prodotto lo marginalizza esattamente,
quindi lo stesso modello completa ogni sottoinsieme di attributi.

**Proposizione U2 (attenzione) [D][V].** Con $\kappa(q,k)=e^{s\,q^\top k}$ e
membri gaussiani, $\mathbb E_{k\sim i}[\kappa]=\exp(s\,q^\top g_i+\tfrac12s^2q^\top V_iq)$.
Con un concetto per token ($C=1$, $V=0$) l'operatore 1 è **esattamente**
l'attenzione softmax densa. Il termine $\log C_i$ nei logit non è un'ipotesi: è
la massa dell'operatore 1. *Verifica:* lo strato coincide con l'attenzione della
libreria entro $2\cdot10^{-5}$ quando la finestra o il numero di concetti copre
il prefisso; un modello innestato con finestra sufficiente riproduce i logit del
modello di partenza.

**Proposizione U3 (aggregazione) [D][V].** Per due concetti con le stesse
statistiche normalizzate $\Delta_{il}=-[(C_i+C_l)\log(C_i+C_l)-C_i\log C_i-C_l\log C_l]<0$:
unire copie conviene sempre, in tabella e in attenzione. È la proprietà
aggregante della CP originale, ottenuta senza un termine dedicato.

**Proposizione U4 (errore della generalizzazione) [D][V].** Per l'ordine zero
($V_i$ omesso), se ogni membro dista al più $\rho$ dal centro del proprio
concetto, $\lVert\hat y_{denso}(q)-\hat y(q)\rVert\le(e^{2s\lVert q\rVert\rho}-1)\max\lVert y\rVert$
(Teorema 6). La quantità che limita l'errore è la dispersione entro i concetti,
cioè $1-\eta^2$: la varianza spiegata è la misura di qualità comune.

**Proposizione U5 (giudizio = causalità) [D].** Poiché le statistiche sono
additive, $p^{(-t)}$ si ottiene sottraendo il contributo di $t$. In un prefisso
ordinato le statistiche al tempo $t$ contengono solo $\tau<t$: il giudizio
dell'operatore 3 è la verosimiglianza del token successivo. Escludere la riga
nel tabulare ed essere causali nel linguaggio sono la stessa condizione.

**Proposizione U6 (limite) [D].** Uno stato di $P$ numeri a $b$ bit non può
restituire esattamente più di $Pb$ bit del passato (Proposizione 5). Un numero
fisso di concetti è quindi una compressione con perdita; la perdita è quella
limitata da U4.

**Dove i due casi differiscono, e perché.**

| | Tabella | Attenzione |
|---|---|---|
| varianza usata nell'appartenenza | predittiva, $\kappa_0=1$: si giudicano righe nuove | ordine zero: i membri sono dati memorizzati, non un campione |
| riassegnazione dei membri | sì (le righe non hanno ordine) | no: il passato non si rilegge, pena la causalità |
| quante generalizzazioni | scelte dall'operatore 3, con la regola dell'errore standard | fissate dal budget di stato: è la risorsa che si vuole limitare |
| posizione | assente | i concetti riassumono chiavi non ruotate e vengono ruotati alla posizione media dei membri: è un'approssimazione, non coperta da U4 |
| ciò che si addestra | nulla oltre le statistiche | le proiezioni del modello linguistico; i concetti non hanno parametri |

Che cosa **non** è dimostrato: che con pochi concetti l'errore di U4 sia piccolo
sulle chiavi di un modello reale. Le misure sono in
[CP-V9-RESULTS.md](CP-V9-RESULTS.md).

Legenda: **[D]** dimostrato qui, **[V]** verificato numericamente dallo script,
**[C]** congettura o argomento euristico da formalizzare, **[S]** da misurare.
Il piano operativo è in [REVISION-PLAN.md](REVISION-PLAN.md).

---

## A. Validazione del principio attuale

### A.1 Ciò che è corretto

| Affermazione della pagina | Esito |
|---|---|
| $\Delta_j=\max(\mu_j-\min_j,\max_j-\mu_j)$ coincide con la forma del codice $(\max-\min)/2+\lvert\mu-(\min+\max)/2\rvert$ | **[D]** identità $\max(a,b)=\tfrac{a+b}2+\tfrac{\lvert a-b\rvert}2$ |
| $q_i=\sum_jR_{ij}^2/\sum_jR_{ij}\in[0,1]$; esempi 0,82 e 0,5 | **[D]** è la media contrarmonica, $\le\max_jR_{ij}$ |
| $h_i=(C_i^2-\sum_tT_{it}^2)/C_i^2$, pari a $(C-1)/C$ per gruppi netti | **[D]** $h_i=1-1/n^{\rm eff}_i$ con $n^{\rm eff}_i=C_i^2/\sum_tT_{it}^2$ (numerosità efficace di Kish) |
| Radicando di $CP_*$ in $[0,\sum_iF_i\tilde q_i]$ | **[D]** $z^\top Sz\le(\sum_iz_i)^2-\sum_iz_i^2\le\sum_iF_i\tilde q_i-\sum_iF_i^2\tilde q_i$, quindi il radicando è $\ge\sum_iF_i^2\tilde q_i\ge0$ |
| $r$ copie identiche: $CP_*=\sqrt{\tilde q/r}$ | **[D]** |
| $0\le B(\pi,F)\le1$ | **[D]** Cauchy–Schwarz |
| Statistiche di solo prefisso nel linguaggio | **[V]** dai test di causalità esistenti |

La pagina è inoltre onesta su tre limiti: la CP non misura la dispersione
interna, non identifica il numero di concetti e non è una probabilità.
Il problema è che questi tre limiti non sono marginali: sono la causa dei
difetti osservati (frammentazione su Letter, risultati misti, nessun effetto
misurabile sul linguaggio). I punti seguenti li rendono precisi.

### A.2 Difetti del criterio CP

**D1. La qualità $q$ non è monotona nell'evidenza. [D][V]**
$\partial q_i/\partial R_{ij}=(2R_{ij}-q_i)/\sum_lR_{il}$: è negativa per ogni
attributo con $R_{ij}<q_i/2$. Un concetto distinto su due attributi (0,9 e 0,3)
vale $q=0{,}75$, *meno* dello stesso concetto valutato sul solo primo attributo
($q=0{,}9$). Il massimo assoluto $q=1$ è raggiunto da un concetto estremo su **un
solo** attributo e identico alla media su tutti gli altri; quattro attributi
tutti a 0,8 valgono 0,8. Il gradiente spinge quindi ogni concetto a
caratterizzarsi su un attributo e a spegnere gli altri: è l'opposto di una
descrizione congiunta («piove **e** palazzetto»).

**D2. A parità di centroidi la CP è cieca alla dispersione. [V]**
Due insiemi con gli stessi centroidi e deviazione interna 0,01 oppure 0,20
ottengono la stessa $CP_*$ (0,706 e 0,705) con assegnazioni nette, mentre la
varianza spiegata passa da 0,99 a 0,69. Con assegnazioni morbide la CP reagisce
solo indirettamente, perché i centroidi si confondono. Una misura di
«organizzazione» che non distingue gruppi compatti da gruppi sovrapposti non può
guidare il clustering.

**D3. La CP è gonfiabile senza cambiare i dati né la partizione. [V]**
Poiché $a_i^{\rm eff}=(a_i+g_i)/2$ contiene il prototipo libero $a_i$,
allontanare i prototipi dai dati aumenta la rilevanza. Stessi dati, stessa
partizione netta: $CP_*$ sale da 0,706 (prototipi sui centroidi) a 0,864
(prototipi negli angoli). L'obiettivo non è una funzione dei dati e della
partizione: l'ottimizzatore ha una via di fuga che non corrisponde ad alcuna
struttura.

**D4. La penalità di duplicazione è aggirabile e la frammentazione è premiata. [V]**
$S_{ij}=\exp[-\tfrac\beta2(\cdot)]$ usa la stessa $\beta$ appresa del routing:
per due prototipi distinti qualsiasi $S_{ij}\to0$ quando $\beta$ cresce. Su una
sola nuvola omogenea, dividerla in due metà *aumenta* $CP_*$ per ogni $\beta$
provata (0,054 → 0,118 con $\beta=1$; → 0,269 con $\beta=4096$, dove la
sovrapposizione misurata è $7\cdot10^{-7}$). L'unico freno residuo è $h_i$, che
costa $q/n$ per ogni concetto aggiunto: troppo poco. È coerente con i 3.346
prototipi su Letter (circa 5 righe di training per prototipo): il modello
gerarchico si comporta come un classificatore a prototipi quasi 1-NN.

**D5. La normalizzazione per intervallo reintroduce la fragilità agli outlier. [V]**
$\Delta_j$ dipende da min e max. Un solo valore anomalo su un attributo riduce la
rilevanza di *tutti* i concetti su quell'attributo da 0,50 a 0,03. È esattamente
il limite degli intervalli rigidi $[\mathrm{MIN},\mathrm{MAX}]$ che la tesi
(cap. 6, p. 123) chiedeva di superare. Nel linguaggio lo stesso denominatore è
calcolato su prefissi di pochi token, dove min e max sono instabili.

**D6. $T$ non è la posterior di un modello normalizzato. [D]**
$T_{it}\propto\pi_i\exp[-\beta D_i(x_t)]$ è la responsabilità di una miscela
gaussiana diagonale con precisioni $2\beta W_{ij}$ **solo se** si include la
costante di normalizzazione $\tfrac12\sum_j\log W_{ij}$ di ciascun concetto.
Senza quel termine un concetto con pesi concentrati su pochi attributi è una
densità «larga» che non paga la propria larghezza. Non esiste quindi una
verosimiglianza di cui $T$ sia il passo E, e non c'è garanzia di monotonia
dell'ottimizzazione alternata.

**D7. Obiettivo e pesi non sono coerenti fra loro. [D]**
I pesi della distanza derivano dallo scostamento dalla media ($R^{route}$), cioè
da una statistica *fra* gruppi; la distanza serve però a decidere l'appartenenza,
che dipende dalla variabilità *entro* il gruppo. Il paper v8 lo riconosceva con
la «rilevanza duale» $0{,}55R_{shift}+0{,}45R_{tight}$; il nucleo attuale ha
perso il secondo termine, e i coefficienti del paper non hanno derivazione.

### A.3 Teoremi e risultati dei paper

| Enunciato | Esito |
|---|---|
| Teorema 1 (invarianza agli attributi irrilevanti) | **Dimostrazione errata.** Per rumore uniforme $\sigma=1/\sqrt{12}$ e $\max(0,1-2\sigma)\approx0{,}423$, non zero (già annotato in `PAPER-AUDIT.md`). L'enunciato corretto e quantitativo è la Proposizione 3 qui sotto. |
| Teorema 2 (XOR tramite $\omega$) | **Vero ma non necessario.** Le quattro combinazioni $(a_1,a_2)$ sono quattro gruppi separati nello spazio originale: qualunque metodo a prototipi con lettura di classe risolve lo XOR. La difficoltà riguarda gli alberi greedy, non i prototipi. $\omega$ serve a pesare gli attributi in presenza di rumore, non a «linearizzare» lo XOR. |
| Teorema 3 (collasso gravitazionale, $\gamma\ge1$) | **Falso per $\gamma=1$**: il punto simmetrico è neutro (già annotato in `CP-SELECTION.md`). Vale per $\gamma>1$, che però non corrisponde ad alcun modello probabilistico: è una tempra del prior. La versione derivata è B.5. |
| $2^m-2$ predittori supervisionati | **[D]** come conteggio. La risposta rigorosa è che *una* densità congiunta fornisce tutte le condizionali (B.3), proprietà di qualunque modello generativo. |
| Complessità lineare $O(knm)$ | **[D]** per il routing a $k$ fissato. |
| Paper LM: 38×, −99,2% KV, PPL 1,51, infilling 100% | **Non sostenuti.** Il codice di riferimento calcola Root/min/max, masse e rilevanza sull'intera sequenza (fuga di informazione dal futuro); la latenza «ibrida» è una media pesata di due misure, non una misura (`LM-AUDIT.md`). Corpus sintetico di poche frasi. Vanno ritirati, non riformulati. |
| «Immunità al context rot» | **Non dimostrata**, e per un'informazione vista una sola volta il layer attuale ha la proprietà opposta (C.1, punto 3). |

---

## B. Fondazione proposta: un solo principio

### B.1 Principio

> Un insieme di concetti è buono quanto **riduce l'incertezza sugli attributi**.

È il contenuto della tesi: la cardinalità $\lVert R\rVert=\prod_a\lVert R\rVert_a$
è il volume della regione in cui un concetto confina le sue tuple, e
generalizzare bene significa confinare molto con pochi concetti. La versione
continua del volume di un concetto è $\prod_j s_{ij}$, con $s_{ij}$ deviazione
standard entro il concetto; il suo logaritmo è, a costanti additive, l'entropia.
La tabella collega i tre linguaggi.

| Tesi (insiemistico) | Statistico | Informazionale |
|---|---|---|
| ampiezza $\lVert R\rVert_a$ del nodo | deviazione $s_{ij}$ entro il concetto | entropia condizionata dell'attributo |
| ampiezza $\lVert\mathrm{Root}\rVert_a$ | deviazione globale $\sigma_j$ | entropia marginale |
| rapporto $\lVert R\rVert_a/\lVert\mathrm{Root}\rVert_a$ | $s_{ij}/\sigma_j$ | informazione guadagnata |
| generalizzazione $\gamma(R_1,R_2)$ | somma delle statistiche sufficienti | fusione di due concetti |
| attributo irrilevante: ampiezza del nodo ≈ ampiezza della radice | $s_{ij}\approx\sigma_j$ | informazione mutua nulla |

### B.2 Statistiche sufficienti: l'«algebra» del modello

Per ogni concetto $i$ si conservano, per gli attributi numerici,

$$
C_i=\sum_tT_{it},\qquad U_i=\sum_tT_{it}x_t,\qquad Q_i=\sum_tT_{it}x_t^{\,2},
$$

da cui $g_i=U_i/C_i$ e $s_{ij}^2=Q_{ij}/C_i-g_{ij}^2$. Per gli attributi
categorici $U$ contiene già i conteggi per valore. Tre proprietà, tutte **[D]**:

1. **Additività.** Le terne $(C,U,Q)$ formano un monoide commutativo rispetto
   alla somma. Aggiungere un'osservazione, un token del prefisso o un intero
   concetto è la stessa operazione. La generalizzazione $\gamma$ della tesi è la
   somma di due terne; un albero di generalizzazione è una sequenza di somme.
2. **Località causale.** Il prefisso di una sequenza è una somma parziale: le
   statistiche sono causali per costruzione e richiedono memoria $O(km)$
   indipendente dalla lunghezza. Nessun min/max.
3. **Robustezza.** $\sigma_j^2$ e $s_{ij}^2$ non dipendono dai valori estremi
   come un intervallo: un outlier sposta la varianza di $O(1/n)$ del suo
   quadrato, non ridefinisce la scala (corregge D5).

Il nucleo attuale conserva $C$, $U$ e $\sum_tT_{it}^2$; manca $Q$.

### B.3 Modello e routing

Ogni concetto è una densità prodotto: gaussiana per gli attributi numerici,
categorica per quelli nominali, con varianze regolarizzate verso la radice
(B.5). Per un insieme $O$ di attributi osservati:

$$
T_{it}=\frac{\pi_i\,p_i(x_{t,O})}{\sum_l\pi_l\,p_l(x_{t,O})},\qquad
\log p_i(x_O)=-\frac12\sum_{j\in O}\left[\frac{(x_j-g_{ij})^2}{v_{ij}}+\log v_{ij}\right]+\text{cost.}
$$

- È il routing attuale con $\beta W_{ij}=1/(2v_{ij})$ **più** il termine
  $\log v_{ij}$ mancante (corregge D6). I pesi non sono più postulati: sono le
  precisioni entro il concetto (corregge D7). $\beta$ scompare come parametro
  libero e con essa l'aggiramento di D4.
- **Completamento universale [D].** Poiché la densità è un prodotto,
  marginalizzare gli attributi mancanti equivale a ometterli dalla somma:
  $p(x_U\mid x_S)=\sum_iT_i(x_S)\,p_i(x_U)$ per *ogni* coppia $(S,U)$. È
  l'enunciato rigoroso dell'autocompletamento della tesi: un modello, tutte le
  $2^m-2$ direzioni, senza approssimazioni aggiuntive.
- **Energia libera [D].** Con
  $\mathcal F(T,\theta)=\sum_{t,i}T_{it}\,[\log T_{it}-\log\pi_i-\log p_i(x_t)]$
  vale $\mathcal F\ge-\log p(X\mid\theta)$, con uguaglianza per il routing sopra.
  Aggiornare $T$ a parametri fissi e i parametri a $T$ fisso non aumenta mai
  $\mathcal F$: la convergenza monotona che i paper attribuivano ad Adam diventa
  una proprietà dell'obiettivo. È anche la versione corretta dell'«Energia
  Libera Algebrica» $-CP+\lambda_HH(T)$ del paper v8, il cui coefficiente
  $\lambda_H$ non era specificato: qui vale 1 e non è una scelta.

### B.4 Rilevanza e CP come varianza spiegata

**Proposizione 1 (decomposizione) [D].** Per ogni assegnazione morbida con
$\sum_iT_{it}=1$ e ogni attributo $j$:

$$
\sigma_j^2=\underbrace{\sum_iF_i\,s_{ij}^2}_{\text{entro i concetti}}
+\underbrace{\sum_iF_i\,(g_{ij}-\mu_j)^2}_{\text{fra i concetti}} .
$$

*Dimostrazione.* $\sum_tT_{it}(x_{tj}-\mu_j)^2=\sum_tT_{it}(x_{tj}-g_{ij})^2+C_i(g_{ij}-\mu_j)^2$
perché il termine misto si annulla per definizione di $g_{ij}$; si somma su $i$
e si divide per $n$, usando $\sum_iF_ig_{ij}=\mu_j$. ∎

Si definiscono la **rilevanza** del concetto $i$ sull'attributo $j$ e la
**rilevanza dell'attributo**:

$$
r_{ij}=1-\frac{s_{ij}^2}{\sigma_j^2},\qquad
\eta_j^2=\sum_iF_i\,r_{ij}=\sum_iF_i\frac{(g_{ij}-\mu_j)^2}{\sigma_j^2}\in[0,1].
$$

L'uguaglianza centrale è la Proposizione 1 divisa per $\sigma_j^2$. Conseguenze:

- La rilevanza «di scostamento» e quella «di compattezza» del paper v8 **sono
  la stessa quantità in media**: non servono i coefficienti 0,55 e 0,45.
- $\eta_j^2$ sta in $[0,1]$ senza clip, senza min/max, senza il prototipo
  libero (corregge D3 e D5) e *diminuisce* quando i gruppi si allargano
  (corregge D2).
- $\eta_j^2$ è il rapporto di correlazione: la frazione di varianza
  dell'attributo spiegata dai concetti. Un attributo irrilevante ha $\eta_j^2\approx0$.

$$
\boxed{\;CP^2=\frac{\sum_j\omega_j\,\eta_j^2}{\sum_j\omega_j}\;}
$$

con $\omega_j\ge0$ pesi di utilità (uniformi senza informazioni a priori). $CP$
è la radice della varianza spiegata media: resta in $[0,1]$, è monotona in ogni
$\eta_j^2$ (corregge D1) ed è una funzione dei soli dati e delle assegnazioni.

**Proposizione 2 (Category Utility) [D][V].** Su un attributo categorico
codificato con indicatori, con $p_{iv}$ frequenza del valore $v$ nel concetto
$i$ e $p_v$ frequenza globale:

$$
\sum_iF_i\sum_v(p_{iv}-p_v)^2=\sum_iF_i\sum_vp_{iv}^2-\sum_vp_v^2=K\cdot CU .
$$

Lo scarto fra concetti è esattamente $K$ volte la Category Utility di
Gluck–Corter usata da COBWEB, cioè la misura che la tesi citava come
riferimento. La CP proposta è quindi la Category Utility estesa agli attributi
continui e normalizzata, non un indice nuovo. *Dimostrazione:* si sviluppa il
quadrato e si usa $\sum_iF_ip_{iv}=p_v$. ∎

**Legame con l'informazione [D].** Per concetti gaussiani
$J_j=\tfrac12\sum_iF_i\log(\sigma_j^2/s_{ij}^2)$ è il guadagno di informazione
gaussiano (logaritmo del rapporto di ampiezza della tabella B.1) e per Jensen
$J_j\ge-\tfrac12\log(1-\eta_j^2)\ge\tfrac12\eta_j^2$. $CP^2$ è dunque un minorante
dell'informazione catturata, privo delle singolarità del logaritmo.
$\mathcal F$ (B.3) e $CP$ sono calcolate dalle stesse statistiche $(C,U,Q)$:
la prima si ottimizza, la seconda si legge e si mostra.

**Proposizione 3 (attributi irrilevanti) [D] per l'attesa, [C] per la soglia.**
Sia $X_j$ indipendente dai concetti. Allora (i) in popolazione $r_{ij}=0$ per
ogni $i$ e il contributo di $j$ ai logit del routing è identico per tutti i
concetti: si cancella esattamente nella softmax; (ii) su $n$ osservazioni e $K$
concetti $\mathbb E[\eta_j^2]=(K-1)/(n-1)$; (iii) la differenza di logit fra due
concetti dovuta a $m_{irr}$ attributi irrilevanti ha media nulla e deviazione
$O(\sqrt{m_{irr}/C_{\min}})$, mentre il segnale degli attributi rilevanti è
$\tfrac12\sum_{j\in rel}d_j^2$ con $d_j$ separazione standardizzata dei
centroidi. Il routing resta corretto finché

$$
\sum_{j\in rel}d_j^2\;\gg\;\sqrt{m_{irr}/C_{\min}} .
$$

Questo sostituisce il Teorema 1: non un'immunità per qualunque $m_{irr}$, ma una
soglia che dipende da separazione e sostegno. Il punto (ii) è verificato su 100
estrazioni: $\eta^2$ medio 0,00751 su 48 attributi di rumore contro il valore
nullo 0,00752, e 0,987 sugli attributi rilevanti **[V]**. Il punto (iii) va formalizzato con
una disuguaglianza di concentrazione.

### B.5 Sostegno, massa e numero di concetti

La varianza spiegata cresce sempre con $K$ (vale 1 con un concetto per riga).
Nessun indice di sola dispersione identifica $K$: serve un prezzo per la
complessità, e deve derivare dal modello.

**Sostegno come regolarizzazione [D].** Con un prior di forza $\kappa$ centrato
sulla radice, la varianza a posteriori del concetto è

$$
v_{ij}=\frac{C_i\,s_{ij}^2+\kappa\,\sigma_j^2}{C_i+\kappa}
\quad\Longrightarrow\quad
\tilde r_{ij}=1-\frac{v_{ij}}{\sigma_j^2}=\frac{C_i}{C_i+\kappa}\,r_{ij}.
$$

Un concetto con una sola riga ha rilevanza quasi nulla perché non ha evidenza
per restringere la radice. Con $\kappa=1$ il fattore $C_i/(C_i+1)$ è
praticamente il fattore $h_i=(C_i-1)/C_i$ già introdotto in
`CP-PAIR-SUPPORT.md`: quella correzione viene conservata e acquista una
derivazione. La stessa $v_{ij}$ entra nel routing, quindi il sostegno agisce
anche sulle assegnazioni e non solo sul punteggio. Sotto l'ipotesi nulla
$\eta^2$ si corregge con $1-(1-\eta^2)\tfrac{n-1}{n-K}$, che ha media zero in
assenza di struttura **[V]** ($-4\cdot10^{-6}$ sui 48 attributi di rumore).

**Numero di concetti: generalizzazione giudicata da chi non ha visto [D] per
la forma, [V] per l'effetto.** Il passo M esatto per le masse è $\pi_i=F_i$:
il fattore di Bhattacharyya $B(\pi,F)$ diventa superfluo. Per il numero di
concetti la revisione ha implementato una regola senza costanti di penalità:

1. si parte da molti concetti specifici e si **generalizza**: la coppia fusa è
   quella con il minimo aumento di lunghezza di descrizione,
   $\Delta_{il}=\ell(i\cup l)-\ell(i)-\ell(l)$ con
   $\ell(i)=\tfrac12\sum_j\omega_jC_{ij}\log v_{ij}-\sum_{j,v}\omega_jN_{ijv}\log p_{ijv}-C_i\log C_i$.
   Due copie dello stesso concetto hanno sempre $\Delta<0$: l'aggregazione
   della CP originale è una conseguenza, non un termine aggiunto;
2. ogni livello è **giudicato** dalla log-probabilità di ciò che va previsto,
   calcolata per ciascuna osservazione dopo aver sottratto il suo contributo
   dalle statistiche (possibile perché sono additive);
3. resta il livello più generale entro un errore standard dal migliore.

La sequenza delle fusioni è l'albero di generalizzazione della tesi. La regola
MML di Figueiredo–Jain, $\pi_i\propto\max(0,C_i-N_p/2)$, resta l'alternativa in
forma chiusa: non è stata adottata perché su tabelle con molti attributi e
poche righe (Sonar: 60 attributi, 166 righe) ammette al più due concetti.
Misure e limiti in [CP-V9-RESULTS.md](CP-V9-RESULTS.md): Letter passa da 1.539
a 412–476 concetti con accuratezza maggiore; senza target il conteggio non è
garantito in presenza di rumore non gaussiano.

### B.6 Obiettivi nelle tre modalità

| Modalità | Attuale | Proposto |
|---|---|---|
| Senza target | $\lambda(1-CP_*)$ | verosimiglianza congiunta; giudizio: log-probabilità di tutti gli attributi di ogni riga non vista; $CP$ è il valore riportato |
| Classificazione | $\mathrm{NLL}(y\mid x)+\lambda(1-CP_*)$ | stessa verosimiglianza con il target pesato quanto tutti gli attributi; giudizio: log-probabilità del target dati gli attributi |
| Regressione | dispersione pesata $+\lambda(1-CP_*)$ | come sopra: il target numerico è un attributo gaussiano letto ma escluso dal routing |
| Linguaggio | $\mathrm{NLL}+\lambda\,\overline{(1-CP_*)}$ | solo NLL (ed eventuale KL del teacher); i concetti non sono un regolarizzatore ma la *memoria*, e la loro qualità è l'errore di approssimazione di C.3 |

Per la classificazione la forma proposta è un ibrido generativo–discriminativo:
$\lambda\to0$ dà il classificatore puro, $\lambda$ grande il modello congiunto
che permette anche il completamento. La regressione non richiede più una
formula separata.

---

## C. Attenzione: che cosa si può e non si può ottenere

### C.1 Che cosa calcola il layer attuale

Nel ramo a concetti di `ACANConceptAttentionLayer` e delle sue varianti
(`attention.py`, `shared_attention.py`, `cp_attention.py`), con
$C_{i,t}=\sum_{\tau\le t}T_{i\tau}$:

$$
c_t=\sum_iT_{it}\,\frac{\sum_{\tau\le t}T_{i\tau}v_\tau}{C_{i,t}}
=\sum_{\tau\le t}A_{t\tau}\,v_\tau,\qquad
A_{t\tau}=\sum_i\frac{T_{it}\,T_{i\tau}}{C_{i,t}} .
$$

**Proposizione 4 [D][V].** $A$ è un nucleo di attenzione causale stocastico per
righe, di rango al più $k$. Il layer è quindi attenzione lineare (Katharopoulos
et al., 2020) con mappa di caratteristiche $\varphi(x)=T(x)$ sul simplesso e
normalizzazione per slot; equivalentemente è una rete ricorrente con stato
$k\times d$. Tre conseguenze:

1. **Non è una sostituzione dell'attenzione densa, è una sua compressione di
   rango $k$.** Va confrontato con attenzione lineare, Performer, Nyströmformer,
   Routing Transformer, clustered attention, Infini-attention (finestra locale
   più memoria compressiva: l'architettura più vicina), Mamba, RWKV, xLSTM. I
   paper lo confrontano solo con l'attenzione densa.
2. **Dentro uno slot non c'è ordine né recenza.** I pesi $T_{i\tau}/C_{i,t}$
   non dipendono dalla posizione: lo slot è una media corrente.
3. **Diluizione $1/C$.** Un token visto una sola volta pesa al più $1/C_{i,t}$,
   che decresce come $k/L$ (con $L=256$, $k=8$ nessun token passato pesa più di 0,033 **[V]**). Per
   un'informazione rara in un contesto lungo l'effetto è l'opposto
   dell'«immunità al context rot» dichiarata.

Altri scostamenti rilevati nel codice, da trattare come difetti:

- la finestra locale calcola `scores = (q*kw).sum(-1)/sqrt(d)`: **una sola testa**
  su tutte le $d$ dimensioni, mentre le proiezioni copiate dal modello base sono
  state addestrate per una softmax *per testa* con scala $\sqrt{d_{head}}$. I
  pesi preaddestrati sono riusati in un calcolo diverso anche dentro la finestra;
- il routing dipende dal solo token corrente e dalle statistiche del prefisso,
  non da una query: ciò che si legge dalla memoria non può dipendere da ciò che
  serve;
- costanti senza derivazione: 0,55/0,45 (contesto/locale e rilevanza), 0,75/0,25,
  $1-1{,}5\,\mathrm{std}$, $\gamma=1{,}15$, soglia di pruning 0,02;
- nel profilo `cp-joint` un doppio ciclo Python su batch e token con un calcolo
  $O(k^2d)$ per token in float64: lineare in $L$ ma con costante enorme
  (4,9–28 token/s misurati).

### C.2 Il limite che nessuna memoria costante supera

**Proposizione 5 (capacità) [D].** Sia lo stato di decodifica composto da
$P$ numeri a $b$ bit ($P=N_{layer}(k+w)d$ a meno di costanti per ACAN). Il
compito di richiamo associativo — $N$ coppie chiave→valore con valori in un
alfabeto $V$, poi una chiave — è risolto esattamente solo se

$$
P\,b\;\ge\;N\log_2\lvert V\rvert .
$$

*Dimostrazione.* La risposta dipende dal contesto solo attraverso lo stato. Se
gli stati possibili sono meno delle $\lvert V\rvert^N$ tabelle, due tabelle
diverse producono lo stesso stato; esiste una chiave su cui differiscono, e su
quella chiave almeno una delle due risposte è sbagliata. ∎

Quindi «memoria $O(1)$ in $L$» e «nessuna perdita» sono incompatibili per
*qualunque* architettura, non solo per ACAN. L'attenzione densa paga $O(Ld)$ di
memoria proprio per conservare tutto. L'obiettivo «risolvere il problema
dell'attenzione quadratica» va perciò riformulato in modo falsificabile:

> A un budget di stato fissato $(k,w)$, minimizzare l'errore rispetto
> all'attenzione densa, con un limite dimostrato dell'errore e una frontiera
> qualità/memoria misurata.

### C.3 Lo strato a concetti derivato dall'attenzione densa

Sia una testa di attenzione con query $q$, chiavi $k_\tau$, valori $v_\tau$:
$\mathrm{out}(q)=\sum_\tau p_\tau v_\tau$, $p_\tau\propto e^{q^\top k_\tau}$.
Si partizionano le chiavi in slot con statistiche $(C_i,g_i,V_i)$ sulle chiavi
(B.2) e valore medio $\bar v_i$. La massa di attenzione che la softmax densa
assegna allo slot $i$ è

$$
\sum_{\tau\in i}e^{q^\top k_\tau}=C_i\;\mathbb E_i\!\left[e^{q^\top k}\right]
\;\approx\;C_i\,\exp\!\left(q^\top g_i+\tfrac12\,q^\top V_i\,q\right),
$$

esatta se le chiavi dello slot sono gaussiane (funzione generatrice dei
momenti). Lo strato proposto è una **sola softmax** sulle $w$ chiavi recenti,
conservate esatte, e sui $k$ slot:

$$
\mathrm{out}(q)=\frac{\sum_{\tau\in W}e^{q^\top k_\tau}v_\tau
+\sum_iC_i\,e^{q^\top g_i+\frac12q^\top V_iq}\,\bar v_i}
{\sum_{\tau\in W}e^{q^\top k_\tau}+\sum_iC_i\,e^{q^\top g_i+\frac12q^\top V_iq}} .
$$

Un token entra nello slot quando esce dalla finestra (somma di statistiche:
nessun doppio conteggio). Proprietà:

1. **La massa gravitazionale è derivata [D].** Il logit dello slot contiene
   $\log C_i$ con coefficiente **esattamente 1**: è il «prior di massa» dei
   paper, ora conseguenza dell'aggregazione e non ipotesi. $\gamma=1{,}15$ è una
   deviazione ingiustificata. (Lo stesso termine compare come *proportional
   attention* nel token merging.)
2. **La rilevanza è derivata [D].** Il termine $\tfrac12q^\top V_iq$ pesa le
   direzioni in cui lo slot è disperso: è la correzione di secondo ordine, la
   controparte in attenzione di $s_{ij}^2$.
3. **Esattezza ai limiti [D].** Con $w\ge L$, oppure con uno slot per token, lo
   strato coincide con l'attenzione densa. L'innesto può quindi conservare la
   funzione del modello preaddestrato e degradare con continuità al diminuire
   di $(k,w)$; non richiede di riaddestrare da uno stato casuale.
4. **Un solo normalizzatore.** Spariscono 0,55/0,45 e la soglia di pruning.
5. **Costo.** Per token e per testa $O((k+w)d_{head})$ in tempo e memoria.

**Teorema 6 (errore di approssimazione) [D][V].** Se ogni chiave dista al più
$\rho$ dal centroide del proprio slot ($\lVert k_\tau-g_{c(\tau)}\rVert\le\rho$),
lo strato di ordine zero (senza il termine in $V_i$) soddisfa

$$
\lVert\mathrm{out}_{dense}(q)-\mathrm{out}_{slot}(q)\rVert
\;\le\;\left(e^{2\lVert q\rVert\rho}-1\right)\max_\tau\lVert v_\tau\rVert .
$$

*Dimostrazione.* Lo strato di ordine zero equivale ad assegnare a ogni token il
punteggio del proprio centroide: $\tilde p_\tau\propto e^{q^\top g_{c(\tau)}}$.
Per Cauchy–Schwarz $\lvert q^\top k_\tau-q^\top g_{c(\tau)}\rvert\le\lVert q\rVert\rho$,
quindi numeratore e normalizzatore cambiano ciascuno al più di un fattore
$e^{\pm\lVert q\rVert\rho}$ e $\tilde p_\tau/p_\tau\in[e^{-2\lVert q\rVert\rho},e^{2\lVert q\rVert\rho}]$.
Allora $\sum_\tau\lvert\tilde p_\tau-p_\tau\rvert\le e^{2\lVert q\rVert\rho}-1$ e la
tesi segue dalla disuguaglianza triangolare. Le chiavi della finestra hanno
$\rho=0$. ∎

**Il legame organico fra le due metà del progetto.** $\rho$ è la dispersione
entro lo slot *nello spazio delle chiavi*. Massimizzare la varianza spiegata
delle chiavi (la $CP$ di B.4) significa minimizzare la quantità che limita
l'errore dell'attenzione. La CP smette di essere un regolarizzatore con peso
$\lambda$ arbitrario e diventa la misura di qualità della memoria; il
clustering concettuale della tesi e l'attenzione sub-quadratica sono lo stesso
problema. Ne segue anche che cosa raggruppare: le **chiavi per testa**, non
$\mathrm{sigmoid}(h)$.

Verifica su chiavi sintetiche raggruppate ($L=2048$, $k=16$, stato pari allo
0,8% della cache densa, norma media dell'uscita 2,97) **[V]**: errore medio
0,0072 con $\log C_i$; 0,0071 con la correzione gaussiana; **1,06 senza il
termine di massa**; il limite del Teorema 6 è rispettato in tutte le 200 query.
È un caso ideale: indica che la derivazione è corretta, non che le chiavi di un
modello reale siano altrettanto raggruppabili.

**Misura sul modello reale.** A pesi invariati, su SmolLM2-135M, riassumere
chiavi già ruotate dalla posizione peggiora la sola finestra (NLL 5,81 contro
5,51): quelle chiavi non stanno vicino al proprio centroide e l'ipotesi del
Teorema 6 non vale. Riassumendo le chiavi **prima** della rotazione e ruotando
il concetto alla posizione media dei membri, i concetti migliorano la finestra
(5,26), ma restano lontani dall'attenzione densa (2,76). È la forma adottata
nel modello unico. Dettagli in [CP-V9-RESULTS.md](CP-V9-RESULTS.md).

### C.4 Questioni aperte, da non presentare come risolte

- **RoPE [S], misurato.** Riassumere chiavi ruotate è dannoso; il modello riassume chiavi non ruotate. Le chiavi sono ruotate in funzione della posizione; il centroide
  di chiavi ruotate a posizioni lontane attenua le componenti ad alta frequenza.
  Occorre scegliere fra statistiche su chiavi non ruotate con una posizione
  media per slot, e slot separati per bande di posizione. Va misurato.
- **Raggruppabilità reale [S].** Il Teorema 6 è utile solo se $\lVert q\rVert\rho$
  è piccolo con $k\ll L$ sulle chiavi di un modello preaddestrato. Si misura
  senza addestrare nulla (fase 2).
- **Politica di assegnazione e fusione [C].** Assegnazione al centroide più
  vicino, nascita di slot fino al budget, fusione dei due slot più vicini
  quando il budget è pieno (la somma di B.2). Che questa politica online sia
  vicina all'ottimo offline non è dimostrato.
- **Covarianza chiave–valore.** L'uso di $\bar v_i$ ignora la covarianza fra
  punteggio e valore entro lo slot; è coperto dal Teorema 6 ma può dominare
  l'errore pratico.
- **Capacità.** La Proposizione 5 resta: oltre la capacità dello stato il
  richiamo esatto fallisce. Il prodotto finale deve dichiarare la propria
  frontiera, non negarla.

---

## Riepilogo delle correzioni

| Elemento attuale | Sostituzione | Motivo |
|---|---|---|
| $\Delta_j$ da min/max | $\sigma_j$ da $(C,U,Q)$ | D5; causale senza cummin/cummax |
| $R_{ij}=\mathrm{clip}(\omega_j\lvert a^{eff}_{ij}-\mu_j\rvert/\Delta_j)$ | $r_{ij}=1-s_{ij}^2/\sigma_j^2$ | D2, D3; Proposizione 1 |
| $q_i=\sum R^2/\sum R$ | media pesata di $\eta_j^2$ | D1; Proposizione 2 |
| $W_{ij}\propto R_{ij}^2+\varepsilon$, $\beta$ appresa | precisioni $1/v_{ij}$ e termine $\log v_{ij}$ | D6, D7 |
| $h_i$ moltiplicativo | regolarizzazione $C_i/(C_i+\kappa)$ dentro $v_{ij}$ | stessa idea, derivata e attiva anche nel routing |
| $S$, $B(\pi,F)$, trasferimenti di massa per tentativi | passo M con estinzione $\max(0,C_i-N_p/2)$ | D4 |
| $\lambda(1-CP_*)$ come obiettivo | energia libera $\mathcal F$; $CP$ riportata | monotonia garantita |
| $\gamma\log C$, $\gamma=1{,}15$ | $\log C_i$ | C.3, punto 1 |
| contesto e finestra mescolati 0,55/0,45 | una softmax su finestra e slot | C.3, punto 4 |
| routing su $\mathrm{sigmoid}(h)$ | slot sulle chiavi, per testa | Teorema 6 |
| Teoremi 1 e 3, tabelle del paper LM | Proposizioni 3, 5 e Teorema 6; nuove misure | A.3 |

## Riferimenti da verificare nella revisione bibliografica

Gluck & Corter (1985) e Fisher (1987) per la Category Utility; Dempster, Laird &
Rubin (1977) e Neal & Hinton (1998) per EM ed energia libera; Figueiredo & Jain
(2002) per la selezione MML delle componenti; Lasserre, Bishop & Minka (2006)
per gli ibridi generativo–discriminativi; Katharopoulos et al. (2020) per
l'attenzione lineare; Vyas et al. (2020) per la clustered attention; Bolya et
al. (2023) per la proportional attention; Munkhdalai et al. (2024) per
Infini-attention; Jelassi et al. (2024) per i limiti di copia dei modelli a
stato fisso. Le citazioni sono indicate a memoria e vanno controllate sulle
fonti prima di entrare in un paper. Nei paper attuali vanno verificate o rimosse
la voce «Context Rot Study (2026)», la voce Medium di Perez (2025) e i
collegamenti a Wikipedia usati come fonte primaria.
