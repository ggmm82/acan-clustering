"""A self-contained explanation of the cp-v9 concept model."""
from dash import dcc, html

TEXT = r"""
## Come il modello cerca concetti nei dati

Un concetto è una descrizione riutilizzabile di più osservazioni: per esempio
«piove e il luogo è un palazzetto». Il principio è uno solo:

> un insieme di concetti è buono quanto **riduce l'incertezza su ciò che va
> previsto**, in osservazioni che i concetti non hanno visto.

Che cosa va previsto dipende dal compito. Senza target sono tutti gli attributi:
è clustering non supervisionato. Con una classe o una quantità è quel target:
è un modello supervisionato basato su concetti. Nel linguaggio è il token
successivo. Le operazioni sono le stesse nei tre casi; cambia soltanto che
cosa viene giudicato. Questa pagina descrive il nucleo **cp-v9**, usato dai
nuovi modelli. I modelli salvati in precedenza continuano a usare le loro formule.

### 1. Un concetto è un riassunto che si può sommare

Ogni riga è un vettore $x$ di attributi. Quelli numerici sono riportati a media
zero e deviazione uno con le statistiche del solo training; una categoria resta
un attributo unico con i suoi valori, senza ordinamenti arbitrari.

Ogni concetto $i$ conserva soltanto conteggi e somme pesati dalle appartenenze
$T_{it}$ (definite nella sezione 2):

$$
C_i=\sum_tT_{it},\qquad U_i=\sum_tT_{it}x_t,\qquad Q_i=\sum_tT_{it}x_t^2,\qquad
N_{iv}=\sum_tT_{it}\,[x_t=v].
$$

Da questi numeri si ricavano il centro e la dispersione del concetto su ogni
attributo numerico, e la frequenza di ogni valore categorico:

$$
g_{ij}=\frac{U_{ij}}{C_i},\qquad s_{ij}^2=\frac{Q_{ij}}{C_i}-g_{ij}^2 .
$$

Le terne si **sommano**: aggiungere una riga, togliere una riga o unire due
concetti è la stessa operazione. Tutto ciò che segue usa solo questa proprietà.

Un cluster non è una classe. Due concetti possono prevedere la stessa classe:
nel gioco, quattro combinazioni di tempo e luogo conducono a due esiti.

### 2. Appartenenza: quale concetto spiega meglio la riga

Un concetto con poche righe non ha evidenza per essere molto preciso. La sua
dispersione viene quindi avvicinata a quella generale, che dopo la
normalizzazione vale uno, con un peso $\kappa=1$ pari a una riga:

$$
v_{ij}=\frac{C_i\,s_{ij}^2+\kappa}{C_i+\kappa},\qquad
p_{iv}=\frac{N_{iv}+\kappa\,p_v}{C_i+\kappa}.
$$

Con una sola riga $v=1/2$: il concetto resta largo. Con molte righe la
correzione scompare. L'appartenenza è la probabilità a posteriori del concetto,
calcolata sui soli attributi osservati $O$:

$$
T_{it}=\frac{F_i\,p_i(x_{t,O})}{\sum_lF_l\,p_l(x_{t,O})},\qquad F_i=\frac{C_i}{\sum_lC_l},
$$

$$
\log p_i(x_O)=-\frac12\sum_{j\in O}\left[\frac{(x_j-g_{ij})^2}{v_{ij}}+\log v_{ij}\right]
+\sum_{j\in O}\log p_{i,x_j}+\text{cost.}
$$

Il primo addendo riguarda gli attributi numerici, il secondo quelli categorici.
Gli attributi su cui il concetto è compatto pesano di più; il termine
$\log v_{ij}$ fa pagare a un concetto la propria larghezza. Un attributo
mancante viene semplicemente omesso dalla somma: non serve inventare un valore.
Per questo un solo modello può completare qualunque sottoinsieme di attributi:

$$
p(x_U\mid x_S)=\sum_iT_i(x_S)\,p_i(x_U).
$$

Il valore da prevedere non entra mai in $O$. T è una probabilità secondo il
modello, non una garanzia calibrata che la risposta sia corretta.

### 3. Generalizzare: unire i due concetti che costa meno unire

L'apprendimento parte da molti concetti specifici, circa uno ogni quattro
righe, e procede dal basso verso l'alto. Unire due concetti significa sommare
le loro terne. Il costo dell'unione è l'aumento della lunghezza della
descrizione dei dati:

$$
\ell(i)=\frac12\sum_j\omega_jC_i\log v_{ij}-\sum_{j,v}\omega_jN_{ijv}\log p_{ijv}-C_i\log C_i,
$$

$$
\Delta_{il}=\ell(i\cup l)-\ell(i)-\ell(l).
$$

A ogni passo si uniscono le coppie con $\Delta$ minimo e si aggiornano le
appartenenze. Due copie dello stesso concetto hanno sempre $\Delta<0$: unirle
conviene. La tendenza ad aggregare non è un termine aggiunto, è una conseguenza.
Il peso $\omega_j$ vale uno per gli attributi; per un target vale il numero
degli attributi, così che il target conti quanto tutti gli altri insieme.

La sequenza delle unioni è un albero di generalizzazione: in basso i casi
particolari, in alto un solo concetto che descrive tutto.

**Attributi che variano insieme.** Un concetto può conservare anche la somma dei
prodotti fra attributi numerici, che si somma come le altre:

$$
S_i=\sum_tT_{it}\,x_tx_t^\top,\qquad
\Sigma_i=\frac{S_i-C_i\,g_ig_i^\top+\kappa I}{C_i+\kappa}.
$$

Così un concetto descrive un gruppo allungato in diagonale con una sola forma,
invece che con molti concetti a gradini. Questa descrizione più ricca si usa
solo se prevede meglio le osservazioni non viste (sezione 5).

### 4. Rilevanza: quanta variabilità di un attributo spiegano i concetti

Per ogni attributo la variabilità totale si divide esattamente in una parte
interna ai concetti e una parte fra i concetti:

$$
\sigma_j^2=\sum_iF_i\,s_{ij}^2+\sum_iF_i\,(g_{ij}-\mu_j)^2 .
$$

La **rilevanza** di un concetto su un attributo è la frazione di variabilità
che il concetto elimina; la rilevanza dell'attributo è la sua media:

$$
r_{ij}=1-\frac{s_{ij}^2}{\sigma_j^2},\qquad
\eta_j^2=\sum_iF_i\,r_{ij}=\frac{\sum_iF_i\,(g_{ij}-\mu_j)^2}{\sigma_j^2}.
$$

Essere distinti dalla media e essere compatti sono quindi la stessa misura.
Un attributo irrilevante ha rilevanza vicina a zero, perché dentro ogni
concetto varia quanto varia in generale. Per un attributo categorico la stessa
formula usa le frequenze dei valori e coincide con la Category Utility del
clustering concettuale. La CP mostrata nell'app è:

$$
CP=\sqrt{\frac1m\sum_j\eta_j^2}.
$$

Sta fra zero e uno e scende quando i gruppi si allargano. **Viene riportata,
non ottimizzata:** da sola crescerebbe sempre aumentando il numero di concetti.

### 5. Quanti concetti: giudica chi non ha visto

Ogni livello dell'albero viene giudicato. Per ciascuna riga si sottrae il suo
contributo dalle terne, e si chiede ai concetti così ottenuti di prevederla:

$$
C_i^{(-t)}=C_i-T_{it},\qquad U_i^{(-t)}=U_i-T_{it}x_t,\qquad Q_i^{(-t)}=Q_i-T_{it}x_t^2 ,
$$

$$
J=\frac1n\sum_t\log p^{(-t)}(\text{ciò che va previsto}_t\mid\text{ciò che è osservato}_t).
$$

Un concetto fatto di una sola riga non può sostenere sé stesso: tolta la riga,
non resta nulla. È il principio del sostegno da osservazioni diverse, applicato
esattamente. Fra tutti i livelli visitati si conserva **il più generale** il
cui punteggio è entro un errore standard dal migliore:

$$
K^*=\min\{K:\;J_K\ge\max_{K'}J_{K'}-\mathrm{se}\}.
$$

Non si indica un numero di concetti, una soglia di purezza o di similarità.
La ricerca è però un procedimento per passi successivi: fino a 5.000 righe viene
ripetuta cinque volte da punti di partenza diversi, e lo stesso punteggio
sceglie fra tutto ciò che è stato visitato.

**Livelli.** A un solo livello la scala di riferimento è quella di tutta la
tabella: gruppi vicini fra loro, per quanto compatti, restano fusi. Ogni
concetto viene quindi riesaminato con lo stesso modello sulle proprie righe e
sulla propria scala; il riferimento del livello fine è il concetto che lo
contiene, non più la tabella intera:

$$
v_{figlio}=\frac{C\,s_{figlio}^2+\kappa\,v_{genitore}}{C+\kappa}.
$$

Un livello più fine sceglie i propri concetti con metà delle sue righe e viene
tenuto solo se, sull'altra metà, prevede meglio del concetto che lo contiene
di più di un errore standard. La previsione attraversa i livelli:

$$
p(y\mid x)=\sum_iT_i(x)\,p_i(y\mid x),
$$

dove $p_i$ è la lettura del concetto oppure, se è stato approfondito, quella dei
suoi figli. Non ci sono soglie di purezza, di righe minime o di profondità.

### 6. Le tre modalità e il linguaggio

**Clustering non supervisionato.** Va previsto ogni attributo: $J$ è la
log-probabilità dell'intera riga non vista.

**Classificazione.** Ogni concetto ha una distribuzione sulle classi e la
previsione è la loro miscela:

$$
p(y\mid x)=\sum_iT_i(x)\,p_i(y).
$$

$J$ è la log-probabilità della classe corretta di ogni riga non vista.
Un numero può essere un nome di classe, come 0/1 oppure codici 1…26: in quel
caso scegliere esplicitamente **Classi, anche codici numerici**.

**Regressione.** Il target è un attributo numerico come gli altri, letto ma
mai usato per l'appartenenza:

$$
\hat y(x)=\sum_iT_i(x)\,g_{i,y}.
$$

**Linguaggio: lo stesso modello.** Tabella e attenzione sono un caso solo.
Il peso di un concetto è la sua massa per la somiglianza attesa fra ciò che si
osserva e i suoi membri:

$$
T_i(o)\;\propto\;C_i\;\mathbb E_{z\sim i}[\kappa(o,z)].
$$

In una tabella la riga è osservata esattamente e questa formula è
l'appartenenza della sezione 2. In uno strato di attenzione il prefisso è una
tabella: ogni token passato è una riga con una chiave $k$ e un valore $v$, e la
query $q$ è una riga di cui si cerca il valore. La somiglianza è
$\exp(q^\top k)$ e si ottiene:

$$
\mathrm{out}(q)=\frac{\sum_{\tau\in W}e^{q^\top k_\tau}v_\tau+\sum_iC_i\,e^{q^\top g_i}\,\bar v_i}
{\sum_{\tau\in W}e^{q^\top k_\tau}+\sum_iC_i\,e^{q^\top g_i}} .
$$

I token recenti $W$ sono concetti di un solo membro; quelli più vecchi vengono
generalizzati come le righe. La coppia da unire è quella che costa meno a chi
legge: l'attenzione che i due concetti ricevono per la distanza dei loro membri
dal centro comune,

$$
\Delta^{lettore}_{il}=(a_i+a_l)\,\rho_{i\cup l}-a_i\rho_i-a_l\rho_l .
$$

L'attenzione ricevuta $a$ è un'altra quantità che si somma. I token che le
query continuano a leggere, come i primi del contesto, restano quindi esatti da
soli. Con un concetto per token la formula **è** l'attenzione ordinaria: per
questo un modello innestato con finestra sufficiente coincide con quello di
partenza.
In un prefisso ogni token è già «non visto» dai riassunti che lo precedono:
la regola della sezione 5 e la causalità sono la stessa regola. L'errore
introdotto dai concetti è limitato dalla distanza delle chiavi dal centro del
proprio concetto, cioè da quanta variabilità delle chiavi i concetti spiegano.
Uno stato di dimensione fissa non può conservare tutto un contesto
arbitrariamente lungo: si può ridurre la perdita, non annullarla.

### 7. Cosa dimostrano le verifiche

Le verifiche numeriche controllano formule, additività delle terne, esclusione
del target, causalità e conservazione dei modelli salvati. Non dimostrano da
sole che il modello generalizzi. Un test usato ripetutamente per scegliere
miglioramenti diventa uno strumento di sviluppo: serve poi una verifica
indipendente, che per questi dati non esiste ancora.

Misure su split 80/20, senza impostazioni per dataset:

| Dataset | Modello attuale | Concetti | Un solo livello, attributi separati | Concetti |
|---|---:|---:|---:|---:|
| Gioco | 100% | 4 | 100% | 4 |
| Adult minimale | 75,3% | 4 | 75,3% | 4 |
| Sonar | 86,5% | 13–18 | 86,5% | 13–18 |
| Titanic | 77,7% | 32–37 | 77,7% | 32–37 |
| Letter | 97,8% | 153–171 | 93,3% | 412–476 |

Su Letter gli attributi variano insieme e il modello lo usa: supera il
**95,78%** storico e i metodi di confronto. Sugli altri dati la descrizione più
ricca non prevede meglio e non viene adottata. Gioco **senza target** dà 4
concetti in dieci prove su dieci.

Limiti misurati:

- I livelli automatici compaiono di rado su questi dati: servono quando la
  tabella ha strutture a scale diverse. In un esempio con due famiglie di tre
  gruppi vicini, un livello vede solo le due famiglie e il secondo livello i
  sei gruppi.
- Senza target, se gli attributi irrilevanti non somigliano a una campana, il
  modello può aggiungere concetti oltre quelli attesi. Riportare gli attributi
  a una scala gaussiana per ranghi lo evita, ma cancella i vuoti fra gruppi
  ben separati: è disponibile, non predefinito.
- Titanic resta sotto i migliori metodi di confronto.
- Con attributi che variano insieme l'apprendimento è molto più lento
  (Letter: circa un quarto d'ora).
- **Linguaggio.** Sostituendo l'attenzione di un modello preaddestrato in tutti
  i livelli, senza addestrare nulla, l'errore di previsione è 2,76 con
  l'attenzione completa, 3,23 tenendo esatti i primi token e gli ultimi 124, e
  3,01 con 64 token esatti più 64 concetti generalizzati per il lettore.
  I concetti recuperano parte del divario; non lo chiudono, e lo strato è
  ancora più lento dell'attenzione completa.

Le formule spiegano il comportamento e i suoi limiti; i risultati sui dati
stabiliscono quanto quel comportamento sia utile nel problema concreto.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
