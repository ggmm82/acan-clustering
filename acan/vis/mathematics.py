"""A self-contained explanation of the concept graph, the one model of the app."""
from dash import dcc, html

TEXT = r"""
## Il grafo dei concetti

Un concetto è una descrizione riutilizzabile di più osservazioni: per esempio
«piove e il luogo è un palazzetto». Il modello è **uno solo**, per le tabelle e
per il linguaggio. Ha un solo tipo di oggetto, il concetto, e tre operazioni:
riconoscere, generalizzare, giudicare.

> Un insieme di concetti è buono quanto permette di **leggere ciò che serve**
> da osservazioni che i concetti non hanno visto.

Che cosa si legge dipende dal compito. Senza target si legge ogni attributo:
è clustering non supervisionato. Con una classe o una quantità si legge quel
target. Nel linguaggio un token legge i valori dei token che lo precedono.
Le formule sono le stesse nei tre casi.

### 1. Nodi: un concetto è un riassunto che si può sommare

Ogni osservazione ha una parte con cui la si **riconosce**, $z$, e una parte che
si **legge**, $y$. In una tabella $z$ sono gli attributi noti e $y$ il target (o
gli attributi mancanti); in uno strato di attenzione $z$ è la chiave di un token
e $y$ il suo valore.

Un concetto $i$ conserva soltanto somme sulle osservazioni che gli appartengono,
pesate dalle appartenenze $T_{it}$:

$$
C_i=\sum_tT_{it},\qquad U_i=\sum_tT_{it}z_t,\qquad Q_i=\sum_tT_{it}z_t^2,\qquad
Y_i=\sum_tT_{it}y_t,\qquad A_i=\text{letture ricevute}.
$$

Da queste somme si ricavano il centro, la dispersione e la lettura del concetto:

$$
g_i=\frac{U_i}{C_i},\qquad s_i^2=\frac{Q_i}{C_i}-g_i^2,\qquad \bar y_i=\frac{Y_i}{C_i}.
$$

Per un attributo categorico le somme sono i conteggi dei valori. Tutte le somme
crescono in proporzione al numero di attributi: il modello non conserva
relazioni fra coppie di attributi, quindi il suo costo non cresce più che
linearmente con le colonne. Aggiungere un'osservazione, toglierla o unire due
concetti è sempre una somma.

Un cluster non è una classe: due concetti possono prevedere la stessa classe.

### 2. Riconoscere: massa per somiglianza attesa

Il peso di un concetto per un'osservazione $o$ è la sua massa per la
somiglianza attesa fra $o$ e i suoi membri:

$$
T_i(o)=\frac{C_i\;\mathbb E_{z\sim i}[\kappa(o,z)]}{\sum_lC_l\;\mathbb E_{z\sim l}[\kappa(o,z)]},\qquad
\hat y(o)=\sum_iT_i(o)\,\bar y_i .
$$

In una tabella la riga è osservata esattamente e la formula diventa

$$
\log\big(C_i\,\mathbb E[\kappa]\big)=\log C_i-\frac12\sum_{j\in O}\left[\frac{(o_j-g_{ij})^2}{v_{ij}}+\log v_{ij}\right]
+\sum_{j\in O}\log p_{i,o_j},
$$

dove $O$ sono gli attributi osservati: uno mancante viene semplicemente omesso,
e lo stesso modello completa qualunque sottoinsieme di attributi. Il target non
entra mai in $O$. Un concetto con poche osservazioni non può essere molto
preciso: la sua dispersione è avvicinata a quella generale, che vale uno, con
il peso di un'osservazione,

$$
v_{ij}=\frac{C_i\,s_{ij}^2+1}{C_i+1}.
$$

In uno strato di attenzione la somiglianza fra una query $q$ e una chiave $k$ è
$e^{q^\top k}$ e la stessa formula dà

$$
\mathrm{out}(q)=\frac{\sum_{\tau\in W}e^{q^\top k_\tau}v_\tau+\sum_iC_i\,e^{q^\top g_i}\,\bar v_i}
{\sum_{\tau\in W}e^{q^\top k_\tau}+\sum_iC_i\,e^{q^\top g_i}} ,
$$

dove i token recenti $W$ sono concetti di un solo membro. Con un concetto per
token questa **è** l'attenzione ordinaria.

### 3. Generalizzare: unire dove chi legge perde meno

Leggere un concetto al posto dei suoi membri introduce un errore limitato da
due quantità: quante letture il concetto riceve e quanto i suoi membri distano
dal centro. Il raggio di un concetto è la sua dispersione nella parte
riconosciuta, sulla scala generale:

$$
\rho_i=\sqrt{\frac1m\sum_j\frac{s_{ij}^2}{\sigma_j^2}} .
$$

Unire due concetti significa sommarne le somme. Il costo dell'unione è la
perdita di chi legge:

$$
\Delta_{il}=(a_i+a_l)\,\rho_{i\cup l}-a_i\,\rho_i-a_l\,\rho_l .
$$

In una tabella ogni riga legge i concetti a cui appartiene, quindi le letture
ricevute sono la massa: $a_i=C_i$. Nell'attenzione $a_i$ è l'attenzione che il
concetto ha ricevuto dalle query precedenti. Il costo non è mai negativo e vale
zero solo fra due copie dello stesso concetto: unire copie è sempre gratuito.

Si parte da molti concetti specifici e si unisce ogni volta la coppia con
$\Delta$ minimo. La sequenza delle unioni è un albero di generalizzazione: in
basso i casi particolari, in alto un solo concetto. Nel linguaggio i token che
le query continuano a leggere, come i primi del contesto, hanno un costo alto
e restano esatti da soli.

### 4. Archi e rilevanza: la struttura che si può ispezionare

**Archi.** Due concetti sono collegati quando qualche osservazione esita fra
loro. Il peso dell'arco somma, sulle osservazioni, il prodotto delle due
appartenenze maggiori:

$$
E_{il}=\frac1n\sum_tT_{it}\,T_{lt}\quad\text{per i due concetti a cui }t\text{ appartiene di più.}
$$

Gli archi mostrano quali concetti sono vicini e dove passano i confini incerti.
Nel grafo dell'app sono le linee tratteggiate fra i gruppi.

**Rilevanza:** quanta variabilità di un attributo i concetti spiegano. Per ogni
attributo la variabilità totale si divide esattamente in una parte interna ai
concetti e una parte fra i concetti:

$$
\sigma_j^2=\sum_iF_i\,s_{ij}^2+\sum_iF_i\,(g_{ij}-\mu_j)^2,\qquad F_i=\frac{C_i}{n}.
$$

$$
r_{ij}=1-\frac{s_{ij}^2}{\sigma_j^2},\qquad
\eta_j^2=\sum_iF_i\,r_{ij},\qquad CP=\sqrt{\frac1m\sum_j\eta_j^2}.
$$

Un attributo irrilevante ha rilevanza vicina a zero. La CP sta fra zero e uno
ed è **riportata, non ottimizzata**: crescerebbe sempre aumentando i concetti.

**Livelli.** I concetti non stanno tutti sullo stesso piano. Sotto la radice,
che è la tabella intera, il livello 1 contiene i concetti generali; i livelli
successivi contengono i comportamenti particolari, cioè ciò che si distingue
solo dopo aver tolto quello che il livello sopra già spiega. La radice
standardizza la tabella con media e deviazione; ogni livello fa lo stesso per
il livello sotto, con il centro e la dispersione del concetto di ogni
osservazione:

$$
z'_t=z_t-\sum_pT_{pt}\,g_p,\qquad \sigma'^2_{tj}=\sum_pT_{pt}\,s^2_{pj}.
$$

Quanto un livello aggiunge a quello sopra lo misura la **CP a coppie**: la
media, su tutte le coppie di osservazioni, di «stesso concetto» per quanto le
due si somigliano più della coppia media del livello sopra:

$$
CP^\circ=\frac1{n(n-1)}\sum_iC_i(C_i-1)\Bigl(1-\frac1m\sum_j\frac{\hat s^2_{ij}}{\bar\sigma'^2_{ij}}\Bigr).
$$

Vale zero per un solo concetto, per concetti di una sola osservazione e per un
livello misurato contro sé stesso; cresce solo con concetti insieme grandi e
distinti. Il livello 1 è il taglio che la rende massima sulla scala della
radice; ogni livello successivo è il massimo contro il livello sopra.

Due livelli consecutivi sono uniti da una **matrice densa**
$P_{pi}=\Pr(\text{concetto }p\text{ sopra}\mid\text{concetto }i)$: un concetto
può discendere da più concetti del livello sopra. Se lo stesso comportamento
compare dentro due concetti generali diversi, visto attraverso di essi è un
solo comportamento e diventa **un solo nodo con due genitori**, non due
frammenti come in un albero. Nel grafo dell'app i livelli sono le righe sopra
i gruppi e le frecce fra le righe sono le celle della matrice.

### 5. Giudicare: decide chi non ha visto

Ogni passo della generalizzazione viene giudicato. Per ciascuna osservazione si sottrae
il suo contributo dalle somme e si chiede ai concetti così ottenuti di leggerla:

$$
C_i^{(-t)}=C_i-T_{it},\qquad U_i^{(-t)}=U_i-T_{it}z_t,\qquad Q_i^{(-t)}=Q_i-T_{it}z_t^2 ,
$$

$$
J=\frac1n\sum_t\log p^{(-t)}(y_t\mid z_t).
$$

Un concetto fatto di una sola osservazione non può sostenere sé stesso: è il
principio del sostegno da osservazioni diverse. Il **livello di lavoro**, quello
che fa le previsioni, è il più generale fra quelli il cui punteggio è entro un
errore standard dal migliore:

$$
K^*=\min\{K:\;J_K\ge\max_{K'}J_{K'}-\mathrm{se}\}.
$$

La CP a coppie non sa fermarsi da sola, quindi anche i livelli intermedi
passano dal giudizio: un livello resta solo se, come descrizione della tabella,
legge le osservazioni non viste meglio del livello sopra.

Nel linguaggio la stessa regola è la causalità: un token è letto solo da somme
che non lo contengono. Lì il numero di concetti non è scelto dal giudizio ma
fissato dalla memoria che si vuole usare.

### 6. Le modalità

**Clustering non supervisionato.** Si legge ogni attributo: $J$ è la
log-probabilità dell'intera riga non vista.

**Classificazione.** Ogni concetto ha una distribuzione sulle classi e la
previsione è la loro miscela:

$$
p(y\mid x)=\sum_iT_i(x)\,p_i(y).
$$

Un numero può essere un nome di classe, come 0/1 oppure codici 1…26: in quel
caso scegliere esplicitamente **Classi, anche codici numerici**.

**Regressione.** Il target è un attributo numerico come gli altri, letto ma
mai usato per riconoscere:

$$
\hat y(x)=\sum_iT_i(x)\,g_{i,y}.
$$

**Linguaggio.** Il prefisso è una tabella: ogni token passato è una riga con
una chiave e un valore, e la query è una riga di cui si cerca il valore. Le
chiavi vengono riassunte prima della rotazione di posizione e il concetto è
riportato alla posizione media dei suoi membri. Uno stato di dimensione fissa
non può conservare tutto un contesto arbitrariamente lungo: si riduce la
perdita, non la si annulla.

### 7. Cosa dimostrano le verifiche

Le verifiche numeriche controllano le formule, che tabella e attenzione usino
lo stesso costo di unione, l'esclusione del target, la causalità e che senza
compressione lo strato coincida con l'attenzione ordinaria. Non dimostrano da
sole che il modello generalizzi.

Misure su split 80/20 ripetuti tre volte, senza impostazioni per dataset:

| Dataset | Accuratezza | Concetti per livello (un seme) | Tempo |
|---|---:|---|---:|
| Gioco | 100% | 4 | < 1 s |
| Adult minimale | 75,3% | 3 → 5 | 1 s |
| Sonar | 83,3% | 3 → 7 → 26 | 4 s |
| Titanic | 79,0% | 3 → 9 → 32 | 3 s |
| Letter | 97,9% | 3 → 12 → 37 → 91 → 614 | circa 2 minuti |

Gioco **senza target** dà un solo livello di 4 concetti in otto tabelle su
otto. Due famiglie lontane con gli stessi tre gruppi interni danno 2 concetti
generali e 3 comportamenti, ciascuno con due genitori: un albero ne darebbe 6.

Limiti misurati:

- **Letter.** I concetti sono cercati descrivendo ogni attributo
  separatamente, così la ricerca resta lineare nel numero di attributi. A
  ricerca finita, il livello di lavoro può descrivere insieme gli attributi che
  variano insieme; lo fa solo se le osservazioni non viste sono lette meglio.
  Su Letter accade sempre e l'accuratezza passa da 93,6% a 97,9%, oltre il
  **95,78%** storico. Richiede attributi numerici senza valori mancanti.
- **Titanic.** Su venti split il modello è a 78,1%, quattro o cinque punti
  sotto i metodi ad alberi. Su Sonar, sugli stessi venti split, è a 85,5% ed è
  il migliore fra quelli provati: tre soli split non bastavano a dirlo.
- La CP a coppie preferisce concetti grandi: gruppi adiacenti lungo una
  direzione compaiono prima uniti e si separano al livello successivo.
- Con poche osservazioni il numero di livelli cambia da una prova all'altra.
- L'addestramento fa due ricerche invece di una: su Letter circa due minuti
  contro uno.
- **Linguaggio.** Sostituendo l'attenzione di un modello preaddestrato in tutti
  i livelli, senza addestrare nulla, l'errore di previsione è 2,76 con
  l'attenzione completa, 3,23 tenendo esatti i primi token e gli ultimi 124, e
  3,01 con 64 token esatti più 64 concetti. I concetti recuperano parte del
  divario, non lo chiudono, e lo strato è ancora più lento dell'attenzione
  completa. Leggere il contesto attraverso due livelli con comportamenti
  condivisi, a parità di memoria, è risultato **peggiore** dei concetti su un
  solo livello: nel linguaggio il modello usa i concetti del prefisso e, sotto,
  la finestra esatta.

Le formule spiegano il comportamento e i suoi limiti; i risultati sui dati
stabiliscono quanto quel comportamento sia utile nel problema concreto.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
