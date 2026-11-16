"""The concept graph step by step: what happens to a table and to a text, with the formula of every step."""
from dash import dcc, html

TEXT = r"""
## Il modello, passo per passo

Questa pagina segue i dati dall'ingresso al risultato e dà la formula di ogni
passo. Prima una tabella, poi un testo: il modello è lo stesso, e alla fine una
tabella mette i due percorsi uno accanto all'altro.

**L'idea in tre righe.** Un *concetto* è un riassunto di più osservazioni fatto
solo di somme. *Riconoscere* un'osservazione vuol dire pesare i concetti per
quanto le somigliano. *Generalizzare* vuol dire sommare due concetti, scegliendo
la coppia che costa meno a chi poi li legge. Un insieme di concetti è buono
quanto permette di leggere ciò che serve da osservazioni che non ha visto.

---

## Parte A · Una tabella

### Passo 1 · La tabella diventa numeri

Ogni colonna numerica è portata a media zero e deviazione uno, con la media e la
deviazione delle righe di addestramento. Ogni colonna categorica diventa un
gruppo di caselle, una per valore. Un valore mancante non viene inventato: è
segnato come non osservato e semplicemente non partecipa ai calcoli.

Ogni riga ha una parte con cui la si **riconosce**, $z$ (gli attributi noti), e
una parte che si **legge**, $y$ (il target, se c'è; altrimenti ogni attributo a
turno). Il target non entra mai nel riconoscere.

### Passo 2 · Ogni riga è un concetto di un solo membro

Un concetto $i$ conserva soltanto somme sui suoi membri:

$$
C_i=\sum 1,\qquad U_i=\sum z,\qquad Q_i=\sum z^2,\qquad N_i=\text{conteggi dei valori categorici}.
$$

Da qui si ricavano centro e dispersione:

$$
g_i=\frac{U_i}{C_i},\qquad s_i^2=\frac{Q_i}{C_i}-g_i^2 .
$$

Aggiungere una riga, toglierla o unire due concetti è sempre una somma. Le somme
crescono con il numero di attributi, non con le coppie di attributi.

### Passo 3 · Quanto costa unire due concetti

Leggere un concetto al posto dei suoi membri introduce un errore limitato da due
quantità: quante letture riceve e quanto i membri distano dal centro. Il
**raggio** è la dispersione nella parte riconosciuta, sulla scala del livello
sopra ($\sigma_j^2$, che alla radice vale uno):

$$
\rho_i=\sqrt{\frac1m\sum_j\frac{s_{ij}^2}{\sigma_j^2}}
$$

(per un attributo categorico al posto della varianza c'è l'impurità
$1-\sum_v p_{iv}^2$). Il **costo** di unire $i$ e $l$ è la perdita di chi legge:

$$
\Delta_{il}=(a_i+a_l)\,\rho_{i\cup l}-a_i\,\rho_i-a_l\,\rho_l ,
$$

dove $a$ sono le letture ricevute. In una tabella ogni riga legge i concetti a
cui appartiene, quindi $a_i=C_i$. Il costo non è mai negativo e vale zero solo
fra due copie dello stesso concetto.

### Passo 4 · Una passata a blocchi: i concetti specifici

Le righe entrano una volta sola, in blocchi di $B$ righe. Due blocchi uguali,
appena completi, diventano un blocco che copre il doppio delle righe:

$$
\mathcal B_j=\operatorname{gen}_B\big(\mathcal B_{j-1}\cup\mathcal B'_{j-1}\big),
$$

dove $\operatorname{gen}_B$ unisce la coppia di $\Delta$ minimo finché restano
$B$ concetti: da $2B$ a $B$ sono $B$ unioni, un numero che dipende dalle
dimensioni e non dai dati. Dopo l'ultima riga restano al massimo
$B\,(\lfloor\log_2(n/B)\rfloor+1)$ concetti, ciascuno somma di righe vere: sono
i **concetti specifici**.

$B$ è un quarto dei concetti specifici ammessi, fra 32 e 128. Una tabella fino a
256 righe salta questo passo: ogni riga resta un concetto specifico.

### Passo 5 · Riconoscere: a quali concetti appartiene una riga

Il peso del concetto $i$ per la riga $o$ è la sua massa per la somiglianza
attesa fra $o$ e i suoi membri:

$$
\log\big(C_i\,\mathbb E[\kappa]\big)=\log C_i-\frac12\sum_{j\in O}\left[\frac{(o_j-g_{ij})^2}{v_{ij}}+\log v_{ij}\right]
+\sum_{j\in O}\log p_{i,o_j},\qquad
T_i(o)=\frac{C_i\,\mathbb E_i[\kappa]}{\sum_lC_l\,\mathbb E_l[\kappa]} ,
$$

con $O$ gli attributi osservati. Un concetto con pochi membri non può essere
molto preciso: la sua dispersione è avvicinata a quella del livello sopra con il
peso di un'osservazione, $v_{ij}=(C_i\,s_{ij}^2+1)/(C_i+1)$.

### Passo 6 · Dai concetti specifici a uno solo

Si ripete finché resta un solo concetto:

1. si uniscono le coppie di $\Delta$ minimo, un ventesimo dei concetti alla volta;
2. ogni riga viene riassegnata ai concetti con i pesi $T_i$ del passo 5 e le
   somme vengono ricalcolate (tre volte).

La sequenza è un percorso dal particolare al generale: ogni suo punto è un
possibile insieme di concetti.

### Passo 7 · Giudicare ogni punto del percorso

Per ogni riga si toglie il suo contributo dalle somme e si chiede ai concetti
rimasti di leggerla:

$$
C_i^{(-t)}=C_i-T_{it},\quad U_i^{(-t)}=U_i-T_{it}z_t,\quad Q_i^{(-t)}=Q_i-T_{it}z_t^2,
\qquad J=\frac1n\sum_t\log p^{(-t)}(y_t\mid z_t).
$$

Un concetto fatto di una sola riga non può sostenere sé stesso. È il numero
«errore sulle righe non viste» che la pagina mostra durante l'addestramento
(è $-J$: meno è meglio).

### Passo 8 · Scegliere il livello di lavoro e i livelli sopra

Il **livello di lavoro**, quello che fa le previsioni, è il più generale fra
quelli il cui giudizio è entro un errore standard dal migliore:

$$
K^*=\min\{K:\;J_K\ge\max_{K'}J_{K'}-\mathrm{se}\}.
$$

I **livelli sopra** sono i tagli del percorso che rendono massima la CP a
coppie, cioè quanto due righe dello stesso concetto si somigliano più della
coppia media del livello sopra:

$$
CP^\circ=\frac1{n(n-1)}\sum_iC_i(C_i-1)\Bigl(1-\frac1m\sum_j\frac{\hat s^2_{ij}}{\bar\sigma'^2_{ij}}\Bigr).
$$

Un livello resta solo se, come descrizione della tabella, legge le righe non
viste meglio del livello sopra.

La ricerca viene fatta due volte. La seconda guarda la tabella *attraverso* i
concetti generali: a ogni riga si toglie il centro del suo concetto e la si
misura sulla sua dispersione,

$$
z'_t=z_t-\sum_pT_{pt}\,g_p,\qquad \sigma'^2_{tj}=\sum_pT_{pt}\,s^2_{pj},
$$

così ciò che si distingue solo dentro un concetto generale diventa visibile, e
un comportamento che compare in più concetti generali diventa **un solo nodo
con più genitori**. Due livelli consecutivi sono uniti dalla matrice
$P_{pi}=\Pr(\text{concetto }p\text{ sopra}\mid\text{concetto }i)$: nel grafo
dell'app sono le frecce fra le righe.

Infine, se il giudizio lo conferma, il livello di lavoro descrive insieme gli
attributi numerici che variano insieme (la loro covarianza, ancora una somma).

### Passo 9 · Prevedere

**Classe.** Ogni concetto ha una distribuzione sulle classi e la previsione è
la loro miscela: $p(y\mid x)=\sum_iT_i(x)\,p_i(y)$.

**Quantità.** Il target è un attributo numerico letto ma mai usato per
riconoscere: $\hat y(x)=\sum_iT_i(x)\,g_{i,y}$.

**Senza target.** Si legge ogni attributo dagli altri: lo stesso modello
completa qualunque casella mancante.

### Che cosa mostra il grafo

**Archi.** Due concetti sono collegati quando qualche riga esita fra loro:
$E_{il}=\frac1n\sum_tT_{it}T_{lt}$ sui due concetti a cui la riga appartiene di
più (le linee tratteggiate).

**Rilevanza.** Quanta variabilità di un attributo i concetti spiegano:

$$
r_{ij}=1-\frac{s_{ij}^2}{\sigma_j^2},\qquad \eta_j^2=\sum_i\frac{C_i}{n}\,r_{ij},\qquad CP=\sqrt{\frac1m\sum_j\eta_j^2}.
$$

La CP sta fra zero e uno ed è riportata, non ottimizzata: crescerebbe sempre
aumentando i concetti.

### Una tabella più grande della memoria

Oltre 20.000 righe la tabella viene letta tre volte, un pezzo alla volta:

1. **scala**: media e deviazione di ogni attributo, e 20.000 righe estratte a
   caso con la stessa probabilità;
2. **blocchi** (passo 4) su tutte le righe; pezzi diversi possono essere letti
   da processi diversi, perché le somme si sommano;
3. **livelli** (passi 5–8) sulle righe estratte, a partire dai concetti dei
   blocchi; poi ogni riga della tabella viene aggiunta ai concetti del livello
   di lavoro, così le loro somme sono sull'intera tabella.

Tempo proporzionale alle righe; memoria: un pezzo di tabella più i blocchi.

---

## Parte B · Un testo

Il modello linguistico è una pila di strati. In ogni strato ogni token deve
leggere i token che lo precedono: è qui che entrano i concetti.

### Passo 1 · Ogni token produce tre vettori

Una **domanda** $q$ (che cosa cerco), una **chiave** $k$ (come mi riconoscono)
e un **valore** $v$ (che cosa porto). Nel vocabolario della parte A: la chiave
è $z$, il valore è $y$, e la domanda è una riga di cui si cerca il valore.

### Passo 2 · Gli ultimi token restano esatti

Gli ultimi $w$ token (la finestra, 16 sul processore) sono letti uno per uno:
ciascuno è un concetto di un solo membro.

### Passo 3 · Quanto è stato letto un token

Finché un token è nella finestra dei token successivi, riceve da ciascuno una
quota di attenzione. La loro somma è la sua lettura:

$$
a_p=\sum_{t=p}^{p+w-1}\frac{e^{\lambda\,q_t\cdot k_p}}{\sum_{u=t-w+1}^{t}e^{\lambda\,q_t\cdot k_u}} .
$$

È nota nel momento in cui il token esce dalla finestra. Un concetto somma le
letture dei suoi membri; chi non è mai stato letto pesa un millesimo della
media, così le copie si uniscono comunque per prime.

### Passo 4 · I token usciti dalla finestra entrano nei blocchi

Sono gli stessi blocchi della parte A, passo 4, con i token al posto delle
righe: un blocco di livello $j$ copre $2^j$ token consecutivi e ha al massimo
$B$ concetti (16 sul processore). Finché ne ha meno resta esatto; altrimenti
$\operatorname{gen}_B$ unisce le coppie di $\Delta$ minimo, con le letture del
passo 3 e il raggio

$$
\rho_i^2=\frac1d\Big(\frac{S_i}{C_i}-\frac{\lVert U_i\rVert^2}{C_i^2}\Big),\qquad S_i=\sum\lVert k\rVert^2 .
$$

Così due token uguali si uniscono gratis e un token che è stato letto molto
resta da solo. Le chiavi sono sommate prima della rotazione di posizione; il
centro è riportato alla posizione media dei suoi membri quando viene letto.

### Passo 5 · Che cosa legge un token

Il token in posizione $t$ ha $n=t-w+1$ token più vecchi della finestra. Legge
la finestra esatta e i blocchi della scrittura binaria di $n$: se
$n=\sum_jb_j2^j$, un blocco di livello $j$ per ogni $b_j=1$, i più grandi sui
token più antichi. Ogni token vecchio è coperto una volta sola. Vicino legge
blocchi piccoli e precisi, lontano blocchi grandi e generali:

$$
\#\{\text{vettori letti}\}\;\le\;w+B\,(\lfloor\log_2 n\rfloor+1).
$$

### Passo 6 · La lettura

È la formula del passo 5 della parte A con la somiglianza $e^{\lambda q\cdot k}$:

$$
\mathrm{out}(q)=\frac{\sum_{\tau\in W}e^{\lambda q\cdot k_\tau}\,v_\tau+\sum_iC_i\,e^{\lambda q\cdot g_i}\,\bar v_i}
{\sum_{\tau\in W}e^{\lambda q\cdot k_\tau}+\sum_iC_i\,e^{\lambda q\cdot g_i}} ,
$$

dove $W$ è la finestra e $i$ scorre sui concetti dei blocchi letti. Un concetto
pesa quanto i token che riassume. Se $B\ge n$ nessuna unione avviene e questa
**è** l'attenzione ordinaria.

### Passo 7 · Mentre impara e mentre scrive

I blocchi dipendono solo dalla posizione, quindi sono gli stessi per ogni testo
della stessa lunghezza.

- **Mentre impara** un livello è lo stesso calcolo per tutti i blocchi e per
  tutti i token del pezzo insieme: $\lfloor\log_2 T\rfloor$ passi, ciascuno
  $O(T\,B\,d)$, contro $O(T^2d)$ dell'attenzione ordinaria.
- **Mentre scrive** i blocchi sono un contatore in base due: i token che
  escono dalla finestra riempiono un blocco, e due blocchi uguali diventano il
  loro genitore con la stessa $\operatorname{gen}_B$. La lettura è quella
  dell'addestramento: esattamente in doppia precisione; in precisione singola
  due unioni di costo quasi uguale possono scambiarsi e la lettura cambia di
  poco.

Nessun token legge il futuro: un token è letto solo da somme che non lo
contengono. È lo stesso principio del passo 7 della parte A.

---

## Parte C · Tabella e testo a confronto

| | Tabella | Testo |
|---|---|---|
| osservazione | una riga: attributi e target | un token: chiave e valore |
| concetto | somme dei membri | somme dei membri |
| riconoscere | massa × somiglianza attesa | massa × somiglianza attesa |
| costo di un'unione | letture × raggio | letture × raggio |
| letture ricevute | la massa | attenzione ricevuta nella finestra |
| concetti specifici | i blocchi rimasti dopo l'ultima riga | i blocchi della scrittura binaria dell'età |
| riassegnazione dei membri | sì | no: il passato non si rilegge |
| quanti concetti | scelti dal giudizio | $B$ per blocco, fissati dalla memoria |
| livelli sopra | scelti dalla CP a coppie | i blocchi stessi, per età |

---

## Parte D · Che cosa è stato misurato

Le verifiche automatiche controllano le formule, che tabella e testo usino lo
stesso costo, l'esclusione del target, che nessun token legga il futuro, che con
abbastanza concetti lo strato sia l'attenzione ordinaria e che la memoria non
superi il limite del passo 5. Non dimostrano da sole che il modello generalizzi:
per quello servono le misure.

**Tabelle** (divisioni dei test di qualità, accuratezza su righe non viste):

| Dataset | Accuratezza | Concetti | Tempo |
|---|---:|---:|---:|
| Iris | 97,7% | 3 | 0,1 s |
| Wine | 97,8% | 3 | 0,5 s |
| Breast cancer | 95,2% | 20 | 4 s |
| Sonar | 90,0% | 92 | 4 s |
| Vehicle | 79,6% | 28 | 2 s |
| Vote | 96,1% | 7 | 1 s |
| Titanic | 81,2% | 25 | 1 s |
| credit-g | 70,6% | 35 | 10 s |
| Letter | 97,6% | 389 | 91 s |

Senza target, Gioco dà 4 concetti in otto tabelle su otto.

**Tabelle grandi** (sintetica: 10 numeri, 3 categorie, valori mancanti, 12
classi): un milione di righe in 211 secondi e due milioni in 370, con due
processi; la memoria resta intorno a 1 GB. Un miliardo di righe non è stato
misurato.

**Testo** (modellini da 64 dimensioni addestrati da zero, finestra 16, 16
concetti per blocco):

| Prova | Attenzione ordinaria | Solo la finestra | Concetti |
|---|---:|---:|---:|
| Memoria, risposta a 49 token: giuste | 100% | 5,5% | 100% |
| Memoria lontana, fino a 177 token: giuste | 100% | — | 99% |
| Testo, bit per carattere (meno è meglio) | 2,518 | 2,451 | 2,522 |

**Limiti noti.**

- Sui pezzi di testo corti l'attenzione ordinaria è ancora più veloce; i
  concetti la superano sui contesti lunghi (numeri nella scheda Informatica).
- Le prove sul testo sono su modellini e su un solo seme utile: dicono che il
  meccanismo funziona, non che un modello grande converserà bene.
- Leggere tutte le righe di una tabella grande, invece delle sole 20.000
  estratte, non ha aumentato l'accuratezza nella prova sintetica: garantisce
  che le somme siano sull'intera tabella.
- Su tabelle con poche righe e molti attributi (Sonar) il livello di lavoro ha
  molti concetti: accurato, ma meno leggibile.
- La CP a coppie preferisce concetti grandi: gruppi adiacenti compaiono prima
  uniti e si separano al livello successivo.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
