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

Il testo è una tabella. Non c'è una rete neurale: ci sono gli stessi concetti
della parte A, con le stesse operazioni.

### Passo 1 · Il testo diventa token, i token diventano righe

Il tokenizer divide il testo in token (parole, pezzi di parola, segni), ciascuno
con un numero fra $0$ e $V-1$. Ogni posizione $t$ del testo è una **riga**:

- i suoi **attributi** sono i token che la precedono, $x_{t-1},x_{t-2},\dots$;
- il suo **target** è il token in quella posizione, $x_t$.

Prevedere il testo è prevedere il target di una riga dai suoi attributi.

### Passo 2 · I concetti per contesto

Un concetto di livello $j$ raccoglie tutte le righe con gli **stessi ultimi $j$
token**: il loro contesto $c=(x_{t-j},\dots,x_{t-1})$. Il livello 0 ha un solo
concetto, tutte le righe. Il concetto conserva solo somme:

$$
C_c=\#\{t:\ \text{contesto }c\},\qquad
N_c(y)=\#\{t:\ \text{contesto }c,\ x_t=y\},\qquad
D_c=\#\{y:\ N_c(y)>0\}.
$$

I livelli sono annidati come nella parte A: il concetto $c$ di livello $j$ è
contenuto nel concetto del suo contesto più corto (senza il token più lontano),
di livello $j-1$, che è il suo **livello sopra**. Le somme di un concetto sono
già comprese in quelle del livello sopra.

### Passo 3 · Leggere: il livello sopra come riferimento

Nella parte A un concetto con pochi membri non può essere molto preciso e viene
avvicinato al livello sopra (passo 5). Qui è la stessa regola, sulla
distribuzione del token che segue:

$$
p_j(y)=\frac{N_{c_j}(y)+w_j\,p_{j-1}(y)}{C_{c_j}+w_j},\qquad w_j=a_j\,D_{c_j},
\qquad p_{-1}(y)=\frac1V .
$$

Si parte dal livello 0 e si scende finché il contesto è stato visto; un contesto
mai visto lascia la lettura com'è. Il peso $w_j$ cresce con $D$: un concetto
seguito da molti token diversi è un concetto largo, e si fida di più del livello
sopra; uno seguito sempre dallo stesso token è stretto, e si fida di sé.

### Passo 4 · I concetti del testo in corso

Gli stessi concetti si costruiscono anche sul testo che si sta leggendo, usando
**solo le posizioni precedenti** di quel testo:

$$
p^{\text{testo}}_j(y)=\frac{N^{<t}_{c_j}(y)+b_j\,D^{<t}_{c_j}\,p^{\text{testo}}_{j-1}(y)}{C^{<t}_{c_j}+b_j\,D^{<t}_{c_j}},
\qquad p^{\text{testo}}_0=p_{\text{finale}}\ \text{del passo 3}.
$$

Così ciò che il testo ha già detto si può ripetere, a qualunque distanza: un nome
apparso all'inizio, la risposta a una domanda. Nessuna posizione legge sé stessa
o il futuro.

### Passo 5 · Giudicare i pesi

I pesi $a_j$ e $b_j$ sono scelti dal giudizio, come i livelli della parte A: si
prendono i valori (fra $2^{-5}$ e $2^{8}$) che leggono meglio pezzi di testo che
il modello **non ha contato**,

$$
(a,b)=\arg\max\ \frac1n\sum_t\log p(x_t\mid x_{<t}),
$$

un peso alla volta, due giri. Metà dei pezzi di validazione sceglie i pesi,
l'altra metà misura l'errore: i due insiemi non si toccano.

### Passo 6 · Imparare è contare, a blocchi

Un pezzo di testo diventa le sue tabelle di somme. Due tabelle si sommano
concetto per concetto: $C$, $N$ si sommano, $D$ si ricalcola dai $N$. I pezzi si
uniscono come i blocchi della parte A, passo 4: due tabelle che coprono lo
stesso numero di pezzi diventano una tabella che ne copre il doppio. Ogni
posizione viene sommata circa $\log_2(\text{pezzi})$ volte, e alla fine le somme
sono quelle dell'intero testo. Un testo successivo della coda aggiunge le sue
somme a quelle del precedente: nulla viene dimenticato.

### Passo 7 · Restare nella memoria: generalizzare dove chi legge perde meno

Il numero di contesti diversi cresce con il testo. Oltre la memoria disponibile
si generalizzano nel livello sopra i concetti che costano meno a chi legge. Il
costo di un concetto $c$ è quanto peggio i suoi membri sarebbero letti dal
livello sopra $\bar c$:

$$
\Delta_c=\sum_yN_c(y)\,\log\frac{N_c(y)/C_c}{p_{\bar c}(y)} .
$$

È la stessa idea del costo della parte A (letture × perdita), con le letture
uguali ai membri. Un concetto che dice la stessa cosa del suo livello sopra
costa zero e va per primo; le sue somme sono già nel livello sopra, quindi non
si perde nulla oltre la distinzione. Il livello 0 non si generalizza mai.

### Passo 8 · Scrivere

Per scrivere, si calcola $p(\cdot\mid x_{<t})$ su tutto il vocabolario con i
passi 3 e 4, si sceglie un token (il più probabile, oppure estratto con
temperatura $T$: $p^{1/T}$ rinormalizzata), lo si aggiunge e si ripete. Il
testo scritto entra subito nei concetti del testo in corso.

---

## Parte C · Tabella e testo a confronto

| | Tabella | Testo |
|---|---|---|
| riga | una riga della tabella | una posizione del testo |
| attributi | le colonne note | i token precedenti |
| target | la colonna da prevedere | il token in quella posizione |
| concetto | somme dei membri | somme dei membri |
| livello sopra | il concetto più generale | il contesto più corto |
| lettura | massa × somiglianza, dispersione avvicinata al livello sopra | conteggi avvicinati al livello sopra |
| costo di un'unione | letture × raggio | letture × perdita di log-verosimiglianza |
| concetti specifici | blocchi di righe uniti a due a due | pezzi di testo uniti a due a due |
| giudizio | righe non viste | pezzi di testo non contati |
| quanti concetti | scelti dal giudizio | tutti quelli visti, fino alla memoria; poi generalizzati |
| memoria a lungo raggio | — | i concetti del testo in corso |

## Parte D · Che cosa è stato misurato

Le verifiche automatiche controllano le formule della tabella, l'esclusione del
target, e per il testo: che la lettura sia esattamente quella del passo 3 e 4
scritta posizione per posizione, che i concetti del testo in corso usino solo le
posizioni precedenti, che pezzi sommati in qualunque ordine diano le somme del
testo intero, che il limite di memoria generalizzi prima i concetti che non
aggiungono nulla, e che salvare e ricaricare non cambi la lettura. Non dimostrano
da sole che il modello generalizzi: per quello servono le misure.

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

**Testo, caratteri.** Stessi dati e stesso protocollo della prova in piccolo
(wikitext, pezzi di 256 caratteri, errore sulle seconde metà dei pezzi):

| | Rete con attenzione | Rete solo finestra | Modello a concetti |
|---|---:|---:|---:|
| Testo, bit per carattere (meno è meglio) | 2,518 | 2,451 | 2,039 |
| Memoria, risposta a 49 token | 100% | 5,5% | 100% |
| Memoria lontana, fino a 177 token | 100% | — | 100% |
| Tempo, un thread | 219 s di addestramento | 213 s | 73 s di conteggio |

Con la memoria dei concetti limitata a 18 MB (il 79% dei concetti generalizzato
nel livello sopra) l'errore passa da 2,039 a 2,079 bit per carattere.

**Testo, token.** Stesso testo, stesso tokenizer BPE da 8.000 token, 272.000
token per imparare, 64 pezzi di 256 token mai visti:

| | Rete micro nata da zero (12 milioni di pesi) | Modello a concetti |
|---|---:|---:|
| Errore, nat per token (meno è meglio) | 5,652 al suo meglio, poi risale a 5,921 | 5,452 |
| Tempo | 433 s per arrivare al suo meglio (1,4 milioni di token letti) | 20 s (ogni token contato una volta) |

La rete legge il testo più volte e dopo circa cinque letture comincia a impararlo
a memoria; il modello a concetti lo conta una volta.

**Limiti noti.**

- **Contesti simili.** Due contesti diversi non si aiutano: «il gatto dorme» e
  «il cane dorme» non condividono nulla oltre il livello sopra comune. Una rete
  neurale impara che gatto e cane si somigliano; il modello a concetti, per ora,
  no. Unire i token che sono seguiti da token simili è il prossimo passo.
- Le misure sul testo sono su testi di pochi MB: dicono che il modello funziona
  e come si confronta con una rete della stessa misura, non che conversi bene.
- Leggere tutte le righe di una tabella grande, invece delle sole 20.000
  estratte, non ha aumentato l'accuratezza nella prova sintetica: garantisce
  che le somme siano sull'intera tabella.
- Su tabelle con poche righe e molti attributi (Sonar) il livello di lavoro ha
  molti concetti: accurato, ma meno leggibile.
- La CP a coppie preferisce concetti grandi: gruppi adiacenti compaiono prima
  uniti e si separano al livello successivo.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
