"""The concept model step by step: one story for the rows of a table and the positions of a text."""
from dash import dcc, html

TEXT = r"""
## Il modello, passo per passo

C'è un solo modello. Le sue osservazioni sono **righe**: una riga di una tabella,
oppure una posizione di un testo. Questa pagina le segue dall'ingresso alla
previsione e dà la formula di ogni passo. Ogni passo vale per entrambe; dove una
riga di tabella e una posizione di testo si comportano diversamente, il passo
dice come e perché.

**L'idea in tre righe.** Un *concetto* è un riassunto di più righe fatto solo di
somme. *Riconoscere* una riga vuol dire pesare i concetti per quanto le
somigliano. *Generalizzare* vuol dire sommare un concetto con un altro o con il
suo livello sopra, dove chi poi legge perde meno. Un insieme di concetti è buono
quanto permette di leggere ciò che serve da righe che non ha visto.

---

### Passo 1 · I dati diventano righe

**Una tabella.** Ogni colonna numerica è portata a media zero e deviazione uno,
con la media e la deviazione delle righe di addestramento. Ogni colonna
categorica diventa un gruppo di caselle, una per valore. Un valore mancante non
viene inventato: è segnato come non osservato e non partecipa ai calcoli.

**Un testo.** Il tokenizer lo divide in token (parole, pezzi di parola, segni),
ciascuno con un numero fra $0$ e $V-1$. **Il testo è una tabella**: ogni
posizione $t$ è una riga, i token che la precedono $x_{t-1},x_{t-2},\dots$ sono i
suoi attributi (categorie con $V$ valori) e il token in quella posizione, $x_t$,
è la colonna da prevedere.

In entrambi i casi una riga ha una parte con cui la si **riconosce**, $z$ (gli
attributi noti), e una parte che si **legge**, $y$ (il target; in una tabella
senza target, ogni attributo a turno). Il target non entra mai nel riconoscere.

### Passo 2 · Un concetto è fatto di somme

Un concetto $i$ conserva soltanto somme sui suoi membri:

$$
C_i=\sum 1,\qquad U_i=\sum z,\qquad Q_i=\sum z^2,\qquad N_i(v)=\#\{\text{membri con il valore }v\},\qquad D_i=\#\{v:\ N_i(v)>0\}.
$$

Da qui centro e dispersione degli attributi numerici: $g_i=U_i/C_i$,
$s_i^2=Q_i/C_i-g_i^2$. Aggiungere una riga, toglierla o unire due concetti è
sempre una somma; le somme crescono con il numero di attributi, non con le
coppie di attributi.

Nel testo un concetto di livello $j$ raccoglie le posizioni con gli **stessi
ultimi $j$ token**, il contesto $c=(x_{t-j},\dots,x_{t-1})$: ne servono solo
$C_c$, i token che lo seguono $N_c(y)$ e quanti sono diversi $D_c$, perché il
contesto è uguale per tutti i membri. Il livello 0 è un solo concetto, tutte le
righe.

I concetti sono **annidati in livelli**: ogni concetto è contenuto in uno più
generale, il suo **livello sopra**, e le sue somme sono già comprese in quelle
del livello sopra. In una tabella i livelli si trovano cercandoli (passo 8); nel
testo sono dati dalla lunghezza del contesto: il livello sopra di un contesto è
lo stesso contesto senza il token più lontano.

### Passo 3 · Riconoscere: a quali concetti appartiene una riga

**Righe di una tabella.** I numeri possono essere vicini o lontani, quindi una
riga appartiene un po' a più concetti. Il peso del concetto $i$ per la riga $o$ è
la sua massa per la somiglianza attesa fra $o$ e i suoi membri:

$$
\log\big(C_i\,\mathbb E[\kappa]\big)=\log C_i-\frac12\sum_{j\in O}\left[\frac{(o_j-g_{ij})^2}{v_{ij}}+\log v_{ij}\right]
+\sum_{j\in O}\log p_{i,o_j},\qquad
T_i(o)=\frac{C_i\,\mathbb E_i[\kappa]}{\sum_lC_l\,\mathbb E_l[\kappa]} ,
$$

con $O$ gli attributi osservati e $v_{ij}$, $p_i$ letti come al passo 4.

**Posizioni di un testo.** Due token o sono uguali o non lo sono, e due contesti
pure: una posizione appartiene a **esattamente un concetto per livello**, quello
del suo contesto, e i concetti si trovano contando, senza cercarli. È la stessa
regola nel caso limite: la somiglianza fra due contesti diversi è zero. Usata
sul testo, la ricerca per somiglianza della tabella trova troppi pochi concetti
e mescolati (3,47 bit per carattere, contro 2,04 contando i contesti).

**Il testo in corso.** Gli stessi concetti si contano anche sul testo che si sta
leggendo, usando **solo le posizioni precedenti**: $C^{<t}_c$, $N^{<t}_c(y)$,
$D^{<t}_c$. Così ciò che il testo ha già detto si può ripetere a qualunque
distanza (un nome apparso all'inizio, la risposta a una domanda). Nessuna
posizione legge sé stessa o il futuro.

### Passo 4 · Leggere: il livello sopra come riferimento

Un concetto con pochi membri non può essere molto preciso: le sue frequenze e la
sua dispersione sono avvicinate a quelle del livello sopra,

$$
p_i(v)=\frac{N_i(v)+\kappa D_i\,p_{\text{sopra}}(v)}{C_i+\kappa D_i},\qquad
v_{ij}=\frac{C_i\,s_{ij}^2+\kappa\,\sigma_j^2}{C_i+\kappa} .
$$

Un concetto i cui membri hanno molti valori diversi ($D$ grande) è largo e si
appoggia di più al livello sopra; uno con un solo valore è stretto e si fida di
sé. Un concetto senza membri lascia la lettura com'è. È **una sola funzione** nel
codice, `categorical_reading`, per una categoria di una tabella e per il token
che segue un contesto.

Nel testo la lettura scende i livelli dal contesto vuoto al più lungo visto, ogni
livello con il precedente come riferimento, poi continua con i concetti del
testo in corso:

$$
p_j(y)=\frac{N_{c_j}(y)+\kappa_jD_{c_j}\,p_{j-1}(y)}{C_{c_j}+\kappa_jD_{c_j}},\quad p_{-1}(y)=\frac1V;
\qquad
p^{\text{testo}}_j(y)=\frac{N^{<t}_{c_j}(y)+b_jD^{<t}_{c_j}\,p^{\text{testo}}_{j-1}(y)}{C^{<t}_{c_j}+b_jD^{<t}_{c_j}},\quad p^{\text{testo}}_0=p_{6}.
$$

La tabella sceglie un $\kappa$; il testo un $\kappa_j$ per livello e un $b_j$ per
livello del testo in corso. Mentre la tabella cerca i suoi concetti $\kappa=1$ e
$D$ non conta.

### Passo 5 · Giudicare su ciò che non si è visto

Ogni scelta del modello è presa nello stesso modo: si tiene ciò che legge meglio
righe che non ha visto.

**Una tabella.** Per ogni riga si toglie il suo contributo dalle somme e si
chiede ai concetti rimasti di leggerla:

$$
C_i^{(-t)}=C_i-T_{it},\quad U_i^{(-t)}=U_i-T_{it}z_t,\quad Q_i^{(-t)}=Q_i-T_{it}z_t^2,
\qquad J=\frac1n\sum_t\log p^{(-t)}(y_t\mid z_t).
$$

Un concetto fatto di una sola riga non può sostenere sé stesso.

**Un testo.** Il testo di validazione non viene mai contato: metà dei suoi pezzi
sceglie i pesi, l'altra metà misura l'errore,

$$
(\kappa,b)=\arg\max\ \frac1n\sum_t\log p(x_t\mid x_{<t}),
$$

un peso alla volta fra $2^{-5},\dots,2^{8}$, due giri. $\kappa$ della tabella è
scelto fra gli stessi valori. Il numero «errore su righe (o testo) mai viste» che
le pagine mostrano è $-J$: meno è meglio.

### Passo 6 · Contare a blocchi

Le righe entrano una volta sola, in blocchi. Due blocchi uguali, appena completi,
diventano un blocco che copre il doppio delle righe:

$$
\mathcal B_j=\operatorname{gen}_B\big(\mathcal B_{j-1}\cup\mathcal B'_{j-1}\big).
$$

**Una tabella.** $\operatorname{gen}_B$ unisce la coppia di costo minimo (passo 7)
finché restano $B$ concetti: da $2B$ a $B$ sono $B$ unioni, un numero che dipende
dalle dimensioni e non dai dati. Dopo l'ultima riga restano al massimo
$B\,(\lfloor\log_2(n/B)\rfloor+1)$ concetti, ciascuno somma di righe vere: i
**concetti specifici**. $B$ è un quarto dei concetti specifici ammessi, fra 32 e
128; fino a 256 righe ogni riga resta un concetto specifico.

**Un testo.** Ogni pezzo (un milione di token) diventa le sue tabelle di somme,
una per livello; due tabelle che coprono lo stesso numero di pezzi si sommano
concetto per concetto. Ogni posizione è sommata circa $\log_2(\text{pezzi})$
volte, e alla fine le somme sono quelle dell'intero testo. Un testo successivo
della coda aggiunge le sue somme a quelle del precedente: nulla viene
dimenticato.

**Una tabella più grande della memoria** si legge tre volte, un pezzo alla volta:
la scala di ogni attributo e 20.000 righe estratte a caso; i blocchi su tutte le
righe (pezzi diversi anche da processi diversi, perché le somme si sommano); i
livelli sulle righe estratte, poi ogni riga aggiunta ai concetti finali. Un
milione di righe sintetiche ha richiesto 211 secondi e due milioni 370, con due
processi e circa 1 GB di memoria.

### Passo 7 · Generalizzare dove chi legge perde meno

Generalizzare toglie una distinzione: due concetti diventano uno, o un concetto
torna nel suo livello sopra. Il costo è sempre la stessa idea: **letture ×
perdita** di chi poi legge.

**Una tabella: unire due concetti.** Il **raggio** è la dispersione nella parte
riconosciuta, sulla scala del livello sopra (per una categoria, l'impurità
$1-\sum_vp_{iv}^2$),

$$
\rho_i=\sqrt{\frac1m\sum_j\frac{s_{ij}^2}{\sigma_j^2}},\qquad
\Delta_{il}=(a_i+a_l)\,\rho_{i\cup l}-a_i\,\rho_i-a_l\,\rho_l ,
$$

con $a=C$ le letture ricevute. Il costo non è mai negativo e vale zero solo fra
due copie dello stesso concetto.

**Un testo: restare nella memoria.** I contesti diversi crescono con il testo.
Oltre il limite di memoria (uno per tutte le somme) un concetto torna nel suo
livello sopra $\bar c$ se costa poco per byte. Il costo è la log-verosimiglianza
che i suoi membri perdono, ciascuno letto **senza sé stesso** come nel giudizio:

$$
\Delta_c=\sum_yN_c(y)\,\log\frac{p^{-}_c(y)}{p_{\bar c}(y)},\qquad
p^{-}_c(y)=\frac{N_c(y)-1+\kappa D_c\,p_{\bar c}(y)}{C_c-1+\kappa D_c}.
$$

Un contesto visto una volta sola non dice nulla del suo unico membro: costa zero
e va per primo. Le sue somme restano nel livello sopra. Su wikitext-2 con 400 MB
di limite l'errore passa da 4,615 a 4,602 tenendo 49 MB di somme invece di 252.

### Passo 8 · I livelli: dal particolare al generale

**Una tabella.** Dai concetti specifici si uniscono le coppie di costo minimo, un
ventesimo dei concetti alla volta, riassegnando le righe con i pesi del passo 3
(tre volte), finché resta un solo concetto. Ogni punto del percorso è un insieme
possibile di concetti, giudicato come al passo 5. Il **livello di lavoro** è il
più generale entro un errore standard dal migliore:

$$
K^*=\min\{K:\;J_K\ge\max_{K'}J_{K'}-\mathrm{se}\}.
$$

I **livelli sopra** sono i tagli del percorso che rendono massima la CP a coppie,
quanto due righe dello stesso concetto si somigliano più della coppia media del
livello sopra,

$$
CP^\circ=\frac1{n(n-1)}\sum_iC_i(C_i-1)\Bigl(1-\frac1m\sum_j\frac{\hat s^2_{ij}}{\bar\sigma'^2_{ij}}\Bigr),
$$

e un livello resta solo se legge le righe non viste meglio del livello sopra. La
ricerca si fa due volte: la seconda guarda la tabella *attraverso* i concetti
generali,

$$
z'_t=z_t-\sum_pT_{pt}\,g_p,\qquad \sigma'^2_{tj}=\sum_pT_{pt}\,s^2_{pj},
$$

così ciò che si distingue solo dentro un concetto generale diventa visibile, e
un comportamento che compare in più concetti generali diventa **un solo nodo con
più genitori**, uniti dalla matrice $P_{pi}=\Pr(\text{concetto }p\text{
sopra}\mid\text{concetto }i)$. Se il giudizio lo conferma, il livello di lavoro
descrive insieme gli attributi numerici che variano insieme (la loro covarianza,
ancora una somma).

**Un testo.** I livelli sono le lunghezze di contesto, da 0 a 6 token, e si
tengono tutti i concetti visti finché la memoria basta (passo 7). Oltre i 3–4
token quasi non si guadagna: su wikitext-2 l'errore è 4,591 con 4 token, 4,589
con 6, 8 o 12, perché un contesto esatto di 5 o più token quasi non si ripete.

### Passo 9 · Valori che si somigliano: i concetti dei token e i percorsi

Due valori diversi di una categoria, nei passi precedenti, sono semplicemente
diversi: «gatto» e «cane» come due città in una colonna. Per il testo il modello
può vedere che si somigliano **usando sé stesso**: tratta i token come righe di
una tabella e ne cerca i concetti con i passi 3–8. La riga del token $v$ dice con
che frequenza ciascuno dei 256 token più frequenti lo precede e lo segue,

$$
r_v=\Big(\sqrt{\tfrac{\#(u\,v)}{\#(\cdot\,v)}}\Big)_{u}\ \oplus\ \Big(\sqrt{\tfrac{\#(v\,u)}{\#(v\,\cdot)}}\Big)_{u},
$$

ridotta alle sue direzioni principali (16 e 48). Ne escono livelli dal generale al
particolare, per esempio $2\to9\to20\to33\to325$ concetti su 8.000 token:
$g(v)$ il concetto generale, $f(v)$ quello particolare.

Il token si legge allora anche **lungo un percorso** di concetti,

$$
p(y)=p\big(g(y)\mid c\big)\cdot p\big(f(y)\mid g(y),c\big)\cdot p\big(y\mid f(y),c\big),
$$

ogni fattore con la lettura del passo 4 e le sue somme: un concetto conta le
posizioni con lo stesso contesto e lo stesso nodo del percorso, e nel contesto
ogni token è preceduto dal suo concetto particolare (fino a 2 token). In ogni
nodo i figli sommano a 1: il risultato è ancora una probabilità esatta su tutto
il vocabolario. Un contesto seguito da «cane» alza tutto il concetto degli
animali, quindi anche «gatto». Il percorso diretto e uno per gerarchia si
**mescolano**, $p(y)=\sum_kw_k\,p_k(y)$, con i pesi dal giudizio.

Su wikitext-2 l'errore scende da 4,589 a 4,451. Su testi grandi con la memoria
limitata peggiora (69 milioni di token, 2 GB: da 3,905 a 3,958), perché le somme
dei percorsi tolgono spazio ai contesti lunghi: per questo è una scelta del
progetto, spenta se non la si accende. Nella tabella i concetti di valori non sono
usati: servirebbero solo a categorie con molti valori, e lì non sono stati
misurati.

### Passo 10 · Prevedere e scrivere

**Una classe.** Ogni concetto ha una distribuzione sulle classi e la previsione è
la loro miscela: $p(y\mid x)=\sum_iT_i(x)\,p_i(y)$.

**Una quantità.** Il target è un attributo numerico letto ma mai usato per
riconoscere: $\hat y(x)=\sum_iT_i(x)\,g_{i,y}$.

**Senza target.** Si legge ogni attributo dagli altri: lo stesso modello
completa qualunque casella mancante.

**Il prossimo token.** È una classe con $V$ valori: la lettura del passo 4 (e del
passo 9) dà la probabilità di ogni token del vocabolario. Per scrivere se ne
sceglie uno, il più probabile oppure estratto con temperatura $T$ ($p^{1/T}$
rinormalizzata), lo si aggiunge e si ripete; il testo scritto entra subito nei
concetti del testo in corso.

Quanto bene, su righe mai viste:

| Tabella | Accuratezza | Concetti | Tempo |
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

Sul testo, contro reti neurali sugli stessi dati. Caratteri di wikitext, pezzi di
256: rete con attenzione 2,518 bit per carattere, rete solo finestra 2,451,
modello a concetti **2,039** (2,079 con il 79% dei concetti generalizzato per
stare in 18 MB). Token BPE da 8.000, gli stessi 64 pezzi di 512 token mai visti
(wikitext-2 e wikitext-103 hanno lo stesso testo di verifica), contro una rete
Llama a 4 strati con 6,2 milioni di pesi e posizione relativa, 90 minuti sullo
stesso processore:

| | Testo per imparare | Memoria | Errore | 0–32 token | 256–512 token | Tempo |
|---|---|---:|---:|---:|---:|---:|
| Rete | wikitext-2 (2 M token) | — | **4,403** | 4,903 | 4,359 | 90 min |
| Concetti | wikitext-2 | libera | 4,589 | 5,078 | 4,486 | 10 s |
| Concetti con percorsi | wikitext-2 | libera | 4,451 | 4,897 | 4,357 | 1 min |
| Rete (legge 16 M token) | wikitext-103 | — | 4,011 | 4,521 | 3,964 | 90 min |
| Concetti | 69 M token di wikitext-103 | 2 GB | **3,905** | **4,195** | **3,834** | 7 min |
| Concetti con percorsi | 69 M token di wikitext-103 | 2 GB | 3,958 | 4,275 | 3,891 | 19 min |

Sul testo grande il modello a concetti è avanti a ogni distanza; la rete
migliorava ancora e aveva letto un quarto del testo. Provati e non adottati
(guadagno nullo o piccolo): conteggi di Kneser-Ney, lettura per somiglianza sulle
posizioni passate, il token da prevedere come vettore di numeri, righe di vettori
lette dai passi 3–8, contesti fatti solo di concetti.

### Che cosa mostra il grafo

**Archi.** Due concetti sono collegati quando qualche riga esita fra loro:
$E_{il}=\frac1n\sum_tT_{it}T_{lt}$ sui due concetti a cui la riga appartiene di
più (le linee tratteggiate). Le frecce fra le righe del grafo sono la matrice
$P$ del passo 8.

**Rilevanza.** Quanta variabilità di un attributo i concetti spiegano:

$$
r_{ij}=1-\frac{s_{ij}^2}{\sigma_j^2},\qquad \eta_j^2=\sum_i\frac{C_i}{n}\,r_{ij},\qquad CP=\sqrt{\frac1m\sum_j\eta_j^2}.
$$

La CP sta fra zero e uno ed è riportata, non ottimizzata: crescerebbe sempre
aumentando i concetti.

### Che cosa garantiscono le verifiche, e i limiti

Le verifiche automatiche controllano le formule, l'esclusione del target, che la
lettura del testo (anche con i percorsi) sia esattamente quella dei passi 4 e 9
scritta posizione per posizione, che la probabilità di ogni token sia la sua
lettura e sommi a 1, che il testo in corso usi solo le posizioni precedenti, che
pezzi sommati in qualunque ordine diano le somme dell'intero testo, che il limite
di memoria generalizzi prima ciò che non aggiunge nulla, e che salvare e
ricaricare non cambi nulla. Non dimostrano da sole che il modello generalizzi:
per quello servono le misure dei passi 6–10.

- **Valori simili.** I concetti dei token sono fissati all'inizio e, con la
  memoria limitata, su testi grandi non convengono; una rete continua a
  migliorare i suoi vettori.
- **Contesto esatto** fino a 6 token; oltre, il testo in corso è ricordato a
  qualunque distanza ma senza sapere a quale.
- Le misure sul testo sono su wikitext (inglese) e su un processore di sviluppo:
  dicono come il modello si confronta con una rete, non che conversi bene, e non
  sono state ripetute sul server né sulla miscela italiano-inglese.
- Leggere tutte le righe di una tabella grande, invece delle sole 20.000
  estratte, non ha aumentato l'accuratezza nella prova sintetica: garantisce che
  le somme siano sull'intera tabella. Un miliardo di righe non è stato misurato.
- Su tabelle con poche righe e molti attributi (Sonar) il livello di lavoro ha
  molti concetti: accurato, ma meno leggibile. La CP a coppie preferisce concetti
  grandi: gruppi adiacenti compaiono prima uniti e si separano al livello
  successivo.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
