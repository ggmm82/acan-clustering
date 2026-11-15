# Concetti algebrici: la lettura del contesto senza cicli

Stato: prototipo in sviluppo (`acan/lm/algebraic_attention.py`), non integrato
nel laboratorio né nel modello nativo. Questo documento raccoglie la
matematica e le misure.

## Che cosa resta uguale

Il concetto è quello del grafo dei concetti (MODEL.md), lo stesso della parte
tabulare: un nodo che conserva solo **somme** delle osservazioni che contiene,

    C = Σ 1,   U = Σ z,   Q = Σ z²,   Y = Σ y,

e che si legge con **massa per somiglianza attesa**:

    logit = log C + E[s·q·z] ≈ log C + s·q·g + ½ s² q²·σ²,   g = U/C,  σ² = Q/C − g².

Nel linguaggio l'osservazione `z` di un token è la sua chiave (con la sua
posizione), `y` il suo valore, `q` la domanda di chi legge. Gli ultimi
`window` token sono letti esatti, come oggi.

## Che cosa cambia: a quale concetto va un'osservazione

Oggi lo strato **cerca** le unioni: quando i concetti sono troppi unisce la
coppia che costa meno a chi legge, una coppia alla volta. È una ricerca
greedy: ogni passo dipende dal precedente, quindi non si parallelizza.

Il prototipo **assegna con una formula**, nota prima di leggere:

- **Livelli per età.** I token vecchi di chi legge in posizione `t` (quelli
  prima della sua finestra, `n = t − window + 1`) sono coperti dai segmenti
  della scrittura binaria di `n` (decomposizione di Fenwick): i token vecchi
  più recenti in segmenti piccoli, i più antichi in segmenti grandi. Al
  massimo `log₂ n + 1` segmenti. I livelli del grafo tabulare (concetti
  generali sopra, particolari sotto) qui sono le dimensioni dei segmenti.
- **Contenuto.** Dentro ogni segmento una chiave va in uno di `2ʳ` gruppi
  secondo i segni di `r` proiezioni fisse (fissate dal seme, senza parametri):
  chiavi diverse restano separate.

Poiché l'assegnazione è nota, ogni statistica di ogni concetto per ogni
lettore è una **differenza di somme cumulative**. Tutto il calcolo è fatto di
somme cumulative, prodotti fra matrici e una softmax: niente ricerca e
nessun ciclo che dipenda dai dati. L'unica ripetizione è sui livelli
(`log₂ T + 1` passi fissi, come i layer di una rete).

**Memoria mentre scrive.** Le somme dei segmenti si tengono come un contatore
binario: all'arrivo di un token due segmenti uguali si sommano. Per lettore e
per layer: `window` token esatti più al massimo `(log₂ T + 1)·2ʳ` concetti.
Con un milione di token e `r = 2`: 16 + 84 vettori, contro un milione
dell'attenzione normale.

## Verifiche

- La versione veloce coincide con una versione di riferimento scritta con
  i cicli (differenze sotto 10⁻⁹), con e senza il termine di varianza.
- Nessuna posizione legge il futuro (gradiente nullo verso i token
  successivi) e i token più vecchi sono letti, attraverso i concetti.
- I segmenti di un lettore coprono esattamente i suoi token vecchi.

## Velocità (CPU, un thread, addestramento, modello di 256 dimensioni e 4 layer)

| Token per pezzo | Attenzione normale | Concetti attuali | Concetti algebrici |
|---:|---:|---:|---:|
| 256 | 2.241 | 378 | 903 |
| 1.024 | 1.747 | 468 | 975 |
| 2.048 | 933 | 384 | 899 |
| 4.096 | 400 | — | 688 |
| 8.192 | 177 | — | 551 |

Token al secondo. A 16.384 token l'attenzione normale ha esaurito la memoria
del server; i concetti algebrici no. Sui pezzi corti l'attenzione normale
resta più veloce (le sue moltiplicazioni fra matrici sono le più ottimizzate
che esistano); dai 2.048 token in su i concetti algebrici la superano.

## Prima la tabella: `acan/core/algebraic.py`

Il modello tabulare algebrico riusa tutto `ConceptModel` (codifica, valori
mancanti, somme dei concetti, lettura, giudizio sulle righe non viste,
covarianza adottata se il giudizio la conferma, previsione) e cambia solo
l'assegnazione: ogni concetto si divide in due alla propria media lungo la
propria direzione, calcolata in forma chiusa (con un target la direzione di
Fisher fra le sue classi, altrimenti la sua direzione principale). Ogni
attributo pesa uno anche nelle direzioni. Il giudizio sceglie solo la
profondità: la più semplice che legge le righe non viste come la migliore,
confrontate riga per riga.

Con target, stessi split dei test di qualità:

| Dataset | Soglia | Algebrico | Attuale | Tempo algebrico / attuale |
|---|---:|---:|---:|---|
| Iris (10 split) | 0,95 | 0,977 | 0,973 | 0,0 / 0,1 s |
| Wine (10) | 0,95 | 0,967 | 0,978 | 0,0 / 0,5 s |
| Breast cancer (2) | 0,92 | 0,956 | 0,939 | 0,2 / 8 s |
| Sonar (5) | 0,83 | 0,843 | 0,867 | 0,1 / 2,5 s |
| Vehicle (3) | 0,75 | 0,810 | 0,790 | 0,4 / 6 s |
| Vote (5) | 0,94 | 0,949 | 0,961 | 0,1 / 1,4 s |
| Titanic (10) | 0,80 | 0,807 | 0,818 | 0,1 / 2,8 s |
| credit-g (5) | 0,63 | 0,750 | 0,671 | 0,8 / 38 s |
| Letter (1) | — | 0,973 | 0,974 | 39 / 98 s |

Tutte le soglie sono superate, credit-g di molto.

**Senza target non funziona ancora.** Sul gioco (quattro combinazioni di due
attributi più rumore) le direzioni principali tagliano le combinazioni invece
di separarle: 8-16 concetti e combinazioni mescolate, dove il modello attuale
ne trova esattamente quattro. Anche la struttura dei livelli è diversa: ogni
livello divide in due, quindi i concetti generali sono sempre due.

## Concetti come massimo della CP della tesi (`CUConceptModel`)

La CP della tesi, con frequenze e rilevanze al quadrato, è la Category Utility
(MATHEMATICAL-FOUNDATION, proposizione 2): CP² è la media delle η² degli
attributi e lo scarto fra concetti vale K volte la CU. È **non supervisionata**:
ogni attributo pesa uno, il target compreso; prevedere è completare.

Ogni concetto si divide nei figli che massimizzano la CU sulle proprie righe:

- candidati in forma chiusa: i valori di un attributo categorico, la soglia
  migliore di un attributo numerico o di una direzione principale (somme
  cumulative), e le celle di due o tre candidati presi insieme;
- passi di Lloyd nella metrica della CP: con le celle date possono solo
  alzarla, e assorbono le celle residue delle combinazioni;
- la divisione per K fissa da sola il numero di figli (sul gioco le 4
  combinazioni valgono 0,125 contro 0,083 di un solo attributo);
- un concetto si divide solo se la sua CU batte quella che la stessa ricerca
  trova con gli attributi rimescolati indipendentemente (19 volte, 5%): una
  divisione che spiega solo l'attributo di cui è fatta passa anche sul
  rimescolato, quindi i concetti non si moltiplicano sul rumore.

**Due fasi, un solo criterio di lettura.**

1. *Struttura* (non supervisionata, la CP della tesi): i livelli del grafo sono
   solo le divisioni significative. Dove la struttura finisce, i concetti sono
   quelli naturali della tabella.
2. *Previsione* (solo se c'è un target): sotto la struttura ogni concetto si
   divide alla propria media lungo la direzione di Fisher fra le sue classi
   (forma chiusa); in alternativa la stessa divisione parte dalla radice.

Le previsioni leggono i concetti naturali, a meno che un livello più profondo
legga le righe non viste meglio **oltre l'errore** (le stesse righe, confronto
riga per riga, ogni riga letta da concetti che non l'hanno vista): allora il
livello che le legge meglio. Nessun parametro scelto a mano: la struttura dice
quanti concetti ci sono, il giudizio dice se servono concetti più fini per
prevedere.

Senza target: sul gioco, in 8 casi su 8, esattamente le 4 combinazioni e mai
mescolate; due famiglie: 2 concetti generali, poi 8 gruppi (i veri sono 6);
Iris in 10 split su 10: livelli 2 e 3, 3 concetti; stesso seme, stesso modello.

Con target, stessi split dei test di qualità:

| Dataset | Soglia | CP della tesi | Attuale |
|---|---:|---:|---:|
| Iris | 0,95 | 0,977 (3 concetti) | 0,973 (3) |
| Wine | 0,95 | 0,978 (4) | 0,978 (5) |
| Breast cancer | 0,92 | 0,956 (16) | 0,939 (12) |
| Sonar | 0,83 | 0,857 (32) | 0,867 (18) |
| Vehicle | 0,75 | 0,816 (16) | 0,790 (35) |
| Vote | 0,94 | 0,943 (16) | 0,961 (6) |
| Titanic | 0,80 | 0,815 (49) | 0,818 (28) |
| credit-g | 0,63 | 0,751 (8) | 0,671 (45) |
| Letter (1 split) | — | 0,975 (2.605) | 0,974 |

Tutte le soglie sono superate e tutti i test di comportamento senza target
passano. Restano da migliorare: Letter è lenta (358 s contro 98 s: la ricerca
della struttura per ogni concetto è ancora un ciclo Python sui concetti) e Vote
è vicina alla soglia.

## Qualità del linguaggio

Rimandata: prima la tabella, compresa la parte senza target.

## Verso il modello unico, tabulare e generativo

Il nodo e la lettura sono già quelli tabulari: stesse somme, stessa formula
di somiglianza attesa con la varianza. Resta diversa la **partizione**: nella
tabella la trova una ricerca (il criterio CP), qui la dà una formula. La
direzione per unificarle è la stessa nei due casi: una partizione gerarchica
data da un codice (i bit dei segni delle proiezioni, e nel linguaggio i bit
dell'età), con le statistiche additive di ogni prefisso del codice. Nella
tabella questo vorrebbe dire concetti come celle di un albero di proiezioni,
calcolati per gruppi in un solo passaggio; va misurato sui dataset tabulari
prima di adottarlo.
