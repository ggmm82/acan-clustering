# Concetti algebrici: il grafo dei concetti senza cicli

Stato: prototipo misurato, **non integrato** nell'apprendimento lungo. Il piano
per portarlo nel prodotto è in [ALGEBRAIC-REVISION-PLAN.md](ALGEBRAIC-REVISION-PLAN.md).

## Il livello del linguaggio: `acan/lm/graph_scan.py`

### Che cosa resta uguale al modello del main

Tutto ciò che è matematica del concetto (MODEL.md, `concept_attention.py`):

- il nodo conserva solo somme: `C = Σ 1`, `U = Σ chiave` (prima della posizione),
  `P = Σ posizione`, `V = Σ valore`;
- si legge con massa per somiglianza: `logit = log C + s · q · centro`, il centro
  ruotato alla posizione media dei suoi membri;
- gli ultimi `window` token sono letti esatti;
- la struttura è scelta senza gradiente, le somme sono differenziabili;
- con abbastanza concetti la lettura **è** l'attenzione normale (verificato a 10⁻¹⁰).

### Che cosa cambia: come si decide chi va con chi

Nel main, quando i concetti superano il massimo, un ciclo unisce una coppia alla
volta (quella che costa meno al lettore): ogni passo dipende dal precedente, e
il ciclo gira per ogni blocco di 16 token. È il motivo della lentezza.

Qui la struttura è **fissa e diadica**, come i livelli del grafo tabulare:

- un blocco di livello `j` copre `2ʲ` token consecutivi e ha al massimo `bins`
  concetti; è la generalizzazione dei suoi due figli di livello `j−1`;
- finché i figli hanno insieme al massimo `bins` concetti restano esatti;
- altrimenti i loro `2·bins` concetti si sommano in `bins`: la partizione è
  quella che abbassa la dispersione delle chiavi dentro i concetti, cioè alza la
  CP (segni lungo le direzioni principali dei centri, poi un passo di Lloyd, che
  può solo abbassarla). Numero di passi fisso, tutti i blocchi insieme;
- chi legge in posizione `t` ha `n = t − window + 1` token vecchi e legge i
  blocchi della scrittura binaria di `n`: fine vicino, grosso lontano, al massimo
  `bins · (log₂ n + 1)` concetti. **Il numero di concetti non può esplodere.**

Ogni concetto di un livello è la somma di concetti del livello sotto: livelli
annidati e additivi, come nel grafo tabulare.

**Addestramento**: tutti i livelli una volta sola per tutti i lettori
(`log₂ T` prodotti fra matrici, struttura fissa come i layer di una rete).
**Generazione**: gli stessi blocchi tenuti come un contatore binario; quando un
token esce dalla finestra entra al livello 0 e i blocchi uguali si generalizzano
con la stessa funzione. I due modi danno lo stesso risultato (differenze sotto
10⁻¹⁰, nei test).

### Misure (CPU, un thread, un solo seme)

Addestramento, token al secondo, modello micro (256 dimensioni, 4 layer,
finestra 32):

| Token per pezzo | Attenzione normale | Main (32 concetti) | Grafo, 8 per blocco | Grafo, 16 per blocco |
|---:|---:|---:|---:|---:|
| 256 | 3.469 | 511 | 2.238 | 2.358 |
| 1.024 | 2.415 | 354 | 1.984 | 2.291 |
| 2.048 | 1.023 | 331 | 2.226 | 2.161 |
| 4.096 | 485 | — | 2.082 | 2.261 |
| 8.192 | 242 | — | 1.894 | 2.017 |

Generazione, token al secondo di un layer dopo N token di contesto:

| Contesto | Attenzione normale | Main | Grafo, 8 | Grafo, 16 |
|---:|---:|---:|---:|---:|
| 256 | 3.853 | 830 | 2.000 | 2.104 |
| 1.024 | 1.705 | 921 | 2.024 | 1.902 |
| 4.096 | 313 | 937 | 1.919 | 1.794 |
| 16.384 | 59 | 793 | 1.664 | 1.343 |

Vettori tenuti a 16.384 token: attenzione normale 16.448, grafo 48 (8 per
blocco) o 64 (16 per blocco).

Qualità, prova in piccolo (modellini da 64 dimensioni, finestra 16, 16 concetti
nel main):

| Prova | Attenzione normale | Solo finestra | Main | Grafo, 16 | Grafo, 8 | Grafo, 4 |
|---|---:|---:|---:|---:|---:|---:|
| Memoria (risposta a 49 token), % giuste | 100 | 5,5 | 94,2 | **99,9** | 88,0 | 47,1 |
| Memoria lontana (fino a 177 token), % giuste | 100 | — | **93,0** | 24,2 | 25,0 | — |
| Testo, bit per carattere (meno è meglio) | 2,518 | 2,451 | 2,538 | 2,527 | 2,529 | 2,521 |
| Tempo della prova di memoria | 84 s | 83 s | 363 s | 183 s | 220 s | 206 s |

**Che cosa dicono.** Il grafo è da 4 a 6 volte più veloce del main in
addestramento e circa 2 volte in generazione, costante con la lunghezza; supera
l'attenzione normale da circa 1.200 token in su. Sulla memoria vicina e sul
testo è alla pari o meglio del main. **Sulla memoria lontana fallisce**: 24%
contro 93%. Il main tiene esatti i token che le domande continuano a leggere;
la partizione del grafo guarda solo la dispersione delle chiavi e, dopo più
generalizzazioni, mescola le risposte con il riempitivo. Finché questo non è
risolto il grafo non può sostituire il main.

Provato e scartato:

- contenitori per segni di proiezioni casuali (la prima versione,
  `algebraic_attention.py`, tolta): memoria 25%, e più lenta;
- unione a coppie di minimo aumento della dispersione (Ward) a passi fissi:
  memoria 25%;
- dispersione pesata con l'energia delle domande viste fin lì (opzione
  `reader`, spenta): nessun guadagno sulla memoria lontana, peggio sulla vicina.

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

## Coerenza fra tabella e linguaggio: valutazione

- **Uguale nei due**: il nodo (somme), la lettura (massa per somiglianza), il
  criterio di partizione (dispersione dentro i concetti, cioè la CP), i livelli
  annidati e additivi.
- **Diverso**: nella tabella i livelli nascono dai dati e finiscono dove le
  divisioni non sono più significative; nel linguaggio sono fissati dall'età.
- **`CUConceptModel` non è pronto a sostituire il modello tabulare del main**:
  supera le soglie, ma su Letter impiega 358 s contro 98 s e usa 2.605 concetti,
  e per prevedere aggiunge un secondo criterio (Fisher) che non viene dalla CP.
  Resta un prototipo di studio.
