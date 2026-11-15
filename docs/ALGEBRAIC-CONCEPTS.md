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

## Qualità

(prova in piccolo: vedi la sezione aggiunta alla fine delle misure)

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
