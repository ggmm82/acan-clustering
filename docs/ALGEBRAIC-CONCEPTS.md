# Il grafo dei concetti senza ricerca sul contesto (`concept-graph-v2`)

Riferimento del modello: [MODEL.md](MODEL.md). Questa pagina descrive che cosa
cambia nello strato del linguaggio dalla versione 8.7.0, perché, e che cosa è
stato misurato. Codice: [`acan/lm/graph_scan.py`](../acan/lm/graph_scan.py)
(il calcolo), [`acan/lm/graph_attention.py`](../acan/lm/graph_attention.py)
(lo strato nel modello).

## Il problema

Nello strato `concept-graph-v1` i concetti del passato sono un solo gruppo. Ogni
16 token nuovi, quelli che escono dalla finestra entrano come concetti singoli e
un ciclo unisce una coppia alla volta finché il gruppo torna alla dimensione
massima. Ogni unione dipende dalla precedente e ogni tratto di 16 token dipende
dallo stato lasciato dal tratto prima: un pezzo di testo va letto in sequenza.
Misurato sul modello micro: 330-540 token al secondo in addestramento.

## Un solo modello: che cosa è comune a tabella e linguaggio

Le definizioni sono in [`acan/concept_algebra.py`](../acan/concept_algebra.py) e
valgono per tutti e due:

| | Tabella | Linguaggio |
|---|---|---|
| osservazione | una riga: attributi e target | un token: chiave e valore |
| nodo | somme dei membri | somme dei membri |
| lettura | massa per somiglianza attesa | massa per somiglianza attesa |
| generalizzazione | si unisce la coppia di costo minimo | si unisce la coppia di costo minimo |
| costo | `reader_cost`: letture × raggio | `reader_cost`: letture × raggio |
| letture | la massa (ogni riga legge i suoi concetti) | le quote di attenzione ricevute |
| livelli | annidati e additivi, scelti dal criterio a coppie | annidati e additivi, fissati dall'età |

Il modello tabulare è `ConceptModel` e non cambia. Nel linguaggio cambia solo
l'ultima riga: **quando** e **fra quali** concetti avviene la generalizzazione.

## La struttura fissa

Statistiche di un concetto (tutte somme):

    C = Σ 1    U = Σ chiave    S = Σ |chiave|²    P = Σ posizione    V = Σ valore    A = Σ letture

La chiave è presa prima della rotazione di posizione; il centro `U/C` è ruotato
alla posizione media `P/C` quando viene letto. Lettura: `log C + λ q·centro`.

**Letture di un token.** Le quote di attenzione che ha ricevuto dai lettori che
lo avevano nella finestra esatta (softmax sui soli token della finestra). Sono
note quando il token esce dalla finestra e non dipendono dai concetti, quindi
si calcolano per tutti i token insieme.

**Costo di un'unione.** `Δ = (a_i + a_l) ρ_{i∪l} − a_i ρ_i − a_l ρ_l`, con
`ρ² = (S/C − |U|²/C²)/d` e `a = A +` un millesimo della media (come in v1: i
concetti mai letti pesano poco, e due copie dello stesso concetto costano zero).

**Blocchi.** Un blocco di livello `j` copre `2ʲ` token consecutivi e ha al
massimo `B` concetti (`slots`). È la generalizzazione dei suoi due figli:

- se i figli hanno insieme al massimo `B` concetti, restano come sono: il blocco
  è esatto (un concetto per token);
- altrimenti si unisce la coppia di `Δ` minimo finché ne restano `B`. Da `2B` a
  `B` sono `B` unioni: un numero fissato dalle dimensioni, non dai dati.

**Chi legge.** In posizione `t`, con finestra `w`, i token vecchi sono
`n = t − w + 1`. Il lettore legge la finestra esatta e i blocchi della scrittura
binaria di `n`: un blocco di livello `j` per ogni bit acceso, i più grandi sui
token più antichi. Ogni token vecchio è coperto una volta sola. Vettori letti:
al massimo `w + B (⌊log₂ n⌋ + 1)`. Con `B ≥ n` è l'attenzione ordinaria.

## Perché è algebrico

- I blocchi sono gli stessi per ogni testo della stessa lunghezza. In
  addestramento un livello è un solo calcolo per tutti i blocchi e tutti i
  lettori: `⌊log₂ T⌋` passi, come i layer di una rete, ciascuno `O(T·B·d)`.
- I blocchi fino a `B` token sono esatti e si leggono insieme alla finestra,
  come attenzione ordinaria fra tratti di `w` domande e le chiavi appena dietro.
- Dentro un blocco le `B` unioni si fanno per tutti i blocchi insieme. Il costo
  di ogni coppia è una forma chiusa di `C`, `S`, `A` e dei prodotti `U_i·U_l`;
  dopo un'unione cambia solo la riga del concetto nuovo.
- Mentre il modello scrive, i blocchi sono un contatore binario: i token vecchi
  che riempiono un blocco di `B` entrano nel contatore e due blocchi uguali
  diventano il loro genitore, con la stessa funzione. Un prompt intero si carica
  in un colpo (gli stessi blocchi dell'addestramento) e poi si continua un token
  alla volta.

Non c'è nessuna ricerca sul contesto e nessun passo che dipenda dal precedente
lungo il testo. Restano cicli di lunghezza fissa: i livelli e le `B` unioni.

## Garanzie verificate dai test

`tests/test_graph_scan.py`, `tests/test_graph_attention.py`:

- addestramento e scrittura danno la stessa lettura (differenza sotto 10⁻¹⁰ in
  doppia precisione), anche caricando un prompt e continuando, e leggendo a
  tratti di lunghezza qualsiasi. In precisione singola, quella dei modelli
  veri, due unioni di costo quasi uguale possono essere scelte in ordine
  diverso nei due modi: sul modello micro appena nato, leggendo 600 token
  tutti insieme oppure un prompt e poi il seguito, i logit differiscono al
  massimo di 0,018 su un'ampiezza di 1,4 (v1: 0,001); su un testo fatto di
  una sequenza ripetuta, dove le copie sono moltissime, di 0,03. Trattare
  come copie esatte i concetti molto stretti toglie questo scarto ma ha
  peggiorato la memoria lontana (85% invece di 99%) e non è stato adottato;
- con abbastanza concetti, o una finestra che copre tutto, lo strato coincide
  con l'attenzione ordinaria del modello;
- nessuna posizione legge il futuro; i token vecchi sono letti;
- la memoria non supera `w + B (⌊log₂ n⌋ + 1)`;
- due copie si uniscono per prime; un concetto letto resta separato;
- i progetti dei profili precedenti conservano le loro ricevute, e un progetto
  v2 non può usarne una precedente.

## Misure

Processore di sviluppo, un thread. Un solo seme dove non è detto altro.

**Addestramento, token al secondo** (solo i layer di lettura, forma del modello
micro: 256 dimensioni, 4 layer, 4 teste, finestra 16, 16 concetti):

| Token per pezzo | Attenzione normale | v1 | v2 |
|---:|---:|---:|---:|
| 256 | 3.534 | 542 | 1.543 |
| 1.024 | 2.297 | 494 | 1.880 |
| 2.048 | 956 | 437 | 2.025 |
| 4.096 | 476 | — | 1.892 |
| 8.192 | 250 | — | 1.845 |

**Modello nativo micro intero** (vocabolario 32.000, classi di produzione):

| | Attenzione normale | v1 | v2 |
|---|---:|---:|---:|
| Addestramento, 256 token | 886 | 403 | 765 |
| Addestramento, 1.024 token | 907 | 359 | 812 |
| Addestramento, 2.048 token | 581 | 337 | 792 |
| Scrittura di 128 token dopo 64 | 209 | 103 | 122 |
| Scrittura di 128 token dopo 1.024 (prompt compreso) | 127 | 41 | 74 |

**Scrittura, token al secondo di un layer** dopo N token di contesto:

| Contesto | Attenzione normale | v1 | v2 |
|---:|---:|---:|---:|
| 256 | 3.644 | 678 | 1.682 |
| 1.024 | 1.924 | 892 | 1.531 |
| 4.096 | 324 | 1.047 | 819 |
| 16.384 | 67 | 917 | 1.254 |

A 16.384 token: 16.384 vettori per layer con l'attenzione normale, 64 con v2.
Il valore di v2 a 4.096 cade dove molti blocchi si uniscono insieme.

**Qualità** (prova in piccolo: 64 dimensioni, finestra 16, 16 concetti):

| Prova, seme 0 | Attenzione normale | Solo finestra | v1 | v2 |
|---|---:|---:|---:|---:|
| Memoria, risposta a 49 token | 100% | 5,5% | 94,2% | 100% |
| Memoria lontana, fino a 177 token | 100% | — | 93% | 99% |
| Testo, bit per carattere | 2,518 | 2,451 | 2,538 | 2,522 |

Memoria lontana di v2 per distanza (0, 8, 16, 24, 40, 56, 72, 104, 120, 136,
176 token): 100% fino a 120, 99% a 136 e 176.

Altri semi: sui semi 1 e 2 la prova di memoria non è imparata in 2.000 passi
nemmeno dall'attenzione normale (25%, il caso). Memoria vicina: v2 25% e 100%,
v1 26% e 26%. Memoria lontana: v2 25% e 26%, attenzione normale 25% e 26%.

## Che cosa dicono, e che cosa no

- v2 addestra da 2,8 a 4,6 volte più veloce di v1 nei layer di lettura, circa il
  doppio sul modello intero, e scrive da 1,5 a 2,5 volte più veloce per layer
  (con l'eccezione dei momenti di unione). Il suo tempo quasi non cresce con la
  lunghezza.
- Sui pezzi corti l'attenzione normale resta più veloce: nei layer fino a circa
  1.300 token, sul modello intero fino a circa 1.000.
- Sulla qualità v2 non è sotto v1 in nessuna prova fatta.
- La memoria non è più fissa: cresce con il logaritmo del contesto.
- In precisione singola la lettura mentre scrive può scostarsi di poco da
  quella dell'addestramento (vedi le garanzie): l'addestramento non ne risente.
- Le prove sono su modellini e su pochi semi. Su GPU e sui modelli grandi la
  velocità va rimisurata.

## Strade provate e scartate

| Variante | Esito |
|---|---|
| Contenitori per segni di proiezioni casuali | memoria 25%: nessun criterio decide chi va con chi |
| Segni lungo le direzioni principali più passi di Lloyd | memoria vicina 88-100%, lontana 25% |
| Chiavi sommate dopo la rotazione di posizione | memoria 33%: token uguali in posizioni diverse non si riconoscono |
| Unione a coppie pesata con la massa | memoria lontana 25%: i token singoli sono i primi a sparire |
| Unione a coppie con letture uguali per tutti | memoria lontana 38% |
| Dispersione pesata con l'energia delle domande | nessun guadagno |
| **Unione a coppie con le letture ricevute nella finestra** | **memoria lontana 99%: adottata** |

In tutte le varianti fallite il modello sbagliava anche a distanza zero, dove lo
strato è l'attenzione ordinaria: non perdeva informazione lontana, imparava
peggio perché i concetti formati senza le letture mescolavano ciò che serve con
il riempitivo.

**Tabella.** È stato studiato anche un secondo modello tabulare (concetti come
massimo della Category Utility con divisioni significative, più divisioni di
Fisher per la previsione). Superava le soglie dei test, ma su Letter impiegava
358 s contro 98 s con 2.605 concetti e usava due criteri invece di uno. Non è
stato adottato: il modello tabulare resta `ConceptModel`.

## Compatibilità

`concept-graph-v2` è un profilo nuovo. I progetti creati prima continuano con il
loro strato e i loro checkpoint; i file del contratto dei checkpoint non sono
cambiati. I nuovi progetti (apprendimento lungo, avvio guidato, modulo manuale)
e la prova in piccolo usano v2. `slots` è il numero di concetti di un blocco.
