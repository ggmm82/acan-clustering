# Grafo dei concetti a livelli: che cosa è stato misurato

> **Documento storico.** Il modello attuale è uno solo ed è descritto in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md). Dove questa pagina differisce, vale MODEL.md: in particolare la covarianza nel concetto, i livelli automatici e la fusione per lunghezza di descrizione sono stati tolti.

Fondamento: [CONCEPT-GRAPH.md](CONCEPT-GRAPH.md). Piano e criteri:
[NEXT-PR-PLAN.md](NEXT-PR-PLAN.md). Misure precedenti:
[CP-V9-RESULTS.md](CP-V9-RESULTS.md).

## Che cosa è cambiato nel modello

| Parte | Prima | Ora |
|---|---|---|
| Livelli | un solo livello di concetti | ogni concetto può essere riesaminato sulle proprie righe e sulla propria scala; il livello fine ha come riferimento il concetto che lo contiene |
| Quando approfondire | soglie di purezza e righe minime (profili precedenti), o solo a mano | il figlio sceglie i concetti con metà delle sue righe; resta solo se sull'altra metà la divisione batte un concetto solo **e** il ramo batte il genitore di più di un errore standard |
| Forma del concetto | attributi numerici indipendenti | anche covarianza fra gli attributi numerici, come statistica additiva, se giudicata migliore |
| Fusione nei layer di attenzione | minima lunghezza di descrizione | minima perdita per il lettore: attenzione ricevuta × dispersione (`cp-graph-v1`) |

Codice: [`acan/concept_algebra.py`](../acan/concept_algebra.py),
[`acan/core/concepts.py`](../acan/core/concepts.py),
[`acan/core/graph.py`](../acan/core/graph.py),
[`acan/lm/concept_attention.py`](../acan/lm/concept_attention.py).

## Tabulare

Protocollo identico a CP-V9-RESULTS.md: split dell'app 80/20, semi 42/7/21,
nessuna impostazione per dataset.

```bash
python scripts/evaluate_concepts.py file.xlsx file.csv --tree --app --output reports/concepts.json
```

| Dataset | Grafo a livelli | concetti | Un livello (`cp-v9` precedente) | concetti | KNN 5 | Random Forest | HistGB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gioco | 100,00 | 4 / 4 / 4 | 100,00 | 4 / 4 / 4 | 100,00 | 100,00 | 55,00 |
| Adult minimale | 75,33 | 4 / 4 / 4 | 75,33 | 4 / 4 / 4 | 74,00 | 72,33 | 76,00 |
| Sonar | 86,51 | 13 / 18 / 13 | 86,51 | 13 / 18 / 13 | 74,60 | 81,75 | 84,92 |
| Titanic | 77,65 | 32 / 37 / 35 | 77,65 | 32 / 37 / 35 | 77,28 | 79,89 | 81,19 |
| Letter | 97,79 | 158 / 153 / 171 | 93,34 | 476 / 456 / 412 | 95,62 | 96,73 | 96,60 |

Percorso dell'app (codifica dell'interfaccia, salvataggio e inferenza JSON):
Gioco 100,00 (4 concetti), Adult 75,33, Sonar 84,92, Titanic 78,21.
Le baseline su Letter sono sul solo seme 42; il grafo a livelli sui tre semi 42 / 7 / 21.

Lettura:

- **Letter.** Con la covarianza dentro il concetto il modello supera il 95,78%
  storico e i metodi di confronto, con un terzo dei concetti. Era il limite
  indicato come «le relazioni fra attributi costano concetti in più».
- **Il giudizio sceglie.** Sugli altri quattro dataset la covarianza non prevede
  meglio le righe non viste e non viene adottata: i risultati coincidono con
  il livello unico, senza regressioni.
- **I livelli automatici compaiono di rado su questi dati.** Con la regola di
  accettazione su righe fresche nessun approfondimento viene tenuto su Adult,
  Sonar e Titanic; su Letter pochi (2 livelli, 4–6 nodi). Servono quando la
  tabella ha strutture a scale diverse: nell'esempio di prova con due famiglie
  di tre gruppi vicini, un livello trova le 2 famiglie e il secondo i 6 gruppi,
  con e senza target (`tests/test_concept_graph.py`).
- **Perché un solo livello non basta in quel caso.** Il prior di cp-v9 ritira
  la varianza di ogni concetto verso quella della tabella intera: gruppi
  piccoli rispetto alla tabella diventano larghi quanto la loro distanza e si
  fondono. Riesaminare un concetto sulla propria scala è esattamente «ogni
  livello è il prior del successivo».
- **Costo.** Con la covarianza l'apprendimento su Letter richiede circa
  14 minuti contro circa 1.

### Criteri del piano

| Criterio | Esito |
|---|---|
| Gioco: 4 concetti con e senza target | rispettato |
| Letter ≥ 93,3% con non più concetti (passo 2, senza covarianza) | **non rispettato**: 93,27% con 507 concetti; i livelli da soli sono neutri |
| Adult e Titanic non peggiori del livello unico | rispettato (identici) |
| Nessuna soglia di purezza o righe minime nel percorso predefinito | rispettato |
| Letter ≥ 95% senza regressioni su Sonar (passo 3) | rispettato |

### Ciò che non ha funzionato

- **Punteggi normali.** Riportare gli attributi numerici a una scala gaussiana
  per ranghi elimina i concetti spurii con rumore uniforme (4 concetti in 8
  casi sintetici su 8, contro 4 su 8), ma cancella i vuoti fra gruppi ben
  separati: nell'esempio a due scale la radice non trova più le due famiglie.
  Sul supervisionato l'effetto è misto (Titanic +2,2 punti, Adult +1,3,
  Sonar −1,6). Resta disponibile e **non** è predefinito. Il limite dei concetti
  in eccesso senza target con rumore non gaussiano resta quindi aperto.
- **La CP a coppie come regola di divisione.** Usata a ogni nodo dà su Letter
  89,7% con 1.317 concetti (CONCEPT-GRAPH.md); il giudizio sulle righe non
  viste dà di più con meno. La CP a coppie è riportata per livello
  (`pair_levels` nel modello salvato) e non decide la struttura.
- **Livelli costruiti dal basso su un solo adattamento.** Ereditano il prior di
  radice e non risolvono le scale fini; per questo i livelli sono costruiti
  dall'alto. `acan/core/graph.py` resta per descrivere i livelli di un nodo.

## Linguaggio

### Controllo corretto e fusione per il lettore

Tutti i 30 layer di SmolLM2-135M sostituiti, pesi invariati, WikiText-2 test,
4 segmenti da 2.048 token, NLL sulle posizioni ≥ 256
([`scripts/probe_slot_attention.py`](../scripts/probe_slot_attention.py)).

| Configurazione | vettori letti | NLL | errore std |
|---|---:|---:|---:|
| Attenzione densa | tutti | 2,758 | 0,033 |
| Finestra 64 | 64 | 5,506 | 0,046 |
| Primi 4 token + finestra 64 | 68 | 3,387 | 0,04 |
| Primi 4 token + finestra 124 | 128 | 3,229 | 0,04 |
| Finestra 64 + 64 concetti, lunghezza di descrizione (cp-v9) | 128 | 5,260 | 0,049 |
| Primi 4 + finestra 60 + 64 concetti, lunghezza di descrizione | 128 | 3,142 | 0,04 |
| **Finestra 64 + 64 concetti, fusione per il lettore** | 128 | **3,005** | 0,04 |
| Primi 4 + finestra 60 + 64 concetti, fusione per il lettore | 128 | 3,006 | 0,04 |

- **Il controllo delle prove precedenti era sbagliato.** Il modello concentra
  attenzione sui primi token: tenerne esatti quattro porta la sola finestra da
  5,51 a 3,39. Il confronto «5,51 contro 5,26» di CP-V9-RESULTS.md misurava
  soprattutto questo effetto.
- **Contro il controllo corretto i concetti aiutano.** A parità di vettori
  letti: 3,23 senza concetti, 3,14 con concetti a lunghezza di descrizione,
  **3,01 con fusione per il lettore**. Il divario dalla densa passa da 0,47 a
  0,25.
- **La fusione per il lettore conserva i primi token da sola**: con o senza
  trattarli a parte il risultato è lo stesso (3,005 e 3,006).

### Strato innestato, senza addestramento

1.024 token, NLL sulle posizioni ≥ 256, un segmento, profilo `cp-graph-v1`.

| Modello al passo 0 | NLL | prima (`cp-v9`) |
|---|---:|---:|
| Denso di partenza | 2,111 | 2,111 |
| 50% dei layer, finestra 64 + 64 concetti | 2,762 | 4,659 |
| 50% dei layer, finestra 256 + 64 concetti | 2,657 | 3,606 |
| 100% dei layer, finestra 64 + 64 concetti | 2,859 | 4,516 |
| 100% dei layer, finestra 256 + 64 concetti | 2,705 | 3,803 |

Velocità sulla CPU di prova: 38–86 token/s contro circa 400 della densa.

### Grafo completo letto in modo adattivo (un solo strato)

Con l'intero albero dei token conservato e solo la lettura limitata, l'errore
dell'uscita di un layer è 0,10–0,21 in tutti i layer (CONCEPT-GRAPH.md, §3.2).
È il limite superiore di ciò che la lettura adattiva può dare; a stato
limitato la forma che lo realizza è la fusione per il lettore qui sopra.

### Criteri del piano

| Criterio | Esito |
|---|---|
| Uscita identica alla densa senza compressione | rispettato (test) |
| A parità di vettori letti NLL migliore di «primi token + finestra» | rispettato: 3,005 contro 3,229 |
| Decodifica incrementale identica al prefisso intero | rispettato (test) |

### Scostamenti dal piano e limiti

- **Un grafo solo per tutti i layer non è stato realizzato.** La fusione per il
  lettore dipende dall'attenzione di ciascuna testa, quindi la struttura è per
  testa. La misura che una partizione unica serve tutti i layer resta valida
  solo per la partizione a lunghezza di descrizione.
- **La lettura adattiva per query non è nello strato.** A memoria limitata i
  concetti conservati vengono letti tutti; ciò che si adatta alle query è
  quali concetti restano fini.
- **Il divario resta**: 3,01 contro 2,76 su tutti i layer, senza addestramento.
- **Lo strato è ancora più lento dell'attenzione densa** a queste lunghezze.
- Nulla è addestrato. L'effetto dell'addestramento resta da misurare su GPU.

Metriche aggregate: [`results/cp-graph-tabular.json`](results/cp-graph-tabular.json),
[`results/slot-probe.json`](results/slot-probe.json),
[`results/concept-tree.json`](results/concept-tree.json).
