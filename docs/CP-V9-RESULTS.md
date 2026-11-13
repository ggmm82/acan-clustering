# Nucleo `cp-v9`: che cosa è stato misurato

> **Documento storico.** Il modello attuale è uno solo ed è descritto in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md). Dove questa pagina differisce, vale MODEL.md: in particolare la covarianza nel concetto, i livelli automatici e la fusione per lunghezza di descrizione sono stati tolti.

Fondazione: [MATHEMATICAL-FOUNDATION.md](MATHEMATICAL-FOUNDATION.md).
Piano e criteri: [REVISION-PLAN.md](REVISION-PLAN.md).
Codice: [`acan/core/concepts.py`](../acan/core/concepts.py),
adattatore dell'app [`acan/core/concept_adapter.py`](../acan/core/concept_adapter.py).

## Il criterio unico

Un concetto è una terna additiva di statistiche sufficienti (conteggio, somme,
somme dei quadrati; conteggi per valore negli attributi categorici). Esistono
tre operazioni, identiche per righe di una tabella e chiavi di un modello
linguistico:

| Operazione | Definizione |
|---|---|
| **Instradare** | posterior dei concetti dati i soli attributi osservati |
| **Generalizzare** | sommare le statistiche di due concetti; si sceglie la coppia la cui fusione costa meno lunghezza di descrizione |
| **Giudicare** | log-probabilità di ciò che va previsto, calcolata per ogni osservazione da concetti da cui quell'osservazione è stata **sottratta** |

L'apprendimento parte da molti concetti specifici (uno ogni 4 righe, al più
1024) e generalizza dal basso verso l'alto fino a un solo concetto. Fra tutti i
livelli visitati resta **il più generale** il cui punteggio è entro un errore
standard dal migliore. Ciò che va previsto è il target, quando esiste; tutti gli
attributi, quando non esiste; il token successivo, nel linguaggio.

Corrispondenze con il lavoro precedente:

- **Proprietà aggregante della CP originale.** Era data dalla massa che
  moltiplicava la similarità. Qui è il costo di fusione: unire due copie dello
  stesso concetto ha sempre costo negativo (test
  `test_generalization_adds_statistics_and_removes_duplicates`), e fra livelli
  equivalenti vince il più generale.
- **Sostegno da osservazioni diverse** (`CP-PAIR-SUPPORT.md`). Diventa esatto:
  ogni riga è giudicata senza il proprio contributo, non con un fattore
  moltiplicativo.
- **Albero di generalizzazione della tesi.** La sequenza delle fusioni è un
  albero binario costruito dal basso; il «costo di generalizzazione» è
  l'aumento di lunghezza di descrizione.
- **Causalità del linguaggio.** Sottrarre l'osservazione nel tabulare e usare
  il solo prefisso nel linguaggio sono la stessa regola: nessuna osservazione
  sostiene sé stessa.

Non ci sono numero di concetti, soglia di purezza, soglia di similarità, peso
$\lambda$, $\beta$, epoche o learning rate. Restano tre costanti dichiarate,
uguali per tutti i dataset: forza del prior verso la radice $\kappa=1$, righe
per concetto iniziale 4, peso del target pari al numero di attributi.

La CP riportata è la radice della varianza spiegata media
(`explained-variance-v1`); non è la quantità ottimizzata.

## Tabulare

Protocollo: split dell'app (80/20 stratificato), target e filtri suggeriti
dall'app, semi 42/7/21, valori predefiniti senza alcuna impostazione per
dataset. Profilo precedente: `cp-joint-v1` con profondità 3, 100 epoche,
lr 0,08, batch 300, purezza 0,92, minimo 12 righe. «Concetti» del profilo
precedente: prototipi con almeno una riga di training assegnata, su tutti i
nodi. Comando:

```bash
python scripts/evaluate_concepts.py file.xlsx file.csv --current --app --output reports/concepts.json
```

| Dataset | `cp-v9` libreria | concetti | `cp-v9` app | concetti | `cp-joint-v1` | prototipi | KNN 5 | Random Forest | HistGB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gioco | 100,00 | 4 / 4 / 4 | 100,00 | 4 / 4 / 4 | 100,00 | 4 / 4 / 4 | 100,00 | 100,00 | 55,00 |
| Adult minimale | 75,33 | 4 / 4 / 4 | 75,33 | 4 / 4 / 4 | 67,67 | 66 / 68 / 69 | 74,00 | 72,33 | 76,00 |
| Sonar | 86,51 | 13 / 18 / 13 | 84,13 | 13 / 15 / 20 | 82,54 | 34 / 31 / 35 | 74,60 | 81,75 | 84,92 |
| Titanic | 77,65 | 32 / 37 / 35 | 76,72 | 3 / 2 / 33 | 80,45 | 139 / 137 / 147 | 77,28 | 79,89 | 81,19 |
| Letter | 93,34 | 476 / 456 / 412 | – | – | 92,27 (un seme) | 1.539 | 95,62 | 96,73 | 96,60 |

Accuracy media in percento. «Libreria» usa la tabella grezza (valori mancanti
marginalizzati); «app» passa per la codifica dell'interfaccia (mancanti
imputati con indicatore) e per salvataggio e inferenza JSON. Letter completo
supera il limite di 10.000 righe dell'interfaccia; le baseline su Letter sono
sul solo seme 42. Tempo di training su Letter: circa 67 s contro 351 s.

**Senza target.** Gioco, 101 righe: 4 concetti in 10 semi su 10, con rilevanza
0,996 / 0,996 / 0,997 su Tempo, Luogo, Gioco e 0,02–0,03 su Giorno, Mese, Anno.

Lettura:

- Il criterio di passaggio della fase 1 è rispettato sui dati disponibili:
  Gioco con 4 concetti, Letter con un terzo dei prototipi e +1,1 punti.
- **Titanic è peggiore** del profilo precedente (da −2,8 a −3,7 punti) e sotto
  le baseline ad albero. Nel percorso dell'app la scelta oscilla fra 2–3 e 33
  concetti: il punteggio è quasi piatto e la regola dell'errore standard
  sceglie il livello più generale.
- Su Letter restano 2–3 punti dalle baseline: un concetto è una gaussiana
  diagonale, quindi le correlazioni fra attributi costano concetti.
- **Limite senza target.** Su dati sintetici con rumore uniforme la scelta è
  esatta (4 concetti) in 2 dataset su 4 e dà 5–6 negli altri: il rumore
  uniforme è descritto un po' meglio da più gaussiane, e il punteggio continua
  a salire lentamente oltre il livello corretto. Le quattro combinazioni non
  vengono mai mescolate, ma il conteggio non è garantito.
- La generalizzazione è una ricerca greedy: fino a 5.000 righe viene ripetuta
  cinque volte e lo stesso criterio sceglie fra tutto ciò che è stato visitato.
  Le ripartenze sono state introdotte dopo aver visto Gioco senza target fermo
  in ottimi locali in 3 semi su 8.
- Sono dati di sviluppo già esaminati più volte. Le misure supervisionate
  riportate sono la prima esecuzione, senza aggiustamenti successivi, ma non
  sostituiscono un test su dataset mai usati.

## Linguaggio: lo stesso modello nello strato di attenzione

Lo strato [`acan/lm/concept_attention.py`](../acan/lm/concept_attention.py)
applica gli stessi operatori al prefisso: i token recenti sono concetti di un
solo membro (attenzione esatta), quelli più vecchi vengono generalizzati per
fusione. Nel laboratorio generativo è il profilo `cp-v9`, predefinito per i
nuovi progetti.

**Proprietà verificate dai test** (`tests/test_lm_concepts.py`,
`tests/test_lm_controlled.py`):

- con finestra o numero di concetti ≥ lunghezza lo strato coincide con
  l'attenzione densa della libreria entro 2·10⁻⁵;
- un modello innestato (50% o 100% dei livelli) con finestra sufficiente
  riproduce i logit del modello di partenza: l'innesto non distrugge la
  funzione preaddestrata;
- causalità: cambiare i token futuri non cambia le uscite passate;
- decodifica incrementale identica al prefisso intero per qualunque
  spezzettamento; stato limitato;
- gradienti finiti verso le proiezioni; addestramento nel laboratorio con
  ripresa esatta dopo una pausa;
- i costi di fusione dell'attenzione e della tabella sono numericamente la
  stessa funzione.

### Innesto su SmolLM2-135M, senza addestramento

1.024 token di WikiText-2 test, NLL sulle posizioni ≥ 256, un segmento.

| Modello al passo 0 | NLL |
|---|---:|
| Denso di partenza | 2,111 |
| `cp-v9`, finestra 1.024 (50% e 100% dei livelli) | 2,111 |
| `cp-v9`, 50% dei livelli, finestra 256 + 64 concetti | 3,606 |
| `cp-v9`, 50% dei livelli, finestra 64 + 64 concetti | 4,659 |
| `cp-v9`, 100% dei livelli, finestra 256 + 64 concetti | 3,803 |
| `cp-v9`, 100% dei livelli, finestra 64 + 64 concetti | 4,516 |
| Profilo precedente (`shared-algebra-v1`), 50%, finestra 64 + 64 slot | 6,946 |

A parità di stato il punto di partenza passa da 6,95 a 4,66, ed è esatto quando
la finestra copre il contesto. Resta però lontano dal denso: la compressione
costa.

### Che cosa aggiungono i concetti, a pesi invariati

Tutti i 30 livelli sostituiti, 4 segmenti da 2.048 token, NLL sulle posizioni
≥ 256, nessun parametro addestrato
([`scripts/probe_slot_attention.py`](../scripts/probe_slot_attention.py)).

| Configurazione | NLL | errore std |
|---|---:|---:|
| Attenzione densa | 2,758 | 0,033 |
| Finestra 64 | 5,506 | 0,046 |
| Finestra 128 | 4,981 | 0,045 |
| Finestra 64 + 64 concetti, chiavi **non ruotate** (modello adottato) | **5,260** | 0,049 |
| idem, senza il termine di massa $\log C$ | 5,181 | 0,048 |
| Finestra 64 + 64 blocchi temporali, chiavi non ruotate | 5,509 | 0,047 |
| idem, senza $\log C$ | 5,808 | 0,052 |
| Finestra 64 + 64 concetti, chiavi ruotate | 5,813 | 0,053 |
| idem, senza $\log C$ | 5,451 | 0,049 |
| idem, con correzione di secondo ordine | 6,635 | 0,051 |
| Finestra 64 + 16 concetti, chiavi ruotate | 5,722 | 0,057 |
| Finestra 64 + 64 blocchi temporali, chiavi ruotate | 5,646 | 0,055 |

Lettura:

- **I concetti aiutano, ma poco.** Con chiavi non ruotate migliorano la sola
  finestra di 0,25 (circa cinque errori standard). Riassumere chiavi già
  ruotate dalla posizione invece peggiora: è il motivo per cui il modello
  riassume le chiavi prima della rotazione.
- **Il contenuto conta.** Fondere per costo di generalizzazione (5,26) è meglio
  che fondere token vicini nel testo (5,51, pari alla sola finestra):
  l'operatore di generalizzazione porta informazione.
- **La massa.** Con blocchi temporali il termine $\log C$ è necessario (5,51
  contro 5,81); con concetti di contenuto toglierlo dà 5,18 contro 5,26,
  differenza di circa un errore standard e mezzo: non distinguibile. Il modello
  mantiene la forma derivata.
- **Non basta.** A parità di stato raddoppiare la finestra (4,98) rende più dei
  concetti (5,26), e l'attenzione densa è a 2,76. Il criterio 2a del piano
  (aumento di NLL ≤ 0,10) **non è rispettato**.
- La correzione di secondo ordine peggiora sulle chiavi ruotate e non è stata
  adottata: lo strato usa l'ordine zero, quello coperto dal limite d'errore.

### Velocità

Sulla CPU di prova, 1.024 token, un solo passaggio in avanti: denso circa 520
token/s; `cp-v9` con finestra 64 + 64 concetti fra 10 e 24 token/s; con finestra
256 fra 32 e 63. Il costo per token non cresce con la lunghezza, ma la costante
è alta: una fusione per token assorbito, eseguita in Python per ogni livello.
**Oggi lo strato non è più veloce dell'attenzione densa a queste lunghezze.**
La riduzione prevista è fondere più coppie reciprocamente più vicine per passo.

### Conclusione sul linguaggio

1. Il modello è **uno solo**, nel codice e nelle formule, ed è esatto nel limite
   senza compressione.
2. La qualità con compressione, senza addestramento, è insufficiente: i
   concetti recuperano una piccola parte di ciò che la finestra perde.
3. Che l'addestramento (LoRA, con distillazione dal denso) chiuda il divario è
   la domanda aperta. Il laboratorio ora permette di misurarla correttamente:
   si parte dalla funzione del modello originale e si confronta con il
   controllo denso dello stesso budget.

Metriche aggregate: [`results/cp-v9-tabular.json`](results/cp-v9-tabular.json),
[`results/slot-probe.json`](results/slot-probe.json). Nessuna riga dei file
caricati è inclusa.
