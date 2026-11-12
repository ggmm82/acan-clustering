# Nucleo `cp-v9`: che cosa è stato misurato

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

## Linguaggio: prova a pesi invariati

Domanda (fase 2a del piano): le stesse operazioni, applicate alle chiavi di
attenzione di un modello preaddestrato, approssimano l'attenzione densa?
Nessun parametro è addestrato. In **tutti** i 30 livelli di SmolLM2-135M le
chiavi più vecchie della finestra esatta sono riassunte da concetti; lettura
con un'unica softmax su chiavi esatte e concetti. Con finestra ≥ lunghezza lo
strato coincide con l'attenzione della libreria (scarto massimo sui logit
8·10⁻⁵). WikiText-2 test, 4 segmenti da 2.048 token, NLL sulle posizioni ≥ 256.

```bash
python scripts/probe_slot_attention.py --text wikitext2-test.txt --output reports/slot-probe.json
```

| Configurazione | NLL | errore std |
|---|---:|---:|
| Attenzione densa | 2,758 | 0,033 |
| Finestra 64 | 5,506 | 0,046 |
| Finestra 128 | 4,981 | 0,045 |
| Finestra 64 + 64 concetti (formula derivata, con $\log C$) | 5,813 | 0,053 |
| Finestra 64 + 64 blocchi temporali | 5,646 | 0,055 |
| Finestra 64 + 64 concetti, senza $\log C$ | 5,451 | 0,049 |
| Finestra 64 + 64 concetti, correzione gaussiana | 6,635 | 0,051 |

**Esito: negativo.** Il criterio 2a (aumento di NLL ≤ 0,10 e almeno metà di
quello della sola finestra) non è rispettato.

- A pesi invariati questo modello dipende fortemente dal contesto lontano: la
  sola finestra porta la NLL da 2,76 a 5,5.
- Lo strato derivato **peggiora** la sola finestra. Senza il termine di massa
  migliora di 0,055, entro circa un errore standard: non è un effetto
  dimostrato. A parità di stato, raddoppiare la finestra (4,98) è molto meglio
  di aggiungere 64 concetti.
- Il Teorema 6 resta vero ma qui è vuoto: richiede chiavi vicine al proprio
  centroide, e le chiavi in cache sono ruotate dalla posizione (RoPE). Il
  centroide di chiavi a posizioni diverse si contrae, il suo punteggio sale
  verso zero mentre quello delle chiavi vere è spesso molto negativo, e il
  fattore $C_i$ amplifica l'errore. La correzione di secondo ordine lo
  amplifica ancora.
- È la questione che la fondazione segnalava come aperta (C.4, RoPE). La misura
  dice che non è un dettaglio: è l'ostacolo principale.

Conseguenze per il progetto:

1. Il criterio unico è **verificato sul tabulare** e **non verificato sul
   linguaggio**. L'unificazione è reale a livello di operazioni e di codice;
   non è ancora un risultato sul modello generativo.
2. Non si può dichiarare che uno strato a concetti sostituisca l'attenzione
   densa di un modello preaddestrato senza addestramento.
3. Restano due strade, entrambe da misurare prima di investire: concetti su
   chiavi **non ruotate** con posizione trattata a parte, e addestramento
   (LoRA con distillazione dal modello denso) partendo da uno strato che
   conserva la funzione entro la finestra. I controlli necessari per leggerne
   l'esito sono il passo successivo di questa revisione.

Metriche aggregate: [`results/cp-v9-tabular.json`](results/cp-v9-tabular.json),
[`results/slot-probe.json`](results/slot-probe.json). Nessuna riga dei file
caricati è inclusa.
