# Il grafo dei concetti: misure

Modello: [MODEL.md](MODEL.md). Tutte le misure sono rigenerabili.

## Tabulare

Split dell'app 80/20, target e filtri dell'app, semi 42/7/21, nessuna
impostazione per dataset.

```bash
python scripts/evaluate_concepts.py file.xlsx file.csv --app --output reports/concepts.json
```

| Dataset | Grafo dei concetti | concetti | archi | tempo per seme | cp-v9 (lunghezza di descrizione) | con covarianza (tolta) | KNN 5 | Random Forest | HistGB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gioco | 100,00 | 4 / 4 / 4 | 2 | < 1 s | 100,00 | 100,00 | 100,00 | 100,00 | 55,00 |
| Adult minimale | 76,33 | 5 / 5 / 5 | 4–5 | 1 s | 75,33 | 75,33 | 74,00 | 72,33 | 76,00 |
| Sonar | 83,33 | 26 / 17 / 26 | 24–46 | 2 s | 86,51 | 86,51 | 74,60 | 81,75 | 84,92 |
| Titanic | 78,96 | 32 / 37 / 31 | 55–65 | 2 s | 77,65 | 77,65 | 77,28 | 79,89 | 81,19 |
| Letter | 93,56 | 600 / 541 / 562 | 1.075–1.183 | 66 s | 93,34 | 97,79 | 95,62 | 96,73 | 96,60 |

Percorso dell'app (codifica dell'interfaccia, salvataggio e inferenza JSON):
Gioco 100,00, Adult 74,67, Sonar 86,51, Titanic 74,49.
Gioco senza target: 4 concetti in 10 semi su 10. Baseline su Letter sul solo
seme 42.

Lettura:

- Il costo unico del lettore è equivalente alla lunghezza di descrizione sul
  tabulare: meglio su Adult, Titanic e Letter, peggio su Sonar (−3,2 punti su
  42 righe di prova).
- **Qualità ceduta per il costo.** Senza covarianza Letter torna al 93,6%
  invece del 97,8%, e resta sotto i metodi di confronto. In cambio il tempo
  torna a circa un minuto (era circa 14) e non cresce più con il cubo degli
  attributi: su Sonar, 60 attributi, 2 secondi invece di oltre 60.
- Nel percorso dell'app Titanic è più basso della libreria (74,5 contro 79,0):
  la codifica dei valori mancanti dell'interfaccia è diversa. Va indagato.
- Sono dati di sviluppo, esaminati più volte.

## Linguaggio

Tutti i 30 layer di SmolLM2-135M sostituiti, pesi invariati, WikiText-2 test,
4 segmenti da 2.048 token, NLL sulle posizioni ≥ 256
(`scripts/probe_slot_attention.py`).

| Configurazione | vettori letti | NLL |
|---|---:|---:|
| Attenzione densa | tutti | 2,758 |
| Finestra 64 | 64 | 5,506 |
| Primi 4 token + finestra 124 (controllo) | 128 | 3,229 |
| **Finestra 64 + 64 concetti, costo del lettore (il modello)** | 128 | **3,005** |
| idem, con smorzamento di fase | 128 | 3,006 |
| idem, con termine di varianza | 128 | 3,594 |
| idem, raggio su chiave e valore | 128 | 5,165 |
| idem, unendo solo concetti letti insieme | 128 | 5,369 |
| Finestra 64 + 64 concetti, lunghezza di descrizione | 128 | 5,260 |

- Il modello batte il controllo corretto a parità di vettori (3,005 contro
  3,229) e conserva da solo i primi token.
- Le varianti provate per estenderlo peggiorano o non cambiano nulla: il raggio
  va misurato sulla sola parte riconosciuta, e ogni coppia va considerata.
  È la definizione adottata anche nel tabulare.
- Il divario dalla densa resta: 0,25.

Strato innestato, senza addestramento, 1.024 token (un segmento):

| Modello al passo 0 | NLL |
|---|---:|
| Denso di partenza | 2,111 |
| 50% dei layer, finestra 64 + 64 concetti | 2,762 |
| 50% dei layer, finestra 256 + 64 concetti | 2,657 |
| 100% dei layer, finestra 64 + 64 concetti | 2,859 |
| 100% dei layer, finestra 256 + 64 concetti | 2,705 |

Velocità sulla CPU di prova: 38–86 token/s contro circa 400 della densa. Lo
strato è ancora più lento dell'attenzione densa a queste lunghezze. Nulla è
addestrato.

Metriche aggregate: [`results/concept-graph-tabular.json`](results/concept-graph-tabular.json),
[`results/slot-probe.json`](results/slot-probe.json).
