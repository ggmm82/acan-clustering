# Il grafo dei concetti: misure

Modello: [MODEL.md](MODEL.md). Tutte le misure sono rigenerabili.

## Tabulare

Split dell'app 80/20, target e filtri dell'app, semi 42/7/21, nessuna
impostazione per dataset.

```bash
python scripts/evaluate_concepts.py file.xlsx file.csv --app --output reports/concepts.json
```

| Dataset | Grafo dei concetti | concetti per livello (un seme) | concetti di lavoro | tempo per seme | un solo livello, senza covarianza (revisione precedente) | covarianza dentro la ricerca (tolta) | KNN 5 | Random Forest | HistGB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Gioco | 100,00 | 4 | 4 / 4 / 4 | < 1 s | 100,00 | 100,00 | 100,00 | 100,00 | 55,00 |
| Adult minimale | 75,33 | 3 → 5 | 5 / 5 / 6 | 1 s | 76,33 | 75,33 | 74,00 | 72,33 | 76,00 |
| Sonar | 83,33 | 3 → 7 → 26 | 26 / 17 / 26 | 3 s | 83,33 | 86,51 | 74,60 | 81,75 | 84,92 |
| Titanic | 78,96 | 3 → 9 → 32 | 32 / 37 / 31 | 3 s | 78,96 | 77,65 | 77,28 | 79,89 | 81,19 |
| Letter | **97,92** | 3 → 12 → 37 → 91 → 614 | 614 / 563 / 562 | 164–169 s | 93,56 | 97,79 | 95,22 | 96,43 | 96,34 |

Percorso dell'app (codifica dell'interfaccia, salvataggio e inferenza JSON):
Gioco 100,00, Adult 74,67, Sonar 86,51, Titanic 74,49.

### Che cosa fanno i livelli

| Verifica | Esito |
|---|---|
| Gioco senza target, 8 tabelle con rumore uniforme | un livello di 4 concetti in 8 su 8 (a un solo livello: 4 concetti in 4 su 8) |
| Due famiglie lontane, tre gruppi interni disposti in modo diverso | livelli 2 → 5 → 6 in 4 su 4: le famiglie, poi i sei gruppi (due gruppi adiacenti compaiono prima uniti) |
| Due famiglie lontane, gli **stessi** tre gruppi interni | livelli 2 → 2 → 3 in 4 su 4: tre comportamenti, ciascuno con due genitori (matrice a 0,5); un albero dà 6 frammenti. A un solo livello i gruppi restano fusi: punteggio di lettura 0,09 contro 0,71 |
| Due famiglie, gli stessi due comportamenti | livelli 2 → 2, matrice densa a 0,5 in 3 su 3 |
| Concetti con più di un genitore (peso oltre 10%) sui dati reali | Adult 20–67%, Sonar 14–33%, Titanic 12–56% dei concetti di un livello |
| Un livello misurato contro sé stesso | criterio a coppie nullo alla precisione di macchina |

Lettura:

- **Che cosa i livelli risolvono.** Il numero di concetti generali è piccolo e
  stabile (3–4 su tutti i dati reali) e non dipende più dal rumore non
  gaussiano; i gruppi vicini rispetto alla scala della tabella si separano
  senza intervento manuale; un comportamento condiviso è un solo nodo.
- **Che cosa non cambiano.** L'accuratezza: il livello di lavoro è scelto dallo
  stesso giudizio di prima e, sui dati reali provati, è sempre risultato
  migliore letto direttamente sulla tabella che attraverso il livello 1.
- **Attributi che variano insieme.** Nel livello di lavoro, a ricerca finita,
  la covarianza è adottata quando legge meglio le osservazioni non viste. Su
  Letter è adottata in 3 semi su 3 e porta l'accuratezza da 93,6% a **97,9%**
  (98,15 / 97,80 / 97,80), sopra KNN, Random Forest e HistGB, in pochi secondi
  in più; negli altri dataset il giudizio non la adotta (Titanic ha valori
  mancanti, su Sonar 60 attributi sono troppi per concetti di poche righe).
  Sonar resta 3,2 punti sotto la versione con lunghezza di descrizione.
- **Costo.** Due ricerche invece di una: su Letter 164–169 secondi contro 66. Il giudizio è la parte più costosa; ora è eseguito ogni dieci per cento
  di concetti in meno.
- Nel percorso dell'app Titanic è più basso della libreria (74,5 contro 79,0):
  la codifica dei valori mancanti dell'interfaccia è diversa. Va indagato.
- Sono dati di sviluppo, esaminati più volte.

### Il documento «Utilità della CP quadratica al centro dell'apprendimento»

| Affermazione | Esito |
|---|---|
| La CP quadratica è aggregante e individua i concetti generali | confermata: è il criterio che sceglie il livello 1 e i successivi |
| Da sola può fermare la generalizzazione | no: cresce con ogni suddivisione di un concetto reale e, per costruzione, premia i concetti grandi; l'arresto resta al giudizio su osservazioni non viste |
| Letter da 66 s a 4,5 s | non riprodotta: con i livelli il tempo cresce a 164–169 s |
| Sonar a 86,5–87,1% | non riprodotta: 83,3% |
| Ottimizzare $-CP^\circ$ chiude il divario del linguaggio (3,01 → 2,76) | non verificabile senza addestramento; a pesi invariati i livelli non migliorano la lettura (sotto) |

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

Due livelli nel contesto (`scripts/probe_level_attention.py`): pesi invariati,
layer 3, 9, 15, 21, 27, tre segmenti da 768 token, primi 4 token e ultimi 64
esatti, i token più vecchi riassunti. Errore quadratico relativo dell'uscita
dell'attenzione rispetto alla densa, a parità di vettori memorizzati.

| Vettori per i token vecchi | solo finestra | concetti su un livello | due livelli, comportamenti condivisi e matrice densa | i due livelli vincono in |
|---:|---:|---:|---:|---:|
| 16 | 1,143 | **0,473** | 0,663 | 47% delle teste |
| 32 | 1,143 | **0,287** | 0,390 | 36% |
| 64 | 1,143 | **0,250** | 0,479 | 36% |

Esito negativo. Sulle chiavi di un modello preaddestrato gli scostamenti dal
centro non si ripetono fra concetti diversi: la struttura che nelle tabelle
rende i livelli più compatti (M9) qui non c'è, e il valore letto come somma
«concetto + comportamento» perde più di quanto la matrice densa recuperi.
Nello strato di attenzione il modello resta quindi a due livelli soli: i
concetti del prefisso e la finestra esatta. Se un addestramento possa indurre
chiavi con quella struttura non è stato misurato.

Velocità sulla CPU di prova: 38–86 token/s contro circa 400 della densa. Lo
strato è ancora più lento dell'attenzione densa a queste lunghezze. Nulla è
addestrato.

Metriche aggregate: [`results/concept-graph-tabular.json`](results/concept-graph-tabular.json),
[`results/slot-probe.json`](results/slot-probe.json),
[`results/level-attention.json`](results/level-attention.json).
