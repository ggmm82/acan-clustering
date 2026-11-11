# Unico nucleo matematico CP: tabulare e generativo

Il profilo **`cp-joint-v1`** usa lo stesso modello concettuale nei due percorsi.
Non basta avere un'etichetta CP o una loss simile: rilevanza, distanza, routing,
conteggi/somme, centroidi e CP sono implementati una sola volta in
[`acan/algebra.py`](../acan/algebra.py). Gli adattatori sono
[`core/supervised.py`](../acan/core/supervised.py) e
[`lm/cp_attention.py`](../acan/lm/cp_attention.py), classe `CPJointAttention`.
È il profilo preselezionato per i nuovi modelli in entrambe le schede della GUI;
i default delle API precedenti restano compatibili. La spiegazione è anche nella
scheda **Matematica** dell'app.

## Da dove viene l'unificazione

I PDF ricaricati coincidono byte per byte con quelli del repository:

| Documento | Riferimento utilizzato | SHA-256 |
| --- | --- | --- |
| [Algebraic Concept Attention](../paper/Algebraic_Concept_Attention_Scientific_Paper.pdf) | §3: prototipi, rilevanza, competizione e CP | `1fdd1a5f6e235d291c29364cae2dd1bd138df74db42256bb9ec36dd528f40396` |
| [ACAN-LM vs. MiniCPM](../paper/ACAN_LM_vs_MiniCPM_Scientific_Paper.pdf) | §§2.1–2.2: slot, riepiloghi cumulativi e finestra locale | `22df94993bdd71d03616375990ef8a4c05d1c1aa52f0d0711d07225dc3f5d65f` |

Il nuovo profilo sceglie una formulazione comune basata sulla CP. Non aggiunge
al solo LM termini di routing assenti nel tabulare. Rispetto ai paper, usa
rilevanza root-shift limitata, un solo routing a distanza e CP supervisionata
normalizzata. Non riproduce integralmente la rilevanza duale, il prior di massa
a due stadi o il termine bilineare del paper LM. I profili precedenti restano
disponibili con identificatori e contratti propri, per poterli confrontare e caricare.

La sostituzione delle statistiche globali della frase con statistiche di prefisso
è necessaria per la generazione autoregressiva. Usare il testo futuro per
calcolare Root, min, max o CP violerebbe la causalità anche con una loss next-token.

## Equazioni effettive del nucleo

Siano X e i prototipi apprendibili A in [0,1]. O è l'insieme dei canali osservabili.
Le colonne target del tabulare sono escluse da O. Ogni canale costante ha omega
azzerato e denominatore sostituito con 1 per la sola protezione numerica.

$$
\Delta_j=(max_j-min_j)/2+|Root_j-(min_j+max_j)/2|,
\quad R^{route}_{ij}=\mathrm{clip}_{[0,1]}(\omega_j|A_{ij}-Root_j|/\Delta_j).
$$

$$
W_{ij}=\frac{(R^{route}_{ij})^2+10^{-4}}
{\sum_{q\in O}[(R^{route}_{iq})^2+10^{-4}]},
\quad D_{it}=\sum_{j\in O}W_{ij}(X_{tj}-A_{ij})^2,
\quad T_{it}=\mathrm{softmax}_i(-\beta D_{it}).
$$

$$
C_i=C_i^{prev}+\sum_tT_{it},\quad
U_i=U_i^{prev}+\sum_tT_{it}X_t,\quad G_i=U_i/C_i,\quad
F_i=C_i/\sum_qC_q.
$$

La stessa funzione di rilevanza è applicata a G_eff=(G+A)/2 per ottenere R.
Il clipping protegge anche i prototipi che cadono fuori dall'intervallo di un
prefisso corto: senza questo limite CP/√d potrebbe superare 1.

$$
CP=\sqrt{\sum_iF_i\frac{\sum_jR_{ij}^2}{\sum_jR_{ij}/d}},
\quad \overline{CP}=CP/\sqrt d
=\sqrt{\sum_iF_i\frac{\sum_jR_{ij}^2}{\sum_jR_{ij}}}.
$$

I denominatori usano un pavimento 10^-12; anche il radicando ha quel pavimento,
quindi una rilevanza nulla produce CP normalizzata 10^-6. Frequenze e R bounded
mantengono il risultato al più 1. Il termine CP non è una probabilità di correttezza
della classe o del token, né una garanzia automatica di separazione delle classi.

## Una sola logica di training

La funzione comune `prediction_cp_objective` calcola:

$$
\mathcal L=\mathcal L_{prediction}+\lambda(1-\overline{CP}).
$$

Nel tabulare la previsione della classe c è la miscela
p(c|x)=sum_i T_i(x) B_ic, dove B contiene le probabilità categoriche apprendibili
dei prototipi. Si usa la NLL della classe corretta e λ=0,1; per più target
categorici si fa la media delle NLL. I riepiloghi usano soltanto il training.
Tutti i record di training partecipano, non soltanto il campione visualizzato.
Senza target rimane il termine CP; il nuovo profilo non implementa una NLL di
regressione. La classificazione di target numerici richiede la loro conversione
categorica mediante l'app o prima del training.

Nel generativo si applica la formulazione proposta dall'utente:

$$
\mathcal L_{LM}=-\frac1N\sum_t\log p(w_{t+1}|w_{\le t})
+\lambda\,\mathrm{media}_{\ell,t}(1-\overline{CP}_{\ell,t}).
$$

La media include batch e soli layer ACAN, indipendentemente da quanto sono
numerosi. Sono escluse le ultime posizioni senza target next-token nel blocco.
La CP di ogni posizione usa soltanto quel prefisso; i riepiloghi precedenti
mantengono le assegnazioni dei token passati, senza riassegnarli dopo aver visto
il futuro. La CP resta collegata al grafo di autograd. Non è calcolata da un float
staccato dal grafo o aggiunta soltanto al log.

La GUI propone λ=0,1 per i nuovi progetti del profilo congiunto, con controllo
esplicito λ=0. L'API mantiene il default 0 per i chiamanti precedenti. Il valore
ottimale linguistico non è stato determinato. Con distillazione la loss di
previsione è la combinazione NLL/KL già dichiarata, a cui viene aggiunta la CP.
Validazione, early stopping e checkpoint migliore linguistici usano la NLL separata.

## Adattatori, parametri e persistenza

| Aspetto | Tabulare interattivo | Generativo |
| --- | --- | --- |
| Input al nucleo | Min-max del training, target one-hot | Sigmoid delle rappresentazioni del token |
| Statistiche Root/min/max | Solo training, congelate per inferenza | Solo prefisso già letto, per layer |
| Concetti | Stessa equazione su batch di righe | Stessa equazione su un token e riepiloghi precedenti |
| Omega | Coupling stimato dal training e congelato | Parametro non negativo tramite valore assoluto |
| Ottimizzazione | Adam NumPy, gradienti analitici esatti | AdamW PyTorch, autograd |
| Lettura dell'output | Miscela delle probabilità dei prototipi | Stati di valore dei concetti, finestra locale causale, testa next-token |
| Budget di questa versione | Tutte le righe, batch ≤300, k≤256, epoche richieste | Su CPU blocco ≤32, k≤8; un passo del nucleo per token |

Le diverse codifiche e teste sono necessarie per i due compiti. Non sono due
formule di rilevanza o due definizioni di CP. La CP retropropaga fino ai prototipi
che determinano il routing in entrambi i percorsi. I valori dei parametri e
l'ottimizzatore possono differire senza cambiare le equazioni del modello comune.

Nel tabulare la predizione usa le stesse assegnazioni e i parametri congelati,
senza aggiornare riepiloghi sul test. Ogni figlio della gerarchia usa lo stesso
profilo, con routing basato soltanto sugli attributi. La necessità di approfondire
un ramo è decisa dalla purezza delle etichette osservate nei suoi record di
training; le probabilità apprese dei prototipi possono essere sicure anche su
rami misti e non vengono usate come sostituto della purezza osservata. Il JSON conserva temperatura,
prototipi, frequenze, rilevanza del routing, identificatore e obiettivo misurato.
Il campione visualizzato è limitato a 300 righe; la CP e le frequenze esportate
sono ricalcolate su tutte le righe di training.

Nel generativo configurazione, contratto, peso CP e metriche separate sono nei
checkpoint e nell'export. Il nuovo profilo richiede un nuovo progetto; la ripresa
non converte vecchi pesi in un diverso modello. I checkpoint dei profili precedenti
possono usare soltanto i loro contratti compatibili esatti.

## Verifiche e misure

`tests/test_supervised_cp.py` controlla gradienti con differenze finite, routing
senza target, persistenza delle predizioni, isolamento delle query e gerarchia.
`tests/test_lm_joint_cp.py` confronta lo stesso nucleo in NumPy e PyTorch, inclusi
gradienti della loss congiunta; verifica gradienti CP non nulli, assenza di
dipendenza dal futuro, media della penalità ed equivalenza streaming. I graft al
50% e al 100% sono entrambi coperti. Il trainer controllato è verificato anche
con λ=0,1 e gradient checkpointing: pausa/ripresa produce gli stessi pesi del
training non interrotto.

### Letter ricaricato

File `letter.arff`: SHA-256
`a31f2d9ad119a53e182e7371d48981982dbb2b0a5e9f0aa7b73f373857b64e2e`.
Si usa il file dell'utente, non la copia pubblica scaricata durante il recupero.
Preprocessing addestrato solo sul training; split stratificato seed42; 100 epoche,
lr=0,08, k=256, λ=0,1, radice. La selezione del checkpoint usa esclusivamente
l'obiettivo congiunto sul training. Nessun tuning o scelta della profondità sui
risultati held-out. Sono misure esplorative su un solo split prescritto.

| Protocollo | Modello | Training | Held-out | Accuracy | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| File completo | Radice | 16.000 | 4.000 | 86,38% | 86,31% |
| GUI seed42, campione 10.000 | Radice | 8.000 | 2.000 | 87,20% | 87,15% |
| File completo | Gerarchia, massimo 3 livelli | 16.000 | 4.000 | 91,43% | 91,40% |
| GUI seed42, campione 10.000 | Gerarchia, massimo 3 livelli | 8.000 | 2.000 | 90,60% | 90,49% |

Salvataggio/caricamento conserva le predizioni, e l'inferenza non modifica il
modello. La gerarchia è addestrata con massimo 3 livelli, soglia di purezza 0,92,
minimo 12 record, 100 epoche alla radice e 50/40 nei figli. Il file completo
produce 65 nodi, 7.391 prototipi e 2 livelli effettivi; il protocollo GUI produce
50 nodi, 2.817 prototipi e 2 livelli effettivi, arrestandosi quando gli split non sono
necessari o utilizzabili. Non si impone un terzo livello a rami già puri.
La struttura è scelta soltanto sui record di training, senza selezionare la
profondità in base all'accuracy held-out.

Il **95,78% precedente era un risultato multilivello gerarchico**, come precisato
dall'utente. La radice non è un confronto equivalente con quel risultato.
La ricostruzione attuale è stata quindi verificata anche con la gerarchia:
**95,78% e 93,90% non sono stati recuperati o riprodotti** e non vengono attribuiti
a questo codice. Il protocollo/configurazione esatti delle modifiche non committate
non erano presenti nel branch recuperato. Non si pubblicano record del dataset.
Dettagli: [JOINT-LETTER-RESULTS.json](JOINT-LETTER-RESULTS.json) e
[JOINT-HIERARCHY-RESULTS.json](JOINT-HIERARCHY-RESULTS.json).

### Confronto linguistico con sola NLL

Prova meccanica locale: Llama casuale, 2 layer, hidden16, vocabolario32, blocchi
sintetici distinti di 8 token; 32 blocchi training, 8 validazione, 8 test;
16 passi, stessi pesi iniziali e stessi batch per ogni coppia λ=0/0,1.
Non sono pesi di un LLM preaddestrato e non è un benchmark di qualità linguistica.

| Layer ACAN | λ | NLL validazione | NLL held-out |
| --- | ---: | ---: | ---: |
| 50% | 0 | 3,478660 | 3,470446 |
| 50% | 0,1 | 3,479396 | 3,471056 |
| 100% | 0 | 3,480135 | 3,469845 |
| 100% | 0,1 | 3,480203 | 3,469916 |

La CP è effettivamente ottimizzata e i gradienti sono finiti; in questa prova
**non migliora la NLL held-out**. Il confronto non determina λ ottimale o vantaggi
su modelli preaddestrati. Servono corpus e budget adeguati, più seed e selezione
sulla validazione prima di un test finale indipendente.
Dettagli: [JOINT-LM-SMOKE.json](JOINT-LM-SMOKE.json).

### Riproduzione

```sh
python -m pip install -e '.[dev,ui,llm]'
OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_joint_cp --letter /percorso/letter.arff
OPENBLAS_NUM_THREADS=1 python -m scripts.evaluate_joint_cp --letter /percorso/letter.arff --depth 3 --output reports/joint-hierarchy.json
python -m scripts.probe_joint_lm
python -m unittest discover -s tests -v
python scripts/check_paper.py --require-torch
python scripts/check_lm.py
```

La prova Letter richiede il file e non viene lanciata automaticamente dal normale
uso dell'app. Il gate LM include esplicitamente i controlli del nucleo congiunto.
