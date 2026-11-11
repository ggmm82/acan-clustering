# CP condivisa: proposta sperimentale e verifiche

Un unico nucleo `cp-joint-v1` con definizione `exclusive-cp-v1` è proposto per nuovi modelli tabulari e generativi. Le vecchie implementazioni rimangono controlli di regressione e supporto ai checkpoint; non sono nuove opzioni di modello nella GUI. La PR è destinata alla prova web, non certifica un miglioramento universale.

## Base matematica e differenze dagli studi

Conserviamo prototipi, frequenze, rilevanza rispetto alla radice e CP normalizzata dei documenti in `paper/`. La CP originale è neutra rispetto alla duplicazione di un concetto: dividere F fra copie con la stessa R non cambia la somma. La proposta introduce una sola estensione esplicita della sua funzione obiettivo, condivisa dai due adattatori. Non è una conseguenza già dimostrata dei teoremi dei paper.

Posto `q_i = sum(R_i²) / sum(R_i)` e `z_i = F_i sqrt(q_i)`, definiamo `S_ii=0` e `S_ij=exp[-beta(D_ij+D_ji)/2]`, con la stessa distanza dei prototipi usata dal routing. Le masse apprese pi sono sul simplesso e il routing è `T_it ∝ pi_i exp(-beta D_it)`. La funzione è:

$$
CP_* = \left(\sum_i\sqrt{\pi_iF_i}\right)\sqrt{\sum_iF_iq_i-z^TSz}.
$$

La somma esterna è l’affinità di Bhattacharyya: impedisce che una massa a priori quasi nulla rappresenti arbitrariamente un concetto frequentemente usato. Non viene interpretata come probabilità della classe. Il termine di sovrapposizione penalizza copie dei concetti. Entrambi i fattori sono limitati: per Cauchy–Schwarz il primo non supera uno; il radicando è fra zero e `sum(F_i q_i)`. Con r copie identiche e pi=F, CP* vale `sqrt(q/r)` invece di `sqrt(q)`. Queste proprietà sono dimostrazioni del criterio; non sono una prova di recupero del K semanticamente vero su dati arbitrari.

L’obiettivo resta `NLL + lambda*(1-CP*)`, con lambda=0,1 nel tabulare e configurabile nel linguaggio. Il linguaggio media sui layer ACAN e sui token con target successivo, usando soltanto il prefisso osservato. La normalizzazione, il routing, la CP e le operazioni sul simplesso sono le stesse funzioni NumPy/PyTorch in `acan/algebra.py`. Codifica e teste di previsione restano adattatori distinti.

Le masse sono ottimizzate sul simplesso tramite proiezione e passi per coordinate sui suoi estremi, accettati dallo stesso obiettivo di training. Le direzioni usano la sovrapposizione già definita da S. Non si fondono i prototipi, non si richiede K=4 e non si usano soglie di similarità. Solo le masse esattamente zero escono dal modello tabulare salvato; nel linguaggio sono mascherate. Il budget di slot è un limite superiore. Gli ottimizzatori e i budget numerici non sono identici bit per bit fra gli adattatori.

## Precisazioni teoriche emerse

- Nel teorema gravitazionale del paper, per slot coincidenti vale `log(Ca/Cb)_next = gamma*log(Ca/Cb)`. Il punto simmetrico è instabile per **gamma>1**; gamma=1 è neutro. La formulazione gamma>=1 richiede questa precisazione.
- Sul rumore uniforme in [0,1], sigma=1/sqrt(12): `max(0,1-2*sigma)` è circa **0,423**, non zero. L’asserzione di decadimento della rilevanza duale non segue da quel termine senza ulteriori ipotesi/correzioni.
- Il nucleo joint precedente non conteneva il routing gravitazionale completo né l’espansione esplicita Phi(X). La proposta CP* non rivendica di averli ripristinati o di dimostrare i teoremi complessivi. Il passaggio alla selezione delle masse è un’estensione sperimentale dichiarata.

## Protocollo e risultati

### Correzioni successive alla prima proposta

La testa categorica inizializza ora i logits con `log(p)`, inversa della softmax,
anziché con i log-odds della sigmoid: il precedente codice trasformava un seme
binario 0,99/0,01 in circa 0,9999/0,0001, introducendo confidenza artificiale.
La funzione obiettivo e il nucleo CP non cambiano.

I nuovi modelli marginalizzano anche i rami della gerarchia:
`p_node(y|x) = sum_i T_i(x) p_i(y|x)`, dove `p_i` è la testa del concetto
terminale oppure la previsione del figlio. Prima, la sola presenza di un figlio
faceva abbandonare la miscela e scegliere un unico ramo. Una sostituzione con un
figlio dalla stessa distribuzione ora lascia invariata la previsione. Il percorso
mostrato è quello con maggiore probabilità congiunta; la sua probabilità non è
una confidenza calibrata della classe. I JSON precedenti conservano il loro
comportamento grazie al metadato `hierarchy_readout`; riaddestrare per usare la
correzione. È una correzione dell'adattatore gerarchico, non una nuova CP o un
secondo modello generativo. La tabella seguente registra la prima proposta;
i confronti dopo queste correzioni sono in corso.

Parametri fissati prima dei confronti: split stratificato 80/20, seed 42/7/21, profondità massima 3, 100 epoche, lr=0,08, batch massimo 300, purezza 0,92 e minimo 12 righe per espandere un ramo. Statistiche, codifica e imputazione vengono apprese sul training. Nessun parametro è scelto in base ai risultati del test. Sono confrontati legacy-v8, il precedente joint della PR8 e la proposta. Tre split ripetuti non equivalgono a tre dataset indipendenti; le medie sono esplorative.

| Dataset | Legacy CP | Joint precedente | Proposta CP* |
|---|---:|---:|---:|
| iris.arff | 95.56% | 94.44% | 96.67% |
| sonar.xlsx | 72.22% | 81.75% | 84.13% |
| adultMinimal.xlsx | 76.33% | 68.00% | 69.67% |
| titanic.csv | 64.25% | 76.91% | 76.16% |
| vote.arff | 83.14% | 92.34% | 93.49% |
| mushroom.arff | 88.70% | 100.00% | 100.00% |
| breast-cancer.arff | 70.69% | 69.54% | 66.67% |

Tutte le 63 esecuzioni conservano le predizioni dopo salvataggio/caricamento e non modificano il modello in inferenza. Adult è il file **minimale da 499 righe**, non il dataset Adult completo. Su Breast Cancer il legacy raggiunge 70,69% predicendo la classe maggioritaria: balanced accuracy 50%; la proposta misura 58,06% balanced, contro 60,09% del joint precedente. Restano regressioni effettive, soprattutto rispetto al joint su Breast Cancer e al legacy su Adult minimale.

Vote contiene una riga senza attributi osservati. La verifica non la elimina: usa il preprocessing nativo già appreso dal training. È emerso un errore del controllo preliminare della GUI, corretto nella PR. Le ricevute annotano i casi che richiedevano l’API nativa nel codice precedente.

Gioco: quattro concetti sul file completo e sui cinque split seed 1/7/21/42/99; accuratezza test 100% nei controlli eseguiti. La riga con target `?` è esclusa dal training. Una prova sintetica indipendente cambia le date e controlla le quattro relazioni senza passare il numero atteso al modello.

Letter gerarchico, seed42: **92,025%** sulle 4.000 righe di test, contro **91,425%** del joint precedente; prototipi totali **4.339** contro **7.391**. Sul campione GUI 10.000→8.000/2.000, invece, **89,00%** contro **90,60%**. Il precedente **95,78% non è stato recuperato**. Non si presenta il risultato sul file completo come prova di assenza di regressioni.

Ricevute aggregate senza righe dei dati: [63 confronti](results/cp-generalization.json), [Letter](results/cp-letter.json). Includono hash dei file e dei sorgenti sperimentali congelati, split, accuratezza bilanciata, macro F1, tempi e conteggi. Le successive modifiche della PR alla GUI, ai controlli di input, ai metadati e al costo CP del solo decoding non cambiano la formula tabulare misurata.

## Efficienza e portata

Il training è più costoso: CP* aggiunge un confronto O(k²d), oltre al routing O(nkd), alle ricerche di passo e ai controlli strutturali. Non si rivendica un miglioramento generale del tempo di training. La riduzione dei prototipi salvati è misurata. In decoding linguistico il confronto fra prototipi per la loss viene saltato, con output identico verificato: costo per token O(kd+wd) e memoria dello stato indipendente dalla lunghezza del contesto, a budget k/w fissati.

Per il generativo sono verificati valori/gradienti comuni, causalità, streaming, supporto sparso, checkpoint, ripresa esatta e controllo dello stato casuale. **Non è stato dimostrato un miglioramento della qualità linguistica su un LLM preaddestrato.** I nuovi contratti impediscono di caricare silenziosamente vecchi checkpoint LM con formule diverse. Un modello tabulare joint precedente resta utilizzabile per inferenza ma va riaddestrato prima di aggiungere rami con la nuova formula.

## Riproduzione e prova web

Usare l’immagine/checkout della PR e avviare un nuovo training: un JSON precedente non viene convertito. Caricare GiocoSet.xlsx, target Gioco, lasciare 100 epoche e il modello CP unico. Verificare i quattro gruppi e salvare il JSON. Per ripetere Letter gerarchico scegliere tre livelli iniziali; il livello predefinito resta uno.

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/check_lm.py --output reports/lm-audit.json
python -m scripts.evaluate_joint_cp --letter letter.arff --depth 3 --output reports/letter.json
python scripts/verify_cp_generalization.py --baseline /path/to/PR8-checkout --output reports/cp-comparison.json titanic.csv adultMinimal.xlsx sonar.xlsx vote.arff mushroom.arff iris.arff breast-cancer.arff
```

Il confronto usa i profili storici soltanto come controlli. Le prove web devono verificare anche regressioni e tempi, non soltanto il numero di gruppi.
