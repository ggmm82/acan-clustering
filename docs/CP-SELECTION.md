> Aggiornamento: il modello corrente usa la CP con sostegno fra osservazioni diverse.

> **Documento storico.** Descrive una versione precedente del modello. Il modello attuale è in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md); dove questa pagina differisce, valgono quelle.

> Recupera quattro gruppi su tutte le 101 righe di Gioco senza target.
> Formula e nuove misure: [CP-PAIR-SUPPORT.md](CP-PAIR-SUPPORT.md).
> Questa pagina conserva la diagnosi delle versioni precedenti.

# CP condivisa: proposta sperimentale e verifiche

Un unico nucleo `cp-joint-v1` con definizione `exclusive-cp-v1` è proposto per nuovi modelli tabulari e generativi. Le vecchie implementazioni rimangono controlli di regressione e supporto ai checkpoint; non sono nuove opzioni di modello nella GUI. La PR è destinata alla prova web, non certifica un miglioramento universale.

## Base matematica e differenze dagli studi

Conserviamo prototipi, frequenze, rilevanza rispetto alla radice e CP normalizzata dei documenti in `paper/`. La CP originale è neutra rispetto alla duplicazione di un concetto: dividere F fra copie con la stessa R non cambia la somma. La proposta introduce una sola estensione esplicita della sua funzione obiettivo, condivisa dai due adattatori. Non è una conseguenza già dimostrata dei teoremi dei paper.

Posto `q_i = sum(R_i²) / sum(R_i)` e `z_i = F_i sqrt(q_i)`, definiamo `S_ii=0` e `S_ij=exp[-beta(D_ij+D_ji)/2]`, con la stessa distanza dei prototipi usata dal routing. Le masse apprese pi sono sul simplesso e il routing è `T_it ∝ pi_i exp(-beta D_it)`. La funzione è:

$$
CP_* = \left(\sum_i\sqrt{\pi_iF_i}\right)\sqrt{\sum_iF_iq_i-z^TSz}.
$$

La somma esterna è l’affinità di Bhattacharyya: impedisce che una massa a priori quasi nulla rappresenti arbitrariamente un concetto frequentemente usato. Non viene interpretata come probabilità della classe. Il termine di sovrapposizione penalizza copie dei concetti. Entrambi i fattori sono limitati: per Cauchy–Schwarz il primo non supera uno; il radicando è fra zero e `sum(F_i q_i)`. Con r copie identiche, frequenze uniformi F_i=1/r e pi=F, CP* vale `sqrt(q/r)` invece di `sqrt(q)`. Queste proprietà sono dimostrazioni del criterio; non sono una prova di recupero del K semanticamente vero su dati arbitrari.

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
secondo modello generativo. La prima tabella registra la proposta iniziale; la sezione successiva confronta
le correzioni con gli stessi split e con uno split aggiuntivo.

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

Tutte le 63 esecuzioni conservano le predizioni dopo salvataggio/caricamento. Il vecchio controllo automatico di non mutazione serializzava nuovamente un oggetto non interrogato: quel flag non costituisce una prova. Lo script ora interroga direttamente l’istanza ricaricata; un test separato verifica la gerarchia, le probabilità, i percorsi e l’assenza di mutazioni. Adult è il file **minimale da 499 righe**, non il dataset Adult completo. Su Breast Cancer il legacy raggiunge 70,69% predicendo la classe maggioritaria: balanced accuracy 50%; la proposta misura 58,06% balanced, contro 60,09% del joint precedente. Restano regressioni effettive, soprattutto rispetto al joint su Breast Cancer e al legacy su Adult minimale.

Vote contiene una riga senza attributi osservati. La verifica non la elimina: usa il preprocessing nativo già appreso dal training. È emerso un errore del controllo preliminare della GUI, corretto nella PR. Le ricevute annotano i casi che richiedevano l’API nativa nel codice precedente.

Gioco supervisionato: quattro concetti sulle 100 righe con target osservato e sui cinque split seed 1/7/21/42/99; accuratezza test 100% nei controlli eseguiti. La riga con target `?` è esclusa dal training. Una prova sintetica indipendente cambia le date e controlla le quattro relazioni senza passare il numero atteso al modello.

Letter gerarchico, seed42: **92,025%** sulle 4.000 righe di test, contro **91,425%** del joint precedente; prototipi totali **4.339** contro **7.391**. Sul campione GUI 10.000→8.000/2.000, invece, **89,00%** contro **90,60%**. Il precedente **95,78% non è stato recuperato**. Non si presenta il risultato sul file completo come prova di assenza di regressioni.

Ricevute aggregate senza righe dei dati: [63 confronti](results/cp-generalization.json), [Letter](results/cp-letter.json). Includono hash dei file e dei sorgenti sperimentali congelati, split, accuratezza bilanciata, macro F1, tempi e conteggi. Le successive modifiche della PR alla GUI, ai controlli di input, ai metadati e al costo CP del solo decoding non cambiano la formula tabulare misurata.

## Verifica delle regressioni dopo le correzioni

Il commit `46b88ce` corregge l'inizializzazione della softmax e conserva la miscela
probabilistica lungo i rami. La CP*, lambda, il criterio per le masse e i parametri
di training restano gli stessi per tutti i file. Medie sugli stessi tre split:

| Dataset | CP* iniziale | Dopo le correzioni |
|---|---:|---:|
| iris.arff | 96.67% | 95.56% |
| sonar.xlsx | 84.13% | 84.13% |
| adultMinimal.xlsx | 69.67% | 72.67% |
| titanic.csv | 76.16% | 81.19% |
| vote.arff | 93.49% | 92.34% |
| mushroom.arff | 100.00% | 100.00% |
| breast-cancer.arff | 66.67% | 67.24% |

Letter completo: **92,70%**, 3.703 prototipi e 62 nodi,
contro 92,025%, 4.339 prototipi e 68 nodi della prima proposta. Letter nel campione
web 10.000→8.000/2.000: **89,85%**, 1.701 prototipi,
contro 89,00% della prima proposta e 90,60% della PR8. Il 95,78% storico resta
non recuperato. Gioco supervisionato conserva quattro concetti e 100% sui cinque split; ripetuta
anche la prova Excel → training → quattro gruppi → salvataggio nel browser.

Una verifica diagnostica sui medesimi modelli già addestrati, cambiando soltanto
la lettura della gerarchia, porta Adult da 69,67% a 73,67% e la NLL media da 2,453
a 0,614; Letter web da 89,00% a 89,80%, NLL da 1,109 a 0,333. Questo isola
l'incoerenza fra miscela appresa e scelta rigida dei rami. Non è il risultato
del nuovo training completo, che include anche l'inizializzazione corretta.

Le correzioni rispondono a regressioni viste su questi test: riusarli serve alla
diagnosi, non costituisce una verifica finale indipendente. Dopo aver fissato
le correzioni è stato aggiunto seed99, sugli stessi file e con gli stessi parametri:

| Dataset | Prima, seed99 | Dopo, seed99 |
|---|---:|---:|
| iris.arff | 96.67% | 96.67% |
| sonar.xlsx | 78.57% | 73.81% |
| adultMinimal.xlsx | 69.00% | 80.00% |
| titanic.csv | 79.33% | 78.21% |
| vote.arff | 93.10% | 93.10% |
| mushroom.arff | 100.00% | 100.00% |
| breast-cancer.arff | 68.97% | 68.97% |

Adult migliora anche qui; Titanic e Sonar non confermano il miglioramento.
Breast Cancer mantiene la stessa accuracy, ma la balanced accuracy scende da
62,55% a 60,83%. Sui primi tre split la balanced accuracy di Breast Cancer passa
da 58,06% a 58.46%, ancora sotto 60,09% della PR8.
Iris e Vote perdono parte del vantaggio della prima proposta. Questi limiti
restano aperti: le correzioni non dimostrano una superiorità universale, né una
qualità linguistica migliore. Nessuna soglia o numero di concetti è stato scelto
in funzione del singolo dataset.

[35 esecuzioni aggiuntive, Letter e diagnosi della gerarchia](results/cp-regressions.json).
126 test locali passati; controlli browser delle formule su 550/1440 px e di
Gioco superati. La CI del commit con il codice `46b88ce` ha superato anche i
controlli generativi, Docker e la pubblicazione della build verificata.
Le ricevute storiche distinguono ora il solo roundtrip GUI dal controllo
sull'istanza ricaricata: `null` indica che la non mutazione non era verificata
direttamente in quella misura. Il nuovo controllo e il test della gerarchia
interrogano l'oggetto ricaricato e ne confrontano lo stato prima/dopo.

## Clustering non supervisionato: controllo sul file originale

La presenza del target cambia il problema: NLL e pesi degli attributi usano le
etichette nel percorso supervisionato. Il risultato di classificazione non va
presentato come prova del clustering non supervisionato. Senza target la NLL
è zero e viene ottimizzata soltanto la CP; una colonna Gioco presente nel file
è allora un attributo osservato come gli altri, non una risposta da predire.

Il confronto fra implementazioni, con stessa preparazione e 100 epoche, chiarisce
il risultato dichiarato dall'utente:

| Dati, sempre senza target | Righe | Controllo precedente | CP attuale |
|---|---:|---:|---:|
| File originale, tutte le colonne e la riga con `?` | 101 | 4 | **5** |
| Righe con esito osservato, tutte le colonne | 100 | 4 | 4 |
| Split di training 80/20 della GUI sul file originale | 80 | 2 | 3 |
| Righe complete, ma colonna Gioco rimossa | 100 | 2 | 14 |

**La regressione sulle 101 righe resta aperta.** Anche prolungando lo stesso training a 300 e 1.000 epoche rimangono cinque gruppi; aumentare semplicemente il budget non risolve questa prova. Il controllo precedente è il
profilo legacy-v8 conservato nel repository. Escludere la riga incompleta o
confrontare cinque colonne con sei non risolve quella regressione. La prima
diagnosi aveva rimosso la colonna Gioco: viene conservata solo come confronto
fra problemi diversi, non come prova contro il risultato originale.

Il valore mancante viene codificato come categoria distinta. Una prova separata
con imputazione tramite frequenze categoriche ha comunque prodotto cinque gruppi:
non è stata adottata e non si attribuisce il problema soltanto alla codifica.
Non sono stati imposti quattro cluster, aggiunte fusioni o scartate righe per
far tornare il conteggio. [Ricevute degli otto controlli](results/cp-unsupervised.json).

La causa matematica della proliferazione non è eliminata dalla penalità dei
duplicati: per prototipi separati S è quasi zero e, a parità di qualità e masse
coerenti, CP* non penalizza direttamente K. La NLL può favorire frammenti che
memorizzano le etichette; nella gerarchia ogni nodo ha un proprio obiettivo e
l'espansione dipende anche da purezza/variabilità, righe minime e profondità.
Non c'è una penalità globale per la dimensione dell'albero.

Le correzioni riducono i prototipi di Letter completo da 4.339 a 3.703 e del
campione web da 2.036 a 1.701. Non è una riduzione universale: per esempio Adult
passa dall'intervallo 354–395 a 370–444 e Mushroom da 49–121 a 71–155 nei tre split.
Resta da trovare e verificare un criterio che concili la qualità dei concetti,
la robustezza a missingness/rumore e la complessità, senza un K scelto sul test.

La scheda Matematica ora espone un racconto autonomo di questi concetti, con
motivazioni, esempi e limiti; i riferimenti tecnici e ai paper restano in questo
rapporto. Il [piano della prossima PR dell'interfaccia](NEXT-UI-PR.md) riguarda un
run successivo: compiti espliciti, classi numeriche, parametri effettivi, lettura
dei concetti e navigazione dello storico generativo oltre 500 punti.

## Efficienza e portata

La marginalizzazione tabulare valuta tutti i rami; usa routing vettorizzato e blocchi di 256 osservazioni, senza tagliare probabilità piccole con soglie. Il costo cresce con i nodi della gerarchia. Le nuove prove sono state eseguite in concorrenza: i tempi grezzi nelle ricevute non sono un confronto controllato di velocità.

Il training è più costoso: CP* aggiunge un confronto O(k²d), oltre al routing O(nkd), alle ricerche di passo e ai controlli strutturali. Non si rivendica un miglioramento generale del tempo di training. La riduzione dei prototipi salvati è misurata. In decoding linguistico il confronto fra prototipi per la loss viene saltato, con output identico verificato: costo per token O(kd+wd) e memoria dello stato indipendente dalla lunghezza del contesto, a budget k/w fissati.

Per il generativo sono verificati valori/gradienti comuni, causalità, streaming, supporto sparso, checkpoint, ripresa esatta e controllo dello stato casuale. **Non è stato dimostrato un miglioramento della qualità linguistica su un LLM preaddestrato.** I nuovi contratti impediscono di caricare silenziosamente vecchi checkpoint LM con formule diverse. Un modello tabulare joint precedente resta utilizzabile per inferenza ma va riaddestrato prima di aggiungere rami con la nuova formula.

## Riproduzione e prova web

Usare l’immagine/checkout della PR e avviare un nuovo training: un JSON precedente non viene convertito. Caricare GiocoSet.xlsx, target Gioco, lasciare 100 epoche e il modello CP unico. Verificare i quattro gruppi e salvare il JSON. Per ripetere Letter gerarchico scegliere tre livelli iniziali; il livello predefinito resta uno.

```sh
OPENBLAS_NUM_THREADS=1 python -m pytest -q
python scripts/check_lm.py --output reports/lm-audit.json
python -m scripts.evaluate_joint_cp --letter letter.arff --depth 3 --output reports/letter.json
python scripts/verify_cp_generalization.py --baseline /path/to/PR8-checkout --output reports/cp-comparison.json titanic.csv adultMinimal.xlsx sonar.xlsx vote.arff mushroom.arff iris.arff breast-cancer.arff
python -m scripts.evaluate_cp_clustering GiocoSet.xlsx --compare-legacy --output reports/gioco-all.json
python -m scripts.evaluate_cp_clustering GiocoSet.xlsx --compare-legacy --complete-columns Gioco --output reports/gioco-complete.json
python -m scripts.evaluate_cp_clustering GiocoSet.xlsx --compare-legacy --gui-split --output reports/gioco-gui.json
python -m scripts.evaluate_cp_clustering GiocoSet.xlsx --compare-legacy --complete-columns Gioco --drop-columns Gioco --output reports/gioco-without-column.json
```

Per il confronto prima/dopo le correzioni, usare il checkout `fb62fb4` come `--baseline` e ripetere con `--seeds 42 7 21` oppure `--seeds 99`; il record `joint` di quel checkout rappresenta la prima proposta CP*, oltre al controllo legacy.

Il confronto usa i profili storici soltanto come controlli. Le prove web devono verificare anche regressioni e tempi, non soltanto il numero di gruppi.
