# Verifica dei due paper e dei dataset caricati

I due PDF allegati coincidono byte per byte con quelli congelati nel repository:
`Algebraic_Concept_Attention_Scientific_Paper.pdf` (SHA-256
`1fdd1a5f6e235d291c29364cae2dd1bd138df74db42256bb9ec36dd528f40396`) e
`ACAN_LM_vs_MiniCPM_Scientific_Paper.pdf` (SHA-256
`22df94993bdd71d03616375990ef8a4c05d1c1aa52f0d0711d07225dc3f5d65f`).
Non sono stati modificati per adattarli ai risultati del codice.

## Cosa è comune e cosa è stato corretto

Il paper tabulare, Eq. `v8_routing`, e il paper LM, Eqs. (1)–(2), condividono
la distanza quadratica pesata dalla rilevanza e la competizione fra slot con prior
logaritmico di massa. Sono ora operatori comuni in `acan/algebra.py`, utilizzati
sia dal nuovo adattatore LM sia dal profilo tabulare sperimentale.

- `paper-routing-v1` corregge il routing tabulare: distanza quadratica e due stadi
  con `gamma * log(mass_fraction + epsilon)`. Il codice precedente moltiplicava la
  massa per una similarità mista dot-product/distanza: non è la stessa formula.
- L'LM possedeva già quella forma di routing, con l'ulteriore termine bilineare
  previsto dal suo Eq. (2). `shared-algebra-v1` riusa gli operatori comuni e conserva
  output, gradienti e causalità. Il termine bilineare appartiene all'adattatore LM,
  non viene aggiunto arbitrariamente all'equazione tabulare.
- Il profilo tabulare viene salvato nei nodi JSON, insieme ad alpha/beta/gamma;
  predizione e approfondimenti mantengono quel profilo. Per l'inferenza le masse
  sono quelle del training: la valutazione non ricalcola i centroidi sui campioni
  da predire. La metrica usa soltanto gli attributi osservati.
- I default sperimentali alpha=2, beta=8, gamma=1.15 provengono dal pacchetto LM
  fornito. Sono una configurazione dichiarata per confrontare gli operatori,
  non una scelta dimostrata ottimale per ogni dataset tabulare.

Non sono due modelli sostitutivi scelti automaticamente in base al punteggio:
l'app mantiene il profilo attuale `legacy-v8` come default. Il routing del paper è
un esperimento esplicito nelle impostazioni avanzate, perché il confronto reale
ha mostrato regressioni e instabilità. I benchmark restano su richiesta.

## Cosa NON è ancora uniformato o certificato

Il profilo sperimentale corregge il routing, **non implementa l'intero paper**:

- conserva la rilevanza root-shift e la CP del codice tabulare fornito; il paper
  descrive anche una rilevanza duale shift/tightness e un termine di entropia;
- omega resta un accoppiamento fisso ricavato dai dati nel tabulare, mentre è
  apprendibile nell'LM. L'espansione Phi completa e il protocollo di selezione delle
  interazioni non sono specificati nel PDF al di fuori dell'esempio XOR; il
  coefficiente lambda_H non viene fornito. Non sono stati inventati;
- il linguaggio richiede statistiche causali sul prefisso e usa next-token CE,
  eventualmente KL del teacher; il grafo tabulare usa CP sui record del training.
  Questi obiettivi e i relativi punteggi non sono intercambiabili;
- i layer Transformer non costituiscono un albero padre–figlio fra gli slot.

Il test degli operatori verifica le formule, non i teoremi o le percentuali dei
paper. La conformità complessiva resta **parziale**, riportata dai gate e dai
manifest di release. Vedi anche `PAPER-AUDIT.md` e `LM-AUDIT.md`.

## Confronto reale eseguito qui

Sono stati confrontati i due routing sugli stessi file utente, con holdout 20%,
seed 7/21/42/84/123, uno stesso nodo iniziale, massimo 100 epoche, batch 300 e
identico budget/ottimizzatore. Target e filtri sono quelli suggeriti dall'app e
sono registrati nel JSON. Non sono state cercate combinazioni per far salire un
punteggio. Gli holdout si sovrappongono: sono ripetizioni esplorative, non cinque
prove indipendenti da cui ricavare intervalli di confidenza. I risultati sono
validazione, non un test finale dopo aver scelto il profilo.

Accuracy media sui cinque split:

| Dataset | Attuale | Solo routing del paper |
| --- | ---: | ---: |
| GiocoSet | 100.0% | 100.0% |
| Irrilevanti | 100.0% | 100.0% |
| AdultMinimal | 65.2% | 74.4% |
| Titanic | 78.3% | 54.1% |
| Anneal | 76.1% | 75.0% |
| Anneal.ORIG | 76.1% | 75.0% |
| Audiology | 40.0% | 23.5% |
| Autos | 30.2% | 43.9% |

Le due copie `anneal.arff` sono identiche. Anneal e Anneal.ORIG sono varianti dello
stesso dataset: non sono due prove indipendenti. Su Anneal l'accuracy del profilo
attuale coincide con la maggioranza: la balanced accuracy è appena 20%. Il nuovo
routing arriva mediamente a circa 20.2%: non risolve il problema. Su Titanic il
nuovo routing è molto instabile (deviazione standard dell'accuracy circa 28.8
punti); l'attuale circa 2.9 punti. Su AdultMinimal migliora anche la stabilità.

KNN, CART, Random Forest, Extra Trees, HistGradientBoosting, XGBoost e CatBoost sono
stati misurati separatamente sul solo seed 42. Non confrontare quei valori singoli
con le medie ACAN sui cinque split come se avessero lo stesso protocollo. I
risultati negativi e quelli delle baseline sono conservati in
`ROUTING-COMPARISON.json`, senza righe dei dataset. I precedenti benchmark del paper
usano dati sintetici/strutturati e protocolli diversi: non garantiscono le stesse
percentuali sui file caricati o sulla lingua naturale.

Ripetizione locale esplicita:

```bash
python scripts/compare_paper_routing.py file.arff file.xlsx file.csv \
  --output results.json --seeds 7 21 42 84 123 --baselines
```

Il comando non parte da solo nell'app. Le baseline vengono eseguite soltanto con
`--baselines`, sul seed 42; massimo cinque seed per invocazione. Il default del
benchmark semplice `scripts/evaluate_datasets.py` resta il profilo attuale; per
l'esperimento passare `--routing-profile paper-routing-v1`.

## Correzioni GUI e persistenza

ARFF: categorie decodificate da byte, spazi nominali Weka e quoting normalizzati,
valori mancanti riconosciuti secondo la convenzione corrente dell'app (`?`),
colonne presenza/assenza non scartate come costanti. Un caricamento con risposta
JSON corretta è verificato prima del training. Il pulsante attende che target e
filtri siano sincronizzati: prima un click molto rapido poteva avviare il training
con target vuoto, cambiando involontariamente il compito in non supervisionato.
Evaluation segnala quando viene predetta una sola classe, senza avviare baseline.

I modelli salvati mantengono il profilo; nessuna migrazione automatica modifica i
pesi. I checkpoint controllati PR6 e PR7 precedenti agli operatori condivisi sono
riconosciuti solo tramite ricevute sorgenti note e fingerprint completo di
configurazione/base/corpus/token. La continuazione viene verificata contro i pesi
esatti di un training non interrotto; una ricevuta legacy non viene accettata per
un nuovo progetto con operatori condivisi.

I gate richiedono formule letterali, differenze finite, parità NumPy/Torch,
causalità/streaming LM, compatibilità checkpoint e serializzazione del profilo.
Nessuna soglia di qualità negativa è stata rimossa per promuovere l'esperimento.
