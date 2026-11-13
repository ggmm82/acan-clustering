# Verifica sui dataset forniti

> **Documento storico.** Descrive una versione precedente del modello. Il modello attuale è in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md); dove questa pagina differisce, valgono quelle.


Misure locali CPU con ACAN nativo, configurazione iniziale GUI: un livello, 100 epoche, learning rate 0,08, tau 0,2, alpha 0,5, batch 300. Split 80/20 stratificato quando possibile, seed 42. Nessuna ricerca dei parametri. Le percentuali descrivono un singolo split, non una garanzia di generalizzazione.

| Dataset | Training / validazione | ACAN accuracy | KNN | Migliore baseline misurata |
|---|---:|---:|---:|---|
| irrilevanti.xlsx | 800 / 200 | 100.00% | 80.50% | CART 100.00% |
| sonar.xlsx | 166 / 42 | 71.43% | 73.81% | XGBoost 92.86% |
| titanic.csv | 712 / 179 | 77.65% | 81.01% | HistGradientBoosting 82.68% |
| GiocoSet.xlsx | 80 / 20 | 100.00% | 100.00% | KNN 100.00% |
| adultMinimal.xlsx | 399 / 100 | 80.00% | 79.00% | HistGradientBoosting 81.00% |

Tutti i sette stimatori di confronto (KNN, CART, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost) hanno completato tutti e cinque i dataset senza errori. Metriche, tempi, versioni e protocollo completi sono nel JSON adiacente; non contiene righe o predizioni dei dataset.

## Correzioni e limiti

- Titanic: target suggerito `Survived`, interpretato come classificazione; esclusione modificabile di `PassengerId`, `Name`, `Ticket` e `Cabin` (alta cardinalità rispetto ai valori osservati). Training: 14 colonne codificate, sotto i limiti interattivi.
- GiocoSet: un target `?` escluso esplicitamente dal training e dalla validazione; non è una terza classe.
- Colonne stringa riconosciute anche con pandas 3; misure numeriche conservate come tali. La modalità target può essere forzata a classificazione o regressione nelle impostazioni avanzate.
- Inferenza dei nuovi modelli GUI con schema appreso sul solo training: livelli categorici assenti dal batch valorizzati a zero, imputazione numerica e indicatori di dati mancanti coerenti con il training. Schema incluso nel JSON salvato. I vecchi modelli senza schema mantengono il percorso precedente: riaddestrare per beneficiare della correzione.
- Nessuna modifica alle formule, all’obiettivo, ai prototipi o al routing nativo ACAN. Nessun aumento dei limiti di memoria; diagnostica preventiva prima della matrice one-hot.
- Il confronto completo resta su richiesta. La validazione viene riutilizzata per decidere gli approfondimenti: non va trattata come test finale indipendente dopo tali scelte.
- Su Sonar ACAN resta sotto XGBoost (92,86%); questa correzione non dimostra una superiorità di ACAN.

I file originali restano fuori dal repository. Per ripetere localmente, usare lo script `scripts/evaluate_datasets.py` con i propri percorsi.
