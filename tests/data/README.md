# Dati pubblici per i test di regressione

Tabelle pubbliche usate da `tests/test_quality_regression.py`, per controllare
in CI la qualità del modello su dati reali.

| File | Origine | Licenza |
|---|---|---|
| `sonar.csv` | UCI Machine Learning Repository, «Connectionist Bench (Sonar, Mines vs. Rocks)» (Sejnowski e Gorman) | CC BY 4.0 |
| `vehicle.arff` | UCI Machine Learning Repository, «Statlog (Vehicle Silhouettes)»; Turing Institute, Glasgow (Siebert, 1987) | CC BY 4.0 |
| `vote.arff` | UCI Machine Learning Repository, «Congressional Voting Records» (Schlimmer, 1987) | CC BY 4.0 |

`sonar.csv` contiene gli stessi valori della tabella UCI, con le colonne
numerate da 1 a 60 e la classe in `Class`. I file ARFF sono le copie di OpenML,
con le note originali in testa.
