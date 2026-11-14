# Confronto con i tre file storici allegati dopo la PR8

> **Documento storico.** Descrive una versione precedente del modello. Il modello attuale è in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md); dove questa pagina differisce, valgono quelle.


L'utente conferma che il precedente risultato >0,9 era **accuracy dalla matrice di
confusione**. Non viene reinterpretato come un indice di errore differente.
Questi tre file costituiscono una versione diversa dallo ZIP v8 già confrontato.

## Differenze effettive

| Passaggio | File storici | Percorso attuale legacy-v8 |
| --- | --- | --- |
| Routing nel forward attivo | softmax(C × sigmoid(G) × Xᵀ), competizione sui gruppi | miscela prodotto scalare/metrica pesata da omega, tau e sharpening |
| Inizializzazione | G/F casuali; chiamata initialize_G_from_data commentata nel learner | inizializzazione dai dati e accoppiamento omega |
| R usata dalla CP | dal solo centroide osservato TX/C | dalla miscela 0,5 TX/C + 0,5 sigmoid(G), con omega |
| Gruppi conservati | soltanto slot con almeno un T > 0,5 | assegnazione argmax per ogni record prima del pruning |
| Training | continuo fino allo stop, batch casuale a ogni iterazione, nessun best-state | budget/early stopping e ripristino best-state, campioni riproducibili |
| Predizione | singolo gruppo, distanza con R normalizzata per riga | soft nella foglia per default, hard per attraversare i rami |
| Gerarchia gLearn nei file | stub: restituisce soltanto la radice | espansione ricorsiva e approfondimento su richiesta |

`calculate_T_from_G` nel file storico è definito ma il suo uso nel forward è
commentato. Non va scambiato con l'algoritmo effettivamente eseguito. La maschera
delle label è registrata ma non viene usata dal forward per cambiare la CP.

Sono modifiche algoritmiche, non soltanto differenze di interfaccia o di backend.
Anche i due percorsi recenti paper-routing e CP diretta sono profili distinti,
come già dichiarato, e non sono una riproduzione bit per bit della versione storica.

## Prova eseguita

Sono state eseguite le definizioni delle classi originali senza modificarle, con
un harness che limita il ciclo infinito nLearn a budget dichiarati. TensorBoard,
JSONPath e i thread GUI non sono necessari al calcolo; i loro import opzionali
sono omessi dal loader AST, non sostituiti con algoritmi diversi.

Letter completo: 16.000 righe training, 4.000 validazione, split stratificato seed42,
seed NumPy/Torch42, CPU float32, batch300, Adam lr0,1. Target rimosso dagli input
di validazione dal metodo originale evaluateError. Sono state mantenute statistiche
root, campionamento a ogni passo, forward/gradienti e predizione originali.

| Prova | Passi | Accuracy | Gruppi conservati |
| --- | ---: | ---: | ---: |
| Originale, attributi numerici | 100 | 4,275% | 2 |
| Originale, attributi numerici | 1.000 | 4,275% | 2 |
| Originale, attributi numerici | 3.000 | 4,175% | 2 |
| Cambiata soltanto R/CP con miscela 50/50 | 100 | 3,800% | 1 |
| Cambiata soltanto R/CP con miscela 50/50 | 1.000 | 3,925% | 1 |
| Originale, tutti gli attributi resi categorici | 100 | 7,875% | 11 |
| Originale, tutti gli attributi resi categorici | 1.000 | 8,500% | 11 |

Il confronto di miscela conserva inizializzazione, sequenza dei batch, routing,
ottimizzatore e regola di predizione dell'originale. È un controllo isolato del
calcolo R/CP, non una riproduzione di tutte le differenze del codice corrente.
La prova categorica cambia il preprocessing ed è dichiarata separatamente.

Il peggioramento nella prova di miscela conferma che quella modifica può cambiare
la dinamica; **non spiega da sola il divario dal >90% storico**, perché il punto di
partenza qui riprodotto è anch'esso basso. Aumentare i passi non recupera quel
risultato in questa prova. La CP aumenta mentre l'accuracy resta bassa: il suo
valore non certifica la separazione delle classi.

Rimangono da identificare checkpoint/versione esatta, impostazioni, seed e durata
del risultato storico. Non è provato che i file forniti, con altri batch/seed/GPU,
non possano mai produrlo; non è stato riprodotto nel protocollo esplicito qui sopra.
Non è corretto correggere una riga e dichiarare recuperato il 90% senza misurarlo.

Hash dei tre file e risultati numerici in `HISTORICAL-VERSION-COMPARISON.json`.
Nessuna riga del dataset né i sorgenti allegati sono copiati nel report pubblico.
