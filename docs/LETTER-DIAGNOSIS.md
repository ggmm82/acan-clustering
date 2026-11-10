# Letter: diagnosi misurata nella PR8

File utente: 20.000 righe, 16 attributi numerici, target `class` con 26 lettere.
Il confronto usa tutte le righe: 16.000 training, 4.000 validazione, split
stratificato seed 42, preprocessing addestrato soltanto sul training. La GUI
campiona prima 10.000 righe, quindi i numeri non coincidono necessariamente
con la sessione web dell'utente. Nessuna riga del file è inclusa nel report.

| Confronto | Accuracy |
| --- | ---: |
| Profilo attuale, radice sola, soft | 8,325% |
| Stessa radice, singolo gruppo | 9,400% |
| Profilo attuale, profondità massima 3, soft | 11,900% |
| Stessa gerarchia, singolo gruppo | 11,650% |
| Solo routing del paper, radice | 4,925% |
| Nuovo CP diretto, radice | 12,350% |
| ZIP originale, NumPy, radice | 9,150% |
| ZIP originale, Torch CPU, radice | 9,225% |
| KNN 5 | 95,625% |
| Extra Trees 200 | 97,375% |

La radice attuale conserva solo 5 gruppi; la gerarchia di profondità 3 contiene
6 nodi e 24 gruppi complessivi. Cambiare predizione e approfondire migliora poco.
Il collasso/coarse-graining e l'allineamento fra obiettivo CP e separazione delle
classi sono quindi problemi sostanziali: non basta abilitare più livelli o un
altro criterio di voto. Cinque gruppi hard possono emettere al massimo cinque
etichette; una miscela soft può emetterne di più, quindi non le si applica lo
stesso limite formale. Anche la nuova CP diretta rimane insufficiente su Letter.

Il comportamento negativo è presente anche nel pacchetto v8 fornito, con entrambi
i backend. Non è dimostrato che il solo porting spieghi il divario dal risultato
storico >0,9. Per riprodurre quel risultato servono la versione e il protocollo
che lo produssero; non sono stati ricostruiti inventando un classificatore diverso
sotto il nome ACAN. Le formule del paper e gli operatori corretti non costituiscono
una garanzia di accuracy sui dati reali.

Il confronto ZIP/codice corrente usa batch 300, massimo 100 epoche, lr=0,08,
tau=0,2 e miscela alpha=0,5. Il nuovo profilo CP ha invece massimo 30 passi su T
e 32 slot, come dichiarato nell'app: non è lo stesso budget né lo stesso problema
ottimizzato. Le misure sono esplorative su una sola validazione, non un test finale
indipendente dopo la scelta di un profilo. Tempo e gruppi riportati nel JSON.

L'app ora segnala anche copertura parziale delle classi, numero di gruppi alla
radice e nodi del grafo, senza lanciare baseline automaticamente. L'unificazione
del solver CP tabulare/LM non viene presentata come risoluzione di questo problema.

Ripetizione esatta del campionamento GUI: 10.000 righe scelte con seed 42,
poi 8.000 training e 2.000 validazione stratificata. Radice soft **13,10%**,
radice hard **13,80%**; gerarchia fino a 3 livelli (9 nodi, 29 gruppi) soft
**17,20%**, hard **17,35%**. Anche con il protocollo dell’app entrambi i cambi
migliorano poco e non riproducono il risultato storico >90%. Questi numeri
non vanno confusi con il confronto sul file completo nella tabella precedente.
