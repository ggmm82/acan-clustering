# Piano di revisione: il grafo dei concetti senza cicli (prossima PR)

Punto di partenza: main alla versione 8.6.0 (PR #16), in linea su
https://acan.srv1757745.hstgr.cloud/. Misure e matematica:
[ALGEBRAIC-CONCEPTS.md](ALGEBRAIC-CONCEPTS.md).

## Che cosa c'è oggi in linea

- Apprendimento lungo del modello nativo a concetti, coda, salvataggi, ripresa,
  domande di prova, dati opzionali (codice, bash, musica, immagini, suoni, video),
  schede Matematica e Informatica. Nessun riferimento a SmolLM2 nella pagina.
- Il limite è dichiarato nella scheda Informatica: lo strato unisce i concetti con
  un ciclo in sequenza ed è da 10 a 25 volte più lento dell'attenzione normale sui
  testi corti. Misurato qui: 330-510 token/s in addestramento sul modello micro.
- Da correggere: la riga di stato dice ancora «profilo di verifica 135M» (residuo
  del vecchio modello di base).

## Obiettivo della PR

Sostituire il ciclo con la struttura fissa di `acan/lm/graph_scan.py` **solo se**
la qualità non scende. Nessun ibrido: tutti i layer leggono dal grafo.

## Criteri di accettazione (misurati, non stimati)

| Misura | Soglia |
|---|---|
| Addestramento, modello micro, 256-2.048 token | almeno 4 volte il main |
| Generazione per layer | almeno 2 volte il main |
| Memoria vicina (prova in piccolo) | non sotto il main (94%) |
| **Memoria lontana** | non sotto il main (93%) — oggi 24%: **blocca l'adozione** |
| Testo, bit per carattere | non sopra il main (2,538) |
| Concetti per lettore | al massimo `bins · (log₂ n + 1)`, per costruzione |
| Addestramento = generazione | differenza sotto 10⁻⁶ in precisione singola |
| Tutte le misure | su 3 semi, non uno |
| Progetti già salvati | restano leggibili (contratto dei checkpoint invariato) |

## Passi

1. **Memoria lontana** (prima di tutto il resto). Ipotesi da misurare, in ordine:
   a. più concetti nei blocchi grandi (`bins` che cresce con il livello: la
      memoria resta logaritmica);
   b. i token letti restano esatti: massa di lettura ricevuta nella finestra come
      peso della partizione (il criterio del lettore del main, ma causale e in
      forma chiusa);
   c. più passi di Lloyd e inizializzazione migliore con molti concetti.
   Se nessuna raggiunge il 93% su 3 semi, la PR si ferma qui e il main resta.
2. **Ripetere tutte le misure su 3 semi**, compresa la prova «full» del laboratorio.
3. **Strato per il modello nativo**: una classe nuova con il confine di
   `ACANConceptAttentionLayer` (stessi nomi dei pesi, cache per la generazione),
   profilo nuovo `concept-graph-v2`. I file del contratto non si toccano; i
   progetti esistenti continuano a usare `concept-graph-v1`.
4. **Precisione**: verificare addestramento = generazione in float32 e bf16 (un
   segno vicino a zero può cambiare la partizione).
5. **GPU**: stessa misura di velocità su GPU, dove l'attenzione normale è molto
   ottimizzata.
6. **Interfaccia**: scelta del profilo alla creazione della coda, quarto modellino
   «grafo» nella prova in piccolo, token al secondo fra le grandezze mostrate,
   testo della scheda Informatica aggiornato, riga «135M» corretta.
7. **Tabella**: il modello tabulare resta `ConceptModel`. `CUConceptModel` resta
   prototipo; la sola unificazione in questa PR è riusare le stesse funzioni di
   nodo, lettura e partizione. Sostituirlo è una PR successiva, con criterio:
   stesse soglie, tempo e numero di concetti non peggiori del main su Letter.
8. **Documenti**: MODEL.md, GENERATIVE.md, scheda Matematica.

## Che cosa non entra

- Nessuna modifica ai file del contratto dei checkpoint.
- Nessun cambio del modello tabulare in produzione.
