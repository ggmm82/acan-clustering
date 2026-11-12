# Prossima PR: un percorso comprensibile per usare e verificare ACAN

Piano per un **run successivo**. Questa PR non ridisegna l'interfaccia e non
modifica i progetti generativi in esecuzione. Destinazione:
https://acan.srv1757745.hstgr.cloud/ . L'11 novembre 2026 sono stati letti soltanto
la pagina pubblica e il layout Dash; nessun training remoto è stato avviato,
fermato o modificato. Il piano considera anche lo screenshot fornito dall'utente.

## Obiettivo e vincoli

Una persona deve capire quale compito sta eseguendo, quali dati il modello può
vedere, cosa ha imparato e quali controlli servono prima di fidarsi del risultato.
Rimane un solo nucleo CP. Preset di risorse o modalità di uso non devono diventare
algoritmi alternativi, né nascondere una selezione supervisionata dei concetti.

La semplificazione dell'interfaccia non risolve matematicamente il numero di
prototipi. Il clustering di Gioco sulle 101 righe complete è ora recuperato: la CP con
sostegno fra osservazioni diverse trova quattro gruppi senza target. Conservare
questo controllo nella prossima PR. Non introdurre un K imposto, fusioni
visive presentate come fusioni del modello o regole basate sul nome del dataset.

## 1. Partire dall'obiettivo della persona

Tre ingressi, con un testo breve e un esempio concreto:

- **Scoprire gruppi:** nessuna colonna da prevedere. Mostrare tutte le colonne
  osservate, compresi eventuali esiti presenti nel file. Distinguere clustering
  dell'intero file da uno studio di stabilità su campioni: mantenere la correzione già applicata: senza target il training usa tutte le
  righe, mentre il protocollo 80/20 riguarda la classificazione. Mostrare righe totali,
  mancanti e realmente utilizzate prima di avviare.
- **Prevedere una classe:** indicare la colonna da prevedere e spiegare che le
  etichette influenzano il training. Una classe può essere testo, intero o decimale.
  Il test non deve entrare in statistiche, selezione dei parametri o prototipi.
- **Adattare un modello al testo:** spiegare che impara a proseguire testi simili
  al corpus; non promettere automaticamente un assistente conversazionale.
  Collegare corpus, training, prova del testo e confronto sul test in un percorso.

Lo stato corrente resta visibile quando si cambia scheda. Caricare un modello
esistente deve mostrarne compito, dati attesi, versione matematica e impostazioni
senza far apparire come modificabili parametri di un training già concluso.

## 2. Classi numeriche e qualità dei dati

Il layout attuale espone già `Classificazione`, ma sotto le impostazioni avanzate.
`Automatico` considera classi solo codici interi con al massimo 20 valori;
altrimenti sceglie regressione, che la testa CP corrente non supporta.
Il controllo effettuato su 26 codici numerici fallisce in Automatico e riesce in
Classificazione. Non è necessario cambiare CP per accettare quei nomi di classe.

Portare la scelta semantica vicino al target, con anteprima dei valori e del
numero di classi. Chiedere una scelta esplicita nei casi ambigui. Non proporre
regressione come funzione disponibile finché non esiste una testa continua
verificata. L'errore deve spiegare cosa correggere, senza rimandare a profili
che non compaiono più nell'app.

Mostrare la politica dei mancanti: una categoria assente non deve essere
confusa nell'esposizione con una nuova classe osservata. Distinguere righe prive
di target, che non partecipano alla classificazione, da righe con attributi
mancanti. Non scartare quelle righe nel clustering per ottenere il K desiderato.

Accettazione: codici 0/1, 1…26, classi decimali, valori mancanti e classi test mai
viste; salvataggio e ricaricamento conservano la stessa codifica e predizione.

## 3. Poche impostazioni, con valori effettivi verificabili

Schermata normale: compito, dati, budget e avvio. Un pannello avanzato contiene
solo parametri attivi e spiega l'effetto pratico, senza rimandi all'implementazione.

| Controllo attuale | Proposta |
|---|---|
| Epoche / passi | Budget esplicito; distinguere letture del dataset da aggiornamenti dell'ottimizzatore. |
| Learning rate | Avanzato, con spiegazione della stabilità; conservare il valore nel riepilogo. |
| Batch tabulare | Mostrare il batch effettivo: oggi è al massimo 300 e il campo non regola la gerarchia. Allineare il controllo al comportamento prima di mostrarlo come globale. |
| Livelli iniziali | Un livello come punto di partenza; approfondimento esplicito e confronto prima/dopo. |
| Purezza e righe minime | Avanzato, solo quando l'espansione automatica li usa; spiegare il rischio di frammentazione. |
| Tau, alpha e bilanciamento storico | Non mostrare comandi inoperanti nei nuovi training; conservarli solo come metadati di modelli precedenti. |
| Routing con una sola scelta | Mostrarlo come descrizione del nucleo, non come menu fittizio. |
| Slot e token generativi | Capacità e costo, con i limiti reali del dispositivo prima dell'avvio. |
| Peso CP, LoRA, distillazione | Avanzato; distinguere organizzazione, parametri aggiornati e imitazione del modello di partenza. |

Non cambiare silenziosamente i parametri di un progetto ripreso. Prima dell'avvio
mostrare un riepilogo leggibile dei valori richiesti e di quelli effettivi.

## 4. Leggere il grafo senza confondere concetti, classi e gerarchia

Conservare navigazione locale, paginazione, zoom, fullscreen e percorso verso la
radice già presenti. Aggiungere una sintesi che distingua:

- prototipi del nodo visualizzato, prototipi totali nell'albero e nodi;
- massa appresa, frequenza morbida e numero di righe assegnate prevalentemente;
- classi previste, che non sono il numero dei cluster;
- dimensione dei gruppi e andamento della frammentazione durante l'apprendimento.

Mostrare distribuzioni delle dimensioni e il legame fra aumento dei prototipi,
CP, errore di training e validazione. Una CP alta non deve essere etichettata
come buona generalizzazione. L'approfondimento deve esporre beneficio misurato,
costo e possibilità di rimuovere il ramo. Un'aggregazione grafica deve essere
chiaramente una modalità di visualizzazione, senza cambiare il modello salvato.

## 5. Storico generativo completo e navigabile

Diagnosi: `Store.history` legge oggi `ORDER BY step DESC LIMIT 500`. Il grafico
riceve solo quella coda, quindi lo zoom non può recuperare i passi più vecchi.
Non basta cambiare i limiti dell'asse. Le metriche sono conservate nel database;
la ripresa da checkpoint, invece, elimina esplicitamente i passi successivi al
checkpoint ripreso. Non promettere di recuperare dati già eliminati.

Implementazione proposta:

1. Endpoint paginato per intervallo di passi e cursore, con lo stesso controllo
   di proprietà del progetto. 500 diventa dimensione di pagina, non limite storico.
2. Vista iniziale di tutto il percorso; per serie lunghe usare un riepilogo per
   intervalli che conservi estremi e punti di validazione. Zoom/caricamento su
   richiesta mostra i punti esatti di quell'intervallo.
3. Comandi «Intero percorso», «Ultimi 500» e intervallo di passi esplicito.
   Il polling conserva zoom e posizione con uno stato stabile per progetto.
4. Download completo CSV/JSON, indipendente dal campione visualizzato. Non
   scaricare milioni di punti per ogni progetto a ogni polling di cinque secondi.
5. Segnare validazioni realmente eseguite e checkpoint; distinguere l'ultima
   misura ripetuta da una nuova valutazione. Conservare il riferimento al modello
   originale e spiegare su quanti token è stato misurato.
6. Distinguere storico delle metriche da disponibilità dei pesi: avere una misura
   al passo 50 non significa poter generare da quel checkpoint. La conservazione
   del migliore e degli ultimi checkpoint deve essere visibile.
7. Documentare le riprese dopo un arresto: percorso confermato dal checkpoint e
   tentativi successivi eventualmente scartati. Se si vuole conservare anche
   questi ultimi, progettare un registro dei tentativi prima di cambiare le
   cancellazioni necessarie alla ripresa esatta; non mescolare curve di traiettorie diverse.

Accettazione: oltre 1.500 punti, lettura del primo e dell'ultimo, navigazione fra
pagine senza lacune/duplicati, refresh durante lo zoom, ripresa, isolamento fra
progetti, esportazione completa e nessuna modifica del training in corso.

## 6. Dare uno scopo al laboratorio generativo

Riepilogo iniziale: «che testo sto insegnando», «quanto apprendimento è avvenuto»,
«è migliorato su testi non letti?» e «che cosa produce?». NLL e perplessità restano
accessibili con spiegazioni: più basso significa migliore previsione sul campione,
non qualità generale garantita né comprensione dimostrata.

Lo screenshot mostra NLL di validazione 4,611 contro riferimento 4,829 e un
campione di controllo di 256 token. È un segnale su quel controllo; va seguito
con test separato e continuazioni, senza presentarlo come risultato LLM definitivo.

Rendere «Prova il modello» un percorso chiaro: salvare/pausare quando richiesto,
scegliere il checkpoint disponibile, proporre un inizio di testo nella lingua del
corpus, mostrare il testo prodotto e consentire il confronto con l'originale.
L'azione non cambia i pesi né avvia un nuovo training. Gli approfondimenti su
layer, gradienti e memoria restano separati dall'attività principale.

## Sequenza del prossimo run e criterio di completamento

1. Inventario delle schermate attuali e dei controlli effettivi; prototipo delle
   tre attività mantenendo componenti e persistenza esistenti.
2. Flusso dati/target e parametri, con controlli del compito prima dell'avvio.
3. API dello storico e navigazione del grafico generativo, con test oltre 500 passi.
4. Lettura dei concetti e percorso di prova del linguaggio.
5. Browser mobile/desktop: upload → training → lettura → salvataggio → ricaricamento;
   classificazione numerica, clustering senza target, generativo ripreso e storico.

La PR dell'interfaccia deve essere indipendente da nuove formule CP. Nuovi
criteri per scegliere K o per trattare la missingness richiedono confronti
matematici e sperimentali propri. Nessuna migrazione distruttiva dei progetti,
nessun riaddestramento o riavvio automatico dei job esistenti, nessuna pubblicazione
sul sito reale come parte di questo run.
