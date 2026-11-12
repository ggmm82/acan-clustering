# Correzioni dell'interfaccia nella PR11

Il nucleo matematico e i parametri del modello non sono modificati da queste
correzioni. I job generativi in corso non sono avviati, fermati o riaddestrati.

- **Model:** target vuoto significa clustering; un target presente significa
  classificazione, compresi numeri interi e decimali. La scelta nascosta in
  Automatico che oltre20 codici diventava regressione è rimossa dal flusso web.
- I menu con un solo modello e le opzioni non disponibili sono nascosti. Batch,
  purezza e righe minime vengono disabilitati quando non agiscono sul percorso
  scelto. Il batch richiesto ha massimo300, coerente con il limite effettivo.
  Ogni impostazione spiega l'effetto; i parametri generativi hanno aiuti analoghi.
- Il grafo distingue numero di nodi, prototipi complessivi e gruppi del nodo.
- Il generativo spiega il suo scopo: adattare continuazioni al corpus, osservare
  la validazione e provare i checkpoint.

## Storico oltre500 passi

Lo stato del worker restituisce una panoramica dall'inizio alla fine, fino a circa
1.001 punti, mediante campionamento regolare. Non nasconde il primo passo dopo
il raggiungimento di500 record. Una serie campionata può omettere picchi: non è
l'esportazione delle misure e non viene presentata come tale.

L'endpoint autenticato `/jobs/<id>/history` fornisce pagine esatte con start,
limit e un end facoltativo; next_step permette di scorrere senza lacune.
L'interfaccia offre finestre di500 passi, un passo iniziale e comandi prima/dopo.
Il grafico conserva zoom e posizione durante il polling e ha uno slider temporale.
L'accesso continua a verificare la proprietà del progetto. Un vecchio worker
viene segnalato se non fornisce il nuovo storico.

I checkpoint disponibili restano distinti dalle metriche: leggere un passo non
implica possederne ancora i pesi. La ripresa da checkpoint conserva le regole
esistenti; non vengono recuperate metriche già eliminate da una ripresa precedente.

Le prove controllano oltre1.500 record, prime/ultime misure, paginazione, intervalli,
parametri non validi e isolamento fra proprietari. Nel browser vengono verificati
training e salvataggio con26 classi numeriche decimali, filtri e navigazione del
grafo su schermi piccoli e grandi.
