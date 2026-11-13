# Piano della prossima revisione: il grafo dei concetti a livelli

> **Documento storico.** Il modello attuale è uno solo ed è descritto in [MODEL.md](MODEL.md), con le misure in [MODEL-RESULTS.md](MODEL-RESULTS.md). Dove questa pagina differisce, vale MODEL.md: in particolare la covarianza nel concetto, i livelli automatici e la fusione per lunghezza di descrizione sono stati tolti.

Fondamento: [CONCEPT-GRAPH.md](CONCEPT-GRAPH.md). Stato di partenza:
[CP-V9-RESULTS.md](CP-V9-RESULTS.md).

## Stato

| Passo | Esito |
|---|---|
| 1. Operatori e struttura del grafo | fatto |
| 2. Tabulare a livelli | fatto; livelli costruiti dall'alto con accettazione su righe fresche. Da soli sono neutri su Letter (criterio non rispettato). Punteggi normali disponibili ma non predefiniti |
| 3. Covarianza nel concetto | fatto; Letter oltre il 97% |
| 4. Linguaggio | fatto in forma diversa: fusione guidata dal lettore per testa (`cp-graph-v1`). Non realizzati il grafo unico per tutti i layer e la lettura adattiva per query |
| 5. Misure e documentazione | [CP-GRAPH-RESULTS.md](CP-GRAPH-RESULTS.md) |

## Obiettivo

Un solo oggetto, il grafo dei concetti a livelli, usato dal modello tabulare e
da tutti i layer del modello linguistico, con gli stessi tre operatori e la
stessa regola: **ogni livello è il prior del successivo, e un livello più fine
si usa solo dove serve**.

Non è un obiettivo di questa revisione dimostrare che il modello linguistico
compresso eguagli quello denso: richiede addestramento su GPU. È un obiettivo
misurare, a pesi invariati e con i controlli corretti, quanto il grafo
recupera.

## Passi

Ogni passo è un commit che lascia l'app utilizzabile e i test verdi.

### 1. Statistiche e operatori del grafo

- `acan/concept_algebra.py`: somiglianza media incrociata $\bar s_{il}$ (G2),
  funzione a coppie $CP^\circ$ dalle statistiche, prior di livello (G7).
  Stesse funzioni per NumPy e PyTorch.
- Struttura `ConceptGraph`: livelli annidati, statistiche di un nodo uguali
  alla somma dei figli, potatura di un ramo per somma.
- Test: identità di G1 e G3 sulle funzioni di produzione; additività fra
  livelli; un concetto senza sostegno coincide con il genitore; parità
  NumPy/PyTorch.

### 2. Tabulare a livelli

- I livelli sono i tagli dell'albero di generalizzazione; ogni livello usa come
  riferimento la varianza del livello superiore.
- Previsione con il prior di livello al posto delle soglie di purezza e righe
  minime; la profondità massima usata è scelta dal giudizio sulle righe non
  viste.
- Attributi numerici in punteggi normali (statistiche del solo training):
  risolve i concetti spurii con rumore non gaussiano (4 concetti in 8 casi su
  8 nelle prove, contro 4 su 8).
- $CP^\circ$ per livello riportata nell'app accanto alla varianza spiegata.
- L'app mostra il grafo per livelli; l'approfondimento manuale resta.

### 3. Covarianza nel concetto (se il passo 2 non basta su Letter)

- Statistica aggiuntiva: somma dei prodotti incrociati, additiva.
- Solo quando gli attributi sono pochi rispetto alle righe del concetto;
  ritirata verso la diagonale con lo stesso $\kappa$.

### 4. Linguaggio: un grafo per il contesto

- Il grafo è costruito una volta sulle chiavi non ruotate di tutti i layer
  innestati (i layer sono attributi del token) e letto da ogni layer.
- Lettura adattiva: si aprono i nodi con il maggiore errore stimato
  (massa × dispersione) fino al numero di vettori disponibile; con apertura
  completa lo strato coincide con l'attenzione densa.
- I primi token del contesto restano sempre esatti.
- Potatura: i rami non aperti si riducono al genitore; lo stato resta limitato.
- Apertura per livelli in forma vettoriale, per non introdurre un altro ciclo
  per token.
- Test: esattezza con apertura completa, causalità, decodifica incrementale
  identica al prefisso intero, gradienti finiti, ripresa esatta nel trainer.
- Il profilo attuale a finestra più concetti piatti resta caricabile.

### 5. Misure e documentazione

- Tabulare: stessi split e semi di CP-V9-RESULTS.md, più dataset non ancora
  usati nello sviluppo, letti una sola volta.
- Linguaggio, a pesi invariati: NLL del modello completo su WikiText-2 con
  lettura adattiva, contro attenzione densa, «primi token + finestra» a parità
  di vettori, e concetti piatti; livello medio aperto per layer; curva
  errore/stato con la potatura; token al secondo.
- Scheda Matematica e fondamento aggiornati; risultati negativi inclusi.

## Criteri di passaggio

Fissati ora; un esito negativo si registra senza cambiare la soglia.

| Passo | Si considera riuscito se |
|---|---|
| 2 | Gioco: 4 concetti con e senza target; Letter ≥ 93,3% con non più concetti di oggi; Adult e Titanic non peggiori di cp-v9 a un livello oltre un errore standard; nessuna soglia di purezza o righe minime nel percorso predefinito |
| 3 | Letter ≥ 95% senza regressioni su Sonar |
| 4 | con apertura completa uscita identica alla densa; a parità di vettori letti la NLL è migliore di «primi token + finestra»; la decodifica incrementale coincide con il prefisso intero |
| 5 | ogni numero dei documenti è rigenerabile con un comando |

## Rischi

| Rischio | Segnale | Risposta |
|---|---|---|
| Il guadagno della lettura adattiva misurato su un singolo strato non si ritrova nella NLL del modello completo | passo 5: NLL non migliore del controllo | si riporta l'esito; il grafo resta nel tabulare e la parte linguistica resta sperimentale |
| La potatura fa risalire l'errore troppo presto | curva errore/stato ripida | stato più grande nei layer bassi, dove la risoluzione necessaria è maggiore |
| Il prior di livello non basta a evitare il sovradattamento in profondità | Adult peggiora con i livelli | profondità scelta dal giudizio, non fissata |
| Costo della lettura adattiva | token/s inferiori al profilo attuale | apertura per livelli; in mancanza, lettura adattiva solo nei layer bassi |
| Compatibilità dei progetti generativi esistenti | test di ripresa | nuove ricevute in `acan/lm/compatibility.py`, come nella revisione precedente |

## Fuori da questa revisione

Addestramento su larga scala e benchmark di contesto lungo (GPU); il termine
ausiliario di compattezza delle chiavi durante l'addestramento; il nuovo paper.
