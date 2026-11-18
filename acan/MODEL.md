# Il modello a concetti

## Matematica

La coda **Addestramento lungo** usa per impostazione predefinita il contatore
di contesti. Apprende i conteggi delle continuazioni per suffissi ordinati
e legge dal suffisso generale al più specifico. Il grafo tabulare sul testo
qui descritto resta sperimentale. Le sue misure sono nella sezione Informatica.

### Dalle osservazioni ai concetti

Una riga di tabella contiene attributi osservabili e, se prevista, una colonna da leggere.
Anche il testo diventa una tabella. Per esempio nella frase «il cane corre» la riga che
deve prevedere «corre» contiene i vettori delle parole o parti di parola già lette.
Lo stesso `ConceptModel` costruisce il grafo in entrambi i casi.

Il concetto $i$ conserva la massa $C_i=\sum_t T_{ti}$ e, per ciascuna colonna
numerica osservata, la massa presente, la somma pesata dei valori e dei loro
quadrati. Il centro è la somma divisa per la massa presente. La dispersione è
il secondo momento meno il quadrato del centro. Per una categoria si conservano
i conteggi pesati. Sommare le statistiche di due concetti equivale a riunirne
le osservazioni. Nel caso di un target numerico si legge un centro, espresso
nelle unità originali. Per un target categorico si legge una distribuzione.

### Riconoscere e generalizzare

Per una riga $z$ si confrontano i soli attributi disponibili. Per una colonna
numerica il concetto attribuisce maggiore densità ai valori vicini al suo centro,
tenendo conto della sua varianza. Per una categoria conta la frequenza.
Indicando con $L_i(z)$ la verosimiglianza delle osservazioni, l'appartenenza è

$$
T_i(z)=\frac{C_i L_i(z)}{\sum_j C_j L_j(z)}.
$$

Un concetto stretto spiega bene una piccola regione. Uno più disperso spiega
più valori, ma assegna meno densità a ciascuno. Per decidere una fusione si
misura l'aumento del raggio dei concetti, pesato per la loro massa:

$$
\Delta(i,j)=(C_i+C_j)\rho_{i\cup j}-C_i\rho_i-C_j\rho_j.
$$

Il raggio è la radice della dispersione media degli attributi riconosciuti
rispetto alla scala del livello superiore. Si fondono le coppie per le quali
si perde meno precisione. Questo produce più possibili tagli degli stessi dati.

### Perché la CP può scegliere un livello

La CP a coppie chiede quanto due osservazioni nello stesso gruppo si somiglino
più di una coppia presa al livello superiore. Se $r_{ij}$ è la dispersione
relativa della colonna $j$ nel concetto $i$, stimata su membri distinti, il
criterio usato per confrontare i tagli è

$$
\mathrm{CP}^{\circ}=\frac{\sum_i C_i\max(C_i-1,0)\,\frac1m\sum_j[1-\min(2,r_{ij})]}{\max(n(n-1),\epsilon)}.
$$

Le categorie usano l'impurità $1-\sum_v p(v)^2$ come dispersione. Un gruppo
isolato contiene poche coppie. Un gruppo indistinto ha dispersione simile a
quella del livello sopra e aggiunge poca somiglianza. Il criterio premia
quindi gruppi insieme numerosi e coerenti. Il modello cerca anche strutture
residue dopo aver tolto il centro spiegato dal livello superiore. Un concetto
specifico può avere più genitori.

Per scegliere il livello di lavoro interviene anche la previsione: si sottrae
il contributo di una riga dalle statistiche e si misura la probabilità di ciò
che si deve leggere. Tra livelli con giudizio entro un errore standard dal
migliore si preferisce quello con meno nodi. Senza target si giudica ogni
attributo dagli altri. Nel testo il seguito fornisce il bersaglio direttamente
dal corpus. Il numero di concetti dipende dagli attributi osservati e dalla
struttura dei dati. La CP da sola non garantisce un numero universale di gruppi.

### Dal concetto alla probabilità del token

Per ogni concetto si contano le continuazioni. Se $N_i(y)$ è il conteggio del
token $y$, $M_i$ la somma dei conteggi, $D_i$ il numero di token distinti e
$q_i^{\rm sopra}(y)$ la previsione dei genitori, la lettura categorica vale

$$
q_i(y)=\frac{N_i(y)+a_i q_i^{\rm sopra}(y)}{M_i+a_i},
\qquad a_i=\kappa\max(1,D_i).
$$

Con poche osservazioni pesa di più il genitore. I conteggi risalgono dai figli
tramite le loro appartenenze ai genitori; la previsione scende dalla radice,
che parte da un prior uniforme. Prima di osservare il seguito si ottiene una
miscela $p(y\mid z)=\sum_i T_i(z)q_i(y)$. Durante l'apprendimento il seguito è
noto: la frazione di un conteggio assegnata al concetto vale

$$
r_i(z,y)=\frac{T_i(z)q_i(y)}{\sum_j T_j(z)q_j(y)}.
$$

Le quote sommano a uno. Un concetto riceve più credito se spiega sia il
prefisso sia la continuazione osservata. Durante la generazione si applica
la stessa lettura categorica anche alle statistiche temporanee del documento.
A ogni livello il genitore dà il prior ai conteggi globali; il risultato dà
il prior ai conteggi del documento. Dopo aver rivelato il token il documento
aggiunge il suo conteggio con i pesi del prefisso. Il checkpoint globale
rimane disponibile per il documento successivo.

## Informatica

### Dal corpus al checkpoint

1. Per una prova esplicita con il profilo sperimentale `concept-table-v1` si prepara il corpus e si avvia la
   coda. Il programma separa i documenti, elimina i duplicati nel corpus e
   costruisce il tokenizer BPE sui documenti di addestramento campionati. I
   testi successivi della stessa coda usano lo stesso vocabolario.
2. Il primo passo usa fino a 200.008 token del train. Per ogni tipo di token conta
   i vicini immediati a sinistra e a destra. Normalizza le due distribuzioni,
   ne prende la radice e le riduce con SVD a 32 coordinate. Così due token con
   vicini simili entrano nella tabella con vettori vicini.
3. Il programma scorre il testo creando blocchi di righe. Le 256 colonne note
   sono gli otto vettori precedenti. Le 32 colonne bersaglio sono il vettore
   del token successivo. Il modello tabulare impara il grafo e i collegamenti
   tra livelli. I primi otto token formano la prima finestra completa.
4. Le appartenenze ai concetti danno i conteggi iniziali. Una rilettura dello
   stesso tratto distribuisce i conteggi secondo la formula $r_i$.
   Per questa lettura il riconoscimento osserva l'ultimo token e usa
   $\kappa=0{,}5$, scelte misurate su WikiText-2. A ogni passo successivo si
   aggiungono le nuove osservazioni agli stessi concetti, usando le probabilità
   globali all'inizio del passo. Vettori e struttura sono fissati dal primo
   tratto: su un dominio nuovo possono diventare poco rappresentativi.
5. Il worker misura l'errore su documenti separati. Salva atomicamente le
   statistiche di riconoscimento, i collegamenti, i vettori, i conteggi e il
   contesto al confine del passo. Una ripresa riparte dall'ultimo checkpoint.
   Un nuovo testo in coda eredita ciò che il grafo ha imparato e inizia un
   nuovo contesto. I checkpoint storici conservano la semantica con cui furono scritti.

### Dal prompt al prossimo token

Il tokenizer del progetto trasforma il prompt in identificatori. Per ogni
posizione il grafo riconosce il prefisso, poi registra il token letto nella
memoria temporanea del documento. Una volta arrivato alla fine del prompt,
il worker valuta la distribuzione sull'intero vocabolario. A temperatura zero
sceglie il massimo; a temperatura positiva campiona. Aggiunge il token al
documento e ripete il passo. Generazione, validazione ed esportazione leggono
le stesse statistiche.

Con il grafo e il vocabolario fissati, ogni nuovo token aggiorna somme nei
livelli. Il costo per passo dipende da concetti, collegamenti e vocabolario.
La memoria temporanea può arrivare a un conteggio per coppia concetto/token.
Non serve una matrice tra tutte le posizioni. Questa compressione perde
l'ordine del passato lontano: avere memoria limitata e predire bene in media
non garantisce di recuperare un fatto isolato.

### Come leggere la qualità

La NLL è la media di $-\log p(y\mid z)$ sul vero token successivo. Più è
bassa, più probabilità il modello assegna ai seguiti osservati. Nella prova
di sviluppo su 32.704 predizioni, il grafo a 83 concetti ha ottenuto
6,085 nat/token; il precedente contatore ha ottenuto 5,452 sugli stessi
blocchi. In quella prova i vettori erano stimati sull'intero train. Nel worker
i vettori vengono stimati sul tratto iniziale per poter continuare in streaming.
La metrica del singolo progetto è quella misurata sul suo corpus.

Una prova con il percorso di apprendimento del worker ha letto tutti i
2.729.601 token di training di WikiText-2 con vocabolario BPE 8.000 e passi
da 200.008 token. Il grafo ha trovato livelli da 26 e 42 concetti. Sugli
stessi 64 blocchi da 512 token (32.704 previsioni, primo token dato) la NLL
è scesa da 6,1674 dopo il primo passo a 5,9701 dopo l'intero train. Il
contatore storico, sul train completo con questi stessi blocchi, misurava
4,5199. Sono misure di previsione sul testo inglese; la qualità di dialogo
richiede una verifica sul corpus che si intende usare.
