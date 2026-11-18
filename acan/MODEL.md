# Il modello a concetti

## Matematica

Una tabella e una sequenza di testo pongono la stessa domanda: quali osservazioni
si possono riunire senza perdere informazione utile alla previsione? ACAN usa
il modello tabulare per proporre gruppi di osservazioni simili. Nel testo
la coda lunga permette di provare quei gruppi dentro il predittore di suffissi
ordinati. Il contatore di suffissi resta la scelta predefinita.

### Dalle osservazioni ai concetti

In una tabella una riga descrive un caso attraverso i suoi attributi. Nel testo
si costruisce una riga per **tipo di token**. I suoi attributi descrivono quali
token lo precedono e lo seguono nei documenti di training. Due token che occupano
posizioni simili possono condividere evidenza anche se le loro grafie sono diverse.
Il modello tabulare trova i concetti in queste righe. Per prevedere in un punto
preciso del testo serve poi l'ordine dei token del prefisso.

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

Nel tabulare il giudizio predittivo sottrae la riga in esame dalle statistiche
e misura la probabilità degli attributi da leggere. Tra livelli entro un errore
standard dal migliore preferisce quello con meno nodi. Senza target cerca di
prevedere ogni attributo dagli altri. Nel testo la CP propone i tagli della
tabella dei token. Il giudizio finale confronta quei tagli usando la probabilità
del **vero token successivo** su documenti di validazione separati.

### Dal concetto alla probabilità del token

Supponiamo di aver letto «ho visto il». L'ordine più corto registra cosa segue
«il». Un livello intermedio registra cosa segue un token del gruppo di «visto»
seguito esattamente da «il». Il livello più preciso registra cosa segue proprio
«visto il». Anche quando la frase precisa compare poche volte, i suoi esempi
contribuiscono alla stima del gruppo. Se la frase è frequente, la sua
distribuzione specifica acquista più peso. Il passaggio si ripete mantenendo
esatti i token più vicini alla posizione da prevedere.

Sia $N_s(y)$ il numero di volte che il token $y$ è comparso dopo il contesto $s$,
$M_s$ il totale delle continuazioni e $D_s$ il numero di continuazioni distinte.
La radice parte da una distribuzione uniforme sul vocabolario. A ogni livello
si aggiorna la distribuzione del livello precedente con la stessa lettura categorica:

$$
q_s(y)=\frac{N_s(y)+a_s q_{\mathrm{prec}}(y)}{M_s+a_s},
\qquad a_s=\kappa_s\max(1,D_s).
$$

La somma su tutti i token vale uno. Un contesto mai osservato restituisce la
distribuzione precedente. Il rapporto $M_s/(M_s+a_s)$ dice quanto incidono le
osservazioni locali. Per un taglio CP il lettore somma i conteggi degli stessi
suffissi che condividono il gruppo del token più lontano. La validazione decide
a quali profondità inserire questo passaggio e con quale peso. Prova anche a
saltarlo. Il criterio è la perdita logaritmica per token

$$
\mathcal L=-\frac{1}{N}\sum_t\log q(w_{t+1} \mid w_{\le t}).
$$

La CP entra nella struttura che si confronta, la NLL decide quale struttura
legge meglio. L'apprendimento usa conteggi e confronto fra tagli. I livelli
successivi affinano una distribuzione con evidenza progressivamente più specifica.

## Informatica

### Dal corpus al checkpoint

1. La coda suddivide i documenti, rimuove duplicati e prepara il tokenizer BPE
   su un campione del training. Ogni file successivo usa lo stesso vocabolario.
2. Nella lettura sperimentale il worker scorre **tutto il train richiesto**.
   Per ogni token registra il seguito dei suffissi ordinati disponibili
   (profondità predefinita 6). Conserva la coda necessaria attraverso i blocchi
   di I/O. Un nuovo file eredita le statistiche e inizia un nuovo contesto.
3. Dai bigrammi conservati ricava due distribuzioni di vicini per tipo di token.
   Su ciascun lato distingue fino a 512 vicini frequenti più una categoria
   residua. Prende la radice delle frequenze relative e usa una SVD centrata
   fino a 32 dimensioni. Il modello tabulare scopre i concetti sui profili di
   fino a 8.000 tipi frequenti (almeno tre osservazioni in uscita). Assegna anche
   gli altri tipi con sufficiente evidenza. I rari condividono una classe residua.
   Il limite di ricerca è 256 concetti, il numero scelto può essere inferiore.
4. Nelle valutazioni intermedie il worker ristima la struttura quando il numero
   di token letti è almeno raddoppiato dall'ultima stima e la rivaluta alla fine
   del file. Dopo ogni cambio di partizione rigenera le statistiche raggruppate
   proiettando le tabelle esatte conservate. Non occorre rileggere il corpus.
   Il limite alle statistiche può eliminare distinzioni rare. Ogni osservazione
   contribuisce al momento della lettura, la sua chiave precisa può essere persa.
5. Il worker verifica sulla validazione i pesi, il taglio e l'opzione di usare
   i soli suffissi esatti. Alle profondità da 1 a 3 può inserire il gruppo del
   token più lontano. Un secondo insieme di blocchi misura la NLL mostrata nel
   pannello. Se è disponibile un solo blocco, le due metà servono ai due giudizi.
   Il checkpoint salva conteggi, partizione, pesi e ultimi token per la ripresa.

### Dal prompt al prossimo token

Il tokenizer converte il prompt negli identificatori usati dal training. Un
cursore percorre il prompt in ordine e aggiorna le statistiche del documento.
Alla fine legge prima i livelli del corpus, dal generale al particolare. Aggiorna
poi la stessa distribuzione con le continuazioni già viste nel documento, dal
conteggio senza contesto ai suffissi di lunghezza 4 per impostazione predefinita.
Un'associazione vista una volta può così riemergere se il suo indizio ricorre.
A temperatura zero sceglie il token più probabile, altrimenti campiona. Solo dopo
aver scelto il token lo registra nel documento e procede alla posizione successiva.

Con profondità, budget e vocabolario fissati, il primo passaggio sul prompt
ha costo lineare nella sua lunghezza. Ogni nuovo token aggiorna un numero
limitato di indici. La distribuzione completa dipende anche dalla dimensione
del vocabolario. Quando il numero di celle di un livello del documento supera
la quota, viene dimenticato il suffisso inserito da più tempo a quel livello.
Non occorre conservare una matrice fra tutte le posizioni. Il ricordo a distanza
dipende dall'indizio che si ripete e dalle statistiche ancora presenti.

### Come leggere la qualità

Su WikiText-2 inglese, vocabolario BPE 8.000, entrambe le letture hanno visto
2.729.601 token. I tagli e i pesi sono stati scelti su 32 blocchi di validazione.
La tabella riporta la NLL su 64 ulteriori blocchi di test, 32.704 predizioni.
Ogni coppia usa le stesse posizioni e lo stesso limite alle statistiche.

| Budget delle statistiche | Suffissi esatti | Suffissi con gruppi CP | Gruppi rimescolati | Classi nel taglio scelto |
|---|---:|---:|---:|---:|
| 256 MiB | 4,5403 | 4,4963 | 4,5401 | 193 |
| 1.024 MiB | 4,5023 | 4,4588 | 4,5017 | 208 |

Rimescolare le identità dei token mantenendo le dimensioni dei gruppi quasi
annulla il vantaggio. Con 256 MiB il taglio genera anche oltre 600.000 suffissi
distinti al terzo livello raggruppato. Le chiavi sono interi a 64 bit: per i
suffissi lunghi la codifica può avere collisioni. Il numero di contesti ordinati e il
numero di classi di token misurano cose diverse. Il limite della tabella
riguarda le statistiche residenti. Raggruppamento, ordinamento e selezione
hanno richiesto fino a circa 1,4 GiB di memoria del processo nella prova da
256 MiB. Ripetendo il confronto a 256 MiB con il contesto mantenuto tra i
blocchi di training come nel worker, la NLL passa da 4,5403 a **4,4946**
(errore standard della differenza 0,0021 nat/token). Sugli stessi blocchi,
tenendo fissi i pesi e limitando i suffissi visibili a uno, due, tre oppure sei
token, la NLL vale rispettivamente 4,8204, 4,5223, 4,5012 e 4,4946. Queste
differenze misurano il contributo delle posizioni precedenti all'ultima.

Con gli stessi indizi casuali dell'esperimento precedente, a distanza di
300.000 token le due letture recuperano entrambe 16 associazioni su 16 e
completano 16 sequenze su 16 al primo posto sull'intero vocabolario. Il merito
di questo risultato è della memoria per suffissi esatti del documento. A 1.000
token la lettura con CP completa 15 sequenze su 16 contro 16 del contatore.
L'indizio viene ripetuto esattamente e il riempimento esclude le chiavi e le
risposte. Questa prova non misura il recupero da una parafrasi.

Nella precedente prova in cui una riga conteneva otto vettori e si prevedeva
il vettore del seguito, il grafo del worker aveva livelli da 26 e 42 concetti.
La NLL sul test storico era 5,9701 contro 4,5199 del contatore. Il nuovo
lettore con 1.024 MiB raggiunge 4,4779 su quegli stessi blocchi. La prova
di sviluppo con 83 concetti e NLL 6,085 era un esperimento distinto.
Le misure riguardano previsione su testo inglese e indizi esatti. Qualità del
dialogo, astrazione semantica a grande distanza e trasferimento ad altri
corpora richiedono misure dedicate.
