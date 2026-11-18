# Il modello a concetti

## Matematica

### Una riga da completare

Una riga contiene informazioni note e una risposta da leggere. In una tabella
di automobili gli attributi possono essere peso e cilindrata, la risposta il
prezzo. Nel testo la risposta è la categoria del prossimo token. Il problema in
entrambi i casi è riconoscere la riga da ciò che si conosce e leggere la risposta
dalle righe di training che le somigliano.

Seguiamo la frase «il gatto nero dorme sul tappeto del salotto». Per semplicità
supponiamo che ogni parola sia un token. Alla posizione successiva la prima
colonna descrive «salotto». La seconda identifica il concetto di «del salotto».
La terza identifica il concetto di «sul tappeto del salotto». La quarta riassume
le otto parole già lette. Il punto che potrebbe seguire è un possibile valore
della colonna da completare. In un'altra frase «della cucina» può ricevere lo
stesso concetto di «del salotto» se il modello ha trovato somiglianze sufficienti.

### Che cosa conserva un concetto

Durante l'apprendimento una riga può contribuire a più concetti. Indichiamo con
$T_{ri}$ la sua quota nel concetto $i$. La massa del concetto e la sua media per
un attributo numerico presente sono

$$
C_i=\sum_r T_{ri},\qquad
\mu_{ij}=\frac{\sum_{r:\,j\,\mathrm{presente}}T_{ri}x_{rj}}
                   {\sum_{r:\,j\,\mathrm{presente}}T_{ri}} .
$$

Il modello conserva anche la somma dei quadrati, da cui ricava la dispersione.
Per una categoria conserva la somma delle quote per ciascun valore. Una fusione
somma le statistiche dei due gruppi. Una misura assente non contribuisce alla
massa osservata di quella colonna.

### Come riconosce i gruppi e sceglie i livelli

Un valore numerico vicino al centro di un concetto stretto ha una densità
elevata. Un concetto disperso spiega un intervallo più largo, assegnando meno
densità a ciascun valore. Per le categorie conta quanto erano frequenti fra i
membri. Se $L_i(x)$ è la verosimiglianza degli attributi presenti, la quota di
appartenenza della nuova riga è

$$
T_i(x)=\frac{C_i L_i(x)}{\sum_h C_h L_h(x)} .
$$

Il modello cerca gruppi che si possono riunire con poca perdita. Chiama raggio
$\rho$ la radice della dispersione media delle colonne osservabili, rispetto
alla scala data dal livello superiore. Il costo di riunire due gruppi è

$$
\Delta_{ij}=(C_i+C_j)\rho_{i\cup j}-C_i\rho_i-C_j\rho_j .
$$

Fondere prima le coppie a costo basso produce possibili tagli della tabella.
La CP a coppie misura se due membri distinti dello stesso gruppo si somigliano
più di due righe del livello precedente. Indicando con $r_{ij}$ la dispersione
relativa dell'attributo $j$, il punteggio di un taglio è

$$
\mathrm{CP}^{\circ}=
\frac{\sum_i C_i\max(C_i-1,0)\,\frac1m\sum_j [1-\min(2,r_{ij})]}
     {\max(n(n-1),\epsilon)} .
$$

Un gruppo piccolo offre poche coppie come evidenza. Un gruppo grande che
riproduce la dispersione della popolazione aggiunge poca somiglianza. Per le
categorie la dispersione è la probabilità che due estrazioni diano valori
diversi, stimata a partire dalle frequenze. Il modello verifica inoltre quanto
bene legge una riga togliendo il contributo di quella riga dalle statistiche.
Tra tagli entro un errore standard dal migliore preferisce il più semplice.
Una struttura residua può giustificare un livello più specifico. Anche un solo
gruppo è un risultato possibile.

### Dal concetto alla probabilità del token

Per un prezzo si legge una media dei valori numerici nei concetti riconosciuti.
Per un token si legge la distribuzione della colonna categorica. Siano $N_i(y)$
la massa del token $y$, $M_i$ la massa totale presente del target e $D_i$ il
numero di valori distinti del target nel concetto. Il modello calcola

$$
q_i(y)=\frac{N_i(y)+a_i q_i^{\rm sopra}(y)}{M_i+a_i},
\qquad a_i=\kappa\max(1,D_i),\qquad
p(y\mid x)=\sum_i T_i(x)q_i(y).
$$

Il livello superiore dà una stima di riferimento quando il concetto ha poca
evidenza. La validazione interna sceglie il peso di questa stima. Tornando a
«del salotto», il punto riceve probabilità se era un seguito frequente nei
concetti ai quali appartiene **la riga intera**. Durante la predizione il
riconoscimento usa soltanto le colonne note. Per vedere il calcolo, supponiamo che i seguiti osservati nel concetto
siano «.», «.» ed «e». Ci sono tre osservazioni e due valori distinti. Con
$\kappa=1$ e una probabilità del punto pari a $1/4$ nel livello superiore,
quel concetto assegna al punto $(2+2/4)/(3+2)=1/2$. La riga combina questo
valore con quelli degli altri concetti secondo le proprie appartenenze.

La qualità sul testo si misura
con la NLL, la media del logaritmo negativo della probabilità assegnata al
seguito realmente osservato.

## Informatica

### Dal corpus al checkpoint

1. La prova di ricerca usa il tokenizer BPE 8.000 imparato sul train di
   WikiText-2. Ogni intestazione di articolo apre un documento. Gli identificatori
   del tokenizer servono per leggere il testo e identificare le categorie target.
2. Ogni tipo di token diventa una riga di frequenze dei vicini sinistri e destri,
   misurate sul training all'interno di ogni articolo. Su ciascun lato si
   distinguono i 24 vicini più frequenti più una categoria residua e si prende
   la radice delle frequenze relative. Si confrontano due embedding alla stessa
   dimensione. La SVD conserva le principali direzioni di variazione. L'alternativa
   usa come coordinate le probabilità di appartenenza assegnate dal modello
   tabulare ai profili. Entrambe le rappresentazioni restano fisse sul test.
3. Per costruire una finestra di due token, il modello tabulare legge i vettori
   della metà più vecchia e di quella più recente in colonne distinte. Per una
   finestra di quattro token legge i due concetti di coppia, ancora in ordine.
   La costruzione prosegue per raddoppio. In queste tabelle ausiliarie il vettore
   del seguito è un target di training. Per riconoscere un tratto si forniscono
   esclusivamente le due metà. Come rappresentante discreto del tratto si
   prende il concetto con la maggiore appartenenza.
4. A ogni posizione la tabella finale ha il vettore dell'ultimo token, una
   colonna categorica per ciascuna scala appresa e un target categorico, il
   vero token seguente. Una finestra che oltrepassa l'inizio del documento
   produce un valore mancante. Una **scala** è una lunghezza scelta per le
   colonne. Un **livello** è una distinzione che il modello trova *dentro*
   una di queste tabelle. Le venti scale previste vanno da 1 a 524.288 token.
5. La prova campiona uniformemente fino a 2.048 tratti eleggibili per apprendere
   ogni composizione e 12.000 posizioni distribuite sul train per la struttura
   finale. Le colonne dei concetti sono costruite a blocchi su disco. Una passata
   su tutti i 2.729.601 token completa le somme del livello di lavoro con il
   dizionario intero e il prior empirico del train. Le probabilità restano quelle
   del modello tabulare. I campioni pongono un limite alle strutture che si possono
   scoprire. Il numero di concetti e livelli entro il budget è scelto dal giudizio.
6. I primi 32 blocchi di validazione scelgono l'embedding. Altri 32 controllano
   la scelta. È possibile salvare localmente lo stato dell'esperimento e
   verificare che, dopo il caricamento, lo stesso prompt dia le stesse probabilità.

### Dal prompt al prossimo token

Il tokenizer legge il prompt. Nell'esempio, quando arriva «salotto», il modello
forma la coppia «del salotto», poi unisce il concetto di «sul tappeto» al concetto
di quella coppia. Le finestre già complete riempiono la riga. La lettura categorica
del modello finale produce una distribuzione sull'intero vocabolario. Si può
scegliere il token più probabile, registrarlo come ultima osservazione e ripetere.

Per ogni scala il cursore conserva in un buffer circolare la metà più vecchia
che servirà a completare le finestre successive. Coprire una finestra massima
$W$ richiede $O(\log W)$ colonne e fino a $O(W)$ identificatori in questi buffer.
A modelli fissati, leggere un prompt di $n$ token richiede $O(n\log W)$
composizioni. Generare un token richiede $O(\log W)$ composizioni più la
lettura sul vocabolario. Il prefisso e la generazione incrementale danno la
stessa previsione del calcolo per blocchi. Il checkpoint di questa prova può
essere usato da un processo di ricerca locale. La coda lunga dell'app usa il
contatore di suffissi; i progetti già salvati mantengono il loro lettore.

### Che cosa hanno mostrato le prove

Su 64 blocchi di test da 512 token (32.704 previsioni), il lettore diadico
ottiene **7,1808 nat/token**. Il contatore raggiunge **4,5000** con gli stessi
confini di articolo e gli stessi blocchi. Il precedente contatore con gruppi
CP ottiene 4,4555. I riferimenti storici erano rispettivamente 4,5023 e 4,4588
con il precedente trattamento dei confini. Il valore 4,403 della rete Llama
a quattro strati è il riferimento fornito per il progetto, senza una nuova
misura in questa prova.

**Il contesto non ha migliorato la previsione.** Il modello ha trovato
distinzioni nelle singole scale. Per la tabella con target token, il miglior
taglio nel giudizio interno conteneva un solo concetto. Anche prima di
applicare la preferenza per i tagli semplici, il giudizio favoriva quella
soluzione. La passata sul train intero ha quindi aggiornato un'unica
distribuzione di continuazioni. Questo spiega perché mascherare una scala
alla volta, rimescolare gli identificatori dei concetti o dare al modello
tutto il prefisso dell'articolo lascia la NLL invariata. La scelta della
struttura è stata fatta su un campione di 12.000 righe, mentre il dizionario
del target comprende 8.000 tipi. La scarsità di ripetizioni utili al giudizio
è una possibile spiegazione del taglio scelto.

**Embedding.** Con otto coordinate per entrambe le rappresentazioni, la NLL
di selezione è 7,2106 sia con SVD sia con appartenenze CP. In caso di parità
il protocollo conserva il primo candidato, qui CP. La misura riguarda questa
tabella finale e questi profili; non consente di dichiarare una rappresentazione
semanticamente migliore dell'altra.

| Posizione nel blocco | NLL diadica sul test |
|---|---:|
| 1–31 | 7,1829 |
| 32–127 | 7,1497 |
| 128–255 | 7,1960 |
| 256–511 | 7,1846 |

**Scale e livelli.** Il train contiene 629 articoli. Il più lungo ha 22.891
token. La maggiore scala con esempi di training è quindi 16.384. Nel campione
di test, anche dando tutto il prefisso dell'articolo, 7.663 posizioni hanno
la scala 8.192 disponibile e nessuna ha la scala 16.384. Ogni numero nella
colonna di destra è il numero di concetti in un livello appreso per la
rappresentazione CP.

| Lunghezza del tratto | Concetti nei livelli appresi |
|---:|---|
| 1, embedding CP | 5 → 4 → 7 → 8 |
| 2 | 4 → 3 → 14 |
| 4 | 3 → 4 → 7 → 11 |
| 8 | 3 → 7 → 11 |
| 16 | 5 → 6 → 7 → 8 |
| 32 | 7 → 8 → 9 → 10 |
| 64 | 6 → 7 → 8 → 9 |
| 128 | 5 → 6 → 7 → 10 |
| 256 | 8 → 7 |
| 512 | 7 |
| 1.024 | 8 → 9 → 10 |
| 2.048 | 10 → 5 → 6 → 8 → 9 → 10 → 11 → 14 |
| 4.096 | 8 → 14 |
| 8.192 | 6 → 6 → 7 → 8 |
| 16.384 | 3 → 5 → 6 → 9 |
| 32.768–524.288 | Nessun tratto di training abbastanza lungo |
| Tabella che predice il token | 1 |

**Richiamo e costo.** A 1.000, 10.000, 100.000 e 300.000 token la tabella
ottiene 0 risposte corrette su 16 al primo posto sull'intero vocabolario in
ciascuna prova di associazione, copia e riformulazione lessicale. Limitando
le risposte a sedici candidati ottiene 1 su 16. La prova di associazione ha
richiesto circa 0,29 ms per token generato dopo il prefisso corto e 0,24 ms
dopo quello lungo. Leggere il prefisso con 300.000 token di riempimento ha
richiesto circa 0,25 s. Il fit ha richiesto 381 s con embedding CP e 322 s con
SVD in questo ambiente, con picco del processo di circa 4,22 GiB. Un prompt
lungo può dunque essere elaborato rapidamente anche se il modello non ne
recupera il contenuto.

Sul tabulare, il giudizio per un target categorico ora sottrae l'osservazione
dal solo valore coinvolto e aggiorna massa e numero di valori distinti. Su
8.000 categorie e 64 concetti, quattro giudizi da 256 righe sono passati da
6,74 s a 0,11 s. Lo scarto massimo nei punteggi è stato inferiore a
$2\times10^{-15}$. Sui dieci file di esempio il confronto prima/dopo conserva
livelli e risultati. GiocoSet completo produce quattro gruppi senza target
e Letter raggiunge il 97,80% su 4.000 righe di test.
