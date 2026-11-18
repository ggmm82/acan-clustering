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

### Perché affinare la lettura del contesto

Se due contesti portano spesso a seguiti diversi, accorparli fa perdere informazione.
Viceversa, separare «del salotto» da ogni altra coppia sulla base di una sola occorrenza
rende la stima del seguito fragile. Il modello confronta due risoluzioni. Prima considera
un gruppo ampio, poi propone di dividerlo ponendo una soglia su una coordinata osservata.
Nei due sottogruppi misura la dispersione **su tutte le coordinate** rispetto al genitore.
Tiene una divisione con CP a coppie positiva. Le frequenze di training pesano le coppie.
Questa è la costruzione delle regioni candidate.

Per decidere quali regioni usare per la previsione, il modello accumula i seguiti di
**tutte** le righe di training in ciascuna regione. Applica quindi la stessa formula
categorica $q_i$ vista sopra. Ora la regione è osservata per intero oppure assente
e il peso di appartenenza è rispettivamente uno oppure zero. Diverse risoluzioni
possono contribuire alla stessa previsione. La validazione confronta la probabilità
del token vero, regola la forza del livello precedente e può scartare un affinamento.
È questa seconda verifica che distingue una separazione geometrica utile da una che
migliora anche la lettura del seguito. Una regione mai osservata eredita la stima già
disponibile.

Due regioni che hanno sempre dato lo stesso unico seguito, osservato lo stesso numero
di volte, applicano esattamente la stessa funzione alla stima precedente. Possono
condividere il prototipo di lettura conservando distinti gli indirizzi con cui vengono
riconosciute. Questa uguaglianza vale per qualunque prefisso e per ogni categoria.

## Informatica

### Dal corpus al checkpoint

1. Il tokenizer BPE viene appreso sul train. Per ciascun token si misurano separatamente
   i vicini sinistri e destri all'interno dello stesso articolo. Nella prova finale si
   distinguono i 96 vicini più frequenti per lato e una categoria residua. Le radici
   delle frequenze relative formano il profilo numerico del token.
2. Si provano due modi di rappresentare il profilo. La SVD ne conserva le principali
   direzioni di variazione. Il modello a concetti può invece usare le distanze espresse
   dalle sue log-probabilità di appartenenza. La trasformazione $\log(1-\log p)$
   conserva differenze fra appartenenze molto piccole. La validazione sceglie la
   rappresentazione prima di aprire il test.
3. Una finestra di due token accosta i vettori della metà più vecchia e della metà
   recente, in quest'ordine. Il modello propone tagli CP su coppie campionate da tutto
   il train. A ogni regione assegna un vettore calcolato dai seguiti di tutte le sue
   occorrenze. È la media dei vettori del token successivo, con un'osservazione di
   riferimento dal genitore. Le finestre di quattro token leggono questi vettori delle
   due coppie. La stessa operazione costruisce ogni scala successiva. Il vettore di
   una regione viene fissato nel training. Sul prompt basta riconoscere le due metà.
4. Alla posizione successiva a «del salotto», il modello può riconoscere sia il concetto
   preciso di «salotto» sia una regione più ampia per il tratto precedente. Le due metà
   possono essere lette con precisioni diverse. Il modello conta le categorie target
   in queste regioni con somme sparse. Le colonne di training possono risiedere su
   disco e i conteggi di blocchi disgiunti si sommano. Il giudizio su testo separato
   seleziona le letture e i loro pesi. Il numero di concetti emerge entro il budget
   di ricerca, che limita campione, profondità e direzioni provate.
5. Il checkpoint conserva le soglie, i vettori dei tratti, le categorie osservate e
   le letture selezionate. Nella prova finale 5.307.315 regioni dei livelli di lettura
   condividono 1.421.564 prototipi. La condivisione riduce le matrici da 370 a 182 MB
   con le stesse probabilità. Le regioni sono indirizzi di contesto e i prototipi
   sono distribuzioni riutilizzabili del seguito.

### Come una parola composta da più token aiuta il richiamo

«Doctor» e «physician» possono terminare in token diversi. Per conservare il
significato oltre i frammenti, una colonna aggiuntiva segue i confini di parola
nel prefisso BPE. Il train misura quali parole ricorrono nelle cinque posizioni
precedenti e successive, dentro la stessa riga di testo. Un vicino vale di più
se è vicino in posizione e se ricorre con quella parola più spesso di quanto
suggerisca la sua frequenza generale. Una SVD riassume questi profili in 64
coordinate. Le soglie CP raggruppano le parole con profili simili.

Nel documento corrente si osservano le tre parole precedenti e il frammento
corrente. Tre posizioni riconoscono una regione CP, la quarta fornisce il
vettore da confrontare. La validazione prova ciascuna posizione in questo ruolo.
Nella frase «for the doctor is» il vettore variabile può essere quello di
«doctor». Se ricompare la regione con un altro vettore, le osservazioni passate
con parole più simili ricevono più peso. Con vettori di norma unitaria $u$ e $v$
il peso relativo è proporzionale a $\exp(16 u^\top v)$. È una densità gaussiana
sulla distanza fra i vettori unitari. Il token da predire resta la categoria BPE.

La massa e i conteggi pesati entrano nella lettura categorica già descritta.
Il riferimento generale pesa meno quando le categorie osservate nella regione
sono ripetutamente insolite rispetto al corpus e la parola variabile assume
valori diversi. Per misurarlo si confronta la probabilità sequenziale dei seguiti
con due riferimenti iniziali, uniforme e proporzionale alle frequenze del train.
Il rapporto si aggiorna **dopo** aver osservato ciascun seguito. La validazione
decide la forza di questa correzione. Ogni regione confronta al massimo 128
parole osservate. Le parole sconosciute lasciano disponibili le altre colonne.

### Dal prompt al prossimo token

Il cursore registra «salotto» e completa le finestre che terminano in quella posizione.
Un buffer circolare per scala gli restituisce la metà più vecchia. Le soglie apprese
trasformano ciascuna coppia di metà in una regione, poi il modello tabulare completa
la colonna target. Le somme temporanee del documento leggono gli stessi concetti
e incorporano soltanto seguiti già osservati. Così il seguito incontrato poco prima
o molte pagine prima può cambiare la distribuzione. Dopo avere scelto il prossimo
token, il cursore lo aggiunge al prefisso e ripete il calcolo. Un documento nuovo
riparte con memoria vuota.

Per una finestra massima $W$, le scale richiedono $O(\log W)$ composizioni per token
a geometria fissata e fino a $O(W)$ posizioni nei buffer. Si aggiungono le letture
sparse e il costo di emettere una distribuzione sul vocabolario. La memoria del
documento conserva l'epoca corrente e la precedente, ciascuna lunga fino a $W$ token.
Con venti scale il budget è 524.288 token per epoca. Sul train WikiText-2 l'articolo
più lungo ha 22.891 token e la maggiore scala compositiva addestrata è 16.384.

### Verifiche

Su WikiText-2 con BPE 8.000 il confronto usa 64 blocchi di test, 32.704 previsioni
e gli stessi confini di articolo. Una NLL inferiore indica più probabilità sul token vero.

| Lettura | NLL sul test |
|---|---:|
| Prima tabella diadica, un solo concetto finale | 7,1808 |
| Contatore di riferimento | 4,5000 |
| Tabella con affinamenti CP e memoria del documento | **4,4794** |
| Precedente esperimento con suffissi e gruppi CP | 4,4555 |

La nuova tabella migliora sul contatore in 51 blocchi su 64. Il vantaggio medio è
0,0205 nat/token e l'errore standard appaiato per blocco è 0,0083. Tenendo solo
l'ultimo token e le frequenze del documento si ottiene 4,8501. Rimescolando le
identità dei concetti si sale a 6,9824. Sulle 4.933 posizioni con una coppia
letterale finale mai vista nel train si passa da 4,4170 con il solo token recente
a 4,3424 con il contesto. La differenza maggiore proviene dalle prime quattro
posizioni. Nelle bande 1–31, 32–127, 128–255 e 256–511 la NLL completa è
rispettivamente 4,9508, 4,6564, 4,4571 e 4,3672.

Sulla metà di controllo della validazione, la SVD a otto coordinate ottiene 4,5724
e la rappresentazione con 64 coordinate CP ottiene 4,6870. La SVD è stata scelta
per rappresentare i token. I tagli dei tratti e la lettura del target usano i concetti.
La memoria del documento contribuisce al risultato, con una NLL di 4,8877 quando è
disabilitata. A 300.000 token di distanza si recuperano al primo posto 16 associazioni
su 16, 16 copie su 16 e 16 richieste lessicali identiche su 16. Quando la richiesta usa un sinonimo,
la coordinata della parola porta il risultato da 0 a 8 su 16 a tutte e quattro le
distanze provate. Sui 64 blocchi principali, con e senza questa coordinata, la NLL
arrotondata è 4,4794. Su altri 64 blocchi disgiunti si passa da 4,58650 a 4,58621.
La differenza più netta riguarda il richiamo. Nella prova di associazione il tempo
mediano per generare un token è circa 4 ms dopo 1.000 token e 8 ms dopo 300.000.

Sui dieci dataset tabulari forniti il confronto riproduce le predizioni precedenti.
GiocoSet completo dà quattro gruppi senza etichette. Togliendo la colonna «Gioco»
ne dà sei. Letter raggiunge il 97,80% sulle 4.000 righe tenute fuori dal training.
Il benchmark sul testo usa 2.729.601 token originali. In una prova distinta, la somma
sparsa di un solo livello elabora 1 GiB di coppie regione–target ottenute ripetendo
le righe di training. La coda web resta sul contatore. Il checkpoint della lettura
diadica si può addestrare e interrogare con i comandi di ricerca.
