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
La terza identifica il concetto di «sul tappeto del salotto». Per leggere otto token si accostano due tratti di quattro. La stessa operazione raddoppia ancora la lunghezza del contesto. Il punto che potrebbe seguire è un possibile valore
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

### Come due concetti formano un contesto

Nella frase «sul tappeto del salotto», due righe di due token descrivono «sul
tappeto» e «del salotto». Ciascuna contiene le coordinate del primo token seguite
da quelle del secondo. Il modello tabulare individua il concetto di ciascuna
coppia tramite la CP descritta sopra. Il concetto trasmette al livello seguente
il centro delle sue coppie osservate. Per formare una riga di quattro token si
affiancano il centro della coppia più vecchia e il centro di quella recente.
Invertire le due metà cambia la riga. L'apprendimento usa tutte le coppie complete
che ricadono nello stesso articolo.

Un concetto ampio può fornire evidenza per «della cucina» anche con pochi esempi
esatti. L'identità esatta rimane disponibile come colonna categorica. Per esempio
si può leggere il concetto della metà più vecchia insieme al valore esatto della
metà recente. Le frequenze dei seguiti di quella combinazione affinano la
probabilità precedente con la stessa formula $q_i$. Quando la combinazione non
è mai comparsa, $M_i=N_i(y)=0$ e la formula restituisce la probabilità precedente.

Un figlio distingue, dentro il concetto padre, una combinazione più precisa delle due metà. Per esempio, due coppie possono avere lo stesso concetto nella metà sinistra e due identità distinte nella metà destra. Alcuni articoli di training servono a costruire le distribuzioni candidate. Altri articoli dello stesso train servono a giudicarle. Per ciascuna osservazione di giudizio si calcola

$$
g=\log p_{\rm figlio}(y)-\log p_{\rm padre}(y).
$$

Si conserva un figlio con almeno due osservazioni di giudizio quando il guadagno medio supera il proprio errore standard. La CP individua somiglianze nelle colonne osservate. Il giudizio sul token decide quali distinzioni aiutano a prevedere. Una tabella priva di target può fermarsi dopo la scoperta dei concetti. Nel testo il corpus fornisce automaticamente il target spostando la sequenza di una posizione.

Nella memoria del documento si applica la stessa lettura categorica. Qui $N_i(y)$ conta quante volte **prima della posizione corrente** il contesto è ricaduto nel concetto $i$ ed è stato seguito da $y$. La distribuzione di riferimento è la previsione già ottenuta dal corpus. Due prefissi diversi possono perciò condividere i seguiti osservati nel documento se attivano lo stesso concetto o sotto-concetto. Un sinonimo aiuta soltanto se la sua codifica e il resto del prefisso danno sufficiente evidenza alla stessa risposta.

## Informatica

### Dal corpus al checkpoint

Il testo viene diviso per documento fra training, validazione e test. Il tokenizer impara sul training a scomporre le parole in token. Ogni token riceve un identificatore. Le frequenze dei vicini sinistri e destri forniscono un profilo del suo uso. Una SVD comprime questi profili in otto coordinate. L’identificatore rimane disponibile quando è importante distinguere due token con usi simili.

Si prendono due token consecutivi e si affiancano le coordinate nello stesso ordine del testo. Sono le colonne numeriche di una normale tabella ACAN. Il riconoscitore assegna la coppia a un concetto. La media delle coppie assegnate descrive il concetto al livello successivo. Due metà di due token producono una riga di quattro, due metà di quattro una riga di otto. Quando la descrizione del centro supera 32 coordinate, si conservano le direzioni di maggiore varianza fra i centri, pesati per la loro massa. Coppie identiche possono entrare nel training come una sola riga con la loro molteplicità. Al confine del documento una metà incompleta ha codice zero.

Quando compare il token successivo, il modello può completare la riga. La sua categoria aggiorna i conteggi dei contesti che terminavano **prima** del token. I conteggi arrivano sia alle letture generali sia alle combinazioni più specifiche. Le probabilità dei token veri in validazione determinano quanto peso concedere a ogni lettura. La procedura di confronto delle scale congela i pesi già scelti prima di provare una scala maggiore e misura il guadagno su una seconda parte della validazione.

Per la coda lunga, il worker apprende la geometria su finestre di training distribuite lungo il primo file, fino a 262.144 token. Poi percorre per intero i file della coda a pezzi. Ogni posizione aggiorna la distribuzione del target della radice e le tabelle sparse delle letture scelte. Le chiavi delle due metà possono registrare seguiti nuovi. Entro il budget di conteggio, un livello che supera la sua quota conserva le chiavi con più osservazioni per byte. Il budget riguarda queste tabelle. Geometria, memoria del documento e lavoro temporaneo occupano altra RAM.

Il checkpoint contiene il tokenizer nella cartella del progetto, i parametri dei riconoscitori, i centri, le categorie apprese e i loro conteggi, i pesi di lettura e il tratto finale del pezzo. Quest’ultimo permette di riprendere dal token successivo anche se un contesto attraversava il punto di salvataggio. All’inizio di un nuovo file la coda conserva gli apprendimenti precedenti e inizia con un prefisso vuoto.

### Dal prompt al prossimo token

Il tokenizer del checkpoint converte il prompt in identificatori. Il cursore ricostruisce le coppie con gli stessi parametri del training. Legge il prefisso in blocchi, registra i seguiti già incontrati nel documento e tiene in coda i token necessari per comporre la scala più lunga. Per la prossima posizione parte dalle appartenenze del token vicino. Applica nell’ordine le letture di corpus selezionate e poi quelle locali. Il risultato è una probabilità per ciascun token del vocabolario, comprese le alternative mai seguite a quel particolare prefisso.

A temperatura zero si sceglie il token con probabilità massima. Con temperatura positiva si campiona dalla distribuzione ripesata. Solo dopo la scelta il token entra nella memoria e nel prefisso. La stessa sequenza di operazioni produce la posizione successiva. A parametri fissati, il calcolo per un nuovo token dipende dalla finestra appresa, dai conteggi consultati e dal vocabolario. Nessun passo costruisce una matrice di attenzione fra tutte le posizioni del documento.

### Verifiche sul testo

Su WikiText-2 si misurano gli stessi 32.704 token in tutte le righe. Entrambi i modelli possono leggere il prefisso dall’inizio dell’articolo. Il contatore usa fino a sei token vicini come chiave. I valori sono NLL in nat/token, con memoria del documento spenta.

| Scala provata | Coppie sul controllo | Coppie sul test | Contatore sul test |
|---:|---:|---:|---:|
| 6 | 4,91266 | 4,84475 | 4,91300 |
| 8 | 4,91098 | 4,84207 | 4,91300 |
| 16 | 4,90990 | 4,84097 | 4,91300 |
| 32 | 4,90987 | 4,84090 | 4,91300 |
| 64 | 4,90987 | 4,84090 | 4,91300 |
| 128 | 4,90913 | 4,84216 | 4,91300 |
| 256 | 4,90913 | 4,84216 | 4,91300 |
| 512 | 4,90913 | 4,84216 | 4,91300 |


Il controllo seleziona la scala 128. Le prove a 256 e 512 mantengono la stessa distribuzione. Sul test, il valore più basso della serie si osserva a 32. Il guadagno fra 6 e 32 è 0,00385 nat/token. Il guadagno maggiore rispetto al contatore si vede già a sei token.

A 32 token, la memoria delle sole identità esatte porta la NLL di test a 4,08816. La memoria dei soli concetti e sotto-concetti dà 4,30614. La validazione combina le due fonti e il test scende a 4,08543, contro 4,11082 del contatore con memoria. Nel test di richiamo si assegna un seguito casuale a ciascuna di sedici espressioni. Con una richiesta sinonima, la lettura combinata trova la risposta 2 volte su 16 fra i candidati a ciascuna distanza fra 1.000 e 300.000 token. Sull’intero vocabolario la risposta arriva prima 0 volte su 16. Ricordare un seguito letterale e riconoscere una parafrasi restano capacità distinte. [Risultati per blocco, scala e compito](../experiments/text_table/results/native-long-context.json).
