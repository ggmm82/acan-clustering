"""Explanations describe implemented profiles, including explicit limitations."""
from dash import dcc, html

TEXT = r"""
## Come funziona ACAN

Una riga del dataset diventa un vettore di numeri. Le categorie sono codificate
in colonne e le variabili numeriche normalizzate. **Il target è la colonna che
vuoi prevedere**: durante la valutazione viene nascosto al predittore.

### 1. I gruppi G: riassunti dei dati

$X\in\mathbb{R}^{n\times m}$ contiene n righe e m attributi. Ogni riga di G
rappresenta un gruppo. T indica quanto ogni record appartiene a ciascun gruppo:

$$T_{it}\geq0,\quad \sum_i T_{it}=1,\quad C_i=\sum_t T_{it},\quad
G_i^{post}=\frac{\sum_tT_{it}X_t}{C_i}.$$

Per esempio, un passeggero può appartenere al 70% a un gruppo e al 30% a un altro.
I numeri sono pesi di appartenenza, non probabilità certificate di sopravvivenza.

### 2. La rilevanza R: quali attributi distinguono un gruppo

Nel codice tabulare, R misura lo scostamento del gruppo dalla media della radice,
normalizzato rispetto all'intervallo dei dati e pesato da omega. La CP è calcolata
con la media fra il centroide osservato e il prototipo iniziale/apprendibile.

$$R_{ij}=\omega_j\frac{|G^{eff}_{ij}-Root_j|}{
(max_j-min_j)/2+|Root_j-(min_j+max_j)/2|}.$$

Le colonne costanti hanno rilevanza zero. Omega è ricavato dal training: non è
un parametro appreso dall'ottimizzatore tabulare. La rilevanza duale completa del
paper non è implementata da questa formula.

### 3. Cluster Probability: cosa viene premiato

Poniamo $F_i=C_i/n$ e $\overline R_i=\sum_j R_{ij}/m$. Il codice calcola:

$$CP=\sqrt{\sum_i F_i\frac{\sum_j R_{ij}^2}{\overline R_i}}.$$

I denominatori hanno protezioni numeriche. La CP favorisce gruppi con attributi
caratteristici. **Una CP più alta non garantisce accuracy migliore**: per questo
Evaluation usa righe escluse dal training, accuracy e balanced accuracy.

### 4. Routing: come vengono scelti i gruppi

Il profilo del paper usa una distanza quadratica pesata dalla rilevanza:

$$D_{it}=\frac1m\sum_j\frac{R_{ij}^2}{\operatorname{media}_j(R_{ij}^2)+\epsilon}
(X_{tj}-G_{ij})^2,\qquad
T_{it}=\operatorname{softmax}_i[-\beta D_{it}+\gamma\log(F_i+\epsilon)].$$

La distanza favorisce gruppi compatibili. La massa favorisce quelli già popolati:
se è troppo forte può amplificare squilibri casuali. Il profilo legacy usa una
formula diversa, mantenuta per confrontare modelli esistenti.

### 5. CP diretta: la nuova opzione nell'app

Scegli **Model → Impostazioni avanzate → Routing ACAN → CP diretta** e premi
**Scopri concetti**. Il nuovo profilo ottimizza direttamente T con l'obiettivo:

$$J(T)=CP(T)-8\langle T,D\rangle/n+H(T)/n+
0.5\sum_i[(F_i+\epsilon)\log(F_i+\epsilon)-(F_i+\epsilon)].$$

Qui H è la somma delle entropie delle assegnazioni. Un passo modifica T usando
il gradiente di J; viene accettato soltanto se aumenta l'obiettivo effettivo.
Il valore di CP da solo può anche diminuire quando aumenta J.

Il budget è limitato a 300 righe campionate dal training, 32 slot e 30 passi
(o meno, se scegli meno epoche). I prototipi iniziali restano fissi durante questi
passi. Non si usa Adam sui prototipi in questa modalità. Ogni riga da prevedere
viene valutata indipendentemente usando i riepiloghi congelati del training e
solo le colonne osservate: non si apprendono i dati del test. Il modello salvato
conserva questi riepiloghi e il profilo scelto.

È una nuova formulazione sperimentale ispirata ad ACAN: i coefficienti sono
espliciti e non sono una certificazione dell'intero paper. Confronta i profili
con gli stessi dati e split; avvia il benchmark soltanto quando vuoi misurarlo.

### 6. Gerarchia e laboratorio linguistico

Nel grafo, il doppio clic su un gruppo addestra un figlio sui suoi record. Puoi
rimuovere il ramo e confrontare di nuovo la validazione. I layer dell'LLM sono
trasformazioni successive: non costituiscono automaticamente questo albero.

Nell'LLM ogni token diventa un vettore. I concetti riassumono il prefisso già letto;
una finestra locale conserva anche informazioni sui token recenti. La testa del
modello trasforma il contesto in probabilità per il token successivo.
Il contesto della frase non viene eliminato.

**CP diretta è disponibile nel tabulare e nel laboratorio generativo** per nuovi
progetti: scegli il routing CP diretta nelle impostazioni del rispettivo percorso.
Entrambi usano lo stesso solver e lo stesso obiettivo J. Nel tabulare le statistiche
sono congelate dopo il training; nell'LLM vengono accumulate sul prefisso e salvate
nella cache. Le rappresentazioni linguistiche sono portate in [0,1] con sigmoid
per applicare la stessa rilevanza; omega è reso non negativo con valore assoluto.
Il budget linguistico è di tre passi per token, con ricerca del passo controllata:
su CPU può essere più lento del routing precedente. La finestra locale e la testa
linguistica restano necessarie per generare testo ordinato.

Il training linguistico usa NLL next-token (eventualmente con teacher KL), che
aggiorna i parametri attraverso questo routing. Una linea rossa di validazione che
sale mentre la blu scende suggerisce sovra-adattamento o divergenza dal compito:
confronta il miglior checkpoint, il modello base e gli stessi prompt. CP e NLL
non hanno una soglia universale di bontà. I checkpoint precedenti conservano il
profilo con cui sono stati addestrati; scegliere CP richiede un nuovo progetto.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
