"""Implemented mathematics and explicit source-paper correspondence."""
from dash import dcc, html

TEXT = r"""
## Un unico nucleo matematico basato sulla CP

Il modello **CP condivisa con selezione dei concetti** (`cp-joint-v1`) è usato nei nuovi training del
tabulare e del laboratorio generativo. Entrambi chiamano le stesse
funzioni in `acan/algebra.py`: `concept_assignments`, `concept_step`,
`normalized_cluster_probability`, `exclusive_cluster_probability`, `mass_assignments` e `prediction_cp_objective`.
Il tabulare usa NumPy con gradienti analitici; il generativo usa PyTorch con
autograd. I test confrontano valori, loss e gradienti sugli stessi dati.

La proposta è un’estensione **sperimentale** della CP degli studi originali, identificata nei nuovi checkpoint da `exclusive-cp-v1`. Non è una dimostrazione di miglioramento universale. Il modello impara a prevedere il target e a organizzare i concetti.
La CP entra esplicitamente nella loss e il suo gradiente aggiorna anche i prototipi
che determinano il routing. Non è soltanto un indicatore mostrato nel grafico.

### 1. Rappresentazioni, prototipi e attributi osservabili

$X\in[0,1]^{n\times d}$ contiene righe normalizzate oppure rappresentazioni dei
token trasformate con sigmoid. $A\in[0,1]^{k\times d}$ contiene i prototipi
apprendibili. Nel tabulare le colonne categoriche del target hanno probabilità
normalizzate con softmax; non vengono usate per assegnare una riga a un concetto.

Root, min e max sono statistiche **del solo training** nel tabulare; nell'LLM
sono statistiche **del solo prefisso già letto**, aggiornate a ogni token e layer.
Il coefficiente non negativo $\omega$ pesa i canali: è stimato dal training nel
tabulare e apprendibile nel generativo. Le equazioni del nucleo sono le stesse.

### 2. La stessa rilevanza e lo stesso routing

$$
\Delta_j=(max_j-min_j)/2+|Root_j-(min_j+max_j)/2|,
$$

$$
R^{route}_{ij}=\operatorname{clip}_{[0,1]}
\left(\omega_j\frac{|A_{ij}-Root_j|}{\Delta_j}\right).
$$

Quando $\Delta_j$ è zero, il canale ha rilevanza zero. Per il sottoinsieme O degli
attributi osservabili, poniamo:

$$
W_{ij}=\frac{(R^{route}_{ij})^2+10^{-4}}
{\sum_{q\in O}[(R^{route}_{iq})^2+10^{-4}]},\qquad
D_{it}=\sum_{j\in O}W_{ij}(X_{tj}-A_{ij})^2,
$$

$$
\pi_i\geq0,\qquad \sum_i\pi_i=1,\qquad
T_{it}=\frac{\pi_i\exp(-\beta D_{it})}{\sum_q\pi_q\exp(-\beta D_{qt})}.
$$

Il routing vede soltanto gli attributi osservabili. Nel tabulare il target resta
escluso anche quando si dividono i record per addestrare i figli della gerarchia.
Se non ci sono attributi osservati, l'inferenza usa le frequenze del training.
Non viene eseguito alcun aggiornamento sul test.

### 3. Gli stessi riepiloghi dei concetti e la stessa CP

$$
C_i=C_i^{precedente}+\sum_t T_{it},\qquad
U_i=U_i^{precedente}+\sum_tT_{it}X_t,\qquad G_i=U_i/C_i.
$$

Nel training tabulare un batch parte da riepiloghi vuoti. Nel generativo, al
passo t, i riepiloghi precedenti contengono soltanto i token prima di t.
Le assegnazioni passate restano congelate; nessun token futuro entra nella CP.
La frequenza è $F_i=C_i/\sum_q C_q$. Usiamo $G^{eff}=(G+A)/2$ per calcolare R
con la stessa formula limitata a [0,1] del punto precedente.

$$
CP=\sqrt{\sum_i F_i\frac{\sum_j R_{ij}^2}{(\sum_j R_{ij})/d}},\qquad
\overline{CP}=\frac{CP}{\sqrt d}
=\sqrt{\sum_i F_i\frac{\sum_jR_{ij}^2}{\sum_jR_{ij}}}.
$$

Il codice protegge i denominatori con epsilon. La normalizzazione elimina la
crescita automatica con il numero di canali; il clipping evita rilevanze illimitate
quando un prototipo cade fuori dall'intervallo di un prefisso ancora corto.
La CP normalizzata resta in [0,1], a parte il pavimento numerico $10^{-6}$.

La CP originale non distingue un concetto da copie identiche che ne dividono la frequenza. La proposta aggiunge **esclusività** e **coerenza delle probabilità**, usando la stessa distanza già presente nel routing:

$$
q_i=\frac{\sum_jR_{ij}^2}{\sum_jR_{ij}},\quad z_i=F_i\sqrt{q_i},\quad
S_{ii}=0,\quad S_{ij}=\exp\left[-\frac\beta2(D_{ij}+D_{ji})\right]\quad(i\ne j).
$$

Qui la distanza fra prototipi usa gli stessi pesi W del routing. L’affinità di Bhattacharyya confronta le masse apprese con le frequenze osservate:

$$
B(\pi,F)=\sum_i\sqrt{\pi_iF_i},\qquad
CP_* = B(\pi,F)\sqrt{\sum_iF_iq_i-z^TSz}.
$$

È un indice di organizzazione, **non una probabilità calibrata della classe**. La costruzione è limitata: poiché ogni S è fra zero e uno e F somma a uno, il radicando è non negativo e non supera la CP originale al quadrato; per Cauchy–Schwarz anche B è fra zero e uno. Quindi CP* è fra zero e uno, salvo i pavimenti numerici.
Con r copie identiche, qualità q e masse uguali alle frequenze, si ottiene CP* = sqrt(q/r), mentre un unico rappresentante vale sqrt(q). Questo rompe la neutralità dei duplicati senza prescrivere il numero di concetti. **Non prova che l’ottimizzazione non convessa trovi sempre il numero semanticamente corretto.**

### 4. La stessa loss: previsione più organizzazione dei concetti

$$
\mathcal L=\mathcal L_{previsione}+\lambda(1-CP_*).
$$

Nel tabulare supervisionato $\mathcal L_{previsione}$ è la NLL della classe
corretta, prevista mescolando con T le probabilità dei prototipi. Con più target
categorici si fa la media delle loro NLL. Il coefficiente tabulare è **0,1**.
Il profilo congiunto richiede target categorici: per regressione sono disponibili
i profili precedenti. Senza target si ottimizza il solo termine CP.

Nel linguaggio la stessa loss diventa:

$$
\mathcal L_{LM}=-\frac1N\sum_t\log p(w_{t+1}\mid w_{\leq t})
+\lambda\,\operatorname{media}_{\ell,t}(1-CP_{*,\ell,t}).
$$

Si fa la media sui **layer ACAN**, sul batch e sulle posizioni con un token
successivo da prevedere. L'ultima posizione del blocco non ha un target next-token
e viene esclusa. Duplicare i layer o il batch non moltiplica la penalità.
Il generativo conserva una testa linguistica e una finestra locale causale per
produrre testo: questi sono adattatori di lettura, non formule CP alternative.

Il **peso CP è configurabile** nei nuovi progetti generativi; la GUI propone
0,1 come esperimento, non come valore ottimale dimostrato. Con **0** si ottiene
il controllo con sola NLL. Con distillazione abilitata, il termine di previsione
è la combinazione NLL/KL dichiarata nel progetto, cui si aggiunge la stessa CP.
Validazione e scelta del checkpoint migliore restano basate sulla NLL separata.

### 5. Training, persistenza e gerarchia

Il tabulare congiunto usa tutte le righe di training, batch fino a 300 e fino a
256 prototipi; le epoche sono quelle richieste nell'app. Adam aggiorna prototipi,
probabilità delle classi e temperatura del routing. Il checkpoint è selezionato
sull'obiettivo congiunto di training, senza consultare il test. JSON conserva
prototipi, rilevanza del routing, temperatura, frequenze, obiettivo e profilo.

Le masse sono parametri del simplesso, aggiornati tramite proiezione euclidea e ottimizzazione per coordinate dello **stesso obiettivo di training**. Gli estremi con massa zero sono candidati numerici: vengono accettati solo se l’obiettivo non aumenta. La direzione usa la sovrapposizione S già definita dalla CP. Non vengono mediati i prototipi né usate soglie di similarità per unirli.

$$
K_{attivo}=\left|\{i:\pi_i>0\}\right|.
$$

Tabulare e generativo chiamano gli stessi operatori di proiezione e di scelta delle coordinate. Nel tabulare il controllo delle masse considera tutto il training; nel generativo i batch di training, mantenendo il dropout identico nei confronti e conservando lo stato casuale per la ripresa. I rispettivi adattatori e ottimizzatori non sono bit per bit identici.
Le masse esattamente zero sono eliminate dal modello tabulare salvato e mascherate negli slot linguistici. I budget 256 e 8 sono limiti di capacità, non numeri di concetti imposti. Non sono state introdotte regole per Gioco, Letter o altri dataset.

Ogni figlio della gerarchia viene addestrato con lo stesso nucleo sui suoi record.
La scelta di approfondire usa la purezza delle etichette nei record di training,
non la sicurezza delle probabilità apprese dei prototipi.
Nei nuovi modelli, la previsione somma anche i rami della gerarchia con le loro
probabilità, conservando la miscela appresa durante il training:

$$
p_v(y\mid x)=\sum_i T_{v,i}(x)\,p_{v,i}(y\mid x).
$$

Qui p del concetto è la sua testa categorica se terminale, oppure la previsione
del figlio. Espandere un concetto con un figlio dalla stessa distribuzione non
cambia la previsione. Il percorso visualizzato è il percorso completo più
probabile; la sua probabilità non è una confidenza calibrata della classe.
La testa categorica si inizializza con log(p), inversa della softmax. I JSON
precedenti conservano il vecchio comportamento; riaddestrare per usare la
marginalizzazione, identificata da `hierarchy_readout=conditional-mixture-v1`.
I layer linguistici sono trasformazioni successive: non sono automaticamente
l'albero tabulare. Nel generativo il budget CPU del profilo congiunto è 32 token
per blocco e 8 slot per layer; ogni token esegue un passo del nucleo comune.
Le diverse dimensioni e i budget sono configurazioni, non equazioni diverse. Il confronto fra prototipi della nuova CP aggiunge lavoro O(k²d) al training. In inferenza linguistica tale termine viene saltato senza cambiare l’output: il costo ricorrente resta O(kd + wd), con memoria indipendente dalla lunghezza del prefisso. Il training tabulare è più lento della versione precedente.

### 6. Collegamento con i due documenti di partenza

**Algebraic Concept Attention Scientific Paper**, §3, fornisce il ponte fra
prototipi, rilevanza, competizione fra concetti e CP. **ACAN-LM vs. MiniCPM**,
§§2.1–2.2, fornisce gli slot linguistici, i riepiloghi cumulativi e la finestra
locale. I PDF allegati coincidono con quelli conservati in `paper/`.

`cp-joint-v1` è una formulazione comune esplicita ispirata a questi elementi:
condivide la rilevanza root-shift limitata, il routing a distanza, i riepiloghi e
la CP, e aggiunge la loss supervisionata normalizzata. Non riproduce integralmente
il routing a due stadi con prior di massa e termine bilineare del paper LM,
né la rilevanza duale shift/tightness del paper tabulare. In questo profilo quei
termini non vengono aggiunti soltanto a uno dei due percorsi. La causalità impone
statistiche di prefisso al posto delle statistiche globali della frase.

I profili precedenti (`legacy-v8`, `paper-routing-v1`, `cp-routing-v1`,
`shared-algebra-v1`) restano disponibili per confronto e caricamento dei modelli
esistenti. Scegliere il nuovo nucleo richiede un nuovo training; i checkpoint non
vengono convertiti silenziosamente.

La revisione dei teoremi dei documenti ha inoltre isolato due precisazioni. Per slot coincidenti, la dinamica gravitazionale ideale soddisfa log(Ca/Cb) al passo successivo = gamma log(Ca/Cb): la simmetria è instabile per gamma > 1, mentre gamma = 1 è neutro. Inoltre, sul rumore uniforme, max(0, 1 - 2 sigma) vale circa 0,423, non zero; non basta quindi a dimostrare il decadimento della rilevanza dichiarato nel paper. Queste osservazioni **non costituiscono una nuova prova dei teoremi complessivi**, né la proposta implementa integralmente quella dinamica gravitazionale. La selezione qui descritta è l’estensione CP* esplicita sopra.

### 7. Cosa dimostrano le verifiche

Verifichiamo parità NumPy/PyTorch di valori e gradienti, differenze finite,
CP causale, streaming, persistenza, gerarchia e ripresa del training.
Accuracy e NLL devono essere misurate su dati esclusi dal training.
**Il 95,78% gerarchico citato nella sessione precedente non è un risultato recuperato né
una prova di miglioramento dell'LLM.** Le nuove misure riproducibili e il confronto
linguistico con sola NLL sono documentati in `docs/JOINT-CP.md`.

La proposta conserva **4 concetti su Gioco** sui cinque split controllati. Dopo la correzione della lettura gerarchica e dell'inizializzazione, Letter completo misura **92,70% su 4.000 righe di test**; il campione GUI misura **89,85%** (PR8: 91,425% e 90,60%). Il 95,78% storico resta non recuperato. Adult e Titanic migliorano nella media dei tre split iniziali; Breast Cancer resta debole, Iris e Vote peggiorano e lo split aggiuntivo mostra variabilità. I 63 confronti iniziali e le 35 esecuzioni successive sono documentati, insieme a limiti e ricevute, in `docs/CP-SELECTION.md`. Sono prove diagnostiche sugli stessi file, non una garanzia generale. Per il generativo sono verificate le proprietà meccaniche; la qualità linguistica su modelli preaddestrati resta da dimostrare.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
