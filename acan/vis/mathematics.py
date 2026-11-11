"""Implemented mathematics and explicit source-paper correspondence."""
from dash import dcc, html

TEXT = r"""
## Un unico nucleo matematico basato sulla CP

Scegli **Modello congiunto NLL + CP** (`cp-joint-v1`) nelle impostazioni del
modello tabulare e del laboratorio generativo. Entrambi chiamano le stesse
funzioni in `acan/algebra.py`: `concept_assignments`, `concept_step`,
`normalized_cluster_probability` e `prediction_cp_objective`.
Il tabulare usa NumPy con gradienti analitici; il generativo usa PyTorch con
autograd. I test confrontano valori, loss e gradienti sugli stessi dati.

Il modello comune impara a **prevedere bene mantenendo organizzati i concetti**.
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
T_{it}=\operatorname{softmax}_i(-\beta D_{it}),\quad
\sum_iT_{it}=1.
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

### 4. La stessa loss: previsione più organizzazione dei concetti

$$
\mathcal L=\mathcal L_{previsione}+\lambda(1-\overline{CP}).
$$

Nel tabulare supervisionato $\mathcal L_{previsione}$ è la NLL della classe
corretta, prevista mescolando con T le probabilità dei prototipi. Con più target
categorici si fa la media delle loro NLL. Il coefficiente tabulare è **0,1**.
Il profilo congiunto richiede target categorici: per regressione sono disponibili
i profili precedenti. Senza target si ottimizza il solo termine CP.

Nel linguaggio la stessa loss diventa:

$$
\mathcal L_{LM}=-\frac1N\sum_t\log p(w_{t+1}\mid w_{\leq t})
+\lambda\,\operatorname{media}_{\ell,t}(1-\overline{CP}_{\ell,t}).
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

Ogni figlio della gerarchia viene addestrato con lo stesso nucleo sui suoi record.
La scelta di approfondire usa la purezza delle etichette nei record di training,
non la sicurezza delle probabilità apprese dei prototipi.
I layer linguistici sono trasformazioni successive: non sono automaticamente
l'albero tabulare. Nel generativo il budget CPU del profilo congiunto è 32 token
per blocco e 8 slot per layer; ogni token esegue un passo del nucleo comune.
Le diverse dimensioni e i budget sono configurazioni, non equazioni diverse.

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

### 7. Cosa dimostrano le verifiche

Verifichiamo parità NumPy/PyTorch di valori e gradienti, differenze finite,
CP causale, streaming, persistenza, gerarchia e ripresa del training.
Accuracy e NLL devono essere misurate su dati esclusi dal training.
**Il 95,78% gerarchico citato nella sessione precedente non è un risultato recuperato né
una prova di miglioramento dell'LLM.** Le nuove misure riproducibili e il confronto
linguistico con sola NLL sono documentati in `docs/JOINT-CP.md`.
"""

layout = html.Div(dcc.Markdown(TEXT, mathjax=True), className='p-3', style={'maxWidth':'960px','margin':'auto'})
