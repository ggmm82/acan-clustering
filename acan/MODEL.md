# ACAN 10.0 · Modello Algebrico Unificato a Concetti per Tabelle e Linguaggio

## Matematica

**Sommario.** *ACAN 10.0 formula l'apprendimento su dati tabulari e la modellazione generativa del linguaggio come due istanze di un unico modello probabilistico basato su concetti nella famiglia esponenziale. Ogni osservazione — una riga di tabella con attributi parzialmente noti o una posizione di un testo preceduta dal suo prefisso — è descritta dall'appartenenza a posteriori a un insieme di concetti rappresentati tramite statistiche sufficienti additive. L'apprendimento dei concetti ubbidisce a una sola equazione, il passo del gradiente naturale di Fisher sui dati completi nello spazio delle attese $S_k' = (1 - \rho)\,S_k + \rho\,\widehat{S}_k$: quando le osservazioni sono fisse ($\rho = 1$, tabelle) essa coincide esattamente con il passo M dell'algoritmo Expectation-Maximization; quando le rappresentazioni evolvono attraverso strati profondi e i dati scorrono in flussi ($0 < \rho < 1$, linguaggio), essa fornisce l'aggiornamento online di Robbins-Monro dei concetti latenti accoppiato alla discesa del gradiente predittivo sulle proiezioni.*

---

### 1 · Formulazione unificata: dalle tabelle alle sequenze di token

Un sistema che apprende dai dati risolve un unico problema fondamentale: **osservata una parte di una configurazione strutturata, inferire la distribuzione di probabilità delle componenti mancanti attraverso un insieme finito e ispezionabile di concetti**.

Sia $\mathcal{X} = \mathcal{X}_1 \times \dots \times \mathcal{X}_m$ uno spazio prodotto di attributi numerici ($\mathcal{X}_j = \mathbb{R}$) e categoriali ($\mathcal{X}_j = \{1, \dots, V_j\}$). Ogni esempio consiste in una tupla $x \in \mathcal{X}$ osservata su un sottoinsieme di coordinate note $\Omega \subseteq \{1, \dots, m\}$, e il compito predittivo consiste nello stimare $p(x_j \mid x_\Omega)$ per ogni coordinata ignota $j \notin \Omega$.

Questa astrazione comprende senza modifiche sia il dominio tabulare sia quello testuale:

- **Dominio tabulare.** Una tabella di $n$ righe fornisce osservazioni indipendenti $\{x_r\}_{r=1}^n$ su $m$ colonne fisse, con eventuali valori mancanti ($\Omega_r \subset \{1, \dots, m\}$). Un'unica distribuzione congiunta appresa su tutte le colonne risponde a qualsiasi interrogazione condizionata $p(x_{r,j} \mid x_{r,\Omega_r})$.
- **Dominio sequenziale e linguistico.** Dato un vocabolario finito $\mathcal{V} = \{1, \dots, V\}$ con simbolo di confine documento $\mathrm{eos} \in \mathcal{V}$, un documento $(y_1, \dots, y_T)$ preceduto da $y_0 = \mathrm{eos}$ genera, per la regola della catena, $T$ problemi causali di completamento:

$$
\log p(y_1, \dots, y_T) = \sum_{t=0}^{T-1} \log p(y_{t+1} \mid y_0, \dots, y_t).
$$

  In ogni posizione $t$, il prefisso osservato $y_{\le t} = (y_0, \dots, y_t)$ costituisce la parte nota della riga, riassunta in uno stato continuo $h_t \in \mathbb{R}^D$, mentre il token successivo $y_{t+1} \in \mathcal{V}$ è l'unica colonna categoriale da completare.

| Elemento formale | Istanza tabulare (`ConceptModel`) | Istanza generativa (`NetText` / `ConceptNet`) |
|---|---|---|
| Osservazione elementare | Riga $x_r$ su colonne note $\Omega_r$ | Posizione $t$ con prefisso causale $y_{\le t}$ |
| Oggetto elementare | Concetto $(C_k, \mu_k, \sigma_k^2, N_k)$ sulle colonne | Concetto $(C_k, g_k, s_k^2)$ in ogni strato e concetto-token $(C_y, e_y, \sigma^2)$ |
| Appartenenza latente | $T_k(x_\Omega) \propto C_k\,L_k(x_\Omega)$ | $T_k(q_t) \propto C_k\,\mathcal{N}(q_t;\,g_k, \operatorname{diag}(s_k^2))$ |
| Regola sui concetti | $S_k' = (1-\rho)S_k + \rho\widehat{S}_k$ con $\rho = 1$ (EM esatto) | $S_k' = (1-\rho_t)S_k + \rho_t\widehat{S}_k$ con $0 < \rho_t < 1$ + $\nabla\mathcal{L}$ sulle proiezioni |
| Scelta dell'architettura | Agglomerazione gerarchica + leave-one-out a 1 SE | Morfogenesi conservativa (larghezza, strati, concetti, teste) a 1 SE |
| Inferenza con memoria | Media pesata / stimatore gerarchico di Dirichlet ($\kappa$) | Lettura gerarchica a due livelli + memoria episodica con sentinella ($e^{s_{t0}} \equiv \kappa$) |

---

### 2 · Geometria dei concetti, statistiche sufficienti ed embedding dei token

#### 2.1 · Statistiche sufficienti additive

Un **concetto** di indice $k \in \{1, \dots, K\}$ è definito esclusivamente dalle sue statistiche sufficienti del primo e secondo ordine, accumulate con pesi di appartenenza $T_{rk} \in [0, 1]$:

$$
C_k = \sum_{r=1}^{n} T_{rk}, \qquad
U_{kj} = \sum_{r:\,j\in\Omega_r} T_{rk}\,x_{rj}, \qquad
Q_{kj} = \sum_{r:\,j\in\Omega_r} T_{rk}\,x_{rj}^2, \qquad
N_{kj}(v) = \sum_{r:\,j\in\Omega_r} T_{rk}\,[x_{rj} = v].
$$

Poiché l'operatore $S_k = (C_k, U_k, Q_k, N_k)$ è lineare nelle osservazioni, soddisfa l'**additività esatta di partizione**: $S_k(\mathcal{D}_1 \cup \mathcal{D}_2) = S_k(\mathcal{D}_1) + S_k(\mathcal{D}_2)$. Due concetti si fondono sommando i rispettivi vettori $S_k$, e una tabella suddivisa in blocchi arbitrari produce statistiche identiche alla lettura monolitica.

Indicando con $C_{kj}^{\mathrm{obs}} = \sum_{r:\,j\in\Omega_r} T_{rk}$ la massa effettivamente osservata sulla colonna numerica $j$ e con $\sigma_{0,j}^2$ la varianza di riferimento del livello superiore, centro e varianza regolarizzati con forza a priori $\kappa \ge 0$ sono dati in forma chiusa da (`centre_variance`):

$$
\mu_{kj} = \frac{U_{kj}}{\max(C_{kj}^{\mathrm{obs}},\,\varepsilon)}, \qquad
\sigma_{kj}^2 = \frac{\max\!\big(Q_{kj} - C_{kj}^{\mathrm{obs}}\mu_{kj}^2,\,0\big) + \kappa\,\sigma_{0,j}^2}{C_{kj}^{\mathrm{obs}} + \kappa}.
$$

Quando un attributo numerico assume pochi valori discreti ravvicinati, un pavimento di risoluzione proporzionale alla spaziatura minima impedisce il collasso verso masse puntuali degeneri ($\sigma_{kj} \to 0$).

#### 2.2 · Embedding dei token come concetti gaussiani e fattorizzazione per frequenza

Nei modelli neurali standard la matrice di *embedding* e il proiettore lineare di uscita (*LM head*) sono parametrizzati empiricamente per prodotto scalare. In ACAN 10.0 essi derivano direttamente dalla geometria dei concetti: ogni token $y \in \{1, \dots, V\}$ è un concetto isotropo in $\mathbb{R}^D$ dotato di centro (**embedding**) $e_y \in \mathbb{R}^D$, massa a priori $C_y > 0$ (inizializzata dalla frequenza empirica del corpus) e deviazione standard condivisa $\sigma = e^{\log s}$.

Dato lo stato finale $h \in \mathbb{R}^D$ prodotto dalla rete per il prefisso corrente, la probabilità a posteriori che $h$ appartenga al concetto del token $y$ si ottiene applicando la regola di Bayes:

$$
p_{\mathrm{rete}}(y \mid h)
= \frac{C_y\,\mathcal{N}(h;\,e_y, \sigma^2 I_D)}{\sum_{y'=1}^{V} C_{y'}\,\mathcal{N}(h;\,e_{y'}, \sigma^2 I_D)}
= \operatorname{softmax}_y\!\left(\frac{h^\top e_y - \tfrac{1}{2}\lVert e_y\rVert_2^2}{\sigma^2} + \log C_y\right),
$$

poiché il termine quadratico $-\lVert h\rVert_2^2 / (2\sigma^2)$ e la costante di normalizzazione gaussiana non dipendono da $y$ e si elidono esattamente nel rapporto. Il termine $-\tfrac{1}{2}\lVert e_y\rVert_2^2/\sigma^2 + \log C_y$ agisce come intercetta intrinseca geometricamente vincolata: token rari o lontani dall'origine non possono gonfiare arbitrariamente i propri logit aumentando solo la norma euclidea di $e_y$.

Per abbattere il costo $\mathcal{O}(VD)$ su grandi vocabolari ($V = 32.000$), il vocabolario ordinato per frequenza decrescente viene partizionato in $G \approx \sqrt{V}$ **concetti-gruppo** disgiunti $\{G_1, \dots, G_G\}$, ciascuno con centro $c_g \in \mathbb{R}^D$ e massa $M_g = \sum_{y \in G_g} C_y$. La distribuzione sul vocabolario si fattorizza esattamente in due letture di appartenenza annidate:

$$
p_{\mathrm{rete}}(y \mid h) = p(G_y \mid h)\,p(y \mid h, G_y),
$$

riducendo il numero di confronti vettoriali per posizione da $V$ a circa $2\sqrt{V}$ ($\approx 360$ per $V = 32.000$).

---

### 3 · Appartenenza a posteriori, attenzione causale e memoria episodica

#### 3.1 · Operatore di appartenenza

Sia $L_k(z)$ la verosimiglianza del concetto $k$ sull'osservazione $z$ (prodotto delle densità gaussiane sulle coordinate continue note e delle probabilità multinomiali sulle coordinate categoriali note). La quota a posteriori $T_k(z)$ è:

$$
T_k(z) = \frac{C_k\,L_k(z)}{\sum_{k'=1}^{K} C_{k'}\,L_{k'}(z)}.
$$

Nel caso gaussiano diagonale su vettori $q \in \mathbb{R}^d$, il log-numeratore sviluppa il quadrato senza mai costruire tensori tridimensionali posizione $\times$ concetto $\times$ dimensione:

$$
\log\big(C_k\,\mathcal{N}(q;\,g_k, \operatorname{diag}(s_k^2))\big)
= \log C_k - \frac{1}{2}\left(\sum_{r=1}^{d} q_r^2 s_{kr}^{-2} - 2\sum_{r=1}^{d} q_r (g_{kr} s_{kr}^{-2}) + \sum_{r=1}^{d} g_{kr}^2 s_{kr}^{-2}\right) - \sum_{r=1}^{d} \log s_{kr} - \frac{d}{2}\log(2\pi).
$$

#### 3.2 · Strato di attenzione causale fra concetti

Una rete di profondità $L$ trasforma la sequenza degli embedding iniziali $h_i^{(0)} = e_{y_i} \in \mathbb{R}^D$ attraverso $L$ strati identici nella forma. In ciascuno strato $\ell$, lo spazio è suddiviso in $H$ teste indipendenti di dimensione $d = D / H$, ognuna dotata di $K_\ell$ concetti gaussiani $\{(C_{hk}, g_{hk}, s_{hk})\}_{k=1}^{K_\ell}$.

Alla posizione $i \in \{0, \dots, T-1\}$, detto $h_i \in \mathbb{R}^D$ lo stato in ingresso allo strato:
1. **Proiezione e appartenenza:** si calcolano interrogazione $q_i = W_Q h_i \in \mathbb{R}^{H \times d}$, valore $v_i = W_V h_i \in \mathbb{R}^{H \times d}$ e le appartenenze $T_{hk}(q_{i,h})$ su ciascuna testa.
2. **Scrittura selettiva causale:** non tutte le posizioni di un testo portano uguale contenuto informativo. Una direzione normalizzata $\widehat{h}_i = \sqrt{D}\,h_i / \max(\lVert h_i\rVert_2, \varepsilon)$ modula una forza di scrittura limitata per testa:

$$
a_{i,h} = \frac{1}{2} + \sigma(u_h^\top \widehat{h}_i) \in \left(\frac{1}{2},\,\frac{3}{2}\right),
$$

   dove all'inizializzazione $u_h = 0$ restituisce la media uniforme $a_{i,h} = 1$. Ogni concetto accumula lungo il prefisso $j \le i$ la massa causale e il momento primo dei valori:

$$
M_{hk}(i) = \varepsilon + \sum_{j=0}^{i} a_{j,h}\,T_{hk}(q_{j,h}), \qquad
S_{hk}(i) = \sum_{j=0}^{i} a_{j,h}\,T_{hk}(q_{j,h})\,v_{j,h}.
$$

3. **Lettura relazionale fra concetti:** la posizione $i$ interroga i baricentri causali $S_{hk'}(i)/M_{hk'}(i)$ attraverso una matrice stocastica di indirizzamento concettuale $\tilde{A}_{h,kk'} = \operatorname{softmax}_{k'}(A_{h,kk'})$ (inizializzata a $A_h = 1{,}5\,I_{K_\ell}$ per $K_\ell > 1$, così che ogni concetto legga in partenza il proprio riassunto anziché collassare sulla media uniforme $1/K_\ell$):

$$
r_{i,h} = \sum_{k=1}^{K_\ell} T_{hk}(q_{i,h}) \sum_{k'=1}^{K_\ell} \tilde{A}_{h,kk'}\,\frac{S_{hk'}(i)}{M_{hk'}(i)}.
$$

4. **Contesto locale a finestra e apertura residua:** l'ordine sintattico a corto raggio delle ultime $W$ posizioni è catturato da una convoluzione causale direzionale $c_i^{\mathrm{near}} = \sum_{w=1}^{W} h_{i-w} W_w^{\mathrm{near}}$ (con $h_{<0} = 0$). Lo stato aggiornato è regolato da uno scalare d'apertura apprendibile $g_\ell \in \mathbb{R}$:

$$
\tilde{h}_i = \mathrm{LN}_2\!\Big(\mathrm{LN}_1\!\big(h_i + W_O[r_i \,\|\, c_i^{\mathrm{near}}] + b_O\big)\Big), \qquad
h_i' = h_i + g_\ell\,(\tilde{h}_i - h_i).
$$

Il costo per token di questo strato dipende esclusivamente dal numero di concetti $H K_\ell d$ e dalla finestra $W$, restando $\mathcal{O}(1)$ rispetto alla lunghezza $T$ del documento.

#### 3.3 · Memoria episodica del documento con sentinella

Poiché $(M_{hk}, S_{hk})$ comprimono il passato remoto in medie sufficienti di concetto, i dettagli lessicali puntuali (nomi propri, identificatori di codice, cifre apparse una sola volta nel prefisso) sono preservati da una **memoria episodica non parametrica**. Ogni posizione passata $j < i$ registra un indirizzo chiave $k_j^{\mathrm{mem}} = W_K^{\mathrm{mem}} h_j^{(L)}$ associato al token effettivamente seguito $y_{j+1}$. La posizione corrente $i$, con interrogazione $q_i^{\mathrm{mem}} = W_Q^{\mathrm{mem}} h_i^{(L)}$, confronta le affinità $s_{ij} = (q_i^{\mathrm{mem}})^\top k_j^{\mathrm{mem}} / \sqrt{D}$ con un indirizzo sentinella appreso $s_{i0} = w_0^\top h_i^{(L)} + b_0$ («nessuna posizione precedente somiglia alla mia»):

$$
p(y \mid y_{\le i}) = \alpha_{i0}\,p_{\mathrm{rete}}(y \mid h_i^{(L)}) + \sum_{j=0}^{i-1} \alpha_{ij}\,[y_{j+1} = y],
\qquad
\alpha_{ij} = \frac{e^{s_{ij}}}{e^{s_{i0}} + \sum_{j'=0}^{i-1} e^{s_{ij'}}},
\quad
\alpha_{i0} = \frac{e^{s_{i0}}}{e^{s_{i0}} + \sum_{j'=0}^{i-1} e^{s_{ij'}}}.
$$

Ponendo $E_i(y) = \sum_{j < i} e^{s_{ij}}[y_{j+1} = y]$, $M_i^{\mathrm{mem}} = \sum_{j < i} e^{s_{ij}}$ e $\kappa_i = e^{s_{i0}}$, l'equazione assume la forma identica dello stimatore gerarchico tabulare:

$$
p(y \mid y_{\le i}) = \frac{E_i(y) + \kappa_i\,p_{\mathrm{rete}}(y \mid h_i^{(L)})}{M_i^{\mathrm{mem}} + \kappa_i}.
$$

Quando nessun passo precedente presenta forte affinità ($M_i^{\mathrm{mem}} \ll \kappa_i$, come all'inizio di un turno di risposta), la distribuzione coincide con quella generale della rete $p_{\mathrm{rete}}$; quando invece compare un precedente quasi identico ($e^{s_{ij}} \gg \kappa_i$), il modello recupera con precisione il seguito osservato nel documento.

---

### 4 · Teorema di unificazione dell'apprendimento: gradiente naturale di Fisher ed EM

In precedenza il modulo tabulare aggiornava i concetti tramite EM mentre il modulo generativo si affidava alla sola retropropagazione euclidea. ACAN 10.0 dimostra e implementa in `acan/concept_algebra.py` (`natural_em_step`, `natural_em_parameters`) che entrambi sono casi particolari di un unico operatore geometrico.

**Proposizione (Identità fra gradiente naturale sui dati completi e passo EM).** *Si consideri una famiglia di $K$ concetti gaussiani diagonali con parametri $\theta_k = (\log c_k,\, g_k,\, \log s_k)$, pesi a priori $\pi_k = c_k / \sum_h c_h$, varianze $\sigma_k^2 = e^{2\log s_k}$ e log-verosimiglianza marginale media su un lotto di $n$ interrogazioni $\{z_t\}_{t=1}^n$:*

$$
\mathcal{L}(\theta) = \frac{1}{n}\sum_{t=1}^{n} \log \sum_{k=1}^{K} \pi_k\,\mathcal{N}\big(z_t;\,g_k, \operatorname{diag}(\sigma_k^2)\big).
$$

*Siano $T_{tk} = p(k \mid z_t, \theta)$ le appartenenze a posteriori e $\widehat{C}_k = \sum_t T_{tk}$, $\widehat{U}_k = \sum_t T_{tk} z_t$, $\widehat{Q}_k = \sum_t T_{tk} z_t^2$ le statistiche sufficienti empiriche sul lotto. Allora:*

1. *I gradienti euclidei di $\mathcal{L}$ rispetto alle coordinate $(\log c_k,\, g_k,\, \log s_k)$ valgono:*

$$
\frac{\partial\mathcal{L}}{\partial\log c_k} = \frac{\widehat{C}_k}{n} - \pi_k, \qquad
\nabla_{g_k}\mathcal{L} = \frac{1}{n}\sum_{t=1}^{n} T_{tk}\,\frac{z_t - g_k}{\sigma_k^2}, \qquad
\nabla_{\log s_k}\mathcal{L} = \frac{1}{n}\sum_{t=1}^{n} T_{tk}\!\left(\frac{(z_t - g_k)^2}{\sigma_k^2} - 1\right).
$$

2. *La matrice di informazione di Fisher della log-verosimiglianza sui dati completi $F_c(\theta)$ è diagonale a blocchi sulle coordinate interne di ciascun concetto, con metriche $F_c(g_k) = \pi_k\,\operatorname{diag}(\sigma_k^{-2})$ e $F_c(\log s_k) = 2\pi_k I_d$. Moltiplicando il gradiente euclideo per $F_c(\theta)^{-1}$ si ottengono le direzioni naturali:*

$$
\tilde{\nabla}_{\pi_k}\mathcal{L} = \widehat{\pi}_k - \pi_k, \qquad
F_c(g_k)^{-1}\nabla_{g_k}\mathcal{L} = \frac{\widehat{U}_k}{n\pi_k} - \frac{\widehat{C}_k}{n\pi_k}g_k, \qquad
F_c(\log s_k)^{-1}\nabla_{\log s_k}\mathcal{L} = \frac{1}{2\pi_k}\!\left(\frac{\widehat{Q}_k - 2\widehat{U}_k g_k + \widehat{C}_k g_k^2}{n\sigma_k^2} - \frac{\widehat{C}_k}{n}\right).
$$

3. *Rappresentando i parametri correnti tramite le statistiche equivalenti di massa $n$, ossia $S_k = (n\pi_k,\, n\pi_k g_k,\, n\pi_k(g_k^2 + \sigma_k^2))$, l'aggiornamento convesso con passo $\rho \in (0, 1]$:*

$$
S_k' = (1 - \rho)\,S_k + \rho\,\widehat{S}_k
$$

   *seguito dalla proiezione chiusa $(\mu_k', \sigma_k'^2) = \mathrm{centre\_variance}(S_k')$, possiede derivata prima in $\rho = 0^+$ (per $\kappa = 0$) **identicamente uguale** al gradiente naturale $F_c(\theta)^{-1}\nabla_\theta\mathcal{L}$ su tutte le coordinate $(\pi_k, g_k, \log s_k)$, mentre per $\rho = 1$ riproduce in un solo passo il punto stazionario esatto del passo M dell'algoritmo EM.*

Questa unificazione risolve simultaneamente tre criticità fondamentali:
- **Equivalenza tabulare ($\rho = 1$):** quando le righe $z_t = x_r$ non dipendono da parametri a monte, $\rho = 1$ garantisce la monotonia della verosimiglianza marginale $\mathcal{L}(\theta') \ge \mathcal{L}(\theta)$ a ogni iterazione senza alcun tasso di apprendimento da tarare.
- **Rimozione della singolarità a $K = 1$ nella rete:** quando uno strato generativo nasce con un solo concetto ($K = 1$), $\operatorname{softmax}$ restituisce $T_{t1} \equiv 1$ costante e il gradiente retropropagato $\partial T_{t1}/\partial g_1 \equiv 0$ lascia $(g_1, \log s_1)$ immobili; l'operatore naturale con $0 < \rho_t < 1$ allinea invece immediatamente centro e dispersione alla distribuzione reale delle interrogazioni $q_t$.
- **Convergenza stocastica di Robbins-Monro e regolarizzazione selettiva:** nella coda generativa, dopo una rampa lineare iniziale di $t_w$ passi, il passo congiunto segue il decadimento

$$
\eta_t = \eta_0\,\min\!\left(1,\,\frac{t+1}{t_w}\right)\!\left(1 + \frac{\max(0, t - t_w)}{t_0}\right)^{-1/2}, \qquad \rho_t = \min(\rho_{\max},\, c_{\mathrm{em}}\eta_t),
$$

  che verifica $\sum_t \eta_t = \infty$ e $\sum_t \eta_t < \infty$ in media quadratica sugli incrementi filtrati. Il decadimento dei pesi ($L_2$) viene applicato esclusivamente agli operatori di proiezione lineare ($W_Q, W_V, W_O, W^{\mathrm{near}}, W_{\mathrm{final}}, W_{Q/K}^{\mathrm{mem}}$), lasciando intatti i log-priori del vocabolario $\log C_y$, le scale $\log s$, le normalizzazioni $\mathrm{LN}$ e le aperture residue $g_\ell$.

---

### 5 · Scelta strutturale dai dati: agglomerazione e morfogenesi conservativa

Nessuna dimensione strutturale di ACAN 10.0 richiede una ricerca manuale di iperparametri: ogni scelta tra un modello più parsimonioso $\mathcal{M}_0$ e un candidato più espressivo $\mathcal{M}_1$ viene decisa da un **test appaiato ad un errore standard** su $n$ osservazioni tenute fuori dall'addestramento. Posto $g_t = \log p_{\mathcal{M}_1}(y_t \mid x_t) - \log p_{\mathcal{M}_0}(y_t \mid x_t)$, con media $\bar{g}$ e deviazione campionaria $s_g$:

$$
\mathcal{M}_1 \succ \mathcal{M}_0 \iff \bar{g} > \frac{s_g}{\sqrt{n - 1}}.
$$

Tra più candidati ammissibili viene selezionato il modello con il minor numero di parametri liberi non battuto di oltre un errore standard dal migliore.

- **Sulle tabelle (dal particolare al generale):** una scansione lineare raggruppa le righe in micro-concetti iniziali; si costruisce poi una sequenza di fusioni agglomerative scegliendo la coppia che minimizza l'aumento di raggio ponderato $\Delta_{ij} = (C_i + C_j)\rho_{i\cup j} - C_i\rho_i - C_j\rho_j$, alternando iterazioni EM ($\rho = 1$). Grazie all'additività di $S_k$, ogni partizione lungo il cammino viene valutata in **leave-one-out esatto** sottraendo al volo il contributo della riga $r$ dalle statistiche del suo concetto senza rileggere la tabella. I livelli superiori del grafo dei concetti sono estratti massimizzando la coesione a coppie $\mathrm{CP}^\circ$, permettendo a concetti specifici di discendere da più genitori generali tramite matrici dense.
- **Nella rete generativa (morfogenesi conservativa dal semplice al complesso):** la rete nasce minimale ($D = 32$, $L = 1$, $H = 1$, $K_1 = 1$, $W = 3$) e, alle scadenze di decisione, confronta sulla finestra di prova del nuovo pezzo la rete corrente («com'è») con quattro morfismi che preservano la lettura iniziale trasferendo anche i momenti di AdamW:
  1. *Raddoppio della larghezza ($D \to 2D$):* duplicazione simmetrica delle coordinate con compensazione delle scale e passo dimezzato ($\eta/2$);
  2. *Aggiunta di uno strato ($L \to L + 1$):* apertura inizializzata chiusa ($g_{L+1} = 0$, sicché $h' = h + 0\cdot(\tilde{h} - h) = h$ preserva esattamente lo stato) ma proiezione d'uscita $W_O \sim \mathcal{N}(0, \sigma_O^2)$ non nulla, garantendo che $\partial\mathcal{L}/\partial g_{L+1} = (\nabla_{h'}\mathcal{L})^\top(\tilde{h} - h) \neq 0$ fin dal primo lotto ed evitando punti di sella a gradiente identicamente nullo;
  3. *Fissione concettuale ($K_\ell \to K_\ell + 1$):* il concetto con maggiore massa empirica viene diviso in due concetti figli di massa dimezzata $C_k / 2$, perturbati in direzioni antipodali sia nei centri $g_k \pm \delta$ sia nelle righe di indirizzamento $A_k \pm \delta_A$;
  4. *Raddoppio delle teste ($H \to 2H$):* aggiunta di teste indipendenti con pesi d'uscita inizialmente nulli verso lo stato residuo.

  Sul primo giro di un pezzo nuovo, la candidata a parità di architettura («com'è», zero parametri aggiunti) avanza non appena migliora la log-verosimiglianza media di validazione ($\bar{g} > 0$), mentre ogni espansione strutturale deve superare la soglia $\bar{g} > s_g/\sqrt{n-1}$.

---

### 6 · Predizione

#### 6.1 · Completamento universale sulle tabelle e pesatura del target

Data una nuova riga $x$ osservata sulle colonne $\Omega$, si calcolano le appartenenze $T_i(x_\Omega)$ usando unicamente gli attributi presenti ($\Omega$).
- **Attributo numerico mancante $j \notin \Omega$:** la predizione puntuale è l'attesa condizionata $\hat{x}_j = \sum_{i=1}^{K} T_i(x_\Omega)\,\mu_{ij}$, accompagnata dalla varianza predittiva della miscela.
- **Attributo categoriale mancante $j \notin \Omega$ (lettura gerarchica concatenata):** detta $D_{ij}$ la cardinalità dei valori distinti osservati nel concetto $i$ e $q^{\mathrm{sopra}}(v)$ la distribuzione predetta dal livello superiore, la stima a posteriori di Dirichlet (`categorical_reading`) vale:

$$
q_{ij}(v) = \frac{N_{ij}(v) + a_{ij}\,q^{\mathrm{sopra}}(v)}{M_{ij} + a_{ij}}, \qquad a_{ij} = \kappa\max(1, D_{ij}), \qquad p(x_j = v \mid x_\Omega) = \sum_{i=1}^{K} T_i(x_\Omega)\,q_{ij}(v),
$$

  con $\kappa$ scelto sulla griglia diadica per validazione leave-one-out. Durante l'apprendimento supervisionato (sia tabulare sia quando `NetText` viene applicato a righe tabulari tramite il vettore di pesi `weight`), la colonna target riceve peso pari alla somma di tutti gli attributi esplicativi ($w_{\mathrm{target}} = m$), impedendo che colonne rumorose o irrilevanti dominino la formazione dei concetti.

#### 6.2 · Generazione autoregressiva causale sul testo

Dato un prompt $(y_0, \dots, y_p)$, il cursore di inferenza costruisce lo stato incrementale composto, per ciascuno strato $\ell$, dalle sole statistiche causali $(M_{hk}, S_{hk})$ e dagli ultimi $W$ stati, oltre agli indirizzi chiave della memoria del documento. Ad ogni passo di generazione:
1. si valuta la distribuzione completa $p(y \mid y_{\le t})$ combinando la lettura a due livelli $p_{\mathrm{rete}}$ e la memoria episodica;
2. si seleziona $\hat{y}_{t+1} = \arg\max_y p(y \mid y_{\le t})$ per $\tau = 0$, oppure si campiona da $p(y \mid y_{\le t})^{1/\tau} / Z(\tau)$ per $\tau > 0$;
3. il token generato $\hat{y}_{t+1}$ aggiorna le statistiche causali $(M_{hk}, S_{hk})$ degli strati concettuali ma **non viene registrato nella memoria episodica del documento** (`remember=False`): la memoria conserva esclusivamente fatti attestati nel testo fornito dall'utente, evitando cicli degeneri di auto-copia e ripetizione di frasi proprie.

#### 6.3 · Metriche di qualità e limiti teorici strutturali

La qualità generativa è misurata dalla log-verosimiglianza negativa media per token su documenti disgiunti mai osservati né usati per decisioni strutturali:

$$
\mathrm{NLL} = -\frac{1}{N_{\mathrm{test}}}\sum_{t=1}^{N_{\mathrm{test}}} \log p(y_t \mid y_{<t}) \quad [\text{nat/token}], \qquad \mathrm{PPL} = e^{\mathrm{NLL}}.
$$

Tre proprietà delimitano strutturalmente il modello:
1. **Fattorizzazione intra-concetto:** dentro un singolo concetto gli attributi sono condizionatamente indipendenti; le interazioni non lineari (come lo XOR) vengono risolte esattamente allocando più concetti (e, nella rete sequenziale, almeno due strati affinché l'interrogazione del secondo strato osservi congiuntamente gli attributi combinati dal primo).
2. **Continuità metrica contro discretizzazione simbolica:** quando `NetText` viene applicato a tabelle serializzate in token, gli attributi numerici vengono quantizzati in fasce discrete prive di ordine metrico intrinseco; se il rapporto fra righe di addestramento e colonne continue è molto basso (es. 166 righe per 60 bande spettrali in `Sonar`), la densità gaussiana continua diretta di `ConceptModel` rimane strutturalmente superiore alla tokenizzazione discreta.
3. **Costo su CPU senza acceleratori:** benché la complessità per token sia lineare nel contesto e indipendente dalla lunghezza del documento, la retropropagazione attraverso più strati continui resta più onerosa di una singola scansione di conteggi su colonne tabulari già fissate.

## Informatica

### 1 · Architettura del sistema e codifica dei dati

ACAN 10.0 separa rigorosamente l'interfaccia analitica dall'esecuzione computazionale pesante mediante due servizi cooperanti:
- **Applicazione interattiva (`acan/vis/`):** espone le viste **Dati**, **Modello**, **Valutazione**, **Previsione**, **Generativo**, **Matematica** e **Informatica**, isolando le sessioni tabulari per scheda del browser e supportando tabelle intere fino a un milione di righe (`FullTables`).
- **Worker generativo persistente (`acan/lm/service.py`, `acan/lm/concept_trainer.py`):** processo dedicato che gestisce il ciclo di vita dei corpus, l'addestramento incrementale a pezzi, il salvataggio atomico dei checkpoint (`safetensors` + registro transazionale SQLite in modalità WAL `FULL`) e l'API HTTP compatibile con lo standard OpenAI (`/v1/models`, `/v1/chat/completions`, `/v1/completions`).

#### Codifica delle tabelle e del testo

1. **Pipeline tabulare (`acan/preprocessing.py`, `acan/vis/datasets.py`):** legge CSV, XLSX, JSON, Parquet e ARFF; calcola medie, scale e vocabolari categoriali **esclusivamente sullo split di addestramento**; codifica le celle mancanti tramite maschere binarie esplicite affinché non alterino né le medie né le varianze dei concetti.
2. **Pipeline testuale e fascicoli multi-dominio (`acan/lm/research_data.py`, `corpus_bundles`):**
   - I testi caricati (JSONL, TXT, Markdown, codice, PDF, DOCX, EPUB, archivi ZIP) vengono deduplicati e ripartiti **per documento intero** negli split `train`, `validation` e `test`, garantendo assenza totale di contaminazione fra prefissi.
   - Un tokenizzatore sub-word BPE (32.000 token di default) converte ogni documento in interi a 32/64 bit terminati da `eos`, mappati su disco tramite `numpy.memmap` per operare in memoria costante anche su corpus di molti gigabyte.
   - Poiché i dataset compositi concatenano in sequenza sorgenti eterogenee (conversazioni, voci enciclopediche italiane e inglesi, codice sorgente, sessioni shell), leggere fette contigue da $10^6$ token produrrebbe deriva di dominio e rimbalzi della NLL di validazione. La funzione `corpus_bundles` raggruppa i documenti di `train[:read]` in fascicoli allineati ai confini `eos` (~4.096 token) e ne applica una **permutazione pseudo-casuale deterministica** fissata dal seme del progetto. La funzione `piece_from_bundles` estrae poi per ogni passo l'intervallo esatto `[step * piece, (step + 1) * piece)`: ogni passo osserva una miscela stazionaria di tutti i domini, preservando al singolo token la biiezione sull'intero corpus e la ripresa deterministica da qualunque checkpoint.

---

### 2 · Algoritmo tabulare: scansione, EM esatto ed esportazione

Su una tabella di addestramento codificata $(X, \Omega)$ con pesi di colonna $w$ ($w_j = m$ sulle colonne target, $w_j = 1$ sulle feature):

1. **Inizializzazione bottom-up a blocchi (`acan/core/scan.py`):** una singola passata divide le righe in blocchi contigui e li riduce localmente tramite fusioni greedy fino a ottenere al più $K_{\max} = 1.024$ micro-concetti con circa quattro righe per concetto.
2. **Raffinamento EM ($\rho = 1$) e cammino agglomerativo (`ConceptModel._em`):** a ogni livello candidato si eseguono iterazioni di `natural_em_dict(..., step=1.0)` alternando il calcolo delle appartenenze $T_{rk}$ e l'aggiornamento delle statistiche $S_k$, per poi fondere le coppie a minimo costo $\Delta_{ij}$ fino alla radice ($K = 1$).
3. **Selezione leave-one-out e DAG multilivello:** per ciascuna partizione candidata si valuta in tempo lineare la log-probabilità leave-one-out sottraendo $T_{rk} x_r$ da $S_k$; si sceglie il livello di lavoro più compatto entro un errore standard dal massimo, si estraggono i livelli superiori tramite $\mathrm{CP}^\circ$ e si testano eventuali blocchi di covarianza piena sul solo livello di lavoro finale.
4. **Esportazione autonoma (`modello_concetti.zip`):** il modello serializza unicamente i dizionari di statistiche sufficienti e la codifica delle colonne (zero righe di addestramento memorizzate), producendo anche un pacchetto Python eseguibile senza dipendenze esterne oltre `numpy` e `pandas`.

---

### 3 · Addestramento su un testo

L'addestramento lungo sulla coda testuale (`run_training` in `acan/lm/concept_trainer.py`, `NetText` in `acan/lm/net_text.py` e `Learner` in `acan/lm/concept_net_training.py`) procede per pezzi di $10^6$ token attraverso cinque meccanismi chiave:

1. **Segmentazione causale e formazione dei lotti:** ogni pezzo estratto da `piece_from_bundles` viene tagliato lungo i confini `eos` in tratti di lunghezza massima $L_{\max} = 512$, ciascuno preceduto da `eos` così che anche il primo token del documento sia predetto. I tratti vengono ordinati per lunghezza simile e raccolti in lotti fino a `4.096` token su CPU (`max(2, min(cpu_count, 4))` thread), minimizzando il padding.
2. **Scansione causale a blocchi (`causal_read`, blocco $B_c = 64$):** invece di materializzare le statistiche cumulative $S_{hk}(i) \in \mathbb{R}^{B \times T \times H \times K \times d}$ per ogni posizione $i$, la sequenza è divisa in blocchi di 64 token: le interazioni intra-blocco sono calcolate con una moltiplicazione matriciale triangolare inferiore, mentre tra un blocco e il successivo viaggiano solo gli accumulatori $(M_{hk}, S_{hk})$.
3. **Lettura e retropropagazione analitica per gruppi (`_GroupedTokenLogProb`):** nella lettura a due livelli $p(G_y \mid h)\,p(y \mid h, G_y)$, un'indicizzazione ingenua `centre[members[g]]` allocherebbe per ogni lotto un tensore tridimensionale $(N, S, D)$ da centinaia di megabyte con accumulo sparso `index_add_` dei gradienti sul vocabolario (~90% del tempo CPU). L'operatore custom `torch.autograd.Function` ordina le $N$ posizioni valide del lotto per indice di gruppo $g = G_{y_t}$, individua i gruppi unici presenti tramite `unique_consecutive` e carica i centri $C_g \in \mathbb{R}^{S \times D}$ **una sola volta per gruppo**, calcolando in forma matriciale esatta sia `log_softmax` sia i gradienti chiusi rispetto a stati $h$, centri $e_y$, log-masse $\log C_y$ e scala $\log\sigma$.
4. **Passo congiunto Fisher-EM + AdamW:** su ciascun lotto:
   - durante il passaggio in avanti ogni strato `ConceptAttention` registra senza grafo autograd le statistiche empiriche $(\widehat{C}_{hk}, \widehat{U}_{hk}, \widehat{Q}_{hk})$ delle interrogazioni $q_t$ via `torch.bmm`;
   - si calcola il gradiente della perdita predittiva, si limita la norma globale a $1{,}0$ (`clip_grad_norm_` con `foreach=True`) e si esegue il passo `AdamW(..., foreach=True)` con tasso di Robbins-Monro $\eta_t$;
   - `Learner._concept_em_step` applica `natural_em_parameters` con passo $\rho_t = \min(0{,}05, 8\eta_t)$ ai concetti $(C_k, g_k, s_k)$ di tutti gli strati, seguito da `_pull_back` sulle sole matrici di proiezione lineare.
5. **Gare di morfogenesi su finestra di prova (`RACE_TRIAL = 65.536`) e riletture finali:**
   - quando il calendario di crescita indice una gara, i cinque candidati («com'è», `larghezza`, `strato`, `concetti`, `teste`) vengono addestrati sulla finestra iniziale del nuovo pezzo (fino a 65.536 token, per al più due tornate) e valutati su un campione stratificato fisso della validazione (`decision_text`, fino a 128 blocchi disgiunti dai blocchi che tracciano la curva della pagina);
   - il solo candidato vincitore legge i restanti ~935.000 token del pezzo; se un passo peggiora oltre un errore standard, viene ripristinato lo snapshot migliore dimezzando $\eta$;
   - sull'ultimo pezzo della coda, la rete stabilizzata esegue riletture complete del corpus finché la validazione migliora di oltre un errore standard, consolidando la memoria parametrica prima dell'esportazione in NumPy.

---

### 4 · Inferenza

L'inferenza e la generazione del testo (`acan/lm/concept_net.py`, `NetCursor`) sono interamente implementate in **NumPy puro**: il modello esportato non richiede PyTorch né GPU per leggere, rispondere o essere distribuito come archivio autonomo (`scrivi.py`).

1. **Prefill del prompt:** data una richiesta (testo libero o conversazione formattata con i delimitatori di turno), `NetText.prediction_cursor` individua l'ultimo confine `eos` e processa l'intero prefisso in un'unica passata vettoriale, inizializzando per ogni strato le somme causali $(M_{hk}, S_{hk})$, il buffer circolare degli ultimi $W$ stati e la matrice degli indirizzi della memoria episodica (`remember=True`).
2. **Equivalenza esatta fra lettura in blocco e streaming:** per costruzione algebrica delle somme causali, leggere $p$ token in parallelo o invocare $p$ volte `cursor.observe(token, remember=True)` produce stati e distribuzioni di probabilità identici a precisione di macchina.
3. **Ciclo di decodifica e isolamento della memoria:** ad ogni nuovo token generato si estrae $\hat{y}$ dalla distribuzione combinata rete + memoria episodica e si avanza lo stato con `cursor.observe(token, remember=False)`, aggiornando i concetti senza inquinare gli indirizzi del documento. L'API `/v1/chat/completions` espone questo stesso cursore sia in risposta singola sia in streaming incrementale Server-Sent Events (SSE), verificando i token e le stringhe di arresto a ogni passo.

---

### 5 · Complessità computazionale e memoria

Indichiamo con $T$ la lunghezza del documento, $D$ la dimensione dello stato, $L$ il numero di strati, $H$ le teste ($Hd = D$), $K$ i concetti per testa, $W$ la finestra locale, $V$ la taglia del vocabolario ($S \approx \sqrt{V}$) e $N_b$ il numero di token in un lotto di addestramento:

| Operazione | Tabulare (`ConceptModel`, $n$ righe, $m$ col.) | Rete a concetti ACAN 10.0 (`ConceptNet`) | Transformer denso standard |
|---|---|---|---|
| Apprendimento per passo / epoca | $\mathcal{O}(n K m)$ (somme esatte, $\rho = 1$) | $\mathcal{O}\!\big(N_b(L D^2 + L D K + S D) + G_{\mathrm{attivi}} S D\big)$ | $\mathcal{O}\!\big(N_b(L D^2 + L T D + V D)\big)$ |
| Selezione strutturale | Leave-one-out in $\mathcal{O}(n K m)$ dalle somme | 5 candidati su $\min(\text{pezzo},\,65.536)$ token | Ricerca esterna su training completi |
| Inferenza per nuovo token | $\mathcal{O}(K m)$ | $\mathcal{O}(L D^2 + L D K + 2\sqrt{V}\,D + T D_{\mathrm{mem}})$ | $\mathcal{O}(L D^2 + L T D + V D)$ |
| Stato degli strati in inferenza | $\mathcal{O}(K m)$ (indipendente da $n$) | $\mathcal{O}(L H K d + L W D)$ (**costante in $T$**) | $\mathcal{O}(L T D)$ (KV-cache lineare in $T$) |

---

### 6 · Validazione sperimentale: versione 10.0 contro baseline (`main`)

Tutte le misure seguenti sono riproducibili su CPU con gli stessi semi casuali e protocolli di validazione/test disgiunti.

#### 6.1 · Apprendimento lungo su coda multi-dominio e velocità su CPU (`V = 32.000`)

Su una coda multi-sorgente di 12 passi (4 domini concatenati in sequenza come nei corpus reali di dialogo, enciclopedia, codice e shell), `main` soffriva di deriva ciclica di dominio e blocco delle gare (`6/7` gare annullate, struttura ferma a `[[1, 1]]`); ACAN 10.0 apprende in modo monotono su tutti i passi:

| Versione | Curva NLL di validazione (passo 1 → 6 → 12) | NLL di test | Perplessità di test | Gare annullate | Concetti finali per testa |
|---|---|---:|---:|---:|---|
| `main` (v9) | `7,022 → 8,289 → 5,514` (oscillante) | 5,878 | 356,9 | 6 / 7 | `[[1, 1]]` (bloccata) |
| **ACAN 10.0** | **`5,800 → 4,707 → 4,532`** (monotona) | **4,451** (**−1,426 nat**) | **85,7** (**4,1× migliore**) | **1 / 8** | `[[8, 5]]` (3,0–4,7 attivi) |

Sul vocabolario completo di produzione (`V = 32.000`, 2 vCPU), `_GroupedTokenLogProb` accelera il singolo passo forward + backward di **3,93×** (`dim = 128, layers = 2`), e l'avanzamento effettivo della coda migliora fino a un ordine di grandezza:

| Fase (`V = 32.000`, pezzi da 100k tok) | Server live (`main`, 1 M tok) | `main` (stesse 2 vCPU) | **ACAN 10.0 (stesse 2 vCPU)** | Accelerazione |
|---|---:|---:|---:|---:|
| Passo con gara a 5 candidati | 58 tok/s | 565 tok/s (176,9 s) | **4.871 tok/s** (20,5 s) | **8,6×** a parità CPU (**84×** vs server su 1 M tok) |
| Passo ordinario | 528 tok/s | 961 tok/s (100,5 s) | **4.087 tok/s** (23,6 s) | **4,3×** a parità CPU (**7,7×** vs server) |

#### 6.2 · Confronto sui dataset tabulari: `main` contro `ACAN 10.0` (tabulare e generativo)

1. **Suite ufficiale di regressione tabulare (`tests/test_quality_regression.py`, `ConceptModel`):** poiché per $\rho = 1$ l'operatore unificato `natural_em_dict` coincide al bit con il passo M esatto, `ACAN 10.0` conserva **identica precisione al terzo decimale** rispetto a `main` su tutti i benchmark multi-split (`Iris` 10 split: **98,00% vs 98,00%**; `Wine` 10 split: **97,78% vs 97,78%**; `Vote` 5 split: **96,09% vs 96,09%**; `Breast Cancer` 2 split: **94,74% vs 94,74%**; `Sonar` 5 split: **90,48% vs 90,48%**; `Vehicle` 3 split: **79,61% vs 79,61%**).
2. **Confronto diretto fra i regimi $\rho = 1$, $\rho = 0{,}5$ e la rete generativa `NetText` (`main` contro `ACAN 10.0`, split 80/20, semi 1–3):** ogni riga tabulare è codificata come documento `[attributi, classe, eos]` con ottili calcolati sul solo training; in `ACAN 10.0` `NetText` applica l'aggiornamento congiunto Fisher-EM + gradiente e pesa il token target quanto tutti gli attributi insieme:

| Dataset | righe / col. | Tabulare `main` ($\rho=1$) | Tabulare `v10.0` ($\rho=1$) | Tabulare `v10.0` ($\rho=0{,}5$) | Generativo `NetText` `main` | **Generativo `NetText` `v10.0`** |
|---|---:|---:|---:|---:|---:|---:|
| **Gioco** (XOR + 3 rumori, avvio `GROWING`) | 300 / 5 | 100,0% | **100,0%** | **100,0%** | 65,6% (`NLL 0,548`) | **100,0%** (`NLL 0,000`) |
| **Gioco** (XOR + 3 rumori, 2 strati) | 300 / 5 | 100,0% | **100,0%** | **100,0%** | 100,0% (`NLL 0,026`, 91 s) | **100,0%** (`NLL 0,000`, 74 s) |
| **Iris** | 150 / 4 | 98,9% | **98,9%** | **98,9%** | 91,1% (52 s) | **94,4%** (35 s) |
| **Wine** | 178 / 13 | 96,3% | **96,3%** | **96,3%** | 94,4% (65 s) | **95,4%** (45 s) |
| **Vote** (categorie e mancanti) | 435 / 16 | 96,6% | 96,6% | **96,9%** | **97,3%** (190 s) | **96,9%** (148 s) |
| **Breast Cancer** | 569 / 30 | 95,6% | 95,6% | 95,9% | 95,6% (288 s) | **96,2%** (98 s) |
| **Sonar** (60 bande continue) | 208 / 60 | 90,5% | **90,5%** | **90,5%** | 52,4% (152 s) | **59,5%** (96 s) |
| **Vehicle** (18 rapporti continui) | 846 / 18 | **79,6%** | **79,6%** | 77,8% | 71,8% (739 s) | **67,1%** (194 s) |

L'esperimento incrociato conferma tre risultati teorici:
- su `ConceptModel` il passo graduale $\rho = 0{,}5$ converge allo stesso punto fisso di $\rho = 1$;
- sulla rete generativa `NetText`, l'introduzione dell'aggiornamento naturale sui concetti e della pesatura bilanciata del target elimina il collasso sulle colonne rumorose (`Gioco` da 65,6% a **100,0%** con `NLL = 0,000`), migliora l'accuratezza e riduce del 25–35% il tempo di addestramento anche a vocabolario piccolo;
- dove pochissime osservazioni devono coprire molte variabili continue (`Sonar`: 166 righe di training per 60 bande continue, ~480 token d'intervallo), la densità gaussiana continua diretta del modulo tabulare (**90,5%**) resta strutturalmente più efficiente della quantizzazione simbolica.

#### 6.3 · Ablazioni su dialoghi bilingui e memoria episodica (SYNTH, `V = 32.000`)

Sui dialoghi italiano/inglese con rete fissa ($D = 96, L = 3, H = 4, K = 64$, 3,4 M parametri):
- **Riletture successive (2.000 dialoghi):** la NLL di test scende da `5,352` (1 passaggio) a `4,966` (2 passaggi) fino a **`4,877`** (3 passaggi), mentre a parità di token letti dati nuovi battono la rilettura (`6.000` dialoghi letti una volta raggiungono test **`4,716`** contro `4,877`).
- **Memoria episodica con sentinella «nessuno somiglia al mio»:** riduce la NLL di validazione da `5,651` (senza memoria) a `4,858`, porta il test a convergenza da `4,832` (valvola semplice) a **`4,805`**, dimezza le risposte tronche sotto i 15 token (`10% → 3%`) e riduce dal `47%` al **`32%`** l'eco lessicale della domanda su 30 prompt mai visti.

[Protocolli sperimentali](../experiments/text_table/README.md).
