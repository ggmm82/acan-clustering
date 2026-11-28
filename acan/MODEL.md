# ACAN 10 · Modello Algebrico Unificato a Concetti per Dati Tabulari e Linguaggio

## Matematica

ACAN 10 formula l'apprendimento non supervisionato, predittivo e generativo — tanto su tabelle relazionali eterogenee quanto su sequenze simboliche di lunghezza arbitraria — all'interno di un'unica teoria probabilistica e geometrica fondata sulle **statistiche sufficienti additive di miscele esponenziali fattorizzate** e sul **gradiente naturale d'informazione di Fisher**.

---

### 1 · Formulazione probabilistica unificata: tabelle relazionali e sequenze simboliche

Sia $\mathcal{X} = \mathcal{X}_1 \times \dots \times \mathcal{X}_m$ uno spazio di osservazione composto da $m$ attributi eterogenei:
- per ogni coordinata continua $j \in \mathcal{J}_{\mathrm{num}}$, $\mathcal{X}_j = \mathbb{R}$;
- per ogni coordinata discreta o simbolica $j \in \mathcal{J}_{\mathrm{cat}}$, $\mathcal{X}_j = \{1, \dots, V_j\}$ è un alfabeto finito di cardinalità $V_j$.

Ogni osservazione $r \in \{1, \dots, n\}$ è una coppia $(x_r, \Omega_r)$, dove $\Omega_r \subseteq \{1, \dots, m\}$ indica il sottoinsieme delle coordinate effettivamente osservate e $x_{r,\Omega_r} = (x_{rj})_{j \in \Omega_r}$ i rispettivi valori misurati, assumendo che il meccanismo di omissione sia non informativo condizionatamente alle variabili osservate (ipotesi di mancanza a caso, MAR).

Questa singola struttura matematica comprende come casi particolari entrambi i domini operativi del modello:

1. **Dominio tabulare relazionale:** ogni riga costituisce un'osservazione $(x_r, \Omega_r)$. Durante l'apprendimento la distribuzione congiunta $p(x_r)$ modella simultaneamente attributi esplicativi e attributi bersaglio; in fase di interrogazione qualunque sottoinsieme $\Omega \subset \{1, \dots, m\}$ può essere fornito come evidenza nota per dedurre la distribuzione condizionata esatta $p(x_{\bar{\Omega}} \mid x_\Omega)$ sulle coordinate incognite $\bar{\Omega} = \{1, \dots, m\} \setminus \Omega$.
2. **Dominio sequenziale e linguistico:** un documento di simboli $(w_1, w_2, \dots, w_T) \in \{1, \dots, V\}^T$, preceduto e seguito dal delimitatore strutturale di fine documento $\langle\mathrm{eos}\rangle$, definisce per fattorizzazione causale esatta una successione di osservazioni condizionate:
   $$p(w_1, \dots, w_T \mid w_0 = \langle\mathrm{eos}\rangle) = \prod_{t=1}^{T} p(w_t \mid w_0, w_1, \dots, w_{t-1})$$
   Alla posizione $t$, il prefisso osservato $(w_0, \dots, w_{t-1})$ svolge il ruolo dell'evidenza nota $x_\Omega$, mentre il simbolo futuro $w_t \in \{1, \dots, V\}$ costituisce l'unica coordinata incognita da prevedere. Una tabella codificata in forma seriale $[a_1, \dots, a_{m-1}, y, \langle\mathrm{eos}\rangle]$ diventa così un documento finito su cui il predittore causale valuta direttamente $p(y \mid a_1, \dots, a_{m-1})$.

---

### 2 · Algebra delle statistiche sufficienti, geometria dei concetti ed embedding dei token

#### 2.1 · Famiglia esponenziale fattorizzata e statistiche sufficienti additive

Un livello concettuale è una miscela finita di $K$ componenti latenti (detti **concetti**) indicizzati da $k \in \{1, \dots, K\}$, con pesi a priori $\pi_k > 0$ tali che $\sum_{k=1}^K \pi_k = 1$. Condizionatamente al concetto $k$, le coordinate osservate sono indipendenti:
$$p(x_{r,\Omega_r} \mid z_r = k) = \prod_{j \in \Omega_r \cap \mathcal{J}_{\mathrm{num}}} \mathcal{N}\!\left(x_{rj} \;\middle|\; \mu_{kj},\, \sigma_{kj}^2\right) \prod_{j \in \Omega_r \cap \mathcal{J}_{\mathrm{cat}}} \mathrm{Cat}\!\left(x_{rj} \;\middle|\; \theta_{kj}\right)$$
dove $\mu_{kj} \in \mathbb{R}$ e $\sigma_{kj}^2 > 0$ sono il centro e la varianza della coordinata continua $j$ nel concetto $k$, mentre $\theta_{kj} = (\theta_{kjv})_{v=1}^{V_j}$ sul simplesso $\Delta^{V_j - 1}$ è la distribuzione categoriale della coordinata discreta $j$.

Assegnata a ogni osservazione $r$ con peso campionario $w_r > 0$ una distribuzione di responsabilità $T_{rk} \ge 0$ ($\sum_{k=1}^K T_{rk} = 1$), l'intera informazione necessaria a stimare il concetto $k$ su un insieme o sotto-insieme di dati $B$ è racchiusa nella quadrupla di **statistiche sufficienti estensive**:
$$C_k(B) = \sum_{r \in B} w_r\,T_{rk}, \qquad U_{kj}(B) = \sum_{\substack{r \in B \\ j \in \Omega_r}} w_r\,T_{rk}\,x_{rj}, \qquad Q_{kj}(B) = \sum_{\substack{r \in B \\ j \in \Omega_r}} w_r\,T_{rk}\,x_{rj}^2$$
$$N_{kjv}(B) = \sum_{\substack{r \in B \\ j \in \Omega_r}} w_r\,T_{rk}\,\mathbb{I}[x_{rj} = v]$$
accompagnate dalla massa effettivamente osservata per ciascuna colonna $C_{kj}^{\mathrm{obs}}(B) = \sum_{r \in B:\, j \in \Omega_r} w_r\,T_{rk}$.

Per qualunque coppia di blocchi disgiunti $B_1 \cap B_2 = \emptyset$, l'operatore di aggregazione è un omomorfismo esatto di monoidi commutativi:
$$\Sigma_k(B_1 \cup B_2) = \Sigma_k(B_1) + \Sigma_k(B_2), \qquad \Sigma_k = \big(C_k,\, C_k^{\mathrm{obs}},\, U_k,\, Q_k,\, N_k\big)$$
Questa proprietà algebrica rende la stima invariante rispetto alla decomposizione dei dati in blocchi di memoria, flussi incrementali o prefissi causali di un documento.

Per impedire degenerazioni singolari ($\sigma_{kj}^2 \to 0$) o probabilità nulle su simboli rari, i parametri canonici vengono ricavati dalle statistiche sufficienti mediante regolarizzazione coniugata empirica verso la dispersione globale $\sigma_{0,j}^2$ del livello radice (con intensità $\kappa > 0$) e lisciamento simmetrico di Dirichlet $\alpha_0 > 0$:
$$\mu_{kj} = \frac{U_{kj}}{C_{kj}^{\mathrm{obs}} + \varepsilon}, \qquad \sigma_{kj}^2 = \frac{\max\!\big(0,\; Q_{kj} - C_{kj}^{\mathrm{obs}}\,\mu_{kj}^2\big) + \kappa\,\sigma_{0,j}^2}{C_{kj}^{\mathrm{obs}} + \kappa}, \qquad \theta_{kjv} = \frac{N_{kjv} + \alpha_0}{C_{kj}^{\mathrm{obs}} + \alpha_0\,V_j}$$

#### 2.2 · Spazio di immersione dei simboli come concetti gaussiani e fattorizzazione gerarchica

Nel modulo linguistico e sequenziale, ogni simbolo del vocabolario $v \in \{1, \dots, V\}$ non è rappresentato da un vettore arbitrario separato dal classificatore finale, bensì è esso stesso un **concetto gaussiano isotropo** nello spazio degli stati $\mathbb{R}^D$, caratterizzato da:
- centro semantico-sintattico $e_v \in \mathbb{R}^D$;
- parametro logaritmico di massa a priori $\ell_v \in \mathbb{R}$, con $\pi_v = \frac{\exp(\ell_v)}{\sum_{u=1}^V \exp(\ell_u)}$ inizializzato dalla distribuzione empirica delle frequenze del corpus;
- scala metrica globale apprendibile $\tau = \exp(\alpha) > 0$.

Dato lo stato contestuale normalizzato $h_t \in \mathbb{R}^D$ prodotto dalla rete alla posizione $t$, la distribuzione predittiva sul simbolo successivo coincide esattamente con la probabilità a posteriori di appartenenza dello stato $h_t$ ai concetti del vocabolario:
$$\log p(w_{t+1} = v \mid h_t) = \ell_v - \frac{\tau^2}{2}\,\|h_t - e_v\|_2^2 - \log Z(h_t) = \underbrace{\left(\ell_v - \frac{\tau^2}{2}\,\|e_v\|_2^2\right)}_{b_v} + \tau^2\,\langle h_t,\, e_v\rangle - \frac{\tau^2}{2}\,\|h_t\|_2^2 - \log Z(h_t)$$
Poiché il termine $-\frac{\tau^2}{2}\|h_t\|_2^2$ è indipendente da $v$, esso si semplifica con la costante di normalizzazione $\log Z(h_t)$, unificando in un'unica identità geometrica l'immersione d'ingresso $e_v$ e la verosimiglianza generativa d'uscita.

Quando il vocabolario $V$ è ampio (ad esempio $V = 32.000$), la valutazione diretta dei $V$ concetti richiederebbe $\mathcal{O}(V D)$ operazioni per posizione. A partire dai profili di informazione mutua puntuale positiva sulle co-occorrenze contestuali destre e sinistre, ACAN 10 ricava le coordinate distributive iniziali dei simboli e li raggruppa mediante iterazioni chiuse di Fisher-EM in **concetti lessicali semantici** (eventualmente suddivisi per fascia di frequenza quando superano la capienza massima per blocco), formando $G$ gruppi $\mathcal{G}_1, \dots, \mathcal{G}_G$ semanticamente omogenei. Ogni gruppo possiede una massa aggregata $\bar{\ell}_g = \log \sum_{v \in \mathcal{G}_g} \exp(\ell_v)$ e un baricentro ponderato:
$$\bar{e}_g = \sum_{v \in \mathcal{G}_g} \frac{\exp(\ell_v)}{\exp(\bar{\ell}_g)}\,e_v \in \mathbb{R}^D$$
La probabilità del simbolo successivo si fattorizza in modo esatto in due scelte di complessità $\mathcal{O}(\sqrt{V}\,D)$ ciascuna — prima il concetto lessicale $g(v)$, poi il simbolo specifico all'interno del solo gruppo osservato:
$$\log p(v \mid h_t) = \log p\big(g(v) \mid h_t\big) + \log p\big(v \;\big|\; g(v),\, h_t\big)$$
riducendo il costo computazionale e di memoria da $\mathcal{O}(V D)$ a $\mathcal{O}(2\sqrt{V}\,D)$ senza alcuna approssimazione campionaria e conservando una distribuzione normalizzata esatta sull'intero vocabolario.

---

### 3 · Appartenenza a posteriori, gerarchia di astrazioni e sotto-concetti e memoria causale

#### 3.1 · Operatore di appartenenza bayesiana ed espansione quadratica

Dato un vettore continuo $q \in \mathbb{R}^d$ (una riga tabulare proiettata o un'interrogazione nello spazio di un gruppo concettuale) e una famiglia di $K$ concetti gaussiani diagonali $\{(\pi_k, g_k, s_k)\}_{k=1}^K$ con $s_k = \exp(\eta_k) \in \mathbb{R}_+^d$, il logaritmo non normalizzato della responsabilità a posteriori vale:
$$\Phi_k(q) = \log \pi_k - \sum_{m=1}^d \eta_{km} - \frac{1}{2}\sum_{m=1}^d \frac{(q_m - g_{km})^2}{s_{km}^2}, \qquad T_k(q) = \frac{\exp\!\big(\Phi_k(q) - \max_j \Phi_j(q)\big)}{\sum_{j=1}^K \exp\!\big(\Phi_j(q) - \max_{j'} \Phi_{j'}(q)\big)}$$
Sviluppando il quadrato con le precisioni $\lambda_{km} = s_{km}^{-2} = \exp(-2\eta_{km})$:
$$\sum_{m=1}^d \frac{(q_m - g_{km})^2}{s_{km}^2} = \langle q \odot q,\, \lambda_k\rangle - 2\,\langle q,\, g_k \odot \lambda_k\rangle + \langle g_k \odot g_k,\, \lambda_k\rangle$$
il calcolo simultaneo su tutte le posizioni di una sequenza si riduce a prodotti matriciali densi tra matrici $(T \times d)$ e $(d \times K)$, evitando la materializzazione di tensori tridimensionali intermedi $(T \times K \times d)$.

#### 3.2 · Grafo gerarchico senza matrici quadratiche, sotto-concetti multi-padre instradati e teste multi-vista

Sia $H^{(\ell-1)} \in \mathbb{R}^{T \times D}$ la sequenza degli stati in ingresso al livello concettuale $\ell \in \{1, \dots, L\}$ in uno spazio metrico compatto di dimensione $D$, con stato pre-normalizzato $\tilde{x}_t = \mathrm{LN}_1\!\big(h_t^{(\ell-1)}\big)$, ripartita su $H_{\mathrm{teste}}$ teste parallele di dimensione $d$ ($H_{\mathrm{teste}}\,d = D$) ciascuna associata a un distinto **punto di vista** interpretativo e dotata di $K_\ell$ concetti organizzati in piramide ($K_1 \le K_2 \le \dots \le K_L$, da centinaia fino a migliaia di astrazioni e sotto-concetti particolari, senza alcuna matrice di costo $K_\ell^2$). Per ogni posizione $t \in \{1, \dots, T\}$ e ogni testa $h \in \{1, \dots, H_{\mathrm{teste}}\}$:

1. **Teste di attenzione multi-vista sul contesto locale e coordinate asimmetriche di lettura e scrittura:** ciascuna testa proietta il proprio punto di vista interrogazione–chiave–valore $(q_{0,t,h}^{(\ell)},\, k_{0,t,h}^{(\ell)},\, v_{t,h}^{(\ell)}) = \big(W_{Q,h}\,\tilde{x}_t,\, W_{K,h}\,\tilde{x}_t,\, W_{V,h}\,\tilde{x}_t\big) \in (\mathbb{R}^d)^3$ e ispeziona selettivamente gli ultimi $W_{\mathrm{vista}}$ simboli (coprendo domande, argomenti e vincoli recenti del contesto in tempo lineare $\mathcal{O}(T W_{\mathrm{vista}} D)$ tramite blocchi contigui) con profilo di prossimità apprendibile $\beta_{h,\delta} \in \mathbb{R}$ ($\delta = t - \tau \in \{0, \dots, W_{\mathrm{vista}} - 1\}$), affiancato dalla traccia causale ravvicinata $r_t^{\mathrm{vicino}} = \sum_{w=1}^{W} \mathbb{I}[t > w]\,U_{\ell,w}\,h_{t-w}^{(\ell-1)}$:
   $$\omega_{t,h,\tau} = \mathrm{softmax}_{\tau \in [\max(1,\,t - W_{\mathrm{vista}} + 1),\, t]}\!\left(\frac{\langle q_{0,t,h}^{(\ell)},\, k_{0,\tau,h}^{(\ell)}\rangle}{\sqrt{d}} + \beta_{h, t-\tau}\right), \qquad r_{t,h}^{\mathrm{vista}} = \sum_{\tau=\max(1,\,t - W_{\mathrm{vista}} + 1)}^{t} \omega_{t,h,\tau}\,v_{\tau,h}^{(\ell)}$$
2. **Residualizzazione gerarchica tabulare senza matrici $K^2$:** adottando la medesima decomposizione dei livelli concettuali del dominio tabulare senza matrici quadratiche $K_\ell \times K_\ell$ o $K_{\ell-1} \times K_\ell$, note al livello superiore $\ell - 1$ le statistiche spiegate dai concetti padre $(\bar{g}_{t,h}^{(\ell-1)}, \bar{s}_{t,h}^{(\ell-1)})$, il livello $\ell \ge 2$ interroga i propri concetti direttamente sullo **scarto standardizzato non ancora spiegato dal livello padre**:
   $$q_{t,h}^{(\ell)} = q_{0,t,h}^{(\ell)} + \mathbb{I}[\ell \ge 2]\,\frac{q_{t,h}^{(\ell-1)} - \bar{g}_{t,h}^{(\ell-1)}}{\bar{s}_{t,h}^{(\ell-1)}}$$
   Poiché la geometria dei residui accoppia in $\mathbb{R}^d$ tutte le teste attraverso lo stato condiviso pre-normalizzato, ogni sotto-concetto particolare $k$ del livello $\ell$ può discendere simultaneamente da più concetti padre $p$ del livello $\ell - 1$ (formando un grafo aciclico diretto multi-ereditario con quota a posteriori misurata $P(p \mid k) = \frac{\sum_t T_{t,h,p}^{(\ell-1)} T_{t,h,k}^{(\ell)}}{\sum_t T_{t,h,k}^{(\ell)}}$), mentre le viste prospettiche $(q_{0,t,h}^{(\ell)}, k_{0,t,h}^{(\ell)}, v_{t,h}^{(\ell)})$ e l'intensità selettiva di scrittura forniscono piena asimmetria direzionale in complessità $\mathcal{O}(d)$ anziché $\mathcal{O}(K_\ell^2)$.
3. **Instradamento per macro-concetti a momenti esatti, attivazione sparsa Top-$m$ e passo residuo:**
   - Quando $K_\ell$ è contenuto ($K_\ell \le 32$), scrittura e lettura valutano direttamente i $K_\ell$ concetti con $T_{t,h,k}^{(\ell)} = \mathrm{softmax}_k \Phi_k(q_{t,h}^{(\ell)})$.
   - Quando un livello si articola in centinaia o migliaia di sotto-concetti ($K_\ell = G_c\,S_\ell > 32$, con $G_c = 16$ famiglie concettuali di $S_\ell$ sotto-concetti fratelli ciascuna), ciascuna famiglia $\mathcal{F}_c = \{(c-1)S_\ell + 1, \dots, c S_\ell\}$ viene sintetizzata analiticamente nel **macro-concetto a momenti esatti** $(\bar{\pi}_{h,c}, \bar{g}_{h,c}, \bar{s}_{h,c})$ con $\bar{\pi}_{h,c} = \sum_{j \in \mathcal{F}_c} \pi_{h,j}$, $\bar{g}_{h,c} = \sum_{j \in \mathcal{F}_c} \frac{\pi_{h,j}}{\bar{\pi}_{h,c}}\,g_{h,j}$ e $\bar{s}_{h,c}^2 = \sum_{j \in \mathcal{F}_c} \frac{\pi_{h,j}}{\bar{\pi}_{h,c}}\big(s_{h,j}^2 + (g_{h,j} - \bar{g}_{h,c})^2\big)$. La memoria causale del prefisso accumula le statistiche sufficienti $(M_{t,h,c}, S_{t,h,c})$ sui $G_c$ macro-concetti con responsabilità $T_{t,h,c}^{(\ell)} = \mathrm{softmax}_c \Phi_c(q_{t,h}^{(\ell)})$ e intensità selettiva $s_{t,h} = \frac{1}{2} + \sigma\!\big(\sqrt{D}\,\langle w_h, h_t^{(\ell-1)}\rangle / (\|h_t^{(\ell-1)}\|_2 + \varepsilon)\big)$:
     $$M_{t,h,c} = \varepsilon + \sum_{\tau=1}^t s_{\tau,h}\,T_{\tau,h,c}^{(\ell)}, \qquad S_{t,h,c} = \sum_{\tau=1}^t s_{\tau,h}\,T_{\tau,h,c}^{(\ell)}\,v_{\tau,h}^{(\ell)}, \qquad r_{t,h}^{\mathrm{conc}} = \sum_{c=1}^{G_c} T_{t,h,c}^{(\ell)}\,\frac{S_{t,h,c}}{M_{t,h,c}}$$
     mentre l'interrogazione $q_{t,h}^{(\ell)}$ seleziona il proprio macro-concetto $c^* = \arg\max_c \Phi_c(q_{t,h}^{(\ell)})$ e valuta **esclusivamente gli $S_\ell = K_\ell / G_c$ sotto-concetti fini** di $\mathcal{F}_{c^*}$ attivando i soli $m_{\mathrm{top}}$ sotto-concetti a maggior densità a posteriori ($\mathcal{M}_{t,h} = \mathrm{Top}\text{-}m_{\mathrm{top}}\{\Phi_j(q_{t,h}^{(\ell)}) : j \in \mathcal{F}_{c^*}\}$), da cui estrae il **prototipo geometrico fine attivo** $\bar{g}_{t,h}^{(\ell)} = \sum_{j \in \mathcal{M}_{t,h}} T_{t,h,j}^{(\ell)}\,g_{h,j}^{(\ell)}$ e la dispersione residua $\bar{s}_{t,h}^{(\ell)}$. Il costo per simbolo scende così da $\mathcal{O}(K_\ell^2 + K_\ell d)$ a **$\mathcal{O}((G_c + S_\ell)\,d) = \mathcal{O}(\sqrt{K_\ell}\,d)$**, consentendo migliaia di concetti attivi con costo sub-lineare.
   - Concatenati i vettori di tutte le teste in $u_t^{(\ell)} = \big[r_t^{\mathrm{conc}};\, r_t^{\mathrm{vista}};\, \bar{g}_{t}^{(\ell)};\, r_t^{\mathrm{vicino}}\big] \in \mathbb{R}^{4D}$, l'aggiornamento a pre-normalizzazione somma un contributo additivo modulato dalla valvola $\gamma_\ell \in \mathbb{R}$:
     $$h_t^{(\ell)} = h_t^{(\ell-1)} + \gamma_\ell\,\mathrm{LN}_2\!\big(W_O\,u_t^{(\ell)} + b_O\big)$$

#### 3.3 · Memoria episodica del documento con indirizzo sentinella

Accanto alla memoria astratta compressa nelle statistiche sufficienti $(M_{t,h,c}, S_{t,h,c})$ e ai concetti gaussiani per ciascun simbolo da predire, una memoria episodica leggera ($D_{\mathrm{mem}} \le D$) preserva l'ancoraggio esatto alle entità, ai numeri e ai termini specifici comparsi nella domanda o nel documento corrente. Ad ogni posizione $\tau \ge 1$ lo stato finale $h_\tau \in \mathbb{R}^D$ deposita un indirizzo chiave $k_\tau = \frac{1}{\sqrt{D_{\mathrm{mem}}}}\,W_{\mathrm{key}}\,h_\tau \in \mathbb{R}^{D_{\mathrm{mem}}}$ associato al simbolo effettivamente seguito $y_\tau = w_{\tau+1}$. Alla posizione $t > 1$, l'interrogazione $q_t^{\mathrm{mem}} = W_{\mathrm{query}}\,h_t$ confronta le affinità $a_{t,\tau} = \langle q_t^{\mathrm{mem}},\, k_\tau\rangle$ ($\tau < t$) con un **indirizzo sentinella** $a_{t,0} = \langle w_{\mathrm{sent}},\, h_t\rangle + b_{\mathrm{sent}}$ (*«predici tramite i concetti generali e lessicali senza copia puntuale»*):
$$\alpha_{t,0} = \frac{\exp(a_{t,0})}{\exp(a_{t,0}) + \sum_{\tau'=1}^{t-1} \exp(a_{t,\tau'})}, \qquad \alpha_{t,\tau} = \frac{\exp(a_{t,\tau})}{\exp(a_{t,0}) + \sum_{\tau'=1}^{t-1} \exp(a_{t,\tau'})} \quad (\tau = 1, \dots, t-1)$$
producendo la mistura convessa $p(w_{t+1} = v \mid h_{1:t}) = \alpha_{t,0}\,p_{\mathrm{concetti}}(v \mid h_t) + \sum_{\tau=1}^{t-1} \alpha_{t,\tau}\,\mathbb{I}[y_\tau = v]$.

---

### 4 · Teorema di unificazione dell'apprendimento: gradiente naturale di Fisher ed EM

Sia $\eta = (\eta_1, \dots, \eta_K)$ la parametrizzazione canonica di una famiglia esponenziale regolare di miscela $p(x, z \mid \eta) = h(x)\exp\{\langle \eta_z, t(x)\rangle - A(\eta_z) + \log \pi_z\}$, e sia $S_k = \nabla A(\eta_k)$ la corrispondente parametrizzazione nei **momenti attesi** (statistiche sufficienti normalizzate per unità di osservazione: massa relativa $c_k = C_k / n$, primo momento $u_k = U_k / n$ e secondo momento $q_k = Q_k / n$).

Fissate al passo $t$ le responsabilità a posteriori $T_{rk}^{(t)} = p(z_r = k \mid x_r, \theta^{(t)})$ su un insieme (o mini-lotto) $B_t$ di massa totale $W(B_t) = \sum_{r \in B_t} w_r$, il funzionale inferiore variazionale di evidenza per osservazione è:
$$\mathcal{F}(S;\, \theta^{(t)}) = \frac{1}{W(B_t)} \sum_{r \in B_t} w_r \sum_{k=1}^K T_{rk}^{(t)} \log p(x_r, z_r = k \mid S) + \mathcal{H}(T^{(t)})$$

**Teorema (Equivalenza tra gradiente naturale sui dati completi ed EM online/esatto).** *Equipaggiando la varietà statistica con la metrica riemanniana d'informazione di Fisher sui dati completi $\mathcal{I}_c(S)$, il gradiente naturale $\widetilde{\nabla}_S \mathcal{F} = \mathcal{I}_c(S)^{-1}\,\nabla_S \mathcal{F}$ valutato in $S^{(t)}$ coincide identicamente con lo scarto lineare tra le statistiche sufficienti empiriche del lotto $\widehat{S}_k(B_t)$ e le statistiche correnti $S_k^{(t)}$:*
$$\widetilde{\nabla}_{S_k} \mathcal{F}\big(S^{(t)};\, \theta^{(t)}\big) = \widehat{S}_k(B_t) - S_k^{(t)}$$
*Pertanto il passo di ascesa lungo il gradiente naturale con passo $\rho_t \in (0, 1]$ assume la forma universale convessa:*
$$S_k^{(t+1)} = (1 - \rho_t)\,S_k^{(t)} + \rho_t\,\widehat{S}_k(B_t)$$

*Dimostrazione.* Nella parametrizzazione duale $(\eta_k, S_k)$ legata dalla trasformata di Legendre $S_k = \nabla A(\eta_k)$, lo jacobiano del cambio di coordinate è l'hessiano della funzione di partizione $\frac{\partial S_k}{\partial \eta_k} = \nabla^2 A(\eta_k)$, che per una famiglia esponenziale coincide con l'informazione di Fisher per singola osservazione completa. Pesando per la frazione a priori $c_k$, il gradiente euclideo rispetto a $\eta_k$ vale $\nabla_{\eta_k}\mathcal{F} = \widehat{S}_k(B_t) - S_k^{(t)}$; applicando l'inversa della metrica di Fisher e trasformando nella coordinata duale $S_k$ mediante la regola della catena $\delta S_k = \frac{\partial S_k}{\partial \eta_k}\,\delta\eta_k$, le matrici di curvatura $\nabla^2 A(\eta_k)$ si cancellano esattamente, restituendo $\widetilde{\nabla}_{S_k}\mathcal{F} = \widehat{S}_k(B_t) - S_k^{(t)}$. Per convessità del dominio dei momenti validi ($q_{kj} \ge u_{kj}^2 / c_k > 0$, $c_k > 0$, $\sum_k c_k = 1$), per ogni $\rho_t \in (0, 1]$ l'interpolazione convessa $S_k^{(t+1)}$ appartiene strettamente all'interno del dominio ammissibile senza bisogno di proiezioni vincolate. $\square$

Questa unica legge governa l'intero sistema:
- **Regime batch esatto ($\rho_t = 1$, apprendimento tabulare e lessicale):** sostituendo $\rho_t = 1$ si ottiene $S_k^{(t+1)} = \widehat{S}_k(X)$, ossia il passo di massimizzazione esatta di Expectation-Maximization su tabelle intere o suddivise in blocchi additivi e sulla formazione dei concetti lessicali, con aumento monotono garantito della log-verosimiglianza marginale $\mathcal{L}(\theta^{(t+1)}) \ge \mathcal{L}(\theta^{(t)})$.
- **Regime stocastico accoppiato ($0 < \rho_t < 1$, apprendimento generativo profondo):** sulle sequenze lunghe, dove le coordinate residualizzate $(q_{t,h}^{(\ell)}, k_{t,h}^{(\ell)})$ dipendono dalle rappresentazioni sottostanti, ad ogni mini-lotto $B_t$ ACAN 10 esegue congiuntamente:
  1. un passo del primo ordine precondizionato (con tasso di apprendimento $\eta_t$ governato da riscaldamento lineare iniziale e decadimento di Robbins-Monro $\eta_t = \eta_0 \min(1, t/T_w)\,\max(\eta_{\min}, (1 + \max(0, t-T_w)/T_0)^{-1/2})$) sulle viste $(W_Q, W_K, W_V, \beta_h)$, sui centri e sulle dispersioni dei concetti e sui concetti del vocabolario, guidato dal gradiente analitico esatto della log-verosimiglianza causale $-\sum_{t} \log p(w_{t+1} \mid h_{1:t})$;
  2. un passo chiuso di gradiente naturale di Fisher $S_k^{(t+1)} = (1 - \rho_t)S_k^{(t)} + \rho_t \widehat{S}_k(B_t)$ accumulato direttamente sulle sotto-popolazioni instradate di ciascun livello con passo accoppiato alla medesima scala del gradiente ($\rho_t = \min(\rho_{\max},\, \beta\,\eta_t)$, $\beta = \frac{1}{2}$, per $K_\ell \ge 2$), in modo che l'adattamento metrico delle masse, dei centri e delle dispersioni mantenga centinaia o migliaia di concetti e sotto-concetti ben distribuiti sulla varietà dei residui senza collassi degeneri;
  3. un decadimento regolarizzativo selettivo applicato esclusivamente alle mappe lineari proiettive attive (escludendo masse a priori, dispersioni, profili posizionali $\beta_h$, normalizzazioni, valvole residue e proiezioni d'interrogazione/chiave finché $K_\ell = 1$).

---

### 5 · Morfogenesi strutturale guidata dai dati: agglomerazione e raffinamento concettuale

ACAN 10 non richiede di fissare arbitrariamente il numero di concetti: in entrambi i domini la gerarchia concettuale emerge dai dati tramite operatori che rispettano la geometria della verosimiglianza.

1. **Riduzione gerarchica e convalida analitica leave-one-out (dominio tabulare):**
   - A partire da una sovra-partizione fine, ad ogni scala vengono fusi i concetti affini sommando direttamente le loro statistiche sufficienti ($\Sigma_{a \cup b} = \Sigma_a + \Sigma_b$) e rifinendo la partizione con $\rho = 1$.
   - Per scegliere la scala ottima senza addestramenti ripetuti, la verosimiglianza fuori campione della riga $r$ si valuta in tempo costante $\mathcal{O}(1)$ **sottraendo algebricamente** il contributo della riga stessa dalle statistiche sufficienti accumulate:
     $$\Sigma_k^{(-r)} = \Sigma_k - w_r\,T_{rk}\,\sigma(x_r)$$
     selezionando il livello che massimizza la densità predittiva incrociata.

2. **Morfogenesi costruttiva piramidale di centinaia e migliaia di concetti senza costo $K^2$ (dominio generativo):**
   - Lo spazio metrico continuo resta ultra-compatto ($D = 32$, con estensione automatica limitata per impostazione predefinita a $D \le 64$ affinché l'intero modello occupi solo pochi megabyte e la ricchezza rappresentativa risieda nella piramide dei concetti, dei sotto-concetti e dei punti di vista anziché in grandi matrici dense). Sui vocabolari linguistici reali ($V \ge 2.048$), la rete parte già con una **piramide multi-vista di centinaia di concetti contestuali** ($L = 2$ livelli, $H_{\mathrm{teste}} = 4$ punti di vista, $K = [16, 64]$, ossia $320$ concetti contestuali affiancati dai $V$ concetti dei singoli token), pronta ad articolarsi in migliaia di sotto-concetti, entro i soli limiti configurati. Ad intervalli controllati del volume di testo appreso, il grafo corrente compete sulla medesima finestra di prova fresca contro **morfismi quasi-isometrici** che preservano la predizione all'istante iniziale ($t = 0$) rompendo al contempo la simmetria dei gradienti:
     - **Raddoppio intercalato dei concetti e sotto-concetti a macro-momenti invariati ($K_\ell \to 2K_\ell$):** privilegiando il livello con il minor rapporto $K_\ell / 2^{\ell-1}$ (così da sviluppare una piramide $K_1 < K_2 < \dots < K_L$ in cui le astrazioni generali si ramificano in sotto-concetti particolari sempre più numerosi), ogni concetto $k$ genera due figli intercalati $(2k-1, 2k)$ con masse dimezzate ($\log \pi' = \log \pi - \log 2$), spostamento antipodale $g_\pm = g \pm u \odot \sigma$ e contrazione analitica della dispersione $\eta' = \eta + \frac{1}{2}\log(1 - u^2)$. Poiché i due figli rimangono nella medesima famiglia $\mathcal{F}_c$ e soddisfano $\frac{1}{2}g_+ + \frac{1}{2}g_- = g$ e $\frac{1}{2}(\sigma'^2 + u^2\sigma^2) + \frac{1}{2}(\sigma'^2 + u^2\sigma^2) = \sigma^2$, **tutti i macro-concetti $(\bar{\pi}_{h,c}, \bar{g}_{h,c}, \bar{s}_{h,c})$ restano matematicamente invariati all'istante zero**.
     - **Raddoppio antipodale delle teste di attenzione per nuovi punti di vista ($H_{\mathrm{teste}} \to 2H_{\mathrm{teste}}$):** ciascuna testa concettuale e prospettica viene duplicata dimezzando le rispettive colonne nei tre blocchi concettuale, prospettico e prototipico della proiezione d'uscita ($\frac{1}{2}W_O \pm \varepsilon_O$), copiando il profilo posizionale $\beta_h$ con perturbazioni opposte infinitesimali ($\pm \varepsilon$) su interrogazioni, chiavi, valori e centri concettuali e scalatura coerente dei momenti dell'ottimizzatore.
     - **Aggiunta di un livello residualizzato di sotto-concetti ($L \to L + 1$):** un nuovo livello di sotto-concetti alimentato dagli scarti standardizzati $(q^{(L)} - \bar{g}_q^{(L)}) / \bar{s}_q^{(L)}$ e $(k^{(L)} - \bar{g}_k^{(L)}) / \bar{s}_k^{(L)}$ parte con **valvola residua a pre-normalizzazione inizialmente chiusa** ($\gamma_{L+1} = 0$), sicché $h_t^{(L+1)} = h_t^{(L)}$ esattamente a $t = 0$ mentre il gradiente apre subito la valvola senza alterare la norma dei livelli precedenti.
     - **Estensione controllata della larghezza ($D \to 2D$, $d \to 2d$):** entro il massimale compatto ($D \le 64$ per impostazione predefinita), le coordinate vengono duplicate con compensazione analitica esatta delle scale prospettiche, delle dispersioni residualizzate e dei macro-concetti.
   - **Plasticità locale transitoria delle componenti nascenti:** quando il grafo ha assimilato milioni di simboli e il passo globale si è raffreddato secondo la legge di Robbins-Monro, i parametri preesistenti conservano il passo raffreddato mentre i soli parametri dei concetti, sotto-concetti e livelli appena scissi o generati ricevono un regime transitorio di plasticità locale calda con attenuazione iperbolica graduale verso il passo globale.
   - **Principio di parsimonia a varianza ridotta:** tutti i candidati osservano l'identica sequenza pseudocasuale di mini-lotti del grafo invariato, cancellando il rumore campionario comune nel confronto appaiato:
     $$\bar{\Delta} > \frac{s_\Delta}{\sqrt{M - 1}}, \qquad \Delta_i = \ell_{\mathrm{cand}}(x_i) - \ell_{\mathrm{base}}(x_i)$$
     Un candidato cresciuto è ammesso solo se migliora la verosimiglianza pre-gara e supera per più di un errore standard il riferimento pre-gara oppure il grafo invariato a fine finestra; fra tutti i candidati entro un errore standard dal migliore, prevale sempre quello con il **minor numero di parametri**.

---

### 6 · Predizione

#### 6.1 · Completamento universale su dati tabulari e bilanciamento dell'attributo bersaglio

Dato un vettore parzialmente osservato $x_\Omega$, la probabilità a posteriori di ciascun concetto $k$ si ottiene per **marginalizzazione analitica immediata** delle coordinate non osservate $\bar{\Omega}$ (il cui integrale su ciascun fattore indipendente vale esattamente $1$):
$$p(z = k \mid x_\Omega) = \frac{\pi_k \prod_{j \in \Omega} p_j(x_j \mid z = k)}{\sum_{k'=1}^K \pi_{k'} \prod_{j \in \Omega} p_j(x_j \mid z = k')}$$
Da questa responsabilità condizionata discende la distribuzione predittiva completa di qualsiasi attributo incognito $u \notin \Omega$:
- **per un attributo continuo $u \in \mathcal{J}_{\mathrm{num}}$**, la densità predittiva è la mistura gaussiana $\sum_k p(z=k \mid x_\Omega)\,\mathcal{N}(x_u \mid \mu_{ku}, \sigma_{ku}^2)$, con valore atteso ottimo in norma quadratica $\hat{x}_u = \sum_{k=1}^K p(z=k \mid x_\Omega)\,\mu_{ku}$ e varianza predittiva totale (aleatoria più epistemica) $\mathrm{Var}(x_u \mid x_\Omega) = \sum_k p(z=k \mid x_\Omega)\big(\sigma_{ku}^2 + (\mu_{ku} - \hat{x}_u)^2\big)$;
- **per un attributo categoriale $u \in \mathcal{J}_{\mathrm{cat}}$**, la distribuzione di classe è $p(x_u = v \mid x_\Omega) = \sum_{k=1}^K p(z=k \mid x_\Omega)\,\theta_{kuv}$ con decisione di Bayes $\hat{x}_u = \arg\max_v p(x_u = v \mid x_\Omega)$.

Quando uno specifico attributo $y$ svolge il ruolo di bersaglio supervisionato accanto a $m - 1$ attributi esplicativi, nella log-verosimiglianza congiunta ad esso viene attribuito peso $w_y = m - 1$ (pari alla somma di tutte le variabili esplicative), impedendo che molte colonne rumorose o ridondanti mettano in ombra la struttura discriminativa del bersaglio — sia nella formulazione tabulare diretta sia nella codifica sequenziale generativa.

#### 6.2 · Decodifica autoregressiva causale sul testo

In generazione testuale, letto il prefisso $(w_0 = \langle\mathrm{eos}\rangle, w_1, \dots, w_t)$, per ciascuno strato $\ell$ lo stato interno dei livelli concettuali dipende dal passato tramite lo stato ricorrente compatto ($M_{\mathrm{mem},\ell} = G_c$ per i livelli instradati a grande $K_\ell$, altrimenti $K_\ell$):
$$\mathcal{C}_t^{(\ell)} = \Big(\big\{M_{t,h,c}^{(\ell)},\, S_{t,h,c}^{(\ell)}\big\}_{h,\,c=1..M_{\mathrm{mem},\ell}},\; \big\{\big(k_{0,\tau,h}^{(\ell)},\, v_{\tau,h}^{(\ell)}\big)\big\}_{\tau=\max(1,\,t-W_{\mathrm{vista}}+1)}^{t},\; \big(h_{t-W+1}^{(\ell-1)}, \dots, h_t^{(\ell-1)}\big)\Big)$$
Ad ogni nuovo simbolo osservato $w_{t+1}$, l'aggiornamento delle somme concettuali, la selezione dei sotto-concetti fini nel macro-concetto attivo e la finestra prospettica limitata a $W_{\mathrm{vista}}$ posizioni costano $\mathcal{O}(L D^2 + L H \sqrt{K}\,d + L W_{\mathrm{vista}} D)$ operazioni — **prive di termini $K^2$ e costanti rispetto alla lunghezza $t$ nei livelli concettuali**. Dalla distribuzione combinata concetti + memoria sentinella $p(v \mid h_{1:t})$, il simbolo successivo viene estratto deterministicamente ($\arg\max_v$) oppure mediante campionamento con temperatura $T_{\mathrm{camp}} > 0$ sulla distribuzione ripesata $p_v^{1/T_{\mathrm{camp}}} / \sum_u p_u^{1/T_{\mathrm{camp}}}$. I simboli autogenerati aggiornano le statistiche concettuali e la finestra prospettica ma non vengono inseriti come nuovi indirizzi nella memoria episodica, evitando cicli auto-rinforzanti di ripetizione lessicale.

#### 6.3 · Misure d'informazione, leggi di scala e limiti strutturali

La qualità predittiva su un insieme di verifica disgiunto di $N_{\mathrm{test}}$ posizioni è misurata dalla **log-verosimiglianza negativa media (NLL)** in nat per simbolo e dalla corrispondente **perplessità**:
$$\mathrm{NLL} = -\frac{1}{N_{\mathrm{test}}} \sum_{t=1}^{N_{\mathrm{test}}} \log p(w_t \mid w_{<t}), \qquad \mathrm{PPL} = \exp(\mathrm{NLL})$$
dove $\mathrm{PPL}$ indica il fattore di ramificazione effettivo medio fra cui il modello rimane incerto ad ogni passo (con valore iniziale $\mathrm{NLL}_0 \approx \log V$, pari a $10{,}37$ nat per $V = 32.000$).

Lungo l'addestramento su grandi corpus testuali, la discesa dell'errore segue due regimi distinti:
1. **Fase iniziale rapida:** nei primi milioni di token la rete assorbe la distribuzione marginale delle frequenze e le regolarità sintattiche corte, abbattendo la NLL di oltre $4{,}5$ nat (la perplessità scende di due ordini di grandezza, da $32.000$ a circa $250$).
2. **Regime asintotico a legge di potenza:** una volta apprese le statistiche locali frequenti, l'affinamento delle dipendenze semantiche rare e a lungo raggio obbedisce alla legge di scala universale dei modelli di linguaggio:
   $$\mathrm{NLL}(N) \approx \mathrm{NLL}_\infty + A\,N^{-\alpha} \qquad (\alpha > 0)$$
   per cui riduzioni successive uguali in nat corrispondono a riduzioni moltiplicative costanti della perplessità ma richiedono volumi crescenti di testo; su un asse verticale lineare che include il punto iniziale $\log V \approx 10{,}37$, tale curva convessa appare visivamente compressa pur continuando a ridurre la perplessità passo dopo passo.

Infine, quando il numero di esempi disponibili $n$ è piccolo rispetto a un elevato numero di coordinate continue $m$ ($n \ll 8\,m$), la stima gaussiana continua diretta conserva l'ordinamento metrico esatto sull'asse reale e presenta efficienza campionaria superiore rispetto alla discretizzazione simbolica in intervalli quantilici, la quale trasforma ogni coordinata continua in simboli categoriali privi di continuità metrica tra intervalli adiacenti.

---

## Informatica

ACAN 10 realizza la teoria matematica precedente in un'architettura software modulare progettata per operare in modo affidabile sia su normali processori multicore (CPU) sia su acceleratori vettoriali (GPU), garantendo riprendibilità transazionale, memoria controllata e inferenza autonoma leggera.

---

### 1 · Architettura distribuita del sistema e rappresentazione dei dati

Il sistema è articolato in due processi disaccoppiati che comunicano tramite protocollo HTTP autenticato:
- **Ambiente analitico interattivo:** gestisce le sessioni utente indipendenti per ciascuna scheda del browser nelle viste dedicate a caricamento e pulizia dei dati, addestramento del grafo concettuale tabulare, analisi diagnostica, previsione interattiva, gestione delle code linguistiche e consultazione dei riferimenti matematici e architetturali. Supporta tabelle fino a un milione di righe mantenendo lo stato isolato per sessione.
- **Servizio persistente di calcolo generativo:** processo autonomo in background che governa l'ingestione a blocchi dei corpus, la preparazione deterministica dei vocabolari, l'esecuzione sequenziale delle code di addestramento, la rotazione transazionale dei salvataggi su disco (registro relazionale con journal write-ahead sincronizzato e archivi binari immutabili dei tensori) e l'esposizione di interfacce standardizzate di completamento e conversazione in streaming (Server-Sent Events), protette da chiavi di spazio di lavoro a 256 bit.

#### Codifica dei dati tabulari e dei flussi testuali

1. **Rappresentazione tabulare:** i formati strutturati (CSV, fogli elettronici, JSON, Parquet, ARFF) vengono tipizzati distinguendo colonne numeriche continue, colonne categoriali e identificatori ad alta cardinalità esclusi automaticamente come chiavi non predittive. Tutte le trasformazioni statistiche (centratura, normalizzazione di scala, dizionari categoriali, quantili) vengono calcolate **esclusivamente sulla partizione di addestramento** e applicate invariate a validazione e test. I valori assenti non vengono sostituiti con medie fittizie, ma rappresentati tramite maschere booleane esplicite che azzerano il loro peso nei prodotti matriciali delle statistiche sufficienti.
2. **Rappresentazione testuale e multimodale:** documenti di testo libero, archivi di dialogo strutturato, codice sorgente, libri elettronici e contenuti multimediali (immagini quantizzate su tavolozza percettiva ed estratti acustici spettrali racchiusi da marcatori dedicati) vengono convertiti in una sequenza di interi tramite un vocabolario sub-lessicale Byte-Pair Encoding (tipicamente $V = 32.000$ simboli) addestrato sul solo testo di training, con inserimento sistematico del delimitatore $\langle\mathrm{eos}\rangle$ ai confini di documento.
3. **Mescolamento deterministico per fasci documentali a taglia limitata:** nei corpus composti dalla concatenazione sequenziale di fonti eterogenee (ad esempio dialoghi, voci enciclopediche, codice e comandi di sistema), la lettura lineare causerebbe forti derive cicliche di dominio tra un passo e l'altro. Il flusso di addestramento viene perciò suddiviso ai confini di $\langle\mathrm{eos}\rangle$ in fasci documentali compresi fra $4.096$ e $16.384$ simboli (imponendo il taglio massimo di $16.384$ simboli anche all'interno di lunghe sequenze prive di delimitatori), permutati una sola volta tramite seme pseudocasuale fisso. In tal modo ogni singolo passo e ogni finestra di gara osservano la medesima miscela bilanciata di domini della validazione, pur conservando conteggi esatti e ripresa deterministica da qualunque passo salvato.

---

### 2 · Pipeline algoritmica per dati tabulari

L'addestramento di un modello concettuale tabulare procede attraverso quattro fasi interamente vettorizzate:

1. **Inizializzazione e scansione a blocchi:** i dati numerici e le codifiche binarie delle categorie vengono letti a blocchi di righe di dimensione prefissata. Le somme di massa, primo momento, secondo momento e frequenze categoriali vengono accumulate con complessità lineare nel numero di righe $n$ e di attributi $m$.
2. **Iterazione a punto fisso ($\rho = 1$):** ad ogni passo le responsabilità a posteriori $T \in \mathbb{R}^{n \times K}$ vengono valutate tramite l'espansione quadratica matriciale stabilizzata in log-spazio; la moltiplicazione matriciale $T^\top X$ e $T^\top (X \odot X)$ produce direttamente le nuove statistiche sufficienti $\widehat{S}_k$, da cui si aggiornano in forma chiusa masse, centri, varianze regolarizzate e probabilità categoriali fino a convergenza della log-verosimiglianza.
3. **Selezione gerarchica multi-scala:** partendo da una griglia fine di concetti, coppie di concetti affini vengono progressivamente unite sommando i rispettivi accumulatori sufficienti; a ciascuna scala la verosimiglianza incrociata leave-one-out viene calcolata algebricamente per sottrazione diretta dalle somme, individuando la risoluzione ottimale senza ricalcolare l'addestramento per singola riga.
4. **Calibrazione ed esportazione autonoma:** sulle attività di classificazione le soglie di decisione vengono calibrate sulla distribuzione predittiva fuori campione; l'intero grafo multi-scala di concetti (centri, varianze, tabelle categoriali e relazioni padre-figlio) viene serializzato in un documento autosufficiente pronto per l'inferenza istantanea.

---

### 3 · Addestramento su un testo

L'addestramento di una coda linguistica è strutturato per operare in modo continuo anche per giorni su server CPU o GPU, con tolleranza completa alle interruzioni:

1. **Partizione tripartita rigorosa del corpus:** prima dell'avvio il corpus viene separato in tre insiemi disgiunti:
   - *flusso di addestramento*, da cui vengono estratti in ordine i passi da un milione di simboli ciascuno;
   - *insieme di decisione strutturale*, costituito da blocchi distribuiti uniformemente lungo la validazione (escludendo i blocchi di misura) e separati da delimitatori $\langle\mathrm{eos}\rangle$ per evitare giunzioni spurie a metà frase fra domini differenti;
   - *insieme di misura indipendente*, mai osservato né dai gradienti né dalle decisioni di crescita, utilizzato ad ogni passo intero per certificare NLL e perplessità fuori campione, oltre alla NLL predittiva sul prossimo segmento di testo non ancora letto.
2. **Raggruppamento per lunghezza e passo d'ottimizzazione accoppiato senza costo $K^2$:** all'interno di ciascun passo i documenti vengono tagliati alla lunghezza massima di contesto (ad esempio $512$ simboli), preceduti da $\langle\mathrm{eos}\rangle$, ordinati per lunghezza affine nei calcoli di validazione (così da evitare lo spreco quadratico di riempimento causato da un singolo documento lungo accanto a molti brevi) e riuniti in mini-lotti di circa $4.096$ simboli. Su ciascun mini-lotto vengono eseguiti in sequenza:
   - la valutazione parallela dei livelli causali di concetti e sotto-concetti con **teste di attenzione multi-vista** sugli ultimi $W_{\mathrm{vista}}$ simboli tramite prodotti matriciali a blocchi contigui (senza matrici quadratiche $T \times T$), **residualizzazione gerarchica tabulare** su lettura e scrittura, **instradamento per macro-concetti a momenti esatti** e **attivazione sparsa Top-$m$ dei sotto-concetti fini** mediante un nucleo differenziabile ibrido che separa le famiglie dense dalle famiglie rare (senza alcuna matrice $K^2$ né sprechi di riempimento), passo residuo a pre-normalizzazione e memoria episodica con indirizzo sentinella;
   - il calcolo fattorizzato a due stadi della log-verosimiglianza sui concetti gaussiani di ciascun token da predire (riuniti in concetti lessicali semantici) mediante un nucleo differenziabile esplicito che accumula analiticamente i gradienti rispetto agli stati $h_t$, ai centri dei simboli $e_v$, alle masse $\ell_v$ e alla scala $\alpha$ senza allocare tensori tridimensionali $(N_{\mathrm{lotto}} \times S \times D)$;
   - il taglio della norma globale del gradiente a $1$, il passo dell'ottimizzatore adattivo (con regimi di plasticità locale dedicati per le valvole residue e per i concetti, sotto-concetti e livelli appena generati o scissi), l'aggiornamento chiuso di Fisher-EM sulle statistiche sufficienti delle sotto-popolazioni instradate e la regolarizzazione selettiva delle sole proiezioni lineari.
3. **Morfogenesi automatica di centinaia e migliaia di concetti e tetti hardware facoltativi:** come nel dominio tabulare, l'utente non deve scegliere il numero di concetti, di teste o di livelli. Sui corpus linguistici ($V \ge 2.048$) la rete si avvia direttamente con una piramide multi-vista di $320$ concetti contestuali ($D = 32, L = 2, H = 4, K = [16, 64]$; larghezza, strati, teste e concetti del primo strato di partenza si possono scegliere (i concetti raddoppiano a ogni strato, fino a 1.024), e la crescita può poi ingrandirli fino ai massimali) oltre ai $V$ concetti dei token, e lasciando a zero i massimali facoltativi espande autonomamente concetti, sotto-concetti, livelli e punti di vista senza tetti (migliaia di concetti attivi senza matrici $K^2$); la pagina propone per impostazione predefinita $D \le 64$ affinché l'intero modello resti compreso fra $4{,}4$ e $8{,}6\text{ MB}$. Gli unici limiti sono quelli configurati: un massimale fissato su una dimensione non ne impone altri alle restanti. L'utente può anche fissare una piramide statica se disattiva la crescita. Quando scatta un controllo di crescita, il grafo corrente e i candidati imparano a parità di mini-lotto e di sequenza casuale l'**intero pezzo nuovo**; finché una crescita vince, la gara si ripete sullo stesso pezzo. Il calendario segue la rete: gara sul primo pezzo; dopo una crescita, gara di nuovo sul pezzo successivo; quando vince la rete com'è, la gara successiva si allontana di 1, 2, 4… pezzi senza un tetto; se la validazione dopo un pezzo non scende, la rete ha smesso di migliorare e la gara si anticipa al pezzo successivo; gara sull'ultimo pezzo. Una finestra breve non basta: un candidato appena cresciuto paga all'inizio la separazione dei concetti scissi e la recupera solo dopo abbastanza testo (sezione 6.2).
4. **Salvataggio atomico, grafo multi-padre per punto di vista in tempo reale e continuità tra code:** ogni $4\text{–}5$ secondi il processo lavoratore trasmette l'avanzamento interno del passo, i punteggi in diretta delle gare e il grafo gerarchico orientato dall'alto verso il basso di astrazioni, concetti intermedi e sotto-concetti con legami multi-padre filtrabile per singolo punto di vista (testa di attenzione), valutato in pochi millisecondi tramite prodotti matriciali sui soli concetti attivi. Al termine dei passi previsti vengono scritti su disco i salvataggi completi, mantenendo gli ultimi salvataggi recenti, il miglior modello assoluto certificato dalla validazione e le pietre miliari permanenti.

---

### 4 · Inferenza

In fase di generazione e di interrogazione tramite API, il modello opera attraverso un motore algebrico puro in virgola mobile a 32 bit su CPU, completamente svincolato dal motore di differenziazione automatica:

1. **Struttura dati a cursore incrementale:** l'oggetto di lettura mantiene per ogni livello $\ell \in \{1, \dots, L\}$ due piccoli array densi per le masse accumulate $M^{(\ell)} \in \mathbb{R}^{H_{\mathrm{teste}} \times M_{\mathrm{mem},\ell}}$ e i momenti $S^{(\ell)} \in \mathbb{R}^{H_{\mathrm{teste}} \times M_{\mathrm{mem},\ell} \times d}$ (con $M_{\mathrm{mem},\ell} = 16$ sui livelli instradati anche quando $K_\ell = 1.024$), un buffer circolare limitato alle ultime $W_{\mathrm{vista}}$ coppie chiave–valore $(k_{0,\tau,h}^{(\ell)}, v_{\tau,h}^{(\ell)})$ per le teste di attenzione multi-vista e agli ultimi $W$ stati $h_{t-W+1:t}^{(\ell-1)}$, oltre agli indirizzi chiave–simbolo $(k_\tau, y_\tau)$ della memoria sentinella del documento corrente.
2. **Pre-lettura parallela del prompt e decodifica ricorrente:** all'arrivo di una richiesta, il cursore osserva anzitutto il simbolo $\langle\mathrm{eos}\rangle$ e processa l'intero prefisso o la domanda fornita dall'utente in forma matriciale vettorizzata, popolando in un'unica passata le somme dei macro-concetti e valutando per ciascuna posizione solo i sotto-concetti fini della famiglia attiva.
3. **Generazione simbolo per simbolo in streaming:** per ogni nuovo token richiesto viene valutata la distribuzione predittiva sui concetti lessicali, sui concetti dei singoli token e sulla memoria sentinella del documento, viene campionato il simbolo successivo e lo stato avanza aggiornando le somme ricorrenti concettuali e il buffer prospettico limitato, emettendo immediatamente i frammenti decodificati in flusso continuo fino al raggiungimento della lunghezza richiesta o di una sequenza di arresto.

---

### 5 · Complessità computazionale e di memoria

Indicando con $n$ il numero di righe e $m$ il numero di colonne di una tabella, e per il modello sequenziale con $T$ la lunghezza del documento, $D$ la dimensione compatta delle coordinate ($32 \le D \le 64$), $L$ il numero di livelli concettuali, $H$ il numero di teste di attenzione ($d$ coordinate per testa), $K$ i concetti per testa ($G_c = 16$, $S_K = K / G_c \approx \sqrt{K}$), $W_{\mathrm{vista}}$ la finestra prospettica locale, $V$ la cardinalità del vocabolario ($S \approx \lceil\sqrt{V}\rceil$) e $N_b$ il numero di simboli in un mini-lotto:

| Operazione | Dominio tabulare ($n$ righe, $m$ attributi) | Dominio generativo a concetti ACAN 10 (zero $K^2$) | Attenzione densa quadratica standard |
|---|---|---|---|
| Apprendimento per epoca / lotto | $\mathcal{O}(n K m)$ (somme sufficienti esatte, $\rho = 1$) | $\mathcal{O}\!\big(N_b(L D^2 + L W_{\mathrm{vista}} D + L H \sqrt{K}\,d + S D) + G_{\mathrm{attivi}} S D\big)$ | $\mathcal{O}\!\big(N_b(L D_{\mathrm{llm}}^2 + L T D_{\mathrm{llm}} + V D_{\mathrm{llm}})\big)$ |
| Selezione della struttura | Convalida leave-one-out analitica in $\mathcal{O}(n K m)$ | Candidati quasi-isometrici sull'intero pezzo nuovo (regola 1-SE) | Ricerca esterna con riaddestramenti completi |
| Inferenza per nuovo simbolo / riga | $\mathcal{O}(K m)$ | $\mathcal{O}(L D^2 + L W_{\mathrm{vista}} D + L H \sqrt{K}\,d + 2\sqrt{V}\,D + T D_{\mathrm{mem}})$ | $\mathcal{O}(L D_{\mathrm{llm}}^2 + L T D_{\mathrm{llm}} + V D_{\mathrm{llm}})$ |
| Stato dei livelli concettuali in inferenza | $\mathcal{O}(K m)$ (indipendente da $n$) | $\mathcal{O}(L H G_c d + L W_{\mathrm{vista}} D)$ (**costante in $T$ e in $K$**) | $\mathcal{O}(L T D_{\mathrm{llm}})$ (crescente linearmente con $T$ su tutti gli $L$ livelli) |

---

### 6 · Caratterizzazione sperimentale della versione 10

#### 6.1 · Apprendimento tabulare continuo, graduale e generativo sequenziale

La tabella seguente riassume l'accuratezza predittiva fuori campione (media su partizioni indipendenti 80/20) ottenuta da ACAN 10 sui benchmark relazionali classici in tre regimi della medesima algebra: aggiornamento esatto $\rho = 1$, aggiornamento graduale $\rho = 0{,}5$ e codifica sequenziale simbolica letta dalla rete generativa a concetti con pesatura bilanciata del bersaglio ($w_y = m - 1$):

| Dataset | Osservazioni / attributi | Valutazione multi-split standard ($\rho = 1$) | Regime continuo esatto ($\rho = 1$) | Regime continuo graduale ($\rho = 0{,}5$) | Regime generativo sequenziale |
|---|---:|---:|---:|---:|---:|
| **Gioco sintetico** (interazione XOR + rumore) | 300 / 5 | — | **100,0%** | **100,0%** | **100,0%** (`NLL 0,000`) |
| **Iris** | 150 / 4 | **98,0%** (10 split) | **98,9%** | **98,9%** | **94,4%** |
| **Wine** | 178 / 13 | **97,8%** (10 split) | **96,3%** | **96,3%** | **95,4%** |
| **Vote** (attributi categoriali con omissioni) | 435 / 16 | **96,1%** (5 split) | **96,6%** | **96,9%** | **96,9%** |
| **Breast Cancer** | 569 / 30 | **94,7%** (2 split) | **95,6%** | **95,9%** | **96,2%** |
| **Sonar** (60 bande spettrali continue) | 208 / 60 | **90,5%** (5 split) | **90,5%** | **90,5%** | **59,5%** |
| **Vehicle** (18 descrittori geometrici continui) | 846 / 18 | **79,6%** (3 split) | **79,6%** | **77,8%** | **67,1%** |

I risultati confermano empiricamente tutte le proprietà della teoria unificata:
- nel dominio tabulare i regimi $\rho = 1$ e $\rho = 0{,}5$ convergono al medesimo equilibrio statistico;
- su tabelle categoriali, miste o con numero adeguato di esempi per attributo (*Gioco*, *Iris*, *Wine*, *Vote*, *Breast Cancer*), la rete generativa sequenziale eguaglia o supera il modello tabulare risolvendo interazioni non lineari pure (XOR al **100,0%** con incertezza residua nulla) e scartando le variabili di disturbo;
- in presenza di decine di variabili continue con poche centinaia di righe (*Sonar*: 60 coordinate continue per 166 esempi di training), la densità gaussiana continua diretta (**90,5%**) sfrutta appieno la metrica euclidea reale rispetto alla quantizzazione simbolica per ottili.

#### 6.2 · Velocità oltre `8.000–13.000` token/s, piramidi di migliaia di concetti senza $K^2$ e dialogo domanda–risposta (`V = 32.000`)

Su code multi-dominio reali da decine di milioni di simboli ($V = 32.000$, dialogo multilingue domanda–risposta, testi enciclopedici, codice sorgente e script di sistema) eseguite su CPU nello spazio metrico compatto ($32 \le D \le 64$):
- **Eliminazione di $K^2$, alta velocità ed estrema leggerezza:** sostituendo le matrici quadratiche $K^2$ con coordinate asimmetriche di lettura/scrittura, macro-concetti a momenti esatti e attivazione sparsa Top-$m$ dei sotto-concetti fini, su 4 core CPU il modello raggiunge **`13.400` simboli/s** con **$320$ concetti contestuali** ($4{,}35\text{ MB}$) e mantiene **`8.300–8.600` simboli/s** anche scalando a **$1.344\text{–}2.624$ concetti contestuali** più $32.000$ concetti token in soli **$4{,}45\text{–}4{,}55\text{ MB}$** (circa $40$ volte più leggero di un piccolo modello linguistico denso).
- **Apertura immediata dei livelli di sotto-concetti e coerenza domanda–risposta:** la residualizzazione gerarchica tabulare su lettura e scrittura unita all'aggiornamento a pre-normalizzazione apre immediatamente le valvole dei livelli di sotto-concetti ($\gamma_2, \gamma_3 > 0{,}5$ già dai primi cicli, portando la NLL di validazione intorno a $5{,}0\text{–}5{,}4$ nat già ai primi due passi), mentre teste di attenzione multi-vista sulla domanda, concetti per ciascun token e memoria sentinella si combinano nella previsione.
- **Grafo multi-padre orientato per punto di vista:** partendo già da centinaia di concetti articolati su $4$ punti di vista complementari e ramificandosi in migliaia di sotto-concetti condivisi fra più concetti padre, il passo accoppiato di Fisher-EM preserva l'attività dell'intera piramide rendendo ogni astrazione, concetto intermedio e sotto-concetto direttamente ispezionabile per singola testa di attenzione.

#### 6.3 · Una coda completa sul server

Coda di dialoghi, testi e comandi (45 passi da un milione di simboli, $V = 32.000$, crescita attiva con larghezza massima $D = 64$), al termine:

| Quantità | Valore |
|---|---|
| Errore su testo mai visto | da $10{,}37$ a $4{,}825$ nat, ancora in discesa di circa $0{,}015$ nat per passo |
| Velocità media | circa $4.400$ simboli/s su CPU, gare comprese |
| Rete | $D = 64$, $L = 2$, $H = 4$, $K = [32, 128]$ |
| Parametri | $2{,}2$ milioni, di cui $2{,}05$ milioni nei concetti dei token e circa $88.000$ nei livelli concettuali |
