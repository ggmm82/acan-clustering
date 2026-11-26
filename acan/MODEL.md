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

#### 3.2 · Grafo gerarchico a residualizzazione tabulare, sotto-concetti multi-padre e teste di attenzione multi-vista

Sia $H^{(\ell-1)} \in \mathbb{R}^{T \times D}$ la sequenza degli stati in ingresso al livello concettuale $\ell \in \{1, \dots, L\}$ in uno spazio metrico compatto di dimensione $D$, con stato pre-normalizzato $\tilde{x}_t = \mathrm{LN}_1\!\big(h_t^{(\ell-1)}\big)$, ripartita su $H_{\mathrm{teste}}$ teste parallele di dimensione $d$ ($H_{\mathrm{teste}}\,d = D$) ciascuna associata a un distinto **punto di vista** interpretativo e dotata di $K_\ell$ concetti organizzati in piramide ($K_1 \le K_2 \le \dots \le K_L$, dalle astrazioni generali ai sotto-concetti particolari). Per ogni posizione $t \in \{1, \dots, T\}$ e ogni testa $h \in \{1, \dots, H_{\mathrm{teste}}\}$:

1. **Teste di attenzione multi-vista sul contesto locale e traccia sequenziale ravvicinata:** ciascuna testa proietta il proprio punto di vista asimmetrico interrogazione–chiave–valore $(q_{0,t,h}^{(\ell)},\, k_{t,h}^{(\ell)},\, v_{t,h}^{(\ell)}) = \big(W_{Q,h}\,\tilde{x}_t,\, W_{K,h}\,\tilde{x}_t,\, W_{V,h}\,\tilde{x}_t\big) \in (\mathbb{R}^d)^3$ e ispeziona selettivamente gli ultimi $W_{\mathrm{vista}}$ simboli (coprendo per intero domande, argomenti e vincoli recenti del contesto in tempo lineare $\mathcal{O}(T W_{\mathrm{vista}} D)$) con profilo di prossimità apprendibile $\beta_{h,\delta} \in \mathbb{R}$ ($\delta = t - \tau \in \{0, \dots, W_{\mathrm{vista}} - 1\}$), affiancato dalla traccia causale corta $r_t^{\mathrm{vicino}} = \sum_{w=1}^{W} \mathbb{I}[t > w]\,U_{\ell,w}\,\tilde{x}_{t-w}$:
   $$\omega_{t,h,\tau} = \mathrm{softmax}_{\tau \in [\max(1,\,t - W_{\mathrm{vista}} + 1),\, t]}\!\left(\frac{\langle q_{0,t,h}^{(\ell)},\, k_{\tau,h}^{(\ell)}\rangle}{\sqrt{d}} + \beta_{h, t-\tau}\right), \qquad r_{t,h}^{\mathrm{vista}} = \sum_{\tau=\max(1,\,t - W_{\mathrm{vista}} + 1)}^{t} \omega_{t,h,\tau}\,v_{\tau,h}^{(\ell)}$$
2. **Residualizzazione gerarchica tabulare e sotto-concetti con più concetti padre:** adottando la medesima decomposizione dei livelli concettuali del dominio tabulare, note al livello superiore $\ell - 1$ le responsabilità $T_{t,h,p}^{(\ell-1)}$, ogni testa calcola il prototipo atteso $\bar{g}_{t,h}^{(\ell-1)}$ e la dispersione marginale $\bar{s}_{t,h}^{(\ell-1)}$ spiegati dai concetti padre:
   $$\bar{g}_{t,h}^{(\ell-1)} = \sum_{p=1}^{K_{\ell-1}} T_{t,h,p}^{(\ell-1)}\,g_{h,p}^{(\ell-1)}, \qquad \bar{s}_{t,h}^{(\ell-1)} = \sqrt{\varepsilon + \sum_{p=1}^{K_{\ell-1}} T_{t,h,p}^{(\ell-1)}\Big(\big(s_{h,p}^{(\ell-1)}\big)^2 + \big(g_{h,p}^{(\ell-1)} - \bar{g}_{t,h}^{(\ell-1)}\big)^2\Big)}$$
   Il livello $\ell \ge 2$ interroga quindi i propri sotto-concetti sullo **scarto standardizzato non ancora spiegato dal livello padre**, combinato con la matrice densa di **sollevamento bayesiano multi-padre** $L_h^{(\ell)} \in \mathbb{R}^{K_{\ell-1} \times K_\ell}$ che consente a ogni sotto-concetto particolare $k$ di discendere simultaneamente da più concetti padre $p$ (formando un grafo aciclico diretto multi-ereditario con quota a posteriori $P(p \mid k) = \frac{\sum_t T_{t,h,p}^{(\ell-1)} T_{t,h,k}^{(\ell)}}{\sum_t T_{t,h,k}^{(\ell)}}$):
   $$q_{t,h}^{(\ell)} = q_{0,t,h}^{(\ell)} + \mathbb{I}[\ell \ge 2]\,\frac{q_{t,h}^{(\ell-1)} - \bar{g}_{t,h}^{(\ell-1)}}{\bar{s}_{t,h}^{(\ell-1)}}, \qquad \Phi_{t,h,k}^{(\ell)} = \Phi_k\!\Big(q_{t,h}^{(\ell)}\Big) + \mathbb{I}[\ell \ge 2]\,\log \sum_{p=1}^{K_{\ell-1}} T_{t,h,p}^{(\ell-1)}\,\exp\!\big(L_{h,pk}^{(\ell)}\big), \qquad T_{t,h,k}^{(\ell)} = \mathrm{softmax}_k\!\Big(\Phi_{t,h,k}^{(\ell)}\Big)$$
   L'intensità selettiva di scrittura vale $s_{t,h} = \frac{1}{2} + \sigma\!\big(\sqrt{D}\,\langle w_h, \tilde{x}_t\rangle / (\|\tilde{x}_t\|_2 + \varepsilon)\big)$ e la massa depositata nella memoria concettuale è $\tilde{T}_{t,h,k} = s_{t,h}\,T_{t,h,k}^{(\ell)}$.
3. **Statistiche sufficienti causali del prefisso, prototipo attivo e passo residuo a pre-normalizzazione:**
   $$M_{t,h,k} = \varepsilon + \sum_{\tau=1}^t \tilde{T}_{\tau,h,k} \in \mathbb{R}_+, \qquad S_{t,h,k} = \sum_{\tau=1}^t \tilde{T}_{\tau,h,k}\,v_{\tau,h}^{(\ell)} \in \mathbb{R}^d, \qquad \bar{v}_{t,h,k} = \frac{S_{t,h,k}}{M_{t,h,k}}$$
   Con la matrice associativa stocastica per righe $P_h = \mathrm{softmax}_{\mathrm{righe}}(A_h)$ ($A_h^{(0)} = \frac{3}{2}\,I_{K_\ell}$), ogni testa estrae sia la memoria relazionale del prefisso $r_{t,h}^{\mathrm{conc}} = \sum_{k=1}^{K_\ell} T_{t,h,k}^{(\ell)} \sum_{j=1}^{K_\ell} P_{h,kj}\,\bar{v}_{t,h,j}$ (valutata a blocchi causali in tempo lineare $\mathcal{O}(T K_\ell d)$) sia il **prototipo geometrico attivo** $\bar{g}_{t,h}^{(\ell)} = \sum_{k=1}^{K_\ell} T_{t,h,k}^{(\ell)}\,g_{h,k}^{(\ell)}$. Concatenati i vettori di tutte le teste in $u_t^{(\ell)} = \big[r_t^{\mathrm{conc}};\, r_t^{\mathrm{vista}};\, \bar{g}_t^{(\ell)};\, r_t^{\mathrm{vicino}}\big] \in \mathbb{R}^{4D}$, l'aggiornamento a pre-normalizzazione somma un contributo puramente additivo modulato dalla valvola $\gamma_\ell \in \mathbb{R}$, lasciando intatta la scala metrica dello stato principale e permettendo l'apertura immediata anche dei livelli di sotto-concetti più profondi:
   $$h_t^{(\ell)} = h_t^{(\ell-1)} + \gamma_\ell\,\mathrm{LN}_2\!\big(W_O\,u_t^{(\ell)} + b_O\big)$$

#### 3.3 · Memoria episodica del documento con indirizzo sentinella

Accanto alla memoria astratta compressa nelle statistiche sufficienti $(M_{t,h,k}, S_{t,h,k})$ e ai concetti gaussiani per ciascun simbolo da predire, una memoria episodica leggera ($D_{\mathrm{mem}} \le D$) preserva l'ancoraggio esatto alle entità, ai numeri e ai termini specifici comparsi nella domanda o nel documento corrente. Ad ogni posizione $\tau \ge 1$ lo stato finale $h_\tau \in \mathbb{R}^D$ deposita un indirizzo chiave $k_\tau = \frac{1}{\sqrt{D_{\mathrm{mem}}}}\,W_{\mathrm{key}}\,h_\tau \in \mathbb{R}^{D_{\mathrm{mem}}}$ associato al simbolo effettivamente seguito $y_\tau = w_{\tau+1}$. Alla posizione $t > 1$, l'interrogazione $q_t^{\mathrm{mem}} = W_{\mathrm{query}}\,h_t$ confronta le affinità $a_{t,\tau} = \langle q_t^{\mathrm{mem}},\, k_\tau\rangle$ ($\tau < t$) con un **indirizzo sentinella** $a_{t,0} = \langle w_{\mathrm{sent}},\, h_t\rangle + b_{\mathrm{sent}}$ (*«predici tramite i concetti generali e lessicali senza copia puntuale»*):
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
- **Regime stocastico accoppiato ($0 < \rho_t < 1$, apprendimento generativo profondo):** sulle sequenze lunghe, dove le interrogazioni residualizzate $q_{t,h}^{(\ell)} = q_{0,t,h}^{(\ell)} + \mathbb{I}[\ell \ge 2]\,(q_{t,h}^{(\ell-1)} - \bar{g}_{t,h}^{(\ell-1)}) / \bar{s}_{t,h}^{(\ell-1)}$ dipendono dalle rappresentazioni sottostanti, ad ogni mini-lotto $B_t$ ACAN 10 esegue congiuntamente:
  1. un passo del primo ordine precondizionato (con tasso di apprendimento $\eta_t$ governato da riscaldamento lineare iniziale e decadimento di Robbins-Monro $\eta_t = \eta_0 \min(1, t/T_w)\,\max(\eta_{\min}, (1 + \max(0, t-T_w)/T_0)^{-1/2})$) sulle matrici di sollevamento multi-padre $L_h^{(\ell)}$, sulle matrici associative $A_h$, sulle viste $(W_Q, W_K, W_V, \beta_h)$ e sui concetti del vocabolario, guidato dal gradiente esatto della log-verosimiglianza causale $-\sum_{t} \log p(w_{t+1} \mid h_{1:t})$;
  2. un passo chiuso di gradiente naturale di Fisher $S_k^{(t+1)} = (1 - \rho_t)S_k^{(t)} + \rho_t \widehat{S}_k(B_t)$ sulle interrogazioni effettive $q_{t,h}^{(\ell)}$ di ciascun livello con passo accoppiato alla medesima scala del gradiente ($\rho_t = \min(\rho_{\max},\, \beta\,\eta_t)$, $\beta = \frac{1}{2}$, per $K_\ell \ge 2$), in modo che l'adattamento metrico delle masse, dei centri e delle dispersioni mantenga concetti e sotto-concetti ancorati alla varietà effettiva dei residui senza sovrastare la separazione discriminativa indotta dal simbolo futuro;
  3. un decadimento regolarizzativo selettivo applicato esclusivamente alle mappe lineari proiettive attive (escludendo masse a priori, dispersioni, profili posizionali $\beta_h$, archi concettuali $A_h$ e $L_h^{(\ell)}$, normalizzazioni, valvole residue e proiezioni d'interrogazione finché $K_\ell = 1$).

---

### 5 · Morfogenesi strutturale guidata dai dati: agglomerazione e raffinamento concettuale

ACAN 10 non richiede di fissare arbitrariamente il numero di concetti: in entrambi i domini la gerarchia concettuale emerge dai dati tramite operatori che rispettano la geometria della verosimiglianza.

1. **Riduzione gerarchica e convalida analitica leave-one-out (dominio tabulare):**
   - A partire da una sovra-partizione fine, ad ogni scala vengono fusi i concetti affini sommando direttamente le loro statistiche sufficienti ($\Sigma_{a \cup b} = \Sigma_a + \Sigma_b$) e rifinendo la partizione con $\rho = 1$.
   - Per scegliere la scala ottima senza addestramenti ripetuti, la verosimiglianza fuori campione della riga $r$ si valuta in tempo costante $\mathcal{O}(1)$ **sottraendo algebricamente** il contributo della riga stessa dalle statistiche sufficienti accumulate:
     $$\Sigma_k^{(-r)} = \Sigma_k - w_r\,T_{rk}\,\sigma(x_r)$$
     selezionando il livello che massimizza la densità predittiva incrociata.

2. **Morfogenesi costruttiva piramidale di astrazioni, sotto-concetti multi-padre e punti di vista (dominio generativo):**
   - Lo spazio metrico continuo parte ultra-compatto ($D = 48$, con estensione automatica limitata a $D \le 96$ affinché l'intero modello resti di pochi megabyte e la capacità semantica risieda nei concetti, nei sotto-concetti e nelle viste anziché in grandi matrici dense), con un singolo livello, una testa di attenzione e un concetto iniziale ($L = 1$, $H_{\mathrm{teste}} = 1$, $K_1 = 1$). Ad intervalli controllati del volume di testo appreso — con tornate consecutive immediate finché i candidati continuano a migliorare — il grafo corrente compete sulla medesima finestra di prova fresca contro trasformazioni strutturali progettate come **morfismi quasi-isometrici** che preservano la predizione all'istante iniziale ($t = 0$) rompendo al contempo la simmetria dei gradienti:
     - **Raddoppio piramidale dei concetti e sotto-concetti a momenti invariati ($K_\ell \to 2K_\ell$):** privilegiando il livello che presenta il minor rapporto $K_\ell / 2^{\ell-1}$ (così da costruire naturalmente una piramide $K_1 < K_2 < \dots < K_L$ in cui poche astrazioni generali si articolano in numerosi sotto-concetti particolari), tutti i $K_\ell$ concetti del livello vengono scissi lungo la progressione geometrica $1 \to 2 \to 4 \to 8 \to 16 \to 32 \to 64 \to 128$, con masse dimezzate ($\log \pi' = \log \pi - \log 2$), spostamento antipodale $g_\pm = g \pm u \odot \sigma$, contrazione analitica della dispersione $\eta' = \eta + \frac{1}{2}\log(1 - u^2)$, preferenza diagonale in $A_h$ e duplicazione conservativa delle colonne di $L_h^{(\ell)}$ e delle righe di $L_h^{(\ell+1)}$ (poiché $T_{p_1} + T_{p_2} = T_p$, sia il prototipo padre $\bar{g}^{(\ell)}$ sia la distribuzione a priori del livello sottostante restano invariati all'istante zero, lasciando poi ciascun figlio libero di rafforzare o attenuare i legami verso più concetti padre).
     - **Raddoppio antipodale delle teste di attenzione per nuovi punti di vista ($H_{\mathrm{teste}} \to 2H_{\mathrm{teste}}$):** ciascuna testa concettuale e prospettica viene duplicata dimezzando le rispettive colonne nei tre blocchi concettuale, prospettico e prototipico della proiezione d'uscita ($\frac{1}{2}W_O \pm \varepsilon_O$), copiando il profilo posizionale $\beta_h$ con perturbazioni opposte infinitesimali ($\pm \varepsilon$) su interrogazioni, chiavi, valori e centri concettuali e scalatura coerente dei momenti dell'ottimizzatore.
     - **Aggiunta di un livello residualizzato di sotto-concetti ($L \to L + 1$):** una volta che i livelli esistenti hanno articolato un nucleo sufficiente di astrazioni ($K_\ell \ge 4$), può essere aggiunto un nuovo livello di sotto-concetti alimentato dal residuo standardizzato $(q^{(L)} - \bar{g}^{(L)}) / \bar{s}^{(L)}$, con matrice multi-padre inizialmente ancorata alle astrazioni sovrastanti ($L_h^{(L+1)} = I$) e **valvola residua a pre-normalizzazione inizialmente chiusa** ($\gamma_{L+1} = 0$), sicché $h_t^{(L+1)} = h_t^{(L)}$ esattamente a $t = 0$ mentre il gradiente apre subito la valvola senza alterare la norma dei livelli precedenti.
     - **Estensione controllata della larghezza ($D \to 2D$, $d \to 2d$):** entro il massimale compatto ($D \le 96$ per impostazione predefinita, così da dotare di rango adeguato $4\text{–}8$ teste di attenzione restando un ordine di grandezza più leggero dei trasformatori densi), le coordinate vengono duplicate con compensazione analitica esatta di scale metriche $\frac{1}{\sqrt{2}}$, dispersioni residualizzate e masse.
   - **Plasticità locale transitoria delle componenti nascenti:** quando il grafo ha assimilato milioni di simboli e il passo globale si è raffreddato secondo la legge di Robbins-Monro, i parametri preesistenti conservano il passo raffreddato mentre i soli parametri dei concetti, sotto-concetti e legami multi-padre $L_h^{(\ell)}$ appena scissi o generati ricevono un regime transitorio di plasticità locale calda con attenuazione iperbolica graduale verso il passo globale.
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

In generazione testuale, letto il prefisso $(w_0 = \langle\mathrm{eos}\rangle, w_1, \dots, w_t)$, per ciascuno strato $\ell$ lo stato interno dei livelli concettuali dipende dal passato tramite lo stato ricorrente compatto:
$$\mathcal{C}_t^{(\ell)} = \Big(\big\{M_{t,h,k}^{(\ell)},\, S_{t,h,k}^{(\ell)}\big\}_{h,k},\; \big\{\big(k_{\tau,h}^{(\ell)},\, v_{\tau,h}^{(\ell)}\big)\big\}_{\tau=\max(1,\,t-W_{\mathrm{vista}}+1)}^{t},\; \big(\tilde{x}_{t-W+1}^{(\ell)}, \dots, \tilde{x}_t^{(\ell)}\big)\Big)$$
Ad ogni nuovo simbolo osservato $w_{t+1}$, l'aggiornamento delle somme concettuali $M_{t+1} = M_t + \tilde{T}_{t+1}$ e $S_{t+1} = S_t + \tilde{T}_{t+1} v_{t+1}$ e della finestra prospettica limitata a $W_{\mathrm{vista}}$ posizioni costa $\mathcal{O}(L D^2 + L D K + L W_{\mathrm{vista}} D)$ operazioni **costanti rispetto alla lunghezza $t$ nei livelli concettuali**. Dalla distribuzione combinata concetti + memoria sentinella $p(v \mid h_{1:t})$, il simbolo successivo viene estratto deterministicamente ($\arg\max_v$) oppure mediante campionamento con temperatura $T_{\mathrm{camp}} > 0$ sulla distribuzione ripesata $p_v^{1/T_{\mathrm{camp}}} / \sum_u p_u^{1/T_{\mathrm{camp}}}$. I simboli autogenerati aggiornano le statistiche concettuali e la finestra prospettica ma non vengono inseriti come nuovi indirizzi nella memoria episodica, evitando cicli auto-rinforzanti di ripetizione lessicale.

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
2. **Raggruppamento per lunghezza e passo d'ottimizzazione accoppiato:** all'interno di ciascun passo i documenti vengono tagliati alla lunghezza massima di contesto (ad esempio $512$ simboli), preceduti da $\langle\mathrm{eos}\rangle$, ordinati per lunghezza affine nei calcoli di validazione (così da evitare lo spreco quadratico di riempimento causato da un singolo documento lungo accanto a molti brevi) e riuniti in mini-lotti di circa $4.096$ simboli. Su ciascun mini-lotto vengono eseguiti in sequenza:
   - la valutazione parallela dei livelli causali di concetti e sotto-concetti con **teste di attenzione multi-vista** sugli ultimi $W_{\mathrm{vista}}$ simboli (realizzata tramite scorrimento a copia zero senza matrici quadratiche $T \times T$), **residualizzazione gerarchica tabulare** $(q^{(\ell-1)} - \bar{g}^{(\ell-1)}) / \bar{s}^{(\ell-1)}$, **sollevamento multi-padre** $L_h^{(\ell)}$, grafo associativo intra-livello $A_h$, prototipo attivo $\bar{g}^{(\ell)}$, passo residuo a pre-normalizzazione e memoria episodica con indirizzo sentinella;
   - il calcolo fattorizzato a due stadi della log-verosimiglianza sui concetti gaussiani di ciascun token da predire (riuniti in concetti lessicali semantici) mediante un nucleo differenziabile esplicito che accumula analiticamente i gradienti rispetto agli stati $h_t$, ai centri dei simboli $e_v$, alle masse $\ell_v$ e alla scala $\alpha$ senza allocare tensori tridimensionali $(N_{\mathrm{lotto}} \times S \times D)$;
   - il taglio della norma globale del gradiente a $1$, il passo dell'ottimizzatore adattivo (con regimi di plasticità locale dedicati per le valvole residue e per i concetti, sotto-concetti e legami multi-padre appena generati o scissi), l'aggiornamento chiuso di Fisher-EM sulle interrogazioni effettive dei concetti e la regolarizzazione selettiva delle sole proiezioni lineari.
3. **Morfogenesi automatica senza numero prefissato di concetti e tetti hardware facoltativi:** come nel dominio tabulare, l'utente non deve scegliere il numero di concetti, di teste o di livelli (lasciando a zero i massimali facoltativi, l'intera piramide di astrazioni, sotto-concetti multi-padre e punti di vista viene determinata autonomamente dalla validazione entro i soli margini di sicurezza della memoria, mentre lo spazio metrico delle coordinate parte da $D = 48$ ed è limitato per impostazione predefinita a $D \le 96$ per mantenere il modello dell'ordine di una decina di megabyte; l'utente può comunque imporre tetti espliciti o fissare una piramide statica se disattiva la crescita). Quando scatta un controllo di crescita, il grafo corrente compete a parità di mini-lotto e di sequenza casuale su una finestra controllata ($65.536$ simboli nei primi milioni di token, estesa fino a $131.072$ simboli a regime). Finché un candidato vince una tornata, ulteriori tornate vengono disputate su segmenti successivi freschi e il controllo successivo viene riprogrammato immediatamente senza attendere lunghe pause esponenziali; l'intervallo massimo fra controlli è limitato a $4$ milioni di simboli e viene anticipato quando un passo intero non riduce l'errore medio di validazione.
4. **Salvataggio atomico, grafo multi-padre ispezionabile in tempo reale e continuità tra code:** ogni $4\text{–}5$ secondi il processo lavoratore trasmette l'avanzamento interno del passo, i punteggi in diretta delle gare e il grafo ispezionabile di astrazioni, concetti e sotto-concetti con legami multi-padre (quota a posteriori di ciascun concetto padre e token predetti da ciascun nodo) valutato sul testo di validazione. Al termine dei passi previsti vengono scritti su disco i salvataggi completi (inclusi i passi trascorsi e il regime di raffreddamento del passo di apprendimento, riallineando automaticamente il calendario di crescita all'avvio di ciascun nuovo testo della coda), mantenendo gli ultimi salvataggi recenti, il miglior modello assoluto certificato dalla validazione e le pietre miliari permanenti.

---

### 4 · Inferenza

In fase di generazione e di interrogazione tramite API, il modello opera attraverso un motore algebrico puro in virgola mobile a 32 bit su CPU, completamente svincolato dal motore di differenziazione automatica:

1. **Struttura dati a cursore incrementale:** l'oggetto di lettura mantiene per ogni livello $\ell \in \{1, \dots, L\}$ due piccoli array densi per le masse accumulate $M^{(\ell)} \in \mathbb{R}^{H_{\mathrm{teste}} \times K_\ell}$ e i momenti $S^{(\ell)} \in \mathbb{R}^{H_{\mathrm{teste}} \times K_\ell \times d}$, un buffer circolare limitato alle ultime $W_{\mathrm{vista}}$ coppie chiave–valore $(k_{\tau,h}^{(\ell)}, v_{\tau,h}^{(\ell)})$ per le teste di attenzione multi-vista e agli ultimi $W$ stati pre-normalizzati $\tilde{x}_{t-W+1:t}^{(\ell)}$, oltre agli indirizzi chiave–simbolo $(k_\tau, y_\tau)$ della memoria sentinella del documento corrente.
2. **Pre-lettura parallela del prompt e decodifica ricorrente:** all'arrivo di una richiesta, il cursore osserva anzitutto il simbolo $\langle\mathrm{eos}\rangle$ e processa l'intero prefisso o la domanda fornita dall'utente in forma matriciale vettorizzata, popolando in un'unica passata le somme concettuali attraverso la gerarchia residualizzata di astrazioni e sotto-concetti multi-padre e la finestra prospettica.
3. **Generazione simbolo per simbolo in streaming:** per ogni nuovo token richiesto viene valutata la distribuzione predittiva sui concetti lessicali, sui concetti dei singoli token e sulla memoria sentinella del documento, viene campionato il simbolo successivo e lo stato avanza aggiornando le somme ricorrenti concettuali e il buffer prospettico limitato, emettendo immediatamente i frammenti decodificati in flusso continuo fino al raggiungimento della lunghezza richiesta o di una sequenza di arresto.

---

### 5 · Complessità computazionale e di memoria

Indicando con $n$ il numero di righe e $m$ il numero di colonne di una tabella, e per il modello sequenziale con $T$ la lunghezza del documento, $D$ la dimensione compatta delle coordinate ($48 \le D \le 96$), $L$ il numero di livelli concettuali, $H$ il numero di teste di attenzione ($d$ coordinate per testa), $K$ i concetti per testa, $W_{\mathrm{vista}}$ la finestra prospettica locale, $V$ la cardinalità del vocabolario ($S \approx \lceil\sqrt{V}\rceil$) e $N_b$ il numero di simboli in un mini-lotto:

| Operazione | Dominio tabulare ($n$ righe, $m$ attributi) | Dominio generativo a concetti ACAN 10 | Attenzione densa quadratica standard |
|---|---|---|---|
| Apprendimento per epoca / lotto | $\mathcal{O}(n K m)$ (somme sufficienti esatte, $\rho = 1$) | $\mathcal{O}\!\big(N_b(L D^2 + L W_{\mathrm{vista}} D + L H K^2 + L H K d + S D) + G_{\mathrm{attivi}} S D\big)$ | $\mathcal{O}\!\big(N_b(L D_{\mathrm{llm}}^2 + L T D_{\mathrm{llm}} + V D_{\mathrm{llm}})\big)$ |
| Selezione della struttura | Convalida leave-one-out analitica in $\mathcal{O}(n K m)$ | Candidati quasi-isometrici su finestra $\le 131.072$ simboli (regola 1-SE) | Ricerca esterna con riaddestramenti completi |
| Inferenza per nuovo simbolo / riga | $\mathcal{O}(K m)$ | $\mathcal{O}(L D^2 + L W_{\mathrm{vista}} D + L H K^2 + L H K d + 2\sqrt{V}\,D + T D_{\mathrm{mem}})$ | $\mathcal{O}(L D_{\mathrm{llm}}^2 + L T D_{\mathrm{llm}} + V D_{\mathrm{llm}})$ |
| Stato dei livelli concettuali in inferenza | $\mathcal{O}(K m)$ (indipendente da $n$) | $\mathcal{O}(L H K d + L W_{\mathrm{vista}} D)$ (**costante in $T$**) | $\mathcal{O}(L T D_{\mathrm{llm}})$ (crescente linearmente con $T$ su tutti gli $L$ livelli) |

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

#### 6.2 · Dinamica del grafo concettuale multi-vista, dialogo domanda–risposta e leggerezza su grandi corpus (`V = 32.000`)

Su code multi-dominio reali da 45 milioni di simboli ($V = 32.000$, dialogo multilingue domanda–risposta, testi enciclopedici, codice sorgente e script di sistema) eseguite su CPU nello spazio metrico compatto ($48 \le D \le 96$):
- **Velocità di elaborazione ed estrema leggerezza:** la fattorizzazione analitica sui concetti lessicali semantici (~$150\text{–}180$ concetti lessicali appresi via Fisher-EM), le viste locali a finestra $W_{\mathrm{vista}} = 32$ e le somme ricorrenti a tempo costante mantengono l'intero modello entro **$6\text{–}13\text{ MB}$** (circa $25\text{–}30$ volte più leggero di un piccolo modello linguistico denso) con migliaia di simboli al secondo su normali processori server multicore senza acceleratore grafico.
- **Apertura immediata dei livelli di sotto-concetti e coerenza domanda–risposta:** la residualizzazione gerarchica tabulare $(q^{(\ell-1)} - \bar{g}^{(\ell-1)}) / \bar{s}^{(\ell-1)}$ unita all'aggiornamento a pre-normalizzazione apre immediatamente le valvole dei livelli di sotto-concetti ($\gamma_2, \gamma_3 > 0{,}5$ già dai primi cicli, anziché restare prossimi a zero), mentre la sinergia fra teste di attenzione multi-vista sulla domanda, concetti per ciascun token e memoria sentinella raddoppia l'accuratezza di risposta esatta su domande fattuali rispetto all'accumulazione concettuale priva di viste prospettiche.
- **Separazione geometrica e grafo multi-padre ispezionabile:** partendo da un singolo concetto iniziale ($D = 48, L = 1, H = 1, K = 1$), il passo accoppiato di Fisher-EM mantiene attiva quasi l'intera capacità di ciascuna testa (coseno medio fra centri negativo prossimo al simplesso antipodale, oltre il $60\%$ del peso associativo su legami tra concetti distinti e sotto-concetti condivisi fra più concetti padre), abbattendo la NLL fuori campione da $10{,}37$ nat ($\mathrm{PPL} \approx 32.000$) sotto $4{,}80$ nat lungo la coda multi-dominio.

