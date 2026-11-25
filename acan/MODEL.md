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

Quando il vocabolario $V$ è ampio (ad esempio $V = 32.000$), la valutazione diretta dei $V$ concetti richiederebbe $\mathcal{O}(V D)$ operazioni per posizione. ACAN 10 ordina i simboli per frequenza decrescente e li partiziona in $G \approx \lceil\sqrt{V}\rceil$ gruppi contigui $\mathcal{G}_1, \dots, \mathcal{G}_G$ di cardinalità massima $S \approx \lceil\sqrt{V}\rceil$. Ogni gruppo possiede una massa aggregata $\bar{\ell}_g = \log \sum_{v \in \mathcal{G}_g} \exp(\ell_v)$ e un baricentro ponderato:
$$\bar{e}_g = \sum_{v \in \mathcal{G}_g} \frac{\exp(\ell_v)}{\exp(\bar{\ell}_g)}\,e_v \in \mathbb{R}^D$$
La probabilità del simbolo successivo si fattorizza in modo esatto in due scelte di complessità $\mathcal{O}(\sqrt{V}\,D)$ ciascuna — prima il gruppo di appartenenza $g(v)$, poi il simbolo specifico all'interno del solo gruppo osservato:
$$\log p(v \mid h_t) = \log p\big(g(v) \mid h_t\big) + \log p\big(v \;\big|\; g(v),\, h_t\big)$$
riducendo il costo computazionale e di memoria da $\mathcal{O}(V D)$ a $\mathcal{O}(2\sqrt{V}\,D)$ senza alcuna approssimazione campionaria e conservando una distribuzione normalizzata esatta sull'intero vocabolario.

---

### 3 · Appartenenza a posteriori, attenzione causale a concetti e memoria episodica

#### 3.1 · Operatore di appartenenza bayesiana ed espansione quadratica

Dato un vettore continuo $q \in \mathbb{R}^d$ (una riga tabulare proiettata o un'interrogazione nello spazio di una testa d'attenzione) e una famiglia di $K$ concetti gaussiani diagonali $\{(\pi_k, g_k, s_k)\}_{k=1}^K$ con $s_k = \exp(\eta_k) \in \mathbb{R}_+^d$, il logaritmo non normalizzato della responsabilità a posteriori vale:
$$\Phi_k(q) = \log \pi_k - \sum_{m=1}^d \eta_{km} - \frac{1}{2}\sum_{m=1}^d \frac{(q_m - g_{km})^2}{s_{km}^2}, \qquad T_k(q) = \frac{\exp\!\big(\Phi_k(q) - \max_j \Phi_j(q)\big)}{\sum_{j=1}^K \exp\!\big(\Phi_j(q) - \max_{j'} \Phi_{j'}(q)\big)}$$
Sviluppando il quadrato con le precisioni $\lambda_{km} = s_{km}^{-2} = \exp(-2\eta_{km})$:
$$\sum_{m=1}^d \frac{(q_m - g_{km})^2}{s_{km}^2} = \langle q \odot q,\, \lambda_k\rangle - 2\,\langle q,\, g_k \odot \lambda_k\rangle + \langle g_k \odot g_k,\, \lambda_k\rangle$$
il calcolo simultaneo su tutte le posizioni di una sequenza si riduce a prodotti matriciali densi tra matrici $(T \times d)$ e $(d \times K)$, evitando la materializzazione di tensori tridimensionali intermedi $(T \times K \times d)$.

#### 3.2 · Strato di attenzione causale multi-testa mediato dai concetti

Sia $H \in \mathbb{R}^{T \times D}$ la sequenza degli stati in ingresso allo strato $\ell \in \{1, \dots, L\}$, ripartita su $H_{\mathrm{teste}}$ teste indipendenti di dimensione $d$ ($H_{\mathrm{teste}}\,d = D$) ciascuna dotata di $K_\ell$ concetti. Per ogni posizione $t \in \{1, \dots, T\}$ e ogni testa $h \in \{1, \dots, H_{\mathrm{teste}}\}$:

1. **Interrogazione, valore e intensità selettiva di scrittura:**
   $$q_{t,h} = W_{Q,h}\,h_t \in \mathbb{R}^d, \qquad v_{t,h} = W_{V,h}\,h_t \in \mathbb{R}^d, \qquad s_{t,h} = \frac{1}{2} + \sigma\!\left(\sqrt{D}\,\frac{\langle w_h,\, h_t\rangle}{\|h_t\|_2 + \varepsilon}\right) \in \left(\frac{1}{2},\, \frac{3}{2}\right)$$
   La responsabilità del concetto $k$ è $T_{t,h,k} = T_k(q_{t,h})$ e la massa depositata nella memoria concettuale è $\tilde{T}_{t,h,k} = s_{t,h}\,T_{t,h,k}$.
2. **Accumulazione causale delle statistiche sufficienti del prefisso:**
   $$M_{t,h,k} = \varepsilon + \sum_{\tau=1}^t \tilde{T}_{\tau,h,k} \in \mathbb{R}_+, \qquad S_{t,h,k} = \sum_{\tau=1}^t \tilde{T}_{\tau,h,k}\,v_{\tau,h} \in \mathbb{R}^d, \qquad \bar{v}_{t,h,k} = \frac{S_{t,h,k}}{M_{t,h,k}}$$
3. **Lettura tramite operatore stocastico di relazione tra concetti:** una matrice apprendibile $A_h \in \mathbb{R}^{K_\ell \times K_\ell}$, inizializzata con dominanza diagonale ($A_h^{(0)} = \frac{3}{2}\,I_{K_\ell}$) affinché ciascun concetto legga primariamente il proprio riassunto prima di apprendere legami incrociati, definisce la matrice stocastica per righe $P_h = \mathrm{softmax}_{\mathrm{righe}}(A_h)$:
   $$r_{t,h} = \sum_{k=1}^{K_\ell} T_{t,h,k} \sum_{j=1}^{K_\ell} P_{h,kj}\,\bar{v}_{t,h,j} = \sum_{j=1}^{K_\ell} \left(\frac{\sum_{k=1}^{K_\ell} T_{t,h,k}\,P_{h,kj}}{M_{t,h,j}}\right) S_{t,h,j}$$
   In addestramento la somma causale è valutata a blocchi di ampiezza $C$ combinando l'interazione causale intra-blocco con lo stato sufficiente accumulato dai blocchi precedenti, preservando la complessità temporale lineare $\mathcal{O}(T K_\ell d)$.
4. **Convoluzione causale a corto raggio e aggiornamento residuo valvolato:** per catturare l'ordine posizionale esatto a breve distanza, una finestra causale di ampiezza $W$ combina gli ultimi $W$ stati:
   $$u_t = \sum_{w=1}^{W} \mathbb{I}[t > w]\,W_{\mathrm{loc},w}\,h_{t-w}$$
   Concatenate le letture delle teste $r_t = [r_{t,1}; \dots; r_{t,H_{\mathrm{teste}}}]$ con $u_t$, lo stato aggiornato attraversa una doppia normalizzazione di strato regolata dallo scalare di apertura residua $\gamma_\ell \in \mathbb{R}$:
   $$\tilde{h}_t = \mathrm{LN}_2\!\Big(\mathrm{LN}_1\!\big(h_t + W_O\,[r_t;\, u_t] + b_O\big)\Big), \qquad h_t^{(\ell)} = h_t + \gamma_\ell\,\big(\tilde{h}_t - h_t\big)$$

#### 3.3 · Memoria episodica del documento con indirizzo sentinella

Mentre le statistiche concettuali $(M_{t,h,k}, S_{t,h,k})$ comprimono la semantica globale del documento in dimensione fissa, i riferimenti letterali esatti (nomi propri, identificatori, cifre introdotte una sola volta nel contesto) sono gestiti da una **memoria episodica associativa causale**.

Ad ogni posizione $\tau \ge 1$, lo stato finale $h_\tau \in \mathbb{R}^D$ deposita un indirizzo chiave $k_\tau = \frac{1}{\sqrt{D_{\mathrm{mem}}}}\,W_{\mathrm{key}}\,h_\tau \in \mathbb{R}^{D_{\mathrm{mem}}}$ associato al simbolo effettivamente seguito $y_\tau = w_{\tau+1}$. Alla posizione corrente $t > 1$, l'interrogazione $q_t^{\mathrm{mem}} = W_{\mathrm{query}}\,h_t$ confronta le affinità $a_{t,\tau} = \langle q_t^{\mathrm{mem}},\, k_\tau\rangle$ per $\tau < t$ con il punteggio scalare di un **indirizzo sentinella** $a_{t,0} = \langle w_{\mathrm{sent}},\, h_t\rangle + b_{\mathrm{sent}}$ che rappresenta l'evento *«nessuna posizione precedente corrisponde al contesto attuale»*:
$$\alpha_{t,0} = \frac{\exp(a_{t,0})}{\exp(a_{t,0}) + \sum_{\tau'=1}^{t-1} \exp(a_{t,\tau'})}, \qquad \alpha_{t,\tau} = \frac{\exp(a_{t,\tau})}{\exp(a_{t,0}) + \sum_{\tau'=1}^{t-1} \exp(a_{t,\tau'})} \quad (\tau = 1, \dots, t-1)$$
La distribuzione predittiva complessiva è la mistura convessa esatta:
$$p(w_{t+1} = v \mid h_{1:t}) = \alpha_{t,0}\,p_{\mathrm{concetti}}(v \mid h_t) + \sum_{\tau=1}^{t-1} \alpha_{t,\tau}\,\mathbb{I}[y_\tau = v]$$
Un'unica normalizzazione esponenziale determina simultaneamente quali occorrenze passate richiamare e quanta fiducia accordare alla rete concettuale ($\alpha_{t,0}$) rispetto alla copia contestuale ($1 - \alpha_{t,0}$): quando nessun antecedente raggiunge affinità superiore alla sentinella, $\alpha_{t,0} \to 1$ e il meccanismo di copia tace automaticamente.

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
- **Regime batch esatto ($\rho_t = 1$, apprendimento tabulare):** sostituendo $\rho_t = 1$ si ottiene $S_k^{(t+1)} = \widehat{S}_k(X)$, ossia il passo di massimizzazione esatta di Expectation-Maximization su tabelle intere o suddivise in blocchi additivi, con aumento monotono garantito della log-verosimiglianza marginale $\mathcal{L}(\theta^{(t+1)}) \ge \mathcal{L}(\theta^{(t)})$.
- **Regime stocastico accoppiato ($0 < \rho_t < 1$, apprendimento generativo profondo):** sulle sequenze lunghe, dove le interrogazioni $q_{t,h} = W_{Q,h}\,h_t$ dipendono dalle rappresentazioni degli strati sottostanti, ad ogni mini-lotto $B_t$ ACAN 10 esegue congiuntamente:
  1. un passo del primo ordine precondizionato (con tasso di apprendimento $\eta_t$ governato da riscaldamento lineare iniziale e decadimento di Robbins-Monro $\eta_t = \eta_0 \min(1, t/T_w)\,\max(\eta_{\min}, (1 + \max(0, t-T_w)/T_0)^{-1/2})$) su tutte le proiezioni lineari, le matrici di relazione $A_h$, i centri del vocabolario e la memoria episodica, guidato dal gradiente esatto della log-verosimiglianza causale $-\sum_{t} \log p(w_{t+1} \mid h_{1:t})$;
  2. un passo chiuso di gradiente naturale di Fisher $S_k^{(t+1)} = (1 - \rho_t)S_k^{(t)} + \rho_t \widehat{S}_k(B_t)$ con passo accoppiato alla medesima scala del gradiente ($\rho_t = \min(\rho_{\max},\, \beta\,\eta_t)$, $\beta = \frac{1}{2}$, per $K_\ell \ge 2$), in modo che l'adattamento metrico delle masse, dei centri e delle dispersioni mantenga i concetti ancorati alla varietà effettiva delle interrogazioni senza sovrastare la separazione discriminativa indotta dal simbolo futuro;
  3. un decadimento regolarizzativo selettivo applicato esclusivamente alle mappe lineari proiettive attive (escludendo masse a priori, dispersioni, normalizzazioni, valvole residue e proiezioni d'interrogazione finché $K_\ell = 1$).

---

### 5 · Morfogenesi strutturale guidata dai dati: agglomerazione e crescita conservativa

ACAN 10 non richiede di fissare arbitrariamente il numero di concetti o la taglia della rete: in entrambi i domini la struttura emerge dai dati tramite operatori che rispettano la geometria della verosimiglianza.

1. **Riduzione gerarchica e convalida analitica leave-one-out (dominio tabulare):**
   - A partire da una sovra-partizione fine, ad ogni scala vengono fusi i concetti affini sommando direttamente le loro statistiche sufficienti ($\Sigma_{a \cup b} = \Sigma_a + \Sigma_b$) e rifinendo la partizione con $\rho = 1$.
   - Per scegliere la scala ottima senza addestramenti ripetuti, la verosimiglianza fuori campione della riga $r$ si valuta in tempo costante $\mathcal{O}(1)$ **sottraendo algebricamente** il contributo della riga stessa dalle statistiche sufficienti accumulate:
     $$\Sigma_k^{(-r)} = \Sigma_k - w_r\,T_{rk}\,\sigma(x_r)$$
     selezionando il livello che massimizza la densità predittiva incrociata.

2. **Morfogenesi costruttiva mediante immersioni conservative (dominio generativo):**
   - La rete può partire dal seme strutturale minimo ($D = 32$, $L = 1$, $H_{\mathrm{teste}} = 1$, $K_1 = 1$). Ad intervalli raddoppianti del volume di testo appreso (o non appena il guadagno su un passo si arresta), la rete corrente compete sulla medesima finestra di prova fresca contro quattro trasformazioni strutturali progettate come **morfismi quasi-isometrici**, tali cioè da preservare la funzione predittiva all'istante iniziale ($t = 0$) pur rompendo la simmetria dei gradienti:
     - **Raddoppio della larghezza ($D \to 2D$, $d \to 2d$):** ogni coordinata di stato e di centro viene duplicata con perturbazioni antipodali minime ($w \pm \varepsilon$, media identica), le mappe di lettura dimezzano i pesi su ciascuna copia, le dispersioni scalano di $\sqrt{2}$, le masse logaritmiche vengono compensate analiticamente di $\sum_m \eta_{km} + d\log 2$ e il passo di ottimizzazione dimezza perché le coordinate evolvono in coppia.
     - **Raddoppio delle teste ($H_{\mathrm{teste}} \to 2H_{\mathrm{teste}}$):** le teste esistenti e le relative statistiche dell'ottimizzatore restano intatte; le nuove teste entrano nella proiezione d'uscita $W_O$ con colonne inizialmente nulle (uscita invariata a $t = 0$) e secondo momento adattivo calibrato sul primo gradiente per evitare sbalzi iniziali.
     - **Aggiunta di uno strato ($L \to L + 1$):** viene appeso un nuovo strato finale con pesi d'uscita non nulli a piccola varianza ma **valvola residua chiusa** ($\gamma_{L+1} = 0$), sicché $h_t^{(L+1)} = h_t^{(L)}$ esattamente a $t = 0$, mentre un passo di apprendimento dedicato ($8\times$) e simmetrico su $\gamma_{L+1} \in [-1{,}5,\, 1{,}5]$ consente al gradiente di aprirlo progressivamente nel segno richiesto dai dati.
     - **Suddivisione di un concetto ($K_\ell \to K_\ell + 1$):** nello strato selezionato a rotazione, il concetto a maggior massa d'occupazione viene scisso in due concetti figli con masse dimezzate ($\log \pi' = \log \pi - \log 2$), dispersioni lievemente contratte ($\eta' = \eta - 0{,}15$ per evitare il punto sella contrattivo della mistura gaussiana), centri spostati antipodalmente ($g \pm \delta$) e preferenza diagonale in $A_h$. Quando lo strato parte da $K_\ell = 1$ (dove $\mathrm{softmax}([\Phi_1]) \equiv 1$ per qualunque parametro), centro e dispersione vengono prima allineati in modo esatto a media e varianza empiriche delle interrogazioni senza alterare di un solo bit l'uscita, attivando poi un passo accelerato ($4\times$) per ruotare rapidamente la proiezione $W_{Q,h}$.
   - **Principio di parsimonia statistica:** su un insieme di validazione disgiunto da quello esposto all'utente, un candidato di crescita viene adottato solo se supera il riferimento di partenza per **più di un errore standard della differenza appaiata**:
     $$\bar{\Delta} > \frac{s_\Delta}{\sqrt{M - 1}}, \qquad \Delta_i = \ell_{\mathrm{cand}}(x_i) - \ell_{\mathrm{rif}}(x_i)$$
     e, fra tutti i candidati entro un errore standard dal migliore, prevale sempre quello con il **minor numero di parametri**. Se un passo ordinario su testo fresco produce una variazione neutra entro un errore standard, i pesi aggiornati vengono mantenuti senza dimezzare prematuramente il passo di apprendimento.

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

In generazione testuale, letto il prefisso $(w_0 = \langle\mathrm{eos}\rangle, w_1, \dots, w_t)$, per ciascuno strato $\ell$ lo stato interno dipende dall'intero passato esclusivamente tramite la tripla ricorrente:
$$\mathcal{C}_t^{(\ell)} = \Big(\big\{M_{t,h,k}^{(\ell)},\, S_{t,h,k}^{(\ell)}\big\}_{h,k},\; \big(h_{t-W+1}^{(\ell-1)}, \dots, h_t^{(\ell-1)}\big)\Big)$$
Ad ogni nuovo simbolo osservato $w_{t+1}$, l'aggiornamento $M_{t+1} = M_t + \tilde{T}_{t+1}$ e $S_{t+1} = S_t + \tilde{T}_{t+1} v_{t+1}$ costa $\mathcal{O}(L D^2 + L D K)$ operazioni **indipendentemente dalla lunghezza $t$ del documento**. Dalla distribuzione combinata rete + memoria episodica $p(v \mid h_{1:t})$, il simbolo successivo viene estratto deterministicamente ($\arg\max_v$) oppure mediante campionamento con temperatura $T_{\mathrm{camp}} > 0$ sulla distribuzione ripesata $p_v^{1/T_{\mathrm{camp}}} / \sum_u p_u^{1/T_{\mathrm{camp}}}$. I simboli autogenerati aggiornano le statistiche concettuali ma non vengono inseriti come nuovi indirizzi nella memoria episodica, evitando cicli auto-rinforzanti di ripetizione lessicale.

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
2. **Raggruppamento per lunghezza e passo d'ottimizzazione ibrido:** all'interno di ciascun passo i documenti vengono tagliati alla lunghezza massima di contesto (ad esempio $512$ simboli), preceduti da $\langle\mathrm{eos}\rangle$, ordinati per lunghezza affine nei calcoli di validazione (così da evitare lo spreco quadratico di riempimento causato da un singolo documento lungo accanto a molti brevi) e riuniti in mini-lotti di circa $4.096$ simboli. Su ciascun mini-lotto vengono eseguiti in sequenza:
   - la valutazione parallela degli strati causali a concetti, della finestra locale e della memoria episodica con sentinella;
   - il calcolo fattorizzato a due stadi della log-verosimiglianza sui gruppi di frequenza mediante un nucleo differenziabile esplicito che accumula analiticamente i gradienti rispetto agli stati $h_t$, ai centri dei simboli $e_v$, alle masse $\ell_v$ e alla scala $\alpha$ senza allocare tensori tridimensionali $(N_{\mathrm{lotto}} \times S \times D)$;
   - il taglio della norma globale del gradiente a $1$, il passo dell'ottimizzatore adattivo (con moltiplicatori di velocità dedicati per l'apertura delle valvole residue e per le interrogazioni dei concetti appena scissi da $K=1$), l'aggiornamento chiuso di Fisher-EM sulle statistiche dei concetti e la regolarizzazione selettiva delle proiezioni lineari.
3. **Gare di morfogenesi su finestra controllata:** quando scatta un controllo di crescita, la rete corrente e i quattro candidati morfogenetici (larghezza doppia, uno strato in più, un concetto scisso in due, il doppio delle teste) vengono addestrati a parità di dimensione del mini-lotto e di sequenza casuale su una finestra iniziale limitata ($65.536$ simboli). Se un candidato migliorativo vince la prima tornata, una seconda tornata viene disputata sul segmento successivo fresco di $65.536$ simboli; il vincitore completa quindi i restanti simboli del milione del passo. I progressi del passo vengono conservati ogniqualvolta non vi sia un peggioramento statisticamente significativo oltre un errore standard.
4. **Salvataggio atomico, telemetria in tempo reale e ripresa:** ogni $4\text{–}5$ secondi il processo lavoratore trasmette l'avanzamento interno del passo e i punteggi live delle gare all'interfaccia grafica, che aggiorna automaticamente la coda attiva con curve continue spezzate solo sui salti effettivi. Al termine dei passi previsti vengono scritti su disco i salvataggi completi (incluso lo stato dell'ottimizzatore), mantenendo gli ultimi salvataggi recenti, il miglior modello assoluto certificato dalla validazione e le pietre miliari permanenti.

---

### 4 · Inferenza

In fase di generazione e di interrogazione tramite API, il modello opera attraverso un motore algebrico puro in virgola mobile a 32 bit su CPU, completamente svincolato dal motore di differenziazione automatica:

1. **Struttura dati a cursore incrementale:** l'oggetto di lettura mantiene per ogni strato $\ell \in \{1, \dots, L\}$ due piccoli array densi per le masse accumulate $M^{(\ell)} \in \mathbb{R}^{H_{\mathrm{teste}} \times K_\ell}$ e i momenti $S^{(\ell)} \in \mathbb{R}^{H_{\mathrm{teste}} \times K_\ell \times d}$, un buffer circolare degli ultimi $W$ stati $h_{t-W+1:t}^{(\ell-1)}$ e la lista sequenziale delle coppie indirizzo-simbolo $(k_\tau, y_\tau)$ della memoria episodica del documento corrente.
2. **Pre-lettura parallela del prompt e decodifica ricorrente:** all'arrivo di una richiesta, il cursore osserva anzitutto il simbolo $\langle\mathrm{eos}\rangle$ e processa l'intero prefisso fornito dall'utente in forma matriciale vettorizzata, popolando in un'unica passata le somme concettuali e gli indirizzi episodici del contesto.
3. **Generazione simbolo per simbolo in streaming:** per ogni nuovo token richiesto viene valutata la distribuzione predittiva combinata (concetti + memoria episodica con sentinella), viene campionato il simbolo successivo e lo stato avanza aggiornando le sole somme ricorrenti concettuali con esclusione della registrazione episodica delle parole autogenerate, emettendo immediatamente i frammenti decodificati in flusso continuo fino al raggiungimento della lunghezza richiesta o di una sequenza di arresto.

---

### 5 · Complessità computazionale e di memoria

Indicando con $n$ il numero di righe e $m$ il numero di colonne di una tabella, e per il modello sequenziale con $T$ la lunghezza del documento, $D$ la dimensione dello stato, $L$ il numero di strati, $H$ il numero di teste ($Hd = D$), $K$ i concetti per testa, $W$ la finestra locale, $V$ la cardinalità del vocabolario ($S \approx \lceil\sqrt{V}\rceil$) e $N_b$ il numero di simboli in un mini-lotto:

| Operazione | Dominio tabulare ($n$ righe, $m$ attributi) | Dominio generativo a concetti ACAN 10 | Attenzione densa quadratica standard |
|---|---|---|---|
| Apprendimento per epoca / lotto | $\mathcal{O}(n K m)$ (somme sufficienti esatte, $\rho = 1$) | $\mathcal{O}\!\big(N_b(L D^2 + L D K + S D) + G_{\mathrm{attivi}} S D\big)$ | $\mathcal{O}\!\big(N_b(L D^2 + L T D + V D)\big)$ |
| Selezione della struttura | Convalida leave-one-out analitica in $\mathcal{O}(n K m)$ | 5 candidati su finestra limitata $\le 65.536$ simboli | Ricerca esterna con riaddestramenti completi |
| Inferenza per nuovo simbolo / riga | $\mathcal{O}(K m)$ | $\mathcal{O}(L D^2 + L D K + 2\sqrt{V}\,D + T D_{\mathrm{mem}})$ | $\mathcal{O}(L D^2 + L T D + V D)$ |
| Stato degli strati in inferenza | $\mathcal{O}(K m)$ (indipendente da $n$) | $\mathcal{O}(L H K d + L W D)$ (**costante in $T$**) | $\mathcal{O}(L T D)$ (crescente linearmente con $T$) |

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

#### 6.2 · Dinamica di apprendimento, morfogenesi e velocità su grandi corpus testuali (`V = 32.000`)

Su code multi-dominio reali da 45 milioni di simboli ($V = 32.000$, dialogo multilingue, testi enciclopedici, codice sorgente e script di sistema) eseguite interamente su CPU:
- **Velocità di elaborazione:** la fattorizzazione analitica per gruppi di frequenza e le gare su finestra controllata sostengono una cadenza effettiva compresa fra **4.000 e 6.500 simboli al secondo** su processori server standard multicore senza acceleratore grafico.
- **Morfogenesi automatica e discesa monotona:** partendo dal seme minimo ($D = 32, L = 1, H = 1, K = 1$), la rete seleziona autonomamente l'aumento delle teste, la suddivisione dei concetti, il raddoppio della larghezza e l'aggiunta di nuovi strati, riducendo con continuità la NLL di validazione da $10{,}37$ nat ($\mathrm{PPL} \approx 32.000$) fino a circa $5{,}15\text{–}5{,}20$ nat nei primi quindici milioni di simboli e sotto $4{,}80$ nat su dialoghi bilingui strutturati.
- **Effetto della memoria episodica con sentinella e delle riletture:** sui dialoghi bilingui italiano/inglese ($D = 96, L = 3, H = 4, K = 64$), l'indirizzo sentinella riduce la NLL fuori campione da $5{,}651$ (rete priva di memoria episodica) a **$4{,}805$**, dimezza la frazione di risposte tronche sotto i 15 simboli ($10\% \to 3\%$) e riduce dal $47\%$ al **$32\%$** la ripetizione letterale delle parole della domanda. A parità di simboli elaborati, l'esposizione a nuovi documenti risulta più informativa della rilettura ripetuta del medesimo sottoinsieme ($4{,}716$ nat contro $4{,}877$ nat).
