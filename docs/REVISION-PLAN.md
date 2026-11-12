# Piano di revisione del progetto ACAN

Base matematica: [MATHEMATICAL-FOUNDATION.md](MATHEMATICAL-FOUNDATION.md).
Questo piano dice che cosa cambiare, in che ordine, e con quali criteri
misurabili decidere se proseguire.

## Stato

| Fase | Stato |
|---|---|
| 0 Misura affidabile | in corso in questa revisione |
| 1 Nucleo sul tabulare | **fatta**: profilo `cp-v9`, predefinito nell'app. Criterio rispettato sui dati disponibili, con Titanic in regressione ([CP-V9-RESULTS.md](CP-V9-RESULTS.md)). Selezione dei concetti per generalizzazione giudicata, al posto della regola MML prevista in 1.4; gerarchia solo su richiesta |
| 2a Raggruppabilità a pesi invariati | **eseguita, esito negativo**: lo strato a concetti non approssima l'attenzione densa di SmolLM2 senza addestramento; causa principale la rotazione posizionale delle chiavi |
| 2b Strato a concetti | subordinata alle due verifiche indicate in CP-V9-RESULTS.md |
| 3–4 | non iniziate; richiedono GPU |

## 1. Obiettivo, riformulato

L'obiettivo dichiarato è un modello linguistico che «risolve il problema
dell'attenzione quadratica». La Proposizione 5 della fondazione mostra che
nessuna architettura con stato costante conserva tutto il contesto: l'obiettivo
va scritto in forma falsificabile.

> **Obiettivo professionale.** Uno strato di attenzione a concetti che, a stato
> di decodifica fissato $(k,w)$ per testa, (1) ha costo per token indipendente
> dalla lunghezza del contesto, (2) ha un errore rispetto all'attenzione densa
> con limite dimostrato e misurato, (3) si innesta su un modello preaddestrato
> senza distruggerne la funzione, (4) dichiara la propria frontiera
> qualità/memoria su benchmark pubblici contro le alternative sub-quadratiche
> esistenti, (5) mantiene statistiche di slot leggibili.

Il risultato pubblicabile non è «più veloce di 38×», ma una curva: perdita di
qualità in funzione dello stato, confrontata a parità di condizioni.

## 2. Stato attuale: che cosa dicono le misure

Dalle due esecuzioni del laboratorio generativo (SmolLM2 a 30 livelli, 15
innestati):

| | Esecuzione A | Esecuzione B |
|---|---|---|
| Slot / finestra / peso CP | 16 / 16 / – | 8 / 16 / 0,1 |
| Passi | 1000 (completata) | 218 (in pausa) |
| NLL training | 2,005 | – |
| NLL validazione | 4,217 | 4,381 |
| NLL base prima dell'innesto | 4,683 | 4,829 |
| Token/s | 28,1 | 4,9 |
| Parametri addestrabili / totali | 2,68 M / 143,8 M | 2,61 M / 143,8 M |

Lettura:

1. **Sovradattamento.** In A la NLL di validazione tocca il minimo (circa 3,6,
   letto dal grafico) verso il passo 400–450 e risale a 4,22 mentre il training
   scende a 2,0. Il valore finale non rappresenta il modello migliore.
2. **Il confronto «uguale o migliore del modello originale» non è valido.** Il
   riferimento è il modello base *senza* adattamento al corpus; il modello
   innestato è stato adattato con LoRA sullo stesso corpus. Il controllo
   corretto è il modello denso con la stessa LoRA, gli stessi dati e gli stessi
   passi. È probabile che quel controllo scenda ben sotto 4,2.
3. **La validazione è troppo piccola.** 256 token di controllo; la NLL base
   differisce di 0,15 fra le due esecuzioni (4,683 e 4,829), più della
   differenza fra molte configurazioni che si vorrebbero confrontare.
4. **Nessuna evidenza che la memoria a concetti contribuisca.** Manca
   l'ablazione con la sola finestra locale. Il gradiente dei parametri ACAN è
   un decimo di quello LoRA (0,088 contro 0,999): l'apprendimento passa
   quasi interamente dagli adattatori.
5. **L'innesto distrugge la funzione preaddestrata.** La NLL parte da circa 6,7
   contro 4,7 del modello base: i 15 livelli superiori ripartono da prototipi
   casuali, e la finestra locale riusa le proiezioni in un calcolo a testa
   singola per cui non sono state addestrate.
6. **Velocità.** 5–28 token/s: il profilo `cp-joint` esegue un doppio ciclo
   Python su batch e token. L'ordine di grandezza è opposto alla tesi
   dell'efficienza, e i 15 livelli inferiori restano densi, quindi il modello
   intero è ancora quadratico.

Sul lato tabulare: Gioco è recuperato (4 concetti), ma Letter usa 3.346
prototipi per 26 classi e i confronti su altri file sono misti; i dataset sono
stati esaminati ripetutamente e non esiste un test mai usato.

## 3. Principi di lavoro

- **Un'affermazione, una misura, un controllo.** Nessun numero senza il
  confronto a parità di dati, passi e parametri addestrabili.
- **Test bloccato.** Un insieme di test per dominio, scelto ora e letto una
  sola volta per fase. Lo sviluppo usa la validazione.
- **Criteri fissati prima.** Soglie di passaggio scritte prima dell'esperimento
  (sezione 5). Un esito negativo si registra, non si riformula.
- **Un solo nucleo.** Una formula vive in un solo file; i profili storici
  restano caricabili ma non ricevono sviluppo.

## 4. Fasi

### Fase 0 — Misura affidabile (1–2 settimane)

Nessun cambio di modello: si rende vero ciò che l'interfaccia mostra.

| # | Intervento | File principali |
|---|---|---|
| 0.1 | Controllo **denso + LoRA** con stesso corpus, passi, rango, seed; mostrato accanto a ogni progetto | `acan/lm/trainer.py`, `research_trainer.py`, `vis/generative.py` |
| 0.2 | Ablazioni a un clic: **sola finestra** (ramo concetti azzerato), $\lambda=0$, soli concetti | `acan/lm/grafting.py`, `research_options.py` |
| 0.3 | Validazione di almeno 50.000 token con intervallo di confidenza (bootstrap per documento); test separato e bloccato | `acan/lm/evaluation.py`, `research_data.py` |
| 0.4 | Sostituire il messaggio «uguale o migliore del modello originale» con il confronto contro 0.1; riportare il **miglior** checkpoint e il passo, non l'ultimo | `acan/vis/generative.py` |
| 0.5 | Throughput e memoria misurati sullo stesso hardware per denso, finestra e ACAN, a $L\in\{256,1024,4096\}$ | `acan/lm/diagnostics.py`, `resources.py` |
| 0.6 | Ritiro formale delle tabelle del paper LM (38×, −99,2%, PPL 1,51, infilling 100%) e dei Teoremi 1 e 3: nota di errata in `paper/` e in `README.md`; i PDF restano come archivio storico | `paper/`, `docs/PAPER-AUDIT.md` |
| 0.7 | `scripts/verify_math_foundation.py` nei controlli di CI | `.github/workflows/ci.yml` |

**Uscita:** una tabella con denso+LoRA, sola finestra, ACAN attuale, con
intervalli. È la linea di base di tutto ciò che segue.

### Fase 1 — Nucleo matematico v9 sul tabulare (2–3 settimane)

Il tabulare è il banco di prova economico della fondazione (parte B).

| # | Intervento |
|---|---|
| 1.1 | `acan/algebra.py`: tipo «statistiche sufficienti» $(C,U,Q)$ con somma, fusione e aggiornamento incrementale; test di associatività, commutatività e parità NumPy/Torch |
| 1.2 | Routing come posterior di miscela diagonale con $\log v_{ij}$, varianze regolarizzate $v_{ij}=(C_is_{ij}^2+\kappa\sigma_j^2)/(C_i+\kappa)$; attributi categorici con verosimiglianza categorica |
| 1.3 | Obiettivo $\mathcal F$; passi E/M in forma chiusa dove possibile; test di monotonia: $\mathcal F$ non aumenta mai |
| 1.4 | Masse con estinzione $\max(0,C_i-N_p/2)$; rimozione di $S$, $B(\pi,F)$ e dei trasferimenti per tentativi |
| 1.5 | $r_{ij}$, $\eta_j^2$ e $CP$ come grandezze **riportate**; regole ed heatmap XAI calcolate da $r_{ij}$ |
| 1.6 | Classificazione e regressione come $\mathrm{NLL}(y\mid x)+\lambda\mathcal F/(nm)$; completamento per marginalizzazione esatta |
| 1.7 | Gerarchia: un figlio si crea solo se riduce $\mathcal F$ più del costo dei suoi parametri (stesso criterio MML), al posto di profondità/purezza/minimo righe |
| 1.8 | Nuovo profilo `cp-v9` con contratto proprio; profili precedenti caricabili in sola inferenza |
| 1.9 | Protocollo: dataset UCI **reali**, 10 ripetizioni di validazione incrociata stratificata, stesse partizioni per ACAN e per le baseline già integrate; test bloccato su dataset mai usati nello sviluppo |

### Fase 2 — Strato a concetti derivato (3–4 settimane)

Si procede in due tempi, il primo senza addestramento.

**2a. Misura della raggruppabilità (pochi giorni, decisiva).** Su SmolLM2
denso, per livello e per testa, si estraggono le chiavi di testi reali e si
misura: varianza spiegata delle chiavi con $k\in\{16,32,64,128\}$ slot online;
$\lVert q\rVert\rho$; errore dell'uscita dell'attenzione e variazione della NLL
quando l'attenzione densa è sostituita **a pesi invariati** dalla formula di
C.3, per $L\in\{512,2048,8192\}$. Stesse misure per le alternative banali:
sola finestra, slot casuali, campionamento uniforme delle chiavi.

**2b. Implementazione.**

| # | Intervento |
|---|---|
| 2.1 | Nuovo modulo `acan/lm/slot_attention.py`: per testa, una softmax su finestra e slot, termine $\log C_i$, correzione $\tfrac12q^\top V_iq$ diagonale; nessun ciclo Python sui token (aggiornamenti vettoriali a blocchi) |
| 2.2 | Proprietà di esattezza come test: con $w\ge L$ o $k\ge L$ l'uscita coincide con l'attenzione densa entro $10^{-5}$ |
| 2.3 | Innesto che conserva la funzione: stesse proiezioni, stesse teste, stessa scala; GQA rispettato senza espansione |
| 2.4 | Trattamento di RoPE: confronto fra chiavi non ruotate con posizione media di slot e slot per bande di posizione |
| 2.5 | Politica di slot: assegnazione al più vicino, nascita fino al budget, fusione dei due più vicini; stato $(C,U,Q,\sum v)$ per slot |
| 2.6 | Tutti i livelli, non solo la metà superiore; maschere di padding e batch di lunghezze diverse |
| 2.7 | Eliminazione delle tre copie quasi identiche di `attend` (`attention.py`, `shared_attention.py`, `cp_attention.py`): i profili storici restano come moduli congelati di sola lettura |

### Fase 3 — Addestramento e valutazione (4–6 settimane, richiede GPU)

| # | Intervento |
|---|---|
| 3.1 | Adattamento con LoRA e distillazione dal modello denso (KL sui logit), partendo dall'innesto che conserva la funzione |
| 3.2 | Qualità: perplessità su WikiText-103 e PG-19; test bloccato |
| 3.3 | Contesto lungo: richiamo associativo e copia a lunghezza crescente (misura diretta della Proposizione 5), ago nel pagliaio, un sottoinsieme di un benchmark pubblico di documenti lunghi |
| 3.4 | Costo: token/s e memoria di picco per $L$ da 1.000 a 64.000 |
| 3.5 | Baseline alla stessa scala: denso, sola finestra, attenzione lineare, fusione della cache KV, un modello a stato (Mamba o RWKV) di dimensione paragonabile |
| 3.6 | Risultato: frontiera qualità/stato con intervalli, più la tabella dei casi in cui ACAN perde |

### Fase 4 — Pubblicazione e rilascio (2–3 settimane)

- Un solo paper nuovo: principio (B), Teorema 6 e Proposizione 5, misure delle
  fasi 1–3, limiti. Bibliografia verificata sulle fonti; posizionamento rispetto
  a clustered attention, attenzione lineare, Infini-attention, fusione KV.
- Scheda **Matematica** dell'app riscritta sulla fondazione.
- Versione 9.0.0: pacchetto, pesi e script di riproduzione; ogni tabella del
  paper rigenerabile con un comando.

## 5. Criteri di passaggio

Le soglie sono proposte iniziali: vanno confermate o corrette **prima** di
eseguire la fase corrispondente, non dopo.

| Dopo | Si prosegue se | Altrimenti |
|---|---|---|
| Fase 0 | la tabella di controllo esiste, con intervalli | – |
| Fase 1 | Gioco: 4 concetti senza target; Irrilevanti: ≥ 99% fino a 48 attributi; Letter: prototipi ridotti di almeno 5× con perdita ≤ 1 punto; sui dataset reali nessuna regressione significativa rispetto al profilo attuale e $\mathcal F$ monotona in tutti i test | si corregge il prior (κ, $N_p$) una volta; se non basta, la selezione di $K$ resta un limite dichiarato e si prosegue solo con il lato attenzione |
| Fase 2a | a pesi invariati, con $k=64$, $w=64$ e $L=2048$, l'aumento di NLL è ≤ 0,10 nat e almeno la metà di quello della sola finestra | **decisione di indirizzo**: se gli slot non battono la sola finestra, l'attenzione a concetti non ha base empirica su questo modello; si valuta la variante addestrata da zero su scala ridotta oppure si concentra il progetto sul prodotto tabulare interpretabile |
| Fase 2b | i test di esattezza passano; NLL al passo 0 entro 0,10 nat dal denso; ≥ 1.000 token/s su CPU per il modello da 135 M | si corregge l'implementazione, non le soglie |
| Fase 3 | a parità di stato ACAN supera sola finestra e attenzione lineare su almeno due delle tre famiglie di misure (qualità, contesto lungo, costo) | si pubblica comunque la frontiera come risultato negativo documentato |

## 6. Rischi principali

| Rischio | Segnale | Risposta |
|---|---|---|
| Le chiavi reali non sono raggruppabili con $k\ll L$ | fase 2a negativa | vedi criterio 2a; è il rischio maggiore e costa pochi giorni scoprirlo |
| RoPE rende inservibili i centroidi | errore alto solo nelle teste posizionali | slot per bande di posizione; finestra più ampia nei livelli bassi |
| La selezione MML elimina strutture rare vere | perdita su classi minoritarie | $\kappa$ e $N_p$ per attributo; riportare sempre l'accuratezza bilanciata |
| Risorse di calcolo insufficienti per la fase 3 | – | limitare la fase 3 a 135 M/360 M e dichiararlo |
| Uso ripetuto degli stessi dati | miglioramenti che non reggono al test bloccato | una sola lettura del test per fase |

## 7. Prime tre azioni

1. Fase 0.1–0.3: controllo denso + LoRA, ablazione a sola finestra, validazione
   ampia. Rispondono alla domanda «la memoria a concetti oggi serve?».
2. Fase 2a: misura a pesi invariati della raggruppabilità delle chiavi.
   Risponde alla domanda «può servire?» prima di investire nell'implementazione.
3. Fase 1.1–1.4 in parallelo: il nucleo v9 sul tabulare, dove ogni proposizione
   della fondazione si verifica in minuti.
