# Un routing comune, derivato da un obiettivo esplicito

Questa è una proposta matematica verificata su piccoli problemi sintetici,
non un nuovo profilo attivato nell'app. Il routing può usare lo stesso operatore
per una riga tabulare o un vettore di token. La causalità limita le statistiche
accessibili all'LLM; non richiede una seconda legge di assegnazione.

## 1. Cosa si ricava già dalla formula del paper

Usiamo T_it ≥ 0, somma_i T_it = 1; i indica uno slot, t un'osservazione.
D_it è la distanza quadratica pesata da R, b_it l'eventuale accoppiamento
previsto dall'adattatore. Poniamo s_it = −β D_it + b_it. Con G, R e una massa
π_i **fissati durante il passo**, l'equazione del paper è la soluzione unica di:

```math
\max_{T\in\Delta^n}\quad
\frac1n\sum_{it}T_{it}\,[s_{it}+\gamma\log(\pi_i+\epsilon)]
+\frac{\tau}{n}\sum_t H(T_{:t}),\qquad \tau>0.
```

```math
T^*_{it}=\operatorname{softmax}_i
\left(\frac{s_{it}+\gamma\log(\pi_i+\epsilon)}{\tau}\right).
```

È una derivazione, non un'altra euristica: il divario fra il valore ottimo e
quello di qualsiasi T è esattamente τ volte la KL media KL(T_:t || T*_:t).
Con τ=1 si recupera la softmax implementata. Il prior normalizzato è
q_i ∝ (π_i+ε)^(γ/τ): la stessa funzione si può scrivere come compatibilità
meno τ KL(T || q), a meno di una costante. Non serve bilanciare artificialmente
tutti gli slot a 1/k.

Il codice usa **frazioni** di massa. Scrivere conteggi C anziché C/n è equivalente
solo senza ε o scalando anche ε di n. Con ε fisso, l'equivalenza non è esatta.

## 2. Quando la massa dipende dalle assegnazioni

Se f_i(T)=somma_t T_it/n, il prior dipende da ciò che stiamo cercando. Non si
può dichiarare che una singola softmax abbia già risolto l'intero problema.
La seguente energia ha precisamente il termine logaritmico desiderato nella
condizione di stazionarietà:

```math
\Psi_\epsilon(f)=\sum_i [(f_i+\epsilon)\log(f_i+\epsilon)-(f_i+\epsilon)],
\qquad \frac{\partial\Psi_\epsilon}{\partial f_i}=\log(f_i+\epsilon).
```

```math
J(T)=\mu\,CP(T)
+\frac1n\sum_{it}T_{it}s_{it}
+\frac{\tau}{n}\sum_t H(T_{:t})
+\gamma\Psi_\epsilon(f(T)).
```

La scelta di includere CP e dei coefficienti μ,τ è una **proposta esplicita**,
non una formula completa già specificata dai PDF. Nel probe CP è proprio quella
del codice tabulare corrente: F=f, G_post=TX/C, G_eff=(G_post+G_param)/2, rilevanza
root-shift pesata da omega e funzione `_compute_cluster_probability`. Questo
verifica come usare la CP esistente; non certifica la rilevanza duale del paper.
G_param, omega, statistiche root e s restano fissi durante l'ottimizzazione di T.

Per un punto stazionario interno e differenziabile:

```math
T_{it}=\operatorname{softmax}_i\left(
\frac{s_{it}+\gamma\log(f_i+\epsilon)
+\mu n\,\partial CP/\partial T_{it}}{\tau}\right).
```

Qui la CP interviene **direttamente nel routing**: il suo gradiente misura
quanto cambia la qualità concettuale modificando ogni assegnazione, includendo
l'effetto su masse, centroidi e rilevanza. La sola CP scalare non dice dove
instradare un elemento; la sua variazione rispetto a T fornisce quell'informazione.
L'equazione è implicita e generalmente non convessa, non una soluzione chiusa.

## 3. Un passo controllato, senza promettere un ottimo globale

Usiamo un passo di salita con penalità KL sulla variazione delle assegnazioni:

```math
T^+_{it}\propto T_{it}\exp\left(\eta n\,
\frac{\partial J}{\partial T_{it}}\right).
```

È la soluzione del sottoproblema **linearizzato**, con penalità
(ηn)^−1 somma_t KL(T^+_:t || T_:t). Ricalcoliamo quindi l'energia reale,
compresa la CP, e dimezziamo η finché il passo soddisfa una condizione di
incremento di Armijo. Se il budget di tentativi termina, T resta invariato e
il passo risulta rifiutato. Il probe usa float64, softmax stabile e un pavimento
numerico 10^−14 per le assegnazioni; rivaluta l'obiettivo anche dopo il pavimento.

Questo controlla il valore dell'obiettivo su un problema a dati/statistiche
fissi. Non dimostra ottimalità globale, miglioramento della validazione,
assenza di sovra-adattamento o monotonia lungo un training a minibatch.
Con μ=1, β=γ=τ=0 il probe testa anche la sola CP. Con l'energia congiunta può
salire J senza che ogni singolo termine aumenti. I punti non differenziabili di
abs/clamp e la saturazione possono arrestare il metodo: il rifiuto è esplicito.

## 4. Instabilità della massa: un risultato verificabile

Due slot coincidenti, stessi punteggi, τ=1 e massa p,1−p producono la dinamica:

```math
p^+=\frac{(p+\epsilon)^\gamma}
{(p+\epsilon)^\gamma+(1-p+\epsilon)^\gamma},\qquad
\left.\frac{dp^+}{dp}\right|_{p=1/2}
=\frac{\gamma}{1+2\epsilon}.
```

Con ε=10^−4 e γ=1,15 il guadagno è circa **1,14977 > 1**: una perturbazione
cresce. Con γ=0,5 si attenua. Al confine γ=1, ε>0 dà un guadagno leggermente
inferiore a 1; con ε=0 è neutro. Quindi l'affermazione di instabilità per ogni
γ≥1 nel Teorema 3 richiede questa precisazione sul caso limite. La deriva per
100 iterazioni nel report è un esperimento della dinamica, **non** una descrizione
del forward attuale, che usa due stadi e non quel ciclo di 100 passi.

Per μ=0, punteggi fissi, 0≤γ<τ, l'energia è strettamente concava nell'interno
del simplesso. Infatti, per una perturbazione U, Cauchy–Schwarz dà:

```math
\sum_i\frac{(\frac1n\sum_t U_{it})^2}{f_i+\epsilon}
\leq\frac1n\sum_{it}\frac{U_{it}^2}{T_{it}}.
```

La curvatura negativa dell'entropia domina quella positiva della massa. Questo
fornisce un regime matematicamente controllabile **per quel sottoproblema**.
Aggiungere CP, aggiornare R/G o il pruning può distruggere la concavità; γ<τ
non è una garanzia universale contro il collasso del modello completo.
Ridurre γ modifica anche la pressione a fondere slot, voluta dal paper: serve
misurare entrambi gli effetti, non scegliere un coefficiente guardando il test.

## 5. Un nucleo, due insiemi di informazioni consentite

Nel tabulare si apprendono le statistiche sui dati di training. In valutazione
la riga può usare soltanto gli attributi osservati; target e altre righe del
test non possono aggiornare i concetti. Nell'LLM il token t può accedere soltanto
al prefisso ≤t; masse/centroidi/varianze non devono usare token futuri. Questo
vincolo vale anche per il calcolo del gradiente di CP usato nel forward.

Lo stesso nucleo può quindi ricevere `(osservazione, maschera, stato dei concetti)`.
Le regole di aggiornamento dello stato dipendono dall'uso batch o causale. In
modalità streaming, cambiare retroattivamente assegnazioni già accumulate
renderebbe incoerente la cache: serve fissarle o ricostruire coerentemente lo stato.
L'equivalenza fra forward sull'intera sequenza e streaming è un requisito.

Il probe ottimizza liberamente T su un batch: **non implementa** questa inferenza
induttiva/causale e non va collegato all'app come se l'avesse già risolta.
CP può contribuire all'apprendimento dei concetti in entrambi gli adattatori.
La previsione linguistica richiede inoltre un obiettivo che colleghi il contesto
al token successivo: una CP elevata non certifica la probabilità della parola
corretta. È possibile condividere il routing senza equiparare CP e NLL.

## 6. Evidenza e prossimo criterio di promozione

Esecuzione esplicita su CPU, senza dati utente né rete:

```bash
python -m unittest discover -s tests -p test_variational_routing.py -v
python scripts/probe_variational_routing.py --output probe.json
```

Quattro test: identità KL dell'ottimo a prior fisso; gradiente dell'energia con
differenze finite sul simplesso; CP identica a quella esistente e permutazione
degli slot; 40 ottimizzazioni sintetiche (20 seed × CP pura/energia congiunta)
di 30 passi con controllo dell'obiettivo effettivo e stabilità di massa analitica
contro differenze finite. Risultati in `VARIATIONAL-ROUTING-PROBE.json`.
I coefficienti del probe sono dichiarati e non sono default proposti per l'app.

Prima della promozione servono un router induttivo e uno streaming coerenti con
questa energia, un protocollo esplicito per R/G/pruning, test causali/cache e
checkpoint versionati; poi confronti a pari budget su split di training/validazione
e un test finale separato. Misurare CP, occupazione, stabilità, balanced accuracy
tabulare, NLL linguistica e costo CPU. Le precedenti misure in
`ROUTING-COMPARISON.json` riguardano i due profili precedenti, non questa proposta.

L'analisi consente di eliminare una distinzione arbitraria fra i **principi** del
routing; non dimostra ancora un algoritmo completo superiore sui dataset caricati.
