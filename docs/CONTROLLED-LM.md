# Laboratorio LLM controllato: cosa fare e cosa misurare

Lo scopo è verificare se ACAN può conservare qualità linguistica e offrire vantaggi
misurabili di memoria o velocità. Avere un training funzionante non dimostra ancora
quel risultato. Su Hostinger CPU si usa SmolLM2-135M; 360M e 1.7B richiedono un
worker CUDA in questa applicazione. Il profilo CPU conservativo rifiuta
prima del caricamento batch maggiori di 1, blocchi oltre 128 token (64 nel
profilo al 100% ACAN), slot/finestra oltre 16 e rank LoRA oltre 8. Questi limiti
riducono il rischio di esaurire la RAM; non sostituiscono la verifica del VPS. Non viene acquistata o attivata alcuna GPU.

Il profilo predefinito dei nuovi progetti è `concept-graph-v1`, lo stesso
modello della scheda tabulare ([MODEL.md](MODEL.md)): nei layer innestati i
token più vecchi del contesto sono letti attraverso concetti, mentre i primi
token e una finestra di quelli recenti restano esatti. I profili descritti in
fondo a questa pagina (`cp-routing-v1`, `cp-joint-v1`) sono storici e restano
solo per riprendere i progetti già avviati.

## Come sapere se funziona

Tre controlli, in quest'ordine:

1. **Testo generato.** Da un checkpoint in pausa, con un inizio di frase nella
   lingua del corpus: deve restare coerente come quello del modello originale.
2. **Controllo denso con lo stesso budget.** È il confronto che decide. Scendere
   sotto il modello di partenza non basta, perché qualunque adattamento al
   corpus lo fa. Se alla fine la NLL di validazione del progetto a concetti è
   entro circa 0,15 da quella del controllo denso, i concetti non costano
   qualità a quella scala; se il controllo è nettamente più basso, la
   differenza è il costo della compressione. Con il campione di validazione
   di 1.024 token, differenze minori di circa 0,15 non si distinguono dal caso.
3. **Test separato.** **Confronta checkpoint e originale** su entrambi i
   progetti, con gli stessi blocchi di test.

Con finestra e concetti piccoli (16+16) e blocchi corti questa prova dice se
il modello impara a leggere il contesto compresso, non se è più veloce: il
vantaggio di costo si vede solo su contesti lunghi, che su CPU non sono
praticabili.

## Percorso dalla GUI

1. Apri **Generativo**, salva la chiave di recupero e scegli un corpus pubblico.
   **WikiText-2** offre articoli inglesi e split ufficiali train/validation/test.
   **Dante** e **Tiny Shakespeare** sono prove letterarie piccole, non materiale
   sufficiente a creare un assistente generalista. Fonte, lingua e licenza sono
   visibili; revisione e checksum vengono verificati prima dell'uso.
2. Scegli una prova di massimo 20 passi. Premi **Scarica e avvia la prova**.
   Il primo avvio prepara i documenti e scarica i pesi originali: i passi di
   ottimizzazione cominciano dopo la preparazione. Il budget non è una promessa
   di durata. La prova guidata usa l'ibrido al 50%, senza teacher.
3. Guarda la validazione confrontata con l'originale. Un valore inferiore è
   migliore sullo stesso campione; non esiste una soglia NLL universale. I token
   elaborati possono includere ripetizioni. Il grafico distingue riferimento
   originale, ultima validazione e miglior checkpoint conservato.
4. Il nuovo protocollo conserva il migliore e mette in pausa dopo tre controlli
   senza un miglioramento di almeno 0,01 NLL rispetto al riferimento significativo.
   Puoi anche premere **Pausa e salva**. Attendi `paused` prima di generare o
   confrontare. **Riprendi** continua con lo stesso budget, corpus e configurazione;
   un progetto `completed` non viene ripreso come se fosse ancora in corso.
5. Seleziona **Migliore conservato**, scrivi un inizio nella lingua del corpus e
   genera. Usa gli stessi prompt quando confronti checkpoint diversi. Il testo
   generato è una prova qualitativa, non un benchmark scientifico da solo.
6. Premi **Confronta checkpoint e originale**. La modalità rapida valuta gli
   stessi quattro blocchi di test e misura prefill, decoding e cache a contesti
   32/128. La modalità CPU lunga usa un solo blocco e contesti 128/1024, con un
   costo superiore. Il profilo ampio è abilitato soltanto con CUDA. Il confronto
   è esplicito, non parte a ogni refresh o a fine training. **Arresta confronto**
   richiede l’annullamento al confine del calcolo corrente, senza cambiare pesi.
7. Se il confronto è favorevole e stabile, aumenta gradualmente dati e budget
   in un nuovo progetto. Non usare ripetutamente lo stesso test per scegliere
   tutti gli iperparametri: in quel caso diventa anch'esso validazione.

## Se la linea rossa sale

Piccole oscillazioni non bastano per diagnosticare un problema. Se la loss di
training scende mentre la validazione peggiora per molti controlli, il quadro è
compatibile con sovra-adattamento: non si deve scegliere automaticamente l'ultimo
checkpoint. La GUI conserva il migliore dei nuovi progetti e spiega l'arresto.
Per vecchi progetti si protegge il migliore tra i file ancora presenti: il minimo
storico di un checkpoint già eliminato non può essere ricostruito dal grafico.

La **Diagnostica del progetto e log** mostra configurazione, gradienti ACAN/LoRA,
learning rate, passaggi equivalenti sul corpus, duplicati rimossi, split, codice
compatibile, memoria cgroup e ultime righe del log. Il JSON è scaricabile dal web.
Non contiene corpus grezzo, chiave workspace o token del servizio. L'accesso resta
limitato al proprio workspace. Non esiste un comando web per eseguire shell o
leggere percorsi arbitrari.

**Avvia controllo denso con stesso budget** crea un progetto separato con SmolLM2
originale + LoRA, stessi dati, split, seed e budget. Se anche questo peggiora,
l'ipotesi dati/ottimizzazione diventa più plausibile; se peggiora soltanto ACAN,
l'innesto e i suoi parametri richiedono maggiore attenzione. È un altro training
esplicito e può essere lento su CPU. Il controllo non altera il progetto ACAN.

## Modalità avanzate dei nuovi progetti

- **Distillazione** aggiunge alla cross-entropy il confronto causale con le
  distribuzioni del modello originale congelato. Il peso predefinito è 0,5 e la
  temperatura è 2 quando l'opzione è selezionata. È una scelta ingegneristica
  esplicita, non un valore numerico imposto dal paper. Richiede più RAM e tempo;
  resta facoltativa su CPU. Patience e miglioramento minimo sono configurabili
  prima del nuovo training nelle impostazioni avanzate; patience 0 disattiva
  la pausa automatica. La loss NLL mostrata rimane cross-entropy; obiettivo
  misto e KL sono riportati separatamente nella diagnostica.
- **50% ACAN** è il percorso ibrido predefinito. **100% ACAN** è sperimentale:
  usa le stesse equazioni causali negli interi livelli e un adattatore di cache
  che mantiene la posizione senza appoggiarsi a una cache densa. I test verificano
  causalità, equivalenza prefisso/streaming e cache residente limitata; non
  certificano qualità linguistica o prestazioni di un modello grande.
- Nessuna delle opzioni promette i risultati del paper. Con metà dei livelli
  ancora densi la loro cache continua a crescere: non è corretto attribuire una
  riduzione del 98% alla cache totale dell'ibrido al 50%. Occorre misurare tutto
  il modello, comprese statistiche e finestra locale ACAN. Per contesti brevi
  l'overhead può rendere la cache più grande o il modello più lento.

## Dati e riproducibilità

I nuovi TXT sono documenti/paragrafi separati da righe vuote; ogni record JSONL
contiene `text` ed eventualmente `split` uguale a `train`, `validation` o `test`.
Senza split espliciti, l'hash del documento assegna circa 80/10/10 ai tre insiemi.
I duplicati esatti dopo normalizzazione degli spazi sono rimossi anche tra split.
Non è un rilevatore di duplicati semantici. Per WikiText si mantengono gli split
ufficiali e si ricompongono gli articoli, senza trasformare ogni intestazione in
un nuovo documento. La valutazione campiona posizioni distribuite negli split;
gli indici sono registrati. I documenti pubblici potrebbero già essere comparsi
nel pre-training del modello originale: non si certifica assenza di contaminazione.

Tokenizzazione, cursori, RNG e ottimizzatore vengono salvati in modo riprendibile.
I moduli del trainer legacy e il suo contratto restano invariati. I nuovi progetti
hanno un'identità aggiuntiva del protocollo controllato e della struttura del
modello; un codice incompatibile non riprende silenziosamente vecchi pesi.

## Scaricare e ispezionare il modello

**Scarica modello ACAN/LoRA e struttura** esporta i pesi addestrabili in safetensors,
la struttura dei livelli, le forme dei tensori, configurazione, riepilogo delle
metriche e revisione esatta della base. Non include corpus, chiavi o optimizer
pickle. È un pacchetto di delta riproducibile, non tutti i pesi originali.

Dopo aver installato questa revisione del progetto con extra `llm` ed estratto
lo ZIP, si può ricostruire il modello:

```python
from acan.lm.exporting import load_export
model, tokenizer = load_export("cartella_estratta", device="cpu")
print(model)
```

Il primo caricamento scarica i pesi originali della revisione registrata e ne
verifica gli hash. `architecture.json` permette l'ispezione senza caricare i pesi.
Il loader non esegue codice remoto del modello e non deserializza optimizer pickle.
Il risultato può essere usato per ulteriori benchmark esterni; la valutazione
limitata della GUI non sostituisce MMLU, GSM8K o un audit linguistico generalista.

## Limiti scientifici e prove

Le formule e il riferimento originale restano congelati; le correzioni causali
approvate sono documentate in [LM-AUDIT.md](LM-AUDIT.md). La CI esegue i controlli
nominati per ripresa esatta, arresto anticipato, migliore conservato, export,
causalità del profilo completo, confronto indipendente e isolamento del debug.
Nessuno può passare saltando PyTorch o usando soltanto i test tabellari.

Le prove reali CPU servono a verificare l'intero percorso, non a dichiarare
superiorità. Nella prova breve da 20 passi l'ibrido non ha ancora recuperato la
qualità della base; nei contesti 32/128 è risultato più lento e non ha risparmiato
cache totale. I report registrano anche questi risultati negativi. Prima di
scalare a circa 1B servono corpus adeguati, qualità stabile e misure su hardware
appropriato. Con la sola CPU Hostinger il percorso realistico è validare il 135M,
non riprodurre un pre-training da centinaia di milioni di token in tempi brevi.

Il [report CPU](CONTROLLED-LM-CPU-SMOKE.json) conserva metriche e protocollo delle
prove, senza corpus o chiavi. Un passo reale con teacher ha misurato circa
2,5–2,6 GiB di RSS del processo. Il Docker annidato dell’ambiente di sviluppo non
applica limiti cgroup: il flag `--memory=4g` non certifica un tetto effettivamente
imposto. Su Hostinger vanno controllati il picco RAM del training e la memoria
del worker, includendo gli altri processi; il consumo di una prova breve non
garantisce il consumo massimo di ogni configurazione o addestramento lungo.

## Guardare dentro i layer dal web

I 30 layer di SmolLM2-135M sono una scelta dell'architettura originale. Sostituire
il 50% superiore con ACAN è un'ipotesi sperimentale, non un risultato che stabilisce
il numero ottimale. I layer non sono i livelli dell'albero ACAN tabellare: non
esistono collegamenti padre–figlio fra i loro slot. Per valutare il graft confronta
originale, controllo denso-LoRA e ibrido sugli stessi split e budget; il profilo
100% ACAN è un ulteriore esperimento, non una garanzia di miglioramento.

1. Nel progetto controllato in esecuzione apri **Dentro il modello · layer, slot e
   gradienti** e premi **Cattura il prossimo passo di training**. La richiesta è
   consumata una sola volta al passo successivo, senza avviare un altro training.
   Se metti in pausa prima che sia acquisita, rimane in attesa della ripresa.
2. La tabella mostra l'ampiezza RMS delle attivazioni in ingresso/uscita, quanto
   cambia il segnale e la norma dei gradienti prima del clipping. Le attivazioni
   riguardano l'ultima posizione del primo elemento del primo microbatch; i
   gradienti comprendono l'accumulo dell'intero passo. Non sono medie dell'intero
   corpus e non provano l'utilità causale di un layer.
3. La mappa mostra la frazione di massa accumulata in ciascuno slot ACAN. Il numero
   di slot effettivi è l'esponenziale dell'entropia delle frazioni: circa 1 indica
   concentrazione, vicino al numero di slot indica distribuzione uniforme. Nessuno
   dei due estremi è automaticamente buono o cattivo. Uno slot in L15 non ha
   necessariamente lo stesso significato dello slot con uguale indice in L16.
4. Per l'inferenza metti in pausa il training, seleziona un checkpoint, scrivi un
   prompt e abilita **Ispeziona i layer durante la generazione**. Scegli al massimo
   32 nuovi token, con prompt massimo 256 token. Premi **Genera dal checkpoint**,
   poi scegli **Ultima generazione ispezionata** nella sezione di ispezione.
5. Il primo campione è il prefill dell'intero prompt; i successivi leggono un nuovo
   token ciascuno usando la memoria precedente. La GUI mostra la lunghezza del
   prefisso, l'ultimo token in ingresso e i cinque candidati successivi con le
   probabilità. Il campionamento può selezionare un candidato diverso dal primo.
   Il token finale emesso non ha un forward aggiuntivo, quindi non viene mostrata
   una sua nuova distribuzione successiva.
6. Scarica il JSON per confrontare catture e conservare risultati. La cattura di
   training più recente resta nel volume; le tracce di inferenza restano nel
   risultato della generazione. Le richieste e i risultati sono isolati per
   workspace. Il report di training non include il testo del corpus.

Valori non finiti richiedono attenzione alla stabilità numerica. Gradienti sempre
nulli su più catture possono indicare pesi scollegati, ma un singolo zero può essere
normale (anche per l'inizializzazione LoRA). Una massa oltre il 95% in uno slot
viene segnalata come concentrazione da indagare: non è una diagnosi automatica di
collasso. Il report JSON contiene anche assegnazioni dell'ultimo token, rilevanza
media, slot attivi e numero di parametri con gradiente per layer.

La cattura non modifica equazioni, output o gradienti e rimuove gli hook anche in
caso di errore. Introduce però sincronizzazioni e lavoro aggiuntivo: **i tempi di
una generazione ispezionata non sono benchmark**. Non salva matrici di attenzione
complete, non etichetta automaticamente i concetti e non applica una loss CP
aggiuntiva. Il trainer legacy resta invariato: su quei progetti è disponibile
l'ispezione in generazione, non la cattura del prossimo passo di training.

### Storico: solver CP precedente (`cp-routing-v1`, PR8)

Nei nuovi progetti la GUI permette `cp-routing-v1`: la stessa energia e lo stesso
solver di `Model → Routing ACAN → CP diretta`, con backend Torch differenziabile
nell'LLM. Su CPU il budget è 32 token per blocco, 8 slot e tre passi per token;
la GUI aggiorna blocco e slot alla selezione, il worker valida gli stessi limiti.
Si tratta di un profilo sperimentale più costoso del routing precedente.

Le statistiche sono causali e la cache contiene anche le somme nello spazio
sigmoid usato dal routing. Le assegnazioni passate rimangono fisse. Sono verificati
forward completo contro streaming, prefisso invariato al cambio dei token futuri,
gradienti finiti e un passo di training nei graft al 50% e 100% su un Llama piccolo
locale. Queste verifiche non misurano qualità linguistica o memoria su un modello
135M in produzione. I progetti precedenti mantengono il routing originale e possono
riprendere soltanto con i relativi contratti verificati; nessuna conversione dei pesi.

La scheda **Matematica** dell'app spiega formule, coefficienti, limiti e differenza
fra CP e NLL. La diagnosi Letter (`LETTER-DIAGNOSIS.md`) mostra che il nuovo routing
non risolve automaticamente la classificazione reale.

Il gate LM richiede almeno 40 test e controlli nominati, inclusa la ripresa CP da checkpoint con pesi
bit per bit uguali al training non interrotto sul modello locale di prova.


### Storico: modello congiunto (`cp-joint-v1`)

Il nuovo profilo usa `concept_step` e `concept_assignments` di `acan/algebra.py`,
gli stessi operatori del modello tabulare supervisionato. Rilevanza root-shift
limitata, routing a distanza, conteggi/somme, centroide e CP normalizzata sono
un unico nucleo matematico; il precedente solver a tre passi resta disponibile
con il suo identificatore originale. Il profilo congiunto esegue un passo per token.

Il trainer aggiunge esplicitamente alla NLL next-token
`cp_weight × media(layer ACAN, batch, posizioni predittive)(1 − CP/√d)`.
La CP viene calcolata nello stesso forward che produce la NLL e mantiene il grafo
PyTorch; non è una metrica staccata. La rilevanza viene limitata a [0,1] anche
quando un prototipo esce dall'intervallo di un prefisso corto. L'ultima posizione
del blocco è esclusa perché non possiede un target next-token nel blocco.
Con teacher KL abilitata si aggiunge la CP alla combinazione NLL/KL esistente.
Validazione, early stopping e conservazione del migliore usano ancora la NLL.

La GUI propone `cp_weight=0.1`, configurabile fino a 1 e azzerabile per il controllo
con sola NLL. L'API mantiene 0 come default per i chiamanti precedenti. Il peso,
il profilo, le metriche separate e l'obiettivo totale sono registrati nella
configurazione/checkpoint ed esportati; non è possibile cambiare la loss di un
progetto già avviato mediante la semplice ripresa. Il profilo resta vincolato a
32 token e 8 slot su CPU. Scegliere il nuovo nucleo richiede un nuovo progetto.

I controlli obbligatori includono parità dei gradienti analitici NumPy con autograd,
gradienti CP non nulli, causalità anche della CP, equivalenza streaming, entrambi
i graft e ripresa esatta con il termine CP e gradient checkpointing attivi.
La prova con λ=0/0,1 usa pesi iniziali e blocchi identici per ogni coppia;
i risultati non dimostrano qualità linguistica generale. Vedi [JOINT-CP.md](JOINT-CP.md).
