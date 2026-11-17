# Il modello a concetti che scrive (`concept-lm-v1`)

Questa pagina descrive il **contatore di contesti esatti attualmente nel prodotto**.
Condivide somme e lettura categorica con il modello tabulare, ma **non usa la
ricerca per somiglianza di ConceptModel**. Non è quindi l'unificazione richiesta
fra tabella e testo vettoriale. Non impiega reti o gradienti.

Il primo prototipo che usa effettivamente `ConceptModel.fit_chunks` sul testo è
separato dal prodotto: [esperimento su200000 righe](../experiments/text_table/README.md).
Trova83 concetti su tre livelli, ma misura6,9207 nat/token contro5,4075 del
contatore con200008 token e4,5199 del contatore sul train completo, sugli stessi
blocchi di questa prova. Non viene integrato: vanno affrontati il bilanciamento
del target vettoriale e il giudizio sul token esatto. Memoria del documento,
affinamento dei vettori e layer successivi restano da verificare.

Codice del prodotto: `acan/lm/concept_lm.py` e `acan/lm/concept_trainer.py`;
la coda, i checkpoint, l'API e i progetti esistenti conservano il loro contratto.

## 1. Il testo come tabella

| Tabella | Testo |
|---|---|
| una riga | una posizione $t$ del testo |
| le colonne note | i token precedenti $x_{t-1}, x_{t-2}, \dots$ |
| la colonna da prevedere | il token $x_t$ |

## 2. Concetti per contesto

Un concetto di livello $j$ raccoglie le posizioni con gli stessi ultimi $j$
token (il loro contesto $c$). Livello 0: tutte le posizioni. Somme:

- $C_c$: posizioni con contesto $c$;
- $N_c(y)$: quante di queste sono seguite da $y$;
- $D_c$: quanti token diversi seguono $c$.

Il livello sopra di $c$ è il suo contesto più corto (senza il token più lontano):
i livelli sono annidati e le somme di un concetto sono contenute in quelle del
suo livello sopra.

## 3. Lettura

$$
p_j(y)=\frac{N_{c_j}(y)+\kappa_jD_{c_j}\,p_{j-1}(y)}{C_{c_j}+\kappa_jD_{c_j}},\qquad p_{-1}=1/V .
$$

È **la stessa funzione** con cui il modello tabulare legge una categoria
(`acan/concept_algebra.py`, `categorical_reading`), scelta allo stesso modo: il
peso del livello sopra è $\kappa$ volte i valori diversi del concetto, con
$\kappa$ dal giudizio. Sulla tabella questa regola tiene tutte le soglie dei test
di qualità (credit-g da 0,706 a 0,731, breast cancer da 0,952 a 0,947, Titanic
da 0,812 a 0,807, gli altri uguali o meglio); sul testo, con $\kappa=1$ fisso
come prima nella tabella, l'errore sarebbe 4,445 bit per carattere invece di
2,039. Un contesto
mai visto lascia la lettura com'è.

**Concetti del testo in corso.** Gli stessi concetti contati solo sulle posizioni
precedenti del testo che si sta leggendo, letti dopo i globali con la lettura
globale come riferimento e pesi $b_j$. Permettono di riprendere ciò che il testo
ha già detto, a qualunque distanza; nessuna posizione legge sé stessa. Il livello
0 sono tutte le posizioni lette finora: alza i token già comparsi nel testo (una
memoria dell'argomento). Senza di lui una domanda in chat era letta solo dai suoi
ultimi token, quelli del formato della conversazione, uguali per ogni domanda, e
domande diverse ricevevano la stessa risposta; con lui la risposta segue
l'argomento, e su wikitext-2 l'errore scende da 4,589 a 4,533.

**Concetti dei token.** Due token diversi per la lettura sopra sono solo diversi.
Il modello tratta i token come righe di una tabella e ne cerca i concetti con il
modello tabulare ([`acan/core/concepts.py`](../acan/core/concepts.py)): la riga
del token $v$ sono le radici delle frequenze con cui ciascuno dei 256 token più
frequenti lo precede e lo segue, ridotte alle direzioni principali (SVD, 16 e 48
dimensioni); la tabella ha gli 8.000 token più frequenti dei primi 20 milioni,
gli altri token visti almeno 3 volte sono assegnati dal modello al concetto più
probabile. Ne escono livelli dal generale al particolare (su wikitext-2:
$2\to9\to20\to33\to325$ e $4\to8\to19\to40\to208$).

**Percorsi.** Un percorso legge il token attraverso i livelli dei concetti:

$$
p(y)=p(g(y)\mid c)\;p(f(y)\mid g(y),c)\;p(y\mid f(y),c),
$$

ogni fattore con la lettura sopra e le sue somme: le chiavi comprendono il nodo
già scelto (il livello 0 di un nodo è uniforme sui suoi figli), e nel contesto il
token a ogni distanza è preceduto dal suo concetto particolare. Il modello
mescola il percorso diretto e uno per gerarchia, $p=\sum_kw_kp_k$, con i pesi
dal giudizio (attesa e massimizzazione sugli stessi pezzi che scelgono i
$\kappa$). I percorsi attraverso i concetti leggono contesti fino a 2 token, il
diretto fino a 6: più percorsi o contesti più lunghi leggono un po' meglio con
memoria libera e peggio dentro il limite, perché tutte le somme lo condividono.

I percorsi sono una scelta del progetto («Concetti dei token e percorsi», spenta
se non la si accende): aiutano sui testi piccoli e peggiorano su quelli grandi
quando la memoria è limitata (§7).

## 4. Giudizio

$\kappa_j$ e $b_j$ si scelgono fra $2^{-5},\dots,2^{8}$, un peso alla volta, due
giri, massimizzando la log-verosimiglianza di metà dei pezzi di validazione;
l'altra metà misura l'errore. Il testo di validazione non viene mai contato.

## 5. Algoritmi

**Chiavi dei contesti.** $k_j=k_{j-1}(V+1)+x_{t-j}+1$ in 64 bit, per tutte le
posizioni insieme (operazioni su vettori). Esatte finché $(V+1)^j<2^{64}$
(quattro token con $V=32.000$); oltre, il numero si avvolge come un hash e due
contesti possono condividere la chiave con probabilità circa $n^2/2^{64}$.

**Conteggio di un pezzo.** Per ogni livello: `numpy.unique` sulle chiavi (ordine
e conteggi) e sulle coppie (concetto, token). Ne esce una **tabella**: chiavi
ordinate, $C$, celle (indice del concetto × $(V+1)$ + token) ordinate con i loro
conteggi; $D$ si ricava dalle celle.

**Somma di due tabelle.** Unione delle chiavi, somma di $C$ per chiave, unione
delle celle (concetto, token) e somma dei conteggi. Costo lineare nelle
dimensioni delle due tabelle (più l'ordinamento).

**Contatore in base due.** Ogni pezzo diventa una tabella; due tabelle che
coprono lo stesso numero di pezzi si sommano in una che ne copre il doppio (come
i blocchi del modello tabulare). Ogni posizione è sommata circa
$\log_2(\text{pezzi})$ volte. Prima di leggere o salvare, il contatore si somma
tutto.

**Memoria.** Il limite vale per tutte le somme, quelle unite e quelle del
contatore (automatico: un quarto della memoria che il processo può usare, cioè
della macchina o del limite del suo contenitore, al massimo 2 GB). Dopo ogni
pezzo, se lo supera, il contatore si unisce tutto e si generalizzano nel livello
sopra i concetti con la minore perdita per byte, finché le somme scendono al 60%
del limite. Le somme si uniscono un livello alla volta come unione di elenchi
ordinati, e la soglia si trova per dimezzamento: il processo resta sotto circa
tre volte il limite. Perdita di
un concetto:

$$
\Delta_c=\sum_y N_c(y)\log\frac{p^-_c(y)}{p_{\bar c}(y)},\qquad
p^-_c(y)=\frac{N_c(y)-1+\kappa D_c\,p_{\bar c}(y)}{C_c-1+\kappa D_c},\qquad
p_{\bar c}(y)=\frac{N_{\bar c}(y)+1}{C_{\bar c}+V} :
$$

ogni membro letto senza sé stesso, come nel giudizio; un contesto visto una volta
costa zero (prima sembrava il più utile da tenere: su wikitext-2 con 400 MB di
limite l'errore del solo percorso diretto passa da 4,615 a 4,602 con 49 MB di
somme invece di 252).

Dove la chiave è esatta il livello sopra si ritrova dalla chiave stessa
($k_{j-1}=\lfloor k_j/(V+1)\rfloor$); oltre, si usa il livello sopra come
distribuzione di tutto il livello. Il livello 0 non si generalizza mai.

**Lettura di molte posizioni.** Per ogni livello: ricerca binaria della chiave
nell'elenco ordinato, poi della cella (concetto, token). **Testo in corso**:
ordinamento stabile per (contesto, posizione) e conteggio delle posizioni
precedenti dello stesso gruppo, senza cicli sulle posizioni.

**Scrittura di un token.** Per ogni livello si trova il concetto del contesto
attuale e si leggono le sue celle (un intervallo contiguo dell'elenco ordinato);
la distribuzione su tutto il vocabolario si aggiorna in $O(V)$ per livello.

**Checkpoint.** Le tabelle di tutti i livelli e i pesi in `trainables.safetensors`,
lo stato in `state.json`, un manifesto con le impronte SHA-256: lo stesso formato
di ogni progetto, quindi pausa, ripresa dopo un riavvio, migliore, permanenti,
coda (il testo successivo somma le sue tabelle a quelle del precedente) ed
esportazione funzionano come sempre.

**API compatibile OpenAI** ([`acan/lm/openai_api.py`](../acan/lm/openai_api.py)).
Il sito risponde a `GET /v1/models`, `POST /v1/chat/completions` e
`POST /v1/completions` come l'API di OpenAI, quindi la libreria `openai`,
LangChain, Open WebUI o curl funzionano cambiando indirizzo e chiave. Le chiavi
(`acan-…`) si creano e si revocano nella scheda Generativo, sezione «API
compatibile OpenAI»; ognuna vede solo i progetti del suo spazio di lavoro, e si
conserva solo la sua impronta SHA-256. Un modello è l'id di un progetto a concetti
(oppure `<id>:<checkpoint>`): risponde con l'ultimo salvataggio letto dal disco,
anche mentre il progetto continua a imparare, senza GPU e senza un processo per
richiesta (gli ultimi due modelli letti restano in memoria). Parametri:
`temperature` (0 = il token più probabile), `top_p`, `max_tokens` (fino a 4.096),
`stop` (fino a 4 stringhe: la risposta finisce prima), `seed`, `stream` (eventi
`data: …` fino a `data: [DONE]`, con `stream_options.include_usage`). La chat è
scritta nel formato dei dati di conversazione (`<|im_start|>ruolo … <|im_end|>`) e
la risposta finisce al fine turno. Il sito passa le richieste al servizio con la
chiave del cliente: il token interno del servizio non esce mai.

## 6. Garanzie verificate dai test

- La lettura è esattamente la formula del §3 calcolata posizione per posizione.
- I concetti del testo in corso usano solo le posizioni precedenti.
- Pezzi sommati in qualunque ordine danno le somme del testo intero (meno i
  contesti che attraversano il confine fra due pezzi).
- Con il limite di memoria i concetti che non aggiungono nulla al livello sopra
  sono i primi a essere generalizzati e il livello 0 resta intero.
- Salvare e ricaricare non cambia la lettura né la scrittura.
- Un apprendimento lungo vero da capo a fondo: tokenizer, pausa, ripresa, coda,
  risposte di prova, scrittura, errore sul test, esportazione.
- L'API: chiavi per spazio di lavoro e revocabili, la risposta uguale alla scrittura
  del modello, lo streaming uguale alla risposta intera, le stringhe di stop, gli
  errori nel formato di OpenAI, la libreria `openai` ufficiale contro un server vero
  e il passaggio attraverso il sito.

## 7. Misure

**Caratteri** (wikitext, protocollo della prova in piccolo: pezzi di 256,
errore sulle seconde metà):

| | Rete con attenzione | Rete solo finestra | Modello a concetti |
|---|---:|---:|---:|
| bit per carattere | 2,518 | 2,451 | 2,039 |
| memoria a 49 token | 100% | 5,5% | 100% |
| memoria fino a 177 token | 100% | — | 100% |

Con la memoria dei concetti limitata a 18 MB (79% dei concetti generalizzati):
2,079 bit per carattere.

**Token** (stesso testo, tokenizer BPE da 8.000, 272.000 token per imparare,
64 pezzi da 256 mai visti):

| | Rete micro nata da zero | Modello a concetti |
|---|---:|---:|
| nat per token | 5,652 al meglio, poi 5,921 | 5,452 |
| tempo | 433 s al suo meglio | 20 s |

**Concetti dei token e percorsi** (wikitext-2, BPE da 8.000, gli stessi 64 pezzi
da 512 token mai visti; wikitext-103 ha lo stesso test):

| | testo per imparare | memoria | errore (nat per token) | tempo |
|---|---:|---:|---:|---:|
| Rete Llama a 4 strati, 6,2 milioni di parametri | wikitext-2 | — | 4,403 | 90 min |
| Concetti, solo percorso diretto | wikitext-2 | libera | 4,589 | 10 s |
| Concetti con 5 percorsi, contesti fino a 6 | wikitext-2 | libera (10 GB) | 4,419 | 4 min |
| Concetti con 3 percorsi, contesti fino a 2 | wikitext-2 | libera (0,9 GB) | 4,451 | 1 min |
| Concetti, solo percorso diretto | wikitext-2 | 400 MB | 4,602 | 10 s |
| Concetti con 3 percorsi, contesti fino a 2 | wikitext-2 | 400 MB | 4,492 | 30 s |
| Concetti, solo percorso diretto | 20 M token di wikitext-103 | 512 MB | 4,156 | 90 s |
| Concetti con 3 percorsi | 20 M token di wikitext-103 | 512 MB | 4,223 | 4 min |
| Rete Llama a 4 strati (legge 16 M token in 90 min) | wikitext-103 | — | 4,011 | 90 min |
| Concetti, solo percorso diretto | 69 M token di wikitext-103 | 2 GB | 3,905 | 7 min |
| Concetti con 3 percorsi | 69 M token di wikitext-103 | 2 GB | 3,958 | 19 min |

Sul testo piccolo i percorsi portano il modello quasi al livello della rete (nei
primi 32 token e oltre i 256 lo superano); sui testi grandi con la memoria
limitata il percorso diretto da solo legge meglio.

Su wikitext-103, a parità di processore, il modello a concetti con il solo
percorso diretto batte la rete a ogni distanza (0–32 token: 4,195 contro 4,521;
256–512: 3,834 contro 3,964) in 7 minuti contro 90. La rete migliorava ancora
(circa 0,05 ogni 10 minuti alla fine) e aveva letto un quarto del testo: con molto
più calcolo è probabile che recuperi.

**Velocità** (un thread, vocabolario 32.000, testo artificiale con frequenze di
una lingua): circa 414.000 token al secondo per imparare, circa 790 per
scrivere. Dopo 8 milioni di token: 21,5 milioni di concetti, 920 MB di somme,
3 GB di picco del processo durante le somme.

## 8. Algebra comune, costruzione dei concetti ancora diversa

Le somme e la lettura condividono funzioni in `acan/concept_algebra.py`.
Questo non rende identici ricerca, giudizio e costruzione dei concetti: il testo
attuale conta chiavi, mentre ConceptModel generalizza righe per somiglianza.
Cambia come una riga trova i suoi concetti, perché cambiano gli attributi:

- in una tabella ci sono numeri: due righe sono più o meno vicine, una riga
  appartiene un po' a più concetti (massa × somiglianza) e i concetti si cercano
  unendo quelli vicini;
- nel testo ci sono solo token: due contesti o sono uguali o no, quindi una
  posizione appartiene a esattamente un concetto per livello, quello del suo
  contesto, e i concetti si trovano contando.

Provato: la ricerca della tabella usata sul testo dà 3,47 bit per carattere
(troppi pochi concetti e mescolati); il conteggio dei contesti dà 2,04. Il
conteggio sulla tabella richiederebbe di rendere uguali numeri vicini, cioè di
cambiare il modello tabulare: non è stato fatto.

## 9. Limiti

- **Valori simili.** I concetti dei token sono fissati all'inizio e ogni token
  ne ha uno per livello; una rete dà a ogni token un vettore che continua a
  cambiare. Nella tabella due valori diversi di una categoria restano diversi: i
  concetti di valori servirebbero solo a categorie con molti valori, e lì non
  sono stati misurati.
- **Scrittura.** Con i percorsi un token costa la lettura di tutti i nodi: circa
  70 token al secondo invece di circa 790.
- **Memoria.** I contesti crescono con il testo; oltre il limite si generalizzano
  i meno utili e qualcosa si perde.
- **Ripetizioni.** Scrivendo il token più probabile il modello ripete la frase
  più frequente: le domande di prova usano temperatura 0,7 con un seme fisso.
- **Chiavi oltre quattro token** con 32.000 token di vocabolario: hash a 64 bit.

## 10. Compatibilità

È l'architettura `concepts` con il profilo `concept-lm-v1`. I progetti creati
prima (reti con o senza concetti) continuano con il loro modello e i loro
checkpoint. La pagina crea ora solo code del modello a concetti; il laboratorio
delle reti resta come storico.
