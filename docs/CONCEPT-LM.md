# Il modello a concetti che scrive (`concept-lm-v1`)

Un solo modello per tabelle e testo: il testo è una tabella, il modello è quello
di [MODEL.md](MODEL.md). Nessuna rete neurale, nessun peso addestrato, nessun
gradiente. Codice: [`acan/lm/concept_lm.py`](../acan/lm/concept_lm.py) (il
modello), [`acan/lm/concept_trainer.py`](../acan/lm/concept_trainer.py)
(apprendimento lungo, generazione, verifica sul test, esportazione). Test:
[`tests/test_concept_lm.py`](../tests/test_concept_lm.py).

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
p_j(y)=\frac{N_{c_j}(y)+a_jD_{c_j}\,p_{j-1}(y)}{C_{c_j}+a_jD_{c_j}},\qquad p_{-1}=1/V .
$$

È la regola del modello tabulare per un concetto con pochi membri (avvicinato al
livello sopra), applicata alla distribuzione del token che segue. Un contesto
mai visto lascia la lettura com'è.

**Concetti del testo in corso.** Gli stessi concetti contati solo sulle posizioni
precedenti del testo che si sta leggendo, letti dopo i globali con la lettura
globale come riferimento e pesi $b_j$. Permettono di riprendere ciò che il testo
ha già detto, a qualunque distanza; nessuna posizione legge sé stessa.

## 4. Giudizio

$a_j$ e $b_j$ si scelgono fra $2^{-5},\dots,2^{8}$, un peso alla volta, due
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

**Memoria.** Oltre il limite (automatico: un decimo della memoria della
macchina, al massimo 2 GB) si generalizzano nel livello sopra i concetti con la
minore perdita per byte, finché le somme scendono al 90% del limite. Perdita di
un concetto:

$$
\Delta_c=\sum_y N_c(y)\log\frac{N_c(y)/C_c}{p_{\bar c}(y)},\qquad
p_{\bar c}(y)=\frac{N_{\bar c}(y)+1}{C_{\bar c}+V} .
$$

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

**Velocità** (un thread, vocabolario 32.000, testo artificiale con frequenze di
una lingua): circa 414.000 token al secondo per imparare, circa 790 per
scrivere. Dopo 8 milioni di token: 21,5 milioni di concetti, 920 MB di somme,
3 GB di picco del processo durante le somme.

## 8. Limiti

- **Contesti simili.** Due contesti diversi non si aiutano, se non attraverso il
  livello sopra comune. Una rete neurale impara che «gatto» e «cane» si
  somigliano; questo modello no. Con molti dati la rete può tornare in vantaggio:
  non è stato misurato su testi grandi. Il prossimo passo è unire i token (e i
  contesti) seguiti da token simili, con lo stesso costo per chi legge.
- **Memoria.** I contesti crescono con il testo; oltre il limite si generalizzano
  i meno utili e qualcosa si perde.
- **Ripetizioni.** Scrivendo il token più probabile il modello ripete la frase
  più frequente: le domande di prova usano temperatura 0,7 con un seme fisso.
- **Chiavi oltre quattro token** con 32.000 token di vocabolario: hash a 64 bit.

## 9. Compatibilità

È l'architettura `concepts` con il profilo `concept-lm-v1`. I progetti creati
prima (reti con o senza concetti) continuano con il loro modello e i loro
checkpoint. La pagina crea ora solo code del modello a concetti; il laboratorio
delle reti resta come storico.
