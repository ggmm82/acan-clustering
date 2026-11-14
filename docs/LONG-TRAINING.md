# Apprendimento lungo dell'LLM a concetti

In cima alla scheda **Generativo**. Il modello è **nativo**: nasce da zero
(`acan/lm/native.py`), con pesi casuali dal seme del progetto, tutti i pesi
che imparano, nessun modello preaddestrato e nessun adattatore. È un decoder
(embedding, blocchi di attenzione e feed-forward, uscita legata agli
embedding) in cui l'attenzione di **ogni** blocco è il layer a concetti.

| Dimensione | Layer | Dimensioni | Teste | Parametri (vocabolario 32.000) | Dove |
|---|---:|---:|---:|---:|---|
| micro | 4 | 256 | 4 | circa 12 milioni | CPU |
| piccolo | 8 | 512 | 8 | circa 50 milioni | CPU lenta o GPU |
| base | 12 | 768 | 12 | circa 138 milioni | GPU |

Sulla CPU: pezzi di 256 token, 16 token esatti e 16 concetti, 8 pezzi per
correzione. Con una GPU: 1.024 token, 64 + 64, 32 pezzi per correzione.

**Tokenizer.** Nasce dal primo testo della coda: BPE sui byte, 32.000 token,
con `<|endoftext|>`, `<|im_start|>` e `<|im_end|>`. I testi successivi della coda
usano lo stesso tokenizer. Imparato su italiano e inglese insieme, non fa
pagare all'italiano i token in più di un vocabolario inglese.

**Misura.** Una prova sulla CPU del server di sviluppo: il modello micro impara
a circa 430 token al secondo (contro circa 25 di SmolLM2 adattato con LoRA);
l'esportazione contiene tutti i pesi e il tokenizer e si ricarica senza altri
file. Su un testo di prova con domande del tipo «quanto fa 7 più uno?», in 600
passi (12 minuti) il modello nato da zero ha imparato il formato della risposta
(«Fa 6. Posso aiutarti ancora?») ma non ancora il calcolo: le risposte di prova
mostrano esattamente questo passaggio, salvataggio dopo salvataggio.

## 1 · I dati

Il worker prepara una miscela di dati italiani e inglesi da versioni fissate di
dataset pubblici (`acan/lm/dialogue_data.py`, elenchi lunghi in
`acan/lm/dialogue_sources.json`). Ogni file è verificato con SHA-256, letto a
blocchi e cancellato; la preparazione si mette in pausa e riprende dal file e
dalla riga raggiunti, anche dopo un riavvio (riparte da sola).

| Fonte | Contenuto | Licenza |
|---|---|---|
| SmolTalk (smol-smoltalk) | conversazioni in inglese | Apache 2.0 |
| SYNTH (PleIAs) | domande e risposte da Wikipedia, ~7,8 MB di italiano per file | CC BY 4.0 |
| Aya dataset | istruzioni scritte da persone, italiano e inglese | Apache 2.0 |
| OpenAssistant OASST2 | conversazioni di volontari, italiano e inglese | Apache 2.0 |
| Everyday conversations ITA (ReDiX) | conversazioni quotidiane in italiano | Apache 2.0 |
| Wikipedia it/en (novembre 2023) | testo enciclopedico | CC BY-SA 3.0 / GFDL |
| FineWeb-2 italiano (file di test) | testo web filtrato | ODC-By 1.0 |

Miscele: **prova** (~310 MB, ~3,7 GB da scaricare), **base** (~1,9 GB,
~17 GB), **esteso** (~5,6 GB, ~34 GB). Le conversazioni sono scritte nel formato
di chat del tokenizer (`<|im_start|>user … <|im_end|>`). Le conversazioni in
italiano sono la parte scarsa (circa 46, 240 e 480 MB nelle tre miscele);
l'italiano in generale viene soprattutto da Wikipedia e FineWeb-2.

**Riferimento fisso.** I file di test ufficiali delle fonti e circa l'1% + 1%
di tutto il resto, scelti con un hash del testo, sono validazione e test: gli
stessi per tutta la durata, quindi i checkpoint si confrontano su testo mai
visto.

## 2 · La coda di apprendimento

Una coda è una serie di progetti, uno per testo (la miscela preparata e i
propri file). Ogni progetto legge il suo testo per il numero di «giri» scelto
(i passi si calcolano dai token) e riparte dai pesi dell'ultimo checkpoint del
precedente (`warm_start`). Si possono aggiungere testi in fondo mentre impara.
Stato e avanzamento di ogni testo sono nella tabella della coda.

**File propri.** Oltre a `.txt` e `.jsonl`: `.md`, `.py`, `.zip` (anche
annidati una volta), `.epub` (in ordine di lettura), `.docx`, `.pdf` (con il
testo, non le scansioni). Il testo viene estratto al termine del caricamento
(`acan/lm/documents.py`) e tagliato in pezzi di al massimo un milione di
caratteri ai confini di paragrafo.

## Seguirlo, provarlo, fermarlo

- **Salvataggi**: ogni N passi e ogni 5 minuti; si tengono gli ultimi tre, il
  migliore sulla validazione e uno ogni M passi per sempre (`milestone-*`).
- **Manutenzione**: «Pausa e salva» ferma al punto esatto; dopo un riavvio del
  server un progetto con ripresa automatica riparte da solo dall'ultimo
  salvataggio (al massimo tre volte allo stesso passo, poi aspetta).
- **Risposte di prova**: a ogni salvataggio il modello risponde alle stesse
  domande (fino a 6, predefinite o scelte), in modo deterministico; la pagina
  mostra la prima e l'ultima risposta e l'errore sul riferimento.
- **Grandezze**: parametri totali e addestrati, layer a concetti, memoria di
  lettura per layer, passo, token, velocità, tempo stimato, errori, passo di
  apprendimento, norma del gradiente, memoria del processo e della GPU, spazio
  dei checkpoint, dei token e libero sul server.
- **Domande libere, download e ispezione**: pausa, poi «Fai scrivere un
  modello» con la casella «Conversazione», esportazione e ispezione dei layer
  da «Segui i progetti».

## Limiti

Un modello nato da zero deve imparare tutto. Sulla CPU il micro elabora
qualche centinaio di token al secondo: la miscela «prova» (circa 80 milioni
di token) richiede pochi giorni per giro, la «base» settimane. Per un chatbot
che risponde bene servono miliardi di token e i modelli più grandi, quindi una
GPU. I pesi di attenzione (domande, chiavi, valori) imparano insieme ai
concetti fin dall'inizio: è il modello a concetti nativo, non un adattamento.
