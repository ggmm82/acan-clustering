# Apprendimento lungo dell'LLM a concetti

In cima alla scheda **Generativo**. Tutti i layer del modello leggono il testo
attraverso i concetti ACAN (frazione 1,0): gli ultimi token esatti più un numero
fisso di concetti per il resto. Si parte dai pesi di SmolLM2 e si addestrano le
matrici LoRA. Sulla CPU il profilo è SmolLM2-135M, pezzi di 64 token, 16 token
esatti e 16 concetti; con una GPU 512 token, 64 + 64, LoRA di rango 64 e i
modelli 360M e 1.7B.

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

Sulla CPU il modello elabora circa 25 token al secondo: qualche decina di
milioni di token al mese. Le miscele più grandi sono pensate per una GPU. LoRA
adatta una piccola parte dei pesi (circa il 2% con il profilo CPU). Un modello
di 135 milioni di parametri resta limitato con o senza concetti.
