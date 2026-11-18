# Misure su WikiText-2

Questa directory conserva i protocolli e le ricevute delle prove sul grafo
tabulare. L'implementazione usata dall'app risiede in `acan/lm`: la scoperta
dei concetti è affidata a `acan/core/concepts.py`. La spiegazione di
apprendimento e previsione si trova nella [guida unica](../../acan/MODEL.md).

Le ricevute in `results/` distinguono i blocchi di test e gli abbinamenti usati
per le singole prove. Nei confronti di sviluppo i vettori erano ricavati dal
train completo. Il precedente grafo a righe causali li ricavava dal primo tratto. La lettura CP nei suffissi ricalcola i profili dai bigrammi conservati.

Per riprodurre le misure storiche, preparare i parquet train, validation e test
di WikiText-2-raw-v1, revisione `b08601e04326c79dfdd32d625aee71d232d685c3`.
Eseguire `python -m experiments.text_table.run --data DIR --output prima.json`,
poi `python -m experiments.text_table.layers --data DIR --output livelli.json
--save-table tabella.pkl` e `python -m experiments.text_table.context --table
tabella.pkl --data DIR --output confronto.json`. I file pickle sono checkpoint
locali delle prove storiche. L'app salva le matrici con safetensors.

Per ripetere la misura dell'intero train con il codice del worker,
eseguire `python -m experiments.text_table.stream --data DIR --output stream.json`
dopo la preparazione del tokenizer. La [ricevuta](results/wikitext2-worker-stream.json)
riporta il primo e l'ultimo passo sugli stessi blocchi.

Per confrontare i gruppi CP nei suffissi ordinati usare i medesimi parquet e
tokenizer del protocollo precedente. Il giudizio sceglie i pesi sui primi 32
blocchi di validazione. Le ricevute riportano i 64 nuovi offset del test,
l'errore standard delle differenze per blocco e il controllo con gruppi permutati.

```sh
python -m experiments.text_table.refinement --data DIR --budget-mib 256 --output confronto.json
python -m experiments.text_table.refinement --data DIR --budget-mib 1024 --output confronto-1024.json
python -m experiments.text_table.refinement --data DIR --budget-mib 256 --continuous --output continuo.json
python -m experiments.text_table.suffix_recall --data DIR --output richiamo.json
python -m experiments.text_table.load_test --data DIR --output carico.json
```

Il confronto storico separa i blocchi di conteggio ogni 32.768 token. Con
`--continuous` entrambe le letture conservano il suffisso al confine come la
nuova coda. La prova di carico ripete e ruota gli ID del train per generare
nuove chiavi. Misura tempi e memoria al crescere del flusso. Il numero di
byte è quello degli identificatori a 32 bit. La prova di qualità usa invece
il train originale e un test separato.

Per misurare l'effetto di una quantità maggiore di testo naturale, il protocollo
`natural_stream` legge entrambi gli shard originali di train di WikiText-103 raw
alla stessa revisione. WikiText-103 e WikiText-2 condividono esattamente i
parquet di validazione e test, verificabili tramite SHA-256. Si conserva il BPE
8.000 già imparato sul train piccolo per confrontare le NLL nella stessa unità.
Il programma tokenizza in streaming, verifica la validazione alla fine di ciascuno
shard e legge i 64 blocchi di test dopo aver completato entrambi.

```sh
python -m experiments.text_table.natural_stream --data DIR \
  --shards train-00000-of-00002.parquet train-00001-of-00002.parquet --output naturale.json
```

Per provare il profilo della coda web, compresi il tokenizer da 32.000, la
preparazione dei documenti, il worker e il caricamento del checkpoint, eseguire
`python -m experiments.text_table.worker_queue --data DIR --output worker.json`.
I dati hanno split espliciti e i duplicati vengono rimossi dal worker.
Le NLL con vocabolari diversi sono espresse in unità di token diverse.
