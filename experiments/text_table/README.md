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
