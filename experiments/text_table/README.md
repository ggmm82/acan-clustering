# Misure su WikiText-2

Questa directory conserva i protocolli e le ricevute delle prove sul grafo
tabulare. L'implementazione usata dall'app risiede in `acan/lm`: la scoperta
dei concetti è affidata a `acan/core/concepts.py`. La spiegazione di
apprendimento e previsione si trova nella [guida unica](../../acan/MODEL.md).

Le ricevute in `results/` distinguono i blocchi di test e gli abbinamenti usati
per le singole prove. Nei confronti di sviluppo i vettori erano ricavati dal
train completo. Il worker li ricava dal primo tratto e continua in streaming.

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
