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

## Contesto a scale diadiche

L'esperimento usa 20 scale sovrapposte, una tabella di composizione per ciascuna
scala con esempi disponibili e un `ConceptModel` con il token come target
categorico. Il contatore con identità esatte e la precedente lettura dei suffissi
con gruppi CP sono i controlli. Le prove di associazione, copia e riformulazione
usano soltanto la tabella diadica.

```sh
OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python -m experiments.text_table.dyadic_run \
  --data DIR --output diadico.json --checkpoint-output selezionato.pkl
python -m experiments.text_table.dyadic_control --data DIR --output controlli.json
python -m experiments.text_table.dyadic_tabular DIR_ALLEGATI tabelle.json
OPENBLAS_NUM_THREADS=1 python -m experiments.text_table.dyadic_judgment > giudizio.json
```

Il tokenizer e i parquet in DIR sono quelli del protocollo WikiText-2 descritto
sopra. Le co-occorrenze e le finestre rispettano gli inizi di articolo. Il test
usa gli stessi 64 offset del confronto con i suffissi. La ricevuta dà sia il
risultato partendo dal blocco vuoto, sia quello con il prefisso dell'articolo.
La prova lessicale cambia «doctor» in «physician» e così via per sedici chiavi
inglesi con risposte assegnate a caso. Gli ID delle risposte sono esclusi dal
riempimento. Il file pickle opzionale serve alla ripetizione locale della prova.

Per completare un prompt con il checkpoint locale appena prodotto:

```python
import pickle
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("DIR/bpe-8000.json")
with open("selezionato.pkl", "rb") as file:
    modello = pickle.load(file)
prefisso = tokenizer.encode("The dog ran across the").ids
print(tokenizer.decode(modello.generate(prefisso, 8)))
```

## Lettura condizionale delle regioni CP

Usare i parquet e il BPE della prova diadica descritta sopra. I due candidati
partono dallo stesso checkpoint locale di `dyadic_run`. Omettendo `--base`
il programma lo ricostruisce dal training. I file di checkpoint contengono
oggetti Python e i comandi caricano il checkpoint prodotto localmente.

```sh
python -m experiments.text_table.dyadic_run --data DIR --output base.json --checkpoint-output base.pkl
python -m experiments.text_table.conditional_run --data DIR --base base.pkl --mode svd --neighbors 96 --output svd.json --checkpoint svd.pkl
python -m experiments.text_table.conditional_run --data DIR --base base.pkl --mode cp_log --neighbors 96 --output cp.json --checkpoint cp.pkl
python -m experiments.text_table.word_refinement --data DIR --base svd.json --output parole.json
python -m experiments.text_table.conditional_eval --data DIR --candidates parole.json svd.json cp.json --output test.json
python -m experiments.text_table.conditional_generate --checkpoint parole.pkl --tokenizer DIR/bpe-8000.json --tokens 16 'The first train'
python -m experiments.text_table.conditional_control --data DIR --checkpoint parole.pkl --output confronto.json
python -m experiments.text_table.conditional_load --data DIR --checkpoint svd.pkl --output carico.json
```

La NLL sui primi 32 blocchi di validazione sceglie il checkpoint. Gli altri 32
servono da controllo. Il test usa gli offset pubblicati, compresi i confini di
articolo dentro ogni blocco. La prova di carico verifica i totali di ciascuna
categoria dopo aver sommato un GiB di righe ripetute in un livello del modello.
[Qualità, ablazioni e richiamo lessicale](results/wikitext2-word-regions.json) ·
[Confronto appaiato](results/word-regions-paired.json) ·
[Somme in streaming](results/conditional-load.json).
