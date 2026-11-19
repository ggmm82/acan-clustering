# Esperimenti sulle coppie di concetti

La [guida del modello](../../acan/MODEL.md) spiega le equazioni, il training e la generazione. Le prove in questa directory misurano la probabilità del prossimo token con lo stesso tokenizer e gli stessi target per ciascuna coppia di modelli.

Il riferimento di confronto è il contatore di suffissi esatti (`counter.py`), che era il modello generativo del sito e non fa più parte del prodotto. Le funzioni comuni (blocchi del protocollo, corpus per articolo, vettori dei token, prove di richiamo) sono in `common.py`.

## Coppie di concetti

Il protocollo breve usa WikiText-2 raw v1, un BPE da 8.000 token appreso sul train e 64 blocchi da 512 token. I 511 seguiti di ciascun blocco danno 32.704 osservazioni. Il contesto e la memoria ripartono a ogni blocco o inizio di articolo.

| Token visibili | Coppie senza memoria | Contatore senza memoria | Coppie con memoria | Contatore con memoria |
|---:|---:|---:|---:|---:|
| 1 | 5,2676 | 5,3004 | 4,8088 | 4,8313 |
| 2 | 4,9028 | 4,9562 | 4,5035 | 4,5426 |
| 4 | 4,8541 | 4,9202 | 4,4600 | 4,5056 |
| 6 | 4,8453 | 4,9141 | 4,4537 | 4,5000 |

La ricevuta storica include il piccolo contributo del raffinamento numerico locale poi rimosso. La prova sulle scale lunghe riparte dai soli figli categorici. [Ricevuta della prova breve e dei dieci dataset tabulari](results/native-pair-scales.json).

## Contesto di articolo

Le posizioni di test sono identiche alla prova breve. Ora ciascun token vede tutto il prefisso del suo articolo, anche quando il blocco misurato comincia a metà frase. Le letture globali sono calibrate su 32 blocchi di validazione. Gli altri 32 decidono se accettare la scala. I pesi delle scale precedenti restano fissi. Dopo due tentativi consecutivi senza riduzione della NLL sul controllo, la ricerca si arresta. Il contatore di riferimento usa al massimo sei token e riceve gli stessi prefissi.

Per riprodurre, mettere in `DIR` i file `train.parquet`, `validation.parquet` e `test.parquet` di WikiText-2-raw-v1, revisione `b08601e04326c79dfdd32d625aee71d232d685c3`. La prima invocazione prepara il tokenizer. Usare una directory `RIGHE` vuota per ciascun corpus, tokenizer e serie di opzioni. I pickle intermedi sono file di lavoro locali prodotti dallo stesso run.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m experiments.text_table.run --data DIR --output iniziale.json
python -m experiments.text_table.pair_layers --data DIR --workdir RIGHE
python -m experiments.text_table.pair_reading --workdir RIGHE --output scale_brevi.json
python -m experiments.text_table.pair_subconcepts --workdir RIGHE
python -m experiments.text_table.long_pairs --data DIR --workdir RIGHE --output scale.json
python -m experiments.text_table.long_counter --data DIR --output contatore.json
python -m experiments.text_table.document_pairs --data DIR --workdir RIGHE --scores scale.json --output memoria.json
python -m experiments.text_table.queue_pairs --data DIR --budget-mib 768 --output coda.json
```

Per la prova di trasferimento su WikiText-103, passare a `document_pairs` anche `--model grafo.pkl`. Per riprodurre il trasferimento della scala 32, filtrare prima la ricevuta delle scale alle sole voci fino a 32. Usare in ordine i due shard del train e lo stesso tokenizer. Il contatore riparte da zero, il grafo conserva la geometria, la distribuzione di base e i predittori selezionati su WikiText-2. Tutte le tabelle sparse dei seguiti ripartono da zero. Entrambi leggono ogni token del nuovo train. Il confronto usa le sezioni ufficiali di validazione e test condivise dalle due edizioni di WikiText.

```sh
python -m experiments.text_table.encode_articles --shards train-0.parquet train-1.parquet --tokenizer DIR/bpe-8000.json --ids train.u16 --starts articoli.npy
python -m experiments.text_table.transfer_pairs --ids train.u16 --starts articoli.npy --model grafo.pkl --data DIR --budget-mib 768 --output trasferimento.json
```

Per verificare la generalizzazione tabulare sui file originali, eseguire i comandi seguenti sulla directory che contiene gli allegati.

```sh
python -m experiments.text_table.dyadic_tabular ALLEGATI tabulare.json
python -m experiments.text_table.pair_tabular ALLEGATI figli_tabulari.json
```
