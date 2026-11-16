# Prototipo: il modello linguistico che è il modello a concetti

Nessuna rete neurale, nessun peso addestrato: il testo è una tabella. Ogni posizione è una riga,
i simboli precedenti sono i suoi attributi, il simbolo successivo è il target.

- **Concetti per contesto** (`model.py`): al livello j stanno le posizioni con gli stessi ultimi
  j simboli. Somme: C (membri), N (conteggi del simbolo che segue), D (simboli diversi che seguono).
  Il livello sopra (contesto più corto) è il riferimento di ogni concetto, con peso a_j·D;
  a_j è scelto dal giudizio (ogni posizione letta da somme che non la contengono).
  Concetti con un solo membro sono lasciati al livello sopra.
- **Concetti del testo letto**: gli stessi concetti costruiti solo dalle posizioni precedenti
  del testo in corso, con la previsione globale come riferimento: il modello può riprendere
  ciò che il testo ha detto prima.

Misure con gli stessi dati e lo stesso protocollo della prova in piccolo (wikitext, caratteri):

| | Rete con attenzione | Rete solo finestra | Modello a concetti |
|---|---:|---:|---:|
| Testo, bit per carattere (pezzi di 256) | 2,518 | 2,451 | 2,172 |
| Memoria, risposta a 49 token | 100% | 5,5% | 100% |
| Memoria lontana, fino a 177 token | 100% | — | 100% |
| Tempo (un thread) | 219 s di addestramento | 213 s | 47 s di costruzione |

    python scripts/concept_lm/measure_text.py 8 4
    python scripts/concept_lm/measure_memory.py
