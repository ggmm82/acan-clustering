# Errata dei due paper in `paper/`

I PDF e il sorgente TeX restano invariati come archivio storico: i loro hash
sono parte dei controlli di rilascio. Le affermazioni seguenti **non vanno
citate come risultati del progetto**. Motivazioni e dimostrazioni sono in
[MATHEMATICAL-FOUNDATION.md](MATHEMATICAL-FOUNDATION.md), sezione A.3; le misure
in [CP-V9-RESULTS.md](CP-V9-RESULTS.md).

## Algebraic Concept Attention (v8)

| Affermazione | Stato |
|---|---|
| Teorema 1, invarianza asintotica agli attributi irrilevanti | Dimostrazione errata: per rumore uniforme $\max(0,1-2\sigma)\approx0{,}423$, non zero. Sostituito dalla Proposizione 3 della fondazione, che dà una soglia dipendente da separazione e sostegno. |
| Teorema 2, XOR risolto dall'accoppiamento $\omega$ | Vero ma non necessario: le quattro combinazioni sono gruppi separati nello spazio originale. |
| Teorema 3, collasso gravitazionale per $\gamma\ge1$ | Falso per $\gamma=1$ (punto neutro). L'aggregazione è ora una conseguenza del costo di fusione. |
| Tabella 4, benchmark Iris, Zoo, Wine, Wisconsin | Dati sintetici con quei nomi, non i dataset UCI. |
| «Convergenza globale di Adam» | Non dimostrata. |
| «Compressione» a 2–4 slot su 140 | Su dati reali la CP di quel profilo aggrega troppo (2 gruppi su Letter, accuratezza 4%); il numero di slot attivi non è una misura di qualità. |

## ACAN-LM vs. MiniCPM

| Affermazione | Stato |
|---|---|
| Speedup 38× a 4.096 token | Non misurato su un modello innestato: la latenza «ibrida» è una media pesata di due misure del codice NumPy di riferimento. Le esecuzioni reali del laboratorio misurano 5–28 token/s. |
| KV-cache −99,2% | Vale per lo stato dei soli livelli ACAN; in un ibrido al 50% la cache dei livelli densi resta. |
| Perplessità 1,51 contro 4,65; 90% con il 70% di distrattori | Corpus sintetico di poche frasi; il codice di riferimento calcola statistiche, masse e rilevanza sull'intera sequenza, quindi usa informazione dal futuro. |
| Infilling bidirezionale 100% | Stessa fuga di informazione. |
| «Immunità al context rot» | Non dimostrata; uno stato di rango $k$ diluisce un'informazione vista una volta come $k/L$ (fondazione, C.1). |
| Complessità $O(1)$ in $L$ senza perdita | Impossibile per qualunque architettura (fondazione, Proposizione 5). |
| Riferimenti «Context Rot Study (2026)», Perez (2025), collegamenti a Wikipedia | Da verificare o sostituire con fonti primarie. |

## Che cosa resta valido

L'impianto concettuale: concetti come comportamenti di gruppo, completamento di
qualunque sottoinsieme di attributi con un solo modello, rilevanza degli
attributi leggibile, costo lineare del routing a numero di concetti fissato.
La formulazione corrente e le sue misure sono nel profilo `cp-v9`.
