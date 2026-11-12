# CP sostenuta da osservazioni diverse

La correzione modifica il nucleo comune, senza imporre un numero di concetti.
Un concetto non può ottenere qualità dalla sola riga che rappresenta.
Date le appartenenze T, si calcola la frazione di coppie che provengono da
osservazioni con indici diversi:

$$
h_i=\frac{(\sum_tT_{it})^2-\sum_tT_{it}^2}{(\sum_tT_{it})^2},\qquad
\widetilde q_i=h_i\frac{\sum_jR_{ij}^2}{\sum_jR_{ij}}.
$$

Questa è un'identità combinatoria sui pesi, non una prova che le righe siano
statisticamente indipendenti o che il numero di cluster sia identificabile.
Con appartenenze nette h=(C−1)/C; un singleton ha qualità sostenuta zero.
Con qualità q uguali, masse coerenti e sovrapposizioni trascurabili, il termine
medio diventa q(1−K/n). Il costo della frammentazione emerge dalle coppie escluse:
non viene aggiunto un coefficiente per K scelto sul risultato di Gioco.

CP conserva penalità delle sovrapposizioni e affinità fra masse e frequenze,
sostituendo q con la qualità sostenuta. Proiezione delle masse e accettazione
per obiettivo restano identiche. In modalità senza target non c'è NLL; in
classificazione e nel linguaggio resta la loss NLL più penalità CP.

NumPy e PyTorch chiamano lo stesso operatore. Nel linguaggio anche le somme
quadratiche delle appartenenze sono accumulate soltanto sul prefisso, con memoria
aggiuntiva proporzionale agli slot, non ai token passati. La correzione non usa
token futuri. Valori, gradienti e accumulo causale sono verificati.

## Risultati

**Gioco: 4 gruppi sulle 101 righe originali senza target**, compresa la riga con
valore mancante. La precedente CP condivisa ne produceva cinque. Nessun nome
colonna, K richiesto, esclusione di righe o fusione entra nella correzione.
Anche la preparazione web senza target conserva ora l'intero file: il protocollo
80/20 continua a essere usato quando esiste una risposta da prevedere.
Le cinque prove supervisionate seed1/7/21/42/99 mantengono quattro concetti e 100%.

Letter gerarchico completo: **92,325%**, 3.346 prototipi; precedente 92,70%, 3.703.
Campione web: **91,05%**, 1.387 prototipi; precedente 89,85%, 1.701.
Il 95,78% storico resta non recuperato. Il miglioramento riguarda soprattutto
il campione web, il recupero di Gioco non supervisionato e la frammentazione;
non si rivendica una superiorità su ogni dataset.

Tre split seed42/7/21, stessi parametri fra dataset:

| File | Accuracy prima | Accuracy corretta | Prototipi prima | Prototipi corretti |
|---|---:|---:|---:|---:|
| adultMinimal.xlsx | 72.67% | 71.33% | 370–444 | 299–442 |
| breast-cancer.arff | 67.24% | 66.67% | 191–200 | 184–191 |
| iris.arff | 95.56% | 96.67% | 23–30 | 17–41 |
| mushroom.arff | 100.00% | 100.00% | 71–155 | 94–113 |
| sonar.xlsx | 84.13% | 82.54% | 85–91 | 33–41 |
| titanic.csv | 81.19% | 80.45% | 419–545 | 453–472 |
| vote.arff | 92.34% | 92.34% | 189–194 | 189–192 |

Lo split aggiuntivo seed99 è incluso nelle ricevute e mostra ancora risultati
misti. Sono dati di sviluppo ripetutamente esaminati, non una conferma esterna.
La CP corretta non garantisce l'ottimo globale né un numero semanticamente vero
su dati arbitrari. Una struttura rara autentica può ricevere meno sostegno:
il compromesso va valutato con dati separati.

## Stabilità e compatibilità

È stato corretto anche un overflow numerico di Adam: quando il quadrato di un
gradiente finito eccede la rappresentazione, il secondo momento viene conservato
come radice e aggiornato con hypot. È lo stesso aggiornamento matematico, senza
clipping o cambiamento della direzione; il percorso ordinario rimane identico.
Il controllo su Gioco passa anche trattando gli overflow come errori.

I nuovi checkpoint dichiarano `pair-supported-cp-v1`. L'inferenza dei modelli
tabulari precedenti resta disponibile, ma nuovi rami non devono mescolare
formule CP diverse. I progetti linguistici esistenti non vengono modificati:
la nuova formula richiede un training coerente con il codice che la definisce.

130 test passati e audit generativo 49/49, senza skip. Controllati differenze
finite, parità dei gradienti, causalità, streaming, checkpoint e ripresa.
[Metriche aggregate e hash](results/cp-pair-support.json). Nessuna riga dei file
caricati è inclusa nelle ricevute.
