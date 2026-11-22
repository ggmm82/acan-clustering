"""Lightweight Dash client for the separate, persistent generative worker."""
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import dash
from . import long_training
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import request as flask_request, jsonify, Response, stream_with_context


TRAINING_HELP = {'lm-length': 'Quanto testo il modello legge in una volta (in token, circa pezzi di parola). Più lungo = più memoria.',
 'lm-batch': 'Quanti pezzi di testo legge insieme. Su CPU lascia 1.',
 'lm-accum': 'Quanti pezzi legge prima di correggersi una volta. Più alto = passi più stabili ma più lenti.',
 'lm-lr': "Quanto si corregge a ogni passo. Se l'errore sale o oscilla molto, abbassalo.",
 'lm-slots': 'Quanti concetti riassumono ogni livello del testo più vecchio (i livelli crescono con il logaritmo della lunghezza). Più concetti = più fedele ma più memoria.',
 'lm-window': 'Quanti token recenti restano esatti, non riassunti.',
 'lm-rank': 'Quanto è grande la parte del modello che si addestra (LoRA). Più alto = più capacità e memoria.',
 'lm-save': 'Ogni quanti passi salvare una copia del modello da cui provare a generare.',
 'lm-patience': 'Dopo quanti controlli senza miglioramenti mettersi in pausa da solo. 0 = mai.',
 'lm-min-delta': "Di quanto deve scendere l'errore per contare come miglioramento.",
 'lm-validation': "Su quanti pezzi di testo mai visto si controllano i progressi. Più pezzi = confronti più affidabili, controlli più lenti. Per confrontare due progetti usa almeno 32.",
 'lm-cp-weight': 'Compromesso fra previsione e organizzazione. Zero: sola NLL.'}


MODELS = ['HuggingFaceTB/SmolLM2-135M', 'HuggingFaceTB/SmolLM2-360M', 'HuggingFaceTB/SmolLM2-1.7B']   # earlier projects only
NATIVE_SIZES = [dict(label='Micro · circa 12 milioni di parametri', value='micro'), dict(label='Piccolo · circa 50 milioni', value='small'),
                dict(label='Base · circa 138 milioni (GPU)', value='base')]


def backend(workspace, path, method='GET', data=None, raw=False, headers=None, timeout=30):
    endpoint = os.environ.get('ACAN_LM_URL', '').rstrip('/')
    token = os.environ.get('ACAN_LM_SERVICE_TOKEN', '')
    if not endpoint or not token:
        raise ValueError('Il servizio che addestra i modelli linguistici non è collegato a questa installazione: questa scheda non può avviare né mostrare progetti. Per chi gestisce il server: avviare il servizio dedicato e impostare ACAN_LM_URL e ACAN_LM_SERVICE_TOKEN.')
    if not isinstance(workspace, str) or not re.fullmatch('[0-9a-f]{64}', workspace):
        raise ValueError('Workspace non ancora pronto: attendi il caricamento della pagina.')
    body = data if raw else json.dumps(data).encode() if data is not None else None
    fields = {'Authorization':'Bearer '+token, 'X-ACAN-Workspace':workspace,
              'Content-Type':'application/octet-stream' if raw else 'application/json'}
    fields.update(headers or {})
    try:
        with urlopen(Request(endpoint+path, data=body, headers=fields, method=method), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            message = json.load(exc).get('error', str(exc))
        except (ValueError, KeyError):
            message = str(exc)
        raise ValueError(message) from None
    except URLError:
        raise ValueError('Worker generativo non raggiungibile.') from None


def inside_figure(trace):
    """The last step learned, opened up: its validation as the network reads the piece, token by token; in a race
    of growth every candidate learns the same piece and has its own line, so the one that goes on can be seen."""
    if not trace:
        return None
    step = math.floor(max(point[0] for point in trace) - 1e-9)
    points = [point for point in trace if step <= point[0] <= step + 1]
    if len(points) < 2:
        return None
    names = {None: 'La rete che impara', "com'è": "Gara · la rete com'è", 'larghezza': 'Gara · il doppio della larghezza',
             'strato': 'Gara · uno strato in più', 'concetti': 'Gara · concetti divisi in due'}
    figure = go.Figure()
    for kind in dict.fromkeys(point[2] if len(point) > 2 else None for point in points):
        line = [point for point in points if (point[2] if len(point) > 2 else None) == kind]
        figure.add_scatter(x=[100 * (point[0] - step) for point in line], y=[point[1] for point in line],
                           mode='lines+markers', name=names.get(kind, f'Gara · {kind}'))
    figure.update_layout(title=f'Dentro il passo {step + 1}: la validazione mentre la rete legge il pezzo',
                         xaxis_title='Pezzo letto (%)', yaxis_title='NLL', margin=dict(t=50, b=40), height=360)
    return figure


layout = dbc.Container([
    html.H4('Il modello a concetti che scrive', className='mt-3'),
    html.P("Gli stessi concetti della scheda Modello, applicati al testo: ogni posizione è una riga e il token successivo è il "
           "target. Qui i concetti formano una rete che impara lungo il gradiente e cresce con il testo. Si usa in cinque passi, dall'alto in "
           "basso: prepari i testi, crei la coda, la segui, provi e scarichi il modello, e se vuoi lo usi da altri programmi. "
           "Come funziona: schede Matematica e Informatica.", className='small'),
    dcc.Store(id='lm-workspace', storage_type='local'), dcc.Store(id='lm-state'),
    dcc.Store(id='lm-chart-view',storage_type='session'),
    dcc.Store(id='lm-active-job', storage_type='session'),
    dcc.Store(id='lm-upload-state'), dcc.Store(id='lm-generation-id'), dcc.Store(id='lm-browser-ready'), dcc.Store(id='lm-evaluation-id'), dcc.Store(id='lm-evaluation-report'),
    html.Div(id='lm-connection', role='status'),
    dcc.Interval(id='lm-poll', interval=5000, disabled=True),
    long_training.layout,
    dbc.Card(dbc.CardBody([
    html.H4('4 · Prova il modello e scaricalo'),
    html.P("Ogni testo della coda è un progetto: il suo modello contiene tutto ciò che la coda ha imparato fino a lui. "
           "Scegli il progetto (di solito l'ultimo che ha finito) e qui vedi il suo grafico, lo fai scrivere e lo scarichi.",
           className='small'),
    dbc.Label('Progetto'),
    dcc.Dropdown(id='lm-job', placeholder='Scegli un progetto: un testo della coda'),
    dbc.Button("Aggiorna l'elenco dei progetti", id='lm-refresh', outline=True, color='primary', size='sm', className='my-2 me-2'),
    html.Div(id='lm-action-status', role='status'), html.Div(id='lm-progress'),
    dbc.Row([
        dbc.Col([dbc.Label('Grafico'),dcc.Dropdown(id='lm-history-mode',value='overview',clearable=False,
            options=[{'label':'Tutto il percorso','value':'overview'},{'label':'500 passi alla volta, punto per punto','value':'window'}])],md=6),
        dbc.Col([dbc.Label('Dal passo'),dbc.Input(id='lm-history-start',type='number',min=0,value=0,step=1),
            dbc.Button('← 500 passi prima',id='lm-history-prev',size='sm',outline=True,color='primary',className='me-1 mt-1'),
            dbc.Button('500 passi dopo →',id='lm-history-next',size='sm',outline=True,color='primary',className='mt-1')],md=6),
    ]),
    html.P("Più in basso è la linea, meno il modello sbaglia a indovinare la parola successiva. Conta la linea su testo mai visto (validazione): se risale mentre quella di addestramento continua a scendere, il modello sta imparando a memoria.",className='small text-muted'),
    dcc.Loading(dcc.Graph(id='lm-loss'),delay_show=150),
    html.Div(dcc.Graph(id='lm-inside'), id='lm-inside-box', style={'display': 'none'}),
    html.H5('Fai scrivere il modello', className='mt-3'),
            html.P("Scegli una copia salvata (la migliore è indicata) e scrivi l'inizio di una frase, oppure una domanda con la casella «Conversazione»: il modello continua. La prova non cambia il modello. Il server fa una cosa alla volta: se una coda sta imparando, la scrittura non parte; aspetta che finisca o metti in pausa la coda al passo 3, e dopo premi «Riprendi»."),
            dcc.Dropdown(id='lm-checkpoint', placeholder='Copia salvata del modello: la migliore e le ultime tre'),
            dbc.Textarea(id='lm-prompt', value='', placeholder="Scrivi l'inizio di una frase, nella lingua del testo di addestramento", rows=3),
            dbc.Label('Quanto testo scrivere (token, circa pezzi di parola; fino a 256, fino a 4.096 per i modelli nativi: un\'immagine sono circa 1.060 token, iniziando con <|image|>)'), dbc.Input(id='lm-new-tokens', type='number', value=64, min=1, max=256),
            dbc.Checkbox(id='lm-full-length', value=True, label='Scrivi sempre fino alla lunghezza richiesta (senza fermarsi a fine paragrafo)'),
            dbc.Checkbox(id='lm-chat', value=False, label='Conversazione: il testo è una domanda e il modello risponde fino a fine turno (per i modelli addestrati sui dialoghi)'),
            html.Div(dbc.Checkbox(id='lm-inspect-generation', value=False), style={'display': 'none'}),
            dbc.Button('Continua la frase', id='lm-generate', className='my-2'),
            html.Div(id='lm-generation-status', role='status'), html.Div(id='lm-output'),
            html.Div(id='lm-generation-metrics'),
            dbc.Button('Scarica questa copia del modello', id='lm-export-model', outline=True, color='primary', className='my-2'),
            html.Small('Contiene i concetti e il tokenizer: si ricarica senza altri file (istruzioni nel README del pacchetto).', className='text-muted d-block'),
            html.Div(id='lm-export-status', role='status'),
    dbc.Accordion([dbc.AccordionItem([
    dbc.Button('Pausa e salva', id='lm-pause', color='warning', className='m-1'),
    dbc.Button('Riprendi', id='lm-resume', className='m-1'),
    dbc.Row([
        dbc.Col(dbc.Input(id='lm-rename-value', placeholder='Nuovo nome del progetto scelto'), md=5, className='mt-1'),
        dbc.Col([dbc.Button('Rinomina', id='lm-rename', outline=True, color='primary', className='me-2 mt-1'),
                 dbc.Button('Elimina il progetto', id='lm-delete', outline=True, color='danger', className='mt-1')], md=7),
    ], className='my-1'),
    dcc.ConfirmDialog(id='lm-delete-confirm', message='Eliminare il progetto scelto con tutte le sue copie salvate? Non si può annullare.'),
    html.Div(id='lm-manage-status', role='status', className='small'),
    ], title='Gestisci il progetto: pausa, ripresa, nome, eliminazione')], start_collapsed=True, className='my-2'),
]), className='my-3', color='primary', outline=True),
    dbc.Card(dbc.CardBody([
    html.H4('5 · Usalo da altri programmi (API compatibile OpenAI)'),
            html.P("Ogni programma scritto per l'API di OpenAI (la libreria «openai», LangChain, Open WebUI, curl) può usare i "
                   "modelli a concetti di questo spazio di lavoro: basta cambiare indirizzo e chiave. Il modello è l'id del progetto; "
                   "risponde con l'ultimo salvataggio, anche mentre continua a imparare.", className='small'),
            dbc.Row([dbc.Col(dbc.Input(id='lm-api-name', placeholder='Nome della chiave (per esempio: Open WebUI)', maxLength=80), md=8),
                     dbc.Col(dbc.Button('Crea una chiave', id='lm-api-create', color='primary'), md=4)], className='mb-2'),
            html.Div(id='lm-api-created', role='status'),
            html.Div(id='lm-api-keys'),
            dbc.Row([dbc.Col(dcc.Dropdown(id='lm-api-revoke-key', placeholder='Chiave da revocare'), md=8),
                     dbc.Col(dbc.Button('Revoca', id='lm-api-revoke', outline=True, color='danger'), md=4)], className='my-2'),
            html.Div(id='lm-api-status', role='status', className='small'),
            dcc.Markdown('''
**Indirizzi** (sullo stesso dominio di questa pagina): `GET /v1/models` · `POST /v1/chat/completions` · `POST /v1/completions`

```python
from openai import OpenAI
client = OpenAI(base_url="https://<questo sito>/v1", api_key="acan-…")
print([m.id for m in client.models.list()])
risposta = client.chat.completions.create(
    model="<id del progetto>",
    messages=[{"role": "user", "content": "Ciao! Chi sei?"}],
    temperature=0.7, max_tokens=128)
print(risposta.choices[0].message.content)
```

Parametri: `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `stream` (le risposte arrivano un pezzo alla volta).
Un modello è un progetto (`<id>`), oppure un suo salvataggio preciso (`<id>:<nome del checkpoint>`).
La chiave si vede solo quando la crei: conservala; se la perdi, revocala e creane un'altra.
''', className='small'),
            dcc.Store(id='lm-api-refresh'),
]), className='my-3', color='primary', outline=True),
    html.Hr(), html.H5('Strumenti avanzati'),
    html.P('Non servono per usare il modello: restano per esperimenti, confronti e debug.', className='small text-muted'),
    dbc.Accordion([dbc.AccordionItem([
            dbc.Card(dbc.CardBody([
                html.H5('Il percorso di validazione'),
                html.P("Testo da usare nei passi 1 e 2: uno pubblico (si scarica una volta) oppure uno tuo, caricato al passo 1 («I tuoi documenti»).", className='small text-muted mb-1'),
                dcc.Dropdown(id='lm-validation-text', placeholder='Scegli il testo', clearable=False, className='mb-3'),
                html.H6('Passo 1 · Prova in piccolo, da zero (CPU, minuti)'),
                html.P("Domanda: un modello che legge il testo lontano attraverso i concetti ricorda ciò che serve, come uno "
                       "con l'attenzione normale, a un costo per token che quasi non cresce con il testo? Tre modellini identici "
                       "imparano da zero, cambiando solo come leggono: attenzione normale, solo le ultime 16 parole, concetti "
                       "(ultime 16 parole esatte più 16 concetti per ogni livello del resto).", className='small mb-1'),
                html.Ul([html.Li("Memoria (criterio): coppie da ricordare, testo di riempimento, poi la domanda; la risposta è a "
                                 "49 token, oltre le 16 parole esatte."),
                         html.Li("Memoria lontana (criterio, solo nella prova completa): la risposta a 177 token. Se qui i "
                                 "concetti falliscono, un modello grande erediterebbe il difetto."),
                         html.Li("Testo (criterio): i concetti non devono peggiorare la previsione del testo normale."),
                         html.Li("Dati necessari (misura): quanto testo leggono i due modelli per arrivare allo stesso livello."),
                         html.Li("Costo (misura): tempo per token e vettori tenuti a 512-4.096 token di contesto.")],
                        className='small'),
                dbc.Row([dbc.Col(dcc.Dropdown(id='lm-small-budget', value='quick', clearable=False, options=[
                            {'label': 'Rapida, senza memoria lontana: 20 minuti', 'value': 'quick'}, {'label': 'Completa: circa un\'ora e un quarto', 'value': 'full'}]), md=5),
                         dbc.Col([dbc.Button('Avvia la prova in piccolo', id='lm-small-start', color='primary', className='me-2'),
                                  dbc.Button('Ferma', id='lm-small-stop', outline=True, color='warning')], md=7)], className='mb-2'),
                html.Div(id='lm-small-status', role='status', className='small'), html.Div(id='lm-small-view'),
                dbc.Button('Scarica tutti i dati della prova (per il debug)', id='lm-small-download', outline=True, color='secondary', size='sm', className='mt-1'), dcc.Download(id='lm-small-file'),
                dcc.Store(id='lm-small-id'), dcc.Store(id='lm-small-result'), dcc.Interval(id='lm-small-poll', interval=5000),
                html.H6('Passo 2 · Prova su un modello nativo vero (micro, nato da zero)', className='mt-3'),
                html.P("Da fare dopo il passo 1. Due modelli nativi micro nascono da zero sullo stesso testo, con lo stesso "
               "tokenizer, gli stessi passi e lo stesso seme: uno con i concetti in ogni layer, uno con l'attenzione normale. "
               "Partono uno dopo l'altro e compaiono sullo stesso grafico. Linee vicine (differenza sotto 0,15 nella verifica "
               "finale) vogliono dire che i concetti non costano qualità al modello che addestrerai.", className='small'),
        dbc.Row([dbc.Col(dbc.InputGroup([dbc.InputGroupText('Passi'), dbc.Input(id='lm-validation-steps', type='number', value=600, min=50, max=100000)]), md=4),
                         dbc.Col(dbc.Button('Avvia il confronto sul modello vero', id='lm-validation-start', color='primary'), md=8)]),
                html.Div(id='lm-validation-status', role='status', className='small mt-1'),
                html.H6('Passo 3 · Decisione', className='mt-3'),
                html.Div(id='lm-decision'),
            ]), className='my-3'),
    ], title='Laboratorio: il modello a concetti contro due reti neurali, in piccolo')], start_collapsed=True, className='my-3'),
    dbc.Accordion([
    dbc.AccordionItem([
            dbc.Card(dbc.CardBody([
                html.H5('Avvio guidato · dati pubblici'),
                html.P("Il modo più semplice per provare: scegli un testo pubblico e quanto a lungo addestrare. L'app scarica il testo, fa nascere da zero un modello a concetti micro (tokenizer imparato dal testo, concetti in ogni layer) e lo addestra. I progetti già esistenti non vengono toccati."),
                dbc.Row([
                    dbc.Col([dbc.Label('Testo pubblico'), dcc.Dropdown(id='lm-public-dataset', value='dante-it', clearable=False, options=[{'label':'Divina Commedia · italiano letterario', 'value':'dante-it'}, {'label':'Tiny Shakespeare · inglese', 'value':'tiny-shakespeare'}, {'label':'WikiText-2 · inglese · split ufficiali', 'value':'wikitext-2'}])], md=6),
                    dbc.Col([dbc.Label('Quanto addestrare'), dcc.Dropdown(id='lm-public-preset', value='check', clearable=False, options=[{'label':'Prova breve · 20 passi', 'value':'check'}, {'label':'Esercizio · 200 passi', 'value':'practice'}])], md=6),
                ]),
                html.Div(id='lm-public-info', className='my-2'),
                dbc.Button('Scarica e avvia la prova', id='lm-public-start', disabled=True),
                html.Div(id='lm-public-status', role='status', className='my-2'),
                html.P('Il primo avvio impara anche il tokenizer dal testo: qualche decina di secondi in più.', className='small text-muted'),
                html.Ol([
                    html.Li('Aspetta che lo stato diventi «running»: il modello legge il testo e si corregge passo dopo passo.'),
                    html.Li("Guarda il grafico: la linea dell'errore su testo mai visto deve scendere. Più basso è meglio."),
                    html.Li('Premi «Pausa e salva», aspetta «paused», scegli la copia salvata migliore e fagli continuare una frase.'),
                    html.Li("Per sapere se i concetti funzionano avvia il controllo con l'attenzione normale («Confronti e serie a mano»): compare sullo stesso grafico."),
                ]),
            ]), className='my-3'),
    ], title='Avvio guidato su un testo pubblico (prova rapida di funzionamento)'),
    dbc.AccordionItem([
            dbc.Row([
                dbc.Col([html.H5('1 · Il tuo testo (facoltativo)'), html.Small('Carica i tuoi documenti al passo 1 della pagina, «I tuoi documenti».', className='text-muted d-block'), html.P('Testo (.txt, .md, .py), .jsonl con un campo «text» per riga, documenti e archivi (.docx, .pdf, .epub, .zip), immagini (.png, .jpg, .webp, .bmp), suoni (.wav, .mp3, .ogg, .flac, .m4a) e video (.gif, .mp4, .webm, .mov): tutto diventa token da solo; per audio e video compressi il worker scarica ffmpeg la prima volta che serve. Le immagini diventano 32×32 a 256 colori, i suoni 8 kHz, le animazioni 16×16 a 4 fotogrammi al secondo.', className='small text-muted'), dcc.Dropdown(id='lm-corpus', placeholder='Scegli un testo già caricato')], md=6),
                dbc.Col([html.H5('2 · Il progetto'),
                         html.Small('Il modello a concetti: nessuna dimensione da scegliere, cresce con il testo che conta.', className='text-muted d-block mb-2'),
                         dbc.Label('Nome del progetto'), dbc.Input(id='lm-name', value='Modello a concetti'),
                         html.Div([dcc.Dropdown(id='lm-base', options=NATIVE_SIZES, value='micro'),
                                   dbc.Input(id='lm-steps', type='number', value=1000)], style={'display': 'none'})], md=6),
            ]),
            dbc.Accordion([dbc.AccordionItem([
                html.P("I valori predefiniti vanno bene per una prova. Non si possono cambiare dopo l'avvio del progetto.", className='small text-muted'),
                dbc.Row([dbc.Col([dbc.Label(label), dbc.Input(id=id, type='number', value=value, min=lo, max=hi),html.Small(TRAINING_HELP.get(id,''),className='text-muted')], md=4, className='mb-2')
                         for label,id,value,lo,hi in [('Lunghezza dei pezzi di prova (token)','lm-length',256,16,2048),
                             ('Salva il modello ogni … passi','lm-save',100,1,10000),('Pezzi di testo per ogni controllo','lm-validation',8,1,64)]]),
                # Kept for callbacks shared with earlier projects (networks): the concept model has no such settings.
                html.Div([dbc.Input(id=id, type='number', value=value) for id, value in (('lm-batch', 1), ('lm-accum', 8), ('lm-lr', .001), ('lm-slots', 16),
                                                                                        ('lm-window', 16), ('lm-patience', 0), ('lm-min-delta', .01))]
                         + [dbc.Input(id='lm-cp-weight',type='number',value=0.), dbc.Input(id='lm-rank',type='number',value=8),
                            dcc.Dropdown(id='lm-routing',value='concept-pairs-v1',options=[{'label':'concept-pairs-v1','value':'concept-pairs-v1'}]),
                            dcc.Dropdown(id='lm-fraction',value=1.,options=[{'label':'1','value':1.}]), dbc.Checkbox(id='lm-distill', value=False),
                            dcc.Dropdown(id='lm-device', options=[{'label':'Processore (CPU)','value':'cpu'}], value='cpu')],
                         style={'display':'none'}),
                html.P('Il testo è letto come una tabella: ogni posizione è una riga, i concetti dei tratti che la precedono (2, 4, 8… token) sono le colonne note, il token che segue è la colonna da completare.', className='small text-muted'),
                html.P("Il testo viene diviso in tre parti che non si mescolano: una per imparare, una per controllare i progressi e scegliere i pesi, una per la verifica finale. Viene conservata sempre la copia migliore.", className='small text-muted'),
                html.P("Il lavoro viene salvato ogni 5 minuti e a ogni pausa: un arresto improvviso fa perdere al massimo i passi dall'ultimo salvataggio.", className='small text-muted'),
            ], title='Impostazioni avanzate')], start_collapsed=True, className='my-3'),
            dbc.Button("Crea e avvia l'addestramento", id='lm-start', className='m-1'),
    ], title='Crea un progetto a mano, con tutte le impostazioni'),
    dbc.AccordionItem([
            html.Hr(),html.H5("5 · Confronto con una rete neurale (facoltativo)"),
            html.P("Per misurare il modello a concetti: una rete neurale piccola nata da zero, sullo stesso testo e con lo stesso tokenizer del progetto scelto. "
                   "Compare sullo stesso grafico, tratteggiata. È solo un riferimento: il modello del progetto resta quello a concetti.", className='small'),
            dbc.Button("Avvia il confronto", id='lm-dense-control', color='primary'),html.Div(id='lm-control-status',role='status'),
            html.Div([dbc.Checklist(id='lm-series-variants', value=[], options=[]), dbc.Button(id='lm-series-start'), html.Div(id='lm-series-status')],
                     style={'display': 'none'}),
    ], title='Confronto con una rete neurale'),
    dbc.AccordionItem([
            html.Hr(),html.H5('6 · Verifica finale su testo mai visto'),
            html.P("Con l'apprendimento in pausa: misura la copia scelta sul testo di verifica, mai usato per imparare né per scegliere i pesi: errore, velocità di scrittura e memoria."),
            dbc.Label('Ampiezza della verifica'),
            dcc.Dropdown(id='lm-eval-profile',value='quick',clearable=False,options=[{'label':'Rapida (CPU): 4 pezzi di testo, contesti di 32 e 128 token','value':'quick'},{'label':'Contesto lungo (CPU, lenta): 1 pezzo, contesti di 128 e 1.024 token','value':'long'},{'label':'Memoria e velocità su testi lunghi (CPU, molto lenta): 512, 2.048 e 4.096 token','value':'memory'},{'label':'Ampia: richiede una GPU','value':'wide','disabled':True}]),
            html.Small('«Testi lunghi» mostra da quale lunghezza i concetti fanno risparmiare memoria: sui testi brevi occupano più dell\'attenzione normale. Oltre la lunghezza del testo di verifica il testo viene ripetuto, perché lì si misurano solo velocità e memoria.', className='text-muted d-block mb-2'),
            dbc.Button('Avvia la verifica', id='lm-evaluate', className='m-1'),
            dbc.Button('Ferma la verifica',id='lm-eval-stop',outline=True,color='warning',className='m-1'),
            dbc.Button('Scarica i risultati della verifica', id='lm-eval-download', outline=True, color='primary', className='m-1'),dcc.Download(id='lm-eval-file'),
            html.Div(id='lm-eval-status',role='status'),html.Div(id='lm-eval-view'),
    ], title='Verifica finale e memoria su testi lunghi'),
    dbc.AccordionItem([
            dbc.Accordion([dbc.AccordionItem([html.Pre(id='lm-debug-view', style={'whiteSpace':'pre-wrap'}),
                dbc.Button('Scarica la diagnostica', id='lm-debug-download', outline=True, color='primary'), dcc.Download(id='lm-debug-file'),
                dcc.Store(id='lm-debug-data')], title='Per esperti · configurazione e registro del progetto')], start_collapsed=True),
            html.Div(            dbc.Accordion([dbc.AccordionItem([
                html.P('Fotografa cosa succede dentro ogni layer, senza modificare il modello: durante il prossimo passo di addestramento, oppure mentre scrive (casella nella sezione 4). I concetti sono riassunti del testo letto, non parole con un significato preciso.', className='small text-muted'),
                dbc.Button('Fotografa il prossimo passo di addestramento', id='lm-layer-capture', outline=True, color='primary'),
                html.Div(id='lm-layer-request-status', role='status'), html.Div(id='lm-layer-status'),
                dcc.Store(id='lm-training-layers'), dcc.Store(id='lm-inference-layers'),
                dcc.Dropdown(id='lm-layer-source', value='training', clearable=False, options=[{'label':"Ultima foto durante l'addestramento",'value':'training'},{'label':'Ultima foto mentre scriveva','value':'inference'}]),
                dcc.Dropdown(id='lm-layer-frame', placeholder='Scegli quale parola scritta guardare'),
                html.Div(id='lm-layer-view'),
                dbc.Button('Scarica la foto dei layer', id='lm-layer-download', outline=True, color='primary', className='mt-2'), dcc.Download(id='lm-layer-file'),
            ], title='Per esperti · dentro il modello, layer per layer')], start_collapsed=True, className='my-3'), style={'display': 'none'}),   # layers exist only in the networks of earlier projects
    ], title='Per esperti: configurazione e registro'),
    dbc.AccordionItem([
            dbc.Button('Salva la chiave di recupero', id='lm-export-key', outline=True, color='secondary', className='my-2'),
            html.Div(id='lm-recovery-input'), dcc.Download(id='lm-key-download'),
            html.P('I tuoi progetti restano legati a questo browser. Per ritrovarli da un altro browser o computer salva la chiave di recupero: chi la possiede può gestirli. Non serve un account.', className='small text-muted'),
    ], title='Ritrovare i progetti da un altro browser'),
    ], start_collapsed=True, always_open=True, className='my-3', id='lm-advanced'),
])


def native_config(corpus, size, name, steps, length, batch, accumulation, lr, slots, window, save_steps, device, patience,
                  min_delta, validation):
    """A project of the manual form: the concept model (the network settings of earlier projects are ignored)."""
    return dict(corpus_id=corpus, name=name, sequence_length=int(length or 256), save_steps=int(save_steps or 100),
                protocol='controlled-v1', architecture='concepts', routing_profile='concept-pairs-v1', vocabulary=32000,
                distillation_weight=0., early_stop_patience=0, validation_batches=int(validation or 8), device='cpu',
                chat_probes=True)


def media_view(text, small=False):
    """Generated text with its images, sounds and animations shown, not as tokens."""
    import base64
    try:
        from acan.lm.media import parts
        pieces = parts(text or '')
    except Exception:                                   # a broken medium is still shown as text
        pieces = [('text', text or '')]
    shown = []
    for kind, value in pieces:
        if kind == 'text':
            if value.strip():
                shown.append(html.Pre(value, className='small mb-1' if small else 'mb-1', style={'whiteSpace': 'pre-wrap'}))
            continue
        data = base64.b64encode(value).decode()
        if kind == 'audio':
            shown.append(html.Audio(src='data:audio/wav;base64,'+data, controls=True, className='d-block my-1'))
        else:
            shown.append(html.Img(src=('data:image/png;base64,' if kind == 'image' else 'data:image/gif;base64,')+data,
                                  style={'imageRendering': 'pixelated', 'width': '128px' if small else '256px'}, className='d-block my-1'))
    return html.Div(shown)


def inspection_panel(report, frame=0):
    if not report or frame is None:
        return html.P((report or {}).get('error') or 'Richiedi una cattura durante il training oppure abilita l’ispezione prima di generare. Poi scegli la sorgente qui sopra.')
    frames = report.get('frames', [])
    if not isinstance(frame, int) or not 0 <= frame < len(frames):
        return html.P(report.get('error', 'Nessuna cattura disponibile.'))
    sample = frames[frame]
    rows, masses = sample['layers'], [r for r in sample['layers'] if 'mass_fractions' in r]
    gradients = report.get('gradients', {})
    header = html.Tr([html.Th(x) for x in ['Layer', 'Tipo', 'RMS ingresso', 'RMS uscita', 'RMS variazione', 'Gradiente prima del clipping', 'Slot effettivi', 'Anomalie']])
    def number(value):
        return f'{value:.4g}' if value is not None else '—'
    body = []
    for row in rows:
        grad = gradients.get(str(row['layer']), {})
        warnings = [row.get('warning', '')]
        if not all(row[key]['finite'] for key in ('input','output','change')) or not row.get('slots_finite',True) or not grad.get('finite',True):
            warnings.append('Valori non finiti: verificare stabilità numerica.')
        body.append(html.Tr([html.Td(x) for x in [row['layer'], row['kind'], number(row['input']['rms']), number(row['output']['rms']), number(row['change']['rms']), number(grad.get('norm')), number(row.get('effective_slots')), ' '.join(warnings)]]))
    panels = [html.P(f"Passo di training: {report.get('step')} · {report['mode']}. RMS misura l’ampiezza dei segnali; slot effettivi misura la distribuzione della massa, non il numero di significati."),
              dbc.Table([html.Thead(header), html.Tbody(body)], responsive=True, bordered=True, size='sm')]
    if masses:
        figure = go.Figure(go.Heatmap(z=[r['mass_fractions'] for r in masses], y=[f"L{r['layer']}" for r in masses], x=list(range(len(masses[0]['mass_fractions']))), zmin=0,zmax=1,colorbar=dict(title='Frazione')))
        figure.update_layout(title='Quota di massa per slot nel prefisso letto', xaxis_title='Slot (identità locale al layer)', yaxis_title='Layer')
        panels.append(dcc.Graph(figure=figure))
    panels.append(html.P(f"Token letti in questo forward: {sample.get('input_tokens', '—')} · prefisso accumulato: {sample.get('prefix_tokens', '—')}."))
    if 'last_input_token' in sample:
        panels.append(html.P(f"Ultimo token in ingresso: {sample['last_input_token']!r}"))
    candidates = sample.get('next_token_candidates', [])
    if candidates:
        panels.append(html.P('Candidati per il token successivo: '+', '.join(f"{c['token'] if c['token'] is not None else c['token_id']!r}: {c['probability']:.1%}" for c in candidates)+'. Sono probabilità proposte; il campionamento può scegliere un altro token.'))
    panels.extend([html.P(report['scope'],className='small text-muted'), html.P(report['limits'],className='small text-muted')])
    return panels


CONTROL_PREFIX = 'Controllo denso · '


SAME_RUN = ('corpus_id', 'base_model', 'total_steps', 'sequence_length', 'batch_size', 'accumulation', 'learning_rate',
            'slots', 'window', 'lora_rank', 'graft_fraction', 'seed')


def paired_job(jobs, job):
    """The dense control of a concept project, or the project of a dense control.

    A control started by an up-to-date worker names its project (``control_of``). Older ones
    are recognized by their name and by every setting they copied from the project; when several
    projects share all of them, nothing is paired rather than a wrong one.
    """
    config = job['config']
    if config.get('architecture') == 'dense-control' and config.get('control_of'):
        return next((other for other in jobs if other['id'] == config['control_of']), None)
    if config.get('architecture') != 'dense-control':
        linked = [other for other in jobs if other['config'].get('control_of') == job['id']]
        if linked:
            return linked[0]
    same = lambda other: all(other['config'].get(key) == config.get(key) for key in SAME_RUN)
    if config.get('architecture') == 'dense-control':
        name = config.get('name', '')[len(CONTROL_PREFIX):] if config.get('name', '').startswith(CONTROL_PREFIX) else None
        found = [other for other in jobs if other['id'] != job['id'] and other['config'].get('architecture') != 'dense-control'
                 and other['config'].get('name') == name and same(other)]
    else:
        found = [other for other in jobs if other['config'].get('architecture') == 'dense-control'
                 and not other['config'].get('control_of')
                 and other['config'].get('name') == CONTROL_PREFIX+config.get('name', '') and same(other)]
    return found[0] if len(found) == 1 else None


def unique_name(name, jobs):
    """A new project never takes the name of an existing one: the date and time are added."""
    import datetime
    name = (name or 'Progetto').strip()
    if any(job['config'].get('name') == name for job in jobs):
        name = f"{name} · {datetime.datetime.now().strftime('%d/%m %H:%M')}"
    return name


def project_label(job, jobs):
    """Name, state and step; projects with the same name are told apart by a short code."""
    name = job['config'].get('name', 'Progetto')
    label = f"{name} · {job['status']} · passo {job['progress'].get('step', 0)}"
    if sum(other['config'].get('name') == name for other in jobs) > 1:
        label += f" · #{job['id'][:4]}"
    return label


def verified(evaluations):
    """Error on the test text of the latest verification, if any."""
    for item in reversed(evaluations or []):
        quality = ((item.get('records') or {}).get('selected_checkpoint') or {}).get('quality') or {}
        if quality.get('nll') is not None:
            return quality['nll'], quality.get('tokens')
    return None


def comparison_summary(concepts, dense, concept_tests=None, dense_tests=None):
    """Best result of the concept project against its dense control, and what it means.

    The verification on the test text is used when both have one: it reads more text than the
    validation during training.
    """
    tested = verified(concept_tests), verified(dense_tests)
    if all(tested):
        (a, tokens), (b, _) = tested
        gap = a-b
        finished = all(job['status'] in ('paused', 'completed') for job in (concepts, dense))
        verdict, color = _verdict(gap)
        return (f"Sul testo di verifica ({tokens:,} token) · concetti {a:.3f}, attenzione normale {b:.3f}, differenza {gap:+.3f}: {verdict}. "
                + ('' if finished else 'Almeno uno dei due è ancora in corso: rifai la verifica alla fine. ')), color
    best = lambda job: next((item.get('validation_nll') for item in job.get('snapshots', []) if item.get('best')), None)
    a, b = best(concepts), best(dense)
    if a is None or b is None:
        return None
    finished = all(job['status'] in ('paused', 'completed') for job in (concepts, dense))
    gap = a-b
    verdict, color = _verdict(gap)
    text = (f"Migliore conservato · concetti {a:.3f}, attenzione normale {b:.3f}, differenza {gap:+.3f}: {verdict}. "
            + ('' if finished else 'Confronto provvisorio: almeno uno dei due è ancora in corso. ')
            + 'Il controllo durante l\'addestramento legge poco testo: differenze sotto 0,15 non si distinguono dal caso. '
              'Per una misura più solida avvia la verifica finale (sezione 6) su entrambi i progetti: il confronto la userà da solo.')
    return text, color


def _verdict(gap):
    if abs(gap) <= .15:
        return 'i concetti non costano qualità a questa scala: è il risultato a favore dell\'idea', 'success'
    if gap > .15:
        return f'la compressione costa {gap:.2f} di errore rispetto all\'attenzione normale', 'warning'
    return 'i concetti risultano migliori dell\'attenzione normale: sorprendente, va ripetuto con un altro seme prima di crederci', 'info'


def series_table(jobs, job, tests):
    """All the projects of a series side by side: what changed, how far they got, their best results."""
    series = (job['config'].get('series') or {}).get('id')
    members = sorted((other for other in jobs if (other['config'].get('series') or {}).get('id') == series),
                     key=lambda other: other['config']['series']['order'])
    if not series or len(members) < 2:
        return None
    rows = []
    for other in members:
        best = next((item.get('validation_nll') for item in other.get('snapshots', []) if item.get('best')), None)
        best = best if best is not None else other['progress'].get('best_validation_nll')
        tested = verified(tests.get(other['id']))
        rows.append(html.Tr([html.Td(other['config']['series']['label']), html.Td(other['status']),
                             html.Td(f"{other['progress'].get('step', 0)}/{other['config']['total_steps']}"),
                             html.Td('—' if best is None else f'{best:.3f}'), html.Td('—' if not tested else f'{tested[0]:.3f}')]))
    return html.Div([html.H6('Serie di esperimenti', className='mt-2'),
                     dbc.Table([html.Thead(html.Tr([html.Th(name) for name in ('Variante', 'Stato', 'Passi', 'Miglior errore su testo mai visto (durante l\'addestramento) ↓', 'Errore sul testo di verifica ↓')])),
                                html.Tbody(rows)], size='sm', bordered=True, responsive=True),
                     html.P('I progetti girano uno dopo l\'altro. Per l\'ultima colonna avvia la verifica finale su ciascuno quando ha finito.',
                            className='small text-muted')])


KIND_LABELS = {'dense': 'Rete con attenzione normale', 'window': 'Rete con le sole ultime 16 parole', 'concepts': 'Modello a concetti'}
VALIDATION_NAME = 'Validazione'
# Thresholds of the validation path, explained in docs/CONTROLLED-LM.md.
MEMORY_SHARE = .8          # concepts must keep at least 80% of what ordinary attention recalls
TEXT_MARGIN = .05          # and lose at most 0.05 bits per character on the text
REAL_MARGIN = .15          # on the native micro model, within the noise of the measurement


def _memory_check(name, dense, concepts, window, chance, distance):
    if dense < .5:
        return (name, None, f"Prova non valida: anche l'attenzione normale ricorda solo il {100*dense:.0f}%. Rifalla con la durata completa.")
    passed = concepts >= MEMORY_SHARE*dense and (window is None or concepts > window+.5*(dense-window))
    alone = '' if window is None else f", sola finestra {100*window:.0f}%"
    return (name, passed, f"Risposta a {distance} token di distanza: concetti {100*concepts:.0f}%, attenzione normale "
                          f"{100*dense:.0f}%{alone} (a caso {100*chance:.0f}%). Serve almeno il {100*MEMORY_SHARE:.0f}% dell'attenzione normale.")


def small_checks(result):
    """The criteria of the small test, as (name, passed, explanation); passed is None when the test is not valid."""
    memory, text = result['memory'], result['text']
    setup = result['memory_setup']
    checks = [_memory_check('Memoria', memory['dense']['accuracy'], memory['concepts']['accuracy'], memory['window']['accuracy'],
                            setup['chance'], setup.get('nearest_answer_distance', '?'))]
    far = (result.get('memory_cases') or {}).get('distance')
    if far:
        checks.append(_memory_check('Memoria lontana', far['dense']['accuracy'], far['concepts']['accuracy'], None,
                                    far['chance'], far['nearest_answer_distance']))
    gap = text['concepts']['bits_per_character']-text['dense']['bits_per_character']
    checks.append(('Testo', gap <= TEXT_MARGIN,
                   f"Concetti {text['concepts']['bits_per_character']:.3f} bit per carattere, attenzione normale "
                   f"{text['dense']['bits_per_character']:.3f} (differenza {gap:+.3f}; serve al massimo {TEXT_MARGIN:+.2f})."))
    return checks


def small_facts(result):
    """What the small test measures without a pass mark: the data needed and the cost per token."""
    facts = []
    found = result.get('efficiency')
    if found:
        text, memory = found['text'], found['memory']
        if text.get('ratio'):
            facts.append(f"Dati necessari, testo: per arrivare allo stesso livello i concetti hanno letto "
                         f"{text['characters']['concepts']:,} caratteri, l'attenzione normale {text['characters']['dense']:,} "
                         f"(rapporto {text['ratio']:.2f}; sotto 1 servono meno dati ai concetti, differenze di pochi punti sono rumore).")
        if memory.get('ratio'):
            facts.append(f"Dati necessari, memoria: risposte quasi sempre giuste dopo {memory['steps']['concepts']:,} passi con i "
                         f"concetti e {memory['steps']['dense']:,} con l'attenzione normale (rapporto {memory['ratio']:.2f}).")
        elif memory['steps'].get('dense') and not memory['steps'].get('concepts'):
            facts.append("Dati necessari, memoria: i concetti non sono arrivati a risposte quasi sempre giuste entro la prova; "
                         f"l'attenzione normale sì, dopo {memory['steps']['dense']:,} passi.")
    cost = result.get('cost')
    if cost:
        last = cost['rows'][-1]
        text = (f"Costo: a {last['length']:,} token di contesto l'attenzione normale tiene {last['dense']['kept_vectors']:,} "
                f"vettori per layer e cresce con il testo; il modello a concetti tiene gli ultimi {last['concepts']['kept_vectors']} token e ne legge i concetti a scale che raddoppiano. "
                f"Tempo per token: attenzione normale da {cost['rows'][0]['dense']['milliseconds_per_token']:.2f} a "
                f"{last['dense']['milliseconds_per_token']:.2f} ms, concetti circa {cost['concepts_flat']:.2f} ms fissi.")
        if cost.get('faster_beyond'):
            text += f" Con questa implementazione i concetti diventano più veloci oltre circa {cost['faster_beyond']:,.0f} token."
        facts.append(text)
    return facts


def small_test_view(test):
    """Phase, then the results table and the criteria, of a small test."""
    status = {'running': 'In corso', 'cancelling': 'Arresto in corso', 'completed': 'Finita', 'cancelled': 'Fermata',
              'failed': 'Non riuscita'}.get(test['status'], test['status'])
    request = test.get('request') or {}
    phase = test.get('phase') or ''
    line = f"{status}{' · '+phase if phase and phase not in ('Finita', 'Fermata') else ''} · testo {request.get('corpus_name', '?')} · durata {request.get('budget', '?')}"
    if test['status'] == 'failed':
        return line, dbc.Alert(test.get('error') or 'Errore sconosciuto', color='danger')
    result = test.get('result')
    if not result:
        return line, None
    rows = [html.Tr([html.Td(KIND_LABELS[kind]), html.Td(f"{100*result['memory'][kind]['accuracy']:.1f}%"),
                     html.Td(f"{result['text'][kind]['bits_per_character']:.3f}")]) for kind in ('dense', 'window', 'concepts')]
    table = dbc.Table([html.Thead(html.Tr([html.Th('Modello'), html.Th('Memoria: risposte giuste (più alto è meglio)'),
                                           html.Th('Testo: bit per carattere (più basso è meglio)')])), html.Tbody(rows)],
                      bordered=True, size='sm', className='mt-2')
    cases = result.get('memory_cases') or {}
    if cases:
        names = {'distance': 'Memoria lontana', 'capacity': 'Capacità'}
        table = html.Div([table, dbc.Table([html.Thead(html.Tr([html.Th('Prova difficile'), html.Th('Attenzione normale'), html.Th('Concetti')])),
            html.Tbody([html.Tr([html.Td(f"{names[name]}: {case['pairs']} coppie, risposta a {case['nearest_answer_distance']} token"),
                                 html.Td(f"{100*case['dense']['accuracy']:.1f}%"), html.Td(f"{100*case['concepts']['accuracy']:.1f}%")])
                        for name, case in cases.items()])], bordered=True, size='sm')])
    checks = [html.Li([html.Strong(f"{name}: " + ('superato' if passed else 'non valido' if passed is None else 'non superato') + '. '),
                       text], className='text-success' if passed else 'text-danger' if passed is False else '')
              for name, passed, text in small_checks(result)]
    checks += [html.Li(fact, className='text-muted') for fact in small_facts(result)]
    samples = dbc.Accordion([dbc.AccordionItem([
        html.P('Inizio dato (dal testo mai visto):', className='small mb-0'),
        html.Pre(result['text']['dense']['prompt'], className='small'),
        *[html.Div([html.Strong(KIND_LABELS[kind]), html.Pre(result['text'][kind]['sample'], className='small', style={'whiteSpace': 'pre-wrap'})])
          for kind in ('dense', 'window', 'concepts')]], title='Cosa scrivono i tre modelli (lettere, non parole: è normale che siano strane)'),
        dbc.AccordionItem([html.Ul([html.Li(
            f"{KIND_LABELS[kind]}: sbagliate prendendo un altro valore della sequenza "
            f"{100*result['memory'][kind].get('wrong_but_from_the_sequence', float('nan')):.1f}%, "
            f"sbagliate con un valore che non c'era {100*result['memory'][kind].get('wrong_not_in_the_sequence', float('nan')):.1f}%, "
            f"{result['memory'][kind].get('parameters', 0):,} parametri, {result['memory'][kind]['seconds']:.0f} s")
            for kind in ('dense', 'window', 'concepts')]),
            html.P("Il primo tipo di errore vuol dire che il modello ha trovato le coppie ma ha letto quella sbagliata; il secondo "
                   "che non le ha trovate. Il file scaricato contiene anche le curve di addestramento di ogni modellino.", className='small')],
            title='Per il debug')], start_collapsed=True, className='mt-2')
    return line, html.Div([table, html.Ul(checks), samples])


def validation_pair(jobs):
    """The concept project and the dense one of the latest comparison started from the validation path."""
    series = {}
    for job in jobs:
        item = job['config'].get('series') or {}
        if item:
            series.setdefault(item['id'], {})[item.get('variant')] = job
    # The worker renames the dense member after the concept one, so the name is read on the latter.
    pairs = [found for found in series.values() if 'chosen' in found and 'dense' in found
             and found['chosen']['config'].get('name', '').startswith(VALIDATION_NAME)]
    if not pairs:
        return None
    latest = max(pairs, key=lambda found: max(job.get('created', 0) or 0 for job in found.values()))
    return latest['chosen'], latest['dense']


def decision_view(small, pair=None, tests=None):
    """Passo 3: every criterion, then whether a large GPU training is justified."""
    items, ready = [], True
    if small and small.get('result'):
        for name, passed, text in small_checks(small['result']):
            items.append((f'Passo 1 · {name}', passed, text))
            ready = ready and passed is True
    else:
        items.append(('Passo 1', None, 'Non ancora eseguito (o non finito).'))
        ready = False
    if pair is None:
        items.append(('Passo 2', None, 'Non ancora avviato.'))
        ready = False
    else:
        concepts, dense = pair
        tested = [verified((tests or {}).get(job['id'])) for job in pair]
        finished = all(job['status'] in ('paused', 'completed') for job in pair)
        if not finished:
            items.append(('Passo 2', None, f"In corso: concetti {concepts['status']} al passo {concepts['progress'].get('step', 0)}, "
                                           f"attenzione normale {dense['status']} al passo {dense['progress'].get('step', 0)}."))
            ready = False
        elif not all(tested):
            items.append(('Passo 2', None, "Entrambi finiti: manca la verifica finale su testo mai visto. Selezionali al passo 4 "
                                           "(«Progetto») e avvia «Verifica finale» negli strumenti avanzati, uno per volta."))
            ready = False
        else:
            gap = tested[0][0]-tested[1][0]
            passed = gap <= REAL_MARGIN
            items.append(('Passo 2', passed, f"Sul testo di verifica: concetti {tested[0][0]:.3f}, attenzione normale {tested[1][0]:.3f}, "
                                             f"differenza {gap:+.3f} (serve al massimo {REAL_MARGIN:+.2f})."))
            ready = ready and passed
    marks = {True: '✔', False: '✘', None: '…'}
    listing = html.Ul([html.Li([html.Strong(f"{marks[passed]} {name}: "), text],
                               className='text-success' if passed else 'text-danger' if passed is False else 'text-muted')
                       for name, passed, text in items], className='mb-2')
    failed = any(passed is False for _, passed, _ in items)
    if ready:
        verdict = dbc.Alert("Pronto per la GPU: i concetti hanno superato tutti i criteri a questa scala. Il passo successivo è lo "
                            "stesso confronto con più testo e un modello più grande; i criteri restano gli stessi.", color='success')
    elif failed:
        verdict = dbc.Alert("Non ancora: almeno un criterio non è superato. Un addestramento grande su GPU erediterebbe lo stesso "
                            "limite; prima va migliorato lo strato a concetti e rifatto il passo che non passa.", color='danger')
    else:
        verdict = dbc.Alert('Decisione sospesa: mancano dei risultati.', color='secondary')
    return html.Div([listing, verdict])


def openai_proxy(path):
    """The OpenAI-compatible API of the service (acan/lm/openai_api.py) at /v1 of this site.

    The request goes to the service with the client's own key (Authorization: Bearer acan-...); the service
    token of the installation is never used here. Answers are passed on as they arrive (streaming).
    """
    endpoint = os.environ.get('ACAN_LM_URL', '').rstrip('/')
    if not endpoint:
        return jsonify(error=dict(message='Il servizio dei modelli linguistici non è collegato a questa installazione.',
                                  type='server_error', param=None, code=None)), 503
    if flask_request.content_length and flask_request.content_length > 1024*1024:
        return jsonify(error=dict(message='Richiesta troppo grande.', type='invalid_request_error', param=None, code=None)), 413
    headers = {'Authorization': flask_request.headers.get('Authorization', ''), 'Content-Type': 'application/json'}
    query = ('?'+flask_request.query_string.decode()) if flask_request.query_string else ''
    body = flask_request.get_data() if flask_request.method == 'POST' else None
    try:
        upstream = urlopen(Request(f'{endpoint}/v1/{path}{query}', data=body, headers=headers, method=flask_request.method),
                           timeout=600)
    except HTTPError as exc:
        return Response(exc.read(), status=exc.code, mimetype='application/json')
    except URLError:
        return jsonify(error=dict(message='Servizio dei modelli linguistici non raggiungibile.', type='server_error',
                                  param=None, code=None)), 502
    kind = upstream.headers.get('Content-Type', 'application/json')
    if kind.startswith('text/event-stream'):
        def relay():
            with upstream:
                for line in upstream:
                    yield line
        response = Response(stream_with_context(relay()), mimetype='text/event-stream')
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['X-Accel-Buffering'] = 'no'
        return response
    with upstream:
        return Response(upstream.read(), status=upstream.status, mimetype=kind.split(';')[0])


def register(app):
    long_training.register(app)
    app.server.add_url_rule('/v1/<path:path>', 'openai_proxy', openai_proxy, methods=['GET', 'POST'])

    @app.callback(Output('lm-api-created', 'children'), Output('lm-api-refresh', 'data', allow_duplicate=True),
                  Input('lm-api-create', 'n_clicks'), State('lm-workspace', 'data'), State('lm-api-name', 'value'),
                  prevent_initial_call=True)
    def create_api_key(clicks, workspace, name):
        try:
            made = backend(workspace, '/api-keys', 'POST', dict(name=name or ''))
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update
        return dbc.Alert([html.Div(f"Chiave «{made['name']}» creata. Copiala adesso: non verrà più mostrata."),
                          html.Code(made['key'], className='user-select-all')], color='success'), made['id']

    @app.callback(Output('lm-api-status', 'children'), Output('lm-api-refresh', 'data', allow_duplicate=True),
                  Input('lm-api-revoke', 'n_clicks'), State('lm-workspace', 'data'), State('lm-api-revoke-key', 'value'),
                  prevent_initial_call=True)
    def revoke_api_key(clicks, workspace, key):
        try:
            if not key:
                raise ValueError('Scegli la chiave da revocare.')
            backend(workspace, f'/api-keys/{key}/revoke', 'POST', {})
        except ValueError as exc:
            return dbc.Alert(str(exc), color='warning'), dash.no_update
        return 'Chiave revocata: i programmi che la usano non hanno più accesso.', key+':revoked'

    @app.callback(Output('lm-api-keys', 'children'), Output('lm-api-revoke-key', 'options'),
                  Input('lm-api-refresh', 'data'), Input('lm-workspace', 'data'))
    def show_api_keys(refresh, workspace):
        import datetime
        try:
            keys = backend(workspace, '/api-keys')['keys']
        except ValueError:
            return html.Small('Le chiavi compaiono quando il servizio dei modelli è collegato.', className='text-muted'), []
        if not keys:
            return html.Small('Nessuna chiave ancora.', className='text-muted'), []
        when = lambda value: datetime.datetime.fromtimestamp(value).strftime('%d/%m/%Y %H:%M') if value else '—'
        rows = [html.Tr([html.Td(item['name']), html.Td(when(item['created'])), html.Td(when(item.get('last_used'))),
                         html.Td('revocata' if item['revoked'] else 'attiva')]) for item in keys]
        table = dbc.Table([html.Thead(html.Tr([html.Th('Nome'), html.Th('Creata'), html.Th('Ultimo uso'), html.Th('Stato')])),
                           html.Tbody(rows)], size='sm', bordered=True)
        options = [dict(label=item['name'], value=item['id']) for item in keys if not item['revoked']]
        return table, options
    @app.server.post('/generative-upload/<action>')
    def upload_proxy(action):
        try:
            workspace = flask_request.headers.get('X-ACAN-Workspace')
            if action == 'begin':
                result = backend(workspace, '/corpora', 'POST', flask_request.get_json())
            elif action in ('chunk','finish'):
                identifier = flask_request.headers.get('X-Corpus-ID', '')
                if not re.fullmatch('[0-9a-f]{32}', identifier):
                    raise ValueError('Identificatore corpus non valido.')
                if action == 'chunk':
                    if flask_request.content_length is None or flask_request.content_length > 2*1024*1024:
                        raise ValueError('Chunk troppo grande.')
                    result = backend(workspace, f'/corpora/{identifier}/chunks', 'POST', flask_request.get_data(), True,
                                     {'X-Offset':flask_request.headers.get('X-Offset','0')})
                else:
                    result = backend(workspace, f'/corpora/{identifier}/finish', 'POST')
            else:
                raise ValueError('Operazione non valida.')
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.server.get('/generative-export/<identifier>/<checkpoint>')
    def export_proxy(identifier, checkpoint):
        if not re.fullmatch('[0-9a-f]{32}', identifier) or not re.fullmatch('[a-z0-9-]{1,128}', checkpoint):
            return jsonify(error='Identificatore non valido.'), 400
        workspace = flask_request.headers.get('X-ACAN-Workspace', '')
        if not re.fullmatch('[0-9a-f]{64}', workspace):
            return jsonify(error='Workspace non pronto.'), 400
        endpoint = os.environ.get('ACAN_LM_URL', '').rstrip('/')
        token = os.environ.get('ACAN_LM_SERVICE_TOKEN', '')
        if not endpoint or not token:
            return jsonify(error='Worker non configurato.'), 400
        try:
            upstream = urlopen(Request(endpoint+f'/jobs/{identifier}/export?checkpoint={checkpoint}',
                headers={'Authorization':'Bearer '+token, 'X-ACAN-Workspace':workspace}), timeout=60)
        except (HTTPError, URLError):
            return jsonify(error='Export non disponibile: aggiorna il worker e verifica il checkpoint.'), 400
        def chunks():
            try:
                while True:
                    chunk = upstream.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()
        return Response(stream_with_context(chunks()), mimetype='application/zip',
                        headers={'Cache-Control':'no-store', 'Content-Disposition':'attachment; filename="acan-model.zip"'})

    app.clientside_callback("""async function(click, workspace, job, checkpoint) {
        if (!click) return window.dash_clientside.no_update;
        if (!workspace || !job || !checkpoint) return 'Scegli un progetto e un checkpoint.';
        try {
            const response = await fetch('/generative-export/'+encodeURIComponent(job)+'/'+encodeURIComponent(checkpoint),
                {headers:{'X-ACAN-Workspace':workspace}});
            if (!response.ok) throw new Error((await response.json()).error);
            const url = URL.createObjectURL(await response.blob());
            const link = document.createElement('a'); link.href=url; link.download='acan-model-'+checkpoint+'.zip';
            document.body.appendChild(link); link.click(); link.remove(); setTimeout(()=>URL.revokeObjectURL(url),60000);
            return 'Scaricato: pesi appresi, struttura e riferimento esatto alla base. I pesi originali saranno scaricati al primo caricamento.';
        } catch(error) {return 'Export non disponibile: '+error.message;}
    }""", Output('lm-export-status','children'), Input('lm-export-model','n_clicks'),
       State('lm-workspace','data'), State('lm-job','value'), State('lm-checkpoint','value'), prevent_initial_call=True)

    app.clientside_callback('''function(tab, workspace) {
        if (tab !== 'Generative') return [window.dash_clientside.no_update, true, window.dash_clientside.no_update];
        if (!workspace) {
            const bytes = new Uint8Array(32); crypto.getRandomValues(bytes);
            workspace = Array.from(bytes, x=>x.toString(16).padStart(2,'0')).join('');
        }
        window.acanLmWorkspace = workspace;
        return [workspace, false, Date.now()];
    }''', Output('lm-workspace','data'), Output('lm-poll','disabled'), Output('lm-browser-ready','data'),
       Input('tabs','value'), State('lm-workspace','data'))

    @app.callback(Output('lm-state','data'), Output('lm-connection','children'),
                  Input('lm-poll','n_intervals'), Input('lm-refresh','n_clicks'), Input('lm-workspace','data'), Input('lm-upload-state','data'), Input('lm-active-job','data'),
                  State('lm-job','value'),
                  State('tabs','value'))
    def poll(interval, refresh, workspace, upload, active, selected, tab):
        if tab != 'Generative' or not workspace:
            return dash.no_update, dash.no_update
        try:
            state = backend(workspace, '/state?'+urlencode(dict(summary=1,selected=selected or '')))
            try:
                state['public_catalog'] = backend(workspace, '/public-corpora')
            except ValueError:
                state['public_catalog'] = {}
            if dash.ctx.triggered_id == 'lm-active-job':
                state['select_job'] = active
            capabilities = backend(workspace, '/capabilities')
            state['capabilities']=capabilities
            hardware = 'CUDA disponibile' if capabilities['cuda'] else 'CPU'
            upgrade=' Per il caricamento rapido dei progetti aggiorna anche il worker.' if 'detail_job' not in state else ''
            return state, f'Worker collegato · {hardware} · persistenza attiva · una sola operazione alla volta.'+upgrade
        except ValueError as exc:
            return None, dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-corpus','options'), Output('lm-job','options'),
                  Output('lm-corpus','value'), Output('lm-job','value'),
                  Input('lm-state','data'), State('lm-active-job','data'),
                  State('lm-upload-state','data'), State('lm-corpus','value'), State('lm-job','value'))
    def options(state, active, upload, corpus, selected):
        if not state:
            return [], [], None, None
        corpora = [item for item in state['corpora'] if item['status']=='ready']
        corpus_ids = {item['id'] for item in corpora}
        uploaded = (upload or {}).get('id')
        chosen_corpus = corpus if corpus in corpus_ids else uploaded if uploaded in corpus_ids else None
        job_ids = {item['id'] for item in state['jobs']}
        chosen_job = state.get('select_job')
        if chosen_job not in job_ids:
            chosen_job = selected if selected in job_ids else active if active in job_ids else None
        return ([dict(label=f"{item['name']} ({item['received']/1048576:.1f} MiB)", value=item['id']) for item in corpora],
                [dict(label=project_label(item, state['jobs']),value=item['id']) for item in state['jobs']],
                chosen_corpus, chosen_job if chosen_job!=selected else dash.no_update)

    @app.callback(Output('lm-base','options'),Output('lm-eval-profile','options'),
                  Output('lm-length','max'),Output('lm-batch','max'),Output('lm-slots','max'),Output('lm-window','max'),Output('lm-rank','max'),
                  Input('lm-state','data'),Input('lm-fraction','value'))
    def available_models(state,fraction):
        cuda=(state or {}).get('capabilities',{}).get('cuda',False)
        return ([dict(item, disabled=item['value']=='base' and not cuda) for item in NATIVE_SIZES],
                [dict(label='Rapida (CPU): 4 pezzi di testo, contesti di 32 e 128 token',value='quick'),dict(label='Contesto lungo (CPU, lenta): 1 pezzo, contesti di 128 e 1.024 token',value='long'),
                 dict(label='Memoria e velocità su testi lunghi (CPU, molto lenta): 512, 2.048 e 4.096 token',value='memory'),dict(label='Ampia: 64 pezzi, contesti fino a 2.048 token' + ('' if cuda else ' · richiede una GPU'),value='wide',disabled=not cuda)],
                2048,8 if cuda else 4,64 if cuda else 32,128 if cuda else 32,64)

    @app.callback(Output('lm-length','value'),Output('lm-slots','value'),Input('lm-routing','value'),
                  State('lm-length','value'),State('lm-slots','value'))
    def routing_budget(profile,length,slots):
        if profile in ('cp-routing-v1','cp-joint-v1'):
            return min(length or 32,32),min(slots or 8,8)
        return dash.no_update,dash.no_update

    @app.callback(Output('lm-pause','disabled'),Output('lm-resume','disabled'),
                  Output('lm-evaluate','disabled'),Output('lm-dense-control','disabled'),
                  Input('lm-state','data'),Input('lm-job','value'))
    def allowed_actions(state,selected):
        job=next((item for item in (state or {}).get('jobs',[]) if item['id']==selected),None)
        if not job:return True,True,True,True
        active=job['status'] in ('queued','preparing','running','pausing')
        busy=(state or {}).get('busy',False)
        controlled=job['config'].get('protocol')=='controlled-v1'
        return (not active,active or job['status']=='completed' or busy,
                not controlled or active or busy or not job.get('snapshots'),
                not controlled or busy or job['config'].get('architecture')=='dense-control')

    @app.callback(Output('lm-public-dataset', 'options'), Output('lm-public-preset', 'options'),
                  Output('lm-public-info', 'children'), Output('lm-public-start', 'disabled'),
                  Input('lm-state', 'data'), Input('lm-public-dataset', 'value'))
    def public_info(state, selected):
        catalog = (state or {}).get('public_catalog', {})
        datasets, presets = catalog.get('datasets', []), catalog.get('presets', [])
        item = next((item for item in datasets if item['id'] == selected), None)
        description = 'Il servizio di addestramento non è collegato o va aggiornato: i testi pubblici non sono disponibili.'
        if item:
            description = html.Div([html.P(item['note']),
                html.A('Fonte del testo', href=item['source'], target='_blank', rel='noopener noreferrer'),
                html.Span(' · '), html.A(item['license'], href=item['license_url'], target='_blank', rel='noopener noreferrer'),
                html.P(f"Download: {item['bytes']/1048576:.2f} MiB; integrità verificata con SHA-256. La prova breve verifica il percorso, non certifica la qualità linguistica."),
                html.P('Worker occupato: metti in pausa il progetto attivo prima di avviare questa prova.') if state.get('busy') else None])
        return ([dict(label=item['title'], value=item['id']) for item in datasets] if datasets else dash.no_update,
                [dict(label=item['title'], value=item['id']) for item in presets] if presets else dash.no_update,
                description, not item or bool((state or {}).get('busy')))

    @app.callback(Output('lm-active-job', 'data', allow_duplicate=True),
                  Output('lm-public-status', 'children'), Output('lm-prompt', 'value'),
                  Input('lm-public-start', 'n_clicks'), State('lm-workspace', 'data'),
                  State('lm-public-dataset', 'value'), State('lm-public-preset', 'value'),
                  prevent_initial_call=True)
    def public_start(clicks, workspace, dataset, preset):
        if not clicks:
            return dash.no_update, dash.no_update, dash.no_update
        try:
            result = backend(workspace, '/public-experiments', 'POST', dict(dataset=dataset, preset=preset), timeout=90)
            return result['job']['id'], result['message'], result['prompt']
        except ValueError as exc:
            return dash.no_update, dbc.Alert(str(exc), color='warning'), dash.no_update

    @app.callback(Output('lm-active-job','data'), Output('lm-action-status','children'),
                  Input('lm-start','n_clicks'), Input('lm-pause','n_clicks'), Input('lm-resume','n_clicks'),
                  State('lm-workspace','data'), State('lm-job','value'), State('lm-corpus','value'),
                  State('lm-base','value'), State('lm-name','value'), State('lm-steps','value'),
                  State('lm-length','value'), State('lm-batch','value'), State('lm-accum','value'),
                  State('lm-lr','value'), State('lm-slots','value'), State('lm-window','value'),
                  State('lm-rank','value'), State('lm-save','value'), State('lm-device','value'), State('lm-distill','value'), State('lm-fraction','value'), State('lm-patience','value'), State('lm-min-delta','value'), State('lm-routing','value'), State('lm-cp-weight','value'), State('lm-state','data'), State('lm-validation','value'), prevent_initial_call=True)
    def control(start, pause, resume, workspace, job, corpus, base, name, steps, length, batch, accumulation, lr, slots, window, rank, save_steps, device, distill, fraction, patience, min_delta, routing, cp_weight, state=None, validation=8):
        try:
            action = dash.ctx.triggered_id
            if action == 'lm-start':
                name = unique_name(name, (state or {}).get('jobs', []))
                config = native_config(corpus, base, name, steps, length, batch, accumulation, lr, slots, window, save_steps,
                                       device, patience, min_delta, validation)
                job = backend(workspace, '/jobs', 'POST', config)['id']
                backend(workspace, f'/jobs/{job}/start', 'POST')
                return job, f'Addestramento di «{name}» avviato; puoi chiudere la pagina.'
            if not job:
                raise ValueError('Seleziona un progetto.')
            backend(workspace, f"/jobs/{job}/{'pause' if action=='lm-pause' else 'start'}", 'POST')
            return job, 'Pausa richiesta: salvataggio al confine del prossimo passo completo.' if action=='lm-pause' else 'Ripresa richiesta: verrà verificato il checkpoint prima di continuare.'
        except ValueError as exc:
            return dash.no_update, str(exc)

    app.clientside_callback("""function(change, figure, previous) {
        if (!change || !figure || !figure.layout || !figure.layout.uirevision) return window.dash_clientside.no_update;
        let range = change['xaxis.range'] || [change['xaxis.range[0]'],change['xaxis.range[1]']];
        if (change['xaxis.autorange'] === true) range = null;
        else if (!range.every(Number.isFinite)) return window.dash_clientside.no_update;
        const views = Object.assign({},previous || {});
        views[figure.layout.uirevision] = range;
        while (Object.keys(views).length > 32) delete views[Object.keys(views)[0]];
        return views;
    }""",Output('lm-chart-view','data'),Input('lm-loss','relayoutData'),
       State('lm-loss','figure'),State('lm-chart-view','data'))

    @app.callback(Output('lm-history-start','value'),Input('lm-history-prev','n_clicks'),
                  Input('lm-history-next','n_clicks'),Input('lm-job','value'),State('lm-history-start','value'),prevent_initial_call=True)
    def history_navigation(previous,next_,job,start):
        if dash.ctx.triggered_id=='lm-job':return 0
        return max(0,int(start or 0)+(500 if dash.ctx.triggered_id=='lm-history-next' else -500))

    @app.callback(Output('lm-history-start','disabled'),Output('lm-history-prev','disabled'),
                  Output('lm-history-next','disabled'),Input('lm-history-mode','value'))
    def history_controls(mode):
        return (mode!='window',)*3

    @app.callback(Output('lm-inside','figure'), Output('lm-inside-box','style'),
                  Input('lm-state','data'), Input('lm-job','value'))
    def inside(state, selected):
        job = next((j for j in (state or {}).get('jobs', []) if j['id'] == selected), None)
        trace = ((job or {}).get('progress') or {}).get('network_trace') or []
        figure = inside_figure(trace)
        return (figure, {'display': 'block'}) if figure is not None else (go.Figure(), {'display': 'none'})

    @app.callback(Output('lm-progress','children'), Output('lm-loss','figure'),
                  Output('lm-checkpoint','options'), Output('lm-checkpoint','value'),
                  Input('lm-state','data'), Input('lm-job','value'), State('lm-checkpoint','value'),
                  Input('lm-history-mode','value'),Input('lm-history-start','value'),State('lm-workspace','data'),State('lm-chart-view','data'))
    def progress(state, selected, checkpoint, history_mode, history_start, workspace, chart_view):
        job = next((j for j in (state or {}).get('jobs',[]) if j['id']==selected), None)
        if not job:
            return 'Seleziona un progetto.', go.Figure(), [], None
        if 'detail_job' in (state or {}) and state['detail_job']!=selected:
            try:job=backend(workspace,f'/jobs/{selected}/view')
            except ValueError as exc:return dbc.Alert(str(exc),color='warning'),go.Figure(),[],None
        metrics = job['progress']
        figure = go.Figure()
        history=job['history']
        history_error=None
        if history_mode=='overview' and job.get('history_view')!='full_overview':
            history_error='Il worker collegato fornisce lo storico precedente: aggiornalo per accedere all’intero percorso.'
        if history_mode=='window':
            try:history=backend(workspace,f'/jobs/{selected}/history?start={max(0,int(history_start or 0))}&end={max(0,int(history_start or 0))+499}&limit=500&plot=1')['rows']
            except ValueError as exc:history=[];history_error=str(exc)
        own = 'Controllo denso' if job['config'].get('architecture') == 'dense-control' else 'Concetti'
        for key, label in [('training_nll','training'),('validation_nll','validazione')]:
            rows = [row for row in history if key in row]
            figure.add_scatter(x=[row['step'] for row in rows], y=[row[key] for row in rows],
                               mode='lines+markers' if job['config'].get('architecture') == 'concepts' else 'lines', name=f'{own} · {label}')
        trace = metrics.get('network_trace') or []
        for kind in dict.fromkeys(point[2] if len(point) > 2 else None for point in trace):
            # the network inside the steps, on the same validation; in a race of growth every candidate over the same step
            points = [point for point in trace if (point[2] if len(point) > 2 else None) == kind]
            figure.add_scatter(x=[point[0] for point in points], y=[point[1] for point in points], mode='lines',
                               name='Validazione mentre impara' if kind is None else f'Gara · {kind}',
                               line=dict(width=1, dash='dot' if kind is None else 'dash'))
        # The project and its dense control on the same chart: same corpus, seed and budget.
        partner, comparison = paired_job((state or {}).get('jobs', []), job), None
        if partner is not None:
            try:
                other = backend(workspace, f"/jobs/{partner['id']}/view")
            except ValueError:
                other = None
            if other is not None:
                rows = [row for row in other.get('history', []) if 'validation_nll' in row]
                name = 'Controllo denso' if own == 'Concetti' else 'Concetti'
                figure.add_scatter(x=[row['step'] for row in rows], y=[row['validation_nll'] for row in rows], mode='lines',
                                   name=f'{name} · validazione', line=dict(dash='dash', width=3))
                def tests(identifier):
                    try:
                        return backend(workspace, f'/jobs/{identifier}/evaluations')
                    except ValueError:
                        return []
                mine, theirs = tests(job['id']), tests(other['id'])
                pair = (job, other, mine, theirs) if own == 'Concetti' else (other, job, theirs, mine)
                comparison = comparison_summary(*pair)
        figure.update_layout(title='Errore di previsione · più basso è meglio', xaxis_title='Passi', yaxis_title='NLL',uirevision=f'{selected}:{history_mode}:{history_start if history_mode=="window" else 0}',xaxis={'rangeslider':{'visible':True}})
        saved_range=(chart_view or {}).get(figure.layout.uirevision) if isinstance(chart_view,dict) else None
        if isinstance(saved_range,list) and len(saved_range)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in saved_range):
            figure.update_xaxes(range=saved_range,autorange=False)
        protected=next((item for item in job['snapshots'] if item.get('best')),None)
        if protected and protected.get('validation_nll') is not None:
            figure.add_hline(y=protected['validation_nll'],line_dash='dot',line_color='purple',annotation_text='Migliore conservato')
        baseline = metrics.get('baseline_nll')
        if baseline is not None:
            figure.add_hline(y=baseline, line_dash='dash', line_color='green', annotation_text='Prima di leggere il testo' if job['config'].get('architecture') == 'concepts' else 'Modello originale sullo stesso corpus')
        text = f"{job['status']} · passo {metrics.get('step',0)}/{job['config']['total_steps']} · token elaborati {metrics.get('tokens_trained',0):,} · ultimo checkpoint {job['checkpoint'] or 'ancora nessuno'}"
        detail = [dbc.Alert(history_error,color='warning') if history_error else None,html.P(text), html.P(job['error'], className='text-danger') if job['error'] else None]
        if comparison is not None:
            detail.insert(0, dbc.Alert(comparison[0], color=comparison[1]))
        if job['config'].get('series'):
            members = [other for other in (state or {}).get('jobs', []) if (other['config'].get('series') or {}).get('id') == job['config']['series']['id']]
            results = {}
            for other in members:
                try:
                    results[other['id']] = backend(workspace, f"/jobs/{other['id']}/evaluations")
                except ValueError:
                    results[other['id']] = []
            table = series_table(members, job, results)
            if table is not None:
                detail.insert(0, table)
        elif partner is None and job['config'].get('architecture') not in ('dense-control', 'concepts') and job['config'].get('protocol') == 'controlled-v1':
            detail.insert(0, html.P('Per sapere se i concetti funzionano, avvia il controllo denso (in fondo alla pagina): '
                                    'comparirà su questo grafico, tratteggiato, con il confronto dei due risultati.', className='small text-muted'))
        for key,label in [('training_nll','Errore sul testo di addestramento (NLL)'),('training_objective','Obiettivo di addestramento'),('training_cp_penalty','Penalità CP media'),('cp_weight','Peso CP'),('validation_nll','Errore su testo mai visto (NLL di validazione)'),('validation_perplexity','Perplessità: fra quante parole in media il modello è incerto'),
                          ('baseline_nll',"Prima di leggere questo testo (il modello non sa ancora nulla)" if job['config'].get('architecture') == 'concepts' else "Errore al passo 0 (pesi casuali per un modello nativo)"),('tokens_per_second','Velocità (token al secondo)'),('trainable_parameters','Parametri che imparano'),('total_parameters','Parametri totali del modello'),('process_peak_rss_kib','Memoria massima usata (KiB)')]:
            # The CP terms belong to earlier profiles only; the objective then equals the training NLL.
            if key in ('training_objective','training_cp_penalty','cp_weight') and job['config'].get('routing_profile') not in ('cp-routing-v1','cp-joint-v1'):
                continue
            if key == 'process_peak_rss_kib' and metrics.get(key) is not None:
                detail.append(html.P(f"Memoria massima usata: {metrics[key]/1024:,.0f} MB"))
                continue
            if metrics.get(key) is not None:
                detail.append(html.P(f"{label}: {metrics[key]:,.5g}"))
        validation = metrics.get('validation_nll')
        if validation is not None and baseline is not None:
            best = protected.get('validation_nll') if protected else None
            parts = [f'Validazione ultimo passo {validation:.3f}']
            if best is not None:
                parts.append(f"migliore conservato {best:.3f}" + (f" al passo {protected['step']}" if protected.get('step') is not None else ''))
            concepts = job['config'].get('architecture') == 'concepts'
            parts.append(f'prima di leggere questo testo {baseline:.3f}' if concepts else f'modello di partenza non adattato {baseline:.3f}')
            if validation > baseline:
                parts.append('ancora peggiore del modello di partenza')
            control = job['config'].get('architecture') == 'dense-control'
            detail.append(dbc.Alert('; '.join(parts)+'. ' + (
                'Ogni passo conta un milione di token del testo e aggiunge le sue somme; l\'errore è misurato sul testo di validazione, che non viene mai contato. '
                'Il confronto con una rete neurale sugli stessi dati è nella scheda Matematica e in «Confronto con una rete neurale».' if concepts else
                'Questo è il controllo denso: è il riferimento con cui confrontare i progetti ACAN dello stesso budget.' if control else
                'Il modello di partenza non ha visto questo corpus, quindi scendere sotto la sua linea non dimostra che i concetti servano: '
                + ('il confronto valido è con il progetto ad attenzione normale con lo stesso budget, che compare tratteggiato sul grafico.' if partner is not None else
                 'il confronto valido è con il controllo denso addestrato con lo stesso budget (strumenti avanzati, «Confronto con una rete neurale»).')),
                color='info'))
            if best is not None and validation > best + .05:
                detail.append(dbc.Alert(f'La validazione è risalita di {validation-best:.3f} rispetto al migliore mentre il training continua a scendere: '
                                        'il modello sta memorizzando il corpus. Usa il checkpoint migliore conservato.', color='warning'))
        if metrics.get('early_stop_reason'):
            detail.append(dbc.Alert(metrics['early_stop_reason']+' Puoi provare il migliore conservato; riprendi solo per un esperimento consapevole.',color='warning'))
        if metrics.get('dataset'):
            data=metrics['dataset'];unique=data['tokens']['train']
            validation_tokens=min(data['tokens']['validation']//(job['config']['sequence_length']+1),job['config'].get('validation_batches',8))*job['config']['sequence_length']
            detail.append(html.P(f'Campione di validazione per controllo: {validation_tokens} token.'))
            if validation_tokens < 10000:
                detail.append(html.P('Con un campione così piccolo differenze di NLL inferiori a circa 0,15 fra progetti non sono distinguibili dal caso: '
                                     'per confrontare due modelli usare «Confronta sul test», che misura l’intero insieme di test.', className='small text-muted'))
            detail.append(html.P(f"Documenti distinti: {data['documents']}; duplicati rimossi: {data['duplicates']}; passaggi equivalenti sul training: {metrics.get('tokens_trained',0)/max(1,unique):.2f}. Il test è escluso dalla scelta dei checkpoint."))
        concept_layer = job['config'].get('routing_profile') in ('concept-graph-v2','concept-graph-v1','cp-v9')
        for key,label in [('gradient_norm',"Ampiezza dell'ultima correzione (norma del gradiente)"),('acan_gradient_norm','Gradiente ACAN'),('lora_gradient_norm','Correzione degli adattatori (solo progetti precedenti)'),('teacher_kl','Distanza dal modello originale (distillazione)')]:
            if key == 'acan_gradient_norm' and (concept_layer or job['config'].get('architecture') == 'dense-control'):
                continue
            if key == 'teacher_kl' and not job['config'].get('distillation_weight'):
                continue
            if metrics.get(key) is not None:detail.append(html.P(f'{label}: {metrics[key]:.5g}'))
        if concept_layer:
            detail.append(html.P('Lo strato a concetti non ha parametri propri: i concetti si formano dalle chiavi del contesto. '
                                 + ('Il modello è nato da zero: imparano tutti i suoi pesi.' if job['config'].get('architecture')=='native' else
                                    'Progetto precedente: imparano i parametri LoRA di un modello preaddestrato.'),
                                 className='small text-muted'))
        if metrics.get('graft') and job['config'].get('architecture') != 'concepts':
            graft=metrics['graft']
            detail.append(html.P(f"Struttura: {len(graft['replaced_layers'])} layer su {graft['total_layers']} leggono il contesto attraverso i concetti "
                                 f"({graft['slots']} concetti{' per livello' if job['config'].get('routing_profile')=='concept-graph-v2' else ''}, ultimi {graft['window']} token esatti)." if graft['replaced_layers'] else
                                 f"Struttura: tutti i {graft['total_layers']} layer usano l'attenzione normale."))
        last_step = max((s['step'] for s in job['snapshots']), default=0)
        if job['config'].get('architecture') == 'concepts':
            detail.append(html.P(f'Un passo conta un milione di token: il testo intero si legge una volta sola, contarlo di nuovo non insegnerebbe nulla. La linea su testo mai visto si aggiorna a ogni salvataggio (ultimo: checkpoint passo {last_step}).', className='small text-muted'))
        else:
            detail.append(html.P(f'La linea di addestramento oscilla perché ogni passo legge un pezzo di testo diverso. La linea su testo mai visto si aggiorna a ogni salvataggio (ultimo: checkpoint passo {last_step}). I token elaborati possono contare più volte lo stesso testo.', className='small text-muted'))
        available = [dict(label=("Migliore conservato · " if s.get('best') else "")+f"Passo {s['step']}"+(f" · validazione {s['validation_nll']:.3f}" if s.get('validation_nll') is not None else ""), value=s['name']) for s in job['snapshots']]
        selected_snapshot = checkpoint if checkpoint in {s['value'] for s in available} else available[0]['value'] if available else None
        if job['config'].get('architecture') == 'concepts':
            # The concept model: the main numbers as tiles, everything else folded.
            status = {'completed': 'ha finito', 'running': 'sta imparando', 'preparing': 'prepara i token', 'paused': 'in pausa',
                      'queued': 'in partenza', 'failed': 'errore', 'created': 'in attesa'}.get(job['status'], job['status'])
            head = [html.P([html.B(job['config'].get('name', 'Progetto')), f' · {status}'], className='mb-1'),
                    html.P(job['error'], className='text-danger') if job['error'] else None,
                    long_training.concept_summary(metrics, job['config'], status=job['status'])]
            fold = long_training.folded('Tutti i dettagli del progetto', [item for item in detail if item is not None],
                                        'lm-project-details', 'mb-2')
            return head+[fold], figure, available, selected_snapshot
        return detail, figure, available, selected_snapshot

    @app.callback(Output('lm-manage-status','children'),Output('lm-refresh','n_clicks',allow_duplicate=True),
                  Input('lm-rename','n_clicks'),State('lm-rename-value','value'),State('lm-workspace','data'),State('lm-job','value'),
                  State('lm-refresh','n_clicks'),prevent_initial_call=True)
    def rename_project(clicks,name,workspace,job,refresh):
        try:
            if not job:raise ValueError('Scegli prima un progetto.')
            renamed=backend(workspace,f'/jobs/{job}/rename','POST',dict(name=name or ''))
            return f"Progetto rinominato in «{renamed['config']['name']}».",(refresh or 0)+1
        except ValueError as exc:
            return str(exc),dash.no_update

    @app.callback(Output('lm-delete-confirm','displayed'),Input('lm-delete','n_clicks'),State('lm-job','value'),prevent_initial_call=True)
    def ask_delete(clicks,job):
        return bool(job)

    @app.callback(Output('lm-manage-status','children',allow_duplicate=True),Output('lm-refresh','n_clicks',allow_duplicate=True),
                  Output('lm-job','value',allow_duplicate=True),
                  Input('lm-delete-confirm','submit_n_clicks'),State('lm-workspace','data'),State('lm-job','value'),
                  State('lm-refresh','n_clicks'),prevent_initial_call=True)
    def delete_project(confirmed,workspace,job,refresh):
        try:
            if not job:raise ValueError('Scegli prima un progetto.')
            backend(workspace,f'/jobs/{job}/delete','POST',{})
            return 'Progetto eliminato.',(refresh or 0)+1,None
        except ValueError as exc:
            return str(exc),dash.no_update,dash.no_update

    @app.callback(Output('lm-active-job','data',allow_duplicate=True),Output('lm-series-status','children'),
                  Input('lm-series-start','n_clicks'),State('lm-workspace','data'),State('lm-series-variants','value'),
                  State('lm-corpus','value'),State('lm-base','value'),State('lm-name','value'),State('lm-steps','value'),
                  State('lm-length','value'),State('lm-batch','value'),State('lm-accum','value'),State('lm-lr','value'),
                  State('lm-slots','value'),State('lm-window','value'),State('lm-rank','value'),State('lm-save','value'),
                  State('lm-device','value'),State('lm-patience','value'),State('lm-min-delta','value'),State('lm-validation','value'),
                  State('lm-state','data'),prevent_initial_call=True)
    def start_series(clicks,workspace,variants,corpus,base,name,steps,length,batch,accumulation,lr,slots,window,rank,save_steps,
                     device,patience,min_delta,validation,state):
        try:
            if not corpus:raise ValueError('Scegli prima il testo nella sezione 1 (anche uno pubblico già scaricato).')
            config=native_config(corpus,base,unique_name(name,(state or {}).get('jobs',[])),steps,length,batch,accumulation,lr,
                                 slots,window,save_steps,device,patience,min_delta,validation)
            result=backend(workspace,'/series','POST',dict(base=config,variants=list(variants or [])))
            return result['jobs'][0]['id'],f"Serie avviata: {len(result['jobs'])} progetti, uno dopo l'altro. Il primo è già partito."
        except ValueError as exc:
            return dash.no_update,str(exc)

    @app.callback(Output('lm-active-job','data',allow_duplicate=True),Output('lm-control-status','children'),
                  Input('lm-dense-control','n_clicks'),State('lm-workspace','data'),State('lm-job','value'),prevent_initial_call=True)
    def dense_control(clicks,workspace,job):
        try:
            if not job:raise ValueError('Seleziona un progetto controllato.')
            result=backend(workspace,f'/jobs/{job}/dense-control','POST',{})
            return result['id'],'Controllo denso avviato con gli stessi dati, seed e budget; progetto ACAN conservato.'
        except ValueError as exc:
            return dash.no_update,str(exc)

    @app.callback(Output('lm-evaluation-id','data'),Output('lm-eval-status','children'),
                  Input('lm-evaluate','n_clicks'),State('lm-workspace','data'),State('lm-job','value'),
                  State('lm-checkpoint','value'),State('lm-eval-profile','value'),prevent_initial_call=True)
    def start_evaluation(clicks,workspace,job,checkpoint,profile):
        try:
            if not job or not checkpoint:raise ValueError('Seleziona progetto e checkpoint.')
            budgets={'quick':dict(max_batches=4,context_lengths=[32,128],new_tokens=8),'long':dict(max_batches=1,context_lengths=[128,1024],new_tokens=4),
                     'memory':dict(max_batches=1,context_lengths=[512,2048,4096],new_tokens=8),'wide':dict(max_batches=64,context_lengths=[128,512,2048],new_tokens=16)}
            if profile not in budgets:raise ValueError('Budget del confronto non valido.')
            result=backend(workspace,f'/jobs/{job}/evaluate','POST',dict(checkpoint=checkpoint,**budgets[profile]))
            return result['id'],'Confronto avviato nel worker: nessun peso viene aggiornato.'
        except ValueError as exc:
            return None,str(exc)

    @app.callback(Output('lm-eval-status','children',allow_duplicate=True),Input('lm-eval-stop','n_clicks'),
                  State('lm-evaluation-id','data'),State('lm-workspace','data'),prevent_initial_call=True)
    def stop_evaluation(clicks,identifier,workspace):
        if not identifier:return 'Nessun confronto selezionato.'
        try:
            backend(workspace,f'/evaluations/{identifier}/cancel','POST',{})
            return 'Arresto richiesto: attendo la fine del calcolo corrente, senza modificare i pesi.'
        except ValueError as exc:return str(exc)

    @app.callback(Output('lm-eval-view','children'),Output('lm-evaluation-report','data'),
                  Input('lm-poll','n_intervals'),Input('lm-evaluation-id','data'),Input('lm-job','value'),
                  State('lm-workspace','data'))
    def evaluation_status(interval,identifier,job,workspace):
        if not identifier or not workspace:return '',None
        try:
            value=backend(workspace,'/evaluations/'+identifier)
            if value['job']!=job:return '',None
            if value['status']=='cancelled':return 'Confronto annullato.',None
            if value['status']=='failed':return dbc.Alert(value['error'],color='warning'),None
            if value['status']!='completed':return 'Confronto in corso…',None
            report=value['result'];rows=[]
            for key,label in [('original','Modello originale'),('selected_checkpoint','Checkpoint selezionato')]:
                if key not in report['records']:                     # a native model has no original to compare with
                    continue
                q=report['records'][key]['quality']
                rows.append(html.Tr([html.Td(label),html.Td(f"{q['nll']:.4f}"),html.Td(f"{q['perplexity']:.2f}" if q['perplexity'] is not None else 'non finita'),
                    html.Td(f"{q['token_accuracy']:.1%}"),html.Td(q['tokens'])]))
            table=dbc.Table([html.Thead(html.Tr([html.Th(x) for x in ['Modello','Errore (NLL) ↓','Perplessità ↓','Parola successiva indovinata ↑','Token di verifica']])),html.Tbody(rows)],responsive=True)
            details=[]
            for item in report['comparison']:
                memory=item['total_cache_reduction']
                details.append(html.Li(f"Testo di {item['context']} token: lettura iniziale {item['prefill_speedup']:.2f} volte la velocità dell'originale, "
                                       f"scrittura {item['decode_speedup']:.2f} volte; memoria "
                                       + (f"{memory:.0%} in meno." if memory >= 0 else f"{-memory:.0%} in più.")))
            if 'original' not in report['records']:              # a native model: its own numbers, nothing to compare with
                speed=[html.Li(f"Testo di {item['prompt_tokens']} token: lettura {item['prefill_tokens_per_second']:,.0f} token al secondo, "
                               f"scrittura {item['decode_tokens_per_second']:,.1f}; memoria di lettura {item['cache']['total']/1024:,.0f} KB")
                         for item in report['records']['selected_checkpoint']['performance']]
                return [table,html.P('Modello nato da zero: non c\'è un modello originale da confrontare. Il confronto giusto è con il suo '
                                     'controllo ad attenzione normale, verificato allo stesso modo.',className='small'),html.Ul(speed)],report
            saving=[item['context'] for item in report['comparison'] if item['total_cache_reduction']>=0]
            longest=max((item['context'] for item in report['comparison']),default=0)
            verdict=(f"Da {min(saving):,} token in su la copia occupa meno memoria del modello originale." if saving else
                     f"Fino a {longest:,} token la copia occupa ancora più memoria del modello originale: il risparmio arriva con testi più lunghi.")
            return [table,html.P('Velocità e memoria rispetto al modello originale (sopra 1 = più veloce):', className='mb-1'),html.Ul(details),html.P(verdict),
                    html.P('Verifica piccola e su testi brevi: dice se la copia è migliore del modello originale su questo testo, non quanto vale in generale.', className='small text-muted')],report
        except ValueError as exc:return str(exc),None

    @app.callback(Output('lm-eval-file','data'),Input('lm-eval-download','n_clicks'),
                  State('lm-evaluation-report','data'),prevent_initial_call=True)
    def download_evaluation(clicks,report):
        return dict(content=json.dumps(report,indent=2),filename='acan-lm-comparison.json') if report else dash.no_update

    @app.callback(Output('lm-debug-view','children'),Output('lm-debug-data','data'),
                  Input('lm-state','data'),Input('lm-job','value'),State('lm-workspace','data'))
    def debug_project(state,job,workspace):
        if not job or not workspace:
            return 'Seleziona un progetto.',None
        try:
            report=backend(workspace,f'/jobs/{job}/diagnostics')
            return json.dumps(report,indent=2,ensure_ascii=False),report
        except ValueError as exc:
            return str(exc),None

    @app.callback(Output('lm-debug-file','data'),Input('lm-debug-download','n_clicks'),
                  State('lm-debug-data','data'),prevent_initial_call=True)
    def download_debug(clicks,report):
        return dict(content=json.dumps(report,indent=2,ensure_ascii=False),filename='acan-lm-diagnostics.json') if report else dash.no_update

    @app.callback(Output('lm-new-tokens','max'), Output('lm-new-tokens','value'),
                  Input('lm-inspect-generation','value'), State('lm-new-tokens','value'))
    def inspection_budget(enabled, count):
        limit = 32 if enabled else 4096                 # the worker allows more than 256 only to native models
        return limit, min(count, limit) if isinstance(count, (int,float)) else min(64, limit)

    @app.callback(Output('lm-generation-id','data'), Output('lm-generation-status','children'),
                  Input('lm-generate','n_clicks'), State('lm-workspace','data'), State('lm-job','value'),
                  State('lm-checkpoint','value'), State('lm-prompt','value'), State('lm-new-tokens','value'), State('lm-inspect-generation','value'),
                  State('lm-full-length','value'), State('lm-chat','value'), prevent_initial_call=True)
    def start_generation(click, workspace, job, checkpoint, prompt, count, inspect, full_length=True, chat=False):
        try:
            if not job or not checkpoint:
                raise ValueError('Seleziona progetto e checkpoint, poi metti in pausa il training.')
            request = dict(checkpoint=checkpoint,prompt=prompt,max_new_tokens=count,inspect_layers=bool(inspect))
            if chat:
                request['chat'] = True
            elif full_length:
                request['full_length'] = True
            result = backend(workspace, f'/jobs/{job}/generate', 'POST', request)
            return result['id'], 'Generazione avviata dal checkpoint selezionato.'
        except ValueError as exc:
            return None, str(exc)

    @app.callback(Output('lm-output','children'), Output('lm-generation-metrics','children'), Output('lm-inference-layers','data'),
                  Input('lm-poll','n_intervals'), Input('lm-generation-id','data'), Input('lm-job','value'), State('lm-workspace','data'))
    def generation_status(interval, generation, selected, workspace):
        if not generation or not workspace:
            return '', [], None
        try:
            item = backend(workspace, '/generations/'+generation)
            if item['job'] != selected:
                return '', [], None
            if item['status'] == 'failed':
                return item['error'], [], None
            if item['status'] != 'completed':
                return 'Generazione in corso…', [], None
            result = item['result']
            description = f"Copia del passo {result['step']} · {result['generated_tokens']} token scritti · {result['tokens_per_second']:.1f} token al secondo"
            if result.get('concept_model'):                    # the concept model: concepts per scale, no layers
                levels = result.get('concepts_per_level') or []
                memory = (f"Concetti: {sum(levels):,} fra token e tratti di 2, 4, 8… token ("
                          + ' · '.join(f'{count:,}' for count in levels) + f"), {result['cache_total_bytes']/1048576:,.1f} MB di conteggi. "
                          "Quello che il testo in corso ha già detto è ricordato a qualunque distanza.")
                return media_view(result['text']), [html.P(description), html.P(memory)], None
            reduction = result['cache_reduction_fraction']
            memory = (f"Memoria per ricordare il testo letto: {result['cache_dense_layers_bytes']/1048576:.2f} MB nei layer normali, "
                      f"{result['cache_acan_layers_bytes']/1048576:.2f} MB nei layer a concetti. ")
            memory += (f"Rispetto all'attenzione normale si risparmia il {reduction:.0%}." if reduction >= 0 else
                       f"Con un testo così breve i concetti occupano più memoria dell'attenzione normale ({-reduction:.0%} in più): "
                       "il risparmio arriva solo con testi lunghi, quando i token da ricordare superano lo spazio fisso dei concetti.")
            panels = [html.P(description), html.P(memory),
                      dbc.Accordion([dbc.AccordionItem([html.P('Quali concetti e canali interni erano più attivi. Sono riassunti del testo, non parole con un significato preciso.', className='small text-muted'),
                                                        html.Pre(json.dumps(result['explanations'], indent=2))], title='Per esperti · concetti e canali più attivi')], start_collapsed=True)]
            return media_view(result['text']), panels, result.get('layer_inspection')
        except ValueError as exc:
            return str(exc), [], None

    @app.callback(Output('lm-layer-request-status','children'), Input('lm-layer-capture','n_clicks'),
                  State('lm-workspace','data'), State('lm-job','value'), prevent_initial_call=True)
    def capture_layers(clicks, workspace, job):
        try:
            if not job:
                raise ValueError('Seleziona un progetto attivo.')
            backend(workspace, f'/jobs/{job}/layers', 'POST', {})
            return 'Cattura richiesta: attendi il prossimo passo completo di training.'
        except ValueError as exc:
            return str(exc)

    @app.callback(Output('lm-training-layers','data'), Output('lm-layer-status','children'),
                  Input('lm-poll','n_intervals'), Input('lm-job','value'), State('lm-workspace','data'))
    def training_layers(interval, job, workspace):
        if not job or not workspace:
            return None, ''
        try:
            result = backend(workspace, f'/jobs/{job}/layers')
            return result['report'], 'Cattura in attesa del prossimo passo; se in pausa, riprendi il training.' if result['pending'] else ''
        except ValueError as exc:
            return None, str(exc)

    @app.callback(Output('lm-layer-frame','options'), Output('lm-layer-frame','value'),
                  Input('lm-layer-source','value'), Input('lm-training-layers','data'), Input('lm-inference-layers','data'), State('lm-layer-frame','value'))
    def layer_frames(source, training, inference, selected):
        report = training if source == 'training' else inference
        frames = (report or {}).get('frames', [])
        selected = selected if isinstance(selected, int) and 0 <= selected < len(frames) else (0 if frames else None)
        return [dict(label=('Prefill · intero prompt' if i == 0 and source == 'inference' else f'Forward {i}'), value=i) for i in range(len(frames))], selected

    @app.callback(Output('lm-layer-view','children'), Input('lm-layer-frame','value'),
                  Input('lm-layer-source','value'), Input('lm-training-layers','data'), Input('lm-inference-layers','data'))
    def view_layers(frame, source, training, inference):
        return inspection_panel(training if source == 'training' else inference, frame)

    @app.callback(Output('lm-layer-file','data'), Input('lm-layer-download','n_clicks'),
                  State('lm-layer-source','value'), State('lm-training-layers','data'), State('lm-inference-layers','data'), prevent_initial_call=True)
    def download_layers(clicks, source, training, inference):
        report = training if source == 'training' else inference
        return dict(content=json.dumps(report,indent=2,ensure_ascii=False), filename='acan-layer-inspection.json') if report else dash.no_update

    @app.callback(Output('lm-validation-text', 'options'), Output('lm-validation-text', 'value'),
                  Input('lm-state', 'data'), State('lm-validation-text', 'value'))
    def validation_texts(state, current):
        if not state:
            return [], dash.no_update
        own = [dict(label=f"Il tuo: {item['name']} ({item['received']/1048576:.1f} MiB)", value=item['id'])
               for item in state.get('corpora', []) if item['status'] == 'ready']
        public = [dict(label=f"Pubblico: {item['title']}", value='public:'+item['id'])
                  for item in (state.get('public_catalog') or {}).get('datasets', [])]
        options = public+own
        values = {item['value'] for item in options}
        chosen = current if current in values else next((item['value'] for item in public if item['value'] == 'public:wikitext-2'), None)
        return options, chosen if chosen != current else dash.no_update

    def validation_corpus(workspace, value):
        if not value:
            raise ValueError('Scegli prima il testo, in cima al percorso.')
        if value.startswith('public:'):
            return backend(workspace, f"/public-corpora/{value[7:]}/import", 'POST', {}, timeout=600)['corpus']['id']
        return value

    @app.callback(Output('lm-small-id', 'data'), Output('lm-small-status', 'children', allow_duplicate=True),
                  Input('lm-small-start', 'n_clicks'), State('lm-workspace', 'data'), State('lm-validation-text', 'value'),
                  State('lm-small-budget', 'value'), prevent_initial_call=True)
    def start_small_test(clicks, workspace, text, budget):
        try:
            corpus = validation_corpus(workspace, text)
            return backend(workspace, '/small-tests', 'POST', dict(corpus_id=corpus, budget=budget))['id'], 'Prova avviata.'
        except ValueError as exc:
            return dash.no_update, dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-small-status', 'children', allow_duplicate=True), Input('lm-small-stop', 'n_clicks'),
                  State('lm-workspace', 'data'), State('lm-small-id', 'data'), prevent_initial_call=True)
    def stop_small_test(clicks, workspace, identifier):
        try:
            if not identifier:
                raise ValueError('Nessuna prova in corso.')
            backend(workspace, f'/small-tests/{identifier}/cancel', 'POST', {})
            return 'Arresto richiesto: la prova si ferma entro pochi secondi.'
        except ValueError as exc:
            return str(exc)

    @app.callback(Output('lm-small-status', 'children'), Output('lm-small-view', 'children'), Output('lm-small-result', 'data'),
                  Output('lm-small-id', 'data', allow_duplicate=True),
                  Input('lm-small-poll', 'n_intervals'), Input('lm-small-id', 'data'), State('lm-workspace', 'data'),
                  State('lm-small-result', 'data'), State('tabs', 'value'), prevent_initial_call='initial_duplicate')
    def follow_small_test(interval, identifier, workspace, known, tab):
        if tab != 'Generative' or not workspace:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        try:
            if not identifier:                          # after a reload: the latest test of this workspace
                tests = backend(workspace, '/small-tests')
                if not tests:
                    return 'Nessuna prova ancora eseguita.', None, None, dash.no_update
                test, found = tests[0], tests[0]['id']
            else:
                found = dash.no_update
                if known and known.get('id') == identifier and known.get('status') in ('completed', 'failed', 'cancelled'):
                    return dash.no_update, dash.no_update, dash.no_update, dash.no_update
                test = backend(workspace, f'/small-tests/{identifier}')
        except ValueError as exc:
            return str(exc), dash.no_update, dash.no_update, dash.no_update
        line, view = small_test_view(test)
        return line, view, test, found

    @app.callback(Output('lm-small-file', 'data'), Input('lm-small-download', 'n_clicks'), State('lm-small-result', 'data'),
                  prevent_initial_call=True)
    def download_small_test(clicks, test):
        if not test:
            return dash.no_update
        return dict(content=json.dumps(test, indent=2, ensure_ascii=False), filename=f"prova-in-piccolo-{test['id'][:8]}.json")

    @app.callback(Output('lm-active-job', 'data', allow_duplicate=True), Output('lm-validation-status', 'children'),
                  Input('lm-validation-start', 'n_clicks'), State('lm-workspace', 'data'), State('lm-validation-text', 'value'),
                  State('lm-validation-steps', 'value'), State('lm-state', 'data'), prevent_initial_call=True)
    def start_validation(clicks, workspace, text, steps, state):
        try:
            corpus = validation_corpus(workspace, text)
            from .long_training import profile
            cuda = ((state or {}).get('capabilities') or {}).get('cuda', False)
            config = dict(profile(cuda, 'micro'), corpus_id=corpus, name=unique_name(VALIDATION_NAME, (state or {}).get('jobs', [])),
                          total_steps=int(steps or 600), save_steps=50, chat_probes=False, auto_resume=False,
                          early_stop_patience=6, early_stop_min_delta=.005, validation_batches=32)
            result = backend(workspace, '/series', 'POST', dict(base=config, variants=['chosen', 'dense']))
            return result['jobs'][0]['id'], ("Confronto avviato: prima il progetto con i concetti, poi quello con l'attenzione "
                                             "normale, in automatico. Li segui al passo 4, scegliendoli in «Progetto».")
        except ValueError as exc:
            return dash.no_update, dbc.Alert(str(exc), color='warning')

    @app.callback(Output('lm-decision', 'children'), Input('lm-small-result', 'data'), Input('lm-state', 'data'),
                  State('lm-workspace', 'data'))
    def decision(small, state, workspace):
        pair = validation_pair((state or {}).get('jobs', []))
        tests = {}
        if pair is not None and all(job['status'] in ('paused', 'completed') for job in pair):
            for job in pair:
                try:
                    tests[job['id']] = backend(workspace, f"/jobs/{job['id']}/evaluations")
                except ValueError:
                    tests[job['id']] = []
        return decision_view(small, pair, tests)

    @app.callback(Output('lm-key-download','data'), Input('lm-export-key','n_clicks'), State('lm-workspace','data'), prevent_initial_call=True)
    def export_key(click, workspace):
        if not workspace:
            return dash.no_update
        return dict(content=json.dumps({'acan_lm_workspace':workspace}), filename='acan-lm-recovery-key.json')
