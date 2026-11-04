"""Session-scoped experimental character generation, without external services."""
import base64
import json
import threading

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from acan.generative import ConceptLanguageModel, demo_corpus
from acan.paper_contract import GAPS
from acan.vis.jobs import TrainingJobs


layout = dbc.Container([
    html.H3('Laboratorio generativo — prototipo linguistico a caratteri'),
    html.P('Estensione sperimentale CPU: routing competitivo tra concetti e loss del token successivo. '
           'Non è un LLM; audio, video e multimodale non sono implementati.'),
    html.Details([html.Summary('Stato rispetto al paper: conformità parziale'),
                  html.Ul([html.Li(gap['detail']) for gap in GAPS]),
                  html.P('Ogni release allega paper-audit.json con i controlli eseguiti. '
                         'Il superamento dei test non dimostra i teoremi del paper.')]),
    html.H5('Corpus e training'),
    html.P('Una frase/documento per riga. Split 70/15/15 prima delle finestre; vocabolario appreso solo sul training. '
           'Massimo 40.000 caratteri, 126 caratteri distinti e 4.096 finestre per split.'),
    dcc.Upload(id='lab-corpus-upload', children=dbc.Button('Carica testo UTF-8'), max_size=200000),
    html.Div(id='lab-upload-status', role='status'),
    dcc.Textarea(id='lab-corpus', value=demo_corpus(), style={'width': '100%', 'height': '160px'}),
    dbc.Row([
        dbc.Col([dbc.Label('Contesto (caratteri)'), dbc.Input(id='lab-context', type='number', value=6, min=1, max=16)]),
        dbc.Col([dbc.Label('Concetti'), dbc.Input(id='lab-slots', type='number', value=12, min=2, max=64)]),
        dbc.Col([dbc.Label('Epoche massime'), dbc.Input(id='lab-epochs', type='number', value=25, min=1, max=100)]),
        dbc.Col([dbc.Label('Learning rate'), dbc.Input(id='lab-lr', type='number', value=.04, min=.0001, max=.2, step=.01)]),
        dbc.Col([dbc.Label('Seed training'), dbc.Input(id='lab-seed', type='number', value=42, min=0, max=4294967295)]),
    ]),
    dbc.Button('Addestra prototipo', id='lab-train', className='my-2'),
    dbc.Button('Stop training', id='lab-stop', disabled=True, className='m-2'),
    html.Div(id='lab-status', role='status'),
    dcc.Interval(id='lab-monitor', interval=1000, disabled=True),
    dcc.Store(id='lab-model'),
    dbc.Button('Scarica modello JSON', id='lab-save', className='my-2'),
    dcc.Download(id='lab-model-download'),
    dcc.Upload(id='lab-model-upload', children=dbc.Button('Carica modello linguistico JSON'), max_size=8*1024*1024),
    html.Div(id='lab-load-status', role='status'),
    html.H5('Valutazione su test tenuto separato'),
    html.P('Selezione dell’epoca solo sulla validation. Perplexity include caratteri, UNK ed EOS. '
           'Il corpus dimostrativo contiene pattern ripetuti: osservare anche la sovrapposizione delle finestre.'),
    html.Div(id='lab-metrics'),
    dcc.Graph(id='lab-loss'),
    dbc.Button('Scarica report esperimento', id='lab-report-save', className='my-2'),
    dcc.Download(id='lab-report-download'),
    html.H5('Generazione autoregressiva'),
    dcc.Textarea(id='lab-prompt', value='il ', style={'width': '100%', 'height': '60px'}),
    dbc.Row([
        dbc.Col([dbc.Label('Token massimi'), dbc.Input(id='lab-length', type='number', value=120, min=1, max=512)]),
        dbc.Col([dbc.Label('Temperatura'), dbc.Input(id='lab-temperature', type='number', value=.8, min=.05, max=2, step=.05)]),
        dbc.Col([dbc.Label('Top-k (0 = tutti)'), dbc.Input(id='lab-top-k', type='number', value=0, min=0)]),
        dbc.Col([dbc.Label('Top-p'), dbc.Input(id='lab-top-p', type='number', value=.95, min=.01, max=1, step=.05)]),
        dbc.Col([dbc.Label('Seed campionamento'), dbc.Input(id='lab-generation-seed', type='number', value=7, min=0)]),
    ]),
    dbc.Button('Genera testo', id='lab-generate', className='my-2'),
    html.Div(id='lab-generation-status', role='status'),
    html.Pre(id='lab-output', style={'whiteSpace': 'pre-wrap'}),
    html.Div(id='lab-trace', style={'overflowX': 'auto'}),
    html.P('I modelli JSON sono esportabili per inferenza; non contengono il corpus né uno stato optimizer per riprendere il training. '
           'Dati e modelli restano distinti per scheda, senza autenticazione.'),
], fluid=True)


class LanguageRun:
    def __init__(self, corpus, context, slots, epochs, lr, seed):
        self.corpus, self.epochs, self.lr = corpus, epochs, lr
        self.model = ConceptLanguageModel(context, slots, seed)
        self.iteration = 0
        self.stop_event = threading.Event()

    def nLearn(self, **kwargs):
        self.model.fit(self.corpus, self.epochs, self.lr,
                       progress=lambda epoch: setattr(self, 'iteration', epoch),
                       cancelled=self.stop_event.is_set)
        self.corpus = None

    def request_stop(self):
        self.stop_event.set()


def register(app, job_key):
    jobs = TrainingJobs(capacity=4)

    @app.callback(Output('lab-corpus', 'value'), Output('lab-upload-status', 'children'),
                  Input('lab-corpus-upload', 'contents'), prevent_initial_call=True)
    def upload(contents):
        try:
            value = base64.b64decode(contents.split(',', 1)[1]).decode('utf-8-sig')
            if not 100 <= len(value) <= 40000:
                raise ValueError('Servono 100–40.000 caratteri.')
            return value, f'{len(value)} caratteri caricati.'
        except Exception as exc:
            return dash.no_update, f'❌ {exc}'

    @app.callback(Output('lab-status', 'children'), Output('lab-stop', 'disabled'),
                  Output('lab-monitor', 'disabled'), Input('lab-train', 'n_clicks'),
                  Input('lab-stop', 'n_clicks'), Input('lab-monitor', 'n_intervals'),
                  State('lab-corpus', 'value'), State('lab-context', 'value'),
                  State('lab-slots', 'value'), State('lab-epochs', 'value'),
                  State('lab-lr', 'value'), State('lab-seed', 'value'), State('tab-session', 'data'),
                  prevent_initial_call=True)
    def train(n_train, n_stop, intervals, corpus, context, slots, epochs, lr, seed, tab):
        try:
            key = job_key(tab)
            if dash.ctx.triggered_id == 'lab-train':
                existing = jobs.get(key)
                if existing and not existing['done'].is_set():
                    return 'Training già in corso in questa scheda.', False, False
                runner = LanguageRun(corpus, context, slots, epochs, lr, seed)
                jobs.start(key, runner, {}, [])
                return 'Training linguistico avviato.', False, False
            if dash.ctx.triggered_id == 'lab-stop':
                jobs.stop(key)
                return 'Stop richiesto.', True, False
            job = jobs.get(key)
            if not job:
                return 'Nessun training in questa scheda.', True, True
            if not job['done'].is_set():
                return f"Epoca {job['learner'].iteration} — training in corso", False, False
            if job['error']:
                return f"❌ {job['error']}", True, True
            return f"✅ Completato: {job['learner'].iteration} epoche", True, True
        except Exception as exc:
            return f'❌ {exc}', True, True

    @app.callback(Output('lab-model', 'data'), Output('lab-load-status', 'children'),
                  Input('lab-monitor', 'disabled'), Input('lab-model-upload', 'contents'),
                  State('tab-session', 'data'), prevent_initial_call=True)
    def model_state(disabled, contents, tab):
        try:
            job = jobs.get(job_key(tab))
            if dash.ctx.triggered_id == 'lab-model-upload':
                if job and not job['done'].is_set():
                    raise ValueError('Attendere la fine del training prima di caricare un modello.')
                if not contents:
                    return dash.no_update, dash.no_update
                text = base64.b64decode(contents.split(',', 1)[1]).decode('utf-8')
                model = ConceptLanguageModel.from_json(text)
                return model.to_json(), '✅ Modello linguistico caricato.'
            if disabled and job and job['done'].is_set() and not job['error']:
                return job['learner'].model.to_json(), '✅ Modello pronto per generazione.'
            return dash.no_update, dash.no_update
        except Exception as exc:
            return dash.no_update, f'❌ {exc}'

    @app.callback(Output('lab-metrics', 'children'), Output('lab-loss', 'figure'),
                  Input('lab-model', 'data'))
    def metrics(text):
        if not text:
            return 'Addestrare o caricare un modello per vedere le misure.', go.Figure()
        try:
            model = ConceptLanguageModel.from_json(text)
            report = model.report
            rows = [dict(metrica=k, valore=v) for k, v in report['test'].items()]
            rows += [dict(metrica=k, valore=report[k]) for k in
                     ('unigram_perplexity', 'bigram_perplexity', 'train_test_window_overlap', 'parameters', 'selected_epoch')]
            figure = go.Figure()
            for field in ('train_nll', 'validation_nll'):
                figure.add_scatter(x=[h['epoch'] for h in model.history], y=[h[field] for h in model.history], name=field)
            figure.update_layout(xaxis_title='Epoca', yaxis_title='Cross-entropy (nat/token)')
            return dbc.Table.from_dataframe(pd.DataFrame(rows), striped=True), figure
        except Exception as exc:
            return f'❌ Report non valido: {exc}', go.Figure()

    @app.callback(Output('lab-output', 'children'), Output('lab-trace', 'children'),
                  Output('lab-generation-status', 'children'), Input('lab-generate', 'n_clicks'),
                  Input('lab-model', 'data'), State('lab-prompt', 'value'), State('lab-length', 'value'),
                  State('lab-temperature', 'value'), State('lab-top-k', 'value'), State('lab-top-p', 'value'),
                  State('lab-generation-seed', 'value'), prevent_initial_call=True)
    def generate(clicks, text, prompt, length, temperature, top_k, top_p, seed):
        if dash.ctx.triggered_id != 'lab-generate':
            return '', [], 'Modello aggiornato: generare un nuovo campione.'
        try:
            if not text:
                raise ValueError('Addestrare o caricare prima un modello linguistico.')
            model = ConceptLanguageModel.from_json(text)
            output, trace = model.generate(prompt or '', length, temperature, top_k, top_p, seed)
            table = dbc.Table.from_dataframe(pd.DataFrame(trace[:100]), striped=True)
            return output, table, f'✅ {len(trace)} token campionati (EOS incluso se presente). Traccia dei primi 100.'
        except Exception as exc:
            return '', [], f'❌ {exc}'

    @app.callback(Output('lab-model-download', 'data'), Input('lab-save', 'n_clicks'),
                  State('lab-model', 'data'), prevent_initial_call=True)
    def save_model(clicks, text):
        return dict(content=text, filename='acan-character-model.json') if text else dash.no_update

    @app.callback(Output('lab-report-download', 'data'), Input('lab-report-save', 'n_clicks'),
                  State('lab-model', 'data'), prevent_initial_call=True)
    def save_report(clicks, text):
        if not text:
            return dash.no_update
        model = ConceptLanguageModel.from_json(text)
        return dict(content=json.dumps(dict(report=model.report, history=model.history), indent=2),
                    filename='acan-language-experiment.json')
