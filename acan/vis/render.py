import os
import secrets
from flask import session
from acan.vis.jobs import TrainingJobs
from acan.vis.inference import infer

import dash  # pip install dash
import dash_cytoscape as cyto  # pip install dash-cytoscape==0.2.0 or higher
from dash import html
from dash import dcc
from dash.dependencies import Output, Input, State
import pandas as pd  # pip install pandas
from acan.core.model import AlgebraicClusterModel, GraphClusterModel
from acan.core.learner import AlgebraicClusterLearner
import matplotlib.pyplot as plt
import io
import gzip
import base64
from scipy.io import arff
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn import metrics
from scipy.io import loadmat
import pyarrow.parquet as pq
import dash_daq as daq
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY],
    requests_pathname_prefix='/', #'/proxy/4000/'
    routes_pathname_prefix='/',
    suppress_callback_exceptions=True)  # Permette callback su componenti generati condizionalmente
app.title='Algebraic Cluster Model'
els={}

# Signed cookies identify browser sessions; learners stay on the server.
app.server.secret_key = os.environ.get('ACAN_SESSION_SECRET') or secrets.token_hex(32)
app.server.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
training_jobs = TrainingJobs()

@app.server.get('/healthz')
def health():
    return {'status': 'ok'}

@app.server.before_request
def ensure_session():
    if 'acan_session' not in session:
        session['acan_session'] = secrets.token_urlsafe(32)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'zip' in filename:
            f=gzip.GzipFile(mode='rb', fileobj=io.BytesIO(decoded))
            root=  AlgebraicClusterModel.loadModel(f.read().decode())
            return  pd.DataFrame(), to_elements(root, root.labels)

    except Exception as e:
        print(e)
        return pd.DataFrame(), html.Div([
            'There was an error processing this file.'
        ])


def getColor(gcm : GraphClusterModel, acm: AlgebraicClusterModel):
    if len(gcm.labels)>0:
        flbs=acm.fullLabels(gcm.labels)
    else:
        flbs=acm.cols
    
    val, err, pos=acm.predictValue(flbs)
    l=len(flbs)
    if l>1:
        ind=flbs.index(pos)/(l-1)
    else:
        ind=1
    red=err*255
    green=ind*val*(1-err)*255
    blue=(1-ind)*255*(1-err)


    color='rgb('+str(red)+','+str(green)+','+str(blue)+')'

    return color


def getGroupColor(gcm: GraphClusterModel, node, group_idx):
    """
    Calcola il colore per un Group specifico basato sulla label prevalente.
    - Stessa label = stesso colore base
    - Errore alto = più rossi
    - Errore basso = colore puro della label
    """
    # Ottieni i valori del gruppo specifico
    group_values = pd.DataFrame(node.Groups, columns=node.cols).iloc[group_idx]
    relevance_row = pd.DataFrame(node.Relevance, columns=node.cols).iloc[group_idx]
 
    if len(gcm.labels) == 0:
        # Se non ci sono label, usa scala di grigi basata sul valore max
        max_val = max(group_values)
        gray = int(128 + max_val * 127)  # 128-255
        return f'rgb({gray},{gray},{gray})'
    
    # Ottieni le label complete
    flbs = node.fullLabels(gcm.labels)
    
    # Trova la label prevalente e il suo valore
    lbl_vals = [(lbl, group_values[lbl]) for lbl in flbs if lbl in group_values.index]
    if not lbl_vals:
        return 'rgb(128,128,128)'
    
    # Ordina per valore decrescente
    lbl_vals.sort(key=lambda x: x[1], reverse=True)
    dominant_label, dominant_val = lbl_vals[0]  # Label prevalente e suo valore
    
    # Calcola errore: 1 - confidenza (quanto è "sicuro" il gruppo della sua label)
    # Se il valore è 1.0 → errore 0 (certezza perfetta)
    # Se il valore è 0.5 → errore alto (incertezza)
    error = max(0, 1 - dominant_val)
    
    # Indice della label prevalente (per assegnare colore base diverso)
    label_idx = flbs.index(dominant_label) if dominant_label in flbs else 0
    num_labels = len(flbs)
    
    # Assegna colore base in base alla label (HSL hue: 0-360)
    # Distribuisci uniformemente i colori sul cerchio cromatico
    hue = (label_idx / max(1, num_labels - 1)) * 300 if num_labels > 1 else 120  # 0=rosso, 120=verde, 240=blu
    
    # Converte HSL a RGB con saturazione/value modulati dall'errore
    # Errore alto → più rosso (saturation bassa, lightness media, red mix alto)
    # Errore basso → colore puro (saturation alta, lightness media-alta)
    import colorsys
    
    # Saturation: bassa se errore alto, alta se errore basso
    saturation = 0.3 + (1 - error) * 0.7  # 0.3 (grigio) a 1.0 (puro)
    
    # Lightness: medio-alto per buona visibilità
    lightness = 0.5 + (1 - error) * 0.2  # 0.5 a 0.7
    
    # Converte HSL a RGB
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    
    # Se c'è errore, mixa con rosso (errore 1.0 = rosso puro)
    red_mix = int(error * 255 + (1 - error) * r * 255)
    green_mix = int((1 - error) * g * 255)
    blue_mix = int((1 - error) * b * 255)
    
    return f'rgb({red_mix},{green_mix},{blue_mix})'


def fs(value, k):
    if value>0:
        return f"{value:.{k-1}e}".rstrip("0").rstrip(".")
    else:
        return ""


def sum_relevances(df_relevance, aggregate_by_max=False):
    """
    Calcola la somma delle rilevanze e arrotonda all'intero.
    
    Per le foglie (singolo gruppo): somma tutte le rilevanze
    Per la radice (multi-gruppo): prende il max per feature tra i gruppi, poi somma
    (così il numero è comparabile con quello delle foglie)
    """
    if df_relevance.empty:
        return 0
    
    if aggregate_by_max and len(df_relevance) > 1:
        # Per la radice: prendi la rilevanza massima per ogni feature (tra tutti i gruppi)
        # poi somma - così il numero è confrontabile con le foglie
        max_per_feature = df_relevance.max(axis=0)
        total = max_per_feature.sum()
    else:
        # Per singolo gruppo (foglia): somma tutto
        total = df_relevance.sum().sum()
    
    # Arrotonda all'intero più vicino
    return int(round(total))


def to_elements(gcm: GraphClusterModel, show_leaves=True):
    """
    Converte il modello in elementi Cytoscape.
    
    Args:
        gcm: GraphClusterModel
        show_leaves: Se False, mostra solo la radice senza i gruppi figli
    """
    elements={}


    for ids, node in gcm.clusterModel.items():
        color=getColor(gcm, node)
        rootNode={}
        # Per la radice: semplicemente il numero di colonne totali (m)
        # Le colonne della radice non sono mascherate, quindi mostriamo il totale
        features_with_relevance = len(node.cols) if hasattr(node, 'cols') and node.cols else 0
        
        # Calcola numero di gruppi attivi (con count > soglia)
        active_groups_count = 0
        if hasattr(node, 'Groups') and len(node.Groups) > 0:
            for k in range(len(node.Groups)):
                if hasattr(node, 'Freqs') and node.Freqs is not None:
                    freq = node.Freqs[k][0] if hasattr(node.Freqs[k], '__len__') else node.Freqs[k]
                elif hasattr(node, 'Counts') and node.Counts is not None:
                    count = node.Counts[k][0] if hasattr(node.Counts[k], '__len__') else node.Counts[k]
                    freq = count / node.n if node.n > 0 else 0
                else:
                    freq = 0
                # Usa direttamente il count se disponibile, altrimenti calcola da freq
                if hasattr(node, 'Counts') and node.Counts is not None and k < len(node.Counts):
                    count = node.Counts[k][0] if hasattr(node.Counts[k], '__len__') else node.Counts[k]
                    n_tuples = int(count)
                else:
                    n_tuples = int(node.n * freq)
                if n_tuples >= 1 or freq > 1e-6:
                    active_groups_count += 1
        
        # Label: mostra id, numero campioni (n, m_relevanti), numero gruppi attivi
        label_text = f"ID:{node.id} | N:({node.n}, {features_with_relevance}) | K:{active_groups_count}"
        
        # Dimensione del nodo basata su n * features (radice quadrata)
        product = node.n * features_with_relevance
        scala_dinamica = (product ** 0.5)
        parent_size = max(50, min(120, 50 + int(scala_dinamica)))  # Base più grande per il padre

        rootNode['data']={'id': str(node.id), 'size':parent_size, 'label':label_text, 'errColor':color}
        rootNode['classes']='node'
        rootNode['grabbable'] = True
        elements[str(node.id)]=rootNode
        #print(str(node.id))

        # Crea nodi Group per ogni gruppo (solo se show_leaves=True)
        if show_leaves and hasattr(node, 'Groups') and len(node.Groups) > 0:
            df_relevance = pd.DataFrame(node.Relevance, columns=node.cols)
            
            # Mostra tutti i gruppi (filtro rimosso - il modello è già pulito)
            active_groups = []
            for k in range(len(node.Groups)):
                # Usa direttamente Counts per n_tuples (più preciso)
                if hasattr(node, 'Counts') and node.Counts is not None and k < len(node.Counts):
                    count = node.Counts[k][0] if hasattr(node.Counts[k], '__len__') else node.Counts[k]
                    n_tuples = int(count)
                elif hasattr(node, 'Freqs') and node.Freqs is not None:
                    freq = node.Freqs[k][0] if hasattr(node.Freqs[k], '__len__') else node.Freqs[k]
                    n_tuples = int(node.n * freq)
                else:
                    n_tuples = 0
                
                # Calcola freq per compatibilità
                freq = n_tuples / node.n if node.n > 0 else 0
                active_groups.append((k, freq, n_tuples))
            
            for k, freq, n_tuples in active_groups:
                # ID univoco per il nodo Group
                group_id = f"{node.id}_G{k}"

                # Calcola colore specifico per questo Group
                group_color = getGroupColor(gcm, node, k)
                
                # Calcola numero di feature attive (con rilevanza > soglia) per questo gruppo
                df_relevance_row = pd.DataFrame(node.Relevance, columns=node.cols).iloc[[k]]
                # Conta feature con rilevanza significativa (> 0.01)
                features_with_relevance = (df_relevance_row > 0.01).sum().sum()
                # Assicura almeno 1 per evitare size=0
                features_with_relevance = max(1, int(features_with_relevance))
                
                # Etichetta compatta: numero tuple e features (n, m)
                groupLabel = f"({n_tuples}, {features_with_relevance})"

                # Dimensione del nodo figlio proporzionale al padre
                # Basata sul rapporto tra (n*m figlio) / (n*m padre)
                child_product = n_tuples * features_with_relevance
                parent_product = node.n * (df_relevance > 0).any(axis=0).sum()  # m totale del padre
                
                if parent_product > 0:
                    ratio = (child_product / parent_product) ** 0.5  # radice quadrata per smoothing
                    group_size = max(20, int(parent_size * ratio))  # min 20, proporzionale al padre
                else:
                    group_size = 30  # default se parent_product è 0

                # Crea nodo Group con colore specifico e dimensione basata su n*m
                groupNode = {
                    'data': {
                        'id': group_id,
                        'size': group_size,
                        'label': groupLabel,
                        'errColor': group_color
                    },
                    'classes': 'group-node',
                    'grabbable': True
                }
                elements[group_id] = groupNode

                # Edge dal nodo padre al Group
                edgeNode = {
                    'data': {
                        'id': f"N{node.id}G{k}",
                        'source': str(node.id),
                        'target': group_id,
                        'size': 1,
                        'label': group_color,
                        'errColor': group_color
                    },
                    'classes': 'directed'
                }
                elements[f"N{node.id}G{k}"] = edgeNode

        # Edge dai Groups ai children (se esistono)
        for k, childId in node.children.items():
            group_id = f"{node.id}_G{k}"
            edgeNode={}
            edgeNode['data']={'id': "N"+str(node.id)+"K"+str(k),'source': group_id, 'target': str(childId), 'size':1, 'label':color, 'errColor':color}
            edgeNode['classes'] = 'directed'
            elements["N"+str(node.id)+"K"+str(k)]=edgeNode

    #print(rootId+" "+str(step))
    return elements


def to_svg(ele):
    plt.pie(ele.df)
    f = io.BytesIO()
    plt.savefig(f, format="svg")
    return f

uplData=dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Dataset Files (Max 150MB)")]),
            style={
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=False,
            max_size=1500000000
        )

uplModel=dcc.Upload(
            id="upload-model",
            children=html.Div(["Drag and Drop or ", html.A("Load Model JSON")]),
            style={
                "lineHeight": "40px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
                "backgroundColor": "#e8f4f8"
            },
            multiple=False,
            max_size=500000000
        )


orgChartData=cyto.Cytoscape(
    id='org-chart',
    autoungrabify=True,
    minZoom=0.1,
    maxZoom=2,
    layout={'name': 'cose', 'animate': False},
    #layout={'name': 'cose', 'animate': True},
    elements={},
    #style={'width': '500px', 'height': '500px'},
    stylesheet=[
        # Group selectors for NODES
        {
            'selector': 'node',
            'style': {

                'width': 'data(size)',
                'height': 'data(size)',
                'background-color': 'data(errColor)',
                'label':'data(label)',
                'font-size': '10px',
                'text-valign': 'center',
                'text-halign': 'center'

            }
        },

        # Stile specifico per nodi Group (più piccoli)
        {
            'selector': '.group-node',
            'style': {
                'shape': 'ellipse',
                'border-width': 1,
                'border-color': '#666',
                'background-opacity': 0.8,
                'font-size': '10px',
                'font-weight': 'bold',
                'color': '#000',
                'text-outline-width': 1,
                'text-outline-color': '#fff'
            }
        },

        # Group selectors for EDGES
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle'
            }
        },
            {
                'selector': '.directed',
                'style': {
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 2
                }
            }
        ]
)

data_layout = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(id="uplColId", children=uplData

                        , width=12)
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id='data-info', children='') , width=12)
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id='data-preview', children='') , width=6)
                ]
            )
    ]
  )


def create_legend():
    """Crea un pannello legenda per interpretare i colori dei nodi"""
    return dbc.Card([
        dbc.CardHeader("Legenda Colori", className="bg-light"),
        dbc.CardBody([
            html.P("I colori indicano:", className="mb-2"),
            html.Ul([
                html.Li([
                    html.Span("Rosso=Alto: ", style={"color": "#ff0000", "fontWeight": "bold"}),
                    "Errore elevato (incertezza)"
                ]),
                html.Li([
                    html.Span("Verde=Alto: ", style={"color": "#00ff00", "fontWeight": "bold"}),
                    "Valore di probabilità alto"
                ]),
                html.Li([
                    html.Span("Blu=Alto: ", style={"color": "#0000ff", "fontWeight": "bold"}),
                    "Cambiamento lungo l'asse classi"
                ])
            ], className="small mb-0")
        ], className="p-2")
    ], className="mb-3", style={"fontSize": "12px"})

model_layout = dbc.Container([
       html.Div(children=[
            # Accordion spiegativo
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P([
                        "Il modello mostra una gerarchia di cluster. ",
                        html.B("Clicca su un nodo"), " per vedere:"
                    ]),
                    html.Ul([
                        html.Li([html.B("Centroide Root:"), " Valore medio di ogni attributo nel nodo"]),
                        html.Li([html.B("Errore (E):"), " Deviazione standard (dispersione dei dati)"]),
                        html.Li([html.B("Gruppi figli:"), " Clicca sul nodo poi sul cluster figlio per vedere sotto-gruppi"]),
                    ]),
                    html.P("Per ogni attributo: valore alto = attributo caratteristico del cluster", className="text-muted mt-2")
                ], title="📖 Come leggere il modello")
            ], start_collapsed=True, className="mb-2"),

            # Upload modello JSON
            dbc.Row([
                dbc.Col(uplModel, width=12)
            ], className="mb-2"),

            dbc.Row(
                [
                    dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(html.P("Class:"), width=2),
                                        dbc.Col(
                                            dcc.Dropdown(
                                            id='dpdnCategory',
                                            value='',
                                            clearable=False,
                                            multi=True,
                                            options=[]), width=9
                                        ),
                                        dbc.Col(
                                            dbc.Button("✕", id='btn-clear-category', size="sm", color="light", style={'padding': '0px 6px', 'fontSize': '10px'}), width=1
                                        )
                                    ]
                                )
                            ], width=3),
                    dbc.Col(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(html.P("Filters:"), width=2),
                                        dbc.Col(
                                            dcc.Dropdown(
                                            id='dpdnFilter',
                                            value='',
                                            clearable=False,
                                            multi=True,
                                            options=[]), width=9
                                        ),
                                        dbc.Col(
                                            dbc.Button("✕", id='btn-clear-filter', size="sm", color="light", style={'padding': '0px 6px', 'fontSize': '10px'}), width=1
                                        )
                                    ]
                                )
                            ], width=3),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(html.P("Labels:"), width=2),
                                    dbc.Col(
                                        dcc.Dropdown(
                                        id='dpdnLabel',
                                        value='',
                                        clearable=False,
                                        multi=True,
                                        options=[]), width=9
                                    ),
                                    dbc.Col(
                                        dbc.Button("✕", id='btn-clear-label', size="sm", color="light", style={'padding': '0px 6px', 'fontSize': '10px'}), width=1
                                    )
                                ]
                            )
                        ], width=3),
                    dbc.Col(daq.BooleanSwitch(disabled=True, id='bsId', label="balanced", on=False) ,  width=1),
                    dbc.Col(dbc.Button(
                                "Learn Model", id="button-learn", className="me-2", n_clicks=0
                            ) ,  width=1),
                    dbc.Col([
                            dbc.Button("Download Model", id="btn-download", className="me-2", n_clicks=0),
                            dcc.Download(id="download-model")
                        ] ,  width=1)
                ]
), dcc.Store(id="model-store"), dcc.Store(id="label-store"), dcc.Store(id="evaluation-trigger", data=0),
            
            # INPUT per batch size (default 300)
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.InputGroup([
                            dbc.InputGroupText("Max Batch:"),
                            dbc.Input(id="input-max-batch", type="number", value=300, min=10, max=10000, step=10)
                        ], size="sm")
                    ], style={'width': '200px'})
                ], width="auto", className="mb-2"),
                dbc.Col([
                    html.Div([
                        dbc.InputGroup([
                            dbc.InputGroupText("Learning Rate:"),
                            dbc.Input(id="input-lr", type="number", value=0.1, min=0.001, max=1.0, step=0.01)
                        ], size="sm")
                    ], style={'width': '200px'})
                ], width="auto", className="mb-2")
            ]),

            # CONTROLLI LEARNING CON THREADING (separati dal grafo per evitare flash)
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button("⏹️ Stop Learning", id="btn-stop-learn", color="danger", size="sm", disabled=True)
                                    ], width="auto"),
                                    dbc.Col([
                                        html.Div(id="learning-status", children="Pronto - Clicca 'Learn Model' per iniziare", 
                                                className="fw-bold text-primary mt-1")
                                    ], width=True)
                                ], align="center")
                            ])
                        ], className="mb-2")
                    ], style={'position': 'relative'})
                ], width=12)
            ]),
            
            # Timer nascosto (non nella card che flasha)
            dcc.Interval(id='learning-monitor', interval=2000, disabled=True),
            
            ]),

             dbc.Row(
                    [
                        dbc.Col(orgChartData, width=8)
                        ,
                        dbc.Col([
                                html.Div(id='node-detail', children=''),
                                html.Div(id='edge-detail', children='')
                                ]
                            , width=4, align='center')
                        ])


    ])



"""
dcc.Dropdown(
    id='dpdn',
    value='cose',
    clearable=False,
    options=[
        {'label': name.capitalize(), 'value': name}
        for name in ['breadthfirst' ,'grid', 'random', 'circle', 'cose', 'concentric']
    ]
    """
evaluation_layout = dbc.Container([
        dbc.Row(dcc.Graph(id='errorIndicator')),
        dbc.Row(html.Div(id='cm-detail', children=''), justify='center'),
        dbc.Row([dbc.Button(
                    "Download prediction", id="button-prediction", className="me-2", n_clicks=0),
                    dcc.Download(id="download-prediction")], justify='end')
    ])


inference_layout = dbc.Container([
    html.P('Usa il modello caricato o addestrato nella scheda Model. Nessun nuovo training.'),
    dcc.Upload(id='inference-upload', children=dbc.Button('Carica CSV per inferenza'), multiple=False),
    dcc.Store(id='inference-input'),
    html.Div(id='inference-upload-status', role='status'),
    dcc.RadioItems(id='inference-mode', options=[
        {'label': 'Predici Labels (ignora eventuali valori target nel CSV)', 'value': 'predict'},
        {'label': 'Completa campi vuoti (mantiene i valori presenti)', 'value': 'complete'},
    ], value='predict'),
    dbc.Button('Esegui inferenza', id='inference-run', className='my-2'),
    html.Div(id='inference-status', role='status'),
    html.Div(id='inference-table', style={'overflowX': 'auto'}),
    dcc.Store(id='inference-result'),
    dbc.Button('Scarica risultati CSV', id='inference-export', disabled=True),
    dcc.Download(id='inference-download'),
])

app.layout = dbc.Container([
    html.H1('Simple Data Mining', style={'textAlign': 'center'}),
    dcc.Tabs(id='tabs', value='Data', children=[
        dcc.Tab(label=name, value=name) for name in ['Data', 'Model', 'Evaluation', 'Inference']
    ]),
    # Keep the graph mounted when switching tabs: destroying Cytoscape while
    # asynchronous layout/resize callbacks are pending causes browser errors.
    html.Div([data_layout, dcc.Store(id='data-store')], id='page-data'),
    html.Div([model_layout, dcc.Store(id='test-store')], id='page-model', style={'display': 'none'}),
    html.Div([evaluation_layout, dcc.Store(id='eval-store')], id='page-evaluation', style={'display': 'none'}),
    html.Div(inference_layout, id='page-inference', style={'display': 'none'}),
], fluid=True)


@app.callback([Output('page-data', 'style'), Output('page-model', 'style'),
               Output('page-evaluation', 'style'), Output('page-inference', 'style')],
              Input('tabs', 'value'))
def show_page(tab):
    return [{'display': 'block' if tab == name else 'none'}
            for name in ['Data', 'Model', 'Evaluation', 'Inference']]


# Callback per pulire i dropdown
@app.callback(
    Output('dpdnCategory', 'value', allow_duplicate=True),
    Input('btn-clear-category', 'n_clicks'),
    prevent_initial_call=True
)
def clear_category(n_clicks):
    if n_clicks:
        return []
    return dash.no_update

@app.callback(
    Output('dpdnFilter', 'value', allow_duplicate=True),
    Input('btn-clear-filter', 'n_clicks'),
    prevent_initial_call=True
)
def clear_filter(n_clicks):
    if n_clicks:
        return []
    return dash.no_update

@app.callback(
    Output('dpdnLabel', 'value', allow_duplicate=True),
    Input('btn-clear-label', 'n_clicks'),
    State('dpdnLabel', 'options'),
    prevent_initial_call=True
)
def clear_label(n_clicks, options):
    if n_clicks:
        return []
    return dash.no_update


@app.callback([Output('label-store', 'data'), Output("dpdnFilter", "options")],
              Input('dpdnLabel', 'value'),
              State("dpdnLabel", "options"))
def set_labels(labels, opts):
    newOps=[]
    for opt in opts:
        if opt['value'] not in labels:
            newOps.append(opt)
    return labels, newOps

@app.callback(
    [Output("dpdnLabel", "options"),Output("dpdnLabel", "value"), Output("dpdnCategory", "options"), Output("dpdnFilter", "value")] ,
    Input('data-store', 'data'),
    State('label-store', 'data'),
    prevent_initial_call=True,
)
def update_labels(data, lbs):
    df=pd.DataFrame(data)
    if len(df)>0:
        filters=[]
        n=len(df)
        for col in df.columns:
            unq=len(df[col].unique())
            if  df[col].dtype == object and (unq==1 or (unq/n)>0.9 or unq>1000):
                filters.append(col)
            if  (df[col].dtype == 'int64' or df[col].dtype == 'float64') and df[col].max()-df[col].min()==0:
                filters.append(col)

        options=[
            {'label': name, 'value': name}
            for name in df.columns
        ]
        lblList=['class', 'label']
        
        labels=[]
        
        if len(lbs)>0 and lbs[0] in df.columns:
            labels.append(lbs[0])
        else:
            for col in df.columns:
                if col.lower() in lblList:
                    labels.append(col)
                    break
        
        if len(labels)==0:
            labels.append(list(df.columns)[len(df.columns)-1])

        return options, labels, options, filters

    return [],[], [], []



@app.callback(
    Output("download-model", "data"),
    Input("btn-download", "n_clicks"),
    State("model-store", "data"),
    prevent_initial_call=True,
)
def download_model(n_clicks, model):
    if model:
        return dict(content=model, filename="model.json")

@app.callback([Output('data-store', 'data'), Output('bsId', 'on'), Output('uplColId', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State('label-store', 'data'),
              prevent_initial_call=True)
def update_data(contents, filename, last_modified, labels):
    balanced=False
    if filename != None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
                
            if 'json' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_json(
                    io.StringIO(decoded.decode('utf-8')))
                
            elif 'xlsx' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')

            elif 'parquet' in filename:
                arrTb=pq.read_table(io.BytesIO(decoded))
                df=arrTb.to_pandas(date_as_object=False)

            elif 'arff' in filename:
                # Assume that the user uploaded an excel file
                data = arff.loadarff(io.StringIO(decoded.decode('utf-8')))
                df = pd.DataFrame(data[0])

            elif 'mat' in filename:
                mat = loadmat(io.BytesIO(decoded))
                mat = {k:v for k, v in mat.items() if k[0] != '_'}
                mdata = mat[next(iter(mat))]  # variable in mat file
                mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
                ndata = {n: mdata[n][0, 0] for n in mdtype.names}
                df=pd.DataFrame(ndata['data'])

            df.columns = df.columns.map(str)

            df=df.astype({col: str for col, dtype in df.dtypes.items() if dtype=='object'})
            
            if labels and len(labels)==1 and labels[0] in df.columns:
                balanced=True
                df = AlgebraicClusterModel.downsample_classes(df, labels)
            
            df=GraphClusterModel.full_cols(df)
            df = df.sample(min(10000, len(df)))
            return df.to_dict(), balanced, uplData
        except Exception as e:
            print(e)
            return pd.DataFrame().to_dict(), balanced, uplData

    

    return pd.DataFrame().to_dict(), balanced, uplData



@app.callback(Output("data-info", "children"),
              Input('data-store', 'data'))
def update_dataInfo(data):
    df=pd.DataFrame(data)
    if len(df)>0:
        df= df.describe(include='all').transpose().reset_index()
    return dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True)


@app.callback([Output("errorIndicator", "figure"), Output("cm-detail", "children"), Output("eval-store", "data")],
              Input("evaluation-trigger", "data"),  # Solo trigger, non model-store
              State('test-store', 'data'),
              State('label-store', 'data'),
              State('dpdnCategory', 'value'),
              State('dpdnFilter', 'value'),
              State("model-store", "data"),  # Spostato da Input a State
              prevent_initial_call=True)
def evaluate_model(trigger, data, labels, categories, filters, model):
    #sd=dcc.send_data_frame(pd.DataFrame().to_excel, "prediction.xlsx", sheet_name="Sheet1")
    cmT=dbc.Table()
    err=0
    # Ritorna valori di default se modello caricato da file senza dati di test
    if model is None:
        return go.Figure(), html.Div("Nessun modello disponibile"), {}

    # Se non ci sono dati di test (es: modello caricato da file), non fare evaluation
    if not data or len(data) == 0 or len(labels) == 0:
        return go.Figure(), html.Div("Caricare dati di test per la valutazione"), {}

    df=pd.DataFrame(data)
    df = df.drop(filters, axis=1, errors='ignore')
    if len(categories)>0:
        df[categories] = df[categories].astype(str)


    gcm=GraphClusterModel.loadModel(model)
    err, Y_test, Y_pred, Y_conf, path=gcm.evaluateError(df, labels)
    err=err*100
    df[Y_pred.columns]=Y_pred
    #sd=dcc.send_data_frame(df.to_excel, "prediction.xlsx", sheet_name="Sheet1")
    Y_pred2=Y_pred.idxmax(axis=1)
    Y_test=Y_test.idxmax(axis=1)
    flbs=gcm.clusterModel[gcm.rootId].fullLabels(labels)
    predLbs=[]
    realLbs=[]
    for pLbl in flbs:
        predLbs.append(pLbl.replace(labels[0],"Pred"))

    for rLbl in flbs:
        realLbs.append(rLbl.replace(labels[0],"Real"))


    cm = metrics.confusion_matrix(Y_test, Y_pred2, labels=flbs)
    cm =pd.DataFrame(cm, columns=predLbs, index=realLbs)
    cm.loc['Pred_Total']= cm.sum(numeric_only=True, axis=0)
    cm.loc[:,'Real_Total'] = cm.sum(numeric_only=True, axis=1)
    cmT=dbc.Table.from_dataframe(cm.reset_index(), striped=True, bordered=False, hover=True)

    return go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = err,
                    title = {'text': "Error"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 25], 'color': 'green'},
                                {'range': [25, 50], 'color': 'yellow'},
                                {'range': [50,100], 'color': 'red'}],
                            }
                )), cmT, gcm.predictValues(Y_pred, Y_conf, path).to_dict()



@app.callback(
    Output("download-prediction", "data"),
    Input("button-prediction", "n_clicks"),
    State("eval-store", "data"),
    State('test-store', 'data'),
    prevent_initial_call=True,
)
def download_prediction(n_clicks, pred, data):
    dfPred=pd.DataFrame(pred)
    df=pd.DataFrame(data)
    df[dfPred.columns]=dfPred
    #sd=dcc.send_data_frame(df.to_excel, "prediction.xlsx", sheet_name="Evaluation")
    return dict(content=df.to_json(orient="records"), filename="prediction.json")
    
    
# ============================================================
# CALLBACK PER THREADING (Apprendimento controllato)
# ============================================================

@app.callback([Output("model-store", "data", allow_duplicate=True), 
               Output('test-store', 'data'),
               Output('learning-status', 'children'),
               Output('btn-stop-learn', 'disabled'),
               Output('learning-monitor', 'disabled')],
              Input('button-learn', 'n_clicks'),
              State('data-store', 'data'),
              State('label-store', 'data'),
              State('dpdnFilter', 'value'),
              State('dpdnCategory', 'value'),
              State('input-max-batch', 'value'),
              State('input-lr', 'value'),
              prevent_initial_call=True
              )
def update_model(n_clicks, data, labels, filters, categories, max_batch, lr):
    if not n_clicks:
        return (dash.no_update,) * 5
    if not data:
        return dash.no_update, dash.no_update, "⚠️ Nessun dato caricato!", True, True
    existing = training_jobs.get(session['acan_session'])
    if existing and not existing['done'].is_set():
        return dash.no_update, dash.no_update, "Training già in corso.", False, False
    try:
        df = pd.DataFrame(data)
        if len(df) < 5:
            raise ValueError('Servono almeno 5 righe per lo split train/test.')
        train = df.sample(frac=0.8, random_state=42)
        test = df.loc[~df.index.isin(train.index)].reset_index(drop=True)
        train = train.drop(filters or [], axis=1, errors='ignore')
        if categories:
            train[categories] = train[categories].astype(str)
        normalized, minmax = GraphClusterModel.normalizeData(GraphClusterModel.full_dummies(train))
        learner = AlgebraicClusterLearner(normalized, batch_size=max_batch or 300,
                                          lr=lr or 0.1, labels=labels or [])
        training_jobs.start(session['acan_session'], learner, minmax, labels or [])
        return dash.no_update, test.to_dict(), "🟢 Apprendimento avviato", False, False
    except Exception as exc:
        return dash.no_update, dash.no_update, f"❌ {exc}", True, True

@app.callback([Output('model-store', 'data', allow_duplicate=True),
            Output('org-chart', 'elements', allow_duplicate=True),
            Output('learning-status', 'children', allow_duplicate=True),
            Output('btn-stop-learn', 'disabled', allow_duplicate=True),
            Output('learning-monitor', 'disabled', allow_duplicate=True),
            Output('evaluation-trigger', 'data', allow_duplicate=True)],
            Input('learning-monitor', 'n_intervals'),
            prevent_initial_call=True)
def monitor_learning(n_intervals):
    job = training_jobs.get(session['acan_session'])
    if job is None:
        return dash.no_update, dash.no_update, 'Nessun training attivo.', True, True, dash.no_update
    learner = job['learner']
    if not job['done'].is_set():
        # Do not run forward_pass/setModel on parameters being updated by training.
        return (dash.no_update, dash.no_update,
                f"🟡 Iterazione: {learner.iteration}", False, False, dash.no_update)
    if job['error']:
        return dash.no_update, dash.no_update, f"❌ {job['error']}", True, True, dash.no_update
    gcm = GraphClusterModel()
    gcm.minMax = job['minmax']
    gcm.labels = job['labels']
    gcm.rootId = 0
    gcm.clusterModel = {0: learner.model}
    learner.model.id = 0
    elements = list(to_elements(gcm, show_leaves=learner.k_active < 100).values())
    return (gcm.to_json(), elements,
            f"✅ Completato: {learner.k_active} gruppi | CP: {learner.cp:.4f}",
            True, True, secrets.randbits(32))


@app.callback([Output('learning-monitor', 'disabled', allow_duplicate=True),
               Output('btn-stop-learn', 'disabled', allow_duplicate=True),
               Output('learning-status', 'children', allow_duplicate=True)],
              Input('btn-stop-learn', 'n_clicks'),
              prevent_initial_call=True)
def stop_learning(n_clicks):
    if n_clicks and training_jobs.stop(session['acan_session']):
        return dash.no_update, dash.no_update, '⏹️ Stop richiesto...'
    return dash.no_update, dash.no_update, dash.no_update

# Restore target controls with the saved model, even before a dataset is uploaded.
@app.callback(
    Output('model-store', 'data', allow_duplicate=True),
    Output('dpdnLabel', 'options', allow_duplicate=True),
    Output('dpdnLabel', 'value', allow_duplicate=True),
    Output('learning-status', 'children', allow_duplicate=True),
    Input('upload-model', 'contents'), State('upload-model', 'filename'),
    prevent_initial_call=True)
def load_model_json(contents, filename):
    if not contents:
        return (dash.no_update,) * 4
    try:
        active = training_jobs.get(session['acan_session'])
        if active and not active['done'].is_set():
            raise ValueError('Attendere la fine del training prima di caricare un modello.')
        if not filename or not filename.lower().endswith('.json'):
            raise ValueError('Selezionare un modello JSON.')
        model_json = base64.b64decode(contents.split(',', 1)[1]).decode('utf-8')
        model = GraphClusterModel.loadModel(model_json)
        if model.rootId not in model.clusterModel:
            raise ValueError('Il modello non contiene un nodo radice valido.')
        options = [{'label': label, 'value': label} for label in model.labels]
        return model_json, options, model.labels, '✅ Modello caricato: pronto per Inference.'
    except Exception as exc:
        return dash.no_update, dash.no_update, dash.no_update, f'❌ Modello non valido: {exc}'


@app.callback(Output('inference-input', 'data'), Output('inference-upload-status', 'children'),
              Input('inference-upload', 'contents'), State('inference-upload', 'filename'),
              prevent_initial_call=True)
def upload_inference(contents, filename):
    try:
        if not filename or not filename.lower().endswith('.csv'):
            raise ValueError('Selezionare un CSV UTF-8.')
        text = base64.b64decode(contents.split(',', 1)[1]).decode('utf-8-sig')
        data = pd.read_csv(io.StringIO(text))
        if data.empty:
            raise ValueError('Il CSV non contiene righe.')
        return text, f'{len(data)} righe caricate: {filename}'
    except Exception as exc:
        return None, f'❌ {exc}'


@app.callback(Output('inference-result', 'data'), Output('inference-table', 'children'),
              Output('inference-status', 'children'), Output('inference-export', 'disabled'),
              Input('inference-run', 'n_clicks'), Input('model-store', 'data'),
              Input('inference-input', 'data'), Input('inference-mode', 'value'),
              prevent_initial_call=True)
def run_inference(n_clicks, model_json, csv_text, mode):
    if dash.ctx.triggered_id != 'inference-run':
        return None, [], 'Pronto: premere Esegui inferenza.', True
    try:
        if not csv_text:
            raise ValueError('Caricare prima un CSV nella scheda Inference.')
        result = infer(model_json, pd.read_csv(io.StringIO(csv_text)), mode)
        table = dbc.Table.from_dataframe(result.head(100).fillna(''), striped=True, bordered=True)
        return (result.to_csv(index=False), table,
                f'✅ Inferenza completata: {len(result)} righe. Anteprima delle prime 100.', False)
    except Exception as exc:
        return None, [], f'❌ {exc}', True


@app.callback(Output('inference-download', 'data'), Input('inference-export', 'n_clicks'),
              State('inference-result', 'data'), prevent_initial_call=True)
def export_inference(n_clicks, csv_text):
    if not csv_text:
        return dash.no_update
    return dict(content=csv_text, filename='acan_predictions.csv')


@app.callback(Output('org-chart', 'elements'),
              Input("model-store", "data"),
              prevent_initial_call=True)
def update_graph(model):
    if not model:
        return []
    gcm=GraphClusterModel.loadModel(model)
    els=to_elements(gcm)
    if model:
        return list(els.values())
    else:
        return {}



"""Input('org-chart', 'mouseoverNodeData'),
Input('org-chart','mouseoverEdgeData'),
Input('org-chart','tapEdgeData'),
Input('org-chart','tapNodeData'),"""

@app.callback(
    Output("node-detail", "children"),
    Input('org-chart','selectedNodeData'),
    State("model-store", "data")
)
def node_info(snd, model):
    nId='D'
    if model:
        if snd != None and len(snd) > 0:
            node_id = snd[0]['id']
            # Gestisci ID nodi Group tipo "0_G3"
            if '_G' in str(node_id):
                # È un nodo Group, estrai parent_id e group_idx
                parts = str(node_id).split('_')
                parent_id = int(parts[0])
                group_idx = int(parts[1][1:])  # "G3" -> 3
                # Mostra i dettagli del gruppo specifico (valori del cluster, non della root)
                return eInfo(model, parent_id, group_idx)
            else:
                # È un nodo cluster normale
                nId = int(node_id)
                return graph_info(model, nId)

import re
import plotly.express as px


@app.callback(
    Output("edge-detail", "children"),
    Input('org-chart','selectedEdgeData'),
    State("model-store", "data")
)
def edge_info(sed, model):
    if model:
        if sed != None and len(sed) > 0:
            edge_id = sed[0]['id']
            source = int(sed[0]['source'])

            # Prova a matchare pattern per edge ai children: N{id}K{k}
            match = re.search(r'K(\d+)$', edge_id)
            if match:
                k = int(match.group(1))
                return eInfo(model, source, k)

            # Prova a matchare pattern per edge ai Groups: N{id}G{k}
            match = re.search(r'G(\d+)$', edge_id)
            if match:
                k = int(match.group(1))
                return eInfo(model, source, k)

            # Se non matcha nessun pattern, cerca il numero alla fine
            match = re.search(r'(\d+)$', edge_id)
            if match:
                k = int(match.group(1))
                return eInfo(model, source, k)


def graph_info(model, nId):
    """
    Mostra informazioni dettagliate su un nodo del grafo.
    Crea una visualizzazione chiara con:
    - Statistiche generali del nodo
    - Distribuzione delle classi (se disponibile)
    - Tabella degli attributi più importanti ordinati per errore
    """
    gcm = GraphClusterModel.loadModel(model)
    node = gcm.clusterModel[nId]
    
    # Card intestazione con info generali
    header_card = dbc.Card([
        dbc.CardHeader(html.H5(f"📊 Nodo {nId} - Dettagli", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.B("Campioni:") , width=6),
                dbc.Col(str(node.n), width=6),
            ]),
            dbc.Row([
                dbc.Col(html.B("Gruppi figli (K):"), width=6),
                dbc.Col(str(len(node.Groups)), width=6),
            ]),
            dbc.Row([
                dbc.Col(html.B("Attributi visibili:"), width=6),
                dbc.Col(str(len(node.cols)), width=6),
            ]),
        ], className="p-2")
    ], className="mb-3")
    
    # Tabella Centroide con barre visive per indicare l'importanza
    df_root = pd.DataFrame(node.Root, columns=node.cols)
    df_error = pd.DataFrame(node.RootError, columns=node.cols)
    
    if len(df_root) > 0:
        df_root = gcm.denormalizeData(df_root)
        df_root = df_root.transpose().reset_index()
        df_root.columns = ['Attributo', 'Centroide']
        
        df_error = df_error.transpose().reset_index()
        df_error.columns = ['Attributo', 'Errore']
        
        # Merge e ordinamento - attributi con errore più basso = più caratteristici
        df_merged = df_root.merge(df_error, on='Attributo')
        df_merged['Errore_%'] = (df_merged['Errore'] * 100).round(2)
        df_merged['Precisione'] = ((1 - df_merged['Errore']) * 100).round(1)
        
        # Ordina per precisione decrescente (più caratteristici per primi)
        df_merged = df_merged.sort_values(by='Precisione', ascending=False)
        
        # Colora le righe in base alla precisione
        def color_precision(val):
            if val >= 90:
                return 'background-color: #d4edda'  # verde
            elif val >= 70:
                return 'background-color: #fff3cd'  # giallo
            else:
                return 'background-color: #f8d7da'  # rosso
        
        # Costruisci tabella manuale con style
        table_rows = []
        for idx, row in df_merged.head(20).iterrows():  # Mostra top 20 attributi
            style = {}
            if row['Precisione'] >= 90:
                style = {'backgroundColor': '#d4edda'}
            elif row['Precisione'] >= 70:
                style = {'backgroundColor': '#fff3cd'}
            
            table_rows.append(html.Tr([
                html.Td(row['Attributo'], style=style),
                html.Td(f"{row['Centroide']:.3f}"),
                html.Td(f"{row['Precisione']:.1f}%", style={**style, 'fontWeight': 'bold'})
            ]))
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Attributo"),
                html.Th("Valore Medio"),
                html.Th("Affidabilità")
            ])),
            html.Tbody(table_rows)
        ], striped=False, bordered=True, hover=True, size="sm")
        
        info_text = html.Div([
            html.P("📌 Attributi ordinati per affidabilità:", className="text-muted small mb-2"),
            html.P("• Verde = Attributo molto rappresentativo del cluster", className="text-muted small mb-1"),
            html.P("• Giallo = Attributo moderatamente rappresentativo", className="text-muted small mb-1"),
            html.P("• Rosso = Attributo con alta varianza", className="text-muted small mb-2")
        ])
        
        data_section = html.Div([
            header_card,
            info_text,
            html.H6(f"Top attributi ({min(20, len(df_merged))} di {len(df_merged)} mostrati):"),
            table
        ])
    else:
        data_section = html.Div([header_card, html.P("Nessun dato disponibile")])
    
    return data_section


def eInfo(model, source, k):
    """
    Mostra informazioni dettagliate su un cluster figlio (edge).
    Visualizza: centroide del cluster, relevance degli attributi.
    - Grafico con valori NORMALIZZATI (0-1)
    - Tabella con valori DENORMALIZZATI (valori reali)
    """
    gcm = GraphClusterModel.loadModel(model)
    node = gcm.clusterModel[source]
    
    # Ottieni dati del figlio (valori normalizzati dal modello)
    df_group_norm = pd.DataFrame(node.Groups, columns=node.cols).iloc[[k]].copy()
    df_relevance = pd.DataFrame(node.Relevance, columns=node.cols).iloc[[k]]
    
    # Card intestazione
    header_card = dbc.Card([
        dbc.CardHeader(html.H5(f"🎯 Gruppo Figlio {k} (dal Nodo {source})", className="mb-0")),
        dbc.CardBody([
            html.P("Questo gruppo rappresenta un sotto-cluster con caratteristiche specifiche.", 
                   className="text-muted small")
        ], className="p-2")
    ], className="mb-3")
    
    if len(df_group_norm) > 0:
        # Prepara dati per la TABELLA (denormalizzati)
        df_group_denorm = gcm.denormalizeData(df_group_norm)
        df_group_denorm = df_group_denorm.transpose().reset_index()
        df_group_denorm.columns = ['Attributo', 'Valore']
        
        # Prepara dati per il GRAFICO (normalizzati 0-1)
        df_group_norm_chart = df_group_norm.transpose().reset_index()
        df_group_norm_chart.columns = ['Attributo', 'Valore_Norm']
        
        df_relevance = df_relevance.transpose().reset_index()
        df_relevance.columns = ['Attributo', 'Importanza']
        
        # Merge per la tabella (con valori denormalizzati)
        df_merged = df_group_denorm.merge(df_relevance, on='Attributo')
        df_merged['Importanza_%'] = (df_merged['Importanza'] * 100).round(1)
        
        # Merge per il grafico (con valori normalizzati)
        df_chart = df_group_norm_chart.merge(df_relevance, on='Attributo')
        df_chart['Valore_Norm'] = df_chart['Valore_Norm'].clip(0, 1)  # Assicura 0-1
        
        # Ordina per importanza
        df_merged = df_merged.sort_values(by='Importanza_%', ascending=False)
        df_chart = df_chart.sort_values(by='Importanza', ascending=False)
        
        # Tabella con colori per importanza RELATIVA al massimo del cluster
        # (così funziona bene anche con valori normalizzati piccoli)
        max_importance = df_merged['Importanza_%'].max() if len(df_merged) > 0 else 100
        
        table_rows = []
        for idx, row in df_merged.head(15).iterrows():
            imp = row['Importanza_%']
            # Colori basati su percentuale del valore massimo (relativo al cluster)
            rel_imp = imp / max_importance if max_importance > 0 else 0
            style = {}
            if rel_imp >= 0.7:  # Top 30%
                style = {'backgroundColor': '#d4edda', 'color': 'darkgreen', 'fontWeight': 'bold'}
            elif rel_imp >= 0.4:  # Top 60%
                style = {'backgroundColor': '#fff3cd'}
            elif rel_imp >= 0.2:  # Top 80%
                style = {'backgroundColor': '#ffe0b2'}
            
            table_rows.append(html.Tr([
                html.Td(row['Attributo']),
                html.Td(f"{row['Valore']:.3f}"),  # Valore denormalizzato
                html.Td([
                    f"{imp:.1f}%",
                    # Barra visiva della rilevanza
                    html.Div(style={
                        'width': f'{rel_imp * 100}%',
                        'height': '4px',
                        'backgroundColor': '#28a745' if rel_imp >= 0.7 else '#ffc107' if rel_imp >= 0.4 else '#fd7e14',
                        'marginTop': '2px'
                    })
                ], style={**style, 'minWidth': '80px'}),
            ]))
        
        table = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Attributo"),
                html.Th("Valore Reale"),
                html.Th(["Rilevanza", html.Br(), html.Small("(relativa al gruppo)")]),
            ])),
            html.Tbody(table_rows)
        ], striped=False, bordered=True, hover=True, size="sm")
        
        # GRAFICO con valori NORMALIZZATI (0-1)
        if len(df_chart) > 0:
            top_attrs = df_chart.head(10)
            fig = px.bar(
                top_attrs, 
                x='Attributo', 
                y='Valore_Norm',
                title='Top 10 Valori Caratteristici (Normalizzati 0-1)',
                labels={'Valore_Norm': 'Valore (0-1)', 'Attributo': ''},
                color='Importanza',
                color_continuous_scale=['red', 'yellow', 'green'],
                range_y=[0, 1]  # Fissa scala 0-1
            )
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                yaxis=dict(
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                )
            )
            # Aggiungi linea di riferimento a 0.5
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                          annotation_text="media", annotation_position="right")
            chart = dcc.Graph(figure=fig, config={'displayModeBar': False})
        else:
            chart = html.Div()
        
        legend = html.Div([
            html.P("📌 Legenda:", className="text-muted small mb-2"),
            html.P("• Grafico: valori NORMALIZZATI (0-1) per confronto", className="text-muted small mb-1"),
            html.P("• Tabella: valori REALI (denormalizzati) del dataset", className="text-muted small mb-1"),
            html.P("• Colori: rilevanza dell'attributo per il cluster", className="text-muted small mb-1")
        ])
        
        return html.Div([
            header_card,
            legend,
            chart,
            html.H6(f"Dettaglio attributi (valori reali):", className="mt-3"),
            table
        ])
    
    return html.Div([header_card, html.P("Nessun dato disponibile")])

"""@app.callback(Output('org-chart', 'layout'),
              Input('dpdn', 'value'))
def update_layout(layout_value):
    if layout_value == 'breadthfirst':
        return {
        'name': layout_value,
        'roots': '[id = "0"]',
        'animate': True
        }
    else:
        return {
            'name': layout_value,
            'animate': False
        }"""



if __name__ == '__main__':
    app.run(debug=True, port=4000)
    #root, cm=AlgebraicClusterLearner.testLearner()
    #elms=to_elements(root, labels=['class'])


# https://youtu.be/iuHFwHgQIwg
# https://dash.plotly.com/cytoscape/styling
# https://dash.plotly.com/cytoscape/reference
# https://js.cytoscape.org/#style/node-body
# https://js.cytoscape.org/#style/edge-line

