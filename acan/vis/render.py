import os
import secrets
import json
import re
from flask import session
from acan.vis.jobs import TrainingJobs
from acan.vis import generative, mathematics
from acan.vis.training import NativeTraining, graph_stats
from acan.vis.evaluation import evaluate
from acan.vis.benchmark_job import BenchmarkJob, signature as benchmark_signature
from acan.vis.refinement import root_signature, remove_branch
from acan.vis.inference import infer
from acan.vis.datasets import clean_frame, target_and_filters, split_for_training, check_budget, read_arff

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
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn import metrics
from scipy.io import loadmat
import pyarrow.parquet as pq
import dash_daq as daq
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix='/', #'/proxy/4000/'
    routes_pathname_prefix='/',
    suppress_callback_exceptions=True)  # Permette callback su componenti generati condizionalmente
app.title='ACAN · Concept Explorer'
els={}

# Signed cookies identify browser sessions; learners stay on the server.
app.server.secret_key = os.environ.get('ACAN_SESSION_SECRET') or secrets.token_hex(32)
app.server.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                         SESSION_COOKIE_SECURE=os.environ.get('ACAN_COOKIE_SECURE', '').lower() == 'true')
training_jobs = TrainingJobs()
benchmark_jobs = TrainingJobs(max_running=1)

@app.server.get('/healthz')
def health():
    return {'status':'ok','version':'8.4.0','build_revision':os.environ.get('ACAN_BUILD_REVISION','development'),'dataset_filter_control':'checkbox-v1'}

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
    parents = {child: (node.id, group) for node in gcm.clusterModel.values()
               for group, child in node.children.items()}

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
        if node.id == gcm.rootId or not show_leaves:
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
                        'source': str(node.id) if node.id == gcm.rootId else f'{parents[node.id][0]}_G{parents[node.id][1]}',
                        'target': group_id,
                        'node_id': node.id, 'group_idx': k,
                        'size': 1,
                        'label': group_color,
                        'errColor': group_color
                    },
                    'classes': 'directed'
                }
                elements[f"N{node.id}G{k}"] = edgeNode

        # Edge dai Groups ai children (se esistono)
        for k, childId in node.children.items():
            if show_leaves:
                continue  # child groups connect directly to this group, without a second root
            group_id = f"{node.id}_G{k}"
            # Cytoscape edge endpoints are immutable. Different IDs prevent the
            # React diff from losing edges when their group nodes are removed.
            edge_id = ('' if show_leaves else 'H') + f'N{node.id}K{k}'
            edgeNode={}
            edgeNode['data']={'id': edge_id,'source': group_id if show_leaves else str(node.id), 'target': str(childId), 'size':1, 'label':color, 'errColor':color}
            edgeNode['classes'] = 'directed'
            elements[edge_id]=edgeNode

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
    style={'width': '100%', 'height': '650px', 'border': '1px solid #dee2e6', 'borderRadius': '8px'},
    responsive=True,
    autoungrabify=False,
    minZoom=0.03,
    maxZoom=4,
    layout={'name': 'breadthfirst', 'directed': True, 'animate': False, 'spacingFactor': 1.3},
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
        {'selector': '.node', 'style': {
            'shape': 'round-rectangle', 'width': 180, 'height': 80,
            'font-size': 14, 'text-wrap': 'wrap', 'text-max-width': 165}},
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
        {'selector': '.local-root', 'style': {
            'shape': 'round-rectangle', 'width': 210, 'height': 72,
            'font-size': 16, 'font-weight': 'bold', 'text-wrap': 'wrap',
            'text-max-width': 195, 'border-width': 2, 'border-color': '#334155'}},
        {'selector': '.local-group', 'style': {
            'shape': 'round-rectangle', 'width': 145, 'height': 74,
            'font-size': 14, 'font-weight': 'bold', 'text-wrap': 'wrap',
            'text-max-width': 135, 'color': '#111827',
            'text-outline-color': '#fff', 'text-outline-width': 1,
            'border-width': 1, 'border-color': '#64748b'}},
        {'selector': '.has-child', 'style': {'border-width': 3, 'border-color': '#2563eb'}},
        {'selector': 'node:selected', 'style': {'border-width': 4, 'border-color': '#7c3aed'}},
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

PARAMETER_HELP = {'input-lr': 'Ampiezza degli aggiornamenti: riduci se il training è instabile.',
 'input-max-batch': 'Righe per aggiornamento, massimo effettivo 300.',
 'train-depth': 'Più livelli possono moltiplicare i prototipi. Parti da uno.',
 'train-min-samples': 'Evita approfondimenti su pochissime righe.',
 'train-purity': 'Più vicina a uno può inseguire il rumore nelle classi.',
 'train-epochs': 'Più epoche aumentano apprendimento e rischio di sovradattamento.'}


def parameter(label, id, value, lo, hi, step):
    return dbc.Col([dbc.Label(label), dbc.Input(id=id, type='number', value=value,
                    min=lo, max=hi, step=step), html.Small(PARAMETER_HELP.get(id,''),className='text-muted')], md=4, className='mb-2',
                   style={'display':'none'} if id in ('train-tau','train-alpha') else {})


model_layout = dbc.Container([
    html.H4('Scopri i concetti, poi approfondisci soltanto dove serve', className='mt-3'),
    html.P('Senza target scopri gruppi su tutte le righe; con un target apprendi a prevederne la classe su uno split 80/20. Scegli se il target è una classe (anche numerica) oppure una quantità continua. Avvia il modello iniziale. Doppio clic su un gruppo per creare un solo livello aggiuntivo sui suoi campioni. La radice rimane unica. Confronta la validazione in Evaluation e rimuovi i rami senza benefici.'),
    uplModel,
    dbc.Row([
        dbc.Col([dbc.Label('Target (svuota per scoperta non supervisionata)'),
                 dcc.Dropdown(id='dpdnLabel', value=[], multi=True, options=[]),
                 dbc.Button('Svuota target', id='btn-clear-label', size='sm', color='link')], md=7),
        dbc.Col([dbc.Button('Scopri concetti', id='button-learn', n_clicks=0, disabled=True, className='m-1'),
                 dbc.Button('Salva modello', id='btn-download', n_clicks=0, className='m-1', outline=True),
                 dcc.Download(id='download-model')], md=5),
    ], className='my-3'),
    html.Div([
        dbc.Label('Che cosa rappresenta il target?'),
        dcc.RadioItems(id='target-task',value='auto',options=[
            {'label':' Automatico · verifica il compito nella diagnostica','value':'auto'},
            {'label':' Classi, anche codici numerici','value':'classification'},
            {'label':' Una quantità da stimare, per esempio un prezzo','value':'regression'}]),
        html.P('Un numero può essere un nome di classe oppure una quantità. Per i prezzi si usa la media nei prototipi e si valutano MAE, RMSE e R². Cambiare compito richiede un nuovo training.',className='small text-muted'),
    ],id='target-task-panel',className='mb-3'),
    dbc.Accordion([dbc.AccordionItem([
        dbc.Row([
            dbc.Col([dbc.Label('Colonne categoriche'), dcc.Dropdown(id='dpdnCategory', value=[], multi=True),
                     dbc.Button('Svuota', id='btn-clear-category', color='link', size='sm')], md=6),
            dbc.Col([dbc.Label('Escludi colonne'), dbc.Checklist(id='dpdnFilter', value=[], options=[], style={'maxHeight':'180px', 'overflowY':'auto'}),
                     dbc.Button('Svuota', id='btn-clear-filter', color='link', size='sm'),
                     dbc.Button('Ripristina filtri suggeriti', id='btn-suggest-filter', color='link', size='sm')], md=6),
        ]),
        html.Div([

            dcc.Dropdown(id='tabular-routing-profile',value='cp-v9',options=[{'label':'Concetti predittivi (cp-v9)','value':'cp-v9'},{'label':'CP condivisa precedente','value':'cp-joint-v1'}]),
        ],style={'display':'none'}),
        html.P('Un solo criterio per tabulare e generativo: i concetti partono specifici e vengono generalizzati per fusione; resta il livello più generale che prevede altrettanto bene le osservazioni non viste. Il numero di concetti non è richiesto né limitato da soglie. Epoche, learning rate, livelli, purezza e campioni minimi non agiscono su questo nucleo: restano per i modelli precedenti, che rimangono caricabili. Formule, misure e limiti nella scheda Matematica.',className='small text-muted'),
        dbc.Row([parameter(*args) for args in [
            ('Epoche', 'train-epochs', 100, 1, 1000, 1),
            ('Learning rate', 'input-lr', .08, .001, 1, .01),
            ('Righe per aggiornamento', 'input-max-batch', 300, 10, 300, 10),
            ('Temperatura τ (solo versione precedente)', 'train-tau', .2, .01, 10, .01),
            ('Miscela α (solo versione precedente)', 'train-alpha', .5, 0, 1, .05),
            ('Campioni minimi del ramo', 'train-min-samples', 12, 2, 10000, 1),
            ('Livelli iniziali automatici (predefinito: 1)', 'train-depth', 1, 1, 6, 1),
            ('Soglia purezza (solo livelli automatici)', 'train-purity', .92, 0, 1, .01),
        ]]),
        html.P('Il batch è modificabile per il primo livello; la gerarchia usa aggiornamenti fino a 300 righe. La purezza e i campioni minimi agiscono solo con più livelli automatici. Approfondire può aumentare i prototipi: confronta il risultato nella validazione.', className='small text-muted'),
        html.Div(daq.BooleanSwitch(disabled=True,id='bsId',on=False),style={'display':'none'}),
    ], title='Impostazioni avanzate')], start_collapsed=True, className='mb-3'),
    dcc.Store(id='model-store'), dcc.Store(id='label-store'),
    dcc.Store(id='evaluation-trigger', data=0), dcc.Store(id='refine-request'), dcc.Store(id='graph-viewport'),
    dbc.Row([
        dbc.Col(dbc.Button('Approfondisci gruppo selezionato', id='refine-selected', outline=True, className='m-1'), md='auto'),
        dbc.Col(dbc.Button('Rimuovi approfondimento selezionato', id='remove-branch', color='warning', outline=True, className='m-1'), md='auto'),
        dbc.Col(dbc.Button('Ferma training', id='btn-stop-learn', color='danger', disabled=True, className='m-1'), md='auto'),
    ]),
    dbc.Accordion([dbc.AccordionItem([html.Pre(id='dataset-debug-view', style={'whiteSpace':'pre-wrap'}),
        dbc.Button('Scarica diagnostica dati', id='dataset-debug-download'), dcc.Download(id='dataset-debug-file'),
        dcc.Store(id='dataset-debug-data')], title='Diagnostica dati e filtri ricevuti dal server')], start_collapsed=True),
    html.Div(id='learning-status', children='Pronto: carica i dati e scopri i concetti.', className='alert alert-light my-2', role='status'),
    dcc.Interval(id='learning-monitor', interval=1000, disabled=True),
    html.Div(id='graph-summary', className='alert alert-info'),
    html.Div([
    dbc.Row([
        dbc.Col(dcc.RadioItems(id='graph-view', options=[{'label': ' Concetti e gruppi', 'value': 'full'},
                   {'label': ' Solo gerarchia', 'value': 'hierarchy'}], value='full', inline=True), md=7),
        dbc.Col([dbc.Label('Disposizione panoramica', html_for='graph-layout'),
            dcc.Dropdown(id='graph-layout', options=['breadthfirst', 'cose', 'concentric'],
                   value='cose', clearable=False)], md=5),
    ], className='mb-3'),
    dbc.Row([
        dbc.Col([dbc.Label('Concetto da esplorare'), dcc.Dropdown(id='graph-focus', clearable=False)], md=7),
        dbc.Col([dbc.Label('Gruppi del concetto'), dcc.Dropdown(id='graph-page', value=1, clearable=False)], md=5),
    ], className='mb-2'),
    dbc.Checkbox(id='graph-empty', value=False, label='Includi prototipi senza assegnazioni hard'),
    html.Div([
        dbc.Button('Radice', id='graph-root', outline=True, size='sm', className='m-1'),
        dbc.Button('Sali al padre', id='graph-up', outline=True, size='sm', className='m-1'),
        dbc.Button('−', id='graph-zoom-out', outline=True, size='sm', className='m-1', title='Riduci zoom'),
        dbc.Button('+', id='graph-zoom-in', outline=True, size='sm', className='m-1', title='Aumenta zoom'),
        dbc.Button('Centra vista', id='graph-fit', outline=True, size='sm', className='m-1'),
        dbc.Button('Schermo intero', id='graph-fullscreen', outline=True, size='sm', className='m-1'),
    ]),
    html.Div(id='graph-window-summary', className='small text-muted my-2', role='status'),
    dbc.Row([dbc.Col(orgChartData, md=8), dbc.Col([html.Div(id='node-detail'), html.Div(id='edge-detail')], md=4)]),
    html.P('Doppio clic sui gruppi con bordo blu per aprire il concetto figlio. Sugli altri gruppi richiede un approfondimento. La panoramica mostra tutta la gerarchia.', className='small text-muted mt-2'),
    ], id='graph-panel'),
    dbc.Accordion([dbc.AccordionItem([
        dbc.Label('Nodo concetto'), dcc.Dropdown(id='concept-node', clearable=False),
        dcc.Graph(id='concept-relevance'), html.Div(id='concept-rules'),
        html.P('Le rilevanze non sono probabilità di correttezza. Le regole sono estratte da extractRules.', className='small text-muted'),
        dbc.Button('Esporta regole di tutta la gerarchia', id='export-rules'), dcc.Download(id='rules-download'),
    ], title='Laboratorio dei concetti · rilevanze e regole native')], start_collapsed=True, className='my-3'),
])


evaluation_layout = dbc.Container([
    html.H4('Benchmark sul tuo dataset', className='mt-3'),
    html.P('Il confronto prima/dopo usa sempre lo stesso holdout 20% (seed 42). Se scegli i rami in base a questi risultati, questo è un set di validazione: le metriche non sono una stima indipendente dopo aver selezionato i rami. Non occorrono altri file: vengono usati i dati già caricati. Metriche per target, senza confondere classificazione e regressione.'),
    html.P('Confronta ACAN con KNN, CART, Random Forest, Extra Trees, HistGradientBoosting, XGBoost e CatBoost: stessi dati, split e target. Budget CPU predefiniti, senza ricerca degli iperparametri.'),
    dbc.Button('Confronta con gli altri modelli', id='benchmark-run', className='m-1'),
    dbc.Button('Ferma confronto', id='benchmark-stop', color='danger', outline=True, className='m-1'),
    html.Div(id='benchmark-status', role='status'),
    dcc.Interval(id='benchmark-monitor', interval=1000, disabled=True), dcc.Store(id='benchmark-report'),
    dcc.Graph(id='errorIndicator'), html.Div(id='cm-detail'),
    dbc.Button('Scarica predizioni e metriche', id='button-prediction', className='my-2'),
    dcc.Download(id='download-prediction'),

])


inference_layout = dbc.Container([
    html.P('Predizione: attraversa la gerarchia nativa e mostra il percorso. Completamento: usa predict_masked, che nel core attuale opera sulla radice. Nessun nuovo training.'),
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

_page_layout = dbc.Container([
    html.Div([html.H1('ACAN · Concept Explorer'), html.P('Dati → concetti → gerarchia → predizione e completamento', className='text-muted')], className='p-4 mb-3 bg-light border rounded'),
    dcc.Tabs(id='tabs', value='Data', children=[
        dcc.Tab(label=name, value=name) for name in ['Data', 'Model', 'Evaluation', 'Inference', 'Generative', 'Matematica']
    ]),
    # Keep the graph mounted when switching tabs: destroying Cytoscape while
    # asynchronous layout/resize callbacks are pending causes browser errors.
    html.Div([data_layout, dcc.Store(id='data-store')], id='page-data'),
    html.Div([model_layout, dcc.Store(id='test-store')], id='page-model', style={'display': 'none'}),
    html.Div([evaluation_layout, dcc.Store(id='eval-store')], id='page-evaluation', style={'display': 'none'}),
    html.Div(inference_layout, id='page-inference', style={'display': 'none'}),
    html.Div(generative.layout, id='page-generative', style={'display': 'none'}),
    html.Div(mathematics.layout, id='page-mathematics', style={'display':'none'}),
], fluid=True)


def serve_layout():
    # New for every page load, including tabs sharing the same browser cookie.
    return html.Div([dcc.Store(id='tab-session', data=secrets.token_urlsafe(24)), _page_layout])


app.layout = serve_layout


app.clientside_callback(
    """function(tab, layout, model) {
        if (tab !== 'Model') return window.dash_clientside.no_update;
        let tries = 0;
        function resize() {
            const el = document.getElementById('org-chart');
            if (!el || !el.offsetWidth || !el._cyreg) {
                if (++tries < 30) setTimeout(resize, 50);
                return;
            }
            const cy = el._cyreg.cy;
            cy.resize();
            cy.fit(cy.elements(), 35);
        }
        setTimeout(resize, 150);
        return Date.now();
    }""",
    Output('graph-viewport', 'data'), Input('tabs', 'value'),
    State('graph-layout', 'value'), State('model-store', 'data'))


def tab_job_key(tab_id):
    if not isinstance(tab_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{32}', tab_id):
        raise ValueError('Sessione della scheda non valida: ricaricare la pagina.')
    return session['acan_session'], tab_id


@app.callback([Output('page-data', 'style'), Output('page-model', 'style'),
               Output('page-evaluation', 'style'), Output('page-inference', 'style'), Output('page-generative', 'style'), Output('page-mathematics','style')],
              Input('tabs', 'value'))
def show_page(tab):
    return [{'display': 'block' if tab == name else 'none'}
            for name in ['Data', 'Model', 'Evaluation', 'Inference', 'Generative', 'Matematica']]


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

@app.callback(Output('dpdnFilter', 'value', allow_duplicate=True),
              Input('btn-suggest-filter', 'n_clicks'), State('data-store', 'data'),
              State('label-store', 'data'), prevent_initial_call=True)
def suggest_filters(clicks, data, labels):
    if not clicks or not data:
        return dash.no_update
    frame = clean_frame(pd.DataFrame(data))
    if frame.empty:
        return []
    return target_and_filters(frame, labels)[1]


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
              State("dpdnLabel", "options"),
              State('data-store', 'data'))
def set_labels(labels, opts, data):
    labels = labels or []
    # A loaded model restores target options only. Feature filters must continue
    # to describe the uploaded table, otherwise the dropdown discards selections.
    columns = list(pd.DataFrame(data).columns) if data else [item['value'] for item in (opts or [])]
    return labels, [{'label': name, 'value': name} for name in columns if name not in labels]

@app.callback(
    [Output("dpdnLabel", "options"),Output("dpdnLabel", "value"), Output("dpdnCategory", "options"), Output("dpdnFilter", "value"),
     Output("dpdnFilter", "options", allow_duplicate=True)] ,
    Input('data-store', 'data'),
    State('label-store', 'data'),
    prevent_initial_call=True,
)
def update_labels(data, lbs):
    df = clean_frame(pd.DataFrame(data))
    if df.empty:
        return [], [], [], [], []
    labels, filters = target_and_filters(df, lbs)
    options = [{'label': name, 'value': name} for name in df.columns]
    filter_options = [item for item in options if item['value'] not in labels]
    return options, labels, options, filters, filter_options



@app.callback(Output('button-learn','children'),Output('target-task-panel','style'),Input('dpdnLabel','value'),Input('target-task','value'))
def training_action_label(labels,task):
    return ('Apprendi a prevedere valori' if task=='regression' else 'Apprendi a prevedere il target',{}) if labels else ('Scopri gruppi senza target',{'display':'none'})


@app.callback(Output('input-max-batch','disabled'),Output('train-min-samples','disabled'),Output('train-purity','disabled'),Input('train-depth','value'),Input('dpdnLabel','value'))
def active_training_parameters(depth,labels):
    hierarchical=int(depth or 1)>1
    return hierarchical,not hierarchical,not hierarchical or not labels


@app.callback(Output('train-tau','disabled'),Output('train-alpha','disabled'),Input('tabular-routing-profile','value'))
def routing_legacy_controls(profile):
    return profile != 'legacy-v8', profile != 'legacy-v8'


@app.callback(Output('button-learn','disabled'), Input('data-store','data'),
              Input('label-store','data'), Input('dpdnLabel','value'), Input('dpdnLabel','options'),
              Input('dpdnFilter','value'), Input('dpdnFilter','options'))
def training_controls_pending(data, labels, selected, options, filters, filter_options):
    if not data or not options or list(labels or []) != list(selected or []):
        return True
    columns = set(data)
    available = {item['value'] for item in (filter_options or [])}
    return not (set(labels or []).issubset(columns) and available == columns-set(labels or [])
                and set(filters or []).issubset(available))


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
            extension = filename.rsplit('.', 1)[-1].casefold()
            if extension == 'csv':
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
                
            elif extension == 'json':
                # Assume that the user uploaded a CSV file
                df = pd.read_json(
                    io.StringIO(decoded.decode('utf-8')))
                
            elif extension == 'xlsx':
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')

            elif extension == 'parquet':
                arrTb=pq.read_table(io.BytesIO(decoded))
                df=arrTb.to_pandas(date_as_object=False)

            elif extension == 'arff':
                df = read_arff(decoded)

            elif extension == 'mat':
                mat = loadmat(io.BytesIO(decoded))
                mat = {k:v for k, v in mat.items() if k[0] != '_'}
                mdata = mat[next(iter(mat))]  # variable in mat file
                mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
                ndata = {n: mdata[n][0, 0] for n in mdtype.names}
                df=pd.DataFrame(ndata['data'])
            else:
                raise ValueError('Formato non supportato: usa CSV, JSON, XLSX, Parquet, ARFF o MAT.')

            df = clean_frame(df)
            original_rows = len(df)
            if original_rows > 10000:
                df = df.sample(10000, random_state=42)
            detected, excluded = target_and_filters(df)
            message = f'{filename}: {original_rows} righe, {len(df.columns)} colonne. Target suggerito: {", ".join(detected)}. Colonne escluse modificabili: {", ".join(excluded) or "nessuna"}.'
            if original_rows > 10000:
                message += ' Campione interattivo di 10.000 righe (seed 42).'
            return df.to_dict(orient='list'), balanced, html.Div([uplData, dbc.Alert(message, color='info')])
        except Exception as e:
            print(e)
            return pd.DataFrame().to_dict(), balanced, html.Div([uplData, dbc.Alert(f'File non elaborabile: {e}', color='danger')])

    

    return pd.DataFrame().to_dict(), balanced, uplData



@app.callback(Output('dataset-debug-view','children'),Output('dataset-debug-data','data'),
              Input('data-store','data'),Input('label-store','data'),Input('dpdnFilter','value'),
              Input('dpdnCategory','value'),Input('target-task','value'))
def dataset_debug(data,labels,filters,categories,task):
    if not data:
        return 'Carica un dataset.',None
    report=dict(received_filters=filters or [],received_targets=labels or [],categorical_columns=categories or [],task=task)
    try:
        from acan.vis.datasets import encoding_size
        train,test,info=split_for_training(data,labels or [],filters,categories,task or 'auto')
        report.update(dataset=info,encoding=encoding_size(train))
        try:
            check_budget(train);report['budget']='ok'
        except ValueError as exc:
            report['budget']=str(exc)
    except Exception as exc:
        report['error']=str(exc)
    return json.dumps(report,indent=2,ensure_ascii=False),report


@app.callback(Output('dataset-debug-file','data'),Input('dataset-debug-download','n_clicks'),
              State('dataset-debug-data','data'),prevent_initial_call=True)
def download_dataset_debug(clicks,report):
    return dict(content=json.dumps(report,indent=2,ensure_ascii=False),filename='acan-dataset-diagnostics.json') if report else dash.no_update


@app.callback(Output("data-info", "children"),
              Input('data-store', 'data'))
def update_dataInfo(data):
    df=pd.DataFrame(data)
    if len(df)>0:
        df= df.describe(include='all').transpose().reset_index()
    return dbc.Table.from_dataframe(df, striped=True, bordered=False, hover=True)


def training_context(tab_id, model_json):
    job = training_jobs.get(tab_job_key(tab_id))
    if not job or not isinstance(job['learner'], NativeTraining):
        return None
    learner = job['learner']
    reference = learner.graph.to_json() if learner.graph is not None else learner.base_model
    if reference and root_signature(reference) == root_signature(model_json):
        return learner
    return None


@app.callback(Output('errorIndicator', 'figure'), Output('cm-detail', 'children'), Output('eval-store', 'data'),
              Input('model-store', 'data'), Input('benchmark-report', 'data'),
              State('tab-session', 'data'))
def evaluate_model(model, benchmark_report, tab_id):
    if not model:
        return go.Figure(), 'Carica o addestra un modello.', None
    try:
        context = training_context(tab_id, model)
        validation = getattr(context, 'validation', None)
        test = validation
        if test is None:
            return go.Figure(), 'Addestra il modello sul dataset caricato in Dati. Il solo JSON non conserva i campioni per il benchmark.', None
        if benchmark_report and benchmark_report.get('signature') == benchmark_signature(model, test):
            report = dict(benchmark_report)
        else:
            report = evaluate(model, test, train=context.data if context else None, before_json=context.base_model if context else None, baselines=False)
        source = 'Validazione 20% del dataset in Dati (riutilizzata per scegliere i rami)'
        report['source'] = source
        table = pd.DataFrame(report['scores'])
        figure = px.bar(table, x='Target', y='Valore', color='Modello', facet_col='Metrica',
                        barmode='group', title=source)
        figure.update_yaxes(matches=None)
        summary = table.pivot(index=['Target', 'Modello'], columns='Metrica', values='Valore').reset_index()
        stats = graph_stats(GraphClusterModel.loadModel(model))
        dataset_info = getattr(context, 'dataset_info', None)
        if dataset_info:
            report['dataset'] = dataset_info
        panels = [html.P(f"{source} · {report['rows']} righe · {stats['nodes']} nodi · {stats['groups']} gruppi · {stats['depth']} livelli. Accuracy/F1 in [0,1]; MAE/RMSE nelle unità originali."),
                  dbc.Table.from_dataframe(summary.round(5), striped=True, responsive=True)]
        panels.extend(dbc.Alert(item['warning'], color='warning') for item in report.get('diagnostics', []))
        if dataset_info:
            panels.append(html.P(f"Training: {dataset_info['train_rows']} righe; validazione: {dataset_info['validation_rows']}; righe senza target escluse: {dataset_info['excluded_missing_targets']}; split stratificato: {'sì' if dataset_info['stratified'] else 'no'}. Target: {dataset_info['target_types']}."))
        if report.get('comparisons'):
            panels.insert(0, html.Div([html.H5('Beneficio del cambiamento'),
                dbc.Table.from_dataframe(pd.DataFrame(report['comparisons']).round(5), striped=True, responsive=True),
                html.P('MAE/RMSE: meno è meglio. Accuracy/F1/R²: più è meglio. Puoi rimuovere il ramo dal grafo se il confronto non giustifica la complessità aggiunta.')]))
        if report['mode'] == 'masked-root':
            panels.insert(0, dbc.Alert(f"Modello non supervisionato: {report['masked_cells']} celle mascherate (seed 42), massimo 100 righe. Completamento sulla radice: approfondire i figli non cambia questa metrica. Per confrontare i predittori supervisionati seleziona un target in Model.", color='info'))
        if report.get('timings'):
            panels.append(html.Div([html.H5('Tempi misurati CPU'),
                html.P('ACAN è già addestrato; il suo tempo indicato riguarda l’inferenza. I tempi di training delle baseline sono misurati ora. Non confrontare un tempo mancante con zero.'),
                dbc.Table.from_dataframe(pd.DataFrame(report['timings']).round(5), striped=True, responsive=True)]))
        for item in report['details']:
            if item['kind'] == 'classification':
                chart = go.Figure(go.Heatmap(z=item['matrix'], x=item['labels'], y=item['labels'], colorscale='Blues'))
                chart.update_layout(title=f"Matrice di confusione · {item['target']}", xaxis_title='Predetto', yaxis_title='Reale')
            else:
                chart = go.Figure(go.Scatter(x=item['truth'], y=item['prediction'], mode='markers'))
                chart.update_layout(title=f"Reale / predetto · {item['target']}", xaxis_title='Reale', yaxis_title='Predetto')
            panels.append(dcc.Graph(figure=chart))
        return figure, panels, report
    except Exception as exc:
        return go.Figure(), dbc.Alert(f'Valutazione non disponibile: {exc}', color='warning'), None


@app.callback(Output('download-prediction', 'data'), Input('button-prediction', 'n_clicks'),
              State('eval-store', 'data'), prevent_initial_call=True)
def download_prediction(clicks, report):
    if report:
        return dict(content=json.dumps(report, indent=2), filename='acan-evaluation.json')
    return dash.no_update


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
              State('train-depth', 'value'),
              State('train-epochs', 'value'),
              State('train-min-samples', 'value'),
              State('train-tau', 'value'),
              State('train-alpha', 'value'),
              State('train-purity', 'value'),
              State('target-task', 'value'),
              State('tabular-routing-profile','value'),
              State('tab-session', 'data'),
              prevent_initial_call=True
              )
def update_model(n_clicks, data, labels, filters, categories, max_batch, lr, depth, epochs, min_samples, tau, alpha, purity, task, routing_profile, tab_id):
    if not n_clicks:
        return (dash.no_update,) * 5
    if not data:
        return dash.no_update, dash.no_update, "⚠️ Nessun dato caricato!", True, True
    try:
        key = tab_job_key(tab_id)
        existing = training_jobs.get(key)
        if existing and not existing['done'].is_set():
            return dash.no_update, dash.no_update, "Training già in corso.", False, False
        train, test, dataset_info = split_for_training(data, labels or [], filters, categories, task)
        check_budget(train)
        if routing_profile not in ('legacy-v8','paper-routing-v1','cp-routing-v1','cp-joint-v1','cp-v9'):
            raise ValueError('Seleziona un profilo di routing ACAN valido.')
        options = dict(routing_profile=routing_profile,max_depth=depth, max_epochs=epochs, min_samples_split=min_samples,
                       tau=tau, alpha=alpha, similarity_threshold=purity, lr=lr,
                       batch_size=max_batch, device='numpy')
        bounds = dict(max_depth=(1, 6), max_epochs=(1, 1000), min_samples_split=(2, 10000),
                      tau=(.01, 10), alpha=(0, 1), similarity_threshold=(0, 1),
                      lr=(.001, 1), batch_size=(10, 10000))
        for name, (lo, hi) in bounds.items():
            value = options[name]
            if value is None or not lo <= value <= hi:
                raise ValueError(f'Parametro {name}: atteso valore tra {lo} e {hi}.')
        for name in ('max_depth', 'max_epochs', 'min_samples_split', 'batch_size'):
            if int(options[name]) != options[name]:
                raise ValueError(f'{name} deve essere intero.')
            options[name] = int(options[name])
        learner = NativeTraining(train, labels or [], options)
        learner.validation = test
        learner.dataset_info = dataset_info
        training_jobs.start(key, learner, {}, labels or [])
        return dash.no_update, test.to_dict(), f"🟢 Apprendimento avviato: {len(train)} training, {len(test)} validazione; {dataset_info['excluded_missing_targets']} righe senza target escluse.", False, False
    except Exception as exc:
        return dash.no_update, dash.no_update, f"❌ {exc}", True, True

@app.callback([Output('model-store', 'data', allow_duplicate=True),
            Output('org-chart', 'elements', allow_duplicate=True),
            Output('learning-status', 'children', allow_duplicate=True),
            Output('btn-stop-learn', 'disabled', allow_duplicate=True),
            Output('learning-monitor', 'disabled', allow_duplicate=True),
            Output('evaluation-trigger', 'data', allow_duplicate=True)],
            Input('learning-monitor', 'n_intervals'),
            State('tab-session', 'data'),
            prevent_initial_call=True)
def monitor_learning(n_intervals, tab_id):
    try:
        job = training_jobs.get(tab_job_key(tab_id))
    except ValueError as exc:
        return dash.no_update, dash.no_update, str(exc), True, True, dash.no_update
    if job is None:
        return dash.no_update, dash.no_update, 'Nessun training attivo.', True, True, dash.no_update
    learner = job['learner']
    if not job['done'].is_set():
        # Do not run forward_pass/setModel on parameters being updated by training.
        return (dash.no_update, dash.no_update,
                f"🟡 Iterazione: {learner.iteration}", False, False, dash.no_update)
    if job['error']:
        return dash.no_update, dash.no_update, f"❌ {job['error']}", True, True, dash.no_update
    gcm = learner.graph
    stats = graph_stats(gcm)
    return (gcm.to_json(), dash.no_update,
            f"✅ Completato: {stats['nodes']} nodi concetto | {stats['depth']} livelli | {stats['groups']} gruppi",
            True, True, secrets.randbits(32))


@app.callback([Output('learning-monitor', 'disabled', allow_duplicate=True),
               Output('btn-stop-learn', 'disabled', allow_duplicate=True),
               Output('learning-status', 'children', allow_duplicate=True)],
              Input('btn-stop-learn', 'n_clicks'),
              State('tab-session', 'data'),
              prevent_initial_call=True)
def stop_learning(n_clicks, tab_id):
    if n_clicks and training_jobs.stop(tab_job_key(tab_id)):
        return dash.no_update, dash.no_update, '⏹️ Stop richiesto...'
    return dash.no_update, dash.no_update, dash.no_update

# Restore target controls with the saved model, even before a dataset is uploaded.
@app.callback(
    Output('model-store', 'data', allow_duplicate=True),
    Output('dpdnLabel', 'options', allow_duplicate=True),
    Output('dpdnLabel', 'value', allow_duplicate=True),
    Output('learning-status', 'children', allow_duplicate=True),
    Input('upload-model', 'contents'), State('upload-model', 'filename'),
    State('tab-session', 'data'),
    prevent_initial_call=True)
def load_model_json(contents, filename, tab_id):
    if not contents:
        return (dash.no_update,) * 4
    try:
        active = training_jobs.get(tab_job_key(tab_id))
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
              Input("model-store", "data"), Input('graph-view', 'value'),
              Input('graph-focus', 'value'), Input('graph-page', 'value'), Input('graph-empty', 'value'),
              prevent_initial_call=True)
def update_graph(model, view='full', focus=None, page=1, include_empty=False):
    if not model:
        return []
    gcm=GraphClusterModel.loadModel(model)
    if view != 'hierarchy':
        from .graph_view import elements
        return elements(gcm,focus,page,include_empty)
    els=to_elements(gcm, show_leaves=False)
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
            source = int(sed[0].get('node_id', str(sed[0]['source']).split('_G')[0]))
            if 'group_idx' in sed[0]:
                return eInfo(model, source, int(sed[0]['group_idx']))

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







@app.callback(Output('benchmark-run', 'disabled'), Input('model-store', 'data'))
def benchmark_available(model):
    return not model or not GraphClusterModel.loadModel(model).labels


@app.callback(Output('benchmark-status', 'children'), Output('benchmark-monitor', 'disabled'),
              Output('benchmark-report', 'data'), Input('benchmark-run', 'n_clicks'),
              Input('benchmark-stop', 'n_clicks'), Input('model-store', 'data'),
              State('tab-session', 'data'), prevent_initial_call=True)
def start_benchmark(run, stop, model, tab_id):
    key = tab_job_key(tab_id)
    if dash.ctx.triggered_id != 'benchmark-run':
        benchmark_jobs.stop(key)
        message = 'Arresto richiesto.' if dash.ctx.triggered_id == 'benchmark-stop' else 'Confronto pronto sui dati correnti.'
        return message, True, None
    try:
        context = training_context(tab_id, model) if model else None
        if context is None:
            raise ValueError('Addestra ACAN sul dataset della pagina per confrontare tutti i modelli sullo stesso training.')
        test = context.validation
        job = BenchmarkJob(model, test, context.data, context.base_model)
        benchmark_jobs.start(key, job, {}, context.labels)
        return 'Confronto in corso in un processo CPU separato; puoi fermarlo.', False, None
    except Exception as exc:
        return str(exc), True, None


@app.callback(Output('benchmark-status', 'children', allow_duplicate=True),
              Output('benchmark-monitor', 'disabled', allow_duplicate=True),
              Output('benchmark-report', 'data', allow_duplicate=True),
              Input('benchmark-monitor', 'n_intervals'), State('tab-session', 'data'), prevent_initial_call=True)
def monitor_benchmark(interval, tab_id):
    job = benchmark_jobs.get(tab_job_key(tab_id))
    if not job:
        return 'Nessun confronto attivo.', True, None
    if not job['done'].is_set():
        return 'Benchmark KNN, ensemble e boosting in corso sui dati della pagina…', False, dash.no_update
    if job['error']:
        return f"Confronto non disponibile: {job['error']}", True, None
    return 'Confronto completato: metriche e tempi misurati sono riportati sotto.', True, job['learner'].report

@app.callback(Output('model-store', 'data', allow_duplicate=True),
              Output('learning-status', 'children', allow_duplicate=True),
              Output('btn-stop-learn', 'disabled', allow_duplicate=True),
              Output('learning-monitor', 'disabled', allow_duplicate=True),
              Input('refine-request', 'data'), Input('refine-selected', 'n_clicks'), Input('remove-branch', 'n_clicks'),
              State('org-chart', 'selectedNodeData'), State('model-store', 'data'),
              State('tab-session', 'data'), prevent_initial_call=True)
def edit_hierarchy(request, refine_clicks, remove_clicks, selected, model_json, tab_id):
    try:
        if not model_json:
            raise ValueError('Addestra prima il modello sul tuo dataset.')
        key = tab_job_key(tab_id)
        active = training_jobs.get(key)
        if active and not active['done'].is_set():
            return dash.no_update, 'Attendi la fine del training corrente.', False, False
        context = training_context(tab_id, model_json)
        if dash.ctx.triggered_id == 'refine-request':
            node_id = request['node_id']
        else:
            if not selected:
                raise ValueError('Seleziona un gruppo o un nodo nel grafo.')
            node_id = selected[0]['id']
        if dash.ctx.triggered_id == 'remove-branch':
            graph = remove_branch(model_json, node_id)
            if context:
                context.base_model = model_json
                context.graph = graph
            return graph.to_json(), 'Ramo rimosso. Evaluation confronta ora prima e dopo la rimozione.', True, True
        if context is None:
            raise ValueError('Per approfondire servono i dati di training di questa pagina. Un JSON caricato da solo non li contiene: addestra il modello sul tuo CSV.')
        from acan.vis.refinement import parse_group
        parse_group(node_id)
        learner = NativeTraining(context.data, context.labels, context.options,
                                 base_model=model_json, branch=node_id)
        learner.validation = context.validation
        learner.dataset_info = getattr(context, 'dataset_info', None)
        training_jobs.start(key, learner, {}, context.labels)
        return dash.no_update, f'Approfondimento di {node_id} in corso: solo un nuovo nodo, radice invariata.', False, False
    except Exception as exc:
        return dash.no_update, str(exc), True, True

@app.callback(Output('concept-node', 'options'), Output('concept-node', 'value'),
              Input('model-store', 'data'))
def list_concepts(model):
    if not model:
        return [], None
    graph = GraphClusterModel.loadModel(model)
    return [dict(label=f'Nodo {key} · {node.n} campioni · {node.k()} gruppi', value=key)
            for key, node in graph.clusterModel.items()], graph.rootId


@app.callback(Output('concept-relevance', 'figure'), Output('concept-rules', 'children'),
              Input('concept-node', 'value'), Input('model-store', 'data'))
def inspect_concept(node_id, model):
    if not model or node_id is None:
        return go.Figure(), []
    graph = GraphClusterModel.loadModel(model)
    if int(node_id) not in graph.clusterModel:
        return go.Figure(), []
    node = graph.clusterModel[int(node_id)]
    figure = go.Figure(go.Heatmap(z=node.Relevance, x=node.cols,
                                 y=[f'G{k}' for k in range(node.k())], colorscale='Viridis'))
    figure.update_layout(title=f'Rilevanza per gruppo · nodo {node.id}', xaxis_title='Attributi', yaxis_title='Gruppi')
    rows = []
    for rule in node.extractRules(graph.minMax):
        rows.append(dict(Concetto=rule['concept_id'], Campioni=rule['support_count'],
                         Frequenza=rule['support_freq'], Attributi='; '.join(
                             f"{item['attribute']} ≈ {item['centroid_value']} (r={item['relevance']})"
                             for item in rule['top_attributes'])))
    return figure, dbc.Table.from_dataframe(pd.DataFrame(rows), striped=True, responsive=True)


@app.callback(Output('rules-download', 'data'), Input('export-rules', 'n_clicks'),
              State('model-store', 'data'), prevent_initial_call=True)
def export_rules(clicks, model):
    if not model:
        return dash.no_update
    graph = GraphClusterModel.loadModel(model)
    rules = {str(key): node.extractRules(graph.minMax) for key, node in graph.clusterModel.items()}
    return dict(content=json.dumps(rules, indent=2), filename='acan-concept-rules.json')

@app.callback(Output('graph-summary', 'children'), Output('org-chart', 'layout'),
              Input('model-store', 'data'), Input('graph-layout', 'value'),
              Input('graph-view', 'value'), Input('graph-focus', 'value'))
def describe_graph(model, layout, view='full', focus=None):
    if not model:
        return 'Carica un modello o avvia la scoperta dei concetti.', {'name': layout, 'animate': False}
    graph = GraphClusterModel.loadModel(model)
    stats = graph_stats(graph)
    root = graph.clusterModel[graph.rootId]
    config = dict(name=layout if view=='hierarchy' else 'preset', animate=False, fit=True, padding=35)
    if layout == 'breadthfirst':
        config.update(directed=True, roots=f'[id = "{graph.rootId}"]', spacingFactor=1.3)
    return (f"Radice {graph.rootId} · {stats['nodes']} nodi concetto · {stats['depth']} livelli · "
            f"{stats['groups']} prototipi complessivi · {root.n} campioni · {len(root.cols)} attributi · routing {getattr(root,'routing_profile','legacy-v8')}. "
            'Seleziona un nodo o un collegamento per esplorare centroidi, rilevanze e dispersione.'), config


@app.callback(Output('graph-focus', 'options'), Output('graph-focus', 'value'),
              Output('graph-page', 'options'), Output('graph-page', 'value'),
              Output('graph-window-summary', 'children'),
              Input('model-store', 'data'), Input('graph-focus', 'value'), Input('graph-page', 'value'),
              Input('graph-empty', 'value'), Input('graph-root', 'n_clicks'), Input('graph-up', 'n_clicks'))
def navigate_graph(model, focus, page, include_empty, root_clicks, up_clicks):
    if not model:
        return [],None,[],1,'Carica un modello per esplorare i concetti.'
    from .graph_view import navigation
    trigger = dash.ctx.triggered_id
    action = {'graph-root':'root','graph-up':'up'}.get(trigger)
    if trigger in ('model-store','graph-focus','graph-empty','graph-root','graph-up'):
        page = 1
    return navigation(GraphClusterModel.loadModel(model),focus,page,include_empty,action)

generative.register(app)


if __name__ == '__main__':
    app.run(debug=True, port=4000)
    #root, cm=AlgebraicClusterLearner.testLearner()
    #elms=to_elements(root, labels=['class'])


# https://youtu.be/iuHFwHgQIwg
# https://dash.plotly.com/cytoscape/styling
# https://dash.plotly.com/cytoscape/reference
# https://js.cytoscape.org/#style/node-body
# https://js.cytoscape.org/#style/edge-line

