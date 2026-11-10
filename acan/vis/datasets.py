"""Dataset typing and actionable resource preflight for the research UI."""
import math
import io
import csv
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

TARGET_NAMES = ('class','label','target','survived','income','gioco','outcome','response')


def _arff_tokens(text):
    values, current, quote, escaped = [], [], None, False
    for char in text:
        if escaped:
            current.append(char); escaped = False
        elif char == '\\' and quote:
            current.append(char); escaped = True
        elif quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char; current.append(char)
        elif char == ',':
            values.append(''.join(current).strip()); current = []
        else:
            current.append(char)
    if quote:
        raise ValueError('Valore ARFF con virgolette non chiuse.')
    values.append(''.join(current).strip())
    return values


def read_arff(content):
    """Normalize lexical spacing/quoting for SciPy's numeric/nominal ARFF reader."""
    from scipy.io import arff
    lines, data = [], False
    for line in content.decode('utf-8-sig').splitlines():
        match = re.match(r'^(\s*@attribute\s+.*?)\{(.*)\}(\s*(?:%.*)?)$', line, re.I)
        if not data and match:
            line = match[1] + '{' + ','.join(_arff_tokens(match[2])) + '}' + match[3]
        elif data and line.strip() and not line.lstrip().startswith(('%', '{')):
            values = []
            for token in _arff_tokens(line):
                if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
                    quote = token[0]
                    token = token[1:-1].replace('\\'+quote, quote).replace('\\\\', '\\')
                values.append(token)
            stream = io.StringIO()
            csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator='').writerow(values)
            line = stream.getvalue()
        lines.append(line)
        if line.strip().casefold() == '@data':
            data = True
    values, _ = arff.loadarff(io.StringIO('\n'.join(lines)))
    return clean_frame(pd.DataFrame(values))


def clean_frame(frame):
    frame = frame.copy()
    frame.columns = frame.columns.map(lambda value: str(value).strip())
    if not frame.columns.is_unique:
        raise ValueError('Nomi colonna duplicati dopo la normalizzazione.')
    for column in frame:
        if not is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: value.decode('utf-8').strip() if isinstance(value,bytes)
                                               else value.strip() if isinstance(value,str) else value)
            frame[column] = frame[column].replace({'?': np.nan, '': np.nan})
    return frame


def target_and_filters(frame, previous=None):
    labels = [name for name in (previous or []) if name in frame]
    if not labels:
        by_name = {name.casefold():name for name in frame}
        labels = next(([by_name[name]] for name in TARGET_NAMES if name in by_name), [frame.columns[-1]])
    filters = []
    for name in frame:
        if name in labels:
            continue
        values = frame[name]
        unique = values.nunique()
        observed = max(int(values.notna().sum()),1)
        identifier = name.casefold().replace(' ','').replace('.','') in {'id','passengerid','rowid','recordid','index'} or name.casefold().endswith('_id')
        text = not is_numeric_dtype(values)
        constant = unique == 0 or (unique == 1 and not values.isna().any())
        if constant or (identifier and unique / observed > .95) or (text and unique > max(32,.5*observed)):
            filters.append(name)
    return labels, filters


def split_for_training(data, labels, filters=(), categories=(), task='auto'):
    if task not in ('auto', 'classification', 'regression'):
        raise ValueError('Tipo di previsione non valido.')
    frame = clean_frame(pd.DataFrame(data))
    raw_rows = len(frame)
    if set(labels) - set(frame):
        raise ValueError('Target assente dal dataset.')
    if set(labels) & set(filters or []):
        raise ValueError('Un target è stato escluso: rimuovilo dai filtri o cambia target.')
    frame = frame.drop(columns=filters or [],errors='ignore')
    for name in categories or []:
        if name in frame:
            frame[name] = frame[name].map(lambda value:str(value) if pd.notna(value) else np.nan)
    types = {}
    for name in labels:
        values = frame[name].dropna()
        integral_class = is_numeric_dtype(values) and 1 < values.nunique() <= 20 and bool((values % 1 == 0).all())
        classification = task == 'classification' or (task == 'auto' and (not is_numeric_dtype(values) or integral_class))
        if classification:
            frame[name] = frame[name].map(lambda value:str(value) if pd.notna(value) else np.nan)
            types[name] = 'classification'
        else:
            frame[name] = pd.to_numeric(frame[name],errors='raise')
            types[name] = 'regression'
    if labels:
        frame = frame.dropna(subset=labels)
        if len(set(frame.columns)-set(labels)) == 0:
            raise ValueError('Serve almeno un attributo di ingresso oltre ai target.')
    if len(frame) < 5:
        raise ValueError('Servono almeno 5 righe con target osservati.')
    stratify = None
    if len(labels)==1 and types[labels[0]]=='classification':
        counts = frame[labels[0]].value_counts()
        if counts.min() >= 2 and len(counts) <= math.ceil(len(frame)*.2) and len(counts) <= math.floor(len(frame)*.8):
            stratify = frame[labels[0]]
    train, validation = train_test_split(frame,test_size=.2,random_state=42,stratify=stratify)
    info = dict(raw_rows=raw_rows, usable_rows=len(frame), excluded_missing_targets=raw_rows-len(frame),
                excluded_columns=list(filters or []), target_types=types, stratified=stratify is not None,
                train_rows=len(train), validation_rows=len(validation), seed=42)
    return train.reset_index(drop=True), validation.reset_index(drop=True), info


def encoding_size(frame):
    from acan import GraphClusterModel
    prepared = GraphClusterModel.full_cols(frame)
    sizes = {str(name):int(prepared[name].nunique(dropna=True)) if not is_numeric_dtype(prepared[name]) else 1
             for name in prepared}
    columns = sum(sizes.values())
    return dict(rows=len(frame), columns=columns, cells=len(frame)*columns,
                largest=sorted(sizes.items(),key=lambda item:item[1],reverse=True)[:6])


def check_budget(frame):
    size = encoding_size(frame)
    if size['rows']>10000 or size['columns']>1000 or size['cells']>2_000_000:
        drivers = ', '.join(f'{name}: {count}' for name,count in size['largest'] if count>1)
        raise ValueError(f"Dopo la codifica: {size['rows']:,} righe, {size['columns']:,} attributi, {size['cells']:,} celle. "
                         f"Espansioni maggiori: {drivers or 'nessuna'}. Usa «Ripristina filtri suggeriti» in Model → Impostazioni avanzate, oppure escludi identificativi/testo ad alta cardinalità nei filtri "
                         "o rimuovi colonne numeriche selezionate per errore come categoriche. "
                         "Limiti: 10.000 righe, 1.000 attributi codificati, 2 milioni di celle.")
    return size
