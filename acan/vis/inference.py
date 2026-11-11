"""Inference helpers shared by the UI and its tests; never train a model."""

import numpy as np
import pandas as pd

from acan import GraphClusterModel


def infer(model_json, dataset, mode='predict'):
    if not model_json:
        raise ValueError('Caricare o addestrare un modello nella scheda Model.')
    if dataset.empty:
        raise ValueError('Il CSV non contiene righe.')
    model = GraphClusterModel.loadModel(model_json)
    root = model.clusterModel[model.rootId]
    data = dataset.copy()
    if mode == 'predict':
        if not model.labels:
            raise ValueError('Modello senza Labels: usare Completa campi vuoti.')
        # Always hide targets, even if the user uploaded a labelled dataset.
        for label in model.labels:
            data[label] = np.nan
    elif mode != 'complete':
        raise ValueError('Modalità non valida.')
    elif not data.isna().any().any():
        raise ValueError('Inserire celle vuote nelle colonne da completare.')

    for col in data.columns[data.isna().any()]:
        if not root.fullLabels([col]):
            raise ValueError(f'Colonna assente dal modello: {col}')
    if mode == 'predict' and model.preprocessing:
        # Prediction has train-fitted imputation and missing-category encoding.
        # An all-missing observation still has a defined prediction; reject an
        # unrelated schema rather than rejecting missing feature values.
        fields = {field['name'] for field in model.preprocessing['fields']}
        if not fields.intersection(dataset.columns):
            raise ValueError('Ogni riga incompleta richiede almeno un attributo noto al modello.')
    else:
        for _, row in data.iterrows():
            if not row.isna().any():
                continue
            known = pd.DataFrame([row.dropna().to_dict()])
            if known.empty:
                raise ValueError('Ogni riga incompleta richiede almeno un attributo noto al modello.')
            encoded = GraphClusterModel.full_dummies(known)
            if not any(col in root.cols for col in encoded.columns):
                raise ValueError('Ogni riga incompleta richiede almeno un attributo noto al modello.')
    if mode == 'predict':
        features = data.drop(columns=model.labels, errors='ignore')
        if model.preprocessing:
            from acan.preprocessing import transform_features
            features = transform_features(features, model.preprocessing)
        predicted, confidence, paths = model.predictDataset(features, model.labels)
        result = dataset.copy()
        for label in model.labels:
            columns = root.fullLabels([label])
            if columns == [label]:
                lo, hi = model.minMax[label]
                result[label] = predicted[label].to_numpy() * (hi - lo) + lo
            else:
                result[label] = predicted[columns].idxmax(axis=1).str[len(str(label)) + 1:].to_numpy()
        result['ACAN_path'] = paths
        result['ACAN_confidence'] = confidence.mean(axis=1).to_numpy()
        return result
    return model.predict_masked(data)
