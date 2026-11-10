"""Measured tabular validation; never fit baselines or ACAN on held-out rows."""
import time
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from acan import GraphClusterModel
from .inference import infer


def encode(train, test):
    """Fit imputation, one-hot schema and scaling exclusively on training rows."""
    train, test = train.copy(), test.copy()
    for col in train:
        if pd.api.types.is_numeric_dtype(train[col]):
            fill = train[col].median()
            fill = 0 if pd.isna(fill) else fill
        else:
            mode = train[col].dropna().mode()
            fill = mode.iloc[0] if len(mode) else '__missing__'
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)
    x = pd.get_dummies(train, dtype=float).astype(float)
    y = pd.get_dummies(test, dtype=float).reindex(columns=x.columns, fill_value=0).astype(float)
    lo, span = x.min(), (x.max() - x.min()).replace(0, 1)
    return ((x - lo) / span).to_numpy(), ((y - lo) / span).to_numpy()


def evaluate(model_json, test, train=None, before_json=None, baselines=True):
    graph = GraphClusterModel.loadModel(model_json)
    if test.empty:
        raise ValueError('Il set di valutazione è vuoto.')
    if not graph.labels:
        return masked_evaluate(model_json, test)
    absent = set(graph.labels) - set(test.columns)
    if absent:
        raise ValueError(f'Target assenti dal CSV: {sorted(absent)}')
    prepared = test.copy()
    if train is not None and not graph.preprocessing:
        # Legacy native inference otherwise imputes a batch's own missing features.
        # Fit this UI preprocessing only on training observations instead.
        for col in train.columns:
            if col in graph.labels or col not in prepared:
                continue
            known = train[col].dropna()
            if not len(known):
                continue
            fill = known.mean() if pd.api.types.is_numeric_dtype(train[col]) else known.mode().iloc[0]
            prepared[col] = prepared[col].fillna(fill)
    started = time.perf_counter()
    predicted = infer(model_json, prepared)
    prediction_seconds = time.perf_counter() - started
    variants = {'ACAN attuale': predicted}
    if before_json:
        variants['ACAN prima del cambiamento'] = infer(before_json, prepared)
    root = graph.clusterModel[graph.rootId]
    records, details, timings, diagnostics = [], [], [], []
    for target in graph.labels:
        valid = test[target].notna()
        if not valid.any():
            continue
        truth = test.loc[valid, target]
        numeric = root.fullLabels([target]) == [target]
        if not numeric:
            distinct = predicted.loc[valid, target].astype(str).nunique()
            observed = truth.astype(str).nunique()
            if distinct == 1 and observed > 1:
                diagnostics.append(dict(target=target, predicted_classes=int(distinct), observed_classes=int(observed),
                    warning=f'Target {target}: ACAN predice una sola classe su {observed} presenti nella validazione. '
                            'L’accuracy può essere fuorviante: controlla balanced accuracy e F1 macro.'))
        candidates = {name: values.loc[valid, target] for name, values in variants.items()}
        if baselines and train is not None and target in train:
            fitted = train.loc[train[target].notna()]
            features = [col for col in train.columns if col not in graph.labels]
            if len(fitted) and features and set(features).issubset(test.columns):
                x, xt = encode(fitted[features], test.loc[valid, features])
                y = fitted[target].to_numpy() if numeric else fitted[target].astype(str).to_numpy()
                candidates['Media training' if numeric else 'Maggioranza training'] = np.repeat(
                    np.mean(y) if numeric else pd.Series(y).mode().iloc[0], len(truth))
                from xgboost import XGBClassifier, XGBRegressor
                from catboost import CatBoostClassifier, CatBoostRegressor
                encoding = None if numeric else LabelEncoder().fit(y)
                fitted_y = y if numeric else encoding.transform(y)
                estimators = [
                    ('KNN', KNeighborsRegressor(n_neighbors=min(5, len(y))) if numeric else KNeighborsClassifier(n_neighbors=min(5, len(y)))),
                    ('CART', DecisionTreeRegressor(max_depth=5, random_state=42) if numeric else DecisionTreeClassifier(max_depth=5, random_state=42)),
                    ('Random Forest', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1) if numeric else RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)),
                    ('Extra Trees', ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1) if numeric else ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=1)),
                    ('HistGradientBoosting', HistGradientBoostingRegressor(max_iter=100, random_state=42) if numeric else HistGradientBoostingClassifier(max_iter=100, random_state=42)),
                    ('XGBoost', XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1, tree_method='hist') if numeric else XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=1, tree_method='hist')),
                    ('CatBoost', CatBoostRegressor(iterations=100, depth=6, random_seed=42, thread_count=1, verbose=False, allow_writing_files=False) if numeric else CatBoostClassifier(iterations=100, depth=6, random_seed=42, thread_count=1, verbose=False, allow_writing_files=False)),
                ]
                for name, estimator in estimators:
                    try:
                        start = time.perf_counter()
                        estimator.fit(x, fitted_y)
                        trained = time.perf_counter() - start
                        start = time.perf_counter()
                        result = np.asarray(estimator.predict(xt)).reshape(-1)
                        elapsed = time.perf_counter() - start
                        candidates[name] = result if numeric else encoding.inverse_transform(result.astype(int))
                        timings.append(dict(Modello=name, Target=target, Training_secondi=trained,
                                            Inferenza_secondi=elapsed, Stato='ok'))
                    except Exception as exc:
                        timings.append(dict(Modello=name, Target=target, Stato='errore', Errore=str(exc)))
        for name, values in candidates.items():
            if numeric:
                scores = dict(MAE=metrics.mean_absolute_error(truth, values),
                              RMSE=metrics.mean_squared_error(truth, values) ** .5)
                if len(truth) > 1:
                    scores['R²'] = metrics.r2_score(truth, values)
            else:
                actual, values = truth.astype(str), np.asarray(values).astype(str)
                scores = dict(Accuracy=metrics.accuracy_score(actual, values),
                              Balanced_accuracy=metrics.balanced_accuracy_score(actual, values),
                              F1_macro=metrics.f1_score(actual, values, average='macro', zero_division=0))
            for metric, value in scores.items():
                records.append(dict(Modello=name, Target=target, Metrica=metric, Valore=float(value), N=len(truth)))
        if numeric:
            details.append(dict(target=target, kind='regression', truth=truth.tolist(),
                                prediction=predicted.loc[valid, target].tolist()))
        else:
            classes = sorted(set(truth.astype(str)) | set(predicted.loc[valid, target].astype(str)))
            matrix = metrics.confusion_matrix(truth.astype(str), predicted.loc[valid, target].astype(str), labels=classes)
            details.append(dict(target=target, kind='classification', labels=classes, matrix=matrix.tolist()))
    if not records:
        raise ValueError('Nessun target osservato utilizzabile per la valutazione.')
    comparisons = []
    for record in records:
        if record['Modello'] != 'ACAN attuale':
            continue
        previous = next((old for old in records if old['Modello'] == 'ACAN prima del cambiamento'
                         and old['Target'] == record['Target'] and old['Metrica'] == record['Metrica']), None)
        if previous:
            delta = record['Valore'] - previous['Valore']
            comparisons.append(dict(Target=record['Target'], Metrica=record['Metrica'],
                                    Prima=previous['Valore'], Dopo=record['Valore'], Delta=delta,
                                    Miglioramento=bool(delta < 0 if record['Metrica'] in ('MAE', 'RMSE') else delta > 0)))
    timings.append(dict(Modello='ACAN attuale', Target='tutti', Inferenza_secondi=prediction_seconds, Stato='ok'))
    protocol = dict(seed=42, train_rows=len(train) if train is not None else None, test_rows=len(test),
                    routing_profile=getattr(root,'routing_profile','legacy-v8'),routing_parameters=getattr(root,'routing_parameters',None),
                    tuned=False, preprocessing='training-only imputation, one-hot and min-max',
                    tree_estimators=200, boosting_iterations=100, knn_neighbors=5, cart_max_depth=5)
    if baselines and train is not None:
        protocol['versions'] = {name: version(name) for name in ('scikit-learn', 'xgboost-cpu', 'catboost')}
    return dict(scores=records, details=details, comparisons=comparisons, timings=timings, protocol=protocol, diagnostics=diagnostics,
                predictions=predicted.to_dict(), mode='supervised', rows=len(test))


def masked_evaluate(model_json, test):
    """Seeded masking on at most 100 held-out rows, with at least one known cell."""
    graph = GraphClusterModel.loadModel(model_json)
    root = graph.clusterModel[graph.rootId]
    cols = [col for col in test if root.fullLabels([col])]
    truth = test[cols].head(100).reset_index(drop=True)
    masked = truth.copy()
    positions = []
    rng = np.random.default_rng(42)
    for row in truth.index:
        known = list(truth.columns[truth.loc[row].notna()])
        if len(known) < 2:
            continue
        hidden = rng.choice(known, size=max(1, len(known) // 5), replace=False)
        for col in hidden:
            if not pd.api.types.is_numeric_dtype(masked[col]):
                masked[col] = masked[col].astype(object)
            masked.loc[row, col] = np.nan
            positions.append((row, col))
    if not positions:
        raise ValueError('Per il benchmark di completamento servono almeno due attributi noti per riga.')
    completed = infer(model_json, masked, 'complete')
    records = []
    for col in cols:
        rows = [row for row, column in positions if column == col]
        if not rows:
            continue
        actual, pred = truth.loc[rows, col], completed.loc[rows, col]
        numeric = root.fullLabels([col]) == [col]
        score = metrics.mean_absolute_error(actual, pred) if numeric else metrics.accuracy_score(actual.astype(str), pred.astype(str))
        records.append(dict(Modello='ACAN masked (radice)', Target=col,
                            Metrica='MAE' if numeric else 'Accuracy', Valore=float(score), N=len(rows)))
    return dict(scores=records, details=[], predictions=completed.to_dict(), mode='masked-root',
                rows=len(truth), masked_cells=len(positions))
