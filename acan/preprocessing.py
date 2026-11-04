"""Train-fitted feature encoding for saved tabular GUI models.

This fixes input alignment/imputation; it does not change ACAN's objective,
attention, prototypes or prediction rule. Legacy models keep their old schema.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype


def fit_features(data, labels):
    from acan.core.model import GraphClusterModel
    features = data.drop(columns=labels, errors='ignore')
    fields = []
    for name in features:
        column = features[name]
        if is_datetime64_any_dtype(column):
            fill = column.dropna().mean()
            fields.append(dict(name=name, kind='datetime', fill=fill.isoformat() if pd.notna(fill) else None,
                               missing=bool(column.isna().any())))
        elif is_numeric_dtype(column) or is_bool_dtype(column):
            fill = column.mean()
            fields.append(dict(name=name, kind='numeric', fill=float(fill) if pd.notna(fill) else 0.,
                               missing=bool(column.isna().any())))
        else:
            fields.append(dict(name=name, kind='categorical'))
    return dict(version=1, fields=fields, encoded_columns=list(GraphClusterModel.full_dummies(features).columns))


def transform_features(data, schema):
    prepared = pd.DataFrame(index=data.index)
    for field in schema['fields']:
        name = field['name']
        values = data[name].copy() if name in data else pd.Series(index=data.index, dtype=object)
        missing = values.isna()
        if field['kind'] == 'categorical':
            prepared[name] = values.astype(object).fillna('_NaN')
        elif field['kind'] == 'datetime':
            values = pd.to_datetime(values).fillna(pd.Timestamp(field['fill']))
            if field['missing']:
                prepared[name+'_NaN'] = missing.astype(float)
            for component in ('day','month','year'):
                prepared[name+'_'+component] = getattr(values.dt, component)
        else:
            if field['missing']:
                prepared[name+'_NaN'] = missing.astype(float)
            prepared[name] = pd.to_numeric(values, errors='raise').fillna(field['fill']).astype(float)
    # Absent one-hot levels are zeros, not unknown/omitted dimensions. Categories
    # absent from training do not create new model dimensions.
    return pd.get_dummies(prepared, dtype=float).reindex(columns=schema['encoded_columns'], fill_value=0.)
