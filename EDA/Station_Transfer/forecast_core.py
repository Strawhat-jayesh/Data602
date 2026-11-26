"""Shared temporal features and estimators, preserved from the first AEB experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

HORIZONS = (1, 3, 6, 12, 24)
ENV_COLS = ('sal', 'ph', 'turb', 'chl', 'temp')
NAMES = {'DO_mg/L': 'do', 'Salinity_ppt': 'sal', 'pH': 'ph',
         'Turb_NTU': 'turb', 'Chl_ug/L': 'chl', 'Temp_C': 'temp'}
STEP = pd.Timedelta(minutes=15)
CONFIGS = {'ridge': [{'alpha': 1.0}, {'alpha': 100.0}],
           'hist_gbm': [{'max_leaf_nodes': 7}, {'max_leaf_nodes': 15}],
           'linear': [{}],
           'polynomial': [{'alpha': 1.0}, {'alpha': 100.0}],
           'decision_tree': [{'max_depth': 5}, {'max_depth': 10}],
           'random_forest': [{'max_depth': 8}, {'max_depth': None}],
           'mlp': [{'hidden_layer_sizes': (32, 16)}, {'hidden_layer_sizes': (64, 32)}]}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_station(path):
    raw = pd.read_csv(path)
    required = {'DateTime', *NAMES}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing required columns: {sorted(required-set(raw.columns))}')
    time = pd.to_datetime(raw.DateTime, format='%m/%d/%Y %H:%M', errors='coerce')
    if time.isna().any() or time.duplicated().any():
        raise ValueError('Timestamps must be parseable and unique before forecasting.')
    if 'Station' in raw and raw.Station.nunique(dropna=False) != 1:
        raise ValueError('Use a single station per input file.')
    if 'Layer' in raw and raw.Layer.nunique(dropna=False) != 1:
        raise ValueError('Use a single sampling layer per input file.')
    data = raw[list(NAMES)].apply(pd.to_numeric, errors='coerce').rename(columns=NAMES)
    if np.isinf(data.to_numpy()).any():
        raise ValueError('Infinite numeric values require explicit quality review.')
    data.index = pd.DatetimeIndex(time, name='origin_time')
    data = data.sort_index()
    grid = pd.date_range(data.index.min(), data.index.max(), freq=STEP, name='origin_time')
    if not data.index.isin(grid).all():
        raise ValueError('Off-grid timestamps require explicit alignment.')
    before = len(data)
    data = data.reindex(grid)  # keep elapsed-time gaps; never compress the time axis
    negative = data.do[data.do < 0]
    audit = {
        'source_sha256': sha(path), 'source_rows': before, 'grid_rows': len(data),
        'inserted_grid_rows': len(data)-before,
        'station': str(raw.Station.iloc[0]) if 'Station' in raw else 'not supplied',
        'layer': str(raw.Layer.iloc[0]) if 'Layer' in raw else 'not supplied',
        'start': str(grid.min()), 'end': str(grid.max()),
        'missing_counts': {c: int(data[c].isna().sum()) for c in data},
        'negative_counts': {c: int((data[c]<0).sum()) for c in data},
        'negative_do_min': float(negative.min()) if len(negative) else None,
        'negative_do_max': float(negative.max()) if len(negative) else None,
        'quality_note': 'No quality flags supplied. Negative DO treatment is an assumption, not a validated DNR rule. Other numeric values are retained and audited.'}
    return data, audit


def apply_do_policy(data, policy):
    d = data.copy()
    if policy == 'clip_zero':
        d['do'] = d.do.clip(lower=0)
    elif policy == 'exclude_negative':
        d.loc[d.do < 0, 'do'] = np.nan
    else:
        raise ValueError(policy)
    return d


def make_features(data, input_set, redundant=False):
    # Timestamp shifts give exactly the intended elapsed lag, including across gaps.
    x = pd.DataFrame(index=data.index)
    x['do'] = data.do
    for lag in (1, 2, 4, 8, 16, 32):
        x[f'do_lag{lag}'] = data.do.shift(freq=lag*STEP).reindex(data.index)
    x['do_roll16_mean'] = data.do.rolling(16, min_periods=8).mean()
    x['do_roll96_mean'] = data.do.rolling(96, min_periods=48).mean()
    if input_set == 'combined':
        for c in ENV_COLS:
            x[c] = data[c]
            for lag in (4, 16):
                x[f'{c}_lag{lag}'] = data[c].shift(freq=lag*STEP).reindex(data.index)
    elif input_set != 'do_history':
        raise ValueError(input_set)
    if redundant:
        x['do_diff1'] = data.do-x.do_lag1
        x['do_diff4'] = data.do-x.do_lag4
        if input_set == 'combined':
            x['temp_diff4'] = data.temp-x.temp_lag4
    assert not any('sat' in c.lower() for c in x)
    return x


def describe_features(data):
    feature_meanings = {'do': 'Current DO (mg/L)'}
    for lag in (1, 2, 4, 8, 16, 32):
        feature_meanings[f'do_lag{lag}'] = f'DO {lag * 15} minutes earlier (mg/L)'
    feature_meanings.update({
        'do_roll16_mean': 'Trailing 4-hour mean DO (mg/L)',
        'do_roll96_mean': 'Trailing 24-hour mean DO (mg/L)',
        'do_diff1': 'DO change over 15 minutes (mg/L)',
        'do_diff4': 'DO change over 1 hour (mg/L)',
        'temp_diff4': 'Temperature change over 1 hour (°C)'})
    environment = {'sal': 'Salinity (ppt)', 'ph': 'pH', 'turb': 'Turbidity (exported units)',
                   'chl': 'Chlorophyll (µg/L)', 'temp': 'Temperature (°C)'}
    for variable, meaning in environment.items():
        feature_meanings[variable] = 'Current ' + meaning
        for lag in (4, 16):
            feature_meanings[f'{variable}_lag{lag}'] = f'{meaning}, {lag // 4} hours earlier'
    missing = set(data.columns) - set(feature_meanings)
    if missing:
        raise ValueError(f'No descriptions for features: {sorted(missing)}')
    return pd.DataFrame({'Predictor': data.columns,
                         'Meaning': [feature_meanings[c] for c in data.columns]})


def target_at(data, hours):
    targets = data.index + pd.Timedelta(hours=hours)
    return pd.Series(data.do.reindex(targets).to_numpy(), index=data.index), targets


def eligible(data, hours):
    y, target_times = target_at(data, hours)
    mask = data.do.notna() & y.notna()
    # Both feature configurations use identical support and a full 24-hour elapsed warmup.
    mask &= data.index >= data.index.min()+pd.Timedelta(hours=24)
    return y, target_times, mask.to_numpy()


def split_masks(data, hours, start, end=None):
    y, times, valid = eligible(data, hours)
    train = valid & (times < start)
    val = valid & (data.index >= start)
    if end is not None:
        val &= times < end
    if not train.any() or not val.any():
        raise ValueError(f'Insufficient eligible observations for {hours}h split {start}')
    assert times[train].max() < data.index[val].min()
    return y, times, train, val


def build_model(name, params):
    if name == 'linear':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LinearRegression())
    if name == 'ridge':
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(**params))
    if name == 'polynomial':
        # Quadratic terms and interactions with L2 regularization; all scales learned on source training.
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(**params))
    if name == 'decision_tree':
        return make_pipeline(SimpleImputer(strategy='median'),
            DecisionTreeRegressor(min_samples_leaf=20, random_state=0, **params))
    if name == 'random_forest':
        return make_pipeline(SimpleImputer(strategy='median'),
            RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_features=.8,
                n_jobs=2, random_state=0, **params))
    if name == 'hist_gbm':
        # Disable automatic random validation; chronology is controlled by explicit CV.
        return make_pipeline(SimpleImputer(strategy='median'),
            HistGradientBoostingRegressor(max_iter=150, learning_rate=.05,
                early_stopping=False, l2_regularization=1., random_state=0, **params))
    if name == 'mlp':
        # Standardize the delta target using source training only. No random internal holdout.
        return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
            TransformedTargetRegressor(transformer=StandardScaler(),
                regressor=MLPRegressor(activation='relu',solver='adam',alpha=.01,
                    batch_size=256,learning_rate_init=.001,max_iter=400,
                    early_stopping=False,n_iter_no_change=25,tol=1e-4,
                    shuffle=True,random_state=0,**params)))
    raise ValueError(name)


def fit_predict(data, hours, features, name, params, train, test):
    y, _ = target_at(data, hours)
    model = build_model(name, params)
    model.fit(features.loc[train], (y-data.do).loc[train])
    predictions = data.do.loc[test].to_numpy()+model.predict(features.loc[test])
    assert np.isfinite(predictions).all()
    return predictions, model


def scores(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    if len(actual) == 0:
        return {'n': 0, 'mae': None, 'rmse': None, 'r2': None}
    r2 = None if len(actual) < 2 or np.all(actual == actual[0]) else float(r2_score(actual, pred))
    return {'n': len(actual), 'mae': float(np.abs(actual-pred).mean()),
            'rmse': float(np.sqrt(np.square(actual-pred).mean())), 'r2': r2}


def detection(actual, pred, threshold=2):
    tn, fp, fn, tp = confusion_matrix(np.asarray(actual)<threshold, np.asarray(pred)<threshold, labels=[False, True]).ravel()
    ratio = lambda a,b: float(a/b) if b else None
    return {'true_negative': int(tn), 'false_positive': int(fp), 'false_negative': int(fn),
            'true_positive': int(tp), 'precision': ratio(tp,tp+fp), 'recall': ratio(tp,tp+fn),
            'f1': ratio(2*tp,2*tp+fp+fn)}


def bootstrap_gain(predictions, days, reps=1000):
    # Resample paired contiguous calendar blocks; retain missing-case support inside each block.
    loss = np.abs(predictions.actual-predictions.do_history)-np.abs(predictions.actual-predictions.combined)
    block = (predictions.index-predictions.index.min())//pd.Timedelta(days=days)
    ag = pd.DataFrame({'loss': loss.to_numpy(), 'block': block}).groupby('block').loss.agg(['sum','count'])
    rng = np.random.default_rng(41)
    draws = rng.integers(0,len(ag),(reps,len(ag)))
    values = ag['sum'].to_numpy()[draws].sum(1)/ag['count'].to_numpy()[draws].sum(1)
    return {'block_days': days, 'n_nonempty_blocks': len(ag), 'bootstrap_reps': reps,
            'gain_ci_low': float(np.quantile(values,.025)), 'gain_ci_high': float(np.quantile(values,.975))}


