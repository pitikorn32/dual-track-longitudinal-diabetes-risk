"""Shared publication metrics and prediction identity checks.

PR-AUC means average precision. Undefined discrimination metrics return NaN;
invalid probabilities, labels, and duplicate prediction occasions fail loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

KEY_COLUMNS = ['PatientId', 'Year', 'target_year']
TARGET = 'Target_AtRisk_Status'


def validate_scores(y_true, probability):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Expected nonempty equally sized label and probability vectors')
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected observed binary target labels')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite probabilities in [0, 1]')
    return y, p


def auc_pr(y_true, probability) -> float:
    y, p = validate_scores(y_true, probability)
    return float(average_precision_score(y, p)) if y.sum() else float('nan')


def auc_roc(y_true, probability) -> float:
    y, p = validate_scores(y_true, probability)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def ranking_metrics(y_true, probability) -> dict[str, float]:
    y, p = validate_scores(y_true, probability)
    return {
        'rows': float(len(y)), 'positives': float(y.sum()),
        'positive_rate': float(y.mean()), 'roc_auc': auc_roc(y, p),
        'pr_auc': auc_pr(y, p), 'brier': float(brier_score_loss(y, p)),
    }


def validate_predictions(frame: pd.DataFrame, *, instance_columns=()) -> None:
    keys = [*instance_columns, *KEY_COLUMNS]
    required = [*keys, TARGET, 'predicted_probability']
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    if frame[keys].isna().any().any():
        raise ValueError('Missing prediction identity')
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate prediction occasions within a model configuration')
    validate_scores(frame[TARGET], frame.predicted_probability)
    if frame.groupby(KEY_COLUMNS, sort=False)[TARGET].nunique().gt(1).any():
        raise ValueError('Inconsistent target labels across model outputs')


def align_to_reference(predictions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Require coverage of every reference occasion and agree on its target."""
    validate_predictions(predictions)
    if reference.duplicated(KEY_COLUMNS).any():
        raise ValueError('Duplicate reference occasions')
    joined = reference[KEY_COLUMNS + [TARGET]].merge(
        predictions, on=KEY_COLUMNS, how='left', validate='one_to_one',
        suffixes=('_reference', ''), indicator=True,
    )
    if not joined['_merge'].eq('both').all():
        raise ValueError('Predictions do not cover the shared evaluation cohort')
    if not joined[TARGET].eq(joined[TARGET + '_reference']).all():
        raise ValueError('Inconsistent target labels against shared cohort')
    return joined.drop(columns=['_merge', TARGET + '_reference'])
