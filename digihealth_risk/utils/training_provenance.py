"""Capture fit-time settings; never infer them for older result files."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd


def training_metadata(model_family: str, train_rows: int, settings: dict) -> dict:
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn']
    if model_family in {'xgboost', 'catboost', 'lightgbm'}:
        packages.append(model_family)
    return {
        'schema_version': 1,
        'model_family': model_family,
        'train_rows': int(train_rows),
        'settings': settings,
        'python': platform.python_version(),
        'packages': {name: version(name) for name in packages},
    }


def collect_training_metadata(root: Path) -> dict:
    """Read fit records from research metrics without relabeling legacy fits.

    Metric sources are hashed independently of the reporting environment.
    A file can have some recorded fits and still appear in unrecorded_sources.
    """
    sources = sorted((root / 'digihealth_risk/phase_1/outputs').glob(
        'phase_1_v2_logistic*_metrics.csv'))
    tree_metrics = root / 'digihealth_risk/phase_4/outputs/phase_4_v2_metrics.csv'
    if tree_metrics.exists():
        sources.append(tree_metrics)
    recorded, unrecorded = [], []
    for path in sources:
        relative = str(path.relative_to(root))
        frame = pd.read_csv(path)
        if 'training_metadata' not in frame or frame.empty:
            unrecorded.append(relative)
            continue
        values = frame.training_metadata.replace(r'^\s*$', pd.NA, regex=True)
        if values.isna().any():
            unrecorded.append(relative)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for value in values.dropna().unique():
            recorded.append({'source': relative, 'source_sha256': digest,
                             'fit': json.loads(value)})
    return {
        'scope': 'Statistical Logistic and phase-4 screening-tree metric files',
        'recorded': recorded,
        'unrecorded_sources': unrecorded,
    }
