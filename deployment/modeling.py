"""Shared modeling helpers for the standalone deployment slice (vendored).

Dependency-free copies of the helpers `export_models.py` needs from the
digihealth_risk phase tree, so this folder runs without importing it:

  * feature engineering and preprocessing  (phase_2/train_tree_models.py)
  * monotonic-constraint rules             (phase_5/train_monotonic_xgboost.py)
  * the no-Year ablation toggle            (phase_7/year_ablation_utils.py)

Keep this file in sync with those upstream modules if the modeling logic changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_SEED = 20260501

LEAKAGE_OR_METADATA_COLUMNS = {
    "PatientId",
    "date_of_birth",
    "DM_status_up_to_year",
    "AtRisk_current_year",
    "first_atrisk_year",
    "target_year",
    "Target_AtRisk_Status",
    "Next_Year_AtRisk_Status",
    "prediction_horizon_years",
    "history_window_years",
    "history_start_year",
    "pulse_pressure",          # v2: removed — VIF=225, = BL_pres1 - BL_pres2 by construction
}


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to read pickles created by NumPy 2.x."""
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add v2 engineered features. Safe to call repeatedly — skips existing columns."""
    df = df.copy()
    if "Year_centered" not in df.columns:
        df["Year_centered"] = df["Year"] - df["Year"].min()
    if "Year_centered_sq" not in df.columns:
        df["Year_centered_sq"] = df["Year_centered"] ** 2
    if "FBS_hinge_100" not in df.columns:
        df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)   # NaN propagates from FBS
    if "FBS_hinge_125" not in df.columns:
        df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    if "FBS_x_Age" not in df.columns:
        df["FBS_x_Age"] = df["FBS"] * df["Age"]
    if "MAX_FBS_x_Age" not in df.columns:
        df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def load_table(path: Path) -> pd.DataFrame:
    install_numpy_pickle_compat()
    df = pd.read_pickle(path).copy()
    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]
    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [
        col
        for col in df.columns
        if col not in LEAKAGE_OR_METADATA_COLUMNS and not df[col].isna().all()
    ]
    numeric_features = [
        col for col in candidates
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    categorical_features = [
        col for col in candidates
        if col not in numeric_features and not pd.api.types.is_datetime64_any_dtype(df[col])
    ]
    return numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "calibration_method": "raw",
        "threshold_strategy": "train_positive_rate",
        "rows": float(len(y_true)),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(((~prediction) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Monotonic-constraint rules (phase_5/train_monotonic_xgboost.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonotoneRule:
    sign: int
    reason: str


BASE_MONOTONE_RULES: dict[str, MonotoneRule] = {
    "total_sugary_week": MonotoneRule(1, "Higher sugary drink intake should not lower risk."),
    "total_veg_fruit_week": MonotoneRule(-1, "Higher vegetable/fruit intake should not raise risk."),
    "total_exercise_week": MonotoneRule(-1, "Higher exercise should not raise risk."),
    "total_phy_activity_week": MonotoneRule(-1, "Higher physical activity should not raise risk."),
    "FBS": MonotoneRule(1, "Higher current fasting blood sugar should not lower risk."),
    "MAX_FBS_up_to_year": MonotoneRule(1, "Higher cumulative maximum fasting blood sugar should not lower risk."),
    "BMI": MonotoneRule(1, "Higher BMI should not lower risk."),
    "Waist": MonotoneRule(1, "Higher waist circumference should not lower risk."),
    # v2: FBS-derived features — same increasing-risk direction as FBS
    "FBS_hinge_100": MonotoneRule(1, "FBS excess above 100 mg/dL (pre-DM threshold) should not lower risk."),
    "FBS_hinge_125": MonotoneRule(1, "FBS excess above 125 mg/dL (DM threshold) should not lower risk."),
    "FBS_x_Age": MonotoneRule(1, "FBS × Age interaction — higher product should not lower risk."),
    "MAX_FBS_x_Age": MonotoneRule(1, "MAX_FBS_up_to_year × Age interaction — higher product should not lower risk."),
}

HISTORY_MONOTONE_BASES = {
    "FBS": 1,
    "BMI": 1,
    "Waist": 1,
}

HISTORY_MONOTONE_SUFFIXES = (
    "_latest",
    "_mean",
    "_min",
    "_max",
    "_range",
    "_slope",
)


# ---------------------------------------------------------------------------
# No-Year ablation toggle (phase_7/year_ablation_utils.py)
# ---------------------------------------------------------------------------

YEAR_FEATURES: tuple[str, ...] = ("Year_centered", "Year_centered_sq")
RAW_YEAR_FEATURE: str = "Year"

_NO_YEAR_PATCHED = False


def patch_drop_year_features() -> None:
    """Exclude Year, Year_centered and Year_centered_sq from training.

    Mirrors year_ablation_utils.patch_drop_year_features: the calendar-time
    features are added to LEAKAGE_OR_METADATA_COLUMNS (so get_feature_columns
    drops them) and engineer_features is wrapped to strip the derived columns it
    produced. Idempotent. Call via the module (modeling.engineer_features,
    modeling.get_feature_columns) so the rebinding is observed.
    """
    global _NO_YEAR_PATCHED, LEAKAGE_OR_METADATA_COLUMNS, engineer_features
    if _NO_YEAR_PATCHED:
        return
    LEAKAGE_OR_METADATA_COLUMNS = set(LEAKAGE_OR_METADATA_COLUMNS) | {
        RAW_YEAR_FEATURE,
        *YEAR_FEATURES,
    }
    original_engineer = engineer_features

    def engineer_features_no_year(df: pd.DataFrame) -> pd.DataFrame:
        out = original_engineer(df)
        drop_cols = [c for c in YEAR_FEATURES if c in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)
        return out

    engineer_features = engineer_features_no_year
    _NO_YEAR_PATCHED = True


def dropped_feature_names() -> tuple[str, ...]:
    return (RAW_YEAR_FEATURE, *YEAR_FEATURES)
