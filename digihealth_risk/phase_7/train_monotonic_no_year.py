"""Phase 7 — Year-features ablation: retrain phase 5 monotonic models.

Mirrors all five phase_5 monotonic trainers but with Year, Year_centered,
and Year_centered_sq excluded from the training set via
`patch_drop_year_features()`.

Families: xgboost, catboost, lightgbm, ebm, logistic
Horizons: N ∈ {1, 2, 3, 4, 5}, fixed M = 5 (matches phase_5)
Variant : monotonic only (the deployed intervention-safe track)

For each (family, horizon) we collect:
    - test/train classification metrics
    - test-set predictions
    - the constrained feature table (for sanity)

Outputs to digihealth_risk/phase_7/outputs/:
    phase_7_no_year_monotonic_metrics.csv
    phase_7_no_year_monotonic_test_predictions.csv
    phase_7_no_year_monotonic_constraints.csv

joblib model artifacts are written under
phase_7/outputs/no_year_monotonic_models/<family>/ so they do not collide
with the deployed phase_5 artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_7.year_ablation_utils import (  # noqa: E402
    dropped_feature_names,
    patch_drop_year_features,
)

patch_drop_year_features()

# Import each family's trainer AFTER the patch.
from digihealth_risk.phase_5 import (  # noqa: E402
    train_monotonic_catboost as cat_mod,
    train_monotonic_ebm as ebm_mod,
    train_monotonic_lightgbm as lgb_mod,
    train_monotonic_logistic as lr_mod,
    train_monotonic_xgboost as xgb_mod,
)

OUT_DIR = ROOT / "digihealth_risk" / "phase_7" / "outputs"
MODEL_ROOT = OUT_DIR / "no_year_monotonic_models"

HORIZONS = [1, 2, 3, 4, 5]
HISTORY_YEARS = 5
VARIANT = "monotonic"


def _redirect_model_dir(module, family: str) -> None:
    """Send a family's joblib artifacts to phase_7/outputs/no_year_monotonic_models/<family>/."""
    target = MODEL_ROOT / family
    target.mkdir(parents=True, exist_ok=True)
    module.MODEL_DIR = target


def _run_family_xgb() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """xgboost trainer exposes run_horizon(horizon) and always trains the monotonic variant."""
    _redirect_model_dir(xgb_mod, "xgboost")
    metrics_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    constraint_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        print(f"[phase_7] xgboost monotonic N={horizon} M={HISTORY_YEARS}")
        metrics, predictions, constraints, _sanity = xgb_mod.run_horizon(horizon)
        metrics.insert(0, "family", "xgboost")
        predictions.insert(0, "family", "xgboost")
        constraints.insert(0, "family", "xgboost")
        metrics_parts.append(metrics)
        prediction_parts.append(predictions)
        constraint_parts.append(constraints)
    return (
        pd.concat(metrics_parts, ignore_index=True),
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(constraint_parts, ignore_index=True),
    )


def _run_family_run_combo(module, family: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """catboost / lightgbm / ebm / logistic share the run_combo(horizon, history, variant) signature."""
    _redirect_model_dir(module, family)
    metrics_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    constraint_parts: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        print(f"[phase_7] {family} {VARIANT} N={horizon} M={HISTORY_YEARS}")
        metrics, predictions, constraints, _scenarios = module.run_combo(
            horizon, HISTORY_YEARS, VARIANT
        )
        metrics.insert(0, "family", family)
        predictions.insert(0, "family", family)
        constraints.insert(0, "family", family)
        metrics_parts.append(metrics)
        prediction_parts.append(predictions)
        constraint_parts.append(constraints)
    return (
        pd.concat(metrics_parts, ignore_index=True),
        pd.concat(prediction_parts, ignore_index=True),
        pd.concat(constraint_parts, ignore_index=True),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[phase_7] dropping features: {dropped_feature_names()}")

    families = [
        ("xgboost", _run_family_xgb, None),
        ("catboost", _run_family_run_combo, cat_mod),
        ("lightgbm", _run_family_run_combo, lgb_mod),
        ("ebm", _run_family_run_combo, ebm_mod),
        ("logistic", _run_family_run_combo, lr_mod),
    ]

    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    constraint_parts: list[pd.DataFrame] = []

    for family, runner, module in families:
        if module is None:
            metrics, predictions, constraints = runner()
        else:
            metrics, predictions, constraints = runner(module, family)
        metric_parts.append(metrics)
        prediction_parts.append(predictions)
        constraint_parts.append(constraints)

    metrics_df = pd.concat(metric_parts, ignore_index=True)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    constraints_df = pd.concat(constraint_parts, ignore_index=True)

    for df in (metrics_df, predictions_df, constraints_df):
        df.insert(0, "year_features", "dropped")

    metrics_df.to_csv(OUT_DIR / "phase_7_no_year_monotonic_metrics.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "phase_7_no_year_monotonic_test_predictions.csv", index=False)
    constraints_df.to_csv(OUT_DIR / "phase_7_no_year_monotonic_constraints.csv", index=False)

    test_rows = metrics_df[metrics_df["split"] == "test"].sort_values(
        ["family", "horizon_years"]
    )
    print("\n[phase_7] Monotonic test PR-AUC by family/horizon (no_year):")
    print(
        test_rows[["family", "horizon_years", "pr_auc", "roc_auc", "brier"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
