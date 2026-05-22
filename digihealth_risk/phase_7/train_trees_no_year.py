"""Phase 7 — Year-features ablation: retrain phase 2 tree grid without Year.

Mirrors phase_2/train_tree_models.py but applies `patch_drop_year_features()`
so the three calendar-time features (Year, Year_centered, Year_centered_sq)
are excluded from the training set.

Trains all 5 tree families across the full N×M grid:
    N (horizon) ∈ {1, 2, 3, 4, 5}
    M (history) ∈ {1, 3, 5}
    model      ∈ {histgb, random_forest, xgboost, lightgbm, catboost}

Outputs to digihealth_risk/phase_7/outputs/:
    phase_7_no_year_trees_metrics.csv
    phase_7_no_year_trees_test_predictions.csv
    phase_7_no_year_trees_feature_importance.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Patch BEFORE importing phase_2 helpers so engineer_features and
# get_feature_columns both see the updated LEAKAGE_OR_METADATA set.
from digihealth_risk.phase_7.year_ablation_utils import (  # noqa: E402
    dropped_feature_names,
    patch_drop_year_features,
)

patch_drop_year_features()

from digihealth_risk.phase_2.train_tree_models import (  # noqa: E402
    run_dataset,
)

PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_7" / "outputs"

MODELS = ["histgb", "random_forest", "xgboost", "lightgbm", "catboost"]
HORIZONS = [1, 2, 3, 4, 5]
HISTORIES = [1, 3, 5]


def phase0_path(horizon: int, history: int) -> Path:
    if horizon == 1 and history == 1:
        return PHASE0_OUT / "phase_0_modeling_table.pkl"
    return PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{history}.pkl"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[phase_7] dropping features: {dropped_feature_names()}")

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        for history in HISTORIES:
            path = phase0_path(horizon, history)
            if not path.exists():
                print(f"[phase_7] skip missing table: {path.relative_to(ROOT)}")
                continue
            print(f"[phase_7] training N={horizon} M={history} from {path.name}")
            metrics_df, predictions, importances = run_dataset(
                path, MODELS, use_class_weights=False
            )
            all_metrics.append(metrics_df)
            all_predictions.extend(predictions)
            all_importances.extend(importances)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importances = pd.concat(all_importances, ignore_index=True)

    metrics["variant"] = "no_year"
    predictions["variant"] = "no_year"
    importances["variant"] = "no_year"

    metrics.to_csv(OUT_DIR / "phase_7_no_year_trees_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / "phase_7_no_year_trees_test_predictions.csv", index=False)
    importances.to_csv(OUT_DIR / "phase_7_no_year_trees_feature_importance.csv", index=False)

    test_rows = metrics[metrics["split"] == "test"].sort_values(
        ["dataset", "pr_auc"], ascending=[True, False]
    )
    print("\n[phase_7] Top test PR-AUC by dataset:")
    print(test_rows[["dataset", "model", "pr_auc", "roc_auc", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
