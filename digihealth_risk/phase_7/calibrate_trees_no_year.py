"""Phase 7 — Year-features ablation: recalibrate phase 4 tree grid without Year.

Mirrors phase_4/calibrate_trees.py but applies `patch_drop_year_features()`
so Year, Year_centered, and Year_centered_sq are excluded from training and
from the per-config calibrators.

Trains + calibrates the same 30 (model × N × M) configurations as phase 4:
    model ∈ {catboost, xgboost}
    N     ∈ {1, 2, 3, 4, 5}
    M     ∈ {1, 3, 5}

For each config, three calibration methods are evaluated: raw / Platt /
isotonic, with thresholds chosen on the calibration split.

Outputs to digihealth_risk/phase_7/outputs/:
    phase_7_no_year_calibration_metrics.csv
    phase_7_no_year_calibration_threshold_table.csv
    phase_7_no_year_calibration_test_predictions.csv
    phase_7_no_year_calibration_curves.csv
    phase_7_no_year_calibration_final_recommendations.csv
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

from digihealth_risk.phase_4.calibrate_trees import (  # noqa: E402
    DEFAULT_CONFIGS,
    final_recommendations,
    run_config,
)

OUT_DIR = ROOT / "digihealth_risk" / "phase_7" / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[phase_7] dropping features: {dropped_feature_names()}")
    print(f"[phase_7] running {len(DEFAULT_CONFIGS)} configs")

    metric_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    curve_parts: list[pd.DataFrame] = []

    for config in DEFAULT_CONFIGS:
        metrics, thresholds, predictions, curves = run_config(config)
        metric_parts.append(metrics)
        threshold_parts.append(thresholds)
        prediction_parts.append(predictions)
        curve_parts.append(curves)

    metrics_df = pd.concat(metric_parts, ignore_index=True)
    thresholds_df = pd.concat(threshold_parts, ignore_index=True)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    curves_df = pd.concat(curve_parts, ignore_index=True)
    recommendations_df = final_recommendations(metrics_df)

    for df in (metrics_df, thresholds_df, predictions_df, curves_df, recommendations_df):
        df.insert(0, "variant", "no_year")

    metrics_df.to_csv(OUT_DIR / "phase_7_no_year_calibration_metrics.csv", index=False)
    thresholds_df.to_csv(OUT_DIR / "phase_7_no_year_calibration_threshold_table.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "phase_7_no_year_calibration_test_predictions.csv", index=False)
    curves_df.to_csv(OUT_DIR / "phase_7_no_year_calibration_curves.csv", index=False)
    recommendations_df.to_csv(
        OUT_DIR / "phase_7_no_year_calibration_final_recommendations.csv", index=False
    )

    print("\n[phase_7] Recommendations (no_year variant):")
    print(recommendations_df.to_string(index=False))


if __name__ == "__main__":
    main()
