"""Phase 4 v2 final candidate calibration and threshold selection.

Changes from v1 (phase_4_final_threshold_calibration.py) driven by Phase 0.2 EDA:
  - Imports from digihealth_risk.phase_2.train_tree_models so pulse_pressure is excluded and
    the v2 LEAKAGE_OR_METADATA_COLUMNS set is used by get_feature_columns().
  - Calls engineer_features() after load_table() to add v2 engineered features:
      FBS_hinge_100, FBS_hinge_125  (hockey-stick at pre-DM/DM thresholds)
      Year_centered_sq              (U-shaped temporal trend, Ljung-Box p=0.03)
      FBS_x_Age                     (Phase 0.2 top cross-lag interaction)
      MAX_FBS_x_Age                 (MAX_FBS_up_to_year × Age, cross-lag r=0.582)
  - Output prefix: phase_4_v2_*

All calibration methods, threshold policies, split logic, and output schema
are identical to v1.

Run from the repository root:
    python digihealth_risk/phase_4/calibrate_trees.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_2.train_tree_models import (  # noqa: E402
    RANDOM_SEED,
    build_model,
    engineer_features,
    get_feature_columns,
    install_numpy_pickle_compat,
    make_preprocessor,
)
from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402


PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_4" / "outputs"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    input_path: Path
    model_name: str
    horizon_years: int
    history_years: int
    role: str


def phase0_path(horizon_years: int, history_years: int = 5) -> Path:
    if horizon_years == 1 and history_years == 1:
        return PHASE0_OUT / "phase_0_modeling_table.pkl"
    return PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon_years}_history_{history_years}.pkl"


DEFAULT_CONFIGS = [
    ModelConfig("n1_m1_catboost", phase0_path(1, 1), "catboost", 1, 1, "candidate"),
    ModelConfig("n1_m1_xgboost", phase0_path(1, 1), "xgboost", 1, 1, "candidate"),
    ModelConfig("n1_m3_catboost", phase0_path(1, 3), "catboost", 1, 3, "candidate"),
    ModelConfig("n1_m3_xgboost", phase0_path(1, 3), "xgboost", 1, 3, "candidate"),
    ModelConfig("n1_m5_catboost", phase0_path(1, 5), "catboost", 1, 5, "candidate"),
    ModelConfig("n1_m5_xgboost", phase0_path(1, 5), "xgboost", 1, 5, "candidate"),
    ModelConfig("n2_m1_catboost", phase0_path(2, 1), "catboost", 2, 1, "candidate"),
    ModelConfig("n2_m1_xgboost", phase0_path(2, 1), "xgboost", 2, 1, "candidate"),
    ModelConfig("n2_m3_catboost", phase0_path(2, 3), "catboost", 2, 3, "candidate"),
    ModelConfig("n2_m3_xgboost", phase0_path(2, 3), "xgboost", 2, 3, "candidate"),
    ModelConfig("n2_m5_catboost", phase0_path(2, 5), "catboost", 2, 5, "candidate"),
    ModelConfig("n2_m5_xgboost", phase0_path(2, 5), "xgboost", 2, 5, "candidate"),
    ModelConfig("n3_m1_catboost", phase0_path(3, 1), "catboost", 3, 1, "candidate"),
    ModelConfig("n3_m1_xgboost", phase0_path(3, 1), "xgboost", 3, 1, "candidate"),
    ModelConfig("n3_m3_catboost", phase0_path(3, 3), "catboost", 3, 3, "candidate"),
    ModelConfig("n3_m3_xgboost", phase0_path(3, 3), "xgboost", 3, 3, "candidate"),
    ModelConfig("n3_m5_catboost", phase0_path(3, 5), "catboost", 3, 5, "candidate"),
    ModelConfig("n3_m5_xgboost", phase0_path(3, 5), "xgboost", 3, 5, "candidate"),
    ModelConfig("n4_m1_catboost", phase0_path(4, 1), "catboost", 4, 1, "candidate"),
    ModelConfig("n4_m1_xgboost", phase0_path(4, 1), "xgboost", 4, 1, "candidate"),
    ModelConfig("n4_m3_catboost", phase0_path(4, 3), "catboost", 4, 3, "candidate"),
    ModelConfig("n4_m3_xgboost", phase0_path(4, 3), "xgboost", 4, 3, "candidate"),
    ModelConfig("n4_m5_catboost", phase0_path(4, 5), "catboost", 4, 5, "candidate"),
    ModelConfig("n4_m5_xgboost", phase0_path(4, 5), "xgboost", 4, 5, "candidate"),
    ModelConfig("n5_m1_catboost", phase0_path(5, 1), "catboost", 5, 1, "candidate"),
    ModelConfig("n5_m1_xgboost", phase0_path(5, 1), "xgboost", 5, 1, "candidate"),
    ModelConfig("n5_m3_catboost", phase0_path(5, 3), "catboost", 5, 3, "candidate"),
    ModelConfig("n5_m3_xgboost", phase0_path(5, 3), "xgboost", 5, 3, "candidate"),
    ModelConfig("n5_m5_catboost", phase0_path(5, 5), "catboost", 5, 5, "candidate"),
    ModelConfig("n5_m5_xgboost", phase0_path(5, 5), "xgboost", 5, 5, "candidate"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4 v2 threshold and calibration analysis.")
    parser.add_argument(
        "--model-key",
        action="append",
        choices=[config.key for config in DEFAULT_CONFIGS],
        help="Model key to run. Can be passed multiple times. Defaults to all Phase 2.3 M=1/M=3/M=5 tree finalists.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 0 table: {path}")
    install_numpy_pickle_compat()
    df = pd.read_pickle(path).copy()
    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]
    return df


def grouped_train_cal_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df, return_calibration=True)


def fit_pipeline(
    model_name: str,
    train_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    feature_columns = numeric_features + categorical_features
    x_train = train_df[feature_columns].copy()
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    positives = y_train.sum()
    negatives = len(y_train) - positives
    scale_pos_weight = float(negatives / positives) if positives else 1.0

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features)),
            ("model", build_model(model_name, scale_pos_weight, use_class_weights=False)),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def predict_probability(
    pipeline: Pipeline,
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    return pipeline.predict_proba(df[numeric_features + categorical_features].copy())[:, 1]


def fit_calibrators(calibration_probability: np.ndarray, y_calibration: np.ndarray) -> dict[str, object | None]:
    clipped = np.clip(calibration_probability, 1e-6, 1 - 1e-6)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(logit(clipped).reshape(-1, 1), y_calibration)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(calibration_probability, y_calibration)
    return {"raw": None, "platt": platt, "isotonic": isotonic}


def apply_calibrator(name: str, calibrator: object | None, probability: np.ndarray) -> np.ndarray:
    if name == "raw":
        return probability
    if name == "platt":
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        return calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1]
    if name == "isotonic":
        return calibrator.predict(probability)
    raise ValueError(f"Unknown calibration method: {name}")


def ranking_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    positives_captured = int(((prediction) & (y_true == 1)).sum())
    flagged = int(prediction.sum())
    return {
        "threshold": float(threshold),
        "flagged_rows": float(flagged),
        "flagged_rate": float(prediction.mean()),
        "positives_captured": float(positives_captured),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(((~prediction) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def candidate_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(probability, 8))
    if len(unique) > 2000:
        unique = np.quantile(probability, np.linspace(0, 1, 2001))
    return np.unique(np.r_[0.0, unique, 1.0])


def threshold_grid(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [{"threshold": threshold, **threshold_metrics(y_true, probability, threshold)} for threshold in candidate_thresholds(probability)]
    )


def top_k_threshold(probability: np.ndarray, top_fraction: float) -> float:
    return float(np.quantile(probability, 1.0 - top_fraction))


def select_recall_threshold(grid: pd.DataFrame, target_recall: float) -> float:
    candidates = grid[grid["recall"] >= target_recall]
    if candidates.empty:
        selected = grid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        selected = candidates.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    return float(selected["threshold"])


def select_thresholds(
    y_calibration: np.ndarray,
    probability: np.ndarray,
    train_positive_rate: float,
) -> dict[str, float]:
    grid = threshold_grid(y_calibration, probability)
    max_f1 = grid.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return {
        "train_positive_rate": float(train_positive_rate),
        "recall_at_least_0_80": select_recall_threshold(grid, 0.80),
        "recall_at_least_0_85": select_recall_threshold(grid, 0.85),
        "max_f1": float(max_f1["threshold"]),
        "top_5_percent": top_k_threshold(probability, 0.05),
        "top_10_percent": top_k_threshold(probability, 0.10),
        "top_20_percent": top_k_threshold(probability, 0.20),
    }


def calibration_curve_rows(
    config: ModelConfig,
    calibration_method: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float | str]]:
    labels = pd.qcut(probability, q=bins, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": probability, "bin": labels})
    grouped = (
        df.groupby("bin", observed=True)
        .agg(rows=("y", "size"), mean_probability=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    grouped.insert(0, "bin_index", np.arange(1, len(grouped) + 1))
    grouped.insert(0, "calibration_method", calibration_method)
    grouped.insert(0, "model_name", config.model_name)
    grouped.insert(0, "model_key", config.key)
    grouped.insert(0, "history_years", config.history_years)
    grouped.insert(0, "horizon_years", config.horizon_years)
    return grouped.to_dict("records")


def run_config(config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Running {config.key}")
    df = load_table(config.input_path)
    df = engineer_features(df)   # v2: add FBS hinges, Year_centered_sq, FBS_x_Age, MAX_FBS_x_Age
    train_df, calibration_df, test_df = grouped_train_cal_test_split(df)
    numeric_features, categorical_features = get_feature_columns(train_df)
    pipeline = fit_pipeline(config.model_name, train_df, numeric_features, categorical_features)

    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_calibration = calibration_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    train_positive_rate = float(y_train.mean())

    p_calibration_raw = predict_probability(pipeline, calibration_df, numeric_features, categorical_features)
    p_test_raw = predict_probability(pipeline, test_df, numeric_features, categorical_features)
    calibrators = fit_calibrators(p_calibration_raw, y_calibration)

    metric_rows = []
    threshold_rows = []
    prediction_frames = []
    curve_rows = []

    base_predictions = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    base_predictions["model_key"] = config.key
    base_predictions["model_name"] = config.model_name
    base_predictions["horizon_years"] = config.horizon_years
    base_predictions["history_years"] = config.history_years

    for calibration_method, calibrator in calibrators.items():
        p_calibration = apply_calibrator(calibration_method, calibrator, p_calibration_raw)
        p_test = apply_calibrator(calibration_method, calibrator, p_test_raw)
        thresholds = select_thresholds(y_calibration, p_calibration, train_positive_rate)
        rank = ranking_metrics(y_test, p_test)

        predictions = base_predictions.copy()
        predictions["calibration_method"] = calibration_method
        predictions["predicted_probability"] = p_test
        prediction_frames.append(predictions)
        curve_rows.extend(calibration_curve_rows(config, calibration_method, y_test, p_test))

        common = {
            "model_key": config.key,
            "model_name": config.model_name,
            "role": config.role,
            "calibration_method": calibration_method,
            "horizon_years": config.horizon_years,
            "history_years": config.history_years,
            "input_path": str(config.input_path.relative_to(ROOT)),
            "train_rows": float(len(train_df)),
            "calibration_rows": float(len(calibration_df)),
            "test_rows": float(len(test_df)),
            "test_positives": float(y_test.sum()),
            "test_positive_rate": float(y_test.mean()),
            **rank,
        }
        for strategy, threshold in thresholds.items():
            threshold_rows.append(
                {
                    "model_key": config.key,
                    "model_name": config.model_name,
                    "calibration_method": calibration_method,
                    "horizon_years": config.horizon_years,
                    "history_years": config.history_years,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                }
            )
            metric_rows.append(
                {
                    **common,
                    "threshold_strategy": strategy,
                    **threshold_metrics(y_test, p_test, threshold),
                }
            )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(threshold_rows),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(curve_rows),
    )


def best_probability_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["threshold_strategy"].eq("train_positive_rate")].copy()
    return (
        base.sort_values(["horizon_years", "pr_auc", "brier"], ascending=[True, False, True])
        .groupby("horizon_years", as_index=False)
        .head(1)
    )


def best_brier_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["threshold_strategy"].eq("train_positive_rate")].copy()
    return (
        base.sort_values(["horizon_years", "model_key", "brier"])
        .groupby(["horizon_years", "model_key"], as_index=False)
        .head(1)
    )


def recall_policy_rows(metrics: pd.DataFrame, strategy: str = "recall_at_least_0_80") -> pd.DataFrame:
    base = metrics[metrics["threshold_strategy"].eq(strategy)].copy()
    return (
        base.sort_values(["horizon_years", "pr_auc", "precision"], ascending=[True, False, False])
        .groupby("horizon_years", as_index=False)
        .head(2)
    )


def final_recommendations(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best_probability = best_probability_rows(metrics)
    recall80 = recall_policy_rows(metrics, "recall_at_least_0_80")
    brier = best_brier_rows(metrics)

    for _, row in best_probability.iterrows():
        rows.append(
            {
                "horizon_years": row["horizon_years"],
                "recommendation_type": "best_pr_auc",
                "model_key": row["model_key"],
                "calibration_method": row["calibration_method"],
                "threshold_strategy": row["threshold_strategy"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "brier": row["brier"],
                "precision": row["precision"],
                "recall": row["recall"],
                "flagged_rate": row["flagged_rate"],
            }
        )

    for _, row in recall80.groupby("horizon_years", as_index=False).head(1).iterrows():
        rows.append(
            {
                "horizon_years": row["horizon_years"],
                "recommendation_type": "recall_0_80_policy",
                "model_key": row["model_key"],
                "calibration_method": row["calibration_method"],
                "threshold_strategy": row["threshold_strategy"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "brier": row["brier"],
                "precision": row["precision"],
                "recall": row["recall"],
                "flagged_rate": row["flagged_rate"],
            }
        )

    for _, row in brier.sort_values(["horizon_years", "brier"]).groupby("horizon_years", as_index=False).head(1).iterrows():
        rows.append(
            {
                "horizon_years": row["horizon_years"],
                "recommendation_type": "best_brier",
                "model_key": row["model_key"],
                "calibration_method": row["calibration_method"],
                "threshold_strategy": row["threshold_strategy"],
                "pr_auc": row["pr_auc"],
                "roc_auc": row["roc_auc"],
                "brier": row["brier"],
                "precision": row["precision"],
                "recall": row["recall"],
                "flagged_rate": row["flagged_rate"],
            }
        )

    return pd.DataFrame(rows).sort_values(["horizon_years", "recommendation_type"])


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    display = df.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    columns = display.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    display_cols = [
        "horizon_years",
        "model_key",
        "calibration_method",
        "threshold_strategy",
        "pr_auc",
        "roc_auc",
        "brier",
        "threshold",
        "flagged_rate",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    recommendation_cols = [
        "horizon_years",
        "recommendation_type",
        "model_key",
        "calibration_method",
        "threshold_strategy",
        "pr_auc",
        "brier",
        "precision",
        "recall",
        "flagged_rate",
    ]

    lines = [
        "# Phase 4 v2 Final Threshold and Calibration Report",
        "",
        "## Scope",
        "Retrains Phase 2.3 tree finalist candidates (`M=1`, `M=3`, and `M=5`) with v2 feature engineering and grouped "
        "train/calibration/test splits. Thresholds are selected on calibration patients and "
        "evaluated on held-out test patients.",
        "",
        "## v2 Feature Changes",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 by construction |",
        "| Added | `FBS_hinge_100` | Hockey-stick at pre-DM threshold (100 mg/dL) |",
        "| Added | `FBS_hinge_125` | Hockey-stick at DM threshold (125 mg/dL) |",
        "| Added | `Year_centered_sq` | U-shaped temporal risk trend (Ljung-Box p=0.03) |",
        "| Added | `FBS_x_Age` | Phase 0.2 top cross-lag interaction |",
        "| Added | `MAX_FBS_x_Age` | MAX_FBS_up_to_year × Age (cross-lag r=0.582) |",
        "",
        "**Split:** 60% train / 20% calibration / 20% test (patient-grouped, seed `20260501`). "
        "Note: the training set is ~25% smaller than in Phases 1–3 (which use 80/20). "
        "This may slightly disadvantage Phase 4 model performance relative to earlier phases.",
        "",
        "## Final Recommendations",
        markdown_table(recommendations[recommendation_cols]),
        "",
        "## Best PR-AUC Per Horizon",
        markdown_table(best_probability_rows(metrics)[display_cols]),
        "",
        "## Recall >= 0.80 Policy",
        "The threshold is chosen on calibration patients to target recall >= 0.80, then evaluated on test patients.",
        markdown_table(recall_policy_rows(metrics, "recall_at_least_0_80")[display_cols]),
        "",
        "## Best Brier Score Per Model",
        "Lower Brier means better probability calibration.",
        markdown_table(best_brier_rows(metrics)[display_cols], max_rows=30),
        "",
        "## Threshold Strategies",
        "- `train_positive_rate`: threshold equals the training positive rate.",
        "- `recall_at_least_0_80`: highest-precision calibration threshold with recall >= 0.80.",
        "- `recall_at_least_0_85`: highest-precision calibration threshold with recall >= 0.85.",
        "- `max_f1`: threshold maximizing F1 on calibration patients.",
        "- `top_5_percent`, `top_10_percent`, `top_20_percent`: flag fixed review-capacity fractions.",
        "",
        "## Notes",
        "- PR-AUC is still the primary model-ranking metric for rare `AtRisk` events.",
        "- Calibration can improve Brier score without improving PR-AUC.",
        "- Choose the clinical horizon first; do not rank different `N` values as the same task.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_by_key = {config.key: config for config in DEFAULT_CONFIGS}
    configs = [config_by_key[key] for key in args.model_key] if args.model_key else DEFAULT_CONFIGS

    metric_parts = []
    threshold_parts = []
    prediction_parts = []
    curve_parts = []
    for config in configs:
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

    metrics_df.to_csv(OUT_DIR / "phase_4_v2_metrics.csv", index=False)
    thresholds_df.to_csv(OUT_DIR / "phase_4_v2_threshold_table.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "phase_4_v2_test_predictions.csv", index=False)
    curves_df.to_csv(OUT_DIR / "phase_4_v2_calibration_curves.csv", index=False)
    recommendations_df.to_csv(OUT_DIR / "phase_4_v2_final_recommendations.csv", index=False)

    report = write_report(metrics_df, recommendations_df)
    (OUT_DIR / "phase_4_v2_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
