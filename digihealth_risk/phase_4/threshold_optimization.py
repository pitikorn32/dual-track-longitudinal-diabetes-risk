"""Phase 5 v2 calibration and threshold optimization.

Changes from v1 (phase_5_calibration_thresholds.py) driven by Phase 0.2 EDA:
  - Imports from digihealth_risk.phase_2.train_tree_models so pulse_pressure is excluded
    and the v2 LEAKAGE_OR_METADATA_COLUMNS set is used by get_feature_columns().
  - Calls engineer_features() after load_table() to add v2 engineered features:
      FBS_hinge_100, FBS_hinge_125  (hockey-stick at pre-DM/DM thresholds)
      Year_centered_sq              (U-shaped temporal trend, Ljung-Box p=0.03)
      FBS_x_Age                     (Phase 0.2 top cross-lag interaction)
      MAX_FBS_x_Age                 (MAX_FBS_up_to_year × Age, cross-lag r=0.582)
  - Slope-hybrid configs use phase_2_v2_modeling_table_with_slopes.pkl
    (v2 slopes: FBS_hinge_100 slope replaces pulse_pressure slope).
  - Output prefix: phase_5_v2_*

All calibration methods, threshold policies, and output schema are identical to v1.

Run from the repository root:
    python digihealth_risk/phase_4/threshold_optimization.py
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

from digihealth_risk.phase_2.train_tree_models import (
    RANDOM_SEED,
    build_model,
    engineer_features,
    get_feature_columns,
    install_numpy_pickle_compat,
    make_preprocessor,
)
from digihealth_risk.utils.patient_split import apply_canonical_split


PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
PHASE2_OUT = ROOT / "digihealth_risk" / "phase_2" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_4" / "outputs"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    input_path: Path
    model_name: str
    description: str


DEFAULT_CONFIGS = [
    ModelConfig(
        key="n1_slope_catboost",
        input_path=PHASE2_OUT / "phase_2_v2_modeling_table_with_slopes.pkl",
        model_name="catboost",
        description="N=1,M=1 Phase 2 v2 slope hybrid CatBoost",
    ),
    ModelConfig(
        key="n1_slope_xgboost",
        input_path=PHASE2_OUT / "phase_2_v2_modeling_table_with_slopes.pkl",
        model_name="xgboost",
        description="N=1,M=1 Phase 2 v2 slope hybrid XGBoost",
    ),
    ModelConfig(
        key="n3_history_catboost",
        input_path=PHASE0_OUT / "phase_0_modeling_table_horizon_3_history_5.pkl",
        model_name="catboost",
        description="N=3,M=5 Phase 2 v2 rolling-history CatBoost",
    ),
    ModelConfig(
        key="n3_history_xgboost",
        input_path=PHASE0_OUT / "phase_0_modeling_table_horizon_3_history_5.pkl",
        model_name="xgboost",
        description="N=3,M=5 Phase 2 v2 rolling-history XGBoost",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 5 v2 calibration and threshold analysis.")
    parser.add_argument(
        "--model-key",
        action="append",
        choices=[config.key for config in DEFAULT_CONFIGS],
        help="Model key to run. Can be passed multiple times. Defaults to all current best models.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
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
    x_train = train_df[numeric_features + categorical_features].copy()
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


def fit_calibrators(cal_probability: np.ndarray, y_cal: np.ndarray) -> dict[str, object]:
    clipped = np.clip(cal_probability, 1e-6, 1 - 1e-6)
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(logit(clipped).reshape(-1, 1), y_cal)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(cal_probability, y_cal)
    return {"raw": None, "platt": platt, "isotonic": isotonic}


def apply_calibrator(name: str, calibrator: object, probability: np.ndarray) -> np.ndarray:
    if name == "raw":
        return probability
    if name == "platt":
        clipped = np.clip(probability, 1e-6, 1 - 1e-6)
        return calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1]
    if name == "isotonic":
        return calibrator.predict(probability)
    raise ValueError(f"Unknown calibrator: {name}")


def threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probability >= threshold
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(((~pred) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "selected_rate": float(pred.mean()),
    }


def ranking_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def candidate_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(probability, 8))
    if len(unique) > 1000:
        unique = np.quantile(probability, np.linspace(0, 1, 1001))
    return np.unique(np.r_[0.0, unique, 1.0])


def select_thresholds(y_cal: np.ndarray, probability: np.ndarray, train_positive_rate: float) -> dict[str, float]:
    thresholds = candidate_thresholds(probability)
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": threshold, **threshold_metrics(y_cal, probability, threshold)})
    metrics = pd.DataFrame(rows)

    max_f1 = metrics.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]

    recall_candidates = metrics[metrics["recall"] >= 0.80]
    if recall_candidates.empty:
        recall80 = metrics.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]
    else:
        recall80 = recall_candidates.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]

    top10 = float(np.quantile(probability, 0.90))
    return {
        "train_positive_rate": float(train_positive_rate),
        "max_f1_on_calibration": float(max_f1["threshold"]),
        "recall_at_least_0_80_on_calibration": float(recall80["threshold"]),
        "top_10_percent_on_calibration": top10,
    }


def calibration_curve_rows(
    model_key: str,
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
    grouped.insert(0, "calibration_method", calibration_method)
    grouped.insert(0, "model_key", model_key)
    grouped["bin_index"] = np.arange(1, len(grouped) + 1)
    return grouped.to_dict("records")


def run_config(config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_table(config.input_path)
    df = engineer_features(df)   # v2: add FBS hinges, Year_centered_sq, FBS_x_Age, MAX_FBS_x_Age
    train_df, cal_df, test_df = grouped_train_cal_test_split(df)
    numeric_features, categorical_features = get_feature_columns(train_df)
    pipeline = fit_pipeline(config.model_name, train_df, numeric_features, categorical_features)

    p_cal_raw = predict_probability(pipeline, cal_df, numeric_features, categorical_features)
    p_test_raw = predict_probability(pipeline, test_df, numeric_features, categorical_features)
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_cal = cal_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    train_positive_rate = float(y_train.mean())

    calibrators = fit_calibrators(p_cal_raw, y_cal)
    metric_rows = []
    threshold_rows = []
    prediction_frames = []
    curve_rows = []

    base_prediction_frame = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    base_prediction_frame["model_key"] = config.key
    base_prediction_frame["model_name"] = config.model_name
    base_prediction_frame["horizon_years"] = int(df["prediction_horizon_years"].iloc[0])
    base_prediction_frame["history_years"] = int(df["history_window_years"].iloc[0])

    for calibration_method, calibrator in calibrators.items():
        p_cal = apply_calibrator(calibration_method, calibrator, p_cal_raw)
        p_test = apply_calibrator(calibration_method, calibrator, p_test_raw)
        thresholds = select_thresholds(y_cal, p_cal, train_positive_rate)

        pred_frame = base_prediction_frame.copy()
        pred_frame["calibration_method"] = calibration_method
        pred_frame["predicted_probability"] = p_test
        prediction_frames.append(pred_frame)
        curve_rows.extend(calibration_curve_rows(config.key, calibration_method, y_test, p_test))

        base_metrics = {
            "model_key": config.key,
            "model_name": config.model_name,
            "description": config.description,
            "calibration_method": calibration_method,
            "input_path": str(config.input_path.relative_to(ROOT)),
            "horizon_years": int(df["prediction_horizon_years"].iloc[0]),
            "history_years": int(df["history_window_years"].iloc[0]),
            "train_rows": float(len(train_df)),
            "calibration_rows": float(len(cal_df)),
            "test_rows": float(len(test_df)),
            "test_positives": float(y_test.sum()),
            "test_positive_rate": float(y_test.mean()),
            **ranking_metrics(y_test, p_test),
        }

        for strategy, threshold in thresholds.items():
            threshold_rows.append(
                {
                    "model_key": config.key,
                    "calibration_method": calibration_method,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                }
            )
            metric_rows.append(
                {
                    **base_metrics,
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


def best_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    filtered = metrics[metrics["threshold_strategy"].eq("recall_at_least_0_80_on_calibration")].copy()
    filtered = filtered.sort_values(["horizon_years", "pr_auc", "brier"], ascending=[True, False, True])
    return filtered.groupby("horizon_years", as_index=False).head(5)


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


def write_report(metrics: pd.DataFrame) -> str:
    ranking_cols = [
        "horizon_years",
        "model_key",
        "calibration_method",
        "roc_auc",
        "pr_auc",
        "brier",
        "threshold_strategy",
        "threshold",
        "precision",
        "recall",
        "specificity",
        "f1",
        "selected_rate",
    ]
    brier = (
        metrics[metrics["threshold_strategy"].eq("train_positive_rate")]
        .sort_values(["horizon_years", "model_key", "brier"])
        .groupby(["horizon_years", "model_key"], as_index=False)
        .head(1)
    )

    lines = [
        "# Phase 5 v2 Calibration and Threshold Report",
        "",
        "## Scope",
        "Calibrates v2-enhanced tree models (slope-hybrid at N=1, rolling-history at N=3) with a grouped "
        "train/calibration/test split. Calibration methods: raw probabilities, Platt scaling, isotonic regression.",
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
        "Slope configs use v2 slope table (`phase_2_v2_modeling_table_with_slopes.pkl`): "
        "`FBS_hinge_100` slope replaces `pulse_pressure` slope.",
        "",
        "## Best Recall-Constrained Rows",
        "Rows below use thresholds selected on the calibration split to target recall >= 0.80, then evaluated on held-out test patients.",
        markdown_table(best_rows(metrics)[ranking_cols], max_rows=20),
        "",
        "## Best Brier Score Per Model",
        "Lower Brier means better probability calibration.",
        markdown_table(brier[ranking_cols], max_rows=20),
        "",
        "## Threshold Strategies",
        "- `train_positive_rate`: threshold equals the training positive rate.",
        "- `max_f1_on_calibration`: threshold maximizing F1 on calibration patients.",
        "- `recall_at_least_0_80_on_calibration`: highest-precision threshold that keeps calibration recall >= 0.80.",
        "- `top_10_percent_on_calibration`: flags approximately the top 10% highest-risk calibration rows.",
        "",
        "## Recommendation",
        "Use PR-AUC for model ranking, Brier score for probability trustworthiness, and the recall-constrained threshold for clinical screening review.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = {config.key: config for config in DEFAULT_CONFIGS}
    configs = [selected[key] for key in args.model_key] if args.model_key else DEFAULT_CONFIGS

    metric_parts = []
    threshold_parts = []
    prediction_parts = []
    curve_parts = []
    for config in configs:
        print(f"Running {config.key}: {config.description}")
        metrics, thresholds, predictions, curves = run_config(config)
        metric_parts.append(metrics)
        threshold_parts.append(thresholds)
        prediction_parts.append(predictions)
        curve_parts.append(curves)

    metrics_df = pd.concat(metric_parts, ignore_index=True)
    thresholds_df = pd.concat(threshold_parts, ignore_index=True)
    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    curves_df = pd.concat(curve_parts, ignore_index=True)
    leaderboard_df = best_rows(metrics_df)

    metrics_df.to_csv(OUT_DIR / "phase_5_v2_calibration_metrics.csv", index=False)
    thresholds_df.to_csv(OUT_DIR / "phase_5_v2_thresholds.csv", index=False)
    predictions_df.to_csv(OUT_DIR / "phase_5_v2_test_predictions.csv", index=False)
    curves_df.to_csv(OUT_DIR / "phase_5_v2_calibration_curves.csv", index=False)
    leaderboard_df.to_csv(OUT_DIR / "phase_5_v2_leaderboard.csv", index=False)

    report = write_report(metrics_df)
    (OUT_DIR / "phase_5_v2_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
