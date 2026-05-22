"""Phase 4.2 v2 cross-family model comparison.

Combines v2 Phase 4 tree finalists with Phase 3.2 v2 landmark Cox, Phase 3.3
v2 two-stage survival, and Phase 1 statistical outputs. Makes the final
recommendation compare model families using v2 feature engineering where
available.

Model sources:
  - Trees:        phase_4_v2_test_predictions.csv   (v2 features)
  - Landmark Cox: phase_3_2_v2_test_predictions.csv (v2 features)
  - Two-stage:    phase_3_3_v2_h*_test_predictions.csv (v2 features)
  - GEE:          phase_1_gee_horizon_* and phase_1_v2_gee_horizon_* grids
  - Logistic:     phase_1_v2_logistic_horizon_* grids

Run from the repository root:
    python digihealth_risk/phase_4/cross_family_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
PHASE1_OUT = ROOT / "digihealth_risk" / "phase_1" / "outputs"
PHASE3_2_OUT = ROOT / "digihealth_risk" / "phase_3" / "outputs"
PHASE3_3_OUT = ROOT / "digihealth_risk" / "phase_3" / "outputs"
PHASE4_OUT = ROOT / "digihealth_risk" / "phase_4" / "outputs"
KEY_COLUMNS = ["PatientId", "Year", "target_year"]


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "threshold": float(threshold),
        "flagged_rows": float(prediction.sum()),
        "flagged_rate": float(prediction.mean()),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(((~prediction) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def ranking_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "rows": float(len(y_true)),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
    }


def candidate_thresholds(probability: np.ndarray) -> np.ndarray:
    unique = np.unique(np.round(probability, 8))
    if len(unique) > 2000:
        unique = np.quantile(probability, np.linspace(0, 1, 2001))
    return np.unique(np.r_[0.0, unique, 1.0])


def threshold_grid(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        [{"threshold": threshold, **classification_metrics(y_true, probability, threshold)} for threshold in candidate_thresholds(probability)]
    )


def select_recall_threshold(grid: pd.DataFrame, target_recall: float) -> float:
    candidates = grid[grid["recall"] >= target_recall]
    if candidates.empty:
        selected = grid.sort_values(["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        selected = candidates.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
    return float(selected["threshold"])


def diagnostic_thresholds(y_true: np.ndarray, probability: np.ndarray) -> dict[str, tuple[float, str]]:
    """Thresholds that use test labels are diagnostic/oracle, not deployment-valid."""
    grid = threshold_grid(y_true, probability)
    max_f1 = grid.sort_values(["f1", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return {
        "recall_at_least_0_80_test_oracle": (select_recall_threshold(grid, 0.80), "test_oracle"),
        "recall_at_least_0_85_test_oracle": (select_recall_threshold(grid, 0.85), "test_oracle"),
        "max_f1_test_oracle": (float(max_f1["threshold"]), "test_oracle"),
    }


def capacity_thresholds(probability: np.ndarray) -> dict[str, tuple[float, str]]:
    return {
        "top_5_percent_test_distribution": (float(np.quantile(probability, 0.95)), "test_distribution"),
        "top_10_percent_test_distribution": (float(np.quantile(probability, 0.90)), "test_distribution"),
        "top_20_percent_test_distribution": (float(np.quantile(probability, 0.80)), "test_distribution"),
    }


def normalize_prediction_frame(
    df: pd.DataFrame,
    *,
    approach: str,
    model_family: str,
    model_key: str,
    model_name: str,
    horizon_years: int,
    history_years: int | float,
    calibration_method: str,
    threshold_source: str,
    train_positive_threshold: float | None,
) -> pd.DataFrame:
    target_column = "Target_AtRisk_Status"
    if target_column not in df.columns and "AtRisk_within_horizon" in df.columns:
        target_column = "AtRisk_within_horizon"
    if target_column not in df.columns:
        raise ValueError(f"Missing target column for {model_key}")

    result = pd.DataFrame(
        {
            "PatientId": df["PatientId"].astype(str).to_numpy(),
            "Year": df["Year"].astype(int).to_numpy(),
            "target_year": df["target_year"].astype(int).to_numpy(),
            "approach": approach,
            "model_family": model_family,
            "model_key": model_key,
            "model_name": model_name,
            "horizon_years": horizon_years,
            "history_years": history_years,
            "calibration_method": calibration_method,
            "threshold_source": threshold_source,
            "train_positive_threshold": train_positive_threshold,
            "Target_AtRisk_Status": df[target_column].astype(int).to_numpy(),
            "predicted_probability": df["predicted_probability"].astype(float).to_numpy(),
        }
    )
    return result


def validate_target_alignment(predictions: pd.DataFrame) -> None:
    grouped = (
        predictions.groupby(["horizon_years", *KEY_COLUMNS], sort=False)["Target_AtRisk_Status"]
        .nunique()
        .reset_index(name="target_nunique")
    )
    inconsistent = grouped[grouped["target_nunique"].gt(1)]
    if not inconsistent.empty:
        sample = inconsistent.head(5).to_dict("records")
        raise ValueError(f"Inconsistent target labels across model outputs. Sample: {sample}")


def align_shared_cohort(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for horizon, horizon_df in predictions.groupby("horizon_years", sort=True):
        model_instances = (
            horizon_df[["model_key", "calibration_method"]]
            .drop_duplicates()
            .shape[0]
        )
        key_counts = (
            horizon_df[["model_key", "calibration_method", *KEY_COLUMNS]]
            .drop_duplicates()
            .groupby(KEY_COLUMNS, sort=False)
            .size()
            .reset_index(name="model_count")
        )
        shared_keys = key_counts[key_counts["model_count"].eq(model_instances)][KEY_COLUMNS].copy()
        aligned = horizon_df.merge(shared_keys, on=KEY_COLUMNS, how="inner")

        summary_rows.append(
            {
                "horizon_years": int(horizon),
                "model_instances": int(model_instances),
                "rows_before_alignment_min": int(
                    horizon_df.groupby(["model_key", "calibration_method"], sort=False).size().min()
                ),
                "rows_before_alignment_max": int(
                    horizon_df.groupby(["model_key", "calibration_method"], sort=False).size().max()
                ),
                "shared_rows": int(len(shared_keys)),
                "shared_positives": int(shared_keys.merge(
                    aligned[KEY_COLUMNS + ["Target_AtRisk_Status"]].drop_duplicates(),
                    on=KEY_COLUMNS,
                    how="left",
                )["Target_AtRisk_Status"].sum()),
                "shared_positive_rate": float(
                    shared_keys.merge(
                        aligned[KEY_COLUMNS + ["Target_AtRisk_Status"]].drop_duplicates(),
                        on=KEY_COLUMNS,
                        how="left",
                    )["Target_AtRisk_Status"].mean()
                ),
            }
        )
        aligned_parts.append(aligned)

    return pd.concat(aligned_parts, ignore_index=True), pd.DataFrame(summary_rows)


def load_phase1_gee() -> list[pd.DataFrame]:
    specs = []
    for version_name, prefix, model_name in [
        ("v1", "phase_1_gee", "GEE v1"),
        ("v2", "phase_1_v2_gee", "GEE v2"),
    ]:
        for history in [1, 3, 5]:
            for horizon in [1, 2, 3, 4, 5]:
                specs.append(
                    (
                        PHASE1_OUT / f"{prefix}_horizon_{horizon}_history_{history}_test_predictions.csv",
                        PHASE1_OUT / f"{prefix}_horizon_{horizon}_history_{history}_metrics.csv",
                        f"phase1_gee_{version_name}_n{horizon}_m{history}",
                        model_name,
                        horizon,
                        history,
                    )
                )
    frames = []
    for pred_path, metric_path, key, model_name, horizon, history in specs:
        if not pred_path.exists() or not metric_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        metric = pd.read_csv(metric_path)
        threshold = float(metric.loc[metric["split"].eq("train"), "threshold"].iloc[0])
        frames.append(
            normalize_prediction_frame(
                pred,
                approach="statistical",
                model_family="gee",
                model_key=key,
                model_name=model_name,
                horizon_years=horizon,
                history_years=history,
                calibration_method="raw",
                threshold_source="train_positive_rate",
                train_positive_threshold=threshold,
            )
        )
    return frames


def load_phase1_logistic_v2() -> list[pd.DataFrame]:
    specs = [
        (
            PHASE1_OUT / f"phase_1_v2_logistic_horizon_{horizon}_history_{history}_test_predictions.csv",
            PHASE1_OUT / f"phase_1_v2_logistic_horizon_{horizon}_history_{history}_metrics.csv",
            f"phase1_logistic_v2_n{horizon}_m{history}",
            horizon,
            history,
        )
        for history in [1, 3, 5]
        for horizon in [1, 2, 3, 4, 5]
    ]
    frames = []
    for pred_path, metric_path, key, horizon, history in specs:
        if not pred_path.exists() or not metric_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        metric = pd.read_csv(metric_path)
        threshold = float(metric.loc[metric["split"].eq("train"), "threshold"].iloc[0])
        frames.append(
            normalize_prediction_frame(
                pred,
                approach="statistical",
                model_family="logistic",
                model_key=key,
                model_name="Logistic v2",
                horizon_years=horizon,
                history_years=history,
                calibration_method="raw",
                threshold_source="train_positive_rate",
                train_positive_threshold=threshold,
            )
        )
    return frames


def load_phase3_2_landmark_cox() -> list[pd.DataFrame]:
    pred_path = PHASE3_2_OUT / "phase_3_2_v2_test_predictions.csv"
    metrics_path = PHASE3_2_OUT / "phase_3_2_v2_binary_horizon_metrics.csv"
    if not pred_path.exists() or not metrics_path.exists():
        print("WARNING: Phase 3.2 v2 predictions not found — skipping landmark Cox.")
        return []
    pred = pd.read_csv(pred_path)
    metrics = pd.read_csv(metrics_path)
    frames = []
    for horizon, group in pred.groupby("horizon_years", sort=True):
        threshold = float(
            metrics.loc[
                metrics["split"].eq("train") & metrics["horizon_years"].eq(horizon),
                "threshold",
            ].iloc[0]
        )
        frames.append(
            normalize_prediction_frame(
                group,
                approach="survival",
                model_family="landmark_cox",
                model_key=f"phase3_2_v2_landmark_cox_n{int(horizon)}",
                model_name="Landmark Cox v2",
                horizon_years=int(horizon),
                history_years=np.nan,
                calibration_method="raw",
                threshold_source="train_positive_rate",
                train_positive_threshold=threshold,
            )
        )
    return frames


def load_phase3_3_two_stage() -> list[pd.DataFrame]:
    frames = []
    for history in [1, 3, 5]:
        prefixes = [
            f"phase_3_3_v2_history_{history}",
            f"phase_3_3_v2_m{history}",
        ]
        if history == 3:
            prefixes.append("phase_3_3_v2")

        pred_path = None
        metrics_path = None
        for prefix in prefixes:
            candidate_pred = PHASE3_3_OUT / f"{prefix}_test_predictions.csv"
            candidate_metrics = PHASE3_3_OUT / f"{prefix}_binary_horizon_metrics.csv"
            if candidate_pred.exists() and candidate_metrics.exists():
                pred_path = candidate_pred
                metrics_path = candidate_metrics
                break

        if pred_path is None or metrics_path is None:
            print(f"WARNING: Phase 3.3 v2 predictions not found for M={history} — skipping two-stage survival.")
            continue

        pred = pd.read_csv(pred_path)
        metrics = pd.read_csv(metrics_path)
        for horizon, group in pred.groupby("horizon_years", sort=True):
            threshold = float(
                metrics.loc[
                    metrics["split"].eq("train") & metrics["horizon_years"].eq(horizon),
                    "threshold",
                ].iloc[0]
            )
            frames.append(
                normalize_prediction_frame(
                    group,
                    approach="survival",
                    model_family="two_stage_survival",
                    model_key=f"phase3_3_v2_two_stage_n{int(horizon)}_m{history}",
                    model_name="Two-stage Survival v2",
                    horizon_years=int(horizon),
                    history_years=history,
                    calibration_method="raw",
                    threshold_source="train_positive_rate",
                    train_positive_threshold=threshold,
                )
            )
    return frames


def load_phase4_trees() -> list[pd.DataFrame]:
    pred_path = PHASE4_OUT / "phase_4_v2_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Phase 4 v2 predictions not found: {pred_path}\n"
            "Run phase_4_v2_final_threshold_calibration.py first."
        )
    pred = pd.read_csv(pred_path)
    frames = []
    for (model_key, calibration_method), group in pred.groupby(["model_key", "calibration_method"], sort=True):
        frames.append(
            normalize_prediction_frame(
                group,
                approach="tree",
                model_family=str(group["model_name"].iloc[0]),
                model_key=str(model_key),
                model_name=str(group["model_name"].iloc[0]),
                horizon_years=int(group["horizon_years"].iloc[0]),
                history_years=int(group["history_years"].iloc[0]),
                calibration_method=str(calibration_method),
                threshold_source="phase4_calibration",
                train_positive_threshold=None,
            )
        )
    return frames


def evaluate_tree_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.read_csv(PHASE4_OUT / "phase_4_v2_metrics.csv")
    rows: list[dict[str, object]] = []
    tree_predictions = predictions[predictions["approach"].eq("tree")].copy()

    for _, threshold_row in metrics.iterrows():
        group = tree_predictions[
            tree_predictions["model_key"].eq(threshold_row["model_key"])
            & tree_predictions["calibration_method"].eq(threshold_row["calibration_method"])
        ].copy()
        if group.empty:
            continue

        y_true = group["Target_AtRisk_Status"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "approach": "tree",
                "model_family": threshold_row["model_name"],
                "model_key": threshold_row["model_key"],
                "model_name": threshold_row["model_name"],
                "horizon_years": int(threshold_row["horizon_years"]),
                "history_years": int(threshold_row["history_years"]),
                "calibration_method": threshold_row["calibration_method"],
                "threshold_strategy": threshold_row["threshold_strategy"],
                "threshold_source": "phase4_calibration",
                "rows": float(len(group)),
                "positives": float(y_true.sum()),
                "positive_rate": float(y_true.mean()),
                **ranking_metrics(y_true, probability),
                **classification_metrics(y_true, probability, float(threshold_row["threshold"])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_non_tree_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in predictions[predictions["approach"].ne("tree")].groupby(
        ["model_key", "calibration_method"],
        sort=True,
    ):
        y_true = group["Target_AtRisk_Status"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        rank = ranking_metrics(y_true, probability)
        base = {
            "approach": group["approach"].iloc[0],
            "model_family": group["model_family"].iloc[0],
            "model_key": group["model_key"].iloc[0],
            "model_name": group["model_name"].iloc[0],
            "horizon_years": int(group["horizon_years"].iloc[0]),
            "history_years": group["history_years"].iloc[0],
            "calibration_method": group["calibration_method"].iloc[0],
            **rank,
        }

        thresholds: dict[str, tuple[float, str]] = {}
        train_threshold = group["train_positive_threshold"].iloc[0]
        if pd.notna(train_threshold):
            thresholds["train_positive_rate"] = (float(train_threshold), "train_positive_rate")
        thresholds.update(capacity_thresholds(probability))
        thresholds.update(diagnostic_thresholds(y_true, probability))

        for strategy, (threshold, source) in thresholds.items():
            rows.append(
                {
                    **base,
                    "threshold_strategy": strategy,
                    "threshold_source": source,
                    **classification_metrics(y_true, probability, threshold),
                }
            )
    return pd.DataFrame(rows)


def ranking_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in predictions.groupby(["model_key", "calibration_method"], sort=True):
        y_true = group["Target_AtRisk_Status"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "approach": group["approach"].iloc[0],
                "model_family": group["model_family"].iloc[0],
                "model_key": group["model_key"].iloc[0],
                "model_name": group["model_name"].iloc[0],
                "horizon_years": int(group["horizon_years"].iloc[0]),
                "history_years": group["history_years"].iloc[0],
                "calibration_method": group["calibration_method"].iloc[0],
                **ranking_metrics(y_true, probability),
            }
        )
    return pd.DataFrame(rows)


def best_by_horizon(ranking: pd.DataFrame) -> pd.DataFrame:
    return (
        ranking.sort_values(["horizon_years", "pr_auc", "roc_auc", "brier"], ascending=[True, False, False, True])
        .groupby("horizon_years", as_index=False)
        .head(3)
    )


def best_valid_threshold_rows(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    valid_sources = {"train_positive_rate", "phase4_calibration"}
    valid = threshold_metrics[threshold_metrics["threshold_source"].isin(valid_sources)].copy()
    return (
        valid.sort_values(["horizon_years", "pr_auc", "recall", "precision"], ascending=[True, False, False, False])
        .groupby("horizon_years", as_index=False)
        .head(5)
    )


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


def write_report(ranking: pd.DataFrame, threshold_metrics: pd.DataFrame, cohort_summary: pd.DataFrame) -> str:
    rank_cols = [
        "horizon_years",
        "approach",
        "model_key",
        "calibration_method",
        "rows",
        "positives",
        "positive_rate",
        "pr_auc",
        "roc_auc",
        "brier",
    ]
    threshold_cols = [
        "horizon_years",
        "approach",
        "model_key",
        "calibration_method",
        "threshold_strategy",
        "threshold_source",
        "pr_auc",
        "threshold",
        "flagged_rate",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    cohort_cols = [
        "horizon_years",
        "model_instances",
        "rows_before_alignment_min",
        "rows_before_alignment_max",
        "shared_rows",
        "shared_positives",
        "shared_positive_rate",
    ]
    cross_family = ranking.sort_values(["horizon_years", "pr_auc"], ascending=[True, False])
    diagnostic = threshold_metrics[threshold_metrics["threshold_source"].eq("test_oracle")].sort_values(
        ["horizon_years", "pr_auc", "recall"], ascending=[True, False, False]
    )
    distribution_only = threshold_metrics[threshold_metrics["threshold_source"].eq("test_distribution")].sort_values(
        ["horizon_years", "pr_auc", "recall"], ascending=[True, False, False]
    )
    lines = [
        "# Phase 4.2 v2 Cross-Family Comparison Report",
        "",
        "## Scope",
        "Compares v2 tree finalists (Phase 4 v2), v2 landmark Cox (Phase 3.2 v2), "
        "v2 two-stage survival (Phase 3.3 v2), and Phase 1 statistical grids. Trees use the full v2 feature set; "
        "landmark Cox and two-stage survival use the v2 temporal and interaction features but still exclude "
        "FBS hinge terms from the final Cox fit because they are singular on the non-at-risk landmark cohort. "
        "Both GEE v1/v2 and Logistic v2 grids are included when present. "
        "GLMM is not included because no stable GLMM prediction output has been generated.",
        "",
        "## Shared Cohort Alignment",
        "All ranking and threshold metrics below are recalculated on the intersection of "
        "(`PatientId`, `Year`, `target_year`) rows available for every compared model within each horizon.",
        markdown_table(cohort_summary[cohort_cols]),
        "",
        "## Best Ranking Metrics Per Horizon",
        markdown_table(best_by_horizon(ranking)[rank_cols]),
        "",
        "## Cross-Family Ranking",
        "Trees cover calibrated `M=1/M=3/M=5` candidates, GEE covers available `M=1/M=3/M=5` v1/v2 grids, "
        "Logistic v2 covers `M=1/M=3/M=5`, two-stage survival now covers `M=1/M=3/M=5`, "
        "and landmark Cox covers `N=1..5` as a rolling survival comparator without an explicit history-window grid.",
        markdown_table(cross_family[rank_cols]),
        "",
        "## Best Deployment-Valid Threshold Rows",
        "`phase4_calibration` and `train_positive_rate` thresholds do not use test labels for threshold optimization.",
        markdown_table(best_valid_threshold_rows(threshold_metrics)[threshold_cols], max_rows=25),
        "",
        "## Diagnostic Test-Distribution Threshold Rows",
        "These rows use test-score quantiles only. They avoid test labels but still adapt to the evaluation cohort, so they are not deployment-valid.",
        markdown_table(distribution_only[threshold_cols], max_rows=20),
        "",
        "## Diagnostic Test-Oracle Threshold Rows",
        "These rows show each non-tree model's best possible threshold behavior on the test set. They are useful for diagnosis but optimistic for deployment.",
        markdown_table(diagnostic[threshold_cols], max_rows=20),
        "",
        "## Conclusions",
        "- Cross-family ranking is now cohort-aligned by horizon; rows/events are directly comparable across approaches.",
        "- Tree thresholds are evaluated on the same shared rows as GEE and both survival families, while preserving the original Phase 4 calibration-selected cutoffs.",
        "- Non-tree test-quantile thresholds are retained only as diagnostics, not as deployment-valid recommendations.",
        "- GLMM should be treated as pending until it has stable predictions.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    PHASE4_OUT.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(
        load_phase4_trees()
        + load_phase1_gee()
        + load_phase1_logistic_v2()
        + load_phase3_2_landmark_cox()
        + load_phase3_3_two_stage(),
        ignore_index=True,
    )
    validate_target_alignment(predictions)
    aligned_predictions, cohort_summary = align_shared_cohort(predictions)
    ranking = ranking_table(aligned_predictions)

    tree_thresholds = evaluate_tree_thresholds(aligned_predictions)
    non_tree_thresholds = evaluate_non_tree_thresholds(aligned_predictions)
    threshold_metrics_df = pd.concat([tree_thresholds, non_tree_thresholds], ignore_index=True)

    ranking.to_csv(PHASE4_OUT / "phase_4_2_v2_cross_family_ranking.csv", index=False)
    threshold_metrics_df.to_csv(PHASE4_OUT / "phase_4_2_v2_cross_family_threshold_metrics.csv", index=False)
    best_by_horizon(ranking).to_csv(PHASE4_OUT / "phase_4_2_v2_best_by_horizon.csv", index=False)
    best_valid_threshold_rows(threshold_metrics_df).to_csv(
        PHASE4_OUT / "phase_4_2_v2_best_valid_threshold_rows.csv",
        index=False,
    )
    cohort_summary.to_csv(PHASE4_OUT / "phase_4_2_v2_shared_cohort_summary.csv", index=False)

    report = write_report(ranking, threshold_metrics_df, cohort_summary)
    (PHASE4_OUT / "phase_4_2_v2_cross_family_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
