"""Phase 2.3 horizon/history grid experiment.

Builds Phase 0 modeling tables for multiple prediction horizons N and history
windows M, then trains the strongest Phase 2 tree candidates on each table.

Run from the repository root:
    python digihealth_risk/phase_2/horizon_history_grid.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_0.build_modeling_tables import (  # noqa: E402
    build_long_table,
    build_modeling_table,
    load_data,
    output_suffix,
    summarize_eda,
)
from digihealth_risk.phase_2.train_tree_models import (  # noqa: E402
    markdown_table,
    run_dataset,
)

PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
PHASE2_OUT = ROOT / "digihealth_risk" / "phase_2" / "outputs"

DEFAULT_HORIZONS = [1, 2, 3, 4, 5]
DEFAULT_HISTORIES = [1, 3, 5]
DEFAULT_MODELS = ["catboost", "xgboost"]
DEFAULT_OUTPUT_PREFIX = "phase_2_3_grid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run N/M horizon-history grid for Phase 2 finalist models.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=DEFAULT_HORIZONS,
        help="Prediction horizons N to evaluate.",
    )
    parser.add_argument(
        "--histories",
        nargs="+",
        type=int,
        default=DEFAULT_HISTORIES,
        help="Historical lookback windows M to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=["histgb", "random_forest", "xgboost", "lightgbm", "catboost"],
        help="Models to train for each N/M table.",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix under digihealth_risk/phase_2/outputs/.",
    )
    parser.add_argument(
        "--force-rebuild-tables",
        action="store_true",
        help="Regenerate Phase 0 tables even if matching files already exist.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use model class weights. Off by default to preserve probability calibration.",
    )
    return parser.parse_args()


def modeling_table_path(horizon_years: int, history_years: int) -> Path:
    suffix = output_suffix(horizon_years, history_years)
    return PHASE0_OUT / f"phase_0_modeling_table{suffix}.pkl"


def ensure_phase0_tables(
    horizons: list[int],
    histories: list[int],
    force_rebuild: bool,
) -> list[Path]:
    PHASE0_OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    long_df = build_long_table(df)
    long_df.to_pickle(PHASE0_OUT / "patient_year_long.pkl")

    paths: list[Path] = []
    for horizon in horizons:
        for history in histories:
            path = modeling_table_path(horizon, history)
            if path.exists() and not force_rebuild:
                paths.append(path)
                continue

            print(f"Building Phase 0 table N={horizon}, M={history}")
            model_df = build_modeling_table(long_df, df, horizon, history)
            suffix = output_suffix(horizon, history)
            model_df.to_pickle(path)
            model_df.head(1000).to_csv(
                PHASE0_OUT / f"phase_0_modeling_table_sample{suffix}.csv",
                index=False,
            )
            report = summarize_eda(df, long_df, model_df, horizon, history)
            (PHASE0_OUT / f"phase_0_eda_report{suffix}.md").write_text(report, encoding="utf-8")
            paths.append(path)

    return paths


def add_rank_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["rank_pr_auc_within_horizon"] = (
        ranked.sort_values(["horizon_years", "pr_auc", "roc_auc"], ascending=[True, False, False])
        .groupby("horizon_years")
        .cumcount()
        + 1
    )
    ranked["rank_pr_auc_within_horizon_history"] = (
        ranked.sort_values(
            ["horizon_years", "history_years", "pr_auc", "roc_auc"],
            ascending=[True, True, False, False],
        )
        .groupby(["horizon_years", "history_years"])
        .cumcount()
        + 1
    )
    return ranked


def summarize_best(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = metrics[metrics["split"].eq("test")].copy()
    test = add_rank_columns(test)

    sort_cols = ["horizon_years", "pr_auc", "roc_auc"]
    best_by_horizon = (
        test.sort_values(sort_cols, ascending=[True, False, False])
        .groupby("horizon_years", as_index=False)
        .head(1)
    )
    best_by_horizon_history = (
        test.sort_values(
            ["horizon_years", "history_years", "pr_auc", "roc_auc"],
            ascending=[True, True, False, False],
        )
        .groupby(["horizon_years", "history_years"], as_index=False)
        .head(1)
    )
    dataset_summary = (
        test.sort_values(["horizon_years", "history_years", "model"])
        .groupby(["horizon_years", "history_years"], as_index=False)
        .agg(rows=("rows", "first"), positives=("positives", "first"), positive_rate=("positive_rate", "first"))
    )
    return best_by_horizon, best_by_horizon_history, dataset_summary


def write_grid_report(
    metrics: pd.DataFrame,
    horizons: list[int],
    histories: list[int],
    models: list[str],
    use_class_weights: bool,
) -> str:
    test = metrics[metrics["split"].eq("test")].copy()
    test = add_rank_columns(test)
    best_by_horizon, best_by_horizon_history, dataset_summary = summarize_best(metrics)

    metric_cols = [
        "horizon_years",
        "history_years",
        "model",
        "rows",
        "positives",
        "positive_rate",
        "pr_auc",
        "roc_auc",
        "brier",
        "precision",
        "recall",
        "specificity",
        "f1",
    ]
    best_cols = [
        "horizon_years",
        "history_years",
        "model",
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "brier",
    ]

    lines = [
        "# Phase 2.3 Horizon/History Grid Report",
        "",
        "## Scope",
        "This experiment varies prediction horizon `N` and historical lookback `M` before final threshold selection.",
        f"Horizons evaluated: `{horizons}`.",
        f"History windows evaluated: `{histories}`.",
        f"Models evaluated: `{models}`.",
        f"Class weighting enabled: `{use_class_weights}`.",
        "",
        "## Interpretation",
        "`N` is the target horizon: source-year `T` predicts `AtRisk_{T+N}`. "
        "`M` is the number of historical years ending at `T` used for rolling clinical features.",
        "",
        "## Best Model Per Horizon",
        markdown_table(best_by_horizon[best_cols]),
        "",
        "## Best Model Per N/M Combination",
        markdown_table(best_by_horizon_history[best_cols]),
        "",
        "## Dataset Size By N/M",
        markdown_table(dataset_summary),
        "",
        "## Full Test Metrics",
        markdown_table(test.sort_values(["horizon_years", "history_years", "model"])[metric_cols]),
        "",
        "## Notes",
        "- Splits are grouped by `PatientId` with seed `20260501`, inherited from Phase 2.",
        "- Threshold metrics use the training positive rate as the cutoff.",
        "- PR-AUC remains the primary metric because positive cases are rare.",
        "- Use these results to choose which horizon/history setup should enter calibration and threshold selection.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    PHASE2_OUT.mkdir(parents=True, exist_ok=True)
    horizons = sorted(set(args.horizons))
    histories = sorted(set(args.histories))
    paths = ensure_phase0_tables(horizons, histories, args.force_rebuild_tables)

    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    all_importances: list[pd.DataFrame] = []

    for path in paths:
        metrics_df, predictions, importances = run_dataset(path, args.models, args.use_class_weights)
        all_metrics.append(metrics_df)
        all_predictions.extend(predictions)
        all_importances.extend(importances)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    importances = pd.concat(all_importances, ignore_index=True)

    test_metrics = add_rank_columns(metrics[metrics["split"].eq("test")])
    best_by_horizon, best_by_horizon_history, dataset_summary = summarize_best(metrics)

    metrics.to_csv(PHASE2_OUT / f"{args.output_prefix}_metrics.csv", index=False)
    test_metrics.to_csv(PHASE2_OUT / f"{args.output_prefix}_test_metrics_ranked.csv", index=False)
    predictions.to_csv(PHASE2_OUT / f"{args.output_prefix}_test_predictions.csv", index=False)
    importances.to_csv(PHASE2_OUT / f"{args.output_prefix}_feature_importance.csv", index=False)
    best_by_horizon.to_csv(PHASE2_OUT / f"{args.output_prefix}_best_by_horizon.csv", index=False)
    best_by_horizon_history.to_csv(
        PHASE2_OUT / f"{args.output_prefix}_best_by_horizon_history.csv",
        index=False,
    )
    dataset_summary.to_csv(PHASE2_OUT / f"{args.output_prefix}_dataset_summary.csv", index=False)

    report = write_grid_report(metrics, horizons, histories, args.models, args.use_class_weights)
    (PHASE2_OUT / f"{args.output_prefix}_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
