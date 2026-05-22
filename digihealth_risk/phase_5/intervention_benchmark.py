"""Phase 6.5 intervention-model comparison and recommendation summary.

This script consolidates the Phase 6 monotonic model family benchmarks into a
single intervention-focused leaderboard. It compares the monotonic variants of:
  - XGBoost
  - LightGBM
  - CatBoost
  - EBM
  - Logistic

Outputs:
  - prediction comparison table
  - safety comparison table
  - overall recommendation summary
  - markdown report

Run from the repository root:
    python digihealth_risk/phase_5/intervention_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "digihealth_risk" / "phase_5" / "outputs"
PHASE4_RANKING_PATH = ROOT / "digihealth_risk" / "phase_4" / "outputs" / "phase_4_2_v2_cross_family_ranking.csv"

MODEL_SPECS = [
    {
        "family": "xgboost",
        "display_name": "Monotonic XGBoost",
        "metrics_path": OUT_DIR / "phase_6_v2_ablation_metrics.csv",
        "safety_path": OUT_DIR / "phase_6_v2_ablation_safety_summary.csv",
        "interpretability_tier": 2,
    },
    {
        "family": "lightgbm",
        "display_name": "Monotonic LightGBM",
        "metrics_path": OUT_DIR / "phase_6_v2_lightgbm_ablation_metrics.csv",
        "safety_path": OUT_DIR / "phase_6_v2_lightgbm_ablation_safety_summary.csv",
        "interpretability_tier": 2,
    },
    {
        "family": "catboost",
        "display_name": "Monotonic CatBoost",
        "metrics_path": OUT_DIR / "phase_6_v2_catboost_ablation_metrics.csv",
        "safety_path": OUT_DIR / "phase_6_v2_catboost_ablation_safety_summary.csv",
        "interpretability_tier": 2,
    },
    {
        "family": "ebm",
        "display_name": "Monotonic EBM",
        "metrics_path": OUT_DIR / "phase_6_v2_ebm_ablation_metrics.csv",
        "safety_path": OUT_DIR / "phase_6_v2_ebm_ablation_safety_summary.csv",
        "interpretability_tier": 1,
    },
    {
        "family": "logistic",
        "display_name": "Monotonic Logistic",
        "metrics_path": OUT_DIR / "phase_6_v2_logistic_ablation_metrics.csv",
        "safety_path": OUT_DIR / "phase_6_v2_logistic_ablation_safety_summary.csv",
        "interpretability_tier": 0,
    },
]

PREDICTION_OUT = OUT_DIR / "phase_6_v2_intervention_model_prediction_comparison.csv"
SAFETY_OUT = OUT_DIR / "phase_6_v2_intervention_model_safety_comparison.csv"
SUMMARY_OUT = OUT_DIR / "phase_6_v2_intervention_model_summary.csv"
REPORT_OUT = OUT_DIR / "phase_6_v2_intervention_model_report.md"


def best_monotonic_rows(metrics_path: Path, family: str, display_name: str, interpretability_tier: int) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    df = df[(df["split"] == "test") & (df["variant"] == "monotonic")].copy()
    best = (
        df.sort_values(["horizon_years", "pr_auc", "roc_auc"], ascending=[True, False, False])
        .groupby("horizon_years", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["family"] = family
    best["display_name"] = display_name
    best["interpretability_tier"] = interpretability_tier
    return best


def best_monotonic_safety(safety_path: Path, family: str, display_name: str, interpretability_tier: int) -> pd.DataFrame:
    df = pd.read_csv(safety_path)
    df = df[df["variant"] == "monotonic"].copy()
    best = (
        df.sort_values(
            ["horizon_years", "directionally_correct_rate", "unexpected_increase_rate", "mean_delta_score"],
            ascending=[True, False, True, True],
        )
        .groupby("horizon_years", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best["family"] = family
    best["display_name"] = display_name
    best["interpretability_tier"] = interpretability_tier
    return best


def load_phase4_leaders() -> pd.DataFrame:
    ranking = pd.read_csv(PHASE4_RANKING_PATH)
    return (
        ranking.sort_values(["horizon_years", "pr_auc", "roc_auc"], ascending=[True, False, False])
        .groupby("horizon_years", as_index=False)
        .head(1)
        .reset_index(drop=True)
        .rename(
            columns={
                "model_key": "pure_leader_model_key",
                "model_name": "pure_leader_model_name",
                "history_years": "pure_leader_history_years",
                "pr_auc": "pure_leader_pr_auc",
                "roc_auc": "pure_leader_roc_auc",
                "brier": "pure_leader_brier",
            }
        )
    )


def markdown_table(df: pd.DataFrame) -> str:
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


def build_comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    safety_parts = []
    for spec in MODEL_SPECS:
        prediction_parts.append(
            best_monotonic_rows(
                spec["metrics_path"],
                spec["family"],
                spec["display_name"],
                spec["interpretability_tier"],
            )
        )
        safety_parts.append(
            best_monotonic_safety(
                spec["safety_path"],
                spec["family"],
                spec["display_name"],
                spec["interpretability_tier"],
            )
        )

    prediction = pd.concat(prediction_parts, ignore_index=True)
    safety = pd.concat(safety_parts, ignore_index=True)
    pure_leaders = load_phase4_leaders()

    prediction = prediction.merge(
        pure_leaders[
            [
                "horizon_years",
                "pure_leader_model_key",
                "pure_leader_model_name",
                "pure_leader_history_years",
                "pure_leader_pr_auc",
                "pure_leader_roc_auc",
                "pure_leader_brier",
            ]
        ],
        on="horizon_years",
        how="left",
    )
    prediction["pr_auc_gap_vs_pure_leader"] = prediction["pr_auc"] - prediction["pure_leader_pr_auc"]
    prediction["roc_auc_gap_vs_pure_leader"] = prediction["roc_auc"] - prediction["pure_leader_roc_auc"]
    prediction["prediction_rank"] = (
        prediction.groupby("horizon_years")["pr_auc"].rank(method="dense", ascending=False).astype(int)
    )

    safety = safety.rename(
        columns={
            "history_years": "safety_history_years",
            "directionally_correct_rate": "safety_directionally_correct_rate",
            "unexpected_increase_rate": "safety_unexpected_increase_rate",
            "mean_delta_score": "safety_mean_delta_score",
            "worst_positive_delta_score": "safety_worst_positive_delta_score",
        }
    )
    safety["safety_rank"] = (
        safety.sort_values(
            ["horizon_years", "safety_unexpected_increase_rate", "safety_mean_delta_score"],
            ascending=[True, True, True],
        )
        .groupby("horizon_years")
        .cumcount()
        .add(1)
    )

    merged = prediction.merge(
        safety[
            [
                "family",
                "horizon_years",
                "safety_history_years",
                "scenario_count",
                "safety_directionally_correct_rate",
                "safety_unexpected_increase_rate",
                "safety_mean_delta_score",
                "safety_worst_positive_delta_score",
                "safety_rank",
            ]
        ],
        on=["family", "horizon_years"],
        how="left",
    )

    summary_rows = []
    for horizon, group in merged.groupby("horizon_years"):
        safe_group = group[group["safety_unexpected_increase_rate"] <= 1e-12].copy()
        best_intervention = safe_group.sort_values(
            ["pr_auc", "roc_auc", "interpretability_tier"],
            ascending=[False, False, True],
        ).iloc[0]
        interpretable_group = safe_group[safe_group["family"].isin(["ebm", "logistic"])].copy()
        best_interpretable = interpretable_group.sort_values(
            ["pr_auc", "roc_auc", "interpretability_tier"],
            ascending=[False, False, True],
        ).iloc[0]
        summary_rows.append(
            {
                "horizon_years": int(horizon),
                "best_pure_predictor": best_intervention["pure_leader_model_key"],
                "best_pure_predictor_pr_auc": float(best_intervention["pure_leader_pr_auc"]),
                "best_intervention_model": best_intervention["display_name"],
                "best_intervention_family": best_intervention["family"],
                "best_intervention_history_years": int(best_intervention["history_years"]),
                "best_intervention_pr_auc": float(best_intervention["pr_auc"]),
                "intervention_pr_auc_gap_vs_pure": float(best_intervention["pr_auc_gap_vs_pure_leader"]),
                "best_interpretable_model": best_interpretable["display_name"],
                "best_interpretable_family": best_interpretable["family"],
                "best_interpretable_history_years": int(best_interpretable["history_years"]),
                "best_interpretable_pr_auc": float(best_interpretable["pr_auc"]),
                "best_interpretable_gap_vs_pure": float(best_interpretable["pr_auc_gap_vs_pure_leader"]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("horizon_years").reset_index(drop=True)
    return merged.sort_values(["horizon_years", "prediction_rank", "display_name"]).reset_index(drop=True), safety.sort_values(["horizon_years", "safety_rank", "display_name"]).reset_index(drop=True), summary


def overall_winner_text(prediction: pd.DataFrame) -> tuple[str, str]:
    safe = prediction[prediction["safety_unexpected_increase_rate"] <= 1e-12].copy()
    overall = (
        safe.groupby(["family", "display_name"], as_index=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            min_pr_auc=("pr_auc", "min"),
            horizons_won=("prediction_rank", lambda s: int((s == 1).sum())),
            mean_gap_vs_pure=("pr_auc_gap_vs_pure_leader", "mean"),
        )
        .sort_values(["horizons_won", "mean_pr_auc", "min_pr_auc"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    best_overall = overall.iloc[0]["display_name"]
    best_interpretable = (
        overall[overall["family"].isin(["ebm", "logistic"])]
        .sort_values(["horizons_won", "mean_pr_auc", "min_pr_auc"], ascending=[False, False, False])
        .iloc[0]["display_name"]
    )
    return str(best_overall), str(best_interpretable)


def horizon_winners_text(summary: pd.DataFrame) -> str:
    """Render summary['best_intervention_model'] as a per-horizon sentence."""
    grouped: dict[str, list[int]] = {}
    for _, row in summary.iterrows():
        grouped.setdefault(str(row["best_intervention_model"]), []).append(int(row["horizon_years"]))

    def fmt(horizons: list[int]) -> str:
        horizons = sorted(horizons)
        label = "year" if len(horizons) == 1 else "years"
        return f"{label} {', '.join(str(h) for h in horizons)}"

    items = sorted(grouped.items(), key=lambda kv: min(kv[1]))
    return "; ".join(f"{model} at {fmt(hs)}" for model, hs in items)


def write_report(prediction: pd.DataFrame, safety: pd.DataFrame, summary: pd.DataFrame) -> str:
    best_overall, best_interpretable = overall_winner_text(prediction)
    horizon_text = horizon_winners_text(summary)
    pred_cols = [
        "horizon_years",
        "display_name",
        "history_years",
        "pr_auc",
        "roc_auc",
        "recall",
        "prediction_rank",
        "pr_auc_gap_vs_pure_leader",
    ]
    safety_cols = [
        "horizon_years",
        "display_name",
        "safety_history_years",
        "safety_directionally_correct_rate",
        "safety_unexpected_increase_rate",
        "safety_mean_delta_score",
        "safety_worst_positive_delta_score",
        "safety_rank",
    ]
    summary_cols = [
        "horizon_years",
        "best_pure_predictor",
        "best_intervention_model",
        "best_intervention_pr_auc",
        "intervention_pr_auc_gap_vs_pure",
        "best_interpretable_model",
        "best_interpretable_pr_auc",
    ]
    lines = [
        "# Phase 6.5 Intervention-Model Comparison",
        "",
        "## Purpose",
        "Compares the completed intervention-ready model families on the same rolling benchmark. "
        "The main goal is to identify the best model for patient-facing what-if simulation, not just the best pure predictor.",
        "",
        "## Labels",
        f"- Best overall intervention-safe family: `{best_overall}`",
        f"- Best interpretable intervention family: `{best_interpretable}`",
        "- Best pure predictor remains the Phase 4.2 horizon-specific leaderboard and is included only as a reference line.",
        "",
        "## Prediction Comparison",
        markdown_table(prediction[pred_cols]),
        "",
        "## Safety Comparison",
        markdown_table(safety[safety_cols]),
        "",
        "## Horizon Recommendations",
        markdown_table(summary[summary_cols]),
        "",
        "## Recommendation",
        "Use the horizon-specific intervention winners when intervention behavior matters. "
        f"In the current results that means {horizon_text}. "
        f"If one single deployment family is preferred for simplicity, {best_overall} is the most balanced overall choice across horizons.",
        "",
        f"Use {best_interpretable} when explanation quality is the top priority and a small predictive tradeoff is acceptable. "
        "Keep Monotonic Logistic as the simplest transparent baseline, not the main deployment choice.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    prediction, safety, summary = build_comparison_tables()
    prediction.to_csv(PREDICTION_OUT, index=False)
    safety.to_csv(SAFETY_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    report = write_report(prediction, safety, summary)
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
