"""Phase 7 — Year-features ablation: compare baseline vs no-Year metrics.

Reads the existing phase 2 / phase 4 / phase 5 metric CSVs (baseline with Year
features) and the phase_7 no-Year metric CSVs, joins them on
(model_key / family / horizon / history / calibration / variant), and emits a
markdown report with PR-AUC / ROC-AUC / Brier deltas.

A negative delta means dropping Year features HURT performance.
A positive delta means dropping Year features HELPED (or made no difference).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "digihealth_risk" / "phase_7" / "outputs"

PHASE2_BASELINE = ROOT / "digihealth_risk" / "phase_2" / "outputs" / "phase_2_v2_metrics.csv"
PHASE4_BASELINE = ROOT / "digihealth_risk" / "phase_4" / "outputs" / "phase_4_v2_metrics.csv"
PHASE5_BASELINES: dict[str, Path] = {
    "xgboost": ROOT / "digihealth_risk" / "phase_5" / "outputs"
    / "phase_6_v2_monotonic_xgboost_metrics.csv",
    "catboost": ROOT / "digihealth_risk" / "phase_5" / "outputs"
    / "phase_6_v2_catboost_ablation_metrics.csv",
    "lightgbm": ROOT / "digihealth_risk" / "phase_5" / "outputs"
    / "phase_6_v2_lightgbm_ablation_metrics.csv",
    "ebm": ROOT / "digihealth_risk" / "phase_5" / "outputs"
    / "phase_6_v2_ebm_ablation_metrics.csv",
    "logistic": ROOT / "digihealth_risk" / "phase_5" / "outputs"
    / "phase_6_v2_logistic_ablation_metrics.csv",
}

NO_YEAR_TREES = OUT_DIR / "phase_7_no_year_trees_metrics.csv"
NO_YEAR_CAL = OUT_DIR / "phase_7_no_year_calibration_metrics.csv"
NO_YEAR_MONO = OUT_DIR / "phase_7_no_year_monotonic_metrics.csv"

METRIC_COLS = ("pr_auc", "roc_auc", "brier")


def _fmt(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.4f}" if isinstance(x, float) and (x > 0 or x < 0) else f"{x:.4f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_"
    display = df.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    cols = display.columns.tolist()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[phase_7] WARN missing: {path.relative_to(ROOT)}")
        return pd.DataFrame()
    return pd.read_csv(path)


# -----------------------------------------------------------------------------
# Comparison 1 — Phase 2 uncalibrated tree grid
# -----------------------------------------------------------------------------


def compare_trees() -> pd.DataFrame:
    base = _safe_read(PHASE2_BASELINE)
    new = _safe_read(NO_YEAR_TREES)
    if base.empty or new.empty:
        return pd.DataFrame()

    base = base[base["split"].eq("test")].copy()
    new = new[new["split"].eq("test")].copy()

    keys = ["model", "horizon_years", "history_years"]
    merged = base.merge(new, on=keys, suffixes=("_baseline", "_no_year"), how="inner")

    for metric in METRIC_COLS:
        merged[f"delta_{metric}"] = merged[f"{metric}_no_year"] - merged[f"{metric}_baseline"]

    cols = (
        keys
        + [c for metric in METRIC_COLS for c in (f"{metric}_baseline", f"{metric}_no_year", f"delta_{metric}")]
    )
    return merged[cols].sort_values(["horizon_years", "history_years", "model"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Comparison 2 — Phase 4 calibrated trees
# -----------------------------------------------------------------------------


def compare_calibration() -> pd.DataFrame:
    base = _safe_read(PHASE4_BASELINE)
    new = _safe_read(NO_YEAR_CAL)
    if base.empty or new.empty:
        return pd.DataFrame()

    # phase_4 metrics duplicate ranking rows per threshold_strategy — keep one row per
    # (model_key, calibration_method)
    base = base[base["threshold_strategy"].eq("train_positive_rate")].copy()
    new = new[new["threshold_strategy"].eq("train_positive_rate")].copy()

    keys = ["model_key", "model_name", "calibration_method", "horizon_years", "history_years"]
    merged = base.merge(new, on=keys, suffixes=("_baseline", "_no_year"), how="inner")

    for metric in METRIC_COLS:
        merged[f"delta_{metric}"] = merged[f"{metric}_no_year"] - merged[f"{metric}_baseline"]

    cols = (
        keys
        + [c for metric in METRIC_COLS for c in (f"{metric}_baseline", f"{metric}_no_year", f"delta_{metric}")]
    )
    return merged[cols].sort_values(["horizon_years", "model_name", "calibration_method"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Comparison 3 — Phase 5 monotonic
# -----------------------------------------------------------------------------


def _load_phase5_baseline() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for family, path in PHASE5_BASELINES.items():
        df = _safe_read(path)
        if df.empty:
            continue
        df = df[df["split"].eq("test")].copy()
        if "variant" in df.columns:
            df = df[df["variant"].eq("monotonic")]
        df["family"] = family
        parts.append(df[["family", "horizon_years", "history_years", "pr_auc", "roc_auc", "brier"]])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def compare_monotonic() -> pd.DataFrame:
    base = _load_phase5_baseline()
    new = _safe_read(NO_YEAR_MONO)
    if base.empty or new.empty:
        return pd.DataFrame()

    new = new[new["split"].eq("test")].copy()
    new = new[["family", "horizon_years", "history_years", "pr_auc", "roc_auc", "brier"]]

    keys = ["family", "horizon_years", "history_years"]
    merged = base.merge(new, on=keys, suffixes=("_baseline", "_no_year"), how="inner")

    for metric in METRIC_COLS:
        merged[f"delta_{metric}"] = merged[f"{metric}_no_year"] - merged[f"{metric}_baseline"]

    cols = (
        keys
        + [c for metric in METRIC_COLS for c in (f"{metric}_baseline", f"{metric}_no_year", f"delta_{metric}")]
    )
    return merged[cols].sort_values(["family", "horizon_years"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Aggregate summaries
# -----------------------------------------------------------------------------


def summarise(deltas: pd.DataFrame, label: str) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    rows = []
    for metric in METRIC_COLS:
        column = f"delta_{metric}"
        rows.append(
            {
                "comparison": label,
                "metric": metric,
                "rows_compared": int(len(deltas)),
                "mean_delta": float(deltas[column].mean()),
                "median_delta": float(deltas[column].median()),
                "min_delta": float(deltas[column].min()),
                "max_delta": float(deltas[column].max()),
                "n_better_or_equal": int((deltas[column] >= 0).sum())
                if metric in ("pr_auc", "roc_auc")
                else int((deltas[column] <= 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    trees_cmp: pd.DataFrame,
    cal_cmp: pd.DataFrame,
    mono_cmp: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    lines = [
        "# Phase 7 — Year-features Ablation Report",
        "",
        "## Scope",
        "",
        "Drops `Year`, `Year_centered`, and `Year_centered_sq` from training and",
        "compares against the existing baselines that retain those features. All",
        "other features (Age, FBS, BMI, lifestyle, history-window slopes, etc.)",
        "are unchanged.",
        "",
        "**Delta convention**: `delta_metric = no_year - baseline`.",
        "- `delta_pr_auc > 0` ⇒ dropping Year helped PR-AUC.",
        "- `delta_pr_auc < 0` ⇒ dropping Year hurt PR-AUC.",
        "- For Brier, the opposite (`< 0` is better).",
        "",
        "## Summary by comparison",
        "",
        _markdown_table(summary),
        "",
        "## Comparison 1 — Phase 2 uncalibrated tree grid (full N×M×model)",
        "",
        _markdown_table(trees_cmp),
        "",
        "## Comparison 2 — Phase 4 calibrated trees (train_positive_rate threshold)",
        "",
        _markdown_table(cal_cmp),
        "",
        "## Comparison 3 — Phase 5 monotonic models (deployed intervention-safe track)",
        "",
        _markdown_table(mono_cmp),
        "",
        "## How to interpret",
        "",
        "- If `delta_pr_auc` is consistently small (|Δ| < 0.005) across the grid,",
        "  the Year features were doing very little — dropping them is safe.",
        "- If deltas are systematically negative, the Year features were absorbing",
        "  in-sample temporal drift and removing them costs predictive power.",
        "- If the deployed (phase 5) track shows the smallest absolute deltas, the",
        "  ablation is a low-cost construct-validity fix for the deployment track.",
        "",
        "## Caveats",
        "",
        "- Same patient-level split (seed 20260501) is used across baseline and",
        "  ablation, so changes reflect feature effects, not split variance.",
        "- Baseline metrics were trained on 2005–2016 with Year features in scope.",
        "  Deployment in any year outside that range is OOD for those features",
        "  regardless of these metrics — see thesis §6 for the construct-validity",
        "  discussion.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trees_cmp = compare_trees()
    cal_cmp = compare_calibration()
    mono_cmp = compare_monotonic()

    summary = pd.concat(
        [
            summarise(trees_cmp, "phase2_trees"),
            summarise(cal_cmp, "phase4_calibrated_trees"),
            summarise(mono_cmp, "phase5_monotonic"),
        ],
        ignore_index=True,
    )

    trees_cmp.to_csv(OUT_DIR / "phase_7_compare_trees.csv", index=False)
    cal_cmp.to_csv(OUT_DIR / "phase_7_compare_calibration.csv", index=False)
    mono_cmp.to_csv(OUT_DIR / "phase_7_compare_monotonic.csv", index=False)
    summary.to_csv(OUT_DIR / "phase_7_compare_summary.csv", index=False)

    report = build_report(trees_cmp, cal_cmp, mono_cmp, summary)
    (OUT_DIR / "phase_7_year_ablation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
