"""Phase 7 — Logistic-only calendar-year ablation.

This script supports the BHI service paper by comparing the interpretable
Logistic Regression baseline with and without calendar-year features. It avoids
tree-model dependencies and uses only the phase-0 table construction, the
canonical patient split rule, and a small ridge-logistic Newton solver.

Run from the submodule root:
    python digihealth_risk/phase_7/logistic_year_ablation.py --history-years 5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_0.build_modeling_tables import (  # noqa: E402
    build_long_table,
    build_modeling_table,
    install_numpy_pickle_compat,
)


OUT_DIR = ROOT / "digihealth_risk" / "phase_7" / "outputs"
RANDOM_SEED = 20260501
RIDGE_ALPHA = 0.01

CONTINUOUS_FEATURES = [
    "Year_centered",
    "Year_centered_sq",
    "Age",
    "FBS",
    "FBS_hinge_100",
    "FBS_hinge_125",
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    "MAX_FBS_up_to_year",
    "years_since_last_fbs",
    "clinical_observed_count",
    "FBS_x_Age",
    "BMI_x_Age",
    "MAX_FBS_x_Age",
]

QUESTIONNAIRE_NUMERIC = [
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "dm_first_degree_relative",
    "cooking_method",
    "sleep_quality",
    "smoking_status",
    "alcohol_status",
]

MISSING_INDICATOR_FEATURES = [
    "FBS",
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    "years_since_last_fbs",
]

HISTORY_FEATURE_PATTERNS = ("_mean", "_slope")
YEAR_FEATURES = {"Year_centered", "Year_centered_sq"}


@dataclass
class Preprocessor:
    continuous_features: list[str]
    continuous_medians: pd.Series
    continuous_means: pd.Series
    continuous_stds: pd.Series
    dummy_columns: list[str]
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    default_data_path = ROOT / "datasets" / "df_final.pkl"
    parent_data_path = ROOT.parent / "datasets" / "df_final.pkl"
    if not default_data_path.exists() and parent_data_path.exists():
        default_data_path = parent_data_path

    parser = argparse.ArgumentParser(description="Run logistic-only year-feature ablation.")
    parser.add_argument("--data-path", type=Path, default=default_data_path)
    parser.add_argument("--history-years", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-7)
    return parser.parse_args()


def load_source(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing source data: {data_path}")
    install_numpy_pickle_compat()
    return pd.read_pickle(data_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Year_centered"] = out["Year"] - out["Year"].min()
    out["Year_centered_sq"] = out["Year_centered"] ** 2
    out["FBS_hinge_100"] = (out["FBS"] - 100).clip(lower=0)
    out["FBS_hinge_125"] = (out["FBS"] - 125).clip(lower=0)
    out["FBS_x_Age"] = out["FBS"] * out["Age"]
    out["BMI_x_Age"] = out["BMI"] * out["Age"]
    out["MAX_FBS_x_Age"] = out["MAX_FBS_up_to_year"] * out["Age"]
    return out


def split_by_patient(df: pd.DataFrame, source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patients = np.asarray(
        sorted(source_df["PatientId"].astype(str).drop_duplicates().tolist()),
        dtype=object,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    test_size = int(round(len(patients) * 0.20))
    test_patients = set(rng.choice(patients, size=test_size, replace=False))

    mask = df["PatientId"].astype(str).isin(test_patients)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def get_continuous_features(df: pd.DataFrame, *, include_year: bool) -> list[str]:
    base = CONTINUOUS_FEATURES if include_year else [f for f in CONTINUOUS_FEATURES if f not in YEAR_FEATURES]
    history_features = [
        col for col in df.columns
        if "_hist_" in col
        and col.endswith(HISTORY_FEATURE_PATTERNS)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return base + QUESTIONNAIRE_NUMERIC + sorted(history_features)


def fit_preprocessor(train_df: pd.DataFrame, *, include_year: bool) -> Preprocessor:
    continuous = get_continuous_features(train_df, include_year=include_year)
    medians = train_df[continuous].median(numeric_only=True)
    filled = train_df[continuous].fillna(medians)
    means = filled.mean()
    stds = filled.std(ddof=0).replace(0, 1)

    cat = train_df[CATEGORICAL_FEATURES].astype("object").fillna("missing")
    dummies = pd.get_dummies(cat, prefix=CATEGORICAL_FEATURES, drop_first=True, dtype=float)

    missing_names = [f"{f}_missing" for f in MISSING_INDICATOR_FEATURES]
    feature_names = (
        ["intercept"]
        + continuous
        + missing_names
        + ["has_fbs_this_year", "is_missing_last_year"]
        + dummies.columns.tolist()
    )
    return Preprocessor(continuous, medians, means, stds, dummies.columns.tolist(), feature_names)


def transform(df: pd.DataFrame, preprocessor: Preprocessor) -> np.ndarray:
    x_cont = df[preprocessor.continuous_features].fillna(preprocessor.continuous_medians)
    x_cont = (x_cont - preprocessor.continuous_means) / preprocessor.continuous_stds

    missing = pd.DataFrame(
        {f"{f}_missing": df[f].isna().astype(float) for f in MISSING_INDICATOR_FEATURES},
        index=df.index,
    )
    binary = pd.DataFrame(
        {
            "has_fbs_this_year": df["has_fbs_this_year"].astype(float),
            "is_missing_last_year": df["is_missing_last_year"].fillna(False).astype(float),
        },
        index=df.index,
    )
    cat = df[CATEGORICAL_FEATURES].astype("object").fillna("missing")
    dummies = pd.get_dummies(cat, prefix=CATEGORICAL_FEATURES, drop_first=True, dtype=float)
    dummies = dummies.reindex(columns=preprocessor.dummy_columns, fill_value=0.0)

    design = pd.concat([x_cont, missing, binary, dummies], axis=1)
    intercept = np.ones((len(design), 1), dtype=float)
    return np.hstack([intercept, design.to_numpy(dtype=float)])


def sigmoid(eta: np.ndarray) -> np.ndarray:
    return np.where(eta >= 0, 1 / (1 + np.exp(-eta)), np.exp(eta) / (1 + np.exp(eta)))


def objective(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    eta = x @ beta
    penalty_beta = beta.copy()
    penalty_beta[0] = 0.0
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * RIDGE_ALPHA * penalty_beta @ penalty_beta)


def fit_logistic(x: np.ndarray, y: np.ndarray, *, max_iter: int, tol: float) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    ridge = np.eye(x.shape[1]) * RIDGE_ALPHA
    ridge[0, 0] = 0.0
    current = objective(beta, x, y)

    for _ in range(max_iter):
        p = sigmoid(x @ beta)
        gradient = x.T @ (p - y) + ridge @ beta
        weights = np.clip(p * (1 - p), 1e-8, None)
        hessian = (x.T * weights) @ x + ridge
        step = np.linalg.solve(hessian + np.eye(hessian.shape[0]) * 1e-8, gradient)

        step_scale = 1.0
        accepted = False
        while step_scale >= 1e-4:
            candidate = beta - step_scale * step
            candidate_obj = objective(candidate, x, y)
            if candidate_obj <= current:
                beta = candidate
                improvement = current - candidate_obj
                current = candidate_obj
                accepted = True
                break
            step_scale *= 0.5

        if not accepted or improvement < tol * max(1.0, abs(current)):
            break
    return beta


def auc_roc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_pr(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    total_pos = (y_true == 1).sum()
    if total_pos == 0:
        return float("nan")
    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(precision, recall))


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    y_bool = y_true == 1
    tp = int((prediction & y_bool).sum())
    fp = int((prediction & ~y_bool).sum())
    tn = int((~prediction & ~y_bool).sum())
    fn = int((~prediction & y_bool).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": float(len(y_true)),
        "positives": float(y_bool.sum()),
        "positive_rate": float(y_bool.mean()),
        "threshold": float(threshold),
        "roc_auc": auc_roc(y_true, probability),
        "pr_auc": auc_pr(y_true, probability),
        "brier": float(np.mean((probability - y_true) ** 2)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def run_variant(
    model_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    horizon: int,
    history_years: int,
    variant: str,
    include_year: bool,
    max_iter: int,
    tol: float,
) -> list[dict[str, float | str | int]]:
    train_df, test_df = split_by_patient(model_df, source_df)
    preprocessor = fit_preprocessor(train_df, include_year=include_year)
    x_train = transform(train_df, preprocessor)
    x_test = transform(test_df, preprocessor)
    y_train = train_df["Target_AtRisk_Status"].to_numpy(dtype=float)
    y_test = test_df["Target_AtRisk_Status"].to_numpy(dtype=float)

    beta = fit_logistic(x_train, y_train, max_iter=max_iter, tol=tol)
    train_probability = sigmoid(x_train @ beta)
    test_probability = sigmoid(x_test @ beta)
    threshold = float(y_train.mean())

    rows: list[dict[str, float | str | int]] = []
    for split, y, probability in (
        ("train", y_train, train_probability),
        ("test", y_test, test_probability),
    ):
        rows.append(
            {
                "variant": variant,
                "model_family": "logistic",
                "horizon_years": horizon,
                "history_years": history_years,
                "split": split,
                "feature_count": len(preprocessor.feature_names),
                **classification_metrics(y, probability, threshold),
            }
        )
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    columns = display.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def build_report(metrics: pd.DataFrame, comparison: pd.DataFrame, data_path: Path) -> str:
    return "\n".join(
        [
            "# Phase 7 — Logistic-Only Year Ablation",
            "",
            f"Input data: `{data_path}`",
            "",
            "Compares the same ridge Logistic Regression feature pipeline with and",
            "without calendar-year terms (`Year_centered`, `Year_centered_sq`).",
            "All results use the canonical patient-level test split and history",
            "window requested at runtime.",
            "",
            "## Test PR-AUC Summary",
            "",
            markdown_table(comparison),
            "",
            "## Full Metrics",
            "",
            markdown_table(metrics),
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_df = load_source(args.data_path)
    long_df = build_long_table(source_df)
    rows: list[dict[str, float | str | int]] = []

    for horizon in sorted(set(args.horizons)):
        print(f"[phase_7] building table N={horizon} M={args.history_years}")
        model_df = build_modeling_table(long_df, source_df, horizon, args.history_years)
        model_df = engineer_features(model_df)
        for variant, include_year in (("with_year", True), ("no_year", False)):
            print(f"[phase_7] fitting logistic {variant} N={horizon} M={args.history_years}")
            rows.extend(
                run_variant(
                    model_df,
                    source_df,
                    horizon=horizon,
                    history_years=args.history_years,
                    variant=variant,
                    include_year=include_year,
                    max_iter=args.max_iter,
                    tol=args.tol,
                )
            )

    metrics = pd.DataFrame(rows)
    test = metrics[metrics["split"].eq("test")].copy()
    comparison = (
        test.pivot_table(
            index=["horizon_years", "history_years"],
            columns="variant",
            values=["pr_auc", "roc_auc", "brier"],
            aggfunc="first",
        )
        .sort_index()
    )
    comparison.columns = [f"{metric}_{variant}" for metric, variant in comparison.columns]
    comparison = comparison.reset_index()
    for metric in ("pr_auc", "roc_auc", "brier"):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_no_year"] - comparison[f"{metric}_with_year"]

    stem = f"phase_7_logistic_year_ablation_history_{args.history_years}"
    metrics.to_csv(OUT_DIR / f"{stem}_metrics.csv", index=False)
    comparison.to_csv(OUT_DIR / f"{stem}_comparison.csv", index=False)
    report = build_report(metrics, comparison, args.data_path)
    (OUT_DIR / f"{stem}_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
