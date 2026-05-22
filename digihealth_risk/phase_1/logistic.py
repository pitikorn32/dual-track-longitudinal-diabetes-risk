"""Phase 1 v2 interpretable logistic model — improved feature engineering.

Changes from v1 (phase_1_interpretable_logistic.py) driven by Phase 0.2 EDA:
  - Removed `pulse_pressure` (VIF=225; = BL_pres1 − BL_pres2 by construction)
    and its missing indicator
  - Added FBS hockey-stick features at clinical thresholds (100 mg/dL pre-DM,
    125 mg/dL DM) to capture non-linear risk structure
  - Added `Year_centered_sq` for the non-linear temporal trend (Ljung-Box p=0.03)
  - Added `MAX_FBS_x_Age` interaction (MAX_FBS has highest cross-lag r=0.582)

Run from the repository root:
    python digihealth_risk/phase_1/logistic.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402

INPUT_PATH = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "phase_0_modeling_table.pkl"
OUT_DIR = ROOT / "digihealth_risk" / "phase_1" / "outputs"
RANDOM_SEED = 20260501
RIDGE_ALPHA = 0.01

CONTINUOUS_FEATURES = [
    "Year_centered",
    "Year_centered_sq",       # v2: non-linear time trend
    "Age",
    "FBS",
    "FBS_hinge_100",          # v2: hockey-stick at pre-DM threshold 100 mg/dL
    "FBS_hinge_125",          # v2: hockey-stick at DM threshold 125 mg/dL
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    # pulse_pressure removed: VIF=225, = BL_pres1 - BL_pres2 by construction
    "MAX_FBS_up_to_year",
    "years_since_last_fbs",
    "clinical_observed_count",
    "FBS_x_Age",
    "BMI_x_Age",
    "MAX_FBS_x_Age",          # v2: cumulative FBS exposure × age
]

CATEGORICAL_FEATURES = [
    "gender",
    "dm_first_degree_relative",
    "cooking_method",
    "sleep_quality",
    "smoking_status",
    "alcohol_status",
]

QUESTIONNAIRE_NUMERIC = [
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
]

MISSING_INDICATOR_FEATURES = [
    "FBS",
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    # pulse_pressure removed along with the feature
    "years_since_last_fbs",
]

HISTORY_FEATURE_PATTERNS = ("_mean", "_slope")


@dataclass
class Preprocessor:
    continuous_features: list[str]
    continuous_medians: pd.Series
    continuous_means: pd.Series
    continuous_stds: pd.Series
    dummy_columns: list[str]
    feature_names: list[str]


@dataclass
class FitResult:
    coefficients: np.ndarray
    covariance: np.ndarray
    train_probability: np.ndarray
    test_probability: np.ndarray
    feature_names: list[str]


def install_numpy_pickle_compat() -> None:
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Phase 1 v2 interpretable logistic model.")
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--output-prefix", default="phase_1_v2_logistic")
    return parser.parse_args()


def load_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing Phase 0 input: {input_path}")
    install_numpy_pickle_compat()
    df = pd.read_pickle(input_path).copy()
    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]

    df["Year_centered"] = df["Year"] - df["Year"].min()

    # v2 engineered features — NaN in FBS propagates naturally to hinge features
    df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)
    df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    df["Year_centered_sq"] = df["Year_centered"] ** 2
    df["FBS_x_Age"] = df["FBS"] * df["Age"]
    df["BMI_x_Age"] = df["BMI"] * df["Age"]
    df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df)


def get_continuous_features(df: pd.DataFrame) -> list[str]:
    history_features = [
        col for col in df.columns
        if "_hist_" in col
        and col.endswith(HISTORY_FEATURE_PATTERNS)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    return CONTINUOUS_FEATURES + QUESTIONNAIRE_NUMERIC + sorted(history_features)


def fit_preprocessor(train_df: pd.DataFrame) -> Preprocessor:
    continuous = get_continuous_features(train_df)
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


def negative_log_likelihood(
    beta: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[float, np.ndarray]:
    eta = x @ beta
    nll = np.sum(np.logaddexp(0, eta) - y * eta)
    probability = special.expit(eta)
    gradient = x.T @ (probability - y)

    beta_penalty = beta.copy()
    beta_penalty[0] = 0.0
    nll += 0.5 * RIDGE_ALPHA * np.dot(beta_penalty, beta_penalty)
    gradient += RIDGE_ALPHA * beta_penalty
    return float(nll), gradient


def fit_logistic(
    train_df: pd.DataFrame, test_df: pd.DataFrame, preprocessor: Preprocessor
) -> FitResult:
    x_train = transform(train_df, preprocessor)
    y_train = train_df["Target_AtRisk_Status"].to_numpy(dtype=float)
    x_test = transform(test_df, preprocessor)

    initial = np.zeros(x_train.shape[1], dtype=float)
    result = optimize.minimize(
        fun=lambda beta: negative_log_likelihood(beta, x_train, y_train),
        x0=initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Logistic optimization failed: {result.message}")

    beta = result.x
    train_probability = special.expit(x_train @ beta)
    test_probability = special.expit(x_test @ beta)
    covariance = cluster_robust_covariance(
        x_train, y_train, train_probability, train_df["PatientId"], beta
    )
    return FitResult(beta, covariance, train_probability, test_probability, preprocessor.feature_names)


def cluster_robust_covariance(
    x: np.ndarray,
    y: np.ndarray,
    probability: np.ndarray,
    patient_ids: pd.Series,
    beta: np.ndarray,
) -> np.ndarray:
    weights = probability * (1 - probability)
    ridge = np.eye(x.shape[1]) * RIDGE_ALPHA
    ridge[0, 0] = 0.0
    bread = np.linalg.pinv((x.T * weights) @ x + ridge)

    residual = y - probability
    score_df = pd.DataFrame(x * residual[:, None])
    score_df["PatientId"] = patient_ids.to_numpy()
    cluster_scores = (
        score_df.groupby("PatientId", sort=False).sum().drop(columns="PatientId", errors="ignore")
    )
    scores = cluster_scores.to_numpy(dtype=float)
    meat = scores.T @ scores

    n_clusters = patient_ids.nunique()
    n_obs, n_params = x.shape
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - n_params))
    return correction * bread @ meat @ bread


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


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
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
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def coefficients_table(fit: FitResult) -> pd.DataFrame:
    se = np.sqrt(np.clip(np.diag(fit.covariance), 0, np.inf))
    z = fit.coefficients / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    lower = fit.coefficients - 1.96 * se
    upper = fit.coefficients + 1.96 * se
    table = pd.DataFrame({
        "feature": fit.feature_names,
        "coefficient": fit.coefficients,
        "std_error_cluster": se,
        "odds_ratio": np.exp(fit.coefficients),
        "or_ci_low": np.exp(lower),
        "or_ci_high": np.exp(upper),
        "p_value": p,
    })
    table = table.replace([np.inf, -np.inf], np.nan)
    table["abs_log_or"] = np.abs(table["coefficient"])
    return table.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or")


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    quantiles = pd.qcut(probability, q=bins, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": probability, "bin": quantiles})
    return (
        df.groupby("bin", observed=True)
        .agg(rows=("y", "size"), mean_probability=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index(drop=True)
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


def write_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    coef_table: pd.DataFrame,
    cal_table: pd.DataFrame,
) -> str:
    report = [
        "# Phase 1 v2 Interpretable Logistic Risk Model",
        "",
        "## Scope",
        "Improved logistic model with patient-cluster robust standard errors. "
        "Feature improvements from Phase 0.2 EDA: removed `pulse_pressure` (VIF=225), "
        "added FBS hockey-stick features at clinical thresholds (100 mg/dL pre-DM, "
        "125 mg/dL DM), quadratic time trend, and MAX_FBS×Age interaction. "
        "When a Phase 0 horizon/history table is supplied, compact history mean/slope "
        "features are also included.",
        "",
        "## Data Split",
        f"Patients are split by `PatientId` with seed `{RANDOM_SEED}`: "
        f"`{train_df['PatientId'].nunique():,}` train patients and "
        f"`{test_df['PatientId'].nunique():,}` test patients.",
        "",
        "## Metrics",
    ]
    metrics_df = pd.DataFrame([
        {"split": "train", "calibration_method": "raw",
         "threshold_strategy": "train_positive_rate", **train_metrics},
        {"split": "test", "calibration_method": "raw",
         "threshold_strategy": "train_positive_rate", **test_metrics},
    ])
    report.append(markdown_table(metrics_df))
    report.extend([
        "",
        "## Most Influential Odds Ratios",
        "Continuous variables are median-imputed and standardized on the training set.",
    ])
    display_cols = [
        "feature", "coefficient", "std_error_cluster",
        "odds_ratio", "or_ci_low", "or_ci_high", "p_value",
    ]
    report.append(
        markdown_table(
            coef_table.loc[coef_table["feature"] != "intercept", display_cols], max_rows=30
        )
    )
    report.extend([
        "",
        "## Test Calibration Deciles",
        markdown_table(cal_table),
        "",
        "## v2 Feature Changes vs v1",
        "| Change | Feature | Reason |",
        "| --- | --- | --- |",
        "| Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 by construction |",
        "| Removed | `pulse_pressure_missing` | Indicator dropped with feature |",
        "| Added | `FBS_hinge_100` | Hockey-stick at pre-DM threshold (100 mg/dL) |",
        "| Added | `FBS_hinge_125` | Hockey-stick at DM threshold (125 mg/dL) |",
        "| Added | `Year_centered_sq` | Non-linear time trend (Ljung-Box p=0.03) |",
        "| Added | `MAX_FBS_x_Age` | Cumulative FBS exposure × age (cross-lag r=0.582) |",
        "",
    ])
    return "\n".join(report)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(args.input_path)
    train_df, test_df = split_by_patient(df)
    preprocessor = fit_preprocessor(train_df)
    fit = fit_logistic(train_df, test_df, preprocessor)

    y_train = train_df["Target_AtRisk_Status"].to_numpy(dtype=float)
    y_test = test_df["Target_AtRisk_Status"].to_numpy(dtype=float)
    threshold = float(y_train.mean())
    train_metrics = classification_metrics(y_train, fit.train_probability, threshold)
    test_metrics = classification_metrics(y_test, fit.test_probability, threshold)
    coef_table = coefficients_table(fit)
    cal_table = calibration_table(y_test, fit.test_probability)

    metrics_df = pd.DataFrame([
        {"split": "train", "calibration_method": "raw",
         "threshold_strategy": "train_positive_rate", **train_metrics},
        {"split": "test", "calibration_method": "raw",
         "threshold_strategy": "train_positive_rate", **test_metrics},
    ])

    predictions = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    predictions["predicted_probability"] = fit.test_probability

    coef_table.to_csv(OUT_DIR / f"{args.output_prefix}_coefficients.csv", index=False)
    metrics_df.to_csv(OUT_DIR / f"{args.output_prefix}_metrics.csv", index=False)
    predictions.to_csv(OUT_DIR / f"{args.output_prefix}_test_predictions.csv", index=False)
    cal_table.to_csv(OUT_DIR / f"{args.output_prefix}_test_calibration.csv", index=False)

    report = write_report(train_df, test_df, train_metrics, test_metrics, coef_table, cal_table)
    (OUT_DIR / f"{args.output_prefix}_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
