"""Frequentist binomial GLMM via GPBoost.

Fits a binary GLMM with logit link and patient-level random intercept on the
rolling patient-year modeling table. This is the "standard GLMM" baseline
requested by the thesis defense (V5 reviewer feedback F7): the V5 main text
reports that the Bayesian VB GLMM from statsmodels did not converge, and the
reviewer asked for a non-Bayesian GLMM benchmark.

CLI is aligned with ``digihealth_risk/phase_1/gee.py`` and ``logistic.py``:
takes ``--input-path`` (a Phase 0 modeling table) and ``--output-prefix``
(used to name ``{prefix}_metrics.csv``, ``{prefix}_test_predictions.csv``,
``{prefix}_coefficients.csv``, and a richer ``{prefix}_metrics.json`` with the
subject-specific vs. marginal Pinheiro--Bates breakdown). Per-horizon and
per-history convenience arguments are accepted for direct invocation; when
``--input-path`` is provided it takes precedence.

Run from the repository root:
    python digihealth_risk/phase_1/glmm_gpboost.py \\
        --input-path digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_3_history_5.pkl \\
        --output-prefix phase_1_v2_glmm_gpboost_horizon_3_history_5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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

import gpboost as gpb


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.utils.patient_split import apply_canonical_split  # noqa: E402


PHASE_0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
OUT_DIR = ROOT / "digihealth_risk" / "phase_1" / "outputs"

CONTINUOUS_FEATURES = [
    "Year_centered",
    "Age",
    "FBS",
    "BMI",
    "Pulse",
    "BL_pres1",
    "Waist",
    "MAX_FBS_up_to_year",
    "years_since_last_fbs",
    "clinical_observed_count",
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
]

# Engineered features. The two FBS hinge terms are excluded because the
# first-onset censoring rule restricts the modeling table to non-at-risk
# rows, which forces source-year FBS <= 100, making both hinge terms
# identically zero. This is the same exclusion the thesis applies to the
# survival family; the rank-deficient design that results breaks the
# Laplace approximation in GPBoost.
ENGINEERED_FEATURES = [
    "FBS_x_Age",
    "MAX_FBS_x_Age",
    "Year_centered_sq",
]

CATEGORICAL_FEATURES = [
    "gender",
    "dm_first_degree_relative",
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

# Rolling-history summary suffixes pulled from the modeling table when M > 1.
# Matches the GEE / logistic convention (Section 3.4.3 of the V5 main text:
# "GEE and logistic used only the _mean and _slope summaries").
HISTORY_FEATURE_SUFFIXES = ("_mean", "_slope")


def collect_history_features(df: pd.DataFrame) -> list[str]:
    """Return rolling-history columns matching the GEE/logistic convention."""
    return sorted(
        col
        for col in df.columns
        if "_hist_" in col
        and col.endswith(HISTORY_FEATURE_SUFFIXES)
        and pd.api.types.is_numeric_dtype(df[col])
    )


def install_numpy_pickle_compat() -> None:
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def modeling_table_path(horizon: int, history: int) -> Path:
    if horizon == 1 and history == 1:
        return PHASE_0_OUT / "phase_0_modeling_table.pkl"
    return PHASE_0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{history}.pkl"


def parse_horizon_history(input_path: Path) -> tuple[int, int]:
    """Parse (N, M) from a Phase 0 modeling table filename.

    Returns ``(1, 1)`` if the filename has no ``horizon_{N}_history_{M}`` suffix.
    """
    match = re.search(r"horizon_(\d+)_history_(\d+)", input_path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 1, 1


def load_modeling_table(input_path: Path) -> pd.DataFrame:
    install_numpy_pickle_compat()
    if not input_path.exists():
        raise FileNotFoundError(f"Modeling table missing: {input_path}")
    df = pd.read_pickle(input_path).copy()
    if "Target_AtRisk_Status" not in df.columns:
        df["Target_AtRisk_Status"] = df["Next_Year_AtRisk_Status"]
    df["Year_centered"] = df["Year"] - df["Year"].min()
    # Engineered features per Section 3.4.2
    df["Year_centered_sq"] = df["Year_centered"] ** 2
    df["FBS_x_Age"] = df["FBS"] * df["Age"]
    df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train = train_df.copy()
    test = test_df.copy()

    feature_names: list[str] = []
    X_train_cols: list[np.ndarray] = []
    X_test_cols: list[np.ndarray] = []

    def add(name: str, train_col: np.ndarray, test_col: np.ndarray) -> None:
        feature_names.append(name)
        X_train_cols.append(np.asarray(train_col, dtype=float))
        X_test_cols.append(np.asarray(test_col, dtype=float))

    history_features = collect_history_features(train)
    continuous_set = [f for f in CONTINUOUS_FEATURES + ENGINEERED_FEATURES if f in train.columns] + history_features
    for feat in continuous_set:
        median = train[feat].median()
        if pd.isna(median):
            median = 0.0
        train_filled = train[feat].fillna(median)
        mean = train_filled.mean()
        std = train_filled.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            std = 1.0
        add(f"{feat}_z", (train_filled - mean) / std, (test[feat].fillna(median) - mean) / std)

    for feat in MISSING_INDICATOR_FEATURES:
        add(f"{feat}_missing", train[feat].isna().astype(int).to_numpy(), test[feat].isna().astype(int).to_numpy())

    add(
        "is_missing_last_year",
        train["is_missing_last_year"].fillna(False).astype(int).to_numpy(),
        test["is_missing_last_year"].fillna(False).astype(int).to_numpy(),
    )
    add("has_fbs_this_year", train["has_fbs_this_year"].astype(int).to_numpy(), test["has_fbs_this_year"].astype(int).to_numpy())

    for feat in CATEGORICAL_FEATURES:
        train_vals = train[feat].astype("object").fillna("missing").astype(str)
        test_vals = test[feat].astype("object").fillna("missing").astype(str)
        levels = sorted(train_vals.unique().tolist())
        for level in levels[1:]:
            add(f"{feat}={level}", (train_vals == level).astype(int).to_numpy(), (test_vals == level).astype(int).to_numpy())

    X_train = np.column_stack(X_train_cols)
    X_test = np.column_stack(X_test_cols)
    return X_train, X_test, feature_names


def fit_glmm(X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray) -> gpb.GPModel:
    gp_model = gpb.GPModel(
        group_data=groups_train.reshape(-1, 1),
        likelihood="bernoulli_logit",
    )
    gp_model.fit(
        y=y_train,
        X=X_train,
        params={
            "trace": False,
            "maxit": 1000,
        },
    )
    return gp_model


def predict_probabilities(
    gp_model: gpb.GPModel,
    X_test: np.ndarray,
    groups_test: np.ndarray,
    sigma2_b: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (subject-specific p, marginal Pinheiro--Bates p)."""
    pred = gp_model.predict(
        X_pred=X_test,
        group_data_pred=groups_test.reshape(-1, 1),
        predict_var=False,
        predict_response=False,
    )
    linear_pred = np.asarray(pred["mu"]).ravel()
    p_subject = 1.0 / (1.0 + np.exp(-linear_pred))
    if sigma2_b is None or sigma2_b <= 0:
        return p_subject, p_subject
    scale = np.sqrt(1.0 + 0.346 * sigma2_b)
    p_marginal = 1.0 / (1.0 + np.exp(-linear_pred / scale))
    return p_subject, p_marginal


def classification_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probability >= threshold
    return {
        "rows": float(len(y_true)),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(((~pred) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def extract_sigma2(gp_model: gpb.GPModel) -> tuple[float | None, list[dict] | dict]:
    try:
        cov_pars = gp_model.get_cov_pars(std_err=False, format_pandas=True)
        if isinstance(cov_pars, pd.DataFrame):
            records = cov_pars.to_dict(orient="records")
            if "Param." in cov_pars.columns:
                vals = cov_pars["Param."].to_numpy()
            else:
                vals = cov_pars.iloc[0].to_numpy()
            sigma2_b = float(np.asarray(vals).ravel()[0])
            return sigma2_b, records
        arr = np.asarray(cov_pars).ravel()
        return float(arr[0]) if arr.size > 0 else None, {f"par_{i}": float(v) for i, v in enumerate(arr)}
    except Exception as exc:  # pragma: no cover
        return None, {"error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit frequentist GLMM (Laplace MLE via GPBoost).")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Phase 0 modeling table to fit on. Overrides --horizon-years/--history-years.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for the output files written to digihealth_risk/phase_1/outputs/.",
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=3,
        help="Convenience: select modeling table by horizon (used when --input-path is not set).",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=5,
        help="Convenience: select modeling table by history window (used when --input-path is not set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.input_path is not None:
        input_path = args.input_path
        horizon, history = parse_horizon_history(input_path)
    else:
        horizon, history = args.horizon_years, args.history_years
        input_path = modeling_table_path(horizon, history)

    prefix = args.output_prefix or f"phase_1_v2_glmm_gpboost_horizon_{horizon}_history_{history}"

    print(f"[glmm-gpboost] Loading: {input_path}")
    print(f"[glmm-gpboost] Output prefix: {prefix}")
    df = load_modeling_table(input_path)
    train_df, test_df = apply_canonical_split(df)
    print(f"[glmm-gpboost] Train rows: {len(train_df):,} ({train_df['PatientId'].nunique():,} patients)")
    print(f"[glmm-gpboost] Test  rows: {len(test_df):,} ({test_df['PatientId'].nunique():,} patients)")

    X_train, X_test, feature_names = prepare_features(train_df, test_df)
    y_train = train_df["Target_AtRisk_Status"].astype(int).to_numpy()
    y_test = test_df["Target_AtRisk_Status"].astype(int).to_numpy()
    groups_train = train_df["PatientId"].astype(str).to_numpy()
    groups_test = test_df["PatientId"].astype(str).to_numpy()

    print(f"[glmm-gpboost] Feature matrix shape: train={X_train.shape}, test={X_test.shape}")
    print(f"[glmm-gpboost] Fitting binary GLMM with patient random intercept (bernoulli_logit)...")
    gp_model = fit_glmm(X_train, y_train, groups_train)

    sigma2_b, cov_pars_vals = extract_sigma2(gp_model)
    print(f"[glmm-gpboost] Covariance parameters: {cov_pars_vals}")
    print(f"[glmm-gpboost] Patient random-intercept variance: {sigma2_b}")

    p_test_subj, p_test_marg = predict_probabilities(gp_model, X_test, groups_test, sigma2_b)
    p_train_subj, p_train_marg = predict_probabilities(gp_model, X_train, groups_train, sigma2_b)

    threshold = float(y_train.mean())  # training positive rate (matches GEE/logistic convention)

    # Standardized metrics CSV row layout (matches GEE/logistic).
    metrics_rows = []
    for split_name, y, p_subj, p_marg in [
        ("train", y_train, p_train_subj, p_train_marg),
        ("test", y_test, p_test_subj, p_test_marg),
    ]:
        for calibration_method, p in [
            ("subject_specific", p_subj),
            ("marginal_pinheiro_bates", p_marg),
        ]:
            row = {
                "split": split_name,
                "calibration_method": calibration_method,
                "threshold_strategy": "train_positive_rate",
                **classification_metrics(y, p, threshold),
            }
            metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / f"{prefix}_metrics.csv", index=False)

    # Test predictions use the marginal prediction (held-out-appropriate).
    pred_df = test_df[["PatientId", "Year", "target_year", "Target_AtRisk_Status"]].copy()
    pred_df["predicted_probability"] = p_test_marg
    pred_df["predicted_probability_subject_specific"] = p_test_subj
    pred_df.to_csv(OUT_DIR / f"{prefix}_test_predictions.csv", index=False)

    # Coefficients
    try:
        coef_obj = gp_model.get_coef(std_err=True, format_pandas=True)
        if isinstance(coef_obj, pd.DataFrame):
            coef_obj.insert(0, "feature", ["intercept"] + feature_names[: len(coef_obj) - 1])
            coef_obj.to_csv(OUT_DIR / f"{prefix}_coefficients.csv", index=False)
    except Exception as exc:  # pragma: no cover
        print(f"[glmm-gpboost] WARN: could not extract coefficients: {exc}")

    # Rich JSON
    metrics_json = {
        "input_path": str(input_path),
        "horizon_years": horizon,
        "history_years": history,
        "sigma2_patient_random_intercept": sigma2_b,
        "n_features": X_train.shape[1],
        "cov_pars": cov_pars_vals,
        "subject_specific": {
            "train": classification_metrics(y_train, p_train_subj, threshold),
            "test": classification_metrics(y_test, p_test_subj, threshold),
        },
        "marginal_pinheiro_bates": {
            "train": classification_metrics(y_train, p_train_marg, threshold),
            "test": classification_metrics(y_test, p_test_marg, threshold),
        },
    }
    (OUT_DIR / f"{prefix}_metrics.json").write_text(json.dumps(metrics_json, indent=2, default=str))

    print(f"[glmm-gpboost] Held-out PR-AUC (marginal): {metrics_json['marginal_pinheiro_bates']['test']['pr_auc']:.4f}")
    print(f"[glmm-gpboost] Held-out ROC-AUC (marginal): {metrics_json['marginal_pinheiro_bates']['test']['roc_auc']:.4f}")
    print(f"[glmm-gpboost] Held-out Brier (marginal):  {metrics_json['marginal_pinheiro_bates']['test']['brier']:.4f}")
    print(f"[glmm-gpboost] Wrote outputs: {OUT_DIR}/{prefix}_*")


if __name__ == "__main__":
    main()
