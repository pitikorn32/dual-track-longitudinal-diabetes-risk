"""Phase 3.3 v2: Two-stage landmark survival — improved feature engineering.

Changes from v1 (phase_3_3_two_stage_survival.py) driven by Phase 0.2 EDA:
  Stage 1: unchanged — STAGE1_FEATURES has no pulse_pressure already.
  Stage 2:
    - Removed `pulse_pressure` (VIF=225; = BL_pres1 − BL_pres2 by construction)
    - Added `Year_centered_sq` (U-shaped temporal trend)
    - Added `FBS_x_Age`, `MAX_FBS_x_Age` interactions
  NOTE: FBS_hinge_100, FBS_hinge_125, and mean_FBS_hinge_100 excluded — the modeling
    table only contains non-at-risk rows (FBS ≤ 100 mg/dL by construction), making all
    three features all-zero (std=0) and causing a singular Cox Hessian.

Run from the repository root:
    python digihealth_risk/phase_3/two_stage_survival.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.duration.hazard_regression import PHReg


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_3.survival_utils import (
    concordance_index,
    hazard_ratio_table,
    markdown_table,
)
from digihealth_risk.utils.patient_split import apply_canonical_split


LONG_PATH = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "patient_year_long.pkl"
OUT_DIR = ROOT / "digihealth_risk" / "phase_3" / "outputs"
RANDOM_SEED = 20260501
BINARY_HORIZONS = [1, 2, 3, 4, 5]
END_YEAR = 2016
DEFAULT_HISTORY_WINDOW = 3
FORECAST_K = 1

import argparse


STAGE1_FEATURES = ["FBS", "BMI", "Pulse", "BL_pres1", "BL_pres2"]

# Stage 2 static features — v2 removes pulse_pressure, adds engineered features
STAGE2_NUMERIC_STATIC = [
    "Year",
    "Year_centered_sq",        # v2: U-shaped temporal trend
    "Age",
    "Waist",
    # pulse_pressure removed (VIF=225; = BL_pres1 − BL_pres2 by construction)
    # FBS_hinge_100 / FBS_hinge_125: all-zero in modeling table (non-at-risk rows
    #   have FBS ≤ 100 mg/dL by construction) → std=0 → singular Cox Hessian
    "FBS_x_Age",               # v2: FBS × Age interaction
    "MAX_FBS_up_to_year",
    "MAX_FBS_x_Age",           # v2: MAX_FBS_up_to_year × Age (cross-lag r=0.582)
    "clinical_observed_count",
    "has_fbs_this_year",
    "years_since_last_fbs",
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
]

CATEGORICAL_FEATURES = [
    "gender",
    "dm_first_degree_relative",
    "sleep_quality",
    "smoking_status",
    "alcohol_status",
]

STAGE2_FORECAST_MEAN_COLS = [f"mean_{feat}" for feat in STAGE1_FEATURES]
# mean_FBS_hinge_100 excluded — mean_FBS forecasts for non-at-risk patients are also
# ≤ 100 mg/dL in practice, making the derived column all-zero
STAGE2_FORECAST_COLS = STAGE2_FORECAST_MEAN_COLS + ["forecast_uncertainty"]
STAGE2_ALL_NUMERIC = STAGE2_NUMERIC_STATIC + STAGE2_FORECAST_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Phase 3.3 v2 two-stage landmark survival model.")
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help=f"Years of lag history for Stage 1 inputs (default: {DEFAULT_HISTORY_WINDOW}).",
    )
    parser.add_argument(
        "--output-prefix",
        default="phase_3_3_v2",
        help="Output prefix written under digihealth_risk/phase_3/outputs/.",
    )
    return parser.parse_args()


def load_long_table() -> pd.DataFrame:
    return pd.read_pickle(LONG_PATH).copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Year_centered" not in df.columns:
        df["Year_centered"] = df["Year"] - df["Year"].min()
    if "Year_centered_sq" not in df.columns:
        df["Year_centered_sq"] = df["Year_centered"] ** 2
    if "FBS_hinge_100" not in df.columns:
        df["FBS_hinge_100"] = (df["FBS"] - 100).clip(lower=0)
    if "FBS_hinge_125" not in df.columns:
        df["FBS_hinge_125"] = (df["FBS"] - 125).clip(lower=0)
    if "FBS_x_Age" not in df.columns:
        df["FBS_x_Age"] = df["FBS"] * df["Age"]
    if "MAX_FBS_x_Age" not in df.columns:
        df["MAX_FBS_x_Age"] = df["MAX_FBS_up_to_year"] * df["Age"]
    return df


def split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return apply_canonical_split(df)


def first_event_and_censor(long_df: pd.DataFrame) -> pd.DataFrame:
    grps = long_df.groupby("PatientId", sort=False)
    first_event = grps.apply(
        lambda g: g.loc[g["AtRisk_current_year"].eq(1), "Year"].min(),
        include_groups=False,
    ).rename("first_atrisk_year_landmark")
    last_obs = grps.apply(
        lambda g: g.loc[g["AtRisk_current_year"].notna(), "Year"].max(),
        include_groups=False,
    ).rename("last_observed_year_landmark")
    return pd.concat([first_event, last_obs], axis=1).reset_index()


def build_landmark_table(long_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = first_event_and_censor(long_df)
    table = long_df.merge(outcomes, on="PatientId", how="left")
    table = table[table["AtRisk_current_year"].eq(0)].copy()
    table = table[table["Year"].lt(END_YEAR)].copy()

    table["event_after_landmark"] = (
        table["first_atrisk_year_landmark"].notna()
        & table["first_atrisk_year_landmark"].gt(table["Year"])
    )
    table["event"] = table["event_after_landmark"].astype(int)
    table["event_or_censor_year"] = np.where(
        table["event"].eq(1),
        table["first_atrisk_year_landmark"],
        table["last_observed_year_landmark"],
    )
    table["duration"] = (table["event_or_censor_year"] - table["Year"]).clip(lower=1).astype(float)
    table = table[table["event_or_censor_year"].gt(table["Year"])].copy()

    status_lookup = long_df.set_index(["PatientId", "Year"])["AtRisk_current_year"]
    for horizon in BINARY_HORIZONS:
        target_idx = pd.MultiIndex.from_arrays(
            [table["PatientId"], table["Year"] + horizon],
            names=["PatientId", "Year"],
        )
        table[f"Target_AtRisk_Status_horizon_{horizon}"] = (
            status_lookup.reindex(target_idx).to_numpy()
        )
    return table.reset_index(drop=True)


def _s1_input_cols(feat: str, history_window: int) -> list[str]:
    return [f"{feat}_lag{lag}" for lag in range(history_window)] + [
        "context_Year",
        "context_Age",
    ]


def build_stage1_history(long_df: pd.DataFrame, df: pd.DataFrame, history_window: int) -> pd.DataFrame:
    feat_lookup = long_df.set_index(["PatientId", "Year"])
    hist = pd.DataFrame(
        {"PatientId": df["PatientId"].values, "Year": df["Year"].values}
    )

    for feat in STAGE1_FEATURES:
        if feat not in feat_lookup.columns:
            continue
        feat_series = feat_lookup[feat]
        for lag in range(history_window):
            idx = pd.MultiIndex.from_arrays(
                [hist["PatientId"], hist["Year"] - lag],
                names=["PatientId", "Year"],
            )
            hist[f"{feat}_lag{lag}"] = feat_series.reindex(idx).values

    hist["context_Year"] = df["Year"].values
    if "Age" in feat_lookup.columns:
        idx0 = pd.MultiIndex.from_arrays(
            [hist["PatientId"], hist["Year"]],
            names=["PatientId", "Year"],
        )
        hist["context_Age"] = feat_lookup["Age"].reindex(idx0).values
    else:
        hist["context_Age"] = np.nan

    return hist


def build_stage1_targets(
    long_df: pd.DataFrame, df: pd.DataFrame, k: int = FORECAST_K
) -> pd.DataFrame:
    feat_lookup = long_df.set_index(["PatientId", "Year"])
    tgt = pd.DataFrame(
        {"PatientId": df["PatientId"].values, "Year": df["Year"].values}
    )

    for feat in STAGE1_FEATURES:
        if feat not in feat_lookup.columns:
            continue
        idx = pd.MultiIndex.from_arrays(
            [tgt["PatientId"], tgt["Year"] + k],
            names=["PatientId", "Year"],
        )
        tgt[f"{feat}_target"] = feat_lookup[feat].reindex(idx).values

    return tgt


def train_stage1_models(
    hist_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    history_window: int,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}

    for feat in STAGE1_FEATURES:
        target_col = f"{feat}_target"
        if target_col not in tgt_df.columns:
            continue

        cols = [c for c in _s1_input_cols(feat, history_window) if c in hist_df.columns]
        mask = tgt_df[target_col].notna().values
        x = hist_df.loc[mask, cols].to_numpy(dtype=float)
        y = tgt_df.loc[mask, target_col].to_numpy(dtype=float)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", BayesianRidge()),
        ])
        pipe.fit(x, y)
        models[feat] = pipe

    return models


def predict_stage1(
    models: dict[str, Pipeline], hist_df: pd.DataFrame, history_window: int
) -> pd.DataFrame:
    preds = pd.DataFrame(index=range(len(hist_df)))
    std_cols: list[str] = []

    for feat, pipe in models.items():
        cols = [c for c in _s1_input_cols(feat, history_window) if c in hist_df.columns]
        x = hist_df[cols].to_numpy(dtype=float)
        x_imp = pipe.named_steps["imputer"].transform(x)
        x_sc = pipe.named_steps["scaler"].transform(x_imp)
        mean_p, std_p = pipe.named_steps["model"].predict(x_sc, return_std=True)

        preds[f"mean_{feat}"] = mean_p
        preds[f"std_{feat}"] = np.abs(std_p)
        std_cols.append(f"std_{feat}")

    if std_cols:
        preds["forecast_uncertainty"] = preds[std_cols].mean(axis=1)

    return preds


def compute_stage1_rmse(
    models: dict[str, Pipeline],
    hist_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    split: str,
    history_window: int,
) -> pd.DataFrame:
    preds = predict_stage1(models, hist_df, history_window)
    rows = []
    for feat in models:
        target_col = f"{feat}_target"
        if target_col not in tgt_df.columns:
            continue
        mask = tgt_df[target_col].notna().values
        if mask.sum() == 0:
            continue
        y_true = tgt_df.loc[mask, target_col].to_numpy(dtype=float)
        y_pred = preds.loc[mask, f"mean_{feat}"].to_numpy(dtype=float)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        rows.append(
            {
                "split": split,
                "feature": feat,
                "n_rows": int(mask.sum()),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                "mean_pred_std": float(preds.loc[mask, f"std_{feat}"].mean()),
            }
        )
    return pd.DataFrame(rows)


def attach_stage1_preds(df: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    result = df.reset_index(drop=True).copy()
    for col in preds.columns:
        result[col] = preds[col].values
    return result


def make_stage2_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first")

    avail_num = [c for c in STAGE2_ALL_NUMERIC if c in df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder),
    ])
    transformers = [("num", num_pipe, avail_num)]
    if avail_cat:
        transformers.append(("cat", cat_pipe, avail_cat))

    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)


def fit_cox(train_df: pd.DataFrame) -> tuple[object, ColumnTransformer, list[str]]:
    preprocessor = make_stage2_preprocessor(train_df)
    x_train = preprocessor.fit_transform(train_df)
    feature_names = [str(n) for n in preprocessor.get_feature_names_out()]
    model = PHReg(
        endog=train_df["duration"].to_numpy(),
        exog=x_train,
        status=train_df["event"].to_numpy(),
        ties="efron",
    )
    result = model.fit(groups=train_df["PatientId"])
    return result, preprocessor, feature_names


def risk_score(
    result: object, preprocessor: ColumnTransformer, df: pd.DataFrame
) -> np.ndarray:
    x = preprocessor.transform(df)
    return np.asarray(x @ result.params, dtype=float)


def baseline_survival_at_horizons(
    train_df: pd.DataFrame, train_score: np.ndarray
) -> dict[int, float]:
    exp_score = np.exp(train_score)
    baseline_hazard = 0.0
    survival_by_time: dict[int, float] = {}
    event_times = sorted(train_df.loc[train_df["event"].eq(1), "duration"].unique())
    for t in event_times:
        n_events = int(
            ((train_df["duration"].eq(t)) & train_df["event"].eq(1)).sum()
        )
        risk_set = train_df["duration"].ge(t).to_numpy()
        risk_sum = float(exp_score[risk_set].sum())
        if risk_sum > 0:
            baseline_hazard += n_events / risk_sum
        survival_by_time[int(t)] = float(np.exp(-baseline_hazard))
    result_map = {}
    for horizon in BINARY_HORIZONS:
        eligible = [t for t in survival_by_time if t <= horizon]
        result_map[horizon] = survival_by_time[max(eligible)] if eligible else 1.0
    return result_map


def event_probability_by_horizon(
    score: np.ndarray, baseline_survival: dict[int, float]
) -> dict[int, np.ndarray]:
    exp_score = np.exp(score)
    return {h: 1.0 - np.power(s0, exp_score) for h, s0 in baseline_survival.items()}


def binary_label_for_horizon(df: pd.DataFrame, horizon: int) -> np.ndarray:
    return df[f"Target_AtRisk_Status_horizon_{horizon}"].astype(int).to_numpy()


def binary_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = probability >= threshold
    return {
        "rows": float(len(y_true)),
        "positives": float(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(
            ((~prediction) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)
        ),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def binary_horizon_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_probabilities: dict[int, np.ndarray],
    test_probabilities: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    base_predictions = test_df[[
        "PatientId", "Year", "duration", "event",
        "first_atrisk_year_landmark", "last_observed_year_landmark",
    ]].copy()

    for horizon in BINARY_HORIZONS:
        target_col = f"Target_AtRisk_Status_horizon_{horizon}"
        train_mask = train_df[target_col].notna().to_numpy()
        test_mask = test_df[target_col].notna().to_numpy()

        y_train = binary_label_for_horizon(train_df.loc[train_mask], horizon)
        y_test = binary_label_for_horizon(test_df.loc[test_mask], horizon)
        threshold = float(y_train.mean())

        for split, y_true, prob in [
            ("train", y_train, train_probabilities[horizon][train_mask]),
            ("test", y_test, test_probabilities[horizon][test_mask]),
        ]:
            metric_rows.append(
                {
                    "split": split,
                    "horizon_years": horizon,
                    "calibration_method": "raw",
                    "threshold_strategy": "train_positive_rate",
                    **binary_metrics(y_true, prob, threshold),
                }
            )

        pred = base_predictions.loc[test_mask].copy()
        pred["horizon_years"] = horizon
        pred["target_year"] = pred["Year"] + horizon
        pred["Target_AtRisk_Status"] = y_test
        pred["predicted_probability"] = test_probabilities[horizon][test_mask]
        prediction_frames.append(pred)

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def write_report(
    landmark_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stage1_rmse: pd.DataFrame,
    cox_metrics: pd.DataFrame,
    binary_metrics_df: pd.DataFrame,
    hr_df: pd.DataFrame,
    history_window: int = DEFAULT_HISTORY_WINDOW,
) -> str:
    lines = [
        "# Phase 3.3 v2 Two-Stage Landmark Survival Report",
        "",
        "## Scope",
        "Improved two-stage survival model. Stage 1 unchanged (BayesianRidge per clinical feature). "
        "Stage 2 removes `pulse_pressure` (VIF=225), adds `Year_centered_sq`, and interaction terms. "
        "FBS hockey-stick features excluded — modeling table contains only non-at-risk rows "
        "(FBS ≤ 100 mg/dL by construction), making them all-zero.",
        "",
        "## v2 Changes vs v1",
        "| Stage | Change | Feature | Reason |",
        "| --- | --- | --- | --- |",
        "| Stage 2 | Removed | `pulse_pressure` | VIF=225; equals BL_pres1 − BL_pres2 |",
        "| Stage 2 | Added | `Year_centered_sq` | U-shaped temporal risk trend |",
        "| Stage 2 | Added | `FBS_x_Age`, `MAX_FBS_x_Age` | Cross-lag top interactions |",
        "| Stage 2 | Excluded | `FBS_hinge_100`, `FBS_hinge_125`, `mean_FBS_hinge_100` | All-zero in non-at-risk modeling table |",
        "",
        "## Architecture",
        f"**Stage 1** — one BayesianRidge per clinical feature, forecasting `k={FORECAST_K}` year ahead",
        f"from `M={history_window}` years of lag history. Posterior predictive mean and std feed Stage 2.",
        "",
        "**Stage 2** — Cox PHReg with static + forecast features. Duration and event identical to Phase 3.2 v2.",
        "",
        "## Data Summary",
        f"Landmark rows: `{len(landmark_df):,}`.",
        f"Train rows: `{len(train_df):,}` across `{train_df['PatientId'].nunique():,}` patients.",
        f"Test rows: `{len(test_df):,}` across `{test_df['PatientId'].nunique():,}` patients.",
        "",
        "## Stage 1 — Feature Forecasting Quality",
        markdown_table(stage1_rmse),
        "",
        "## Stage 2 — Cox Ranking Metrics",
        markdown_table(cox_metrics),
        "",
        "## Fixed-Horizon Binary Metrics",
        markdown_table(binary_metrics_df[binary_metrics_df["split"].eq("test")]),
        "",
        "## Top Hazard Ratios",
        markdown_table(hr_df, max_rows=25),
        "",
        "## Notes",
        "- Stage 1 trained on training patients only; applied to test patients at inference.",
        "- `forecast_uncertainty` = mean(std_FBS, std_BMI, ...) collapses 5 nearly-collinear std features.",
        "- FBS hockey-stick features excluded: non-at-risk modeling table has FBS ≤ 100 mg/dL by construction.",
        "",
    ]
    return "\n".join(lines)


def prefixed_output(prefix: str, suffix: str) -> Path:
    return OUT_DIR / f"{prefix}_{suffix}"


def main() -> None:
    args = parse_args()
    history_window = args.history_window
    output_prefix = args.output_prefix
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    long_df = load_long_table()
    landmark_df = build_landmark_table(long_df)
    landmark_df = engineer_features(landmark_df)  # add FBS hinges, Year_centered_sq, interactions
    train_df, test_df = split_by_patient(landmark_df)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Landmark rows: {len(landmark_df):,}  |  train: {len(train_df):,}  test: {len(test_df):,}")
    print(f"Stage 1 history window: {history_window} years")

    print("Building Stage 1 history matrices...")
    hist_train = build_stage1_history(long_df, train_df, history_window)
    hist_test = build_stage1_history(long_df, test_df, history_window)
    tgt_train = build_stage1_targets(long_df, train_df)
    tgt_test = build_stage1_targets(long_df, test_df)

    print("Training Stage 1 BayesianRidge models...")
    stage1_models = train_stage1_models(hist_train, tgt_train, history_window)
    print(f"  Trained models for: {list(stage1_models.keys())}")

    rmse_train = compute_stage1_rmse(stage1_models, hist_train, tgt_train, split="train", history_window=history_window)
    rmse_test = compute_stage1_rmse(stage1_models, hist_test, tgt_test, split="test", history_window=history_window)
    stage1_rmse = pd.concat([rmse_train, rmse_test], ignore_index=True)

    print("Generating Stage 1 forecasts...")
    s1_train = predict_stage1(stage1_models, hist_train, history_window)
    s1_test = predict_stage1(stage1_models, hist_test, history_window)

    train_s2 = attach_stage1_preds(train_df, s1_train)
    test_s2 = attach_stage1_preds(test_df, s1_test)

    print("Fitting Stage 2 Cox model...")
    cox_result, preprocessor, feature_names = fit_cox(train_s2)
    train_score = risk_score(cox_result, preprocessor, train_s2)
    test_score = risk_score(cox_result, preprocessor, test_s2)
    baseline_survival = baseline_survival_at_horizons(train_s2, train_score)
    train_probs = event_probability_by_horizon(train_score, baseline_survival)
    test_probs = event_probability_by_horizon(test_score, baseline_survival)

    cox_metrics = pd.DataFrame([
        {
            "split": "train",
            "rows": len(train_s2),
            "patients": train_s2["PatientId"].nunique(),
            "events": int(train_s2["event"].sum()),
            "event_rate": float(train_s2["event"].mean()),
            "c_index": concordance_index(
                train_s2["duration"].to_numpy(),
                train_s2["event"].to_numpy(),
                train_score,
            ),
        },
        {
            "split": "test",
            "rows": len(test_s2),
            "patients": test_s2["PatientId"].nunique(),
            "events": int(test_s2["event"].sum()),
            "event_rate": float(test_s2["event"].mean()),
            "c_index": concordance_index(
                test_s2["duration"].to_numpy(),
                test_s2["event"].to_numpy(),
                test_score,
            ),
        },
    ])

    binary_metrics_df, predictions = binary_horizon_evaluation(
        train_s2, test_s2, train_probs, test_probs,
    )
    hr_df = hazard_ratio_table(cox_result, feature_names)

    stage1_rmse.to_csv(prefixed_output(output_prefix, "stage1_rmse.csv"), index=False)
    cox_metrics.to_csv(prefixed_output(output_prefix, "cox_metrics.csv"), index=False)
    binary_metrics_df.to_csv(prefixed_output(output_prefix, "binary_horizon_metrics.csv"), index=False)
    predictions.to_csv(prefixed_output(output_prefix, "test_predictions.csv"), index=False)
    hr_df.to_csv(prefixed_output(output_prefix, "hazard_ratios.csv"), index=False)

    report = write_report(
        landmark_df, train_s2, test_s2,
        stage1_rmse, cox_metrics, binary_metrics_df, hr_df,
        history_window=history_window,
    )
    prefixed_output(output_prefix, "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
