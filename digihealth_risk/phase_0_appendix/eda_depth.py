"""Phase 0.2 In-Depth EDA: Correlation, Autocorrelation, and Phase 1 Bridge.

Reads:
  - patient_year_long.pkl      (82,704 rows × 29 cols, censored at first event)
  - phase_0_modeling_table.pkl (41,930 rows × 35 cols, horizon=1, history=1)

Outputs (digihealth_risk/phase_0_appendix/outputs/):
  phase_0_2_univariate_stats.csv
  phase_0_2_temporal_trends.csv
  phase_0_2_autocorrelation.csv
  phase_0_2_ljung_box.csv
  phase_0_2_cross_lagged_correlation.csv
  phase_0_2_pearson_correlation.csv
  phase_0_2_spearman_correlation.csv
  phase_0_2_vif.csv
  phase_0_2_missing_by_year.csv
  phase_0_2_joint_missingness.csv
  phase_0_2_mar_test.csv
  phase_0_2_phase1_bridge.csv
  phase_0_2_report.md

Run from the repository root:
    python digihealth_risk/phase_0_appendix/eda_depth.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LONG_PATH  = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "patient_year_long.pkl"
MODEL_PATH = ROOT / "digihealth_risk" / "phase_0" / "outputs" / "phase_0_modeling_table.pkl"
OUT_DIR    = ROOT / "digihealth_risk" / "phase_0_appendix" / "outputs"

RANDOM_SEED = 20260501
MAX_SHAPIRO_N = 5_000
LAGS = [1, 2, 3, 4, 5]

CLINICAL_FEATURES = ["FBS", "BMI", "Pulse", "BL_pres1", "BL_pres2", "Waist", "pulse_pressure"]
QUESTIONNAIRE_FEATURES = [
    "total_sugary_week", "total_veg_fruit_week", "total_exercise_week",
    "total_phy_activity_week", "sleep_hours",
]
DERIVED_FEATURES = [
    "Age", "MAX_FBS_up_to_year", "clinical_observed_count",
    "has_fbs_this_year", "years_since_last_fbs",
]
CATEGORICAL_FEATURES = [
    "gender", "dm_first_degree_relative", "sleep_quality",
    "smoking_status", "alcohol_status", "cooking_method",
]
NUMERIC_FEATURES = CLINICAL_FEATURES + QUESTIONNAIRE_FEATURES + DERIVED_FEATURES
TARGET = "Target_AtRisk_Status"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0.2 in-depth EDA")
    p.add_argument("--long-path", type=Path, default=LONG_PATH,
                   help="Path to patient_year_long.pkl")
    p.add_argument("--input-path", type=Path, default=MODEL_PATH,
                   help="Path to phase_0_modeling_table.pkl")
    return p.parse_args()


def load_data(long_path: Path, model_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = pd.read_pickle(long_path)
    mod_df  = pd.read_pickle(model_path)
    return long_df, mod_df


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    display = df.copy()
    for col in display.select_dtypes(include=[np.number]).columns:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    cols = display.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _present_features(df: pd.DataFrame, feat_list: list[str]) -> list[str]:
    return [f for f in feat_list if f in df.columns]


# ---------------------------------------------------------------------------
# Block 1: Univariate Statistics
# ---------------------------------------------------------------------------

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def _cramers_v(ct: np.ndarray) -> float:
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n > 0 else float("nan")


def block1_univariate(mod_df: pd.DataFrame) -> pd.DataFrame:
    y = mod_df[TARGET].astype(int)
    pos_mask = y == 1
    neg_mask = y == 0
    rows: list[dict] = []

    for feat in _present_features(mod_df, NUMERIC_FEATURES):
        col = mod_df[feat]
        valid = col.dropna()
        n_total = len(col)
        n_missing = int(col.isna().sum())

        q25, q75 = (
            float(valid.quantile(0.25)) if len(valid) > 0 else float("nan"),
            float(valid.quantile(0.75)) if len(valid) > 0 else float("nan"),
        )
        skew = float(valid.skew()) if len(valid) > 3 else float("nan")
        kurt = float(valid.kurt()) if len(valid) > 3 else float("nan")

        sw_p = float("nan")
        if len(valid) >= 8:
            sample = valid.sample(min(MAX_SHAPIRO_N, len(valid)), random_state=RANDOM_SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, sw_p = stats.shapiro(sample)

        pos_vals = col[pos_mask].dropna().to_numpy(dtype=float)
        neg_vals = col[neg_mask].dropna().to_numpy(dtype=float)
        cohen_d = _cohen_d(pos_vals, neg_vals)

        mwu_p = float("nan")
        if len(pos_vals) > 0 and len(neg_vals) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, mwu_p = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")

        mi_col = col.fillna(float(col.median()) if col.notna().any() else 0.0)
        mi_score = float(
            mutual_info_classif(
                mi_col.to_numpy().reshape(-1, 1),
                y.to_numpy(),
                random_state=RANDOM_SEED,
            )[0]
        )

        rows.append({
            "feature": feat,
            "type": "numeric",
            "n_rows": n_total,
            "n_missing": n_missing,
            "missing_rate": round(n_missing / n_total, 4),
            "mean": round(float(valid.mean()), 4) if len(valid) > 0 else float("nan"),
            "std": round(float(valid.std(ddof=1)), 4) if len(valid) > 1 else float("nan"),
            "median": round(float(valid.median()), 4) if len(valid) > 0 else float("nan"),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "min": round(float(valid.min()), 4) if len(valid) > 0 else float("nan"),
            "max": round(float(valid.max()), 4) if len(valid) > 0 else float("nan"),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "shapiro_wilk_p": round(float(sw_p), 4),
            "cohen_d_vs_target": round(cohen_d, 4),
            "mannwhitney_p": round(float(mwu_p), 6),
            "mutual_info": round(mi_score, 4),
            "cramers_v": float("nan"),
            "chi2_p": float("nan"),
        })

    for feat in _present_features(mod_df, CATEGORICAL_FEATURES):
        raw = mod_df[feat]
        n_total = len(raw)
        n_missing = int(raw.isna().sum())
        col = raw.astype("object").fillna("missing").astype(str)

        ct = pd.crosstab(col, y).to_numpy()
        cramers = _cramers_v(ct)
        chi2_p = float("nan")
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chi2_p = float(stats.chi2_contingency(ct, correction=False)[1])

        rows.append({
            "feature": feat,
            "type": "categorical",
            "n_rows": n_total,
            "n_missing": n_missing,
            "missing_rate": round(n_missing / n_total, 4),
            "mean": float("nan"), "std": float("nan"), "median": float("nan"),
            "q25": float("nan"), "q75": float("nan"),
            "min": float("nan"), "max": float("nan"),
            "skewness": float("nan"), "kurtosis": float("nan"),
            "shapiro_wilk_p": float("nan"),
            "cohen_d_vs_target": float("nan"),
            "mannwhitney_p": float("nan"),
            "mutual_info": float("nan"),
            "cramers_v": round(cramers, 4),
            "chi2_p": round(chi2_p, 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block 2: Temporal Trends
# ---------------------------------------------------------------------------

def block2_temporal(mod_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    numeric_feats = _present_features(mod_df, NUMERIC_FEATURES)

    for year, grp in mod_df.groupby("Year"):
        row: dict = {
            "Year": int(year),
            "n_rows": int(len(grp)),
            "events": int(grp[TARGET].sum()),
            "event_rate": round(float(grp[TARGET].mean()), 4),
        }
        for feat in numeric_feats:
            col = grp[feat].dropna()
            row[f"{feat}_mean"]   = round(float(col.mean()), 4)   if len(col) > 0 else float("nan")
            row[f"{feat}_std"]    = round(float(col.std(ddof=1)), 4) if len(col) > 1 else float("nan")
            row[f"{feat}_median"] = round(float(col.median()), 4) if len(col) > 0 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Block 3: Within-Patient Autocorrelation
# ---------------------------------------------------------------------------

def block3_autocorrelation(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    long_sorted = long_df.sort_values(["PatientId", "Year"]).reset_index(drop=True)
    numeric_feats = _present_features(long_sorted, NUMERIC_FEATURES)

    ac_rows:  list[dict] = []
    lb_rows:  list[dict] = []

    for feat in numeric_feats:
        feat_df = long_sorted[["PatientId", "Year", feat]].copy()

        # Pooled lag-k Pearson: correlate x_t with x_{t+k} across all patient pairs
        ac_row: dict = {"feature": feat}
        for lag in LAGS:
            lagged = feat_df.copy()
            lagged["Year"] = lagged["Year"] + lag
            lagged = lagged.rename(columns={feat: f"_lag"})
            merged = feat_df.merge(lagged[["PatientId", "Year", "_lag"]],
                                   on=["PatientId", "Year"], how="inner")
            valid = merged[[feat, "_lag"]].dropna()
            if len(valid) > 2:
                r, p = stats.pearsonr(valid[feat].to_numpy(dtype=float),
                                      valid["_lag"].to_numpy(dtype=float))
                ac_row[f"lag{lag}_r"] = round(float(r), 4)
                ac_row[f"lag{lag}_p"] = round(float(p), 6)
            else:
                ac_row[f"lag{lag}_r"] = float("nan")
                ac_row[f"lag{lag}_p"] = float("nan")
        ac_rows.append(ac_row)

        # Ljung-Box on the grand-mean series (population-level trajectory)
        lb_row: dict = {"feature": feat}
        for lag in LAGS:
            lb_row[f"lb_stat_lag{lag}"]   = float("nan")
            lb_row[f"lb_pvalue_lag{lag}"] = float("nan")

        by_year = (
            long_sorted.groupby("Year")[feat].mean().sort_index().dropna()
        )
        if len(by_year) >= 10:
            max_lag = min(max(LAGS), len(by_year) // 2)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lb = acorr_ljungbox(
                        by_year.to_numpy(), lags=list(range(1, max_lag + 1)), return_df=True
                    )
                for lag_val in lb.index:
                    if lag_val in LAGS:
                        lb_row[f"lb_stat_lag{lag_val}"]   = round(float(lb.loc[lag_val, "lb_stat"]), 4)
                        lb_row[f"lb_pvalue_lag{lag_val}"] = round(float(lb.loc[lag_val, "lb_pvalue"]), 6)
            except Exception:
                pass
        lb_rows.append(lb_row)

    return pd.DataFrame(ac_rows), pd.DataFrame(lb_rows)


# ---------------------------------------------------------------------------
# Block 4: Cross-Lagged Correlation (feature@T vs AtRisk@T+k)
# ---------------------------------------------------------------------------

def block4_cross_lagged(long_df: pd.DataFrame) -> pd.DataFrame:
    long_sorted = long_df.sort_values(["PatientId", "Year"]).reset_index(drop=True)
    atrisk = long_sorted[["PatientId", "Year", "AtRisk_current_year"]].copy()
    numeric_feats = _present_features(long_sorted, NUMERIC_FEATURES)

    rows: list[dict] = []
    for feat in numeric_feats:
        feat_df = long_sorted[["PatientId", "Year", feat]].copy()

        for lag in LAGS:
            # Shift outcome backward by lag so it aligns with feature at T
            future = atrisk.copy()
            future["Year"] = future["Year"] - lag
            future = future.rename(columns={"AtRisk_current_year": "_outcome"})

            merged = feat_df.merge(
                future[["PatientId", "Year", "_outcome"]], on=["PatientId", "Year"], how="inner"
            )
            valid = merged[[feat, "_outcome"]].dropna()
            valid = valid[valid["_outcome"].notna()]

            if len(valid) < 10:
                rows.append({
                    "feature": feat, "lag_years": lag,
                    "pearson_r": float("nan"), "pearson_p": float("nan"),
                    "spearman_r": float("nan"), "spearman_p": float("nan"),
                    "n_pairs": 0,
                })
                continue

            x = valid[feat].to_numpy(dtype=float)
            y_out = valid["_outcome"].to_numpy(dtype=float)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr, pp = stats.pearsonr(x, y_out)
                sr, sp = stats.spearmanr(x, y_out)

            rows.append({
                "feature": feat,
                "lag_years": lag,
                "pearson_r": round(float(pr), 4),
                "pearson_p": round(float(pp), 6),
                "spearman_r": round(float(sr), 4),
                "spearman_p": round(float(sp), 6),
                "n_pairs": int(len(valid)),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Block 5: Inter-Feature Correlation and VIF
# ---------------------------------------------------------------------------

def block5_correlation_vif(
    mod_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    features = _present_features(mod_df, NUMERIC_FEATURES)
    feat_df = mod_df[features].copy()

    pearson_corr  = feat_df.corr(method="pearson").reset_index().rename(columns={"index": "feature"})
    spearman_corr = feat_df.corr(method="spearman").reset_index().rename(columns={"index": "feature"})

    # VIF — impute medians, drop constant/all-NaN columns
    feat_filled = feat_df.copy()
    for col in feat_filled.columns:
        median = feat_filled[col].median()
        feat_filled[col] = feat_filled[col].fillna(median if pd.notna(median) else 0.0)
    feat_filled = feat_filled.dropna(axis=1)
    feat_filled = feat_filled.loc[:, feat_filled.std() > 0]

    vif_features = feat_filled.columns.tolist()
    X = feat_filled.to_numpy(dtype=float)
    X_c = np.column_stack([np.ones(len(X)), X])  # add intercept column

    vif_rows: list[dict] = []
    for i, fname in enumerate(vif_features):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vif_val = variance_inflation_factor(X_c, i + 1)
            vif_val = float("nan") if not np.isfinite(vif_val) else round(float(vif_val), 4)
        except Exception:
            vif_val = float("nan")
        vif_rows.append({"feature": fname, "VIF": vif_val})

    vif_df = (
        pd.DataFrame(vif_rows)
        .sort_values("VIF", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return pearson_corr, spearman_corr, vif_df


# ---------------------------------------------------------------------------
# Block 6: Missing Data Analysis
# ---------------------------------------------------------------------------

def block6_missing(
    mod_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_feats = _present_features(mod_df, NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    clinical  = _present_features(mod_df, CLINICAL_FEATURES)

    # Missing rate by Year
    yr_rows: list[dict] = []
    for year, grp in mod_df.groupby("Year"):
        row: dict = {"Year": int(year), "n_rows": int(len(grp))}
        for feat in all_feats:
            row[f"{feat}_missing_rate"] = round(float(grp[feat].isna().mean()), 4)
        yr_rows.append(row)
    missing_by_year_df = pd.DataFrame(yr_rows).sort_values("Year").reset_index(drop=True)

    # Joint missingness matrix (clinical features only)
    miss_mat = mod_df[clinical].isna()
    joint_rows: list[dict] = []
    for f1 in clinical:
        for f2 in clinical:
            joint_rows.append({
                "feature1": f1,
                "feature2": f2,
                "both_missing_rate": round(float((miss_mat[f1] & miss_mat[f2]).mean()), 4),
            })
    joint_df = pd.DataFrame(joint_rows)

    # MAR test: predict each clinical feature's missingness from questionnaire + Age + Year
    observed_feats = _present_features(
        mod_df, QUESTIONNAIRE_FEATURES + ["Age", "Year", "clinical_observed_count"]
    )
    mar_rows: list[dict] = []
    for feat in clinical:
        y_miss = mod_df[feat].isna().astype(int)
        n_missing = int(y_miss.sum())
        if n_missing < 10 or n_missing >= len(y_miss) - 10:
            mar_rows.append({"feature": feat, "n_missing": n_missing, "mar_roc_auc": float("nan"),
                             "interpretation": "insufficient variance in missingness"})
            continue
        X_obs = mod_df[observed_feats].fillna(mod_df[observed_feats].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_obs.to_numpy(dtype=float))
        try:
            lr = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, C=1.0)
            lr.fit(X_scaled, y_miss.to_numpy())
            prob = lr.predict_proba(X_scaled)[:, 1]
            auc = round(float(roc_auc_score(y_miss.to_numpy(), prob)), 4)
            interp = (
                "likely MAR/MNAR" if auc >= 0.65
                else "plausibly MCAR" if auc < 0.55
                else "ambiguous"
            )
            mar_rows.append({"feature": feat, "n_missing": n_missing, "mar_roc_auc": auc,
                             "interpretation": interp})
        except Exception as exc:
            mar_rows.append({"feature": feat, "n_missing": n_missing, "mar_roc_auc": float("nan"),
                             "interpretation": str(exc)})
    mar_df = (
        pd.DataFrame(mar_rows)
        .sort_values("mar_roc_auc", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return missing_by_year_df, joint_df, mar_df


# ---------------------------------------------------------------------------
# Block 7: Phase 1 Bridge
# ---------------------------------------------------------------------------

def block7_phase1_bridge(
    univariate_df: pd.DataFrame,
    cross_lag_df: pd.DataFrame,
    vif_df: pd.DataFrame,
) -> pd.DataFrame:
    lag1 = (
        cross_lag_df[cross_lag_df["lag_years"] == 1][["feature", "pearson_r"]]
        .copy()
        .rename(columns={"pearson_r": "cross_lag1_r"})
    )
    lag1["abs_cross_lag1_r"] = lag1["cross_lag1_r"].abs()

    bridge = univariate_df[[
        "feature", "type", "missing_rate",
        "cohen_d_vs_target", "mutual_info", "cramers_v",
    ]].copy()
    bridge["abs_cohen_d"] = bridge["cohen_d_vs_target"].abs()

    bridge = bridge.merge(lag1[["feature", "cross_lag1_r", "abs_cross_lag1_r"]], on="feature", how="left")
    bridge = bridge.merge(vif_df[["feature", "VIF"]], on="feature", how="left")

    # Rank-based composite score
    for score_col in ["abs_cohen_d", "mutual_info", "abs_cross_lag1_r", "cramers_v"]:
        bridge[f"{score_col}_rank"] = bridge[score_col].rank(pct=True, na_option="bottom")

    rank_cols = ["abs_cohen_d_rank", "mutual_info_rank", "abs_cross_lag1_r_rank", "cramers_v_rank"]
    bridge["predictive_score"] = bridge[rank_cols].mean(axis=1)

    bridge["high_vif"]     = bridge["VIF"].gt(10)
    bridge["high_missing"] = bridge["missing_rate"].gt(0.20)
    # Recommend unless missing > 20% AND weak signal
    bridge["recommended_phase1"] = (
        (~bridge["high_missing"]) | (bridge["predictive_score"] > 0.5)
    )

    out_cols = [
        "feature", "type", "missing_rate",
        "abs_cohen_d", "cross_lag1_r", "mutual_info", "cramers_v",
        "VIF", "predictive_score",
        "high_vif", "high_missing", "recommended_phase1",
    ]
    return (
        bridge[out_cols]
        .sort_values("predictive_score", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    mod_df: pd.DataFrame,
    long_df: pd.DataFrame,
    univariate_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    autocorr_df: pd.DataFrame,
    ljungbox_df: pd.DataFrame,
    cross_lag_df: pd.DataFrame,
    vif_df: pd.DataFrame,
    missing_by_year_df: pd.DataFrame,
    mar_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
) -> str:
    pos_rate = float(mod_df[TARGET].mean())

    uv_display = (
        univariate_df
        .assign(_abs=univariate_df["cohen_d_vs_target"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        [["feature", "type", "missing_rate", "mean", "std", "skewness",
          "shapiro_wilk_p", "cohen_d_vs_target", "mannwhitney_p", "mutual_info", "cramers_v"]]
    )

    lag1_display = (
        cross_lag_df[cross_lag_df["lag_years"] == 1]
        .assign(_abs=cross_lag_df["pearson_r"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        [["feature", "lag_years", "pearson_r", "pearson_p", "spearman_r", "spearman_p", "n_pairs"]]
    )

    tier1 = bridge_df[
        (bridge_df["missing_rate"] <= 0.10) & (bridge_df["predictive_score"] >= 0.6)
    ]
    tier2 = bridge_df[
        (bridge_df["missing_rate"] <= 0.30) & ~bridge_df["feature"].isin(tier1["feature"])
    ]
    tier3 = bridge_df[
        ~bridge_df["feature"].isin(tier1["feature"]) & ~bridge_df["feature"].isin(tier2["feature"])
    ]

    def feat_list(df: pd.DataFrame) -> str:
        items = df["feature"].tolist()
        return ", ".join(f"`{f}`" for f in items) if items else "_(none)_"

    temporal_display_cols = (
        ["Year", "n_rows", "events", "event_rate"]
        + [c for c in temporal_df.columns if c.endswith("_mean") and
           c.split("_mean")[0] in ["FBS", "BMI", "Waist", "Age", "total_sugary_week"]]
    )

    lines = [
        "# Phase 0.2 — In-Depth EDA Report",
        "",
        "## 1. Dataset Overview",
        "",
        f"| Property | Value |",
        f"| --- | --- |",
        f"| Modeling table | {mod_df.shape[0]:,} rows × {mod_df.shape[1]} cols (horizon=1, history=1) |",
        f"| Long table | {long_df.shape[0]:,} rows × {long_df.shape[1]} cols |",
        f"| Unique patients | {long_df['PatientId'].nunique():,} |",
        f"| Year range | {int(long_df['Year'].min())}–{int(long_df['Year'].max())} |",
        f"| Positive rate (target) | {pos_rate:.1%} ({int(mod_df[TARGET].sum()):,} events) |",
        f"| Random seed | {RANDOM_SEED} |",
        "",
        "## 2. Univariate Statistics (sorted by |Cohen's d|)",
        "",
        "Cohen's d: standardised mean difference between positive and negative patients.",
        "Shapiro-Wilk p < 0.05 → non-normal (relevant for GEE/logistic assumptions).",
        "",
        markdown_table(uv_display, max_rows=20),
        "",
        "## 3. Temporal Trends",
        "",
        "Feature population mean and event rate by survey year.",
        "",
        markdown_table(temporal_df[temporal_display_cols]),
        "",
        "## 4. Within-Patient Autocorrelation (pooled lag 1–5)",
        "",
        "Pearson r between feature_t and feature_{t+k} paired within the same patient.",
        "High lag-1 autocorrelation means the feature is stable year-to-year within a patient.",
        "",
        markdown_table(autocorr_df),
        "",
        "### Ljung-Box Test on Population Mean Series",
        "",
        "Applied to the year-averaged feature series (12 data points).",
        "p < 0.05 → significant autocorrelation in the population-level trajectory.",
        "",
        markdown_table(ljungbox_df),
        "",
        "## 5. Cross-Lagged Correlation — feature@T vs AtRisk@T+k (k=1)",
        "",
        "Predictive correlation between each feature and the at-risk outcome 1 year ahead.",
        "Positive r: higher feature value → higher future risk.",
        "",
        markdown_table(lag1_display),
        "",
        "## 6. Inter-Feature Correlation — VIF",
        "",
        "VIF > 10 indicates high multicollinearity; consider dropping or combining features.",
        "",
        markdown_table(vif_df),
        "",
        "## 7. Missing Data Analysis",
        "",
        "### MAR Logistic Regression",
        "",
        "ROC-AUC of predicting each feature's missingness from questionnaire + Age + Year.",
        "AUC ≥ 0.65 → missingness is predictable (MAR or MNAR); use model-based imputation.",
        "AUC < 0.55 → plausibly MCAR; median/mean imputation acceptable.",
        "",
        markdown_table(mar_df),
        "",
        "## 8. Phase 1 Bridge — Feature Recommendations",
        "",
        "Composite predictive score = mean percentile rank across |Cohen's d|, MI, |cross-lag-1 r|.",
        "`high_vif = VIF > 10` | `high_missing = missing_rate > 20%`",
        "",
        markdown_table(bridge_df, max_rows=25),
        "",
        "### Recommended Feature Sets for Phase 1",
        "",
        f"**Tier 1** — high signal, low missing (missing ≤ 10%, score ≥ 0.6):",
        feat_list(tier1),
        "",
        f"**Tier 2** — moderate signal or moderate missing (missing ≤ 30%):",
        feat_list(tier2),
        "",
        f"**Tier 3** — high missing or low signal (require careful imputation):",
        feat_list(tier3),
        "",
        "### Phase 1 Imputation Guidance",
        "",
        "| Mechanism | Features | Recommended imputation |",
        "| --- | --- | --- |",
        "| MAR / MNAR (AUC ≥ 0.65) | FBS, Waist, BL_pres1/2 | Median + missing indicator; "
        "or multiple imputation |",
        "| MCAR (AUC < 0.55) | Questionnaire features | Simple median/mode |",
        "| Structural zero | has_fbs_this_year, years_since_last_fbs | Already encoded |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    long_df, mod_df = load_data(args.long_path, args.input_path)
    print(f"  Long table : {long_df.shape}")
    print(f"  Model table: {mod_df.shape}")

    print("Block 1: Univariate statistics...")
    univariate_df = block1_univariate(mod_df)
    univariate_df.to_csv(OUT_DIR / "phase_0_2_univariate_stats.csv", index=False)

    print("Block 2: Temporal trends...")
    temporal_df = block2_temporal(mod_df)
    temporal_df.to_csv(OUT_DIR / "phase_0_2_temporal_trends.csv", index=False)

    print("Block 3: Within-patient autocorrelation (may take ~1 min)...")
    autocorr_df, ljungbox_df = block3_autocorrelation(long_df)
    autocorr_df.to_csv(OUT_DIR / "phase_0_2_autocorrelation.csv", index=False)
    ljungbox_df.to_csv(OUT_DIR / "phase_0_2_ljung_box.csv", index=False)

    print("Block 4: Cross-lagged correlation (may take ~2 min)...")
    cross_lag_df = block4_cross_lagged(long_df)
    cross_lag_df.to_csv(OUT_DIR / "phase_0_2_cross_lagged_correlation.csv", index=False)

    print("Block 5: Inter-feature correlation and VIF...")
    pearson_df, spearman_df, vif_df = block5_correlation_vif(mod_df)
    pearson_df.to_csv(OUT_DIR / "phase_0_2_pearson_correlation.csv", index=False)
    spearman_df.to_csv(OUT_DIR / "phase_0_2_spearman_correlation.csv", index=False)
    vif_df.to_csv(OUT_DIR / "phase_0_2_vif.csv", index=False)

    print("Block 6: Missing data analysis...")
    missing_by_year_df, joint_df, mar_df = block6_missing(mod_df)
    missing_by_year_df.to_csv(OUT_DIR / "phase_0_2_missing_by_year.csv", index=False)
    joint_df.to_csv(OUT_DIR / "phase_0_2_joint_missingness.csv", index=False)
    mar_df.to_csv(OUT_DIR / "phase_0_2_mar_test.csv", index=False)

    print("Block 7: Phase 1 bridge...")
    bridge_df = block7_phase1_bridge(univariate_df, cross_lag_df, vif_df)
    bridge_df.to_csv(OUT_DIR / "phase_0_2_phase1_bridge.csv", index=False)

    print("Writing report...")
    report = write_report(
        mod_df, long_df, univariate_df, temporal_df,
        autocorr_df, ljungbox_df, cross_lag_df,
        vif_df, missing_by_year_df, mar_df, bridge_df,
    )
    (OUT_DIR / "phase_0_2_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
