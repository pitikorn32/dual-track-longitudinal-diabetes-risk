"""Phase 0 EDA and long-format dataset construction.

Run from the repository root:
    python digihealth_risk/phase_0/build_modeling_tables.py

Outputs are written to digihealth_risk/phase_0/outputs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "datasets" / "df_final.pkl"
OUT_DIR = ROOT / "digihealth_risk" / "phase_0" / "outputs"

YEARS = list(range(2005, 2017))
DEFAULT_HORIZON_YEARS = 1
DEFAULT_HISTORY_YEARS = 1
STATIC_FEATURES = [
    "PatientId",
    "date_of_birth",
    "gender",
    "dm_first_degree_relative",
    "cooking_method",
    "total_sugary_week",
    "total_veg_fruit_week",
    "total_exercise_week",
    "total_phy_activity_week",
    "sleep_hours",
    "sleep_quality",
    "smoking_status",
    "alcohol_status",
]
CLINICAL_FEATURES = ["FBS", "BMI", "Pulse", "BL_pres1", "BL_pres2", "Waist"]


def install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to read pickles created by NumPy 2.x."""
    import sys
    import numpy.core as np_core
    import numpy.core.multiarray as np_multiarray
    import numpy.core.numeric as np_numeric

    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input data: {DATA_PATH}")
    install_numpy_pickle_compat()
    return pd.read_pickle(DATA_PATH)


def status_from_max_fbs(max_fbs: pd.Series) -> pd.Series:
    """Observed data uses non_dm <=100, pre_dm 101-125, dm >=126."""
    status = pd.Series(pd.NA, index=max_fbs.index, dtype="object")
    status[max_fbs <= 100] = "non_dm"
    status[(max_fbs > 100) & (max_fbs <= 125)] = "pre_dm"
    status[max_fbs > 125] = "dm"
    return status


def first_atrisk_year(row: pd.Series) -> float:
    for year in YEARS:
        if row.get(f"AtRisk_{year}") == 1:
            return float(year)
    return np.nan


def years_since_last_observed(values: list[float], years: list[int]) -> dict[int, float]:
    last_seen: int | None = None
    result: dict[int, float] = {}
    for year, value in zip(years, values, strict=True):
        if pd.notna(value):
            last_seen = year
            result[year] = 0.0
        else:
            result[year] = np.nan if last_seen is None else float(year - last_seen)
    return result


def markdown_table(records: list[dict[str, object]]) -> str:
    if not records:
        return ""
    columns = list(records[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        lines.append("| " + " | ".join(str(record.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    long_parts: list[pd.DataFrame] = []

    for year in YEARS:
        part = df[STATIC_FEATURES].copy()
        part["Year"] = year
        for feature in CLINICAL_FEATURES:
            part[feature] = df[f"{feature}_{year}"]
        part["MAX_FBS_up_to_year"] = df[f"MAX_FBS_up_to_{year}"]
        part["DM_status_up_to_year"] = df[f"DM_status_up_to_{year}"].astype("object")
        part["AtRisk_current_year"] = df[f"AtRisk_{year}"]
        part["first_atrisk_year"] = df.apply(first_atrisk_year, axis=1)
        long_parts.append(part)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df["Age"] = long_df["Year"] - pd.to_datetime(long_df["date_of_birth"]).dt.year
    long_df["pulse_pressure"] = long_df["BL_pres1"] - long_df["BL_pres2"]
    long_df["clinical_observed_count"] = long_df[CLINICAL_FEATURES].notna().sum(axis=1)
    long_df["has_fbs_this_year"] = long_df["FBS"].notna().astype("int8")

    fbs_since = {}
    for _, row in df.iterrows():
        mapping = years_since_last_observed([row[f"FBS_{year}"] for year in YEARS], YEARS)
        fbs_since[row["PatientId"]] = mapping
    long_df["years_since_last_fbs"] = [
        fbs_since[pid][year] for pid, year in zip(long_df["PatientId"], long_df["Year"], strict=True)
    ]

    return long_df


def validate_config(horizon_years: int, history_years: int) -> None:
    if horizon_years < 1:
        raise ValueError("--horizon-years must be >= 1")
    if horizon_years >= len(YEARS):
        raise ValueError(f"--horizon-years must be <= {len(YEARS) - 1}")
    if history_years < 1:
        raise ValueError("--history-years must be >= 1")


def output_suffix(horizon_years: int, history_years: int) -> str:
    if horizon_years == DEFAULT_HORIZON_YEARS and history_years == DEFAULT_HISTORY_YEARS:
        return ""
    return f"_horizon_{horizon_years}_history_{history_years}"


def add_history_features(model_df: pd.DataFrame, df: pd.DataFrame, history_years: int) -> pd.DataFrame:
    if history_years <= 1:
        return model_df

    wide = df.set_index("PatientId")
    records: list[dict[str, float | int]] = []
    min_year = min(YEARS)

    for pid, source_year in zip(model_df["PatientId"], model_df["Year"], strict=True):
        start_year = max(min_year, int(source_year) - history_years + 1)
        window_years = list(range(start_year, int(source_year) + 1))
        record: dict[str, float | int] = {"history_start_year": start_year}

        for feature in CLINICAL_FEATURES:
            values = pd.Series([wide.loc[pid, f"{feature}_{year}"] for year in window_years], index=window_years)
            observed = values.dropna()
            prefix = f"{feature}_hist_{history_years}y"
            record[f"{prefix}_observed_count"] = int(observed.size)
            record[f"{prefix}_missing_count"] = int(values.isna().sum())
            record[f"{prefix}_latest"] = float(observed.iloc[-1]) if not observed.empty else np.nan
            record[f"{prefix}_mean"] = float(observed.mean()) if not observed.empty else np.nan
            record[f"{prefix}_min"] = float(observed.min()) if not observed.empty else np.nan
            record[f"{prefix}_max"] = float(observed.max()) if not observed.empty else np.nan
            record[f"{prefix}_std"] = float(observed.std(ddof=0)) if observed.size > 1 else 0.0 if observed.size == 1 else np.nan
            record[f"{prefix}_range"] = (
                float(observed.max() - observed.min()) if not observed.empty else np.nan
            )
            if observed.size >= 2:
                x = observed.index.to_numpy(dtype=float)
                y = observed.to_numpy(dtype=float)
                record[f"{prefix}_slope"] = float(np.polyfit(x - x.min(), y, deg=1)[0])
            else:
                record[f"{prefix}_slope"] = np.nan

        records.append(record)

    history_df = pd.DataFrame(records, index=model_df.index)
    return pd.concat([model_df, history_df], axis=1)


def build_modeling_table(
    long_df: pd.DataFrame,
    df: pd.DataFrame,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    history_years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    validate_config(horizon_years, history_years)
    target_lookup = df.set_index("PatientId")[[f"AtRisk_{year}" for year in YEARS]]
    fbs_lookup = df.set_index("PatientId")[[f"FBS_{year}" for year in YEARS]]
    source_years = [year for year in YEARS if year + horizon_years in YEARS]
    model_df = long_df[long_df["Year"].isin(source_years)].copy()
    model_df["prediction_horizon_years"] = horizon_years
    model_df["history_window_years"] = history_years
    model_df["target_year"] = model_df["Year"] + horizon_years
    model_df["Target_AtRisk_Status"] = [
        target_lookup.loc[pid, f"AtRisk_{year + horizon_years}"]
        for pid, year in zip(model_df["PatientId"], model_df["Year"], strict=True)
    ]

    model_df["is_missing_last_year"] = model_df.apply(
        lambda row: (
            np.nan
            if row["Year"] == YEARS[0]
            else pd.isna(fbs_lookup.loc[row["PatientId"], f"FBS_{int(row['Year']) - 1}"])
        ),
        axis=1,
    )

    # Modeling first onset: keep only rows known to be non-risk at source year,
    # and remove all rows at/after a patient's first at-risk year.
    before_first_event = model_df["first_atrisk_year"].isna() | (model_df["Year"] < model_df["first_atrisk_year"])
    model_df = model_df[
        before_first_event & (model_df["AtRisk_current_year"] == 0) & model_df["Target_AtRisk_Status"].notna()
    ].copy()

    model_df["Target_AtRisk_Status"] = model_df["Target_AtRisk_Status"].astype("int8")
    if horizon_years == 1:
        model_df["Next_Year_AtRisk_Status"] = model_df["Target_AtRisk_Status"]
    model_df["is_missing_last_year"] = model_df["is_missing_last_year"].astype("boolean")
    model_df = add_history_features(model_df, df, history_years)
    return model_df


def summarize_eda(
    df: pd.DataFrame,
    long_df: pd.DataFrame,
    model_df: pd.DataFrame,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    history_years: int = DEFAULT_HISTORY_YEARS,
) -> str:
    lines: list[str] = []
    lines.append("# Phase 0 EDA Report")
    lines.append("")
    lines.append(f"Input file: `{DATA_PATH.relative_to(ROOT)}`")
    lines.append(f"Patient-level shape: `{df.shape[0]:,}` rows x `{df.shape[1]:,}` columns")
    lines.append(f"Duplicate `PatientId` values: `{df['PatientId'].duplicated().sum():,}`")
    lines.append(f"Full long table shape: `{long_df.shape[0]:,}` rows x `{long_df.shape[1]:,}` columns")
    lines.append(f"Censored modeling table shape: `{model_df.shape[0]:,}` rows x `{model_df.shape[1]:,}` columns")
    lines.append(f"Prediction horizon: `{horizon_years}` year(s)")
    lines.append(f"Historical lookback window: `{history_years}` year(s)")
    lines.append("")

    lines.append("## Target Distribution")
    target_rows = []
    for year in YEARS:
        target = df[f"AtRisk_{year}"]
        target_rows.append(
            {
                "Year": year,
                "AtRisk_0": int((target == 0).sum()),
                "AtRisk_1": int((target == 1).sum()),
                "Missing": int(target.isna().sum()),
            }
        )
    lines.append(markdown_table(target_rows))
    lines.append("")

    first_year_counts = df.apply(first_atrisk_year, axis=1).value_counts(dropna=False).sort_index()
    lines.append("## First At-Risk Year")
    first_year_records = []
    for first_year, patients in first_year_counts.items():
        first_year_records.append(
            {
                "first_atrisk_year": "never" if pd.isna(first_year) else int(first_year),
                "patients": int(patients),
            }
        )
    lines.append(markdown_table(first_year_records))
    lines.append("")

    lines.append("## Missingness Summary")
    missing_rows = []
    for feature in CLINICAL_FEATURES:
        values = [df[f"{feature}_{year}"].isna().mean() for year in YEARS]
        missing_rows.append(
            {
                "feature": feature,
                "min_missing_pct": round(min(values) * 100, 2),
                "max_missing_pct": round(max(values) * 100, 2),
                "avg_missing_pct": round(float(np.mean(values)) * 100, 2),
            }
        )
    lines.append(markdown_table(missing_rows))
    lines.append("")

    lines.append("## Status Threshold Check")
    mismatch_rows = []
    for year in YEARS:
        observed = df[f"DM_status_up_to_{year}"].astype("object")
        expected = status_from_max_fbs(df[f"MAX_FBS_up_to_{year}"])
        mismatch = ((expected != observed) & ~(expected.isna() & observed.isna())).sum()
        mismatch_rows.append({"Year": year, "mismatches": int(mismatch)})
    lines.append(markdown_table(mismatch_rows))
    lines.append("")
    lines.append(
        "Observed labels are consistent with `non_dm <= 100`, `pre_dm 101-125`, "
        "and `dm >= 126`. This differs from the README threshold text, so Phase 0 "
        "uses existing `AtRisk_YEAR` labels instead of recomputing targets."
    )
    lines.append("")

    lines.append("## Modeling Table Definition")
    lines.append(
        f"Each modeling row uses source-year `T` features to predict `AtRisk_{{T+{horizon_years}}}`. "
        "Rows are kept only when the patient is known non-risk at `T`, the horizon-year "
        "target is observed, and `T` occurs before the patient's first at-risk year."
    )
    if history_years > 1:
        lines.append(
            f"Historical features use only the last `{history_years}` calendar years ending at `T`; "
            "no values after `T` are used."
        )
    lines.append("")

    lines.append("## Modeling Target Distribution")
    overall_target = model_df["Target_AtRisk_Status"]
    lines.append(
        f"Overall positives: `{int((overall_target == 1).sum()):,}` / "
        f"`{len(overall_target):,}` rows "
        f"({(overall_target.mean() * 100):.2f}%)."
    )
    model_target_rows = []
    for year, group in model_df.groupby("Year", sort=True):
        target = group["Target_AtRisk_Status"]
        model_target_rows.append(
            {
                "source_year": int(year),
                "target_year": int(year + horizon_years),
                "rows": int(len(group)),
                "positives": int((target == 1).sum()),
                "positive_pct": round(float(target.mean()) * 100, 2),
            }
        )
    lines.append(markdown_table(model_target_rows))
    lines.append("")

    lines.append("Generated files:")
    lines.append("- `patient_year_long.pkl`: all patient-year rows for 2005-2016.")
    suffix = output_suffix(horizon_years, history_years)
    lines.append(f"- `phase_0_modeling_table{suffix}.pkl`: censored first-onset table for modeling.")
    lines.append(f"- `phase_0_eda_report{suffix}.md`: this report.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 0 EDA and modeling tables.")
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=DEFAULT_HORIZON_YEARS,
        help="Predict AtRisk status N years after source year T.",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=DEFAULT_HISTORY_YEARS,
        help="Add rolling historical clinical features from the last M years ending at T.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_config(args.horizon_years, args.history_years)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    long_df = build_long_table(df)
    model_df = build_modeling_table(long_df, df, args.horizon_years, args.history_years)
    suffix = output_suffix(args.horizon_years, args.history_years)

    long_df.to_pickle(OUT_DIR / "patient_year_long.pkl")
    model_df.to_pickle(OUT_DIR / f"phase_0_modeling_table{suffix}.pkl")
    model_df.head(1000).to_csv(OUT_DIR / f"phase_0_modeling_table_sample{suffix}.csv", index=False)

    report = summarize_eda(df, long_df, model_df, args.horizon_years, args.history_years)
    (OUT_DIR / f"phase_0_eda_report{suffix}.md").write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
