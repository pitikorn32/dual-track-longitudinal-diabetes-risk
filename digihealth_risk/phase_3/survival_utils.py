"""Shared Phase 3 v2 survival reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def concordance_index(duration: np.ndarray, event: np.ndarray, score: np.ndarray) -> float:
    comparable = 0
    concordant = 0.0
    n = len(duration)
    for i in range(n):
        if event[i] != 1:
            continue
        mask = duration[i] < duration
        if not mask.any():
            continue
        comparable += int(mask.sum())
        concordant += float((score[i] > score[mask]).sum())
        concordant += 0.5 * float((score[i] == score[mask]).sum())
    return float(concordant / comparable) if comparable else np.nan


def hazard_ratio_table(result: object, feature_names: list[str]) -> pd.DataFrame:
    params = np.asarray(result.params, dtype=float)
    se = np.asarray(result.bse, dtype=float)
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": params,
            "std_error": se,
            "hazard_ratio": np.exp(params),
            "hr_ci_low": np.exp(params - 1.96 * se),
            "hr_ci_high": np.exp(params + 1.96 * se),
            "p_value": np.asarray(result.pvalues, dtype=float),
        }
    )
    table["abs_log_hr"] = table["coefficient"].abs()
    return table.replace([np.inf, -np.inf], np.nan).sort_values("abs_log_hr", ascending=False).drop(columns="abs_log_hr")


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
