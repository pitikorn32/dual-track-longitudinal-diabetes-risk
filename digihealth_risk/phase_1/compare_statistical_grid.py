"""Compare GEE, Logistic, and GLMM (GPBoost) across the (N, M) grid.

Reads each family's per-cell ``phase_1_v2_*_metrics.csv`` and produces a
consolidated comparison CSV at::

    digihealth_risk/phase_1/outputs/phase_1_v2_statistical_grid_comparison.csv

Also writes a short markdown report to::

    digihealth_risk/phase_1/outputs/phase_1_v2_statistical_grid_comparison.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "digihealth_risk" / "phase_1" / "outputs"

HORIZONS = [1, 2, 3, 4, 5]
HISTORIES = [1, 3, 5]
FAMILIES = ["gee", "logistic", "glmm_gpboost"]


def read_family_cell(family: str, n: int, m: int) -> dict | None:
    path = OUT / f"phase_1_v2_{family}_horizon_{n}_history_{m}_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # GLMM has two calibration_method rows per split; pick marginal_pinheiro_bates for held-out.
    if "calibration_method" in df.columns and family == "glmm_gpboost":
        df = df[df["calibration_method"] == "marginal_pinheiro_bates"]
    test_row = df[df["split"] == "test"]
    if test_row.empty:
        return None
    row = test_row.iloc[0]
    return {
        "family": family,
        "N": n,
        "M": m,
        "PR-AUC": float(row["pr_auc"]),
        "ROC-AUC": float(row["roc_auc"]),
        "Brier": float(row["brier"]),
        "rows": int(row["rows"]),
        "positives": int(row["positives"]),
        "positive_rate": float(row["positive_rate"]),
    }


def main() -> None:
    rows = []
    for family in FAMILIES:
        for n in HORIZONS:
            for m in HISTORIES:
                cell = read_family_cell(family, n, m)
                if cell is not None:
                    rows.append(cell)
    df = pd.DataFrame(rows)

    csv_path = OUT / "phase_1_v2_statistical_grid_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Build a pivot per metric for the markdown report
    md_lines = ["# Phase 1 v2 Statistical Family Grid Comparison (test set)", ""]
    md_lines.append("Test-set metrics by horizon $N$ and history $M$. GLMM (GPBoost) ")
    md_lines.append("uses the marginal Pinheiro--Bates prediction appropriate for held-out patients.")
    md_lines.append("")

    for metric in ["PR-AUC", "ROC-AUC", "Brier"]:
        pivot = df.pivot_table(index=["family", "M"], columns="N", values=metric)
        pivot = pivot.reindex(level=0, labels=FAMILIES)
        md_lines.append(f"## {metric}")
        md_lines.append("")
        md_lines.append(pivot.to_markdown(floatfmt=".4f"))
        md_lines.append("")

    # Family head-to-head: at each cell, who wins on PR-AUC?
    md_lines.append("## Family head-to-head on PR-AUC (test, per cell)")
    md_lines.append("")
    head = df.pivot_table(index="M", columns=["N"], values="PR-AUC", aggfunc="max")
    md_lines.append("Best PR-AUC across the three families at each (N, M) cell:")
    md_lines.append("")
    md_lines.append(head.to_markdown(floatfmt=".4f"))
    md_lines.append("")

    md_lines.append("Winning family at each (N, M) cell (test PR-AUC):")
    md_lines.append("")
    winners = (
        df.loc[df.groupby(["N", "M"])["PR-AUC"].idxmax(), ["N", "M", "family", "PR-AUC"]]
        .pivot(index="M", columns="N", values="family")
    )
    md_lines.append(winners.to_markdown())
    md_lines.append("")

    md_path = OUT / "phase_1_v2_statistical_grid_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
