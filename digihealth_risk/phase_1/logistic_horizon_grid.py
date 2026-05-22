"""Run Phase 1 v2 logistic across horizon/history tables.

This mirrors the GEE grid wrappers so Logistic v2 can be compared on the same
rolling N/M task family.

Run from the repository root:
    python digihealth_risk/phase_1/logistic_horizon_grid.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE0_OUT = ROOT / "digihealth_risk" / "phase_0" / "outputs"
PHASE1_OUT = ROOT / "digihealth_risk" / "phase_1" / "outputs"
LOGISTIC_SCRIPT = ROOT / "digihealth_risk" / "phase_1" / "logistic.py"
DEFAULT_HORIZONS = [1, 2, 3, 4, 5]
DEFAULT_HISTORY = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 v2 logistic for multiple N/M tables.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=DEFAULT_HORIZONS,
        help="Prediction horizons N to run.",
    )
    parser.add_argument(
        "--history-years",
        type=int,
        default=DEFAULT_HISTORY,
        help="History window M to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Logistic v2 even when the metrics output already exists.",
    )
    return parser.parse_args()


def phase0_table_path(horizon: int, history_years: int) -> Path:
    if horizon == 1 and history_years == 1:
        return PHASE0_OUT / "phase_0_modeling_table.pkl"
    return PHASE0_OUT / f"phase_0_modeling_table_horizon_{horizon}_history_{history_years}.pkl"


def output_prefix(horizon: int, history_years: int) -> str:
    return f"phase_1_v2_logistic_horizon_{horizon}_history_{history_years}"


def run_logistic(input_path: Path, prefix: str) -> None:
    command = [
        sys.executable,
        str(LOGISTIC_SCRIPT),
        "--input-path",
        str(input_path),
        "--output-prefix",
        prefix,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    PHASE1_OUT.mkdir(parents=True, exist_ok=True)

    for horizon in sorted(set(args.horizons)):
        input_path = phase0_table_path(horizon, args.history_years)
        prefix = output_prefix(horizon, args.history_years)
        metrics_path = PHASE1_OUT / f"{prefix}_metrics.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing Phase 0 table for N={horizon}, M={args.history_years}: {input_path}"
            )

        if metrics_path.exists() and not args.force:
            print(
                f"Skipping existing Logistic v2 output N={horizon}, M={args.history_years}: {metrics_path}"
            )
            continue

        print(f"Running Logistic v2 N={horizon}, M={args.history_years}")
        run_logistic(input_path, prefix)


if __name__ == "__main__":
    main()
