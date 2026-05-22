#!/usr/bin/env bash
# Phase 7 — Year-features ablation: retrain phases 2/4/5 without
# Year, Year_centered, Year_centered_sq and produce a comparison report.
#
# Run from the repository root:
#   bash digihealth_risk/phase_7/run_all.sh
#
# Options:
#   --skip-trees       skip the uncalibrated tree grid (phase 2 mirror)
#   --skip-calibration skip the calibrated tree grid (phase 4 mirror)
#   --skip-monotonic   skip the monotonic family grid (phase 5 mirror)
#   --report-only      only run the comparison step against existing outputs
#   --fail-fast        stop at the first failure

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/digihealth_risk/phase_7/outputs/logs"
mkdir -p "${LOG_DIR}"

RUN_TREES=1
RUN_CALIBRATION=1
RUN_MONOTONIC=1
REPORT_ONLY=0
FAIL_FAST=0

for arg in "$@"; do
  case "$arg" in
    --skip-trees) RUN_TREES=0 ;;
    --skip-calibration) RUN_CALIBRATION=0 ;;
    --skip-monotonic) RUN_MONOTONIC=0 ;;
    --report-only) REPORT_ONLY=1 ;;
    --fail-fast) FAIL_FAST=1 ;;
    *) echo "[phase_7] unknown option: $arg" >&2; exit 2 ;;
  esac
done

run_step() {
  local name="$1"
  local script="$2"
  local log="${LOG_DIR}/${name}.log"
  echo "[phase_7] === ${name} ==="
  echo "[phase_7] log: ${log#${ROOT_DIR}/}"
  python "${script}" 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  if [[ "${status}" -ne 0 ]]; then
    echo "[phase_7] ${name} FAILED (exit ${status})" >&2
    if [[ "${FAIL_FAST}" -eq 1 ]]; then
      exit "${status}"
    fi
    return "${status}"
  fi
  return 0
}

cd "${ROOT_DIR}"

if [[ "${REPORT_ONLY}" -eq 0 ]]; then
  [[ "${RUN_TREES}"       -eq 1 ]] && run_step "trees"       "digihealth_risk/phase_7/train_trees_no_year.py"
  [[ "${RUN_CALIBRATION}" -eq 1 ]] && run_step "calibration" "digihealth_risk/phase_7/calibrate_trees_no_year.py"
  [[ "${RUN_MONOTONIC}"   -eq 1 ]] && run_step "monotonic"   "digihealth_risk/phase_7/train_monotonic_no_year.py"
fi

run_step "compare" "digihealth_risk/phase_7/compare_with_baseline.py"

echo "[phase_7] outputs in digihealth_risk/phase_7/outputs/"
