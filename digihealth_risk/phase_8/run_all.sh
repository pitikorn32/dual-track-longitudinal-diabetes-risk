#!/usr/bin/env bash
# Phase 8 -- Logistic-only alternative screening model build.
#
# Runs export_logistic_models.py twice: once for the with-Year variant and
# once for the no-Year (phase 7 ablation) variant. Produces 30 logistic
# screening artifacts total (15 per variant).
#
# Run from the repository root:
#   bash digihealth_risk/phase_8/run_all.sh
#
# Options:
#   --skip-with-year   skip the with-Year variant
#   --skip-no-year     skip the no-Year variant
#   --fail-fast        stop at the first failure

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/digihealth_risk/phase_8/outputs/logs"
mkdir -p "${LOG_DIR}"

RUN_WITH_YEAR=1
RUN_NO_YEAR=1
FAIL_FAST=0

for arg in "$@"; do
  case "$arg" in
    --skip-with-year) RUN_WITH_YEAR=0 ;;
    --skip-no-year)   RUN_NO_YEAR=0 ;;
    --fail-fast)      FAIL_FAST=1 ;;
    *) echo "[phase_8] unknown option: $arg" >&2; exit 2 ;;
  esac
done

run_step() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"
  echo "[phase_8] === ${name} ==="
  echo "[phase_8] log: ${log#${ROOT_DIR}/}"
  python "$@" 2>&1 | tee "${log}"
  local status=${PIPESTATUS[0]}
  if [[ "${status}" -ne 0 ]]; then
    echo "[phase_8] ${name} FAILED (exit ${status})" >&2
    if [[ "${FAIL_FAST}" -eq 1 ]]; then
      exit "${status}"
    fi
    return "${status}"
  fi
  return 0
}

cd "${ROOT_DIR}"

[[ "${RUN_WITH_YEAR}" -eq 1 ]] && run_step "with_year" \
  "digihealth_risk/phase_8/export_logistic_models.py"

[[ "${RUN_NO_YEAR}" -eq 1 ]] && run_step "no_year" \
  "digihealth_risk/phase_8/export_logistic_models.py" --no-year

echo "[phase_8] outputs in digihealth_risk/phase_8/outputs/"
