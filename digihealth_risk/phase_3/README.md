# Phase 3 — Survival Models

## Thesis reference
Section 4.1.3 (Survival Models), Section 5.3 (Survival-Model Results)

## Purpose
Two survival formulations directly comparable to the binary classification families:
1. **Landmark Cox** — each eligible patient-year is a landmark origin; fixed-horizon
   binary labels derived from future AtRisk status.
2. **Two-stage rolling survival** — Stage 1 forecasts near-future covariates (Bayesian
   ridge per feature); Stage 2 fits Cox on forecasted values. History window M∈{1,3,5}
   is explicitly tested (key thesis result: M=1 degenerates; M=3 is strongest).

The baseline Cox model (one-row-per-patient, 2005 baseline) is **not included** — it
does not operate on rolling observations and is not in the final leaderboard.

## Prerequisites
- `digihealth_risk/phase_0/outputs/patient_year_long.pkl` (Phase 0 Step 1)

## Step-by-step

### Step 1 — Landmark Cox
```bash
python digihealth_risk/phase_3/landmark_cox.py
```

### Step 2 — Two-stage survival (M=3 recommended; also run M=1 and M=5)
```bash
python digihealth_risk/phase_3/two_stage_survival.py --history-window 3
python digihealth_risk/phase_3/two_stage_survival.py --history-window 5
python digihealth_risk/phase_3/two_stage_survival.py --history-window 1
```
The M=1 run is expected to produce degenerate (ROC-AUC ≈ 0.5) results — this is
the key negative result reported in Section 5.3.

## Key outputs

| File | Description |
|------|-------------|
| `phase_3_2_v2_test_predictions.csv` | Landmark Cox test predictions |
| `phase_3_2_v2_metrics.csv` | Landmark Cox metrics |
| `phase_3_3_v2_h3_test_predictions.csv` | Two-stage M=3 test predictions |
| `phase_3_3_v2_h5_test_predictions.csv` | Two-stage M=5 test predictions |
| `phase_3_3_v2_h1_test_predictions.csv` | Two-stage M=1 (degenerate baseline) |

## Expected runtime
~10–30 min per run
