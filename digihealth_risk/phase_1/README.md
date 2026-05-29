# Phase 1: Statistical Models (GEE, Logistic, GLMM)

## Thesis reference
Section 4.1.1 (Biostatistical Models), Section 5.1 (Pure Prediction Performance),
Section 5.5.1 (Feature Ablation: Statistical Models)

## Purpose
Trains the three biostatistical model families on the rolling patient-year tables.
GEE and Logistic (refined features) are included in the final leaderboard.
GLMM is exploratory (unstable rolling artifacts, not in final comparison).

## Scripts

| Script | Role |
|--------|------|
| `gee_horizon_grid.py` | GEE across all N/M combinations (final leaderboard entry) |
| `logistic_horizon_grid.py` | Penalised logistic across all N/M (final leaderboard entry) |
| `glmm_exploratory.py` | GLMM with random intercepts, exploratory only |

## Prerequisites
- `digihealth_risk/phase_0/outputs/phase_0_modeling_table.pkl` (Phase 0 Step 1)
- For grid runs: all `phase_0_modeling_table_horizon_{N}_history_{M}.pkl` files (Phase 0 Step 2)

## Step-by-step

### Step 1: GEE horizon/history grid (M=5 recommended for leaderboard)
```bash
python digihealth_risk/phase_1/gee_horizon_grid.py --history-years 5
```
For all M values:
```bash
python digihealth_risk/phase_1/gee_horizon_grid.py --history-years 1
python digihealth_risk/phase_1/gee_horizon_grid.py --history-years 3
python digihealth_risk/phase_1/gee_horizon_grid.py --history-years 5
```

### Step 2: Logistic horizon/history grid
```bash
python digihealth_risk/phase_1/logistic_horizon_grid.py --history-years 5
```
For all M values:
```bash
for M in 1 3 5; do
  python digihealth_risk/phase_1/logistic_horizon_grid.py --history-years $M
done
```

### Step 3: GLMM (optional exploratory)
```bash
python digihealth_risk/phase_1/glmm_exploratory.py
```

## Key outputs

| File | Description |
|------|-------------|
| `phase_1_v2_gee_horizon_{N}_history_{M}_metrics.csv` | GEE metrics per N/M |
| `phase_1_v2_gee_horizon_{N}_history_{M}_test_predictions.csv` | Test-set predictions |
| `phase_1_v2_logistic_horizon_{N}_history_{M}_metrics.csv` | Logistic metrics per N/M |
| `phase_1_v2_logistic_horizon_{N}_history_{M}_test_predictions.csv` | Test-set predictions |

## Expected runtime
~5–15 min per N/M combination (GEE is slower than Logistic)
