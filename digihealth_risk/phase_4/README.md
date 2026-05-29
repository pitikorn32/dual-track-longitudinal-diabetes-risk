# Phase 4: Calibration, Threshold Policy & Final Leaderboard

## Thesis reference
Section 4.2 (Calibration and Fair Final Comparison),
Section 4.4.4 (Threshold and Policy Layer),
Section 5.1 (Final Leaderboard Table)

## Purpose
Three scripts that together produce the final cross-family comparison:
1. **calibrate_trees.py**: applies Platt scaling and isotonic regression to tree
   finalists; evaluates raw vs. calibrated variants.
2. **threshold_optimization.py**: evaluates recall-constrained and F1-optimal
   threshold policies on the calibration subset.
3. **cross_family_comparison.py**: shared-cohort comparison across all families
   (trees, GEE, logistic, landmark Cox, two-stage survival). Produces the final
   leaderboard used in Section 5.1.

## Prerequisites
- `digihealth_risk/phase_0/outputs/`: modeling tables (Phase 0)
- `digihealth_risk/phase_2/outputs/`: tree predictions (Phase 2 Step 1–2)
- `digihealth_risk/phase_1/outputs/`: GEE and logistic predictions (Phase 1)
- `digihealth_risk/phase_3/outputs/`: survival predictions (Phase 3)

## Step-by-step

### Step 1: Calibrate tree finalists
```bash
python digihealth_risk/phase_4/calibrate_trees.py
```

### Step 2: Threshold policy optimization
```bash
python digihealth_risk/phase_4/threshold_optimization.py
```

### Step 3: Final cross-family leaderboard
```bash
python digihealth_risk/phase_4/cross_family_comparison.py
```

## Key outputs

| File | Description |
|------|-------------|
| `phase_4_v2_test_predictions.csv` | Calibrated tree test predictions |
| `phase_4_v2_metrics.csv` | Calibration metrics (Brier, PR-AUC, ROC-AUC) |
| `phase_5_v2_report.md` | Threshold policy results |
| `phase_4_2_v2_cross_family_ranking.csv` | **Final leaderboard** (used by Phase 5) |
| `phase_4_2_v2_best_by_horizon.csv` | Best model per horizon |
| `phase_4_2_v2_cross_family_report.md` | Final recommendation report |

## Expected runtime
~5–20 min total
