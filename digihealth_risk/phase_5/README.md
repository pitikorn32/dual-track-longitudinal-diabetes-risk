# Phase 5: Intervention-Safe Models (Monotonic Constraints)

## Purpose
Trains five model families with monotonic constraints and evaluates directional
consistency under seven specified favorable scenarios on every shared test row. All five families are benchmarked together in
`intervention_benchmark.py`.

## Scripts

| Script | Family |
|--------|--------|
| `train_monotonic_xgboost.py` | Monotonic XGBoost (primary benchmark) |
| `train_monotonic_lightgbm.py` | Monotonic LightGBM |
| `train_monotonic_catboost.py` | Monotonic CatBoost |
| `train_monotonic_ebm.py` | Monotonic EBM (most interpretable) |
| `train_monotonic_logistic.py` | Monotonic logistic (statistical baseline) |
| `evaluate_saved_models.py` | Reevaluate exact saved artifacts on shared rows |
| `intervention_benchmark.py` | Consolidate all 5 families into one report |
| `intervention_scenarios.py` | Per-patient what-if scenario scoring |

## Prerequisites
- `digihealth_risk/phase_0/outputs/`: modeling tables (Phase 0)
- `digihealth_risk/phase_4/outputs/phase_4_2_v2_cross_family_ranking.csv` (Phase 4 Step 3)

## Step-by-step

### Step 1: Train all five monotonic families
```bash
python digihealth_risk/phase_5/train_monotonic_xgboost.py
python digihealth_risk/phase_5/train_monotonic_catboost.py
python digihealth_risk/phase_5/train_monotonic_lightgbm.py
python digihealth_risk/phase_5/train_monotonic_ebm.py
python digihealth_risk/phase_5/train_monotonic_logistic.py
```

### Step 2: Shared-cohort reevaluation and consolidated benchmark
```bash
python digihealth_risk/phase_5/evaluate_saved_models.py
python digihealth_risk/phase_5/intervention_benchmark.py
```

### Step 3: Per-patient scenario scoring (optional; replace the synthetic identifier with an authorized local patient identifier)
```bash
python digihealth_risk/phase_5/intervention_scenarios.py \
  --patient-id "SYNTHETIC-001" --horizons 1 3 5

# Batch preset simulation (100 patients, 3-year horizon)
python digihealth_risk/phase_5/intervention_scenarios.py \
  --max-patients 100 --horizons 3 --preset combined_lifestyle
```

## Key outputs

| File | Description |
|------|-------------|
| `phase_6_v2_ablation_metrics.csv` | XGBoost monotonic metrics |
| `publication_intervention_metrics.csv` | Shared-cohort artifact metrics |
| `publication_intervention_safety.csv` | Matching safety evidence and artifact/cohort hashes |
| `phase_6_v2_catboost_ablation_metrics.csv` | CatBoost ablation |
| `phase_6_v2_lightgbm_ablation_metrics.csv` | LightGBM ablation |
| `phase_6_v2_ebm_ablation_metrics.csv` | EBM ablation |
| `phase_6_v2_logistic_ablation_metrics.csv` | Logistic ablation |
| `phase_6_v2_intervention_model_summary.csv` | **Final intervention recommendation** |
| `phase_6_v2_intervention_model_report.md` | Consolidated report |
| `models_v2/` | Trained XGBoost intervention models |

## Monotonic constraint direction
- **Increasing risk (+1)**: FBS, MAX_FBS, BMI, Waist, total_sugary_week, FBS hinges, interactions
- **Decreasing risk (−1)**: total_exercise_week, total_phy_activity_week, total_veg_fruit_week

## Expected runtime
~15–30 min per family; ~2 hours total for all five
