# Phase 2: Tree-Based Models

## Thesis reference
Section 4.1.2 (Tree-Based Models), Section 5.1 (Pure Prediction Results),
Section 5.2 (Horizon and History Effects), Section 5.5.2 (Tree Feature Ablation)

## Purpose
Trains XGBoost, CatBoost, LightGBM, HistGradientBoosting, and RandomForest on
rolling patient-year tables. Includes a hybrid slope-feature branch (LMM-derived
longitudinal trends) and a full N×M horizon/history grid search.

## Scripts

| Script | Role |
|--------|------|
| `train_tree_models.py` | Train all 5 tree families (v2 features) |
| `lmm_slope_features.py` | Add LMM-shrunk slope columns to modeling table |
| `horizon_history_grid.py` | Grid search over N∈{1..5}, M∈{1,3,5} |

## Prerequisites
- `digihealth_risk/phase_0/outputs/phase_0_modeling_table.pkl` (Phase 0 Step 1)
- For grid: all `phase_0_modeling_table_horizon_{N}_history_{M}.pkl` files
- For slope branch: run `lmm_slope_features.py` first

## Step-by-step

### Step 1: Standard tree models (N=1, M=1 default)
```bash
python digihealth_risk/phase_2/train_tree_models.py
```
For a specific horizon/history table:
```bash
python digihealth_risk/phase_2/train_tree_models.py \
  --input-path digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_3_history_5.pkl
```

### Step 2: Slope-feature hybrid
```bash
python digihealth_risk/phase_2/lmm_slope_features.py
python digihealth_risk/phase_2/train_tree_models.py \
  --input-path digihealth_risk/phase_2/outputs/phase_2_v2_modeling_table_with_slopes.pkl
```

### Step 3: Horizon/history grid (XGBoost + CatBoost)
```bash
python digihealth_risk/phase_2/horizon_history_grid.py
```
Custom subset:
```bash
python digihealth_risk/phase_2/horizon_history_grid.py \
  --horizons 1 3 5 --histories 3 5 --models xgboost catboost
```

## Key outputs

| File | Description |
|------|-------------|
| `phase_2_v2_metrics.csv` | Train/test metrics for all models |
| `phase_2_v2_test_predictions.csv` | Test-set predictions (used by Phase 4) |
| `phase_2_v2_modeling_table_with_slopes.pkl` | Slope-augmented modeling table |
| `phase_2_3_grid_metrics.csv` | Grid search metrics across all N/M |

## Expected runtime
~5–15 min per table; full grid ~2–3 hours
