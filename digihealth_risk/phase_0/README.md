# Phase 0 — Data Engineering

## Thesis reference
Section 3 (Dataset and Problem Formulation) and Section 4.4.1 (Temporal Grid)

## Purpose
Pivots the raw 121-column wide-format clinical dataset into rolling patient-year
modeling tables. Each table encodes a specific prediction horizon N and history
window M. Downstream phases read these tables as their primary input.

## Prerequisites
- `datasets/df_final.pkl` must exist (5.6 MB, 6,892 patients, 121 columns)
- Run from the **repository root** (all commands below assume this)

## Step-by-step

### Step 1 — Default table (N=1, M=1)
```bash
python digihealth_risk/phase_0/build_modeling_tables.py
```
Writes: `digihealth_risk/phase_0/outputs/phase_0_modeling_table.pkl`

### Step 2 — Full N×M grid (all 15 combinations)
```bash
for N in 1 2 3 4 5; do
  for M in 1 3 5; do
    python digihealth_risk/phase_0/build_modeling_tables.py \
      --horizon-years $N --history-years $M
  done
done
```
Writes: `digihealth_risk/phase_0/outputs/phase_0_modeling_table_horizon_{N}_history_{M}.pkl`

Also produces: `patient_year_long.pkl`, `phase_0_eda_report.md`, CSV samples.

## Key outputs

| File | Description |
|------|-------------|
| `phase_0_modeling_table.pkl` | Default rolling table (N=1, M=1), ~42k rows |
| `phase_0_modeling_table_horizon_{N}_history_{M}.pkl` | Grid variant |
| `patient_year_long.pkl` | Long-format censored patient-year panel |
| `patient_split.csv` | Canonical 60/20/20 patient split (auto-generated) |

## Expected runtime
~2 min per table on a standard laptop; ~30 min for the full 15-table grid.

## Extended depth EDA (optional)

`eda_depth.py` is an optional analysis script, not part of the build path:
nothing downstream consumes its outputs. It produces the statistical evidence
behind the v2 feature-engineering choices, supporting thesis Section 3.4.2.

### Run

```bash
python digihealth_risk/phase_0/eda_depth.py
```

Requires Step 1 above (`patient_year_long.pkl` and `phase_0_modeling_table.pkl`).

| File | Key finding |
|------|-------------|
| `phase_0_2_vif.csv` | VIF=225 for `pulse_pressure`, removed in v2 |
| `phase_0_2_ljung_box.csv` | Ljung-Box p=0.03 for `Year_centered`, `Year_centered_sq` added |
| `phase_0_2_cross_lagged_correlation.csv` | `MAX_FBS_x_Age` cross-lag r=0.582, interaction added |
| `phase_0_2_report.md` | Full summary |

Runtime: roughly 5-10 min.
