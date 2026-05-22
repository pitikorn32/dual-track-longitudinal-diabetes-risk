# Phase 0 Appendix — Extended Depth EDA

## Thesis reference
Supporting Section 3.4.2 (nonlinear interaction feature motivation); not a
direct result table but produces the statistical evidence for v2 feature choices.

## Purpose
Deep univariate, temporal-autocorrelation, cross-lag correlation, and
multi-collinearity analysis. Results directly motivated the v2 feature
engineering applied across phases 1–6 (e.g., VIF for pulse_pressure, Ljung-Box
for Year_centered_sq, cross-lag r=0.582 for MAX_FBS×Age).

## Prerequisites
- `digihealth_risk/phase_0/outputs/patient_year_long.pkl` (Step 1 of Phase 0)
- `digihealth_risk/phase_0/outputs/phase_0_modeling_table.pkl` (Step 1 of Phase 0)

## Step-by-step

### Step 1 — Run depth EDA
```bash
python digihealth_risk/phase_0_appendix/eda_depth.py
```

## Key outputs

| File | Key finding |
|------|-------------|
| `phase_0_2_vif.csv` | VIF=225 for pulse_pressure → removed in v2 |
| `phase_0_2_ljung_box.csv` | Ljung-Box p=0.03 for Year_centered → Year_centered_sq added |
| `phase_0_2_cross_lagged_correlation.csv` | MAX_FBS×Age r=0.582 → interaction feature added |
| `phase_0_2_report.md` | Full summary |

## Expected runtime
~5–10 min
