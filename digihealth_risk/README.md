# Research pipeline

HealthCom is the primary workflow; the extended thesis supplies supplementary
analyses. Start with the [repository README](../README.md) and the authoritative
[publication methods and command mapping](../docs/PUBLICATION.md).

## Phase structure

| Phase | Role | Principal entry points |
| --- | --- | --- |
| 0 | Cumulative labels, rolling tables, cohort accounting | `build_modeling_tables.py`, `build_cohort_figure.py`; supplementary `eda_depth.py` |
| 1 | Statistical research models | GEE/logistic horizon grids and GPBoost GLMM comparator |
| 2 | Tree and shrunk-slope comparisons | `train_tree_models.py`, `lmm_slope_features.py`, `horizon_history_grid.py` |
| 3 | Survival comparators | `landmark_cox.py`, `two_stage_survival.py` |
| 4 | Calibration, shared ranking, uncertainty, interpretation | `calibrate_trees.py`, `threshold_optimization.py`, `cross_family_comparison.py`, `bootstrap_significance.py` |
| 5 | Monotonic fits and exact-artifact evaluation | Five trainers, `evaluate_saved_models.py`, `intervention_benchmark.py` |
| 6 | Optional serving export | `export_models.py`; serving fits have their own scope |
| 7 | Calendar-time ablation | No-year trainers and `compare_with_baseline.py` |
| `publication` | Public smoke example and evidence generation | `smoke`, `feature_ablation`, `feature_importance`, `report` modules |

Run every command from the submodule root. The HealthCom profile runs phases
0–5 and 7 plus publication reporting; `--profile thesis` adds thesis supplements.
Deployment is opt-in. All phases reuse the canonical patient assignment in
`utils/patient_split.py`; the shared evaluation contract is in `utils/evaluation.py`.

```bash
bash reproduce.sh --profile healthcom --dry-run
bash reproduce.sh --profile healthcom --fail-fast
bash reproduce.sh --profile thesis --fail-fast
```

The private input schema is documented in `docs/PUBLICATION.md`. Generated
modeling tables, predictions, models, reports, and split caches stay in ignored
`outputs/` directories. Numeric plotting scripts now produce research figures;
authored manuscript schematics remain separate assets. Older phase filenames
containing `phase_6` or `v2` are retained for compatibility and do not imply that
those outputs are deployment artifacts.
