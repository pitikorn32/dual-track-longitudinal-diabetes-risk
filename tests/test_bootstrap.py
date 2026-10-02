import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from digihealth_risk.phase_4.bootstrap_significance import paired_cluster_bootstrap
from digihealth_risk.phase_4.bootstrap_significance import analyze
import pandas as pd


def test_cluster_bootstrap_matches_explicit_repeated_patient_rows():
    y = np.array([0, 1, 1, 0, 1, 0, 0])
    p = np.array([.1, .6, .5, .3, .9, .2, .7])
    patients = np.array(['a', 'a', 'b', 'c', 'c', 'c', 'd'])
    actual = paired_cluster_bootstrap(y, p, p, patients, replicates=40, seed=9)
    ids = np.unique(patients)
    rng = np.random.default_rng(9)
    explicit = []
    for _ in range(40):
        sampled = rng.integers(len(ids), size=len(ids))
        take = np.concatenate([np.flatnonzero(patients == ids[i]) for i in sampled])
        if len(np.unique(y[take])) < 2:
            continue
        explicit.append(average_precision_score(y[take], p[take]))
    assert actual['a_pr_auc_ci_low'] == pytest.approx(np.percentile(explicit, 2.5))
    assert actual['a_pr_auc_ci_high'] == pytest.approx(np.percentile(explicit, 97.5))
    assert actual['delta_pr_auc_ci_low'] == 0
    assert actual['delta_pr_auc_ci_high'] == 0
    assert not actual['gap_excludes_zero']
    assert actual['replicates_valid'] == len(explicit)


def test_degenerate_bootstrap_fails_explicitly():
    with pytest.raises(ValueError, match='both classes'):
        paired_cluster_bootstrap([0, 0], [.1, .2], [.2, .1], ['a', 'b'], replicates=4)


def test_default_survival_reference_preserves_original_m5_comparison():
    frames = []
    for key, approach, history, scores in [
        ('tree', 'tree', 5, [.1, .8, .2, .9]),
        ('logistic', 'statistical', 5, [.2, .7, .3, .8]),
        ('two_stage_m3', 'survival', 3, [.1, .9, .2, .8]),
        ('two_stage_m5', 'survival', 5, [.5, .4, .3, .9]),
    ]:
        frames.append(pd.DataFrame({'PatientId': ['a', 'b', 'c', 'd'], 'Year': 2005,
                      'target_year': 2006, 'horizon_years': 1, 'history_years': history,
                      'Target_AtRisk_Status': [0, 1, 0, 1], 'predicted_probability': scores,
                      'model_key': key, 'model_family': key, 'model_name': key,
                      'approach': approach, 'calibration_method': 'raw'}))
    predictions = pd.concat(frames, ignore_index=True)
    original = analyze(predictions, replicates=5)
    sensitivity = analyze(predictions, replicates=5, survival_reference='best')
    assert original.iloc[1].b_model_key == 'two_stage_m5'
    assert sensitivity.iloc[1].b_model_key == 'two_stage_m3'
