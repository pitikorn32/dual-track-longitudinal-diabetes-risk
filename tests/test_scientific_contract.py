import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from digihealth_risk.phase_0.build_modeling_tables import (
    YEARS, CLINICAL_FEATURES, STATIC_FEATURES, build_long_table,
    build_modeling_table, status_from_max_fbs,
)
from digihealth_risk.phase_1.logistic import auc_pr, auc_roc
from digihealth_risk.phase_4.cross_family_comparison import align_shared_cohort
from digihealth_risk.utils import patient_split


def test_study_thresholds_and_missing_values():
    values = pd.Series([np.nan, 99, 100, 100.5, 119, 120, 125, 125.5, 126])
    actual = status_from_max_fbs(values)
    assert pd.isna(actual.iloc[0])
    assert actual.iloc[1:].tolist() == [
        'non_dm', 'non_dm', 'pre_dm', 'pre_dm', 'pre_dm', 'pre_dm', 'dm', 'dm',
    ]


def test_average_precision_and_tied_roc():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.3, 0.3, 0.8])
    assert auc_pr(y, p) == pytest.approx(average_precision_score(y, p))
    assert auc_roc(y, np.full(4, 0.5)) == pytest.approx(0.5)
    assert np.isnan(auc_roc(np.zeros(4), p))


def test_canonical_split_is_patient_grouped_and_order_independent(tmp_path, monkeypatch):
    source = tmp_path / 'source.pkl'
    cache = tmp_path / 'split.csv'
    ids = [f'synthetic-{i:03}' for i in range(100)]
    pd.DataFrame({'PatientId': ids}).to_pickle(source)
    monkeypatch.setattr(patient_split, 'SOURCE_DATA', source)
    monkeypatch.setattr(patient_split, 'SPLIT_CACHE', cache)
    data = pd.DataFrame({'PatientId': ids * 2})
    train, cal, test = patient_split.apply_canonical_split(data, return_calibration=True)
    groups = [set(frame.PatientId) for frame in (train, cal, test)]
    assert list(map(len, groups)) == [60, 20, 20]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    original = patient_split.load_canonical_split()
    pd.DataFrame({'PatientId': ids[::-1]}).to_pickle(source)
    pd.testing.assert_frame_equal(original, patient_split.load_canonical_split(rebuild=True))
    folded, same_test = patient_split.apply_canonical_split(data)
    assert set(folded.PatientId) == groups[0] | groups[1]
    assert set(same_test.PatientId) == groups[2]


def prediction_rows():
    return pd.DataFrame({
        'PatientId': ['synthetic-a', 'synthetic-b'] * 2,
        'Year': [2005] * 4, 'target_year': [2006] * 4,
        'horizon_years': [1] * 4, 'model_key': ['a', 'a', 'b', 'b'],
        'calibration_method': ['raw'] * 4,
        'Target_AtRisk_Status': [0, 1, 0, 1],
        'predicted_probability': [.2, .7, .3, .8],
    })


def test_shared_cohort_rejects_duplicate_predictions():
    df = prediction_rows()
    with pytest.raises(ValueError, match='Duplicate'):
        align_shared_cohort(pd.concat([df, df.iloc[[0]]]))


def test_shared_cohort_rejects_conflicting_targets():
    df = prediction_rows()
    df.loc[2, 'Target_AtRisk_Status'] = 1
    with pytest.raises(ValueError, match='target'):
        align_shared_cohort(df)


def test_shared_cohort_intersects_keys_not_row_positions():
    df = prediction_rows().drop(index=3).sample(frac=1, random_state=2)
    aligned, summary = align_shared_cohort(df)
    assert set(aligned.PatientId) == {'synthetic-a'}
    assert len(aligned) == 2
    assert summary.shared_rows.tolist() == [1]


def test_publication_comparison_rejects_incomplete_model_grid(monkeypatch):
    from digihealth_risk.phase_4 import cross_family_comparison as comparison
    monkeypatch.setattr(comparison, 'load_phase4_trees', lambda: [prediction_rows()])
    for loader in ['load_phase1_gee', 'load_phase1_logistic_v2', 'load_phase3_2_landmark_cox', 'load_phase3_3_two_stage']:
        monkeypatch.setattr(comparison, loader, lambda: [])
    with pytest.raises(ValueError, match='28 model configurations'):
        comparison.load_shared_predictions()


def test_history_and_cumulative_predictors_ignore_future_readings():
    row = {name: 0 for name in STATIC_FEATURES}
    row.update(PatientId='synthetic-a', date_of_birth=pd.Timestamp('1970-01-01'))
    for year in YEARS:
        for feature in CLINICAL_FEATURES:
            row[f'{feature}_{year}'] = 90.0 if feature == 'FBS' else 25.0
        row[f'MAX_FBS_up_to_{year}'] = 90.0
        row[f'DM_status_up_to_{year}'] = 'non_dm'
        row[f'AtRisk_{year}'] = 0.0
    original = pd.DataFrame([row])
    changed = original.copy()
    for year in range(2008, 2017):
        changed[f'FBS_{year}'] = 140.0
        changed[f'MAX_FBS_up_to_{year}'] = 140.0
        changed[f'DM_status_up_to_{year}'] = 'dm'
        changed[f'AtRisk_{year}'] = 1.0
    a = build_modeling_table(build_long_table(original), original, 1, 3)
    b = build_modeling_table(build_long_table(changed), changed, 1, 3)
    predictors = ['FBS', 'MAX_FBS_up_to_year', *[c for c in a if '_hist_' in c]]
    pd.testing.assert_frame_equal(
        a.loc[a.Year.eq(2007), predictors], b.loc[b.Year.eq(2007), predictors],
    )
    assert b.Year.max() == 2007
    assert b.loc[b.Year.eq(2007), 'Target_AtRisk_Status'].item() == 1
