import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from digihealth_risk.phase_1 import logistic
from digihealth_risk.phase_4 import calibrate_trees


def test_logistic_objective_sums_loss_with_fixed_ridge_and_free_intercept():
    x = np.array([[1.0, -1.0], [1.0, 2.0], [1.0, 0.5]])
    y = np.array([0.0, 1.0, 0.0])
    beta = np.array([0.4, -0.7])
    p = expit(x @ beta)
    penalty = 0.5 * 0.01 * beta[1] ** 2
    expected = -np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)) + penalty
    loss, gradient = logistic.negative_log_likelihood(beta, x, y)
    assert loss == pytest.approx(expected)
    np.testing.assert_allclose(gradient, x.T @ (p - y) + [0, 0.01 * beta[1]])
    doubled, _ = logistic.negative_log_likelihood(beta, np.tile(x, (2, 1)), np.tile(y, 2))
    assert doubled == pytest.approx(2 * loss - penalty)
    assert logistic.LOSS_REDUCTION == 'sum'
    assert logistic.PENALIZE_INTERCEPT is False


@pytest.mark.parametrize('family', ['xgboost', 'catboost'])
def test_screening_fit_stays_unweighted_and_records_actual_training(family, monkeypatch):
    original = calibrate_trees.build_model

    def small_model(name, ratio, use_class_weights):
        model = original(name, ratio, use_class_weights)
        model.set_params(**({'n_estimators': 2, 'n_jobs': 1} if name == 'xgboost'
                            else {'iterations': 2, 'thread_count': 1}))
        return model

    monkeypatch.setattr(calibrate_trees, 'build_model', small_model)
    train = pd.DataFrame({'x': np.arange(20, dtype=float),
                          'Target_AtRisk_Status': [0] * 18 + [1] * 2})
    pipeline = calibrate_trees.fit_pipeline(family, train, ['x'], [])
    params = pipeline.named_steps['model'].get_params()
    if family == 'xgboost':
        assert params['scale_pos_weight'] == 1.0  # The observed class ratio is 9.
    else:
        assert params.get('class_weights') is None
    metadata = pipeline.training_metadata_
    assert metadata['train_rows'] == 20
    assert metadata['settings']['class_weighting_enabled'] is False
    assert metadata['settings']['scale_pos_weight'] == params.get('scale_pos_weight')
    assert metadata['settings']['class_weights'] == params.get('class_weights')
    assert metadata['packages'][family]
    json.dumps(metadata, allow_nan=False)


def test_logistic_fit_records_loss_and_penalty(monkeypatch):
    train = pd.DataFrame({
        'PatientId': np.repeat([f'SYNTHETIC-{i}' for i in range(20)], 3),
        'x': np.sin(np.arange(60)),
        'gender': ['F', 'M'] * 30,
        'has_fbs_this_year': np.arange(60) % 2,
        'is_missing_last_year': np.arange(60) % 3 == 0,
        'Target_AtRisk_Status': np.arange(60) % 5 == 0,
    })
    prep = logistic.Preprocessor(
        ['x'], pd.Series({'x': 0.0}), pd.Series({'x': 0.0}), pd.Series({'x': 1.0}),
        ['gender_M'], ['intercept', 'x', 'has_fbs_this_year', 'is_missing_last_year', 'gender_M'],
    )
    # This fixture tests fitting/provenance independently of the clinical schema.
    monkeypatch.setattr(logistic, 'CATEGORICAL_FEATURES', ['gender'])
    monkeypatch.setattr(logistic, 'MISSING_INDICATOR_FEATURES', [])
    fit = logistic.fit_logistic(train, train.iloc[:8], prep)
    metadata = fit.training_metadata
    assert metadata['train_rows'] == 60
    assert metadata['settings'] == {
        'class_weighting_enabled': False, 'loss_reduction': 'sum',
        'ridge_alpha': 0.01, 'penalize_intercept': False,
    }
    assert metadata['packages']['scipy']
    json.dumps(metadata, allow_nan=False)


def test_manifest_distinguishes_recorded_fits_from_legacy_metrics(tmp_path):
    from digihealth_risk.utils.training_provenance import collect_training_metadata
    phase1 = tmp_path / 'digihealth_risk/phase_1/outputs'
    phase4 = tmp_path / 'digihealth_risk/phase_4/outputs'
    phase1.mkdir(parents=True)
    phase4.mkdir(parents=True)
    record = {'model_family': 'logistic', 'train_rows': 60,
              'settings': {'loss_reduction': 'sum'}, 'python': 'test',
              'packages': {'scipy': 'recorded-fit-version'}}
    path = phase1 / 'phase_1_v2_logistic_horizon_1_history_1_metrics.csv'
    pd.DataFrame({'split': ['train', 'test'], 'training_metadata': [json.dumps(record)] * 2}).to_csv(path, index=False)
    legacy = phase4 / 'phase_4_v2_metrics.csv'
    pd.DataFrame({'model_key': ['old-model'], 'pr_auc': [0.2]}).to_csv(legacy, index=False)
    result = collect_training_metadata(tmp_path)
    assert len(result['recorded']) == 1
    assert result['recorded'][0]['fit'] == record
    assert len(result['recorded'][0]['source_sha256']) == 64
    assert result['unrecorded_sources'] == [str(legacy.relative_to(tmp_path))]
    # Mixed old/new files must not imply that all rows have provenance.
    pd.DataFrame({'training_metadata': [json.dumps(record), None]}).to_csv(path, index=False)
    result = collect_training_metadata(tmp_path)
    assert str(path.relative_to(tmp_path)) in result['unrecorded_sources']


def test_logistic_entry_point_writes_fit_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from digihealth_risk.utils.patient_split import apply_canonical_split

    frame = pd.DataFrame({
        'PatientId': np.repeat([f'SYNTHETIC-{i:02}' for i in range(30)], 3),
        'Year': np.tile([2005, 2006, 2007], 30),
        'target_year': np.tile([2006, 2007, 2008], 30),
        'x': np.sin(np.arange(90)), 'gender': ['F', 'M'] * 45,
        'has_fbs_this_year': np.arange(90) % 2,
        'is_missing_last_year': np.arange(90) % 3 == 0,
        'Target_AtRisk_Status': np.arange(90) % 5 == 0,
    })
    source = tmp_path / 'synthetic.pkl'
    frame[['PatientId']].drop_duplicates().to_pickle(source)
    monkeypatch.setattr(logistic, 'load_data', lambda _: frame)
    monkeypatch.setattr(logistic, 'split_by_patient', lambda df: apply_canonical_split(
        df, source_path=source, cache_path=tmp_path / 'split.csv'))
    monkeypatch.setattr(logistic, 'CONTINUOUS_FEATURES', ['x'])
    monkeypatch.setattr(logistic, 'QUESTIONNAIRE_NUMERIC', [])
    monkeypatch.setattr(logistic, 'CATEGORICAL_FEATURES', ['gender'])
    monkeypatch.setattr(logistic, 'MISSING_INDICATOR_FEATURES', [])
    monkeypatch.setattr(logistic, 'OUT_DIR', tmp_path)
    monkeypatch.setattr(logistic, 'parse_args', lambda: SimpleNamespace(
        input_path=source, output_prefix='synthetic_logistic'))
    logistic.main()
    metrics = pd.read_csv(tmp_path / 'synthetic_logistic_metrics.csv')
    records = metrics.training_metadata.map(json.loads)
    assert records.iloc[0] == records.iloc[1]
    assert records.iloc[0]['train_rows'] == metrics.loc[metrics.split.eq('train'), 'rows'].iloc[0]
    assert records.iloc[0]['settings']['loss_reduction'] == 'sum'
    assert (tmp_path / 'synthetic_logistic_report.md').exists()


def test_tree_configuration_exports_same_fit_metadata_for_each_calibrator(tmp_path, monkeypatch):
    from digihealth_risk.utils.patient_split import apply_canonical_split

    frame = pd.DataFrame({
        'PatientId': np.repeat([f'SYNTHETIC-{i:02}' for i in range(30)], 3),
        'Year': np.tile([2005, 2006, 2007], 30),
        'target_year': np.tile([2006, 2007, 2008], 30),
        'x': np.sin(np.arange(90)),
        'Target_AtRisk_Status': np.arange(90) % 5 == 0,
    })
    source = tmp_path / 'synthetic.pkl'
    frame[['PatientId']].drop_duplicates().to_pickle(source)
    monkeypatch.setattr(calibrate_trees, 'ROOT', tmp_path)
    monkeypatch.setattr(calibrate_trees, 'load_table', lambda _: frame)
    monkeypatch.setattr(calibrate_trees, 'engineer_features', lambda df: df)
    monkeypatch.setattr(calibrate_trees, 'grouped_train_cal_test_split', lambda df: apply_canonical_split(
        df, return_calibration=True, source_path=source, cache_path=tmp_path / 'split.csv'))
    original = calibrate_trees.build_model
    monkeypatch.setattr(calibrate_trees, 'build_model', lambda *args, **kwargs:
                        original(*args, **kwargs).set_params(n_estimators=2, n_jobs=1))
    config = calibrate_trees.ModelConfig('synthetic_xgboost', source, 'xgboost', 1, 1, 'candidate')
    metrics, _, _, _ = calibrate_trees.run_config(config)
    assert set(metrics.calibration_method) == {'raw', 'platt', 'isotonic'}
    assert metrics.training_metadata.nunique() == 1
    record = json.loads(metrics.training_metadata.iloc[0])
    assert record['train_rows'] == metrics.train_rows.iloc[0]
    assert record['model_key'] == config.key
    assert record['settings']['scale_pos_weight'] == 1.0
