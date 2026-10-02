import numpy as np
import pandas as pd
import pytest

from digihealth_risk.phase_2.train_tree_models import engineer_features
from digihealth_risk.phase_5 import monotonic_ablation_utils as scenarios
from digihealth_risk.phase_5.intervention_benchmark import attach_safety


def test_scenario_recomputes_glucose_derivatives_without_recentering_year():
    row = pd.DataFrame({'Year': [2010], 'Year_centered': [5], 'Age': [50.],
                        'FBS': [90.], 'MAX_FBS_up_to_year': [95.]})
    original = engineer_features(row)
    original['FBS'] = 110.
    changed = engineer_features(original)
    assert changed.FBS_hinge_100.item() == 10
    assert changed.FBS_x_Age.item() == 5500
    assert changed.Year_centered.item() == 5


@pytest.mark.parametrize('preset,feature,values', [
    ('reduce_sugary_50', 'total_sugary_week', [-1., 0., 4., np.nan]),
    ('reduce_sugary_zero', 'total_sugary_week', [-1., 0., 4., np.nan]),
    ('bmi_minus_one', 'BMI', [17., 18., 25., np.nan]),
])
def test_favorable_clipping_never_increases_risk_increasing_input(preset, feature, values):
    df = pd.DataFrame({feature: values})
    lower = 18. if feature == 'BMI' else 1.
    artifact = {'train_feature_ranges': {feature: {'min': lower, 'max': 40.}}}
    adjusted = scenarios.PRESET_REGISTRY[preset](df, artifact, df)
    observed = df[feature].notna()
    assert (adjusted.loc[observed, feature] <= df.loc[observed, feature]).all()
    assert pd.isna(adjusted[feature].iloc[-1])


def test_all_rows_and_all_presets_are_checked_in_score_units(monkeypatch):
    size = 5001
    df = pd.DataFrame({'Year': [2005] * size, 'Age': [50.] * size,
                       'FBS': [90.] * size, 'MAX_FBS_up_to_year': [90.] * size,
                       'BMI': [25.] * size, 'total_sugary_week': [4.] * size})
    def predict(artifact, data):
        return data.total_sugary_week.to_numpy() / 10
    monkeypatch.setattr(scenarios, 'predict_probability', predict)
    result = scenarios.scenario_summary(
        {'train_feature_ranges': {'total_sugary_week': {'min': 0, 'max': 10}}},
        train_df=df, test_df=df, horizon=1, history_years=5, variant='monotonic',
    )
    assert set(result.scenario) == set(scenarios.PRESET_REGISTRY)
    assert result.rows.eq(size).all()
    assert result.unexpected_increase_rows.eq(0).all()
    assert result.loc[result.scenario.eq('reduce_sugary_50'), 'mean_delta_score'].item() == pytest.approx(-20)


@pytest.mark.parametrize('field,wrong', [('history_years', 3), ('artifact_sha256', 'other-fit'),
                                        ('cohort_sha256', 'other-rows'), ('scenario_suite', 'old-suite')])
def test_safety_must_describe_the_exact_prediction_configuration(field, wrong):
    identity = {'family': 'ebm', 'horizon_years': 1, 'history_years': 5,
                'model_key': 'ebm-n1-m5', 'artifact_sha256': 'fit',
                'cohort_sha256': 'rows', 'scenario_suite': 'suite',
                'tolerance_score_points': 1e-10}
    prediction = pd.DataFrame([identity])
    safety = pd.DataFrame([{**identity, field: wrong, 'unexpected_increase_rate': 0.}])
    with pytest.raises(ValueError, match='exact prediction'):
        attach_safety(prediction, safety)


def test_safety_selection_follows_prediction_history_not_best_safety():
    identity = {'family': 'ebm', 'horizon_years': 1, 'history_years': 5,
                'model_key': 'ebm-n1-m5', 'artifact_sha256': 'fit',
                'cohort_sha256': 'rows', 'scenario_suite': 'suite',
                'tolerance_score_points': 1e-10}
    safety = pd.DataFrame([
        {**identity, 'unexpected_increase_rate': .1},
        {**identity, 'history_years': 3, 'unexpected_increase_rate': 0.},
    ])
    result = attach_safety(pd.DataFrame([identity]), safety)
    assert result.unexpected_increase_rate.item() == .1
