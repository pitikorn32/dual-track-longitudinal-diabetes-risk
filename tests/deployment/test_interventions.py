import importlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer


FEATURES = ["BMI", "total_sugary_week", "total_exercise_week",
            "total_phy_activity_week", "total_veg_fruit_week"]
RANGES = {
    "BMI": {"min": 18.0, "max": 40.0},
    "total_sugary_week": {"min": 2.0, "max": 10.0},
    "total_exercise_week": {"min": 0.0, "max": 10.0},
    "total_phy_activity_week": {"min": 0.0, "max": 10.0},
    "total_veg_fruit_week": {"min": 0.0, "max": 10.0},
}


@pytest.fixture
def presets(api):
    exporter = importlib.import_module("export_models")
    train = pd.DataFrame({
        "BMI": [18.0, 25.0, 40.0], "total_sugary_week": [2.0, 3.0, 10.0],
        "total_exercise_week": [0.0, 3.0, 10.0],
        "total_phy_activity_week": [0.0, 3.0, 10.0],
        "total_veg_fruit_week": [0.0, 3.0, 10.0],
    })
    return exporter.compute_intervention_presets(train)


def modeling_row(api, **values):
    req = api.PredictRequest(
        horizon_years=1, history_years=1, age=45, year=2016,
        measurements=[{"FBS": 95, "BMI": values.pop("BMI", 25.0)}], **values,
    )
    return api._engineer_features(api.build_modeling_row(req))


@pytest.mark.parametrize("preset,feature,current,expected", [
    ("increase_exercise_to_p75", "total_exercise_week", 12.0, 12.0),
    ("increase_activity_to_p75", "total_phy_activity_week", 12.0, 12.0),
    ("increase_veg_fruit_to_p75", "total_veg_fruit_week", 12.0, 12.0),
    ("reduce_bmi_by_one", "BMI", 17.0, 17.0),
    ("reduce_sugary_to_zero", "total_sugary_week", 1.0, 1.0),
    ("reduce_sugary_50pct", "total_sugary_week", 1.0, 1.0),
    ("reduce_sugary_to_zero", "total_sugary_week", -1.0, -1.0),
    ("reduce_sugary_50pct", "total_sugary_week", -1.0, -1.0),
])
def test_clipping_never_reverses_a_favorable_change(api, presets, preset, feature, current, expected):
    row = modeling_row(api, **{feature: current})
    original = row.copy()
    adjusted, changed = api._apply_preset(row, presets[preset], RANGES)
    assert adjusted[feature].iloc[0] == expected
    assert changed[feature] == {"from": current, "to": expected}
    pd.testing.assert_frame_equal(row, original)


@pytest.mark.parametrize("preset,feature,current,expected", [
    ("increase_exercise_to_p75", "total_exercise_week", 2.0, 6.5),
    ("increase_exercise_to_p75", "total_exercise_week", 8.0, 8.0),
    ("increase_exercise_to_p75", "total_exercise_week", -1.0, 6.5),
    ("reduce_bmi_by_one", "BMI", 25.0, 24.0),
    ("reduce_bmi_by_one", "BMI", 18.0, 18.0),
    ("reduce_bmi_by_one", "BMI", 50.0, 40.0),
    ("reduce_sugary_50pct", "total_sugary_week", 6.0, 3.0),
    ("reduce_sugary_to_zero", "total_sugary_week", 6.0, 2.0),
])
def test_permitted_changes_still_use_training_targets_and_bounds(api, presets, preset,
                                                              feature, current, expected):
    adjusted, _ = api._apply_preset(modeling_row(api, **{feature: current}), presets[preset], RANGES)
    assert adjusted[feature].iloc[0] == expected


def test_missing_values_are_preserved_instead_of_inventing_observations(api, presets):
    row = modeling_row(api, BMI=None)
    for preset in presets.values():
        adjusted, changed = api._apply_preset(row, preset, RANGES)
        assert changed == {}
        pd.testing.assert_frame_equal(adjusted, row)


def test_unavailable_training_target_does_not_create_nan(api):
    row = modeling_row(api, total_exercise_week=2.0)
    adjusted, changed = api._apply_preset(row, {"max_assignments": {"total_exercise_week": np.nan}}, RANGES)
    assert changed == {}
    pd.testing.assert_frame_equal(adjusted, row)


ROUTES = [
    ("/predict/interventions", "with_year", "intervention_ebm_n1_m1"),
    ("/no_year/predict/interventions", "no_year", "intervention_ebm_n1_m1"),
    ("/logistic_only/predict/interventions", "logistic_only_with_year", "intervention_monotonic_logistic_n1_m1"),
    ("/logistic_only/no_year/predict/interventions", "logistic_only_no_year", "intervention_monotonic_logistic_n1_m1"),
]


@pytest.mark.parametrize("route,variant,key", ROUTES)
@pytest.mark.parametrize("case", ["within_range", "beyond_range", "missing", "partially_missing"])
def test_every_route_keeps_favorable_scores_nonincreasing(api, request_api, monkeypatch,
                                                        presets, route, variant, key, case):
    # A monotonic scorer with known coefficients makes a reversed update observable.
    train = pd.DataFrame([[18, 2, 0, 0, 0], [25, 3, 3, 3, 3], [40, 10, 10, 10, 10]], columns=FEATURES)
    artifact = {
        "model_key": key, "track": "intervention", "model_family": "monotonic_logistic",
        "threshold": 0.2, "feature_columns": FEATURES,
        "preprocessor": SimpleImputer(strategy="median").fit(train),
        "coefficients": np.array([-5.0, 0.1, 0.05, -0.02, -0.02, -0.02]),
        "mean_": np.zeros(5), "scale_": np.ones(5),
        "train_feature_ranges": RANGES, "intervention_presets": presets,
    }
    monkeypatch.setitem(api._VARIANT_STORES, variant, {key: artifact})
    values = {"within_range": [25, 6, 3, 3, 3], "beyond_range": [17, 1, 12, 12, 12],
              "missing": [None] * 5, "partially_missing": [17, None, 12, None, 3]}[case]
    body = dict(zip(FEATURES[1:], values[1:]))
    body.update(horizon_years=1, history_years=1, age=45, year=2016,
                measurements=[{"FBS": 95, "BMI": values[0]}], presets=list(presets))
    status, result = request_api(api.app, route, body=json.dumps(body))
    assert status == 200
    assert len(result["scenarios"]) == 7
    for scenario in result["scenarios"]:
        assert scenario["probability"] <= result["baseline"]["probability"]
        assert scenario["delta_risk_score"] <= 0
        for feature, change in scenario["changed_features"].items():
            assert change["from"] is not None
            if feature in {"BMI", "total_sugary_week"}:
                assert change["to"] <= change["from"]
            else:
                assert change["to"] >= change["from"]
    if case == "missing":
        assert all(s["changed_features"] == {} for s in result["scenarios"])
