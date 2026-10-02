"""Paired N=1, M=1 ablation of explicitly listed engineered features.

This regenerates the feature-engineering comparison using the current fitting
code. It is not a recovered historical v1 implementation. Both variants use
the same patient split, missingness indicators, imputation, and base predictors.
"""
from contextlib import contextmanager
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from digihealth_risk.phase_1 import logistic, gee
from digihealth_risk.phase_2.train_tree_models import engineer_features, get_feature_columns
from digihealth_risk.phase_4.calibrate_trees import fit_pipeline, predict_probability
from digihealth_risk.utils.patient_split import apply_canonical_split
from digihealth_risk.utils.evaluation import ranking_metrics

ENGINEERED = {'Year_centered_sq', 'FBS_hinge_100', 'FBS_hinge_125',
              'FBS_x_Age', 'BMI_x_Age', 'MAX_FBS_x_Age'}


@contextmanager
def feature_selection(module, attribute, variant):
    original = getattr(module, attribute)
    if variant == 'without_engineered_terms':
        setattr(module, attribute, [f for f in original if f not in ENGINEERED])
    try:
        yield
    finally:
        setattr(module, attribute, original)


def main():
    results = []
    for family in ['logistic', 'gee', 'xgboost', 'catboost']:
        for variant in ['without_engineered_terms', 'enriched']:
            table = (gee if family == 'gee' else logistic).load_data(logistic.INPUT_PATH)
            if family in ['xgboost', 'catboost']:
                table = engineer_features(pd.read_pickle(logistic.INPUT_PATH))
                if variant == 'without_engineered_terms':
                    table = table.drop(columns=list(ENGINEERED), errors='ignore')
            train, test = apply_canonical_split(table)
            if family == 'logistic':
                with feature_selection(logistic, 'CONTINUOUS_FEATURES', variant):
                    prep = logistic.fit_preprocessor(train)
                    p = logistic.fit_logistic(train, test, prep).test_probability
            elif family == 'gee':
                with feature_selection(gee, 'BASE_CONTINUOUS_FEATURES', variant):
                    prepared_train, prepared_test, names = gee.prepare_features(train, test)
                fitted = smf.gee(gee.build_formula(names), groups='PatientId', data=prepared_train,
                                 family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=100)
                p = fitted.predict(prepared_test)
            else:
                numeric, categorical = get_feature_columns(train)
                fitted = fit_pipeline(family, train, numeric, categorical)
                p = predict_probability(fitted, test, numeric, categorical)
            results.append({'family': family, 'variant': variant, 'horizon_years': 1, 'history_years': 1,
                            'protocol': 'paired_current_feature_set_ablation',
                            **ranking_metrics(test.Target_AtRisk_Status, p)})
            print(f'Feature ablation completed: {family}, {variant}', flush=True)
    output = logistic.ROOT / 'digihealth_risk/phase_4/outputs/publication_feature_ablation.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output, index=False)


if __name__ == '__main__':
    main()
