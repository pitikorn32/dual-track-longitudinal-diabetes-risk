"""Importance from research-model fits and statistical coefficient tables.

Trees are refitted with the phase-4 60% training protocol; their fitted-object
importance is descriptive and is not an explanation of a frozen historic fit.
Linear models use saved research coefficient tables. No serving model is used.
"""
import pandas as pd

from digihealth_risk.phase_4.cross_family_comparison import load_shared_predictions, ranking_table
from digihealth_risk.phase_4.calibrate_trees import (
    phase0_path, load_table, engineer_features, grouped_train_cal_test_split,
    get_feature_columns, fit_pipeline,
)
from digihealth_risk.phase_4.feature_importance_analysis import raw_importance, per_base_shares, stat_base, ROOT


def main():
    shared, _ = load_shared_predictions()
    ranking = ranking_table(shared)
    winners = ranking.sort_values(['horizon_years', 'pr_auc', 'roc_auc', 'brier'], ascending=[True, False, False, True]).groupby('horizon_years').head(1)
    rows = []
    for winner in winners.itertuples():
        n, m, family = int(winner.horizon_years), int(winner.history_years), winner.model_family
        if winner.approach == 'tree':
            table = engineer_features(load_table(phase0_path(n, m)))
            train, _, _ = grouped_train_cal_test_split(table)
            numeric, categorical = get_feature_columns(train)
            model = fit_pipeline(family, train, numeric, categorical)
            artifact = {'model_family': family, 'model': model['model'], 'categorical_features': categorical,
                        'transformed_feature_names': list(model['preprocessor'].get_feature_names_out())}
            shares = per_base_shares(artifact, raw_importance(artifact))
            provenance = 'research_protocol_refit'
        elif winner.approach == 'statistical':
            path = ROOT / f'digihealth_risk/phase_1/outputs/phase_1_v2_{family}_horizon_{n}_history_{m}_coefficients.csv'
            coefficients = pd.read_csv(path)
            coefficients = coefficients[~coefficients.feature.str.lower().eq('intercept')]
            shares = coefficients.assign(base=coefficients.feature.map(stat_base), value=coefficients.coefficient.abs()).groupby('base').value.sum()
            shares = shares / shares.sum()
            provenance = 'saved_research_coefficients'
        else:
            raise ValueError('Research importance for a survival winner needs an explicit method')
        for feature, share in shares.items():
            rows.append({'horizon_years': n, 'history_years': m, 'family': family, 'model_key': winner.model_key,
                         'calibration_method': winner.calibration_method, 'feature': feature,
                         'importance_share': float(share), 'provenance': provenance})
        print(f'Research importance completed: N={n}, {family}', flush=True)
    output = ROOT / 'digihealth_risk/phase_4/outputs/publication_feature_importance.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == '__main__':
    main()
