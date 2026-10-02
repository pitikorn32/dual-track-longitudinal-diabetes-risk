"""Run real engineering, splitting, fitting, and safety checks on synthetic data."""
import json
import tempfile
from pathlib import Path

from digihealth_risk.publication.synthetic import make_cohort
from digihealth_risk.phase_0.build_modeling_tables import build_long_table, build_modeling_table
from digihealth_risk.phase_2.train_tree_models import engineer_features
from digihealth_risk.phase_5.train_monotonic_xgboost import fit_monotonic_model
from digihealth_risk.phase_5.monotonic_ablation_utils import predict_probability, scenario_summary
from digihealth_risk.utils.patient_split import apply_canonical_split
from digihealth_risk.utils.evaluation import ranking_metrics


def run():
    cohort = make_cohort()
    table = engineer_features(build_modeling_table(build_long_table(cohort), cohort, 1, 5))
    with tempfile.TemporaryDirectory(prefix='digihealth-synthetic-') as directory:
        source = Path(directory) / 'synthetic.pkl'
        cohort.to_pickle(source)
        train, cal, test = apply_canonical_split(table, return_calibration=True,
                                               source_path=source, cache_path=Path(directory) / 'split.csv')
        artifact = fit_monotonic_model(train, n_jobs=1)
        artifact['train_feature_ranges'] = {
            column: {'min': float(train[column].min()), 'max': float(train[column].max())}
            for column in artifact['numeric_features']
        }
        probability = predict_probability(artifact, test)
        safety = scenario_summary(artifact, train_df=train, test_df=test, horizon=1, history_years=5, variant='monotonic')
        if safety.unexpected_increase_rows.sum():
            raise AssertionError('Synthetic directional checks failed')
        return {'data': 'wholly synthetic; not publication evidence',
                'train_patients': train.PatientId.nunique(), 'calibration_patients': cal.PatientId.nunique(),
                'test_patients': test.PatientId.nunique(), 'metrics': ranking_metrics(test.Target_AtRisk_Status, probability),
                'scenarios': len(safety), 'direction_checks': int(safety.rows.sum()), 'unexpected_increases': 0}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
