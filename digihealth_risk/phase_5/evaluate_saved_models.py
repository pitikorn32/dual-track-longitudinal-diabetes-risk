"""Evaluate saved intervention artifacts on the publication's exact shared cohort.

No training is performed. All configurations are evaluated before retrospective
ranking, and safety evidence carries the artifact hash and cohort fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_2.train_tree_models import engineer_features, load_table, classification_metrics
from digihealth_risk.phase_4.cross_family_comparison import load_shared_predictions
from digihealth_risk.phase_5.monotonic_ablation_utils import (
    OUT_DIR, HORIZONS, HISTORY_OPTIONS, phase0_path, predict_probability,
    scenario_summary, aggregate_safety, SCENARIO_SUITE, TOLERANCE,
)
from digihealth_risk.utils.patient_split import apply_canonical_split
from digihealth_risk.utils.evaluation import KEY_COLUMNS, TARGET, align_to_reference

METRICS_PATH = OUT_DIR / 'publication_intervention_metrics.csv'
SAFETY_PATH = OUT_DIR / 'publication_intervention_safety.csv'


def cohort_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame[KEY_COLUMNS + [TARGET]].sort_values(KEY_COLUMNS)
    return hashlib.sha256(ordered.to_csv(index=False).encode()).hexdigest()


def evaluate(horizons=HORIZONS):
    shared, _ = load_shared_predictions()
    metric_rows, safety_parts, scenario_parts = [], [], []
    for family in ['xgboost', 'lightgbm', 'catboost', 'ebm', 'logistic']:
        histories = [5] if family == 'xgboost' else HISTORY_OPTIONS
        model_dir = OUT_DIR / ('models_v2' if family == 'xgboost' else f'models_v2_{family}_ablation')
        for horizon in horizons:
            reference = shared.loc[shared.horizon_years.eq(horizon), KEY_COLUMNS + [TARGET]].drop_duplicates()
            for history in histories:
                key = f'phase6_v2_monotonic_{family}_n{horizon}_m{history}'
                path = model_dir / f'{key}.joblib'
                if not path.exists():
                    raise FileNotFoundError(f'Train phase 5 first; missing artifact {path.name}')
                artifact = joblib.load(path)
                for field, expected in [('horizon_years', horizon), ('history_years', history), ('model_key', key)]:
                    if artifact.get(field) != expected:
                        raise ValueError(f'Artifact metadata disagrees with its configuration: {key}, {field}')
                table = engineer_features(load_table(phase0_path(horizon, history)))
                train, test = apply_canonical_split(table)
                # Match labels and keys before selecting rows; no silent row dropping.
                identity = test[KEY_COLUMNS + [TARGET]].assign(predicted_probability=0.)
                aligned = align_to_reference(identity, reference)
                test = aligned[KEY_COLUMNS].merge(test, on=KEY_COLUMNS, validate='one_to_one')
                probabilities = predict_probability(artifact, test)
                metadata = {
                    'family': family, 'variant': 'monotonic', 'horizon_years': horizon,
                    'history_years': history, 'model_key': key,
                    'artifact_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'cohort_sha256': cohort_fingerprint(reference),
                    'scenario_suite': SCENARIO_SUITE, 'tolerance_score_points': TOLERANCE,
                }
                metric_rows.append({**metadata, 'split': 'test', 'model_name': f'monotonic_{family}',
                                    **classification_metrics(test[TARGET].to_numpy(), probabilities, artifact['threshold'])})
                scenarios = scenario_summary(artifact, train_df=train, test_df=test,
                                             horizon=horizon, history_years=history, variant='monotonic')
                safety = aggregate_safety(scenarios)
                for field, value in metadata.items():
                    safety[field] = value
                    scenarios[field] = value
                safety_parts.append(safety)
                scenario_parts.append(scenarios)
                print(f'Evaluated {key}: {len(test)} rows, {len(scenarios)} scenarios', flush=True)
    return pd.DataFrame(metric_rows), pd.concat(safety_parts, ignore_index=True), pd.concat(scenario_parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizons', nargs='+', type=int, choices=HORIZONS, default=HORIZONS)
    parser.add_argument('--output-dir', type=Path, default=OUT_DIR)
    args = parser.parse_args()
    metrics, safety, scenarios = evaluate(args.horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / METRICS_PATH.name, index=False)
    safety.to_csv(args.output_dir / SAFETY_PATH.name, index=False)
    scenarios.to_csv(args.output_dir / 'publication_intervention_scenarios.csv', index=False)


if __name__ == '__main__':
    main()
