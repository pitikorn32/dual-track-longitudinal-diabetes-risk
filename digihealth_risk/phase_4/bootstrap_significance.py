"""Paired patient-cluster bootstrap on the all-family shared evaluation cohort."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from digihealth_risk.phase_4.cross_family_comparison import load_shared_predictions, ranking_table
from digihealth_risk.utils.evaluation import KEY_COLUMNS, TARGET, validate_scores


def paired_cluster_bootstrap(y, a, b, patients, *, replicates=2000, seed=20260501):
    """Resample patients with replacement, retaining every row in each cluster.

    Multiplicity weights are exactly equivalent to physically repeating rows,
    and avoid allocating thousands of concatenated patient-year arrays.
    """
    y, a = validate_scores(y, a)
    _, b = validate_scores(y, b)
    if replicates < 1 or len(patients) != len(y):
        raise ValueError('Positive replicate count and one patient per row required')
    if pd.isna(patients).any():
        raise ValueError('Missing patient identity')
    ids, inverse = np.unique(np.asarray(patients, dtype=str), return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        weights = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))[inverse]
        if len(np.unique(y[weights > 0])) < 2:
            continue
        ap_a = average_precision_score(y, a, sample_weight=weights)
        ap_b = average_precision_score(y, b, sample_weight=weights)
        samples.append([ap_a, ap_b, ap_a - ap_b,
                        roc_auc_score(y, a, sample_weight=weights),
                        roc_auc_score(y, b, sample_weight=weights)])
    if not samples:
        raise ValueError('No bootstrap samples with both classes')
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    result = {'replicates_requested': replicates, 'replicates_valid': len(samples), 'seed': seed,
              'rows': len(y), 'patients': len(ids),
              'a_pr_auc': average_precision_score(y, a), 'b_pr_auc': average_precision_score(y, b)}
    for i, name in enumerate(['a_pr_auc', 'b_pr_auc', 'delta_pr_auc', 'a_roc_auc', 'b_roc_auc']):
        result[name + '_ci_low'] = low[i]
        result[name + '_ci_high'] = high[i]
    result['delta_pr_auc'] = result['a_pr_auc'] - result['b_pr_auc']
    result['gap_excludes_zero'] = bool(low[2] > 0 or high[2] < 0)
    return result


def analyze(predictions, *, replicates=2000, seed=20260501, survival_reference="fixed-m5"):
    ranking = ranking_table(predictions)
    results = []
    for horizon, candidates in ranking.groupby('horizon_years', sort=True):
        ordered = candidates.sort_values(['pr_auc', 'roc_auc', 'brier'], ascending=[False, False, True])
        tree = ordered[ordered.approach.eq('tree')].iloc[0]
        statistical = ordered[ordered.approach.eq('statistical')].iloc[0]
        survival_candidates = ordered[ordered.approach.eq('survival')]
        if survival_reference == 'fixed-m5':
            survival_candidates = survival_candidates[survival_candidates.model_key.str.contains('two_stage') & survival_candidates.history_years.eq(5)]
        survival = survival_candidates.iloc[0]
        winner = ordered.iloc[0]
        cohort = predictions[predictions.horizon_years.eq(horizon)]
        for offset, (comparison, left, right) in enumerate([
            ('tree_vs_statistical', tree, statistical), ('winner_vs_survival', winner, survival),
        ]):
            def select(row):
                return cohort[cohort.model_key.eq(row.model_key) & cohort.calibration_method.eq(row.calibration_method)].sort_values(KEY_COLUMNS)
            a, b = select(left), select(right)
            if not a[KEY_COLUMNS + [TARGET]].reset_index(drop=True).equals(b[KEY_COLUMNS + [TARGET]].reset_index(drop=True)):
                raise ValueError('Bootstrap comparisons require identical shared occasions and targets')
            results.append({
                'horizon_years': int(horizon), 'comparison': comparison,
                'survival_reference': survival_reference,
                'a_model_key': left.model_key, 'a_calibration': left.calibration_method,
                'b_model_key': right.model_key, 'b_calibration': right.calibration_method,
                **paired_cluster_bootstrap(a[TARGET], a.predicted_probability, b.predicted_probability,
                                           a.PatientId, replicates=replicates, seed=seed + 2 * (int(horizon) - 1) + offset),
            })
        print(f'Bootstrap completed horizon {horizon}', flush=True)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260501)
    parser.add_argument('--survival-reference', choices=['fixed-m5', 'best'], default='fixed-m5',
                        help='Original analysis used two-stage M=5; best is a separate sensitivity analysis')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'digihealth_risk/phase_4/outputs')
    args = parser.parse_args()
    shared, _ = load_shared_predictions()
    results = analyze(shared, replicates=args.replicates, seed=args.seed, survival_reference=args.survival_reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = 'publication_bootstrap.csv' if args.survival_reference == 'fixed-m5' else 'publication_bootstrap_best_survival.csv'
    results.to_csv(args.output_dir / filename, index=False)


if __name__ == '__main__':
    main()
