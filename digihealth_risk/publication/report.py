"""Generate aggregate tables, figures, and a provenance manifest from research outputs."""
import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / 'digihealth_risk/phase_4/outputs'
P5 = ROOT / 'digihealth_risk/phase_5/outputs'


def save(fig, directory, name):
    for extension in ['png', 'pdf']:
        fig.savefig(directory / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'digihealth_risk/publication/outputs')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    used, omitted = [], []
    def read(path):
        used.append(path)
        return pd.read_csv(path)
    ranking = read(P4 / 'phase_4_2_v2_cross_family_ranking.csv')
    intervention = read(P5 / 'phase_6_v2_intervention_model_summary.csv')
    top = ranking.sort_values(['horizon_years', 'pr_auc', 'roc_auc', 'brier'], ascending=[True, False, False, True]).groupby('horizon_years').head(5)
    top.to_csv(args.output_dir / 'healthcom_top_five.csv', index=False)
    winners = top.groupby('horizon_years').head(1)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    ax.plot(winners.horizon_years, winners.pr_auc, 'o-', color='#0072B2', label='Screening leader')
    ax.plot(intervention.horizon_years, intervention.best_intervention_pr_auc, 's--', color='#D55E00', label='Intervention leader')
    ax.set(xlabel='Prediction horizon (years)', ylabel='Average precision (PR-AUC)', xticks=range(1, 6),
           title='Two tracks evaluated on identical shared patient-year rows')
    ax.legend()
    save(fig, args.output_dir, 'intervention_vs_screening')

    path = P4 / 'publication_bootstrap.csv'
    if path.exists():
        bootstrap = read(path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained', sharey=True)
        for ax, comparison in zip(axes, ['tree_vs_statistical', 'winner_vs_survival']):
            frame = bootstrap[bootstrap.comparison.eq(comparison)]
            ax.vlines(frame.horizon_years, frame.delta_pr_auc_ci_low, frame.delta_pr_auc_ci_high, color='#0072B2')
            ax.plot(frame.horizon_years, frame.delta_pr_auc, 'o', color='#0072B2')
            ax.axhline(0, color='#555555', linestyle='--', linewidth=1)
            ax.set(title=comparison.replace('_', ' '), xlabel='Horizon (years)', xticks=range(1, 6))
        axes[0].set_ylabel('Difference in average precision\n95% patient-cluster bootstrap interval')
        save(fig, args.output_dir, 'bootstrap_differences')
    else:
        omitted.append('Bootstrap figure: run phase_4/bootstrap_significance.py')

    path = P4 / 'publication_feature_ablation.csv'
    if path.exists():
        ablation = read(path)
        pivot = ablation.pivot(index='family', columns='variant', values='pr_auc')
        ax = pivot.plot.bar(color=['#0072B2', '#D55E00'], rot=0, figsize=(8, 4))
        ax.set(ylabel='Average precision (PR-AUC)', xlabel='Model family',
               title='Current feature-set ablation: one-year horizon and history')
        ax.legend(title='Feature set', fontsize=9)
        save(ax.figure, args.output_dir, 'feature_ablation')
    else:
        omitted.append('Feature ablation figure: run publication.feature_ablation')

    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    styles = ['o-', 's--', '^:']
    for style, (history, group) in zip(styles, ranking.dropna(subset=['history_years']).groupby('history_years')):
        best = group.groupby('horizon_years').pr_auc.max()
        ax.plot(best.index, best.values, style, label=f'History {int(history)} years')
    ax.set(xlabel='Horizon (years)', ylabel='Best shared-cohort average precision', xticks=range(1, 6), title='History-window comparison')
    ax.legend()
    save(fig, args.output_dir, 'history_comparison')

    # Supplementary thesis figures use the same saved research results.
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for style, (family, group) in zip(styles, ranking.groupby('approach')):
        best = group.groupby('horizon_years').pr_auc.max()
        ax.plot(best.index, best.values, style, label=family)
    ax.set(xlabel='Horizon (years)', ylabel='Best shared-cohort average precision', xticks=range(1, 6), title='Model-family trajectories')
    ax.legend()
    save(fig, args.output_dir, 'family_trajectories')
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    survival = ranking[ranking.model_key.str.contains('two_stage')]
    for style, (history, group) in zip(styles, survival.groupby('history_years')):
        ordered = group.sort_values('horizon_years')
        ax.plot(ordered.horizon_years, ordered.pr_auc, style, label=f'History {int(history)} years')
    ax.set(xlabel='Horizon (years)', ylabel='Average precision (PR-AUC)', xticks=range(1, 6), title='Two-stage survival and history length')
    ax.legend()
    save(fig, args.output_dir, 'survival_history_ablation')

    path = ROOT / 'digihealth_risk/phase_7/outputs/phase_7_compare_monotonic.csv'
    if path.exists():
        ablation = read(path)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        markers = ['o', 's', '^', 'D', 'v']
        for marker, (family, group) in zip(markers, ablation.groupby('family')):
            ordered = group.sort_values('horizon_years')
            ax.plot(ordered.horizon_years, ordered.delta_pr_auc, marker=marker, label=family)
        ax.axhline(0, color='#555555', linestyle='--', linewidth=1)
        ax.set(xlabel='Horizon (years)', ylabel='Change in average precision: no-year minus baseline',
               xticks=range(1, 6), title='Calendar-time ablation by monotonic family')
        ax.legend(fontsize=9)
        save(fig, args.output_dir, 'calendar_time_ablation')
    else:
        omitted.append('Calendar ablation figure: run phase 7')

    path = P4 / 'publication_feature_importance.csv'
    if path.exists():
        importance = read(path)
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout='constrained')
        for ax, (horizon, group) in zip(axes.flat, importance.groupby('horizon_years')):
            top_features = group.nlargest(6, 'importance_share').sort_values('importance_share')
            ax.barh(top_features.feature, top_features.importance_share, color='#0072B2')
            ax.set(title=f'N={horizon}: {group.family.iloc[0]}', xlabel='Within-model importance share')
            ax.tick_params(axis='y', labelsize=9)
        axes.flat[-1].axis('off')
        save(fig, args.output_dir, 'research_feature_importance')
    else:
        omitted.append('Research importance figure: run publication.feature_importance')

    manifest = {
        'purpose': 'Local aggregate evidence; not an automatic public-data release',
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'catboost', 'lightgbm', 'statsmodels', 'interpret']},
        'inputs': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in used},
        'omitted_figures': omitted,
        'limitations': ['Private cohort is not distributed', 'Questionnaire timing is unverified',
                        'Retrospective test-set ranking', 'Current feature-set ablation is not recovered historical v1 code'],
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote aggregate evidence to {args.output_dir}; optional figures omitted: {len(omitted)}')


if __name__ == '__main__':
    main()
