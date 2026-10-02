"""Show eligible, held-out, and shared rows separately; never edit thesis builds."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from digihealth_risk.phase_0.build_modeling_tables import load_data, OUT_DIR
from digihealth_risk.utils.patient_split import apply_canonical_split


def compute_counts() -> dict:
    source = load_data()
    shared = pd.read_csv(ROOT / 'digihealth_risk/phase_4/outputs/phase_4_2_v2_shared_cohort_summary.csv').set_index('horizon_years')
    rows = []
    for horizon in range(1, 6):
        name = 'phase_0_modeling_table.pkl' if horizon == 1 else f'phase_0_modeling_table_horizon_{horizon}_history_1.pkl'
        table = pd.read_pickle(OUT_DIR / name)
        train, calibration, test = apply_canonical_split(table, return_calibration=True)
        shared_rows = int(shared.loc[horizon, 'shared_rows'])
        if shared_rows > len(test):
            raise ValueError('Shared evaluation rows exceed eligible held-out rows')
        rows.append({'horizon': horizon, 'eligible_patients': table.PatientId.nunique(),
                     'eligible_rows': len(table), 'train_rows': len(train),
                     'calibration_rows': len(calibration), 'test_rows': len(test),
                     'shared_test_rows': shared_rows, 'test_rows_excluded_by_alignment': len(test) - shared_rows})
    return {'source_patients': source.PatientId.nunique(),
            'baseline_at_risk_patients': int(source.AtRisk_2005.eq(1).sum()), 'rows': rows}


def build_figure(counts: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(counts['rows'])
    rows.to_csv(OUT_DIR / 'publication_cohort_accounting.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    ax.axis('off')
    ax.set_title(f"Cohort accounting: {counts['source_patients']:,} source patients\n"
                 'Patient holdout and shared-row alignment are separate steps', fontsize=13, pad=20)
    shown = rows[['horizon', 'eligible_patients', 'eligible_rows', 'test_rows', 'shared_test_rows', 'test_rows_excluded_by_alignment']]
    labels = ['Horizon\n(years)', 'Eligible\npatients', 'Eligible\nrows', 'Held-out\nrows', 'Shared test\nrows', 'Alignment\nexclusions']
    table = ax.table(cellText=[[f'{int(v):,}' for v in row] for row in shown.to_numpy()], colLabels=labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.text(.5, .02, 'Canonical patient assignment: 60% train / 20% calibration / 20% test.\n'
            'Row fractions can differ because patients contribute different numbers of occasions.', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=10)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT_DIR / f'publication_cohort_flow.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    build_figure(compute_counts())
