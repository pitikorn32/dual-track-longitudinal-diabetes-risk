"""Entirely artificial schema example; never reads or samples the study cohort."""
import numpy as np
import pandas as pd

from digihealth_risk.phase_0.build_modeling_tables import YEARS, status_from_max_fbs


def make_cohort(patients: int = 100, seed: int = 17) -> pd.DataFrame:
    if patients < 20:
        raise ValueError('Use at least 20 synthetic patients for split smoke checks')
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'PatientId': [f'SYNTHETIC-{i:04d}' for i in range(patients)],
        'date_of_birth': pd.to_datetime(rng.integers(1945, 1986, patients).astype(str) + np.array(['-01-01'] * patients)),
    })
    for name in ['gender', 'dm_first_degree_relative', 'cooking_method', 'sleep_quality', 'smoking_status', 'alcohol_status']:
        df[name] = pd.Categorical(rng.integers(0, 2, patients).astype(str))
    for name in ['total_sugary_week', 'total_veg_fruit_week', 'total_exercise_week', 'total_phy_activity_week']:
        df[name] = rng.integers(0, 10, patients).astype(float)
    df['sleep_hours'] = rng.uniform(5, 9, patients)
    onset = rng.integers(2006, 2021, patients)
    maximum = pd.Series(np.nan, index=df.index)
    annual = {}
    for year in YEARS:
        fbs = pd.Series(np.where(year >= onset, rng.uniform(101, 140, patients), rng.uniform(75, 100, patients)))
        fbs[rng.random(patients) < .15] = np.nan
        maximum = pd.concat([maximum, fbs], axis=1).max(axis=1)
        annual[f'FBS_{year}'] = fbs
        for name, low, high in [('BMI', 18, 32), ('Pulse', 55, 100), ('BL_pres1', 95, 150),
                                 ('BL_pres2', 55, 95), ('Waist', 60, 110)]:
            values = rng.uniform(low, high, patients)
            values[rng.random(patients) < .15] = np.nan
            annual[f'{name}_{year}'] = values
        annual[f'MAX_FBS_up_to_{year}'] = maximum
        status = status_from_max_fbs(maximum)
        annual[f'DM_status_up_to_{year}'] = status
        annual[f'AtRisk_{year}'] = status.map({'non_dm': 0., 'pre_dm': 1., 'dm': 1.})
    return pd.concat([df, pd.DataFrame(annual)], axis=1)
