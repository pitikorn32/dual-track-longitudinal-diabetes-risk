from digihealth_risk.publication.smoke import run
from digihealth_risk.publication.synthetic import make_cohort
from digihealth_risk.utils import patient_split
import pandas as pd
import pytest


def test_synthetic_example_is_deterministic_and_explicitly_artificial():
    a, b = make_cohort(20), make_cohort(20)
    pd.testing.assert_frame_equal(a, b)
    assert a.PatientId.str.startswith('SYNTHETIC-').all()
    assert a.shape == (20, 121)


def test_smoke_uses_only_its_synthetic_source_and_isolated_split(tmp_path, monkeypatch):
    monkeypatch.setattr(patient_split, 'SOURCE_DATA', tmp_path / 'absent-private-data.pkl')
    monkeypatch.setattr(patient_split, 'SPLIT_CACHE', tmp_path / 'unused-global-cache.csv')
    result = run()
    assert result['unexpected_increases'] == 0
    assert result['scenarios'] == 7
    assert result['direction_checks'] > 0
    assert not patient_split.SPLIT_CACHE.exists()


def test_custom_cohort_cannot_replace_default_patient_cache(tmp_path):
    with pytest.raises(ValueError, match='isolated split cache'):
        patient_split.load_canonical_split(source_path=tmp_path / 'synthetic.pkl')
