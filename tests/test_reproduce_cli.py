from pathlib import Path
import shutil
import subprocess
import pytest


def invoke(tmp_path, *args):
    # A checkout with no private dataset must support help and dry-run.
    script = tmp_path / 'reproduce.sh'
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'reproduce.sh', script)
    return subprocess.run(['bash', str(script), *args], capture_output=True, text=True)


@pytest.mark.parametrize('args', [('--from-phase', '4'), ('--from-phase=4',)])
def test_resume_dry_run_needs_no_dataset(tmp_path, args):
    result = invoke(tmp_path, *args, '--dry-run')
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'bootstrap_significance.py' in result.stdout
    assert 'evaluate_saved_models.py' in result.stdout
    assert 'GLMM exploratory' not in result.stdout


@pytest.mark.parametrize('args', [('--from-phase',), ('--from-phase', '8'), ('--from-phase=x',),
                                 ('--profile', 'unknown'), ('--profile',)])
def test_invalid_options_fail_before_running(tmp_path, args):
    result = invoke(tmp_path, *args, '--dry-run')
    assert result.returncode != 0
    assert 'STEP 1' not in result.stdout


def test_thesis_profile_adds_supplements_without_deployment(tmp_path):
    result = invoke(tmp_path, '--profile', 'thesis', '--dry-run')
    assert result.returncode == 0, result.stderr
    assert 'build_cohort_figure.py' in result.stdout
    assert 'CMD  python digihealth_risk/phase_6/export_models.py' not in result.stdout
