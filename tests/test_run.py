"""The command line entry point runs end to end on the synthetic data."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def invoke(*args):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "run.py"), *args],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def test_demo_runs():
    code, out = invoke("demo", "--n", "300", "--seed", "1")
    assert code == 0, out
    assert "VIF elimination" in out
    assert "dropped Mileage" in out
    assert "OLS Regression Results" in out


def test_demo_threshold_can_suppress_elimination():
    code, out = invoke("demo", "--n", "300", "--seed", "1", "--threshold", "1000")
    assert code == 0, out
    assert "nothing dropped" in out


def test_fit_requires_the_data_argument():
    code, out = invoke("fit")
    assert code != 0


def test_fit_reports_a_clear_error_on_a_missing_file():
    code, out = invoke("fit", "--data", "no_such_file.xlsx")
    assert code != 0
