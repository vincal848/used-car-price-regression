"""Schema validation on load, and the synthetic fixture used by the other tests."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import pytest

import data


def _minimal_valid_frame():
    cols = {c: [0] for c in data.REQUIRED_COLUMNS}
    cols["Engine"] = ["2.0"]
    cols["Fuel_Type"] = ["Gas"]
    return pd.DataFrame(cols)


def test_load_used_cars_raises_on_missing_columns(tmp_path):
    df = _minimal_valid_frame().drop(columns=["Highway", "Doors"])
    path = tmp_path / "bad.xlsx"
    df.to_excel(path, index=False)

    with pytest.raises(ValueError) as exc:
        data.load_used_cars(path)
    assert "Highway" in str(exc.value)
    assert "Doors" in str(exc.value)


def test_load_used_cars_reads_a_valid_file(tmp_path):
    df = _minimal_valid_frame()
    path = tmp_path / "good.xlsx"
    df.to_excel(path, index=False)

    loaded = data.load_used_cars(path)
    assert set(data.REQUIRED_COLUMNS) <= set(loaded.columns)
    assert len(loaded) == 1


def test_clean_used_cars_fixes_the_electric_engine_token():
    df = pd.DataFrame({
        "Engine": ["2.0", "E", "3.5"],
        "Fuel_Type": ["Gas", "Electric", "Gas"],
    })
    cleaned = data.clean_used_cars(df)
    assert cleaned["Engine"].tolist() == [2.0, 0.0, 3.5]
    assert cleaned["Engine"].dtype == float


def test_chained_assignment_silently_fails_to_set_the_value():
    """Pins the defect clean_used_cars replaces: under copy-on-write (the
    default since pandas 2.0, and the only mode from pandas 3.0), the original's
    `df['Engine'][mask] = value` sets the value on a temporary copy. It warns
    `ChainedAssignmentError` but the source frame is left completely unchanged.
    """
    df = pd.DataFrame({"Engine": ["2.0", "E", "3.5"]})
    with pytest.warns(pd.errors.ChainedAssignmentError):
        df["Engine"][df["Engine"] == "E"] = "0"
    assert df["Engine"].tolist() == ["2.0", "E", "3.5"]


def test_clean_used_cars_does_not_mutate_its_input():
    df = pd.DataFrame({"Engine": ["E"], "Fuel_Type": ["Gas"]})
    data.clean_used_cars(df)
    assert df["Engine"].tolist() == ["E"]


def test_clean_used_cars_merges_fuel_type_synonyms():
    df = pd.DataFrame({
        "Engine": ["2.0"] * 4,
        "Fuel_Type": ["Gas/Electric Hybrid", "Gasoline Fuel", "Gaseous Fuel Compatible", "Other"],
    })
    cleaned = data.clean_used_cars(df)
    assert cleaned["Fuel_Type"].tolist() == ["Gasoline Hybrid", "Gas", "Flexible", "Flexible"]
    # Left as strings, not label-encoded into integers.
    assert not pd.api.types.is_numeric_dtype(cleaned["Fuel_Type"])


def test_numeric_correlations_ignores_non_numeric_columns():
    """`df.corr()` raises TypeError on a frame with object columns under
    pandas >= 2.0; selecting numeric columns first avoids it.
    """
    df = pd.DataFrame({
        "Price": [10000, 20000, 30000],
        "Fuel_Type": ["Gas", "Hybrid", "Electric"],
    })
    corr = data.numeric_correlations(df)
    assert list(corr.columns) == ["Price"]


def test_make_synthetic_used_cars_has_the_documented_columns():
    df = data.make_synthetic_used_cars(n=50, seed=0)
    expected = {"Price", "Age", "Mileage", "EngineSize", "Horsepower", "Doors",
                "Passengers", "Fuel_Type", "Drivetrain"}
    assert set(df.columns) == expected
    assert len(df) == 50


def test_make_synthetic_used_cars_is_deterministic_with_a_seed():
    a = data.make_synthetic_used_cars(n=20, seed=7)
    b = data.make_synthetic_used_cars(n=20, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_make_synthetic_used_cars_seeds_do_differ():
    a = data.make_synthetic_used_cars(n=20, seed=1)
    b = data.make_synthetic_used_cars(n=20, seed=2)
    assert not a["Price"].equals(b["Price"])
