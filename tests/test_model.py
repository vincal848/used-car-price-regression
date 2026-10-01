"""VIF, backward elimination, and the formula-API fit against known coefficients."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import data
import model


def test_vif_of_orthogonal_columns_is_near_one():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "a": rng.normal(size=5000),
        "b": rng.normal(size=5000),
        "c": rng.normal(size=5000),
    })
    vifs = model.vif_table(X)
    assert "const" not in vifs.index
    assert (vifs - 1.0).abs().max() < 0.05


def test_vif_of_a_near_duplicate_column_is_large():
    rng = np.random.default_rng(42)
    a = rng.normal(size=5000)
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=1e-3, size=5000)})
    vifs = model.vif_table(X)
    assert vifs["a"] > 1000
    assert vifs["b"] > 1000


def test_backward_elimination_drops_the_collinear_column_first():
    """Mileage is generated from Age (plus a little EngineSize) in
    make_synthetic_used_cars specifically so this has something to find.
    """
    df = data.make_synthetic_used_cars(n=1000, seed=0)
    numeric_cols = ["Age", "Mileage", "EngineSize", "Horsepower", "Doors", "Passengers"]
    steps, kept = model.backward_vif_elimination(df[numeric_cols], threshold=10)

    assert len(steps) == 1
    assert steps[0]["dropped"] == "Mileage"
    assert steps[0]["vif"] > 100
    assert "Mileage" not in kept
    assert set(kept) == {"Age", "EngineSize", "Horsepower", "Doors", "Passengers"}

    # Everything remaining is actually below the threshold now.
    assert model.vif_table(df[kept]).max() <= 10


def test_backward_elimination_is_a_no_op_when_nothing_exceeds_the_threshold():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    steps, kept = model.backward_vif_elimination(X, threshold=10)
    assert steps == []
    assert kept == ["a", "b"]


def test_fit_ols_recovers_known_coefficients_within_three_standard_errors():
    df = data.make_synthetic_used_cars(n=1000, seed=0)
    formula = ("Price ~ Age + EngineSize + Horsepower + Doors + Passengers"
               " + C(Fuel_Type, Treatment(reference='Gas'))"
               " + C(Drivetrain, Treatment(reference='FWD'))")
    result = model.fit_ols(df, formula)

    checks = {
        "Intercept": data.SYNTHETIC_INTERCEPT,
        "Age": data.SYNTHETIC_COEFS["Age"],
        "EngineSize": data.SYNTHETIC_COEFS["EngineSize"],
        "Horsepower": data.SYNTHETIC_COEFS["Horsepower"],
        "Doors": data.SYNTHETIC_COEFS["Doors"],
        "C(Fuel_Type, Treatment(reference='Gas'))[T.Hybrid]": data.SYNTHETIC_FUEL_EFFECTS["Hybrid"],
        "C(Fuel_Type, Treatment(reference='Gas'))[T.Diesel]": data.SYNTHETIC_FUEL_EFFECTS["Diesel"],
        "C(Fuel_Type, Treatment(reference='Gas'))[T.Electric]": data.SYNTHETIC_FUEL_EFFECTS["Electric"],
        "C(Drivetrain, Treatment(reference='FWD'))[T.RWD]": data.SYNTHETIC_DRIVETRAIN_EFFECTS["RWD"],
        "C(Drivetrain, Treatment(reference='FWD'))[T.AWD]": data.SYNTHETIC_DRIVETRAIN_EFFECTS["AWD"],
    }
    for name, truth in checks.items():
        est, se = result.params[name], result.bse[name]
        assert abs(est - truth) <= 3 * se, "%s: %.2f vs truth %.2f (se %.2f)" % (name, est, truth, se)

    # Passengers has no true effect and is not collinear with anything -- the
    # fit should not find a large, confident effect where there isn't one.
    assert abs(result.params["Passengers"]) <= 3 * result.bse["Passengers"]


def test_no_intercept_r_squared_is_the_uncentered_one():
    """sm.OLS with no constant reports the *uncentered* R^2 (1 - SSE / sum(y^2)),
    which the original relied on for every one of its seven regressions. On
    data where y is far from zero, that is a very different, and much larger,
    number than the usual R^2 against the mean.
    """
    df = data.make_synthetic_used_cars(n=1000, seed=0)
    X = df[["Age", "EngineSize", "Horsepower", "Doors"]]
    y = df["Price"]

    no_intercept = sm.OLS(y, X).fit()
    with_intercept = sm.OLS(y, sm.add_constant(X)).fit()

    assert no_intercept.rsquared == pytest.approx(
        1 - (no_intercept.resid ** 2).sum() / (y ** 2).sum(), abs=1e-10)
    # The two are not even close: the uncentered figure gets credit for the
    # overall price level, which a model with an intercept already explains
    # for free and does not count as "fit".
    assert no_intercept.rsquared - with_intercept.rsquared > 0.1
