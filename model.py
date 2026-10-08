"""VIF-based backward elimination and OLS fitting, done the way the original wasn't:
with an intercept, and with nominal categories as dummies rather than integer codes.
"""

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant


def vif_table(X: pd.DataFrame) -> pd.Series:
    """Variance inflation factor for each column of X, as a Series.

    `variance_inflation_factor` needs a constant column in the design matrix to
    mean what it claims to -- without one, every VIF is computed against a model
    forced through the origin, and comes out inflated (see
    `demonstrate_no_intercept_bias` in validate.py for how much). We add the
    constant, compute VIF for the whole design including it, then drop the
    intercept's own row: a VIF for the intercept isn't a quantity anyone wants.
    """
    X = pd.DataFrame(X).copy()
    design = add_constant(X, has_constant="add")
    vifs = pd.Series(
        [variance_inflation_factor(design.values, i) for i in range(design.shape[1])],
        index=design.columns,
    )
    return vifs.drop("const")


def backward_vif_elimination(X: pd.DataFrame,
                             threshold: float = 10) -> tuple[list[dict], list[str]]:
    """Drop the column with the highest VIF, repeatedly, until every remaining
    column is at or below threshold.

    Returns (steps, kept_columns). `steps` is a list of
    {"dropped": name, "vif": value, "remaining": [...]} dicts in removal order --
    the original project did this by hand, rerunning the regression and eyeing
    the printed VIF list seven times, which is the same procedure but here it is
    reproducible and testable.
    """
    X = pd.DataFrame(X).copy()
    steps = []
    while X.shape[1] > 1:
        vifs = vif_table(X)
        worst = vifs.idxmax()
        if vifs[worst] <= threshold:
            break
        X = X.drop(columns=worst)
        steps.append({
            "dropped": worst,
            "vif": float(vifs[worst]),
            "remaining": list(X.columns),
        })
    return steps, list(X.columns)


def fit_ols(df: pd.DataFrame, formula: str) -> RegressionResultsWrapper:
    """Fit an OLS model through the formula API.

    Formula strings get an intercept by default and should use `C(column)` for
    nominal categories. Both of those are the fix for the original, which passed
    `sm.OLS` a plain design matrix with no constant column at all (forcing the
    fit through the origin) and label-encoded nominal columns like `Fuel_Type`
    and `Drivetrain` into integers, which imposes an ordering on categories that
    have none.
    """
    return smf.ols(formula, data=df).fit()
