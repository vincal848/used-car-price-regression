"""Load the used-car data the course project ran on, and a synthetic stand-in.

The original spreadsheet (`UsedCarData.xlsx`) is not redistributable and is not in
this repository. `load_used_cars` validates that a file has the columns the
original script actually touched -- so a wrong or stale file fails loudly at load
time instead of producing a regression on whatever columns happened to exist.
`make_synthetic_used_cars` builds a dataset with the same kind of structure (a
noisy linear response, a deliberately collinear pair, nominal categories) from
known coefficients, so the modeling code in `model.py` has something to be
checked against even without the real file.
"""

import numpy as np
import pandas as pd

# Columns project1.py reads, either for value_counts() or inside a regression.
# Price is the response; everything else appears as a regressor in at least one
# of the seven models.
REQUIRED_COLUMNS = (
    "Price", "Year", "Kilometres", "Engine", "Fuel_Type", "Drivetrain",
    "Passengers", "Doors", "City", "Highway", "Transmission", "Make", "Model",
    "Body_Type",
)

# True coefficients used by make_synthetic_used_cars, kept here so tests and
# validate.py can check recovered estimates against the values that generated
# the data rather than against numbers copied into two places.
SYNTHETIC_INTERCEPT = 38000.0
SYNTHETIC_COEFS = {
    "Age": -1400.0,
    "EngineSize": 2600.0,
    "Horsepower": 30.0,
    "Doors": -350.0,
}
SYNTHETIC_FUEL_EFFECTS = {"Gas": 0.0, "Hybrid": 2000.0, "Diesel": 500.0, "Electric": 5000.0}
SYNTHETIC_DRIVETRAIN_EFFECTS = {"FWD": 0.0, "RWD": 800.0, "AWD": 1500.0}
SYNTHETIC_NOISE_SD = 1800.0


def load_used_cars(path):
    """Read the course spreadsheet and check it has the columns the models need.

    Raises ValueError naming the missing columns rather than letting a later
    `df[['Year', ...]]` KeyError point at the wrong line.
    """
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("missing expected columns: %s" % ", ".join(missing))
    return df


def clean_used_cars(df):
    """Fix the two things in the raw file that block a numeric regression.

    - `Engine` uses the literal string 'E' for electric vehicles instead of a
      displacement; it becomes 0.0 and the column is cast to float. The original
      wrote this as `df_car['Engine'][df_car['Engine'] == 'E'] = 0`, a chained
      assignment: under copy-on-write (the default since pandas 2.0, and the
      only mode from pandas 3.0) that sets a value on a temporary copy of the
      column and warns `ChainedAssignmentError`, but the original frame is left
      unchanged -- the fix never happens and nothing says so unless warnings are
      being watched. `.loc` assigns into the real frame in one step.
    - `Fuel_Type` has near-duplicate labels for the same category (e.g.
      'Gas/Electric Hybrid' and 'Gasoline Hybrid'). They are merged, but left as
      strings -- the original mapped them to integers 0-4 and fed that into OLS,
      which treats 'Diesel - Gas = Hybrid' as a meaningful statement. `fit_ols`
      is expected to pass this column through `C(...)` instead.
    """
    df = df.copy()
    # Assigned as the string "0", not the int 0: pandas's default string dtype
    # (since pandas 3.0) rejects an int value with a TypeError, same as it would
    # reject it through .loc or through the original's chained assignment.
    df.loc[df["Engine"] == "E", "Engine"] = "0"
    df["Engine"] = df["Engine"].astype(float)

    fuel_synonyms = {
        "Gas/Electric Hybrid": "Gasoline Hybrid",
        "Gasoline Fuel": "Gas",
        "Gaseous Fuel Compatible": "Flexible",
        "Other": "Flexible",
    }
    df["Fuel_Type"] = df["Fuel_Type"].replace(fuel_synonyms)
    return df


def numeric_correlations(df):
    """Correlation matrix over the numeric columns only.

    `df.corr()` on a frame that still has object columns raises
    `TypeError: could not convert string to float` on pandas >= 2.0 -- it used to
    silently drop them, which is exactly what the original script relied on.
    """
    return df.select_dtypes(include="number").corr()


def make_synthetic_used_cars(n=1000, seed=0):
    """A synthetic used-car dataset with known coefficients and one deliberately
    collinear pair.

    `Mileage` is generated from `Age` (12,000 km/year) plus a `Horsepower` term
    and has no effect on `Price` at all -- it exists purely so
    `backward_vif_elimination` has something real to remove. The `Horsepower`
    term matters for more than realism: a pair of columns related only to each
    other has *exactly* the same VIF on both sides, because that VIF is a
    function of their correlation and correlation is symmetric. Coupling
    `Mileage` to a second, independent regressor breaks that tie in `Mileage`'s
    favor reliably (checked over 200 seeds at n=200/300/1000 while writing this),
    without which `backward_vif_elimination` would drop `Age` or `Mileage`
    arbitrarily depending on the sampling noise of the draw.

    `Age`, `EngineSize`, `Horsepower` and `Doors` do determine `Price`, along
    with `Fuel_Type` and `Drivetrain`, which are nominal categories rather than
    ordered codes, to exercise `C(...)` in `fit_ols`. `Passengers` is pure
    noise: present as a regressor, with no true effect and no collinearity, as
    a control.

    This is not a clone of the real file's schema (which needed the
    string-token cleanup in `clean_used_cars`) -- it is a clean dataset built to
    exercise the same modeling code.
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(1, 15, n)
    engine_size = np.clip(rng.normal(2.5, 0.6, n), 1.0, 5.0)
    horsepower = rng.normal(200, 40, n)
    doors = rng.choice([2, 4, 5], n)
    passengers = rng.choice([2, 4, 5, 7], n)
    fuel_type = rng.choice(list(SYNTHETIC_FUEL_EFFECTS), n, p=[0.5, 0.2, 0.1, 0.2])
    drivetrain = rng.choice(list(SYNTHETIC_DRIVETRAIN_EFFECTS), n, p=[0.5, 0.2, 0.3])

    # Deliberately collinear with Age: 12,000 km per year, plus a Horsepower
    # term that breaks the Age/Mileage VIF tie (see the docstring above) so
    # elimination has a deterministic column to drop first.
    mileage = 12000 * age + 500 * horsepower + rng.normal(0, 4000, n)

    fuel_effect = np.array([SYNTHETIC_FUEL_EFFECTS[f] for f in fuel_type])
    drivetrain_effect = np.array([SYNTHETIC_DRIVETRAIN_EFFECTS[d] for d in drivetrain])

    price = (
        SYNTHETIC_INTERCEPT
        + SYNTHETIC_COEFS["Age"] * age
        + SYNTHETIC_COEFS["EngineSize"] * engine_size
        + SYNTHETIC_COEFS["Horsepower"] * horsepower
        + SYNTHETIC_COEFS["Doors"] * doors
        + fuel_effect
        + drivetrain_effect
        + rng.normal(0, SYNTHETIC_NOISE_SD, n)
    )

    return pd.DataFrame({
        "Price": price,
        "Age": age,
        "Mileage": mileage,
        "EngineSize": engine_size,
        "Horsepower": horsepower,
        "Doors": doors,
        "Passengers": passengers,
        "Fuel_Type": fuel_type,
        "Drivetrain": drivetrain,
    })
