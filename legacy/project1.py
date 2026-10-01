"""SUPERSEDED. Kept for reference only -- this file is not part of the package and
is known to be incorrect.

This is the original ISQS 4370 coursework script. It loads `UsedCarData.xlsx` from a
hardcoded path on a machine that no longer exists, label-encodes three columns, and
runs seven `sm.OLS` regressions on `Price`, dropping one variable at a time by eyeing
a printed VIF list. Preserved because the README's "What was wrong" section refers to
it; do not import it.

Known defects, each with a named regression test:

1. Hardcoded, non-portable path (`C:/Users/Owner/Downloads/UsedCarData.xlsx`), and the
   file itself is not included -- it is not redistributable. `data.load_used_cars`
   takes a path argument and validates the result instead of assuming a location.
   See ``test_data.py::test_load_used_cars_raises_on_missing_columns``.

2. Written as a notebook cut-and-pasted into a .py file: `result.summary()`,
   `df_car.dtypes`, and `vif` appear as bare expressions seven times each. In a
   notebook these print; run as a script with `python project1.py`, every one of them
   evaluates and is silently discarded -- the script produces no output at all.
   ``run.py`` prints every VIF step and every summary explicitly.

3. `df_car['Engine'][df_car['Engine'] == 'E'] = 0` and the five `Fuel_Type`/
   `Drivetrain` reassignments below it are chained assignment. Under copy-on-write
   (the default since pandas 2.0, and the only mode from pandas 3.0), this sets the
   value on a temporary copy and warns `ChainedAssignmentError` -- the source frame is
   never actually changed. See
   ``test_data.py::test_chained_assignment_silently_fails_to_set_the_value``.

4. `df_corr = df_car.corr()` is called while `Transmission`, `Make`, `Model`, and
   `Body_Type` are still object columns. On pandas >= 2.0 this raises
   `TypeError: could not convert string to float` rather than silently dropping them,
   as older pandas did. See
   ``test_data.py::test_numeric_correlations_ignores_non_numeric_columns``.

5. Every one of the seven `sm.OLS(df_car.Price, df_car[[...]])` calls passes a design
   matrix with no constant column -- there is no `sm.add_constant` anywhere in the
   file. The fit is forced through the origin. On synthetic data with known
   coefficients this inflates R^2 from 0.82 (with a constant) to 0.98 (without one),
   and the `Doors` coefficient comes out with the wrong *sign* (+1190 against a true
   -350). See ``validate.py`` section "No intercept forces the fit through the
   origin" and ``test_model.py::test_no_intercept_r_squared_is_the_uncentered_one``.

6. `Fuel_Type` and `Drivetrain` are nominal categories (fuel system, wheel
   arrangement) mapped to integers 0-4 and 0-5 and fed into OLS as ordinary numeric
   columns. The regression then reads a change from code 1 to code 2 as a one-unit
   increase of the same kind as a change from code 2 to code 3, which is not true of
   "Gasoline Hybrid" versus "Electric" versus "Flexible". `model.fit_ols` takes a
   formula and expects these columns through `C(...)`, which fits one dummy per
   category instead of imposing an order. See
   ``test_model.py::test_fit_ols_recovers_known_coefficients_within_three_standard_errors``,
   which only holds with `C(...)` in the formula.

7. `import matplotlib.pyplot as plt` is never used.

The replacement is the flat modules at the repository root: `data.py`, `model.py`,
`run.py`.
"""

### New Analysis Model

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import statsmodels.api as sm

# initialize our constants
path = 'C:/Users/Owner/Downloads/'
file = 'UsedCarData.xlsx'

# read in our dataframe
df_car = pd.read_excel(path + file)

# check datatypes
df_car.dtypes

# check value counts for diff columns
df_car['Transmission'].value_counts()
df_car['Drivetrain'].value_counts()
df_car['Make'].value_counts()
df_car['Model'].value_counts()
df_car['Engine'].value_counts()
df_car['Fuel_Type'].value_counts()
df_car['Body_Type'].value_counts()

### do some munging by mapping and converting to int64
df_car['Engine'][df_car['Engine'] == 'E'] = 0
df_car['Engine'] = df_car['Engine'].astype('int64')

df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gas/Electric Hybrid'] = 'Gasoline Hybrid'
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gasoline Fuel'] = 'Gas'
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gaseous Fuel Compatible'] = 'Flexible'
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Other'] = 'Flexible'

df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gas'] = 0
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gasoline Hybrid'] = 1
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Electric'] = 2
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Flexible'] = 3
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Diesel'] = 4
df_car['Fuel_Type'] = df_car['Fuel_Type'].astype('int64')

df_car['Drivetrain'][df_car['Drivetrain'] == 'AWD'] = 0
df_car['Drivetrain'][df_car['Drivetrain'] == 'FWD'] = 1
df_car['Drivetrain'][df_car['Drivetrain'] == '4x4'] = 2
df_car['Drivetrain'][df_car['Drivetrain'] == 'RWD'] = 3
df_car['Drivetrain'][df_car['Drivetrain'] == '4WD'] = 4
df_car['Drivetrain'][df_car['Drivetrain'] == '2WD'] = 5
df_car['Drivetrain'] = df_car['Drivetrain'].astype('int64')

# correlation matrix
df_corr = df_car.corr()

# Regression 1
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Engine', 'Drivetrain', 'Passengers', 'Doors', 'Fuel_Type', 'City', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 2
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Engine', 'Passengers', 'Doors', 'Fuel_Type', 'City', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 3
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Engine', 'Passengers', 'Doors', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 4
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Passengers', 'Doors', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 5
model=sm.OLS(df_car.Price, df_car[['Kilometres', 'Passengers', 'Doors', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 6
model=sm.OLS(df_car.Price, df_car[['Kilometres', 'Passengers', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 7
model=sm.OLS(df_car.Price, df_car[['Kilometres', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif
