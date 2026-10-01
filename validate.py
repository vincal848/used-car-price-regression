"""Regenerate docs/VALIDATION.md and the figure in docs/img/.

Every number in the documentation comes from this script, so the write-up cannot
drift away from the code:

    python validate.py

Run from the repository root. Everything here runs on the synthetic data from
data.make_synthetic_used_cars, since the original UsedCarData.xlsx is not
redistributable and is not in this repository.
"""

import io
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import data
import model

DOCS = ROOT / "docs"
IMG = DOCS / "img"

N = 1000
SEED = 0
NUMERIC_COLS = ["Age", "Mileage", "EngineSize", "Horsepower", "Doors", "Passengers"]
FORMULA_TEMPLATE = (
    "Price ~ {numeric}"
    " + C(Fuel_Type, Treatment(reference='Gas'))"
    " + C(Drivetrain, Treatment(reference='FWD'))"
)


def section_no_intercept_bias(out, df):
    out.write("## No intercept forces the fit through the origin\n\n")
    out.write(
        "The original ran every one of its seven regressions as "
        "`sm.OLS(df_car.Price, df_car[[...]])` with no constant column, which fits "
        "`Price = b1 x1 + ... + bk xk` with no intercept term at all -- the fit is "
        "forced through the origin. Below, the same four predictors "
        "(`Age`, `EngineSize`, `Horsepower`, `Doors`) fit with and without a constant, "
        "on the same synthetic rows with known coefficients.\n\n"
    )

    X = df[["Age", "EngineSize", "Horsepower", "Doors"]]
    y = df["Price"]
    no_int = sm.OLS(y, X).fit()
    with_int = sm.OLS(y, sm.add_constant(X)).fit()
    truth = {"Age": data.SYNTHETIC_COEFS["Age"], "EngineSize": data.SYNTHETIC_COEFS["EngineSize"],
             "Horsepower": data.SYNTHETIC_COEFS["Horsepower"], "Doors": data.SYNTHETIC_COEFS["Doors"]}

    out.write("| Coefficient | True | No intercept | With intercept |\n|---|---|---|---|\n")
    for name, true_val in truth.items():
        out.write(f"| {name} | {true_val:.1f} | {no_int.params[name]:.1f} | "
                  f"{with_int.params[name]:.1f} |\n")
    out.write(f"| R-squared | -- | {no_int.rsquared:.4f} (uncentered) | {with_int.rsquared:.4f} |\n\n")

    worst = max((k for k in truth if k != "Doors"),
                key=lambda k: abs(no_int.params[k] / truth[k] - 1))
    out.write(
        f"Forcing the fit through the origin does not just move the R-squared -- "
        f"`{worst}`'s coefficient comes out {no_int.params[worst]:.1f} against a true "
        f"{truth[worst]:.1f}, and `Doors` flips sign entirely "
        f"({no_int.params['Doors']:.1f} against a true {truth['Doors']:.1f}). The "
        f"no-intercept R-squared ({no_int.rsquared:.4f}) is the *uncentered* figure, "
        f"`1 - SSE / sum(y^2)`: with an intercept added back, the same data gives "
        f"{with_int.rsquared:.4f}, because the uncentered number gets credit for "
        f"`Price` being far from zero, which an intercept already explains for free.\n\n"
    )
    return no_int, with_int, truth


def section_vif_elimination(out, df):
    out.write("## VIF-based backward elimination\n\n")
    out.write(
        "`Mileage` is generated from `Age` (roughly 12,000 km per year) and has no "
        "effect on `Price` at all -- it is there purely to be found and removed.\n\n"
    )

    vifs_before = model.vif_table(df[NUMERIC_COLS])
    out.write("Before elimination:\n\n| Column | VIF |\n|---|---|\n")
    for name, v in vifs_before.items():
        out.write(f"| {name} | {v:.2f} |\n")
    out.write("\n")

    steps, kept = model.backward_vif_elimination(df[NUMERIC_COLS], threshold=10)
    out.write(f"Backward elimination at threshold 10 drops {len(steps)} column(s):\n\n")
    for s in steps:
        out.write(f"- dropped `{s['dropped']}` (VIF {s['vif']:.2f})\n")
    out.write("\n")

    vifs_after = model.vif_table(df[kept])
    out.write("After elimination:\n\n| Column | VIF |\n|---|---|\n")
    for name, v in vifs_after.items():
        out.write(f"| {name} | {v:.2f} |\n")
    out.write(
        f"\n`Horsepower`'s own VIF falls from {vifs_before['Horsepower']:.2f} to "
        f"{vifs_after['Horsepower']:.2f} once `Mileage` is gone -- it was never "
        f"collinear with `Age` on its own; it only looked that way because both "
        f"were feeding into `Mileage`.\n\n"
    )
    return steps, kept


def section_coefficient_recovery(out, df, kept):
    out.write("## Coefficients recovered after elimination\n\n")
    out.write(
        "The full model, fit on the columns elimination kept plus the two nominal "
        f"categories as `C(...)` terms, against the values that generated the data "
        f"(n={N}):\n\n"
    )

    formula = FORMULA_TEMPLATE.format(numeric=" + ".join(kept))
    result = model.fit_ols(df, formula)

    truth = {
        "Intercept": data.SYNTHETIC_INTERCEPT,
        "Age": data.SYNTHETIC_COEFS["Age"],
        "EngineSize": data.SYNTHETIC_COEFS["EngineSize"],
        "Horsepower": data.SYNTHETIC_COEFS["Horsepower"],
        "Doors": data.SYNTHETIC_COEFS["Doors"],
        "Passengers": 0.0,
        "C(Fuel_Type, Treatment(reference='Gas'))[T.Hybrid]": data.SYNTHETIC_FUEL_EFFECTS["Hybrid"],
        "C(Fuel_Type, Treatment(reference='Gas'))[T.Diesel]": data.SYNTHETIC_FUEL_EFFECTS["Diesel"],
        "C(Fuel_Type, Treatment(reference='Gas'))[T.Electric]": data.SYNTHETIC_FUEL_EFFECTS["Electric"],
        "C(Drivetrain, Treatment(reference='FWD'))[T.RWD]": data.SYNTHETIC_DRIVETRAIN_EFFECTS["RWD"],
        "C(Drivetrain, Treatment(reference='FWD'))[T.AWD]": data.SYNTHETIC_DRIVETRAIN_EFFECTS["AWD"],
    }

    out.write("| Term | True | Estimate | SE | within 3 SE |\n|---|---|---|---|---|\n")
    rows = []
    for name, true_val in truth.items():
        est, se = result.params[name], result.bse[name]
        within = abs(est - true_val) <= 3 * se
        rows.append((name, true_val, est, se, within))
        out.write(f"| `{name}` | {true_val:.1f} | {est:.1f} | {se:.1f} | "
                  f"{'yes' if within else 'NO'} |\n")
    out.write(f"\nFormula: `{formula}`\n\n")
    out.write(f"R-squared {result.rsquared:.4f}, adjusted {result.rsquared_adj:.4f}.\n\n")
    return rows


def figure(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 8})

    names = [r[0].replace("C(Fuel_Type, Treatment(reference='Gas'))", "Fuel")
                  .replace("C(Drivetrain, Treatment(reference='FWD'))", "Drive")
             for r in rows]
    truths = [r[1] for r in rows]
    estimates = [r[2] for r in rows]
    errors = [3 * r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(names))
    ax.errorbar(estimates, y_pos, xerr=errors, fmt="o", color="C0",
               label="estimate +/- 3 SE", capsize=3)
    ax.scatter(truths, y_pos, marker="x", color="C1", s=60, label="true value", zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("coefficient value")
    ax.set_title("Coefficients recovered on synthetic data vs. the values that generated it")
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(IMG / "coefficient_recovery.png")
    plt.close(fig)
    print(f"figure written to {IMG / 'coefficient_recovery.png'}")


def main():
    df = data.make_synthetic_used_cars(n=N, seed=SEED)

    out = io.StringIO()
    out.write("# Validation\n\n")
    out.write("Generated by `validate.py`. Do not edit by hand.\n\n")
    out.write(
        f"Everything below runs on `data.make_synthetic_used_cars(n={N}, seed={SEED})`: "
        "the original `UsedCarData.xlsx` is not redistributable and is not in this "
        "repository, so the synthetic data -- built from known coefficients -- is what "
        "pins the modeling code.\n\n"
    )

    section_no_intercept_bias(out, df)
    steps, kept = section_vif_elimination(out, df)
    rows = section_coefficient_recovery(out, df, kept)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "VALIDATION.md").write_text(out.getvalue(), encoding="utf-8")
    print(f"wrote {DOCS / 'VALIDATION.md'}")
    figure(rows)


if __name__ == "__main__":
    main()
