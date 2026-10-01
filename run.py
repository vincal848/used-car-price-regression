"""Run VIF-based backward elimination and print the resulting OLS fit.

    python run.py fit --data C:/path/to/UsedCarData.xlsx
    python run.py demo
"""

import argparse

import data
import model


def _report(label, steps, threshold, formula, result):
    print(label)
    print("\nVIF elimination (threshold=%s)" % threshold)
    if not steps:
        print("  nothing dropped; all VIFs <= threshold")
    for s in steps:
        print("  dropped %s (VIF=%.2f)" % (s["dropped"], s["vif"]))
    print("\nformula: %s\n" % formula)
    print(result.summary())


def cmd_fit(args):
    df = data.load_used_cars(args.data)
    df = data.clean_used_cars(df)

    numeric_cols = ["Year", "Kilometres", "Engine", "Passengers", "Doors", "City", "Highway"]
    steps, kept = model.backward_vif_elimination(df[numeric_cols], threshold=args.threshold)

    formula = "Price ~ " + " + ".join(kept) + " + C(Fuel_Type) + C(Drivetrain)"
    result = model.fit_ols(df, formula)
    _report("fit on %s (n=%d)" % (args.data, len(df)), steps, args.threshold, formula, result)


def cmd_demo(args):
    df = data.make_synthetic_used_cars(n=args.n, seed=args.seed)

    numeric_cols = ["Age", "Mileage", "EngineSize", "Horsepower", "Doors", "Passengers"]
    steps, kept = model.backward_vif_elimination(df[numeric_cols], threshold=args.threshold)

    formula = ("Price ~ " + " + ".join(kept)
               + " + C(Fuel_Type, Treatment(reference='Gas'))"
               + " + C(Drivetrain, Treatment(reference='FWD'))")
    result = model.fit_ols(df, formula)
    _report("synthetic data (n=%d, seed=%d)" % (args.n, args.seed),
             steps, args.threshold, formula, result)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="load UsedCarData.xlsx and fit it")
    p.add_argument("--data", required=True, help="path to the spreadsheet")
    p.add_argument("--threshold", type=float, default=10.0, help="VIF elimination threshold")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("demo", help="run on synthetic data with known coefficients")
    p.add_argument("--n", type=int, default=1000, help="number of synthetic rows")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--threshold", type=float, default=10.0, help="VIF elimination threshold")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
