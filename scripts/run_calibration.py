import argparse
import pandas as pd
from merton_calib.calibrate import calibrate_powell

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with a column of log-returns")
    ap.add_argument("--col", default="r", help="Column name (default: r)")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step of returns")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    rets = df[args.col].values

    # Simple, safe bounds
    bounds = {
        "mu": (-1.0, 1.0),          # drift per dt
        "sigma": (1e-6, 2.0),       # diffusive vol
        "lam": (0.0, 10.0),         # jump intensity per dt
        "mu_j": (-2.0, 2.0),        # mean jump size (log-return)
        "sigma_j": (1e-3, 2.0),     # jump size std
    }

    res = calibrate_powell(rets, bounds=bounds, dt=args.dt, k_max=40, restarts=5, maxiter=4000)
    print("Status:", res["status"])
    print("NLL   :", res["fun"])
    for k, v in res["x"].items():
        print(f"{k:7s}: {v:.6g}")

if __name__ == "__main__":
    main()