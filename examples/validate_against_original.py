"""Validate PyClimExtract against the original Ghana pipeline's output.

Downloads ERA5 for a handful of DHS clusters, re-derives their exposures
with the package, and diffs them against the exposure file the published
analysis used.

Requires a configured ~/.cdsapirc (see README). Downloading ~6 clusters
takes a few minutes; the CDS queues requests.

    python examples/validate_against_original.py --n-clusters 6

Exit status is 0 if every column agrees within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from pyclimextract import add_derived, extract_daily, fetch_windows, summarise_windows
from pyclimextract.adapters import dhs

# The original pipeline used Timedelta(days=30) from the interview date,
# which spans 31 days inclusive. Match it exactly here: we are testing
# whether the package reproduces that pipeline, not whether the pipeline
# was right.
ORIGINAL_LOOKBACK_DAYS = 30
ORIGINAL_WINDOWS = (7, 14, 21, 30)

# Columns to compare, with absolute tolerances.
TOLERANCES = {
    "t2m_mean_30d": 0.01,   # degrees C
    "rh_mean_30d": 0.05,    # percentage points
    "hi_mean_30d": 0.01,    # degrees C
    "tp_sum_30d": 1e-5,     # metres
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default=r"..\Supported_Files_Codes",
                    help="directory with GHGE8AFL.csv, GHIR8CFL.DTA and the exposure CSV")
    ap.add_argument("--reference",
                    default="era5_exposures_7_14_21_30d_by_cluster.csv",
                    help="the exposure file the analysis actually used")
    ap.add_argument("--nc-dir", default="validation_nc")
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    reference = pd.read_csv(os.path.join(args.data_dir, args.reference))
    gps = dhs.read_dhs_gps(os.path.join(args.data_dir, "GHGE8AFL.csv"))
    ind = pd.read_stata(os.path.join(args.data_dir, "GHIR8CFL.DTA"),
                        columns=["v001", "v008"], convert_categoricals=False)

    windows = dhs.dhs_windows(gps, ind, lookback_days=ORIGINAL_LOOKBACK_DAYS)

    # Prefer clusters with a single interview month: where a cluster spans
    # two months it has two windows, and "the" 30-day exposure for that
    # cluster is ambiguous - a question about the original design, not
    # about whether the package reproduces it.
    counts = windows.groupby("point_id").size()
    single = counts[counts == 1].index
    eligible = [p for p in single if p in set(reference["cluster_id"])]
    if not eligible:
        print("No single-window clusters found in the reference file.")
        return 2

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(eligible, size=min(args.n_clusters, len(eligible)), replace=False)
    sel = windows[windows["point_id"].isin(chosen)].reset_index(drop=True)
    print(f"validating {len(sel)} clusters: {sorted(int(c) for c in chosen)}\n")

    if not args.skip_download:
        report = fetch_windows(sel, out_dir=args.nc_dir, variables=("t2m", "d2m", "tp"))
        if report["failed"]:
            print(f"ERROR: {len(report['failed'])} download(s) failed: {report['failed']}")
            return 1

    daily = add_derived(extract_daily(sel, nc_dir=args.nc_dir, on_missing="raise"))
    mine = summarise_windows(daily, windows_days=ORIGINAL_WINDOWS)
    mine = mine.rename(columns={"point_id": "cluster_id",
                                "t2m_c_mean_30d": "t2m_mean_30d",
                                "hi_c_mean_30d": "hi_mean_30d",
                                "tp_mm_sum_30d": "tp_mm_sum_30d"})
    # The reference stores precipitation in metres; the package uses mm.
    mine["tp_sum_30d"] = mine["tp_mm_sum_30d"] / 1000.0

    m = mine.merge(reference, on="cluster_id", suffixes=("_new", "_ref"))
    if m.empty:
        print("ERROR: no clusters matched the reference file.")
        return 1

    print(f"{'column':<16}{'max |diff|':>14}{'tolerance':>12}{'verdict':>10}")
    ok = True
    for col, tol in TOLERANCES.items():
        new, ref = col + "_new", col + "_ref"
        if new not in m or ref not in m:
            print(f"{col:<16}{'(absent)':>14}{tol:>12}{'SKIP':>10}")
            continue
        d = (m[new] - m[ref]).abs().max()
        passed = d <= tol
        ok &= passed
        print(f"{col:<16}{d:>14.6g}{tol:>12}{'PASS' if passed else 'FAIL':>10}")

    # This is the number of days the 30-day SUMMARY consumed, not the length
    # of the downloaded series. A Timedelta(days=30) window spans 31 days,
    # and summarise_windows takes the trailing 30 - so 30 here is correct,
    # and anything less means the download was short.
    print("\ndays contributing to the 30-day summary (expect 30):")
    print(mine[["cluster_id", "n_days_30d"]].to_string(index=False))

    if not ok:
        print("\nMismatch. Check: window length (inclusive endpoints), "
              "grid cell selection, and the heat-index adjustments flag.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
