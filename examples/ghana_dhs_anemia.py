"""Reproduce the Ghana DHS 2022 climate exposures using PyClimExtract.

This replaces the original 01_/02_/03_ scripts in ../CODES with one pass.
Run from a directory holding GHGE8AFL.csv and GHIR8CFL.DTA:

    python examples/ghana_dhs_anemia.py --data-dir ../Supported_Files_Codes

Downloading all 618 cluster windows takes hours and needs a CDS account;
pass --skip-download to work from an existing era5_nc directory.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from pyclimextract import (
    add_derived,
    extract_daily,
    fetch_windows,
    summarise_windows,
    verify_directory,
)
from pyclimextract.adapters import dhs

# The original pipeline used a 30-day window ending at the interview date.
LOOKBACK_DAYS = 29
EXPOSURE_WINDOWS = (7, 14, 21, 30)
VARIABLES = ("t2m", "d2m", "tp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="directory holding the DHS files")
    ap.add_argument("--nc-dir", default="era5_nc", help="where NetCDF files live")
    ap.add_argument("--out", default="era5_exposures_by_cluster.csv")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gps_path = os.path.join(args.data_dir, "GHGE8AFL.csv")
    ind_path = os.path.join(args.data_dir, "GHIR8CFL.DTA")

    gps = dhs.read_dhs_gps(gps_path)
    logging.info("GPS clusters with a fix: %d", len(gps))

    ind = pd.read_stata(ind_path, columns=["v001", "v008"], convert_categoricals=False)
    windows = dhs.dhs_windows(gps, ind, lookback_days=LOOKBACK_DAYS)
    logging.info("exposure windows: %d", len(windows))

    if not args.skip_download:
        report = fetch_windows(windows, out_dir=args.nc_dir, variables=VARIABLES)
        logging.info(
            "downloaded %d, skipped %d, failed %d",
            len(report["downloaded"]), len(report["skipped"]), len(report["failed"]),
        )

    checks = verify_directory(args.nc_dir)
    if not checks.empty:
        logging.info(
            "verified %d files, %d opened cleanly", len(checks), int(checks["opened"].sum())
        )

    daily = add_derived(extract_daily(windows, nc_dir=args.nc_dir))
    logging.info("daily rows: %d across %d clusters", len(daily), daily["point_id"].nunique())

    exposures = summarise_windows(daily, windows_days=EXPOSURE_WINDOWS)
    exposures = exposures.rename(columns={"point_id": "cluster_id"})
    exposures.to_csv(args.out, index=False)
    logging.info("wrote %s (%d clusters)", args.out, len(exposures))


if __name__ == "__main__":
    main()
