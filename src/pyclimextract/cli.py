"""Command-line entry point: ``pyclimextract <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from . import __version__
from .extract import add_derived, extract_daily
from .fetch import DEFAULT_VARIABLES, fetch_windows
from .summarise import summarise_windows
from .verify import verify_directory
from .windows import WINDOW_COLUMNS, validate_windows


def _read_windows(path):
    w = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    w["start_date"] = w["start_date"].dt.date
    w["end_date"] = w["end_date"].dt.date
    validate_windows(w)
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pyclimextract", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download ERA5 for each window")
    f.add_argument("windows", help=f"CSV with columns {WINDOW_COLUMNS}")
    f.add_argument("--nc-dir", default="era5_nc")
    f.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))

    v = sub.add_parser("verify", help="check downloaded NetCDF files")
    v.add_argument("--nc-dir", default="era5_nc")
    v.add_argument("--out", default="era5_verify_summary.csv")

    e = sub.add_parser("extract", help="build per-point exposure summaries")
    e.add_argument("windows", help=f"CSV with columns {WINDOW_COLUMNS}")
    e.add_argument("--nc-dir", default="era5_nc")
    e.add_argument("--out", default="exposures.csv")
    e.add_argument("--windows-days", nargs="+", type=int, default=[30])
    e.add_argument("--daily-out", default=None, help="also write the daily table")
    e.add_argument("--heat-index-adjustments", action="store_true",
                   help="use the full NWS heat index rather than bare Rothfusz")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.command == "fetch":
        report = fetch_windows(_read_windows(args.windows), args.nc_dir,
                               variables=tuple(args.variables))
        print(f"downloaded={len(report['downloaded'])} "
              f"skipped={len(report['skipped'])} failed={len(report['failed'])}")
        return 1 if report["failed"] else 0

    if args.command == "verify":
        checks = verify_directory(args.nc_dir)
        if checks.empty:
            print(f"no .nc files found in {args.nc_dir}")
            return 1
        checks.to_csv(args.out, index=False)
        print(f"{int(checks['opened'].sum())}/{len(checks)} opened -> {args.out}")
        return 0 if checks["opened"].all() else 1

    if args.command == "extract":
        windows = _read_windows(args.windows)
        daily = add_derived(
            extract_daily(windows, args.nc_dir),
            heat_index_adjustments=args.heat_index_adjustments,
        )
        if args.daily_out:
            daily.to_csv(args.daily_out, index=False)
        out = summarise_windows(daily, windows_days=tuple(args.windows_days))
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(out)} points)")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
