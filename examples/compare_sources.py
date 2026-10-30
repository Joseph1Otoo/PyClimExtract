"""Two rainfall products, one window table: ERA5 against CHIRPS.

Demonstrates that the framework is not tied to a single source. The
window table is built once and handed to both products; only the way the
data is reached differs.

    python examples/compare_sources.py --n-clusters 20

ERA5 comes from NetCDF files already downloaded through the CDS. CHIRPS
is read over OPeNDAP with no account and no download step.

The two disagree, and should: ERA5 is a model reanalysis, CHIRPS blends
satellite estimates with rain-gauge observations. Over West Africa CHIRPS
is generally the better rainfall estimate, while ERA5 supplies the
temperature and humidity fields CHIRPS does not carry. Knowing how far
apart they are is part of knowing what an exposure estimate is worth.
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd

from pyclimextract import add_derived, extract_daily, summarise_windows
from pyclimextract.adapters.dhs import read_dhs_gps
from pyclimextract.sources import extract_chirps

_STEM = re.compile(r"^era5_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})_instant\.nc$")


def windows_from_archive(nc_dir, gps):
    seen = {
        (int(m.group(1)), m.group(2), m.group(3))
        for m in (_STEM.match(f) for f in sorted(os.listdir(nc_dir)))
        if m
    }
    win = pd.DataFrame(sorted(seen), columns=["point_id", "start_date", "end_date"])
    win = win.merge(gps[["point_id", "lat", "lon"]], on="point_id", how="inner")
    win["start_date"] = pd.to_datetime(win["start_date"]).dt.date
    win["end_date"] = pd.to_datetime(win["end_date"]).dt.date
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Supported_Files_Codes")
    ap.add_argument("--nc-dir", default="../era5_nc")
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="source_comparison.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gps = read_dhs_gps(os.path.join(args.data_dir, "GHGE8AFL.csv"))
    windows = windows_from_archive(args.nc_dir, gps)

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(windows["point_id"].unique(),
                        size=min(args.n_clusters, windows["point_id"].nunique()),
                        replace=False)
    sel = windows[windows["point_id"].isin(chosen)].drop_duplicates("point_id")
    print(f"comparing {len(sel)} clusters\n")

    # ---- ERA5, from the local NetCDF archive --------------------------
    era5_daily = add_derived(extract_daily(sel, nc_dir=args.nc_dir, on_missing="raise"))
    era5 = summarise_windows(
        era5_daily, windows_days=(30,), mean_vars=(), sum_vars=("tp_mm",),
        by=["point_id", "window_end"],
    ).rename(columns={"tp_mm_sum_30d": "era5_mm"})

    # ---- CHIRPS, over OPeNDAP ------------------------------------------
    # Same window table. No download, no account.
    chirps_daily = extract_chirps(sel)
    chirps = summarise_windows(
        chirps_daily, windows_days=(30,), mean_vars=(), sum_vars=("rain_mm",),
        by=["point_id", "window_end"],
    ).rename(columns={"rain_mm_sum_30d": "chirps_mm"})

    both = era5[["point_id", "window_end", "era5_mm"]].merge(
        chirps[["point_id", "window_end", "chirps_mm"]],
        on=["point_id", "window_end"],
    )
    both["difference_mm"] = both["era5_mm"] - both["chirps_mm"]

    print(both.round(1).to_string(index=False))

    r = both["era5_mm"].corr(both["chirps_mm"])
    print(f"\n30-day rainfall totals, {len(both)} clusters")
    print(f"  ERA5   mean {both.era5_mm.mean():7.1f} mm")
    print(f"  CHIRPS mean {both.chirps_mm.mean():7.1f} mm")
    print(f"  correlation r = {r:.3f}")
    print(f"  mean absolute difference {both.difference_mm.abs().mean():.1f} mm")

    both.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
