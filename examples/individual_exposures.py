"""Build per-interview-window (individual-level) exposures for Ghana DHS.

Cluster-level exposure is an approximation: where a cluster was visited in
two different months, every woman in it is given one window - in practice
the later one, because `summarise_windows` takes trailing days. Keying the
exposure to each woman's own interview month removes that approximation.

    python examples/individual_exposures.py --data-dir ../Supported_Files_Codes \
        --nc-dir ../era5_nc --out ../Supported_Files_Codes/era5_exposures_individual.csv

Output has one row per (cluster, interview month), to be joined to women on
`v001` and `v008`.
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import pandas as pd

from pyclimextract import add_derived, extract_daily, summarise_windows
from pyclimextract.adapters.dhs import cmc_to_date, read_dhs_gps

WINDOWS = (7, 14, 21, 30)
_STEM = re.compile(r"^era5_(\d+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})(_instant)?\.nc$")


def windows_from_archive(nc_dir, gps):
    """Rebuild the window table from the downloaded filenames.

    Useful when the archive outlives the script that produced it - the
    filenames carry the full window definition.
    """
    seen = set()
    for name in sorted(os.listdir(nc_dir)):
        m = _STEM.match(name)
        if m:
            seen.add((int(m.group(1)), m.group(2), m.group(3)))

    win = pd.DataFrame(sorted(seen), columns=["point_id", "start_date", "end_date"])
    win = win.merge(gps[["point_id", "lat", "lon"]], on="point_id", how="left")
    win["start_date"] = pd.to_datetime(win["start_date"]).dt.date
    win["end_date"] = pd.to_datetime(win["end_date"]).dt.date

    orphan = int(win["lat"].isna().sum())
    if orphan:
        logging.warning("%d window(s) have no GPS match and will be dropped", orphan)
        win = win.dropna(subset=["lat", "lon"])
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../Supported_Files_Codes")
    ap.add_argument("--nc-dir", default="../era5_nc")
    ap.add_argument("--out", default="era5_exposures_individual.csv")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    gps = read_dhs_gps(os.path.join(args.data_dir, "GHGE8AFL.csv"))
    win = windows_from_archive(args.nc_dir, gps)
    logging.info("windows: %d across %d clusters", len(win), win["point_id"].nunique())

    daily = add_derived(extract_daily(win, nc_dir=args.nc_dir, on_missing="raise"))

    # by=[point_id, window_end] is the whole point: without it the two
    # windows of a twice-visited cluster pool into one series.
    out = summarise_windows(daily, windows_days=WINDOWS,
                            by=["point_id", "window_end"])

    out = out.rename(columns={"point_id": "v001"})
    for w in WINDOWS:
        out = out.rename(columns={
            f"t2m_c_mean_{w}d": f"t2m_mean_{w}d",
            f"hi_c_mean_{w}d": f"hi_mean_{w}d",
        })
        # Metres, to match the original cluster-level exposure files.
        out[f"tp_sum_{w}d"] = out[f"tp_mm_sum_{w}d"] / 1000.0

    # v008 is the join key on the DHS side; window_end is its date.
    out["v008"] = out["window_end"].map(
        lambda d: (d.year - 1900) * 12 + d.month
    )
    check = out["v008"].map(lambda c: cmc_to_date(c).date())
    bad = int((check != out["window_end"]).sum())
    if bad:
        raise AssertionError(f"{bad} window_end values did not round-trip through CMC")

    out.to_csv(args.out, index=False)
    logging.info("wrote %s (%d rows)", args.out, len(out))


if __name__ == "__main__":
    main()
