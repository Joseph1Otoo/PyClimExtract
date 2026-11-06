"""NASA POWER daily point data.

NASA POWER (Prediction Of Worldwide Energy Resources) serves daily
meteorology at any point on Earth, 1981 to near-present, through a plain
web API: no account, no API key, no file to download. Temperature and
humidity come from the MERRA-2 reanalysis and rainfall from a corrected
blend, on a grid of roughly 0.5 degrees (~50 km).

It is coarser than CHIRPS (0.05-0.25 degrees) or ERA5 (0.25 degrees), but
one request returns years of daily data for a point in seconds, which
makes it the practical choice for teaching, for quick exploration, and
whenever a remote OPeNDAP server is too slow to use::

    from pyclimextract import summarise_windows
    from pyclimextract.sources import extract_power

    daily = extract_power(windows)       # tp_mm, t2m_c, rh, hi_c
    exposure = summarise_windows(daily, windows_days=(30,))

Requests are made once per *location*, covering every window at that
location, so many windows at few places cost few requests.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from ..derive import heat_index
from ..windows import validate_windows

log = logging.getLogger(__name__)

__all__ = ["POWER_URL", "POWER_NAMES", "extract_power", "fetch_power_point"]

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# POWER parameter -> PyClimExtract column, so POWER output lines up with
# the ERA5 and CHIRPS paths downstream.
POWER_NAMES = {
    "PRECTOTCORR": "tp_mm",
    "T2M": "t2m_c",
    "T2M_MAX": "t2m_max_c",
    "T2M_MIN": "t2m_min_c",
    "RH2M": "rh",
    "WS2M": "wind2",
    "ALLSKY_SFC_SW_DWN": "ssrd_MJm2",   # POWER reports MJ/m2/day for daily
}

DEFAULT_PARAMETERS = ("PRECTOTCORR", "T2M", "RH2M")

# POWER marks missing values with this fill value.
_FILL = -999.0


def fetch_power_point(lat, lon, start, end, parameters=DEFAULT_PARAMETERS,
                      community="AG", retries=3, timeout=120):
    """Daily POWER data for one point as a DataFrame indexed by date.

    Columns keep POWER's own parameter names; :func:`extract_power` does
    the renaming. Missing values come back as NaN.
    """
    query = urllib.parse.urlencode({
        "parameters": ",".join(parameters),
        "community": community,
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON",
    })
    url = f"{POWER_URL}?{query}"

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.load(resp)
            break
        except Exception as exc:   # network errors, HTTP 429/5xx
            if attempt == retries:
                raise RuntimeError(f"NASA POWER request failed for ({lat}, {lon}): {exc}") from exc
            wait = 5 * attempt
            log.warning("POWER request failed (%s); retrying in %d s", exc, wait)
            time.sleep(wait)

    values = payload["properties"]["parameter"]
    df = pd.DataFrame(values)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    return df.astype(float).replace(_FILL, np.nan)


def extract_power(windows, parameters=DEFAULT_PARAMETERS, community="AG",
                  add_heat_index=True, fetch=None):
    """Daily NASA POWER data for a window table.

    Parameters
    ----------
    windows
        Window table (see :mod:`pyclimextract.windows`).
    parameters
        POWER parameter names. Those in :data:`POWER_NAMES` are renamed to
        PyClimExtract's columns; any others keep their POWER name.
    community
        POWER user community, which sets units. ``"AG"`` (agroclimatology)
        reports rainfall in mm/day and temperature in degrees C.
    add_heat_index
        Add ``hi_c`` when both temperature and humidity are present.
    fetch
        Replacement for :func:`fetch_power_point`, mainly for testing.

    Returns
    -------
    DataFrame with the same schema as
    :func:`pyclimextract.extract_from_dataset`: one row per window per
    day, carrying ``point_id``, ``time``, ``window_start``, ``window_end``
    and the requested variables.
    """
    validate_windows(windows)
    fetch = fetch or fetch_power_point

    w = windows.copy()
    w["start_date"] = pd.to_datetime(w["start_date"])
    w["end_date"] = pd.to_datetime(w["end_date"])

    frames = []
    locations = w.groupby(["lat", "lon"], sort=False)
    for i, ((lat, lon), group) in enumerate(locations, start=1):
        log.info("POWER location %d of %d (%.3f, %.3f)", i, locations.ngroups, lat, lon)
        series = fetch(lat, lon, group["start_date"].min(), group["end_date"].max(),
                       parameters=parameters, community=community)
        series = series.rename(columns=POWER_NAMES)
        series.index.name = "time"
        series = series.reset_index()

        for row in group.itertuples(index=False):
            days = series[series["time"].between(row.start_date, row.end_date)].copy()
            if days.empty:
                log.warning("no POWER data for point %s in %s..%s",
                            row.point_id, row.start_date.date(), row.end_date.date())
                continue
            days["point_id"] = row.point_id
            days["window_start"] = row.start_date
            days["window_end"] = row.end_date
            frames.append(days)

    if not frames:
        raise RuntimeError("No data extracted from NASA POWER for these windows.")

    daily = pd.concat(frames, ignore_index=True)
    if add_heat_index and {"t2m_c", "rh"} <= set(daily.columns):
        daily["hi_c"] = heat_index(daily["t2m_c"], daily["rh"])

    lead = ["point_id", "time", "window_start", "window_end"]
    return daily[lead + [c for c in daily.columns if c not in lead]]
