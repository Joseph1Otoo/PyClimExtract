"""Exposure windows: where to extract, and over which dates.

A *window table* is the single contract between every stage of the pipeline.
It is a DataFrame with exactly these columns:

    point_id   identifier for the location (any hashable; DHS cluster id, etc.)
    lat        latitude, decimal degrees, WGS84
    lon        longitude, decimal degrees, WGS84
    start_date datetime.date, first day of the exposure window (inclusive)
    end_date   datetime.date, last day of the exposure window (inclusive)

`fetch`, `extract` and `summarise` all take this table, so the download and
the extraction can never disagree about which file belongs to which point -
a failure mode of the original scripts, which rebuilt the table twice.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd

__all__ = ["WINDOW_COLUMNS", "build_windows", "validate_windows", "window_key"]

WINDOW_COLUMNS = ["point_id", "lat", "lon", "start_date", "end_date"]


def build_windows(points, lookback_days, reference_date=None, date_column=None):
    """Build a window table from a table of points.

    Parameters
    ----------
    points
        DataFrame with columns ``point_id``, ``lat``, ``lon``, and either a
        per-point reference date in `date_column` or a single
        `reference_date` applied to all points.
    lookback_days
        Length of the exposure window, in days, ending on the reference date.
        The window is inclusive of both endpoints and therefore spans
        ``lookback_days + 1`` days; pass ``lookback_days=29`` for exactly 30.
    reference_date
        A single date ending every window. Mutually exclusive with
        `date_column`.
    date_column
        Column in `points` holding a per-point reference date.

    Returns
    -------
    DataFrame with WINDOW_COLUMNS, de-duplicated.
    """
    if (reference_date is None) == (date_column is None):
        raise ValueError("Pass exactly one of reference_date or date_column.")

    missing = {"point_id", "lat", "lon"} - set(points.columns)
    if missing:
        raise KeyError(f"points is missing required columns: {sorted(missing)}")

    out = points[["point_id", "lat", "lon"]].copy()

    if date_column is not None:
        if date_column not in points.columns:
            raise KeyError(f"date_column {date_column!r} not found in points.")
        ref = pd.to_datetime(points[date_column])
    else:
        ref = pd.Series(pd.to_datetime(reference_date), index=points.index)

    out["end_date"] = ref.dt.date
    out["start_date"] = (ref - pd.Timedelta(days=int(lookback_days))).dt.date
    out = out[WINDOW_COLUMNS].dropna().drop_duplicates().reset_index(drop=True)

    validate_windows(out)
    return out


def validate_windows(windows):
    """Raise if `windows` is not a well-formed window table."""
    missing = set(WINDOW_COLUMNS) - set(windows.columns)
    if missing:
        raise KeyError(f"window table is missing columns: {sorted(missing)}")

    if windows["lat"].abs().gt(90).any():
        raise ValueError("latitude outside [-90, 90] - are lat and lon swapped?")
    if windows["lon"].abs().gt(180).any():
        raise ValueError("longitude outside [-180, 180] - are lat and lon swapped?")

    bad = windows["start_date"] > windows["end_date"]
    if bad.any():
        raise ValueError(f"{int(bad.sum())} window(s) have start_date after end_date.")
    return True


def window_key(point_id, start_date, end_date):
    """Canonical filename stem for one window.

    Every stage derives file names through this function, so the downloader
    and the extractor cannot drift apart.
    """
    def _d(x):
        if isinstance(x, str):
            return x
        if isinstance(x, (_dt.date, _dt.datetime)):
            return x.strftime("%Y-%m-%d")
        return pd.Timestamp(x).strftime("%Y-%m-%d")

    return f"era5_{point_id}_{_d(start_date)}_{_d(end_date)}"
