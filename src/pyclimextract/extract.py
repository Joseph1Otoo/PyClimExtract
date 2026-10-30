"""Read downloaded NetCDF windows into a tidy daily table."""

from __future__ import annotations

import logging
import os

import pandas as pd
import xarray as xr

from . import derive
from .naming import ACCUMULATED_VARIABLES, standardise
from .windows import validate_windows, window_key

log = logging.getLogger(__name__)

__all__ = ["window_files", "extract_daily", "add_derived"]

# The CDS may return one file, or split instantaneous and accumulated
# variables into two. Try each layout in turn.
_SUFFIX_GROUPS = ((""),)
_SPLIT_SUFFIXES = ("_instant", "_accum")


def window_files(out_dir, point_id, start_date, end_date):
    """Return the existing NetCDF files for one window.

    Handles both the single-file layout and the ``_instant``/``_accum``
    split. Returns an empty list when nothing is present.
    """
    stem = window_key(point_id, start_date, end_date)

    single = os.path.join(out_dir, stem + ".nc")
    if os.path.exists(single):
        return [single]

    split = [os.path.join(out_dir, stem + s + ".nc") for s in _SPLIT_SUFFIXES]
    present = [p for p in split if os.path.exists(p)]
    return present


def _open_window(paths):
    """Open and merge the file(s) for one window into a standardised Dataset."""
    datasets = []
    for p in paths:
        ds = xr.open_dataset(p, engine="netcdf4")
        datasets.append(standardise(ds))

    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets, compat="override", join="inner")


def _to_daily(ds):
    """Aggregate hourly to daily: sum accumulations, mean everything else."""
    data_vars = [str(v) for v in ds.data_vars]
    accum = [v for v in data_vars if v in ACCUMULATED_VARIABLES]
    instant = [v for v in data_vars if v not in ACCUMULATED_VARIABLES]

    daily = xr.Dataset()
    if instant:
        daily = ds[instant].resample(time="1D").mean(keep_attrs=True)
    for v in accum:
        daily[v] = ds[v].resample(time="1D").sum(keep_attrs=True)
    return daily


def extract_daily(windows, nc_dir, on_missing="warn"):
    """Extract a daily time series for every window.

    For each window the value is taken at the grid cell nearest the point's
    own coordinates - not the centre of the downloaded bounding box, which
    can fall in a neighbouring cell.

    Parameters
    ----------
    windows
        Window table (see :mod:`pyclimextract.windows`).
    nc_dir
        Directory holding the downloaded NetCDF files.
    on_missing
        ``"warn"`` (default) logs and skips windows with no files,
        ``"raise"`` stops, ``"ignore"`` skips silently. The original
        scripts skipped silently, so a filename mismatch produced an empty
        result with no indication of why.

    Returns
    -------
    DataFrame with one row per point per day, carrying ``point_id``,
    ``time``, and every variable present in the files.
    """
    validate_windows(windows)
    if on_missing not in {"warn", "raise", "ignore"}:
        raise ValueError("on_missing must be 'warn', 'raise' or 'ignore'")

    frames = []
    missing = 0

    for row in windows.itertuples(index=False):
        paths = window_files(nc_dir, row.point_id, row.start_date, row.end_date)
        if not paths:
            missing += 1
            msg = f"no NetCDF file for point {row.point_id} ({row.start_date}..{row.end_date})"
            if on_missing == "raise":
                raise FileNotFoundError(msg)
            if on_missing == "warn":
                log.warning(msg)
            continue

        ds = _open_window(paths)
        if "time" not in ds.coords:
            log.warning("no time coordinate in %s - skipping", paths[0])
            continue

        daily = _to_daily(ds)

        if "latitude" in daily.coords and "longitude" in daily.coords:
            daily = daily.sel(latitude=row.lat, longitude=row.lon, method="nearest")

        df = daily.to_dataframe().reset_index()
        df["point_id"] = row.point_id
        # Carry window identity through. A point may have several windows
        # (a survey cluster visited in two months, say); without these the
        # windows pool silently and the summary spans both.
        df["window_start"] = row.start_date
        df["window_end"] = row.end_date
        frames.append(df)

    if missing and on_missing == "warn":
        log.warning("%d of %d windows had no files", missing, len(windows))

    if not frames:
        raise RuntimeError(
            f"No data extracted from {nc_dir!r}. "
            f"{missing} of {len(windows)} windows had no matching NetCDF file - "
            "check that the download completed and that the file naming matches "
            "pyclimextract.windows.window_key()."
        )

    return pd.concat(frames, ignore_index=True)


def extract_from_dataset(
    windows,
    dataset,
    accumulated=None,
    lat_name=None,
    lon_name=None,
    time_name=None,
):
    """Extract daily series from ONE gridded dataset covering all windows.

    The source-agnostic path. :func:`extract_daily` expects one file per
    window, named by :func:`pyclimextract.windows.window_key` - the layout
    the CDS downloader produces. This function instead takes a single
    dataset that already spans every point and date, which is how most
    gridded products arrive: a CHIRPS rainfall stack, a MODIS composite,
    a regional climate model run, or any NetCDF/Zarr a user already holds.

    Nothing here is specific to ERA5 or to any survey.

    Parameters
    ----------
    windows
        Window table (see :mod:`pyclimextract.windows`).
    dataset
        An ``xarray.Dataset``, or a path to anything ``xarray.open_dataset``
        can read. Needs a time coordinate and two spatial coordinates.
    accumulated
        Names of variables that accumulate within a day and must be SUMMED
        rather than averaged (rainfall, radiation). Defaults to the ERA5
        accumulations. Pass an empty collection if every variable is a
        state variable.

        Getting this wrong is silent: a daily *mean* of hourly rainfall is
        1/24 of the true total and looks entirely plausible.
    lat_name, lon_name, time_name
        Coordinate names, if they cannot be inferred. Common spellings
        (``lat``/``latitude``, ``lon``/``longitude``, ``time``/``valid_time``)
        are handled automatically.

    Returns
    -------
    DataFrame with the same schema as :func:`extract_daily`: one row per
    point per day, carrying ``point_id``, ``time``, ``window_start``,
    ``window_end`` and every variable in the dataset.

    Notes
    -----
    Data already at daily resolution passes through the daily aggregation
    unchanged, so sub-daily and daily sources can be mixed freely.
    """
    validate_windows(windows)

    ds = dataset
    if not isinstance(ds, xr.Dataset):
        ds = xr.open_dataset(ds)
    ds = standardise(ds)

    time_name = time_name or ("time" if "time" in ds.coords else None)
    lat_name = lat_name or ("latitude" if "latitude" in ds.coords else None)
    lon_name = lon_name or ("longitude" if "longitude" in ds.coords else None)

    for label, name in (("time", time_name), ("latitude", lat_name), ("longitude", lon_name)):
        if name is None or name not in ds.coords:
            raise KeyError(
                f"could not find the {label} coordinate; dataset has "
                f"{list(ds.coords)}. Pass it explicitly via "
                f"{label if label == 'time' else label[:3]}_name."
            )

    accum = set(ACCUMULATED_VARIABLES if accumulated is None else accumulated)
    unknown = accum - set(str(v) for v in ds.data_vars)
    if accumulated is not None and unknown:
        raise KeyError(f"accumulated names not in the dataset: {sorted(unknown)}")

    frames = []
    for row in windows.itertuples(index=False):
        at_point = ds.sel({lat_name: row.lat, lon_name: row.lon}, method="nearest")
        window = at_point.sel({time_name: slice(str(row.start_date), str(row.end_date))})

        if window.sizes.get(time_name, 0) == 0:
            log.warning(
                "no time steps for point %s in %s..%s - is the dataset's time "
                "range shorter than the windows?",
                row.point_id, row.start_date, row.end_date,
            )
            continue

        data_vars = [str(v) for v in window.data_vars]
        daily = xr.Dataset()
        state = [v for v in data_vars if v not in accum]
        if state:
            daily = window[state].resample({time_name: "1D"}).mean(keep_attrs=True)
        for v in (v for v in data_vars if v in accum):
            daily[v] = window[v].resample({time_name: "1D"}).sum(keep_attrs=True)

        df = daily.to_dataframe().reset_index()
        if time_name != "time":
            df = df.rename(columns={time_name: "time"})
        df["point_id"] = row.point_id
        df["window_start"] = row.start_date
        df["window_end"] = row.end_date
        frames.append(df)

    if not frames:
        raise RuntimeError(
            "No data extracted. The dataset's time range or spatial extent "
            "does not overlap the requested windows."
        )

    return pd.concat(frames, ignore_index=True)


def add_derived(daily, heat_index_adjustments=False):
    """Add derived columns to a daily table, where the inputs are present.

    Adds ``t2m_c``, ``rh``, ``hi_c``, ``wind10``, ``tp_mm``, ``ssrd_MJm2``.
    See :func:`pyclimextract.derive.heat_index` for the meaning of
    `heat_index_adjustments`.
    """
    df = daily.copy()

    if "t2m" in df:
        df["t2m_c"] = derive.kelvin_to_celsius(df["t2m"])
    if "t2m" in df and "d2m" in df:
        df["rh"] = derive.relative_humidity(df["t2m"], df["d2m"])
        df["hi_c"] = derive.heat_index(
            df["t2m_c"], df["rh"], adjustments=heat_index_adjustments
        )
    if "u10" in df and "v10" in df:
        df["wind10"] = derive.wind_speed(df["u10"], df["v10"])
    if "tp" in df:
        df["tp_mm"] = derive.metres_to_mm(df["tp"])
    if "ssrd" in df:
        df["ssrd_MJm2"] = derive.joules_to_megajoules(df["ssrd"])

    return df
