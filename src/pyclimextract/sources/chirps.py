"""CHIRPS daily rainfall.

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) is
a quasi-global daily rainfall product, 1981 to near-present, widely used
for African climate and health work because it blends satellite estimates
with station observations.

It is read here over OPeNDAP from the IRI Data Library, so only the cells
and days actually requested cross the network - a 30-day window at one
point is a few kilobytes, not the multi-gigabyte annual file.

No account, no API key, no download step::

    from pyclimextract import extract_from_dataset, summarise_windows
    from pyclimextract.sources import open_chirps

    rain = open_chirps()
    daily = extract_from_dataset(windows, rain, accumulated=["rain_mm"])
    exposure = summarise_windows(daily, windows_days=(7, 30),
                                 mean_vars=(), sum_vars=("rain_mm",),
                                 by=["point_id", "window_end"])
"""

from __future__ import annotations

import logging

import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

__all__ = ["CHIRPS_URLS", "CHIRPS_LATITUDE_LIMIT", "open_chirps", "check_coverage"]

_IRI = "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.daily-improved/.global"

CHIRPS_URLS = {
    "0p25": f"{_IRI}/.0p25/.prcp/dods",
    "0p05": f"{_IRI}/.0p05/.prcp/dods",
}

# CHIRPS is quasi-global: it stops at 50 degrees north and south.
CHIRPS_LATITUDE_LIMIT = 50.0

# The IRI Data Library reports time as Julian day number, which is not a
# CF-standard unit, so xarray cannot decode it. Julian day 2440587.5 is
# the Unix epoch.
_JULIAN_EPOCH = 2440587.5


def _decode_julian_days(values):
    """Julian day numbers -> pandas DatetimeIndex, normalised to midnight.

    IRI stamps each day at 12:00 UTC. Normalising to midnight makes the
    dates compare cleanly against ordinary calendar dates in the window
    table, which carry no time component.
    """
    stamps = pd.Timestamp("1970-01-01") + pd.to_timedelta(
        pd.Index(values) - _JULIAN_EPOCH, unit="D"
    )
    return stamps.normalize()


def open_chirps(resolution="0p25", url=None, variable_name="rain_mm", mask_negative=False):
    """Open CHIRPS daily rainfall as an analysis-ready Dataset.

    Parameters
    ----------
    resolution
        ``"0p25"`` (default, ~28 km) or ``"0p05"`` (~5.5 km). The finer
        grid is much slower over OPeNDAP; prefer it only when the point
        spacing genuinely warrants it. DHS displaces cluster coordinates
        by up to 10 km in rural areas, which already exceeds the 0.05
        degree cell, so 0p25 is usually the honest choice.
    url
        Override the endpoint entirely, e.g. a local mirror or a file you
        have already downloaded.
    variable_name
        What to call the rainfall variable. Default ``rain_mm`` says the
        unit out loud; CHIRPS natively calls it ``prcp``.
    mask_negative
        CHIRPS marks missing cells with a large negative sentinel. Leave
        this False (the default) for the remote dataset: the comparison
        cannot be evaluated lazily, so it would pull the entire global
        record across the network. Mask after extraction instead, which
        :func:`extract_chirps` does for you. Set it True only for a small
        local dataset already in memory.

    Returns
    -------
    ``xarray.Dataset`` with ``time``/``latitude``/``longitude`` and one
    rainfall variable in mm per day.
    """
    if url is None:
        if resolution not in CHIRPS_URLS:
            raise ValueError(
                f"resolution must be one of {sorted(CHIRPS_URLS)}, got {resolution!r}"
            )
        url = CHIRPS_URLS[resolution]

    # decode_times=False because the Julian-day unit is not CF-standard;
    # we decode it ourselves below.
    ds = xr.open_dataset(url, decode_times=False)
    return standardise_chirps(ds, variable_name=variable_name, mask_negative=mask_negative)


def extract_chirps(windows, resolution="0p25", url=None, variable_name="rain_mm",
                   warn_coverage=True):
    """Daily CHIRPS rainfall for a window table, in one call.

    The recommended entry point. It opens the remote dataset, extracts the
    requested points and dates, and blanks the missing-value sentinels -
    in that order, which matters.

    Masking has to happen *after* extraction. Applying it to the remote
    dataset would force the whole global record (tens of gigabytes) across
    the network before any subsetting, because the comparison cannot be
    evaluated lazily by the netCDF backend. Chunking it through dask does
    keep the arithmetic lazy, but turns one range request into thousands
    and is roughly a hundred times slower in practice.

    Parameters
    ----------
    windows
        Window table (see :mod:`pyclimextract.windows`).
    resolution, url, variable_name
        As for :func:`open_chirps`.
    warn_coverage
        Check the windows against CHIRPS coverage first and warn about any
        that fall outside it.

    Returns
    -------
    Daily table, as :func:`pyclimextract.extract_from_dataset` returns,
    with sentinels already blanked.
    """
    from ..extract import extract_from_dataset

    ds = open_chirps(resolution=resolution, url=url,
                     variable_name=variable_name, mask_negative=False)
    if warn_coverage:
        check_coverage(windows, ds)

    daily = extract_from_dataset(windows, ds, accumulated=[variable_name])
    return mask_missing(daily, variable=variable_name)


def mask_missing(daily, variable="rain_mm"):
    """Blank CHIRPS missing-value sentinels in an extracted daily table.

    Negative rainfall is not a dry day: summed into a 30-day total, a
    -9999 sentinel would destroy the exposure silently. Masking belongs
    here, on the small extracted table, rather than on the remote dataset
    - see :func:`extract_chirps` for why.
    """
    out = daily.copy()
    if variable in out.columns:
        out.loc[out[variable] < 0, variable] = float("nan")
    return out


def standardise_chirps(ds, variable_name="rain_mm", mask_negative=True):
    """Turn a raw IRI CHIRPS dataset into the package's conventions.

    Separated from :func:`open_chirps` so it can be tested without a
    network connection.
    """
    renames = {}
    if "T" in ds.coords or "T" in ds.dims:
        renames["T"] = "time"
    if "Y" in ds.coords or "Y" in ds.dims:
        renames["Y"] = "latitude"
    if "X" in ds.coords or "X" in ds.dims:
        renames["X"] = "longitude"
    if renames:
        ds = ds.rename(renames)

    for required in ("time", "latitude", "longitude"):
        if required not in ds.coords:
            raise KeyError(
                f"CHIRPS dataset has no {required} coordinate; found {list(ds.coords)}"
            )

    units = str(ds["time"].attrs.get("units", "")).lower()
    if "julian" in units or ds["time"].dtype.kind == "f":
        ds = ds.assign_coords(time=_decode_julian_days(ds["time"].values))

    if "prcp" in ds.data_vars and variable_name != "prcp":
        ds = ds.rename({"prcp": variable_name})

    if mask_negative and variable_name in ds.data_vars:
        ds[variable_name] = ds[variable_name].where(ds[variable_name] >= 0)

    ds[variable_name].attrs.setdefault("units", "mm")
    ds.attrs.setdefault("source", "CHIRPS v2.0 daily, via IRI Data Library")
    return ds


def check_coverage(windows, dataset=None):
    """Warn about windows CHIRPS cannot cover, and return the offenders.

    CHIRPS stops at 50 degrees north and south, and begins in 1981.
    Without this check such points come back as silent NaN.

    Returns
    -------
    DataFrame of the rows that fall outside coverage (empty if all fine).
    """
    outside_lat = windows["lat"].abs() > CHIRPS_LATITUDE_LIMIT

    too_early = pd.Series(False, index=windows.index)
    too_late = pd.Series(False, index=windows.index)
    if dataset is not None and "time" in dataset.coords:
        first = pd.Timestamp(dataset["time"].values.min())
        last = pd.Timestamp(dataset["time"].values.max())
        starts = pd.to_datetime(windows["start_date"])
        ends = pd.to_datetime(windows["end_date"])
        too_early = starts < first
        too_late = ends > last

    bad = windows[outside_lat | too_early | too_late]
    if len(bad):
        log.warning(
            "%d of %d windows fall outside CHIRPS coverage "
            "(beyond %.0f degrees latitude, or outside the available dates); "
            "their exposures will be missing",
            len(bad), len(windows), CHIRPS_LATITUDE_LIMIT,
        )
    return bad
