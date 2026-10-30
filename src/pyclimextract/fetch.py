"""Download ERA5 single-level data for each exposure window.

Requires the optional ``download`` extra (``pip install pyclimextract[download]``)
and a configured ``~/.cdsapirc`` with the ERA5 licence accepted.
"""

from __future__ import annotations

import logging
import os

from .naming import CDS_VARIABLE_NAMES
from .windows import validate_windows, window_key

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_VARIABLES", "fetch_windows", "build_request"]

DEFAULT_VARIABLES = ("t2m", "d2m", "tp")

_DATASET = "reanalysis-era5-single-levels"


def build_request(lat, lon, start_date, end_date, variables, pad_deg=0.25, grid=(0.25, 0.25)):
    """Build the CDS request payload for one window.

    The area is a box of half-width `pad_deg` around the point. On the
    0.25-degree ERA5 grid a pad of 0.25 guarantees the containing cell plus
    its neighbours, which is what ``method="nearest"`` selection needs.
    """
    unknown = [v for v in variables if v not in CDS_VARIABLE_NAMES]
    if unknown:
        raise ValueError(f"unknown variable short name(s): {unknown}")

    north, south = lat + pad_deg, lat - pad_deg
    west, east = lon - pad_deg, lon + pad_deg

    return {
        "product_type": "reanalysis",
        "variable": [CDS_VARIABLE_NAMES[v] for v in variables],
        "date": f"{start_date}/{end_date}",
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],
        "grid": list(grid),
        "format": "netcdf",
        # Without this the CDS may return a zip archive rather than raw .nc.
        "download_format": "unarchived",
    }


def fetch_windows(
    windows,
    out_dir,
    variables=DEFAULT_VARIABLES,
    pad_deg=0.25,
    grid=(0.25, 0.25),
    skip_existing=True,
    client=None,
):
    """Download one NetCDF per window into `out_dir`.

    Already-downloaded windows are skipped when `skip_existing` is True, so
    an interrupted run can simply be restarted. Failures are logged and the
    run continues; the return value reports what happened.

    Parameters
    ----------
    client
        An object with a ``retrieve(dataset, request, target)`` method.
        Defaults to ``cdsapi.Client()``. Injectable for testing.

    Returns
    -------
    dict with keys ``downloaded``, ``skipped``, ``failed`` (lists of point ids).
    """
    validate_windows(windows)
    os.makedirs(out_dir, exist_ok=True)

    if client is None:
        try:
            import cdsapi
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "cdsapi is required to download. "
                "Install with: pip install pyclimextract[download]"
            ) from exc
        client = cdsapi.Client()

    result = {"downloaded": [], "skipped": [], "failed": []}

    for row in windows.itertuples(index=False):
        stem = window_key(row.point_id, row.start_date, row.end_date)
        target = os.path.join(out_dir, stem + ".nc")

        if skip_existing and os.path.exists(target):
            result["skipped"].append(row.point_id)
            continue

        request = build_request(
            row.lat, row.lon, row.start_date, row.end_date,
            variables, pad_deg=pad_deg, grid=grid,
        )

        try:
            log.info("downloading %s", stem)
            client.retrieve(_DATASET, request, target)
            result["downloaded"].append(row.point_id)
        except Exception:
            log.exception("CDS request failed for point %s", row.point_id)
            result["failed"].append(row.point_id)

    log.info(
        "fetch complete: %d downloaded, %d skipped, %d failed",
        len(result["downloaded"]), len(result["skipped"]), len(result["failed"]),
    )
    return result
