"""Gridded data sources.

Each module here knows how to reach one product and hand back data ready
for the rest of the pipeline: CHIRPS as a gridded ``xarray.Dataset`` for
:func:`pyclimextract.extract_from_dataset`, NASA POWER (a point API) as
the daily table that function would return. Nothing downstream needs to
know which source it came from.

ERA5 is the exception: it arrives through the Copernicus CDS as one file
per request, so it has its own download path in
:mod:`pyclimextract.fetch` rather than living here.
"""

from .chirps import (
    CHIRPS_URLS,
    check_coverage,
    extract_chirps,
    mask_missing,
    open_chirps,
)
from .power import POWER_NAMES, extract_power, fetch_power_point

__all__ = [
    "extract_power",
    "fetch_power_point",
    "POWER_NAMES",
    "extract_chirps",
    "open_chirps",
    "CHIRPS_URLS",
    "check_coverage",
    "mask_missing",
]
