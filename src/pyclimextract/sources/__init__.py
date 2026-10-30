"""Gridded data sources.

Each module here knows how to reach one product and hand back a plain
``xarray.Dataset`` with ``time``/``latitude``/``longitude`` coordinates,
ready for :func:`pyclimextract.extract_from_dataset`. Nothing downstream
needs to know which source it came from.

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

__all__ = [
    "extract_chirps",
    "open_chirps",
    "CHIRPS_URLS",
    "check_coverage",
    "mask_missing",
]
