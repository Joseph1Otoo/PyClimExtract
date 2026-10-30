"""PyClimExtract - gridded climate reanalysis to point exposure windows.

The pipeline is four steps, each taking the window table as its contract::

    from pyclimextract import build_windows, fetch_windows, extract_daily
    from pyclimextract import add_derived, summarise_windows

    windows = build_windows(points, lookback_days=29, date_column="visit_date")
    fetch_windows(windows, out_dir="era5_nc")
    daily = add_derived(extract_daily(windows, nc_dir="era5_nc"))
    exposures = summarise_windows(daily, windows_days=(7, 14, 21, 30))

See ``pyclimextract.adapters.dhs`` for DHS survey helpers.
"""

from __future__ import annotations

from .derive import (
    heat_index,
    kelvin_to_celsius,
    metres_to_mm,
    relative_humidity,
    wind_speed,
)
from .extract import add_derived, extract_daily, extract_from_dataset, window_files
from .fetch import fetch_windows
from .summarise import summarise_windows
from .verify import verify_directory
from .windows import build_windows, validate_windows, window_key

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build_windows",
    "validate_windows",
    "window_key",
    "fetch_windows",
    "verify_directory",
    "extract_daily",
    "extract_from_dataset",
    "add_derived",
    "window_files",
    "summarise_windows",
    "heat_index",
    "relative_humidity",
    "kelvin_to_celsius",
    "wind_speed",
    "metres_to_mm",
]
