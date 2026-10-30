"""Canonical variable names, and the aliases seen in the wild.

ERA5 files name the same variable differently depending on how they were
requested and which CDS backend served them (``t2m`` vs ``2m_temperature``),
and the newer CDS splits a single request into ``*_instant.nc`` and
``*_accum.nc``. Everything that normalises those differences lives here.
"""

from __future__ import annotations

__all__ = [
    "CANONICAL_VARIABLES",
    "ACCUMULATED_VARIABLES",
    "INSTANT_VARIABLES",
    "CDS_VARIABLE_NAMES",
    "canonical_name",
    "standardise",
]

# canonical short name -> aliases that may appear in a NetCDF file
CANONICAL_VARIABLES = {
    "t2m": ("t2m", "2m_temperature", "T2M"),
    "d2m": ("d2m", "2m_dewpoint_temperature", "D2M"),
    "tp": ("tp", "total_precipitation", "TP"),
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
    "ssrd": ("ssrd", "surface_solar_radiation_downwards"),
    "skt": ("skt", "skin_temperature"),
    "sp": ("sp", "surface_pressure"),
    "swvl1": ("swvl1", "volumetric_soil_water_layer_1"),
}

# canonical short name -> the name the CDS API expects in a request
CDS_VARIABLE_NAMES = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "ssrd": "surface_solar_radiation_downwards",
    "skt": "skin_temperature",
    "sp": "surface_pressure",
    "swvl1": "volumetric_soil_water_layer_1",
}

# Hourly accumulations: sum over the day. Everything else: mean over the day.
# Getting this wrong is the classic ERA5 error - a daily MEAN of hourly
# precipitation is 1/24 of the daily total and looks perfectly reasonable.
ACCUMULATED_VARIABLES = frozenset({"tp", "ssrd"})
INSTANT_VARIABLES = frozenset(set(CANONICAL_VARIABLES) - ACCUMULATED_VARIABLES)

_ALIAS_TO_CANONICAL = {
    alias: canon
    for canon, aliases in CANONICAL_VARIABLES.items()
    for alias in aliases
}

_COORD_ALIASES = {
    "valid_time": "time",
    "lat": "latitude",
    "lon": "longitude",
}


def canonical_name(name):
    """Map a variable name to its canonical short name, or None if unknown."""
    return _ALIAS_TO_CANONICAL.get(name)


def standardise(ds):
    """Rename coordinates and variables of an xarray Dataset in place-ish.

    Returns a new Dataset with ``time``/``latitude``/``longitude`` coordinates
    and canonical variable short names. Unknown variables are left untouched.
    """
    renames = {
        old: new
        for old, new in _COORD_ALIASES.items()
        if old in ds.coords and new not in ds.coords
    }
    if renames:
        ds = ds.rename(renames)

    var_renames = {}
    for name in ds.variables:
        canon = canonical_name(str(name))
        if canon and canon != name and canon not in ds.variables:
            var_renames[name] = canon
    if var_renames:
        ds = ds.rename(var_renames)

    return ds
