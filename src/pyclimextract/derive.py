"""Derived meteorological quantities.

Pure functions over array-likes. No I/O, no global state - these are the
pieces most worth unit-testing, because an error here is silent: the
pipeline still runs and still produces plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kelvin_to_celsius",
    "celsius_to_fahrenheit",
    "fahrenheit_to_celsius",
    "relative_humidity",
    "heat_index",
    "wind_speed",
    "metres_to_mm",
    "joules_to_megajoules",
]

# Magnus-Tetens coefficients over water (Alduchov & Eskridge 1996).
_MAGNUS_A = 17.625
_MAGNUS_B = 243.04
_SVP_0C_HPA = 6.1094


def kelvin_to_celsius(k):
    """Kelvin -> degrees Celsius."""
    return np.asarray(k, dtype=float) - 273.15


def celsius_to_fahrenheit(c):
    """Degrees Celsius -> degrees Fahrenheit."""
    return np.asarray(c, dtype=float) * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(f):
    """Degrees Fahrenheit -> degrees Celsius."""
    return (np.asarray(f, dtype=float) - 32.0) * 5.0 / 9.0


def _saturation_vapour_pressure(t_c):
    """Saturation vapour pressure (hPa) at temperature `t_c` (degrees C)."""
    t_c = np.asarray(t_c, dtype=float)
    return _SVP_0C_HPA * np.exp((_MAGNUS_A * t_c) / (_MAGNUS_B + t_c))


def relative_humidity(t2m_k, d2m_k):
    """Relative humidity (%) from 2 m temperature and dewpoint, both in Kelvin.

    Uses the ratio of actual to saturation vapour pressure, both from the
    Magnus formula. Result is clipped to [0, 100]: values marginally above
    100 arise from rounding when dewpoint meets temperature in saturated air.
    """
    t_c = kelvin_to_celsius(t2m_k)
    td_c = kelvin_to_celsius(d2m_k)
    es = _saturation_vapour_pressure(t_c)
    e = _saturation_vapour_pressure(td_c)
    return np.clip(100.0 * (e / es), 0.0, 100.0)


def heat_index(t_c, rh, adjustments=False):
    """NWS heat index (apparent temperature) in degrees Celsius.

    Parameters
    ----------
    t_c
        Air temperature, degrees Celsius.
    rh
        Relative humidity, percent (0-100).
    adjustments
        If False (the default), apply the bare Rothfusz regression whenever
        the air temperature is at or above 80 degrees F, and return the air
        temperature below that.

        If True, apply the full NWS algorithm: the Steadman simple formula
        first, the Rothfusz regression only when their average reaches
        80 degrees F, plus the low-humidity and high-humidity corrections.

    Notes
    -----
    The default is False for backwards compatibility: the Ghana exposure
    files in this project were built with the bare Rothfusz form, and
    changing the default would silently alter published estimates. Use
    ``adjustments=True`` for new work - it is the meteorologically correct
    version, and differs by up to ~1.5 degrees C in dry or very humid air.
    """
    t_c = np.asarray(t_c, dtype=float)
    rh = np.asarray(rh, dtype=float)
    t_f = celsius_to_fahrenheit(t_c)

    rothfusz = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 6.83783e-3 * t_f**2
        - 5.481717e-2 * rh**2
        + 1.22874e-3 * t_f**2 * rh
        + 8.5282e-4 * t_f * rh**2
        - 1.99e-6 * t_f**2 * rh**2
    )

    if not adjustments:
        hi_f = np.where(t_f < 80.0, t_f, rothfusz)
        return fahrenheit_to_celsius(hi_f)

    # Steadman's simple form, used to decide whether the full regression applies.
    simple = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)
    use_full = ((simple + t_f) / 2.0) >= 80.0

    adjusted = rothfusz.copy()

    # Low humidity, warm: subtract a correction.
    low_rh = (rh < 13.0) & (t_f >= 80.0) & (t_f <= 112.0)
    adjusted = np.where(
        low_rh,
        adjusted - ((13.0 - rh) / 4.0) * np.sqrt(np.clip((17.0 - np.abs(t_f - 95.0)) / 17.0, 0, None)),
        adjusted,
    )

    # High humidity, warm: add a correction.
    high_rh = (rh > 85.0) & (t_f >= 80.0) & (t_f <= 87.0)
    adjusted = np.where(high_rh, adjusted + ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0), adjusted)

    hi_f = np.where(use_full, adjusted, simple)
    return fahrenheit_to_celsius(hi_f)


def wind_speed(u, v):
    """Scalar wind speed from orthogonal components (same units in and out)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.sqrt(u**2 + v**2)


def metres_to_mm(x):
    """Metres -> millimetres. ERA5 total precipitation is accumulated in metres."""
    return np.asarray(x, dtype=float) * 1000.0


def joules_to_megajoules(x):
    """J/m^2 -> MJ/m^2. ERA5 radiation fluxes accumulate in J/m^2."""
    return np.asarray(x, dtype=float) / 1e6
