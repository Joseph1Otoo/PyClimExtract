"""Collapse daily series into per-point exposure summaries."""

from __future__ import annotations

import pandas as pd

__all__ = ["DEFAULT_MEAN_VARS", "DEFAULT_SUM_VARS", "summarise_windows"]

# Reasonable defaults for the derived columns add_derived() produces.
DEFAULT_MEAN_VARS = ("t2m_c", "hi_c", "rh", "wind10", "skt", "sp", "swvl1")
DEFAULT_SUM_VARS = ("tp_mm", "ssrd_MJm2")


def summarise_windows(
    daily,
    windows_days=(30,),
    mean_vars=DEFAULT_MEAN_VARS,
    sum_vars=DEFAULT_SUM_VARS,
    time_column="time",
    point_column="point_id",
    by=None,
):
    """Summarise a daily table over one or more trailing exposure windows.

    For each window length *w*, takes the last *w* days of each point's
    series and reduces it: mean for state variables, sum for accumulations.

    Parameters
    ----------
    daily
        Daily table from :func:`pyclimextract.extract.extract_daily`,
        typically after :func:`pyclimextract.extract.add_derived`.
    windows_days
        Window lengths in days, e.g. ``(7, 14, 21, 30)``.
    mean_vars, sum_vars
        Column names to average / total. Names absent from `daily` are
        skipped rather than raising, so the same call works whether or not
        the extended variables were downloaded.
    by
        Columns identifying one series to summarise. Defaults to
        ``[point_column]``. Pass ``["point_id", "window_end"]`` when a point
        has more than one exposure window and each should be summarised
        separately - otherwise the windows pool and the trailing-days
        selection spans both.

    Returns
    -------
    One row per group, with columns ``{var}_mean_{w}d`` and ``{var}_sum_{w}d``.

    Notes
    -----
    A window longer than the available series yields a summary over
    whatever days exist; it is not padded and does not error. Check
    ``n_days_{w}d`` in the output to see how many days actually contributed.
    """
    if time_column not in daily.columns:
        raise KeyError(f"{time_column!r} not found in daily table")

    group_cols = list(by) if by is not None else [point_column]
    for col in group_cols:
        if col not in daily.columns:
            raise KeyError(f"{col!r} not found in daily table")

    mean_present = [v for v in mean_vars if v in daily.columns]
    sum_present = [v for v in sum_vars if v in daily.columns]
    if not (mean_present or sum_present):
        raise ValueError(
            "none of the requested summary variables are present; "
            f"daily table has: {sorted(daily.columns)}"
        )

    ordered = daily.sort_values(group_cols + [time_column])
    out = None

    for w in windows_days:
        tail = ordered.groupby(group_cols, sort=False).tail(int(w))
        grouped = tail.groupby(group_cols, sort=False)

        agg = {f"{v}_mean_{w}d": (v, "mean") for v in mean_present}
        # min_count=1 so a window with no usable values comes back as NaN
        # rather than 0. A plain sum of all-missing days reports zero
        # rainfall, which is indistinguishable from a genuinely dry month
        # and is exactly the kind of failure that survives unnoticed.
        agg.update({
            f"{v}_sum_{w}d": (v, lambda s: s.sum(min_count=1))
            for v in sum_present
        })
        agg[f"n_days_{w}d"] = (time_column, "count")

        part = grouped.agg(**agg).reset_index()
        out = part if out is None else out.merge(part, on=group_cols, how="outer")

    return out
