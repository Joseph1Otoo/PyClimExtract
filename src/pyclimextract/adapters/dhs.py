"""DHS-specific helpers.

Kept apart from the core so that the rest of the package stays survey-agnostic.
Nothing in ``pyclimextract`` outside this module knows what DHS is.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["cmc_to_date", "read_dhs_gps", "dhs_windows"]

# Column name candidates, in the order DHS has used them.
_CLUSTER_ALIASES = ("DHSCLUST", "CLUSTER", "CLUSTER_ID", "V001", "IDCLUST", "CLUST")
_LAT_ALIASES = ("LATNUM", "LAT", "LATITUDE", "Y", "LAT_DEC", "LATITUDE_DECIMAL")
_LON_ALIASES = ("LONGNUM", "LON", "LONGITUDE", "X", "LON_DEC", "LONGITUDE_DECIMAL")


def cmc_to_date(cmc, day=15):
    """Century Month Code -> Timestamp.

    CMC counts months from January 1900 (CMC 1). DHS records interview dates
    as CMC, which has no day component, so a representative day must be
    chosen; the 15th (mid-month) is conventional and bounds the error at
    about two weeks.
    """
    cmc = int(cmc)
    if cmc < 1:
        raise ValueError(f"CMC must be >= 1, got {cmc}")
    year = 1900 + (cmc - 1) // 12
    month = ((cmc - 1) % 12) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _normalise_headers(columns):
    """DHS GPS exports carry type suffixes, e.g. ``LATNUM,N,19,11``."""
    return [
        str(c).split(",")[0].strip().replace(" ", "_").replace(".", "_").upper()
        for c in columns
    ]


def _pick(df, options, what):
    for opt in options:
        if opt in df.columns:
            return opt
    raise KeyError(
        f"could not find a {what} column; tried {list(options)}, "
        f"available: {list(df.columns)[:30]}"
    )


def read_dhs_gps(path):
    """Read a DHS GPS export into ``point_id``, ``lat``, ``lon``.

    Rows whose coordinates are (0, 0) are dropped: DHS uses that for
    clusters with no GPS fix, and left in place they would silently pull
    climate data from the Gulf of Guinea.
    """
    gps = pd.read_csv(path)
    gps.columns = _normalise_headers(gps.columns)

    cluster = _pick(gps, _CLUSTER_ALIASES, "cluster id")
    lat = _pick(gps, _LAT_ALIASES, "latitude")
    lon = _pick(gps, _LON_ALIASES, "longitude")

    out = gps.rename(columns={cluster: "point_id", lat: "lat", lon: "lon"})
    out["point_id"] = pd.to_numeric(out["point_id"], errors="coerce").astype("Int64")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")

    out = out[["point_id", "lat", "lon"]].dropna()
    missing_fix = (out["lat"] == 0) & (out["lon"] == 0)
    if missing_fix.any():
        out = out[~missing_fix]

    return out.drop_duplicates().reset_index(drop=True)


def dhs_windows(gps, individual, lookback_days=29,
                cluster_column="v001", cmc_column="v008"):
    """Build a window table from DHS GPS and individual-recode tables.

    Parameters
    ----------
    gps
        Output of :func:`read_dhs_gps`.
    individual
        DHS recode table with a cluster column and an interview-date CMC
        column (``v008`` in the IR file).
    lookback_days
        Days before the interview date. The window includes both endpoints,
        so the default of 29 gives a 30-day window.

    Returns
    -------
    Window table (see :mod:`pyclimextract.windows`).
    """
    from ..windows import build_windows

    for col in (cluster_column, cmc_column):
        if col not in individual.columns:
            raise KeyError(
                f"column {col!r} not found in the individual table; "
                f"first columns are {list(individual.columns)[:20]}"
            )

    ind = individual[[cluster_column, cmc_column]].rename(
        columns={cluster_column: "point_id", cmc_column: "cmc"}
    )
    ind["point_id"] = pd.to_numeric(ind["point_id"], errors="coerce").astype("Int64")
    ind = ind.dropna(subset=["point_id", "cmc"])
    ind["interview_date"] = ind["cmc"].map(cmc_to_date)

    points = ind.merge(gps, on="point_id", how="inner")
    if points.empty:
        raise ValueError(
            "no clusters matched between the GPS and individual tables - "
            "check that both use the same cluster numbering"
        )

    return build_windows(points, lookback_days=lookback_days,
                         date_column="interview_date")
