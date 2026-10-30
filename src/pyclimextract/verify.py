"""Integrity check over a directory of downloaded NetCDF files."""

from __future__ import annotations

import pathlib

import pandas as pd
import xarray as xr

from .naming import CANONICAL_VARIABLES, standardise

__all__ = ["verify_directory"]


def verify_directory(nc_dir, variables=None):
    """Open every ``.nc`` file in `nc_dir` and report what it contains.

    Parameters
    ----------
    nc_dir
        Directory of downloaded files.
    variables
        Canonical short names to check for. Defaults to all known ones.

    Returns
    -------
    DataFrame with one row per file: ``file``, ``opened``, ``n_times``,
    ``error``, and a boolean column per variable. An empty frame means the
    directory held no ``.nc`` files at all.
    """
    variables = list(variables or CANONICAL_VARIABLES)
    root = pathlib.Path(nc_dir)
    rows = []

    for path in sorted(root.glob("*.nc")):
        rec = {"file": path.name, "opened": False, "n_times": 0, "error": ""}
        rec.update({v: False for v in variables})

        try:
            with xr.open_dataset(path, engine="netcdf4") as ds:
                ds = standardise(ds)
                rec["opened"] = True
                rec["n_times"] = int(ds.sizes.get("time", 0))
                for v in variables:
                    rec[v] = v in ds.variables
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(rec)

    return pd.DataFrame(rows, columns=["file", "opened", "n_times", "error", *variables])
