"""Extraction tests against synthetic NetCDF files - no CDS account needed."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyclimextract import add_derived, extract_daily, window_files, window_key


def _make_nc(path, lat_centre, lon_centre, start, hours=48, t2m_by_cell=None):
    """Write a small 3x3-cell, hourly ERA5-shaped file."""
    lats = np.round(np.array([lat_centre + 0.25, lat_centre, lat_centre - 0.25]), 4)
    lons = np.round(np.array([lon_centre - 0.25, lon_centre, lon_centre + 0.25]), 4)
    times = pd.date_range(start, periods=hours, freq="h")

    if t2m_by_cell is None:
        t2m = np.full((hours, 3, 3), 300.0)
    else:
        t2m = np.broadcast_to(np.asarray(t2m_by_cell, dtype=float), (hours, 3, 3)).copy()

    ds = xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), t2m),
            "d2m": (("time", "latitude", "longitude"), np.full((hours, 3, 3), 295.0)),
            # 1 mm per hour, in metres, so a 24-hour day totals 24 mm.
            "tp": (("time", "latitude", "longitude"), np.full((hours, 3, 3), 0.001)),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)


@pytest.fixture
def one_window(tmp_path):
    lat, lon = 5.0, -2.0
    start, end = dt.date(2022, 6, 1), dt.date(2022, 6, 2)
    stem = window_key(1, start, end)
    _make_nc(tmp_path / f"{stem}.nc", lat, lon, "2022-06-01")
    windows = pd.DataFrame(
        {"point_id": [1], "lat": [lat], "lon": [lon],
         "start_date": [start], "end_date": [end]}
    )
    return windows, tmp_path


class TestWindowFiles:
    def test_finds_single_file(self, one_window):
        _, d = one_window
        assert len(window_files(d, 1, dt.date(2022, 6, 1), dt.date(2022, 6, 2))) == 1

    def test_finds_split_instant_accum_pair(self, tmp_path):
        stem = window_key(9, dt.date(2022, 6, 1), dt.date(2022, 6, 2))
        for suffix in ("_instant", "_accum"):
            _make_nc(tmp_path / f"{stem}{suffix}.nc", 5.0, -2.0, "2022-06-01")
        found = window_files(tmp_path, 9, dt.date(2022, 6, 1), dt.date(2022, 6, 2))
        assert len(found) == 2

    def test_returns_empty_when_absent(self, tmp_path):
        assert window_files(tmp_path, 404, dt.date(2022, 6, 1), dt.date(2022, 6, 2)) == []


class TestExtractDaily:
    def test_hourly_collapses_to_one_row_per_day(self, one_window):
        windows, d = one_window
        daily = extract_daily(windows, d)
        assert len(daily) == 2  # 48 hours -> 2 days

    def test_precipitation_is_summed_not_averaged(self, one_window):
        windows, d = one_window
        daily = add_derived(extract_daily(windows, d))
        # 24 hours x 1 mm. An averaged value would be 1.0 - the classic error.
        assert daily["tp_mm"].iloc[0] == pytest.approx(24.0)

    def test_temperature_is_averaged(self, one_window):
        windows, d = one_window
        daily = add_derived(extract_daily(windows, d))
        assert daily["t2m_c"].iloc[0] == pytest.approx(300.0 - 273.15)

    def test_selects_cell_nearest_the_point_not_the_box_centre(self, tmp_path):
        # Point sits in the southern cell; only that cell is 310 K.
        lat_centre, lon_centre = 5.0, -2.0
        t2m_by_cell = np.array([[300.0, 300.0, 300.0],
                                [300.0, 300.0, 300.0],
                                [310.0, 310.0, 310.0]])
        stem = window_key(1, dt.date(2022, 6, 1), dt.date(2022, 6, 2))
        _make_nc(tmp_path / f"{stem}.nc", lat_centre, lon_centre,
                 "2022-06-01", t2m_by_cell=t2m_by_cell)

        windows = pd.DataFrame({
            "point_id": [1], "lat": [lat_centre - 0.25], "lon": [lon_centre],
            "start_date": [dt.date(2022, 6, 1)], "end_date": [dt.date(2022, 6, 2)],
        })
        daily = add_derived(extract_daily(windows, tmp_path))
        assert daily["t2m_c"].iloc[0] == pytest.approx(310.0 - 273.15)

    def test_missing_window_raises_when_asked(self, tmp_path):
        windows = pd.DataFrame({
            "point_id": [77], "lat": [5.0], "lon": [-2.0],
            "start_date": [dt.date(2022, 6, 1)], "end_date": [dt.date(2022, 6, 2)],
        })
        with pytest.raises(FileNotFoundError):
            extract_daily(windows, tmp_path, on_missing="raise")

    def test_all_missing_gives_an_explanatory_error(self, tmp_path):
        windows = pd.DataFrame({
            "point_id": [77], "lat": [5.0], "lon": [-2.0],
            "start_date": [dt.date(2022, 6, 1)], "end_date": [dt.date(2022, 6, 2)],
        })
        with pytest.raises(RuntimeError, match="No data extracted"):
            extract_daily(windows, tmp_path, on_missing="warn")

    def test_point_id_is_carried_through(self, one_window):
        windows, d = one_window
        daily = extract_daily(windows, d)
        assert set(daily["point_id"]) == {1}


class TestAddDerived:
    def test_adds_expected_columns(self, one_window):
        windows, d = one_window
        daily = add_derived(extract_daily(windows, d))
        for col in ("t2m_c", "rh", "hi_c", "tp_mm"):
            assert col in daily.columns

    def test_humidity_is_plausible(self, one_window):
        windows, d = one_window
        daily = add_derived(extract_daily(windows, d))
        assert (daily["rh"].between(0, 100)).all()
