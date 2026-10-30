"""The source-agnostic path: one dataset, any variables, no file naming."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyclimextract import build_windows, extract_from_dataset, summarise_windows
from pyclimextract.demo import load_demo, make_gridded_data, make_survey


@pytest.fixture
def grid():
    """Daily data, 1 unit of 'rain' per day, constant 'temp'."""
    time = pd.date_range("2022-01-01", periods=60, freq="D")
    lats = np.array([5.5, 5.25, 5.0])
    lons = np.array([-2.0, -1.75, -1.5])
    shape = (len(time), len(lats), len(lons))
    return xr.Dataset(
        {
            "temp": (("time", "latitude", "longitude"), np.full(shape, 25.0)),
            "rain": (("time", "latitude", "longitude"), np.full(shape, 1.0)),
        },
        coords={"time": time, "latitude": lats, "longitude": lons},
    )


@pytest.fixture
def windows():
    return pd.DataFrame({
        "point_id": [1],
        "lat": [5.25],
        "lon": [-1.75],
        "start_date": [dt.date(2022, 1, 11)],
        "end_date": [dt.date(2022, 2, 9)],   # 30 days inclusive
    })


class TestExtractFromDataset:
    def test_slices_to_the_window(self, grid, windows):
        daily = extract_from_dataset(windows, grid, accumulated=["rain"])
        assert len(daily) == 30
        assert daily["time"].min() == pd.Timestamp("2022-01-11")
        assert daily["time"].max() == pd.Timestamp("2022-02-09")

    def test_carries_window_identity(self, grid, windows):
        daily = extract_from_dataset(windows, grid, accumulated=["rain"])
        assert set(daily.columns) >= {"point_id", "time", "window_start", "window_end"}

    def test_accumulated_variables_are_summed(self, grid, windows):
        daily = extract_from_dataset(windows, grid, accumulated=["rain"])
        out = summarise_windows(daily, windows_days=(30,),
                                mean_vars=("temp",), sum_vars=("rain",))
        assert out.loc[0, "rain_sum_30d"] == pytest.approx(30.0)
        assert out.loc[0, "temp_mean_30d"] == pytest.approx(25.0)

    def test_nothing_accumulates_when_told_so(self, grid, windows):
        # With accumulated=[], rain is averaged, not totalled.
        daily = extract_from_dataset(windows, grid, accumulated=[])
        out = summarise_windows(daily, windows_days=(30,),
                                mean_vars=("temp", "rain"), sum_vars=())
        assert out.loc[0, "rain_mean_30d"] == pytest.approx(1.0)

    def test_unknown_accumulated_name_raises(self, grid, windows):
        with pytest.raises(KeyError, match="accumulated names"):
            extract_from_dataset(windows, grid, accumulated=["nope"])

    def test_picks_the_nearest_cell(self, grid, windows):
        g = grid.copy()
        # Make one cell distinctive, then ask for a point inside it.
        g["temp"].loc[{"latitude": 5.0}] = 40.0
        w = windows.assign(lat=[5.02])
        daily = extract_from_dataset(w, g, accumulated=["rain"])
        assert daily["temp"].iloc[0] == pytest.approx(40.0)

    def test_missing_coordinate_raises_with_guidance(self, windows):
        bad = xr.Dataset({"temp": (("time",), np.zeros(5))},
                         coords={"time": pd.date_range("2022-01-01", periods=5)})
        with pytest.raises(KeyError, match="latitude"):
            extract_from_dataset(windows, bad)

    def test_window_outside_dataset_range_is_reported(self, grid):
        w = pd.DataFrame({
            "point_id": [1], "lat": [5.25], "lon": [-1.75],
            "start_date": [dt.date(2030, 1, 1)], "end_date": [dt.date(2030, 1, 30)],
        })
        with pytest.raises(RuntimeError, match="does not overlap"):
            extract_from_dataset(w, grid)

    def test_accepts_a_file_path(self, grid, windows, tmp_path):
        path = tmp_path / "grid.nc"
        grid.to_netcdf(path)
        daily = extract_from_dataset(windows, path, accumulated=["rain"])
        assert len(daily) == 30


class TestDemoData:
    def test_load_demo_shapes(self):
        gridded, survey = load_demo()
        assert set(gridded.data_vars) == {"t2m_c", "rain_mm"}
        assert len(survey) > 0
        assert {"site_id", "lat", "lon", "interview_date"} <= set(survey.columns)

    def test_demo_is_reproducible(self):
        a, _ = make_gridded_data(seed=7), None
        b = make_gridded_data(seed=7)
        np.testing.assert_allclose(a["t2m_c"].values, b["t2m_c"].values)

    def test_temperature_and_rain_are_strongly_opposed(self):
        # The planted structure the tutorial relies on: hot months are dry.
        # The real Ghana data gave r = -0.76; the generator targets about
        # -0.78, so anything weaker than -0.6 means the seasonality has
        # drifted and the suppression demonstration will not land.
        g = make_gridded_data(seed=0)
        t = g["t2m_c"].mean(dim=["latitude", "longitude"]).to_series()
        r = g["rain_mm"].mean(dim=["latitude", "longitude"]).to_series()
        monthly = pd.DataFrame({"t": t, "r": r}).resample("ME").mean()
        assert monthly["t"].corr(monthly["r"]) < -0.6

    def test_some_sites_are_visited_twice(self):
        survey = make_survey(seed=0)
        visits = survey.groupby("site_id")["interview_date"].nunique()
        assert (visits > 1).any()

    def test_interviews_are_dated_to_the_month(self):
        # People at one site in one month must SHARE a window, as they do
        # in a real survey. If every person had their own interview day the
        # per-person vs per-site comparison would be trivially 100% and the
        # tutorial's lesson would not transfer to real data.
        survey = make_survey(seed=0)
        assert (survey["interview_date"].dt.day == 15).all()
        per_site = survey.groupby("site_id")["interview_date"].nunique()
        assert per_site.max() <= 2

    def test_affected_share_resembles_the_real_data(self):
        # In the Ghana DHS 2022, 12% of women sat in a twice-visited
        # cluster. The demo should land in the same ballpark, not at 99%.
        survey = make_survey(seed=0)
        per_site = survey.groupby("site_id")["interview_date"].nunique()
        multi = per_site[per_site > 1].index
        share = survey["site_id"].isin(multi).mean()
        assert 0.10 < share < 0.45

    def test_demo_runs_through_the_whole_pipeline(self):
        gridded, survey = load_demo()
        points = survey.rename(columns={"site_id": "point_id"})
        w = build_windows(points, lookback_days=29, date_column="interview_date")
        daily = extract_from_dataset(w, gridded, accumulated=["rain_mm"])
        out = summarise_windows(daily, windows_days=(7, 30),
                                mean_vars=("t2m_c",), sum_vars=("rain_mm",),
                                by=["point_id", "window_end"])
        assert len(out) == len(w)
        assert out["rain_mm_sum_30d"].notna().all()
