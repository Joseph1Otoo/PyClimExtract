"""A point with more than one exposure window must not pool them silently."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyclimextract import add_derived, extract_daily, summarise_windows, window_key


def _make_nc(path, start, t2m_k, hours=48):
    lats = np.array([5.25, 5.0, 4.75])
    lons = np.array([-2.25, -2.0, -1.75])
    times = pd.date_range(start, periods=hours, freq="h")
    xr.Dataset(
        {
            "t2m": (("time", "latitude", "longitude"), np.full((hours, 3, 3), t2m_k)),
            "d2m": (("time", "latitude", "longitude"), np.full((hours, 3, 3), 290.0)),
            "tp": (("time", "latitude", "longitude"), np.full((hours, 3, 3), 0.001)),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    ).to_netcdf(path)


@pytest.fixture
def two_windows(tmp_path):
    """One point, two windows, 10 K apart - cold in June, hot in August."""
    specs = [
        (dt.date(2022, 6, 1), dt.date(2022, 6, 2), "2022-06-01", 295.0),
        (dt.date(2022, 8, 1), dt.date(2022, 8, 2), "2022-08-01", 305.0),
    ]
    for start, end, ts, t2m in specs:
        _make_nc(tmp_path / f"{window_key(1, start, end)}.nc", ts, t2m)

    windows = pd.DataFrame(
        {
            "point_id": [1, 1],
            "lat": [5.0, 5.0],
            "lon": [-2.0, -2.0],
            "start_date": [s for s, _, _, _ in specs],
            "end_date": [e for _, e, _, _ in specs],
        }
    )
    return windows, tmp_path


class TestWindowIdentity:
    def test_extract_carries_window_columns(self, two_windows):
        windows, d = two_windows
        daily = extract_daily(windows, d)
        assert "window_start" in daily.columns
        assert "window_end" in daily.columns
        assert daily["window_end"].nunique() == 2

    def test_pooling_is_the_default_and_spans_both_windows(self, two_windows):
        windows, d = two_windows
        daily = add_derived(extract_daily(windows, d))
        out = summarise_windows(daily, windows_days=(4,))
        # 4 days of trailing data pools 2 cold + 2 hot days -> midway.
        assert len(out) == 1
        assert out.loc[0, "t2m_c_mean_4d"] == pytest.approx(300.0 - 273.15)

    def test_grouping_by_window_keeps_them_separate(self, two_windows):
        windows, d = two_windows
        daily = add_derived(extract_daily(windows, d))
        out = summarise_windows(
            daily, windows_days=(2,), by=["point_id", "window_end"]
        ).sort_values("window_end")
        assert len(out) == 2
        np.testing.assert_allclose(
            out["t2m_c_mean_2d"].to_numpy(),
            np.array([295.0 - 273.15, 305.0 - 273.15]),
            atol=1e-6,
        )

    def test_precipitation_does_not_double_count_when_grouped(self, two_windows):
        windows, d = two_windows
        daily = add_derived(extract_daily(windows, d))
        pooled = summarise_windows(daily, windows_days=(30,))
        split = summarise_windows(daily, windows_days=(30,), by=["point_id", "window_end"])
        # Pooled sums all 4 days; each split window sums its own 2.
        assert pooled.loc[0, "tp_mm_sum_30d"] == pytest.approx(96.0)
        assert split["tp_mm_sum_30d"].tolist() == pytest.approx([48.0, 48.0])

    def test_unknown_by_column_raises(self, two_windows):
        windows, d = two_windows
        daily = add_derived(extract_daily(windows, d))
        with pytest.raises(KeyError):
            summarise_windows(daily, windows_days=(2,), by=["point_id", "nope"])

    def test_single_window_unaffected_by_grouping(self, tmp_path):
        start, end = dt.date(2022, 6, 1), dt.date(2022, 6, 2)
        _make_nc(tmp_path / f"{window_key(1, start, end)}.nc", "2022-06-01", 300.0)
        windows = pd.DataFrame({
            "point_id": [1], "lat": [5.0], "lon": [-2.0],
            "start_date": [start], "end_date": [end],
        })
        daily = add_derived(extract_daily(windows, tmp_path))
        a = summarise_windows(daily, windows_days=(2,))
        b = summarise_windows(daily, windows_days=(2,), by=["point_id", "window_end"])
        assert a.loc[0, "t2m_c_mean_2d"] == pytest.approx(b.loc[0, "t2m_c_mean_2d"])
