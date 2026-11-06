"""NASA POWER source tests.

Offline tests replace the HTTP call with a fake. The one test that
touches the network is marked `live`:

    pytest -m live
"""

import numpy as np
import pandas as pd
import pytest

from pyclimextract import summarise_windows
from pyclimextract.sources import extract_power, fetch_power_point


def _windows():
    return pd.DataFrame({
        "point_id": ["a1", "a2", "b1"],
        "lat": [5.76, 5.76, 9.40],
        "lon": [-0.22, -0.22, -0.84],
        "start_date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-02-01"]),
        "end_date": pd.to_datetime(["2024-01-10", "2024-03-05", "2024-02-03"]),
    })


class FakeFetch:
    """Records calls and returns constant daily series."""

    def __init__(self):
        self.calls = []

    def __call__(self, lat, lon, start, end, parameters, community):
        self.calls.append((lat, lon, pd.Timestamp(start), pd.Timestamp(end)))
        idx = pd.date_range(start, end, freq="D")
        df = pd.DataFrame({"PRECTOTCORR": 2.0, "T2M": 30.0, "RH2M": 70.0}, index=idx)
        df.iloc[0, 0] = np.nan        # one missing rainfall day
        return df


class TestExtractPower:
    def test_one_request_per_location(self):
        fake = FakeFetch()
        extract_power(_windows(), fetch=fake)
        assert len(fake.calls) == 2

    def test_request_spans_all_windows_at_location(self):
        fake = FakeFetch()
        extract_power(_windows(), fetch=fake)
        lat, lon, start, end = fake.calls[0]
        assert (start, end) == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-05"))

    def test_each_window_gets_only_its_days(self):
        daily = extract_power(_windows(), fetch=FakeFetch())
        n = daily.groupby("point_id").size()
        assert n.to_dict() == {"a1": 10, "a2": 5, "b1": 3}

    def test_schema_and_renaming(self):
        daily = extract_power(_windows(), fetch=FakeFetch())
        assert list(daily.columns[:4]) == ["point_id", "time", "window_start", "window_end"]
        assert {"tp_mm", "t2m_c", "rh", "hi_c"} <= set(daily.columns)
        assert "PRECTOTCORR" not in daily.columns

    def test_heat_index_exceeds_temperature_when_humid(self):
        daily = extract_power(_windows(), fetch=FakeFetch())
        assert (daily["hi_c"] > daily["t2m_c"]).all()

    def test_heat_index_optional(self):
        daily = extract_power(_windows(), fetch=FakeFetch(), add_heat_index=False)
        assert "hi_c" not in daily.columns

    def test_feeds_summarise(self):
        daily = extract_power(_windows(), fetch=FakeFetch())
        out = summarise_windows(daily, windows_days=(10,), mean_vars=("t2m_c",),
                                sum_vars=("tp_mm",)).set_index("point_id")
        # a1: ten days of 2 mm, first one missing
        assert out.loc["a1", "tp_mm_sum_10d"] == pytest.approx(18.0)
        assert out.loc["a1", "t2m_c_mean_10d"] == pytest.approx(30.0)

    def test_window_outside_data_is_skipped(self):
        def short(lat, lon, start, end, parameters, community):
            idx = pd.date_range("2024-01-01", "2024-01-31", freq="D")
            return pd.DataFrame({"T2M": 30.0}, index=idx)

        daily = extract_power(_windows(), fetch=short, parameters=("T2M",))
        assert set(daily["point_id"]) == {"a1"}


@pytest.mark.live
def test_live_power_accra():
    df = fetch_power_point(5.60, -0.19, "2024-01-01", "2024-01-31")
    assert len(df) == 31
    assert df["T2M"].between(20, 35).all()
    assert df["PRECTOTCORR"].ge(0).all()
