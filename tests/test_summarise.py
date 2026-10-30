import numpy as np
import pandas as pd
import pytest

from pyclimextract import summarise_windows


@pytest.fixture
def daily():
    """Two points, 30 days each. Point 1 warms linearly; point 2 is constant."""
    days = pd.date_range("2022-06-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "point_id": [1] * 30 + [2] * 30,
            "time": list(days) * 2,
            "t2m_c": list(np.arange(30, dtype=float)) + [25.0] * 30,
            "tp_mm": [1.0] * 30 + [2.0] * 30,
            "rh": [60.0] * 60,
            "hi_c": [28.0] * 60,
        }
    )


class TestSummariseWindows:
    def test_sums_accumulations_and_means_states(self, daily):
        out = summarise_windows(daily, windows_days=(30,))
        p2 = out[out.point_id == 2].iloc[0]
        assert p2["tp_mm_sum_30d"] == pytest.approx(60.0)
        assert p2["t2m_c_mean_30d"] == pytest.approx(25.0)

    def test_window_takes_the_trailing_days(self, daily):
        # Point 1 runs 0..29; the last 7 days are 23..29, mean 26.
        out = summarise_windows(daily, windows_days=(7,))
        p1 = out[out.point_id == 1].iloc[0]
        assert p1["t2m_c_mean_7d"] == pytest.approx(26.0)

    def test_shorter_window_is_a_subset_not_the_whole_series(self, daily):
        out = summarise_windows(daily, windows_days=(7, 30))
        p2 = out[out.point_id == 2].iloc[0]
        assert p2["tp_mm_sum_7d"] == pytest.approx(14.0)
        assert p2["tp_mm_sum_30d"] == pytest.approx(60.0)

    def test_multiple_windows_merge_on_one_row_per_point(self, daily):
        out = summarise_windows(daily, windows_days=(7, 14, 21, 30))
        assert len(out) == 2
        for w in (7, 14, 21, 30):
            assert f"t2m_c_mean_{w}d" in out.columns

    def test_day_count_is_reported(self, daily):
        out = summarise_windows(daily, windows_days=(7,))
        assert out["n_days_7d"].tolist() == [7, 7]

    def test_short_series_does_not_silently_pad(self):
        short = pd.DataFrame({
            "point_id": [1, 1, 1],
            "time": pd.date_range("2022-06-01", periods=3, freq="D"),
            "t2m_c": [20.0, 22.0, 24.0],
            "tp_mm": [1.0, 1.0, 1.0],
        })
        out = summarise_windows(short, windows_days=(30,))
        # Only 3 days contributed - the caller can see that and decide.
        assert out.loc[0, "n_days_30d"] == 3
        assert out.loc[0, "tp_mm_sum_30d"] == pytest.approx(3.0)

    def test_unsorted_input_is_handled(self, daily):
        shuffled = daily.sample(frac=1.0, random_state=0).reset_index(drop=True)
        a = summarise_windows(daily, windows_days=(7,)).sort_values("point_id")
        b = summarise_windows(shuffled, windows_days=(7,)).sort_values("point_id")
        np.testing.assert_allclose(
            a["t2m_c_mean_7d"].to_numpy(), b["t2m_c_mean_7d"].to_numpy()
        )

    def test_absent_variables_are_skipped_not_fatal(self, daily):
        out = summarise_windows(daily.drop(columns=["rh", "hi_c"]), windows_days=(7,))
        assert "t2m_c_mean_7d" in out.columns
        assert "rh_mean_7d" not in out.columns

    def test_no_recognised_variables_raises(self, daily):
        with pytest.raises(ValueError, match="none of the requested"):
            summarise_windows(daily[["point_id", "time"]], windows_days=(7,))

    def test_missing_time_column_raises(self, daily):
        with pytest.raises(KeyError):
            summarise_windows(daily.drop(columns=["time"]), windows_days=(7,))

    def test_all_missing_sums_to_nan_not_zero(self):
        # A window with no usable data must not report zero rainfall -
        # that is indistinguishable from a genuinely dry month.
        d = pd.DataFrame({
            "point_id": [1] * 5 + [2] * 5,
            "time": list(pd.date_range("2022-06-01", periods=5)) * 2,
            "tp_mm": [np.nan] * 5 + [1.0] * 5,
            "t2m_c": [np.nan] * 5 + [25.0] * 5,
        })
        out = summarise_windows(d, windows_days=(5,)).sort_values("point_id")
        assert np.isnan(out.iloc[0]["tp_mm_sum_5d"])
        assert out.iloc[1]["tp_mm_sum_5d"] == pytest.approx(5.0)

    def test_partially_missing_still_sums_what_is_there(self):
        d = pd.DataFrame({
            "point_id": [1] * 4,
            "time": pd.date_range("2022-06-01", periods=4),
            "tp_mm": [2.0, np.nan, 3.0, np.nan],
            "t2m_c": [25.0] * 4,
        })
        out = summarise_windows(d, windows_days=(4,))
        assert out.loc[0, "tp_mm_sum_4d"] == pytest.approx(5.0)
