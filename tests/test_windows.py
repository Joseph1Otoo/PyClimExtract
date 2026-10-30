import datetime as dt

import pandas as pd
import pytest

from pyclimextract import build_windows, validate_windows, window_key
from pyclimextract.adapters import dhs


@pytest.fixture
def points():
    return pd.DataFrame(
        {
            "point_id": [1, 2, 3],
            "lat": [5.02, 6.70, 9.40],
            "lon": [-2.74, -1.62, -0.84],
            "visit": pd.to_datetime(["2022-06-15", "2022-07-15", "2022-08-15"]),
        }
    )


class TestBuildWindows:
    def test_lookback_is_inclusive_of_both_endpoints(self, points):
        w = build_windows(points, lookback_days=29, date_column="visit")
        span = (w.loc[0, "end_date"] - w.loc[0, "start_date"]).days
        assert span == 29  # 30 calendar days inclusive

    def test_single_reference_date_applies_to_all(self, points):
        w = build_windows(points, lookback_days=6, reference_date="2022-12-31")
        assert set(w["end_date"]) == {dt.date(2022, 12, 31)}
        assert len(w) == 3

    def test_requires_exactly_one_date_source(self, points):
        with pytest.raises(ValueError):
            build_windows(points, lookback_days=7)
        with pytest.raises(ValueError):
            build_windows(points, lookback_days=7,
                          reference_date="2022-01-01", date_column="visit")

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            build_windows(pd.DataFrame({"point_id": [1]}),
                          lookback_days=7, reference_date="2022-01-01")

    def test_duplicates_collapse(self, points):
        doubled = pd.concat([points, points], ignore_index=True)
        w = build_windows(doubled, lookback_days=29, date_column="visit")
        assert len(w) == 3


class TestValidateWindows:
    def test_swapped_coordinates_are_caught(self):
        bad = pd.DataFrame({
            "point_id": [1], "lat": [120.0], "lon": [5.0],
            "start_date": [dt.date(2022, 1, 1)], "end_date": [dt.date(2022, 1, 30)],
        })
        with pytest.raises(ValueError, match="latitude"):
            validate_windows(bad)

    def test_reversed_dates_are_caught(self):
        bad = pd.DataFrame({
            "point_id": [1], "lat": [5.0], "lon": [-1.0],
            "start_date": [dt.date(2022, 2, 1)], "end_date": [dt.date(2022, 1, 1)],
        })
        with pytest.raises(ValueError, match="start_date"):
            validate_windows(bad)


class TestWindowKey:
    def test_stable_across_input_types(self):
        a = window_key(7, dt.date(2022, 1, 1), dt.date(2022, 1, 30))
        b = window_key(7, "2022-01-01", "2022-01-30")
        c = window_key(7, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-30"))
        assert a == b == c == "era5_7_2022-01-01_2022-01-30"


class TestCMC:
    def test_cmc_one_is_january_1900(self):
        assert dhs.cmc_to_date(1) == pd.Timestamp("1900-01-15")

    def test_known_cmc(self):
        # 1474 = 1900 + 122 years, month 10 -> October 2022
        assert dhs.cmc_to_date(1474) == pd.Timestamp("2022-10-15")

    def test_december_boundary(self):
        # CMC 1476 is month 12 of 2022, not month 0 of 2023.
        assert dhs.cmc_to_date(1476) == pd.Timestamp("2022-12-15")
        assert dhs.cmc_to_date(1477) == pd.Timestamp("2023-01-15")

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            dhs.cmc_to_date(0)


class TestDHSGPS:
    def test_reads_suffixed_headers_and_drops_missing_fixes(self, tmp_path):
        csv = tmp_path / "gps.csv"
        csv.write_text(
            "DHSCLUST,LATNUM,N,19,11,LONGNUM\n"
            "1,5.02,x,y,z,-2.74\n"
            "2,0.0,x,y,z,0.0\n",
            encoding="utf-8",
        )
        out = dhs.read_dhs_gps(csv)
        assert list(out.columns) == ["point_id", "lat", "lon"]
        # Cluster 2 has no GPS fix and must be dropped.
        assert out["point_id"].tolist() == [1]
