"""CHIRPS source tests.

The standardisation logic is tested offline against a fake IRI-shaped
dataset. The one test that touches the network is marked `live` and
skipped unless explicitly requested:

    pytest -m live
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyclimextract import extract_from_dataset, summarise_windows
from pyclimextract.sources import chirps


def _fake_iri(n_days=40, start_julian=2459580.5):
    """A dataset shaped the way the IRI Data Library serves CHIRPS."""
    T = start_julian + np.arange(n_days, dtype=float)
    Y = np.array([5.125, 5.375, 5.625])
    X = np.array([-1.375, -1.125, -0.875])
    prcp = np.full((n_days, len(Y), len(X)), 2.0)
    prcp[0, :, :] = -9999.0          # CHIRPS missing sentinel
    ds = xr.Dataset(
        {"prcp": (("T", "Y", "X"), prcp)},
        coords={"T": T, "Y": Y, "X": X},
    )
    ds["T"].attrs["units"] = "julian_day"
    return ds


class TestStandardise:
    def test_renames_coordinates(self):
        ds = chirps.standardise_chirps(_fake_iri())
        assert {"time", "latitude", "longitude"} <= set(ds.coords)
        assert not {"T", "Y", "X"} & set(ds.coords)

    def test_decodes_julian_days(self):
        ds = chirps.standardise_chirps(_fake_iri(start_julian=2459580.5))
        # Julian day 2459580.5 is 2022-01-01.
        assert pd.Timestamp(ds["time"].values[0]) == pd.Timestamp("2022-01-01")

    def test_times_are_normalised_to_midnight(self):
        # IRI stamps noon; dates in the window table carry no time, so a
        # noon stamp would make slice() comparisons behave surprisingly.
        ds = chirps.standardise_chirps(_fake_iri())
        assert all(pd.Timestamp(t).hour == 0 for t in ds["time"].values)

    def test_renames_variable_and_sets_units(self):
        ds = chirps.standardise_chirps(_fake_iri())
        assert "rain_mm" in ds.data_vars
        assert "prcp" not in ds.data_vars
        assert ds["rain_mm"].attrs["units"] == "mm"

    def test_keeps_native_name_when_asked(self):
        ds = chirps.standardise_chirps(_fake_iri(), variable_name="prcp")
        assert "prcp" in ds.data_vars

    def test_missing_sentinel_becomes_nan_not_zero(self):
        # -9999 summed into a 30-day rainfall total would be catastrophic
        # and completely invisible.
        ds = chirps.standardise_chirps(_fake_iri())
        assert np.isnan(ds["rain_mm"].values[0]).all()
        assert ds["rain_mm"].values[1:].min() == pytest.approx(2.0)

    def test_negative_masking_can_be_disabled(self):
        ds = chirps.standardise_chirps(_fake_iri(), mask_negative=False)
        assert ds["rain_mm"].values[0].min() == pytest.approx(-9999.0)

    def test_missing_coordinate_raises(self):
        bad = xr.Dataset({"prcp": (("T",), np.zeros(3))}, coords={"T": [1.0, 2.0, 3.0]})
        with pytest.raises(KeyError, match="latitude"):
            chirps.standardise_chirps(bad)


class TestCoverage:
    def _windows(self, lat):
        return pd.DataFrame({
            "point_id": [1], "lat": [lat], "lon": [-1.0],
            "start_date": [dt.date(2022, 1, 5)], "end_date": [dt.date(2022, 2, 3)],
        })

    def test_flags_points_beyond_fifty_degrees(self):
        bad = chirps.check_coverage(self._windows(63.0))
        assert len(bad) == 1

    def test_passes_points_inside_coverage(self):
        assert len(chirps.check_coverage(self._windows(6.0))) == 0

    def test_flags_dates_outside_the_dataset(self):
        ds = chirps.standardise_chirps(_fake_iri())
        w = pd.DataFrame({
            "point_id": [1], "lat": [5.2], "lon": [-1.1],
            "start_date": [dt.date(1975, 1, 1)], "end_date": [dt.date(1975, 1, 30)],
        })
        assert len(chirps.check_coverage(w, ds)) == 1


class TestEndToEnd:
    def test_chirps_feeds_the_generic_extractor(self):
        ds = chirps.standardise_chirps(_fake_iri(n_days=40))
        windows = pd.DataFrame({
            "point_id": [1], "lat": [5.3], "lon": [-1.1],
            "start_date": [dt.date(2022, 1, 11)], "end_date": [dt.date(2022, 2, 9)],
        })
        daily = extract_from_dataset(windows, ds, accumulated=["rain_mm"])
        out = summarise_windows(daily, windows_days=(30,),
                                mean_vars=(), sum_vars=("rain_mm",),
                                by=["point_id", "window_end"])
        # Days 11..40 are all 2.0 mm; the -9999 day is day 1, outside the window.
        assert out.loc[0, "rain_mm_sum_30d"] == pytest.approx(60.0)
        assert out.loc[0, "n_days_30d"] == 30


class TestMaskMissing:
    def test_blanks_negative_sentinels_in_a_table(self):
        df = pd.DataFrame({"rain_mm": [2.0, -9999.0, 3.0]})
        out = chirps.mask_missing(df)
        assert np.isnan(out["rain_mm"].iloc[1])
        assert out["rain_mm"].sum() == pytest.approx(5.0)

    def test_absent_column_is_a_no_op(self):
        df = pd.DataFrame({"other": [1.0]})
        assert chirps.mask_missing(df).equals(df)


@pytest.mark.live
class TestLive:
    def test_extract_chirps_end_to_end(self):
        windows = pd.DataFrame({
            "point_id": [1, 2],
            "lat": [6.0, 9.4],
            "lon": [-1.0, -0.84],
            "start_date": [dt.date(2022, 6, 1), dt.date(2022, 6, 1)],
            "end_date": [dt.date(2022, 6, 30), dt.date(2022, 6, 30)],
        })
        daily = chirps.extract_chirps(windows)
        assert len(daily) == 60
        assert set(daily["point_id"]) == {1, 2}
        # Southern Ghana in June is deep in the wet season.
        total = daily[daily.point_id == 1]["rain_mm"].sum()
        assert 50 < total < 800

    def test_opening_does_not_pull_the_whole_archive(self):
        # Regression: masking the remote dataset used to be eager, which
        # tried to allocate the entire 36 GB global record before any
        # subsetting. open_chirps must stay lazy.
        ds = chirps.open_chirps("0p25")
        assert ds["rain_mm"].size > 1e9        # the full archive is addressable
        point = ds["rain_mm"].sel(latitude=6.0, longitude=-1.0, method="nearest")
        values = point.sel(time=slice("2022-06-01", "2022-06-05")).values
        assert len(values) == 5                 # but only 5 numbers crossed the wire
