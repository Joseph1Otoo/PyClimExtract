import numpy as np
import pytest

from pyclimextract import derive


class TestConversions:
    def test_kelvin_to_celsius_freezing(self):
        assert derive.kelvin_to_celsius(273.15) == pytest.approx(0.0)

    def test_celsius_fahrenheit_roundtrip(self):
        c = np.array([-40.0, 0.0, 27.3, 100.0])
        back = derive.fahrenheit_to_celsius(derive.celsius_to_fahrenheit(c))
        np.testing.assert_allclose(back, c, atol=1e-10)

    def test_minus_forty_is_the_same_in_both_scales(self):
        assert derive.celsius_to_fahrenheit(-40.0) == pytest.approx(-40.0)

    def test_metres_to_mm(self):
        assert derive.metres_to_mm(0.0254) == pytest.approx(25.4)

    def test_joules_to_megajoules(self):
        assert derive.joules_to_megajoules(2.5e7) == pytest.approx(25.0)


class TestRelativeHumidity:
    def test_saturated_air_is_one_hundred_percent(self):
        # Dewpoint equal to temperature means saturation.
        assert derive.relative_humidity(300.0, 300.0) == pytest.approx(100.0)

    def test_dewpoint_below_temperature_is_unsaturated(self):
        rh = derive.relative_humidity(303.15, 293.15)  # 30 C air, 20 C dewpoint
        assert 50.0 < rh < 60.0

    def test_bounded_to_zero_hundred(self):
        # Supersaturated input (dewpoint above temperature) must still clip.
        rh = derive.relative_humidity(290.0, 300.0)
        assert rh == pytest.approx(100.0)

    def test_known_value(self):
        # 25 C air with 15 C dewpoint is ~53.8% RH by the Magnus formula.
        rh = derive.relative_humidity(298.15, 288.15)
        assert rh == pytest.approx(53.8, abs=0.5)

    def test_vectorised(self):
        rh = derive.relative_humidity(
            np.array([300.0, 300.0]), np.array([300.0, 290.0])
        )
        assert rh.shape == (2,)
        assert rh[0] > rh[1]


class TestHeatIndex:
    def test_below_eighty_fahrenheit_returns_air_temperature(self):
        # 20 C = 68 F, below the Rothfusz threshold.
        assert derive.heat_index(20.0, 50.0) == pytest.approx(20.0)

    def test_hot_humid_feels_hotter_than_air(self):
        hi = derive.heat_index(32.0, 80.0)
        assert hi > 32.0

    def test_hot_dry_feels_cooler_than_air(self):
        # The full NWS algorithm applies a low-humidity correction;
        # dry heat should not read hotter than the air temperature.
        hi = derive.heat_index(35.0, 10.0, adjustments=True)
        assert hi < 35.0

    def test_nws_reference_value(self):
        # NWS table: 90 F at 70% RH -> about 105 F.
        hi_c = derive.heat_index(
            derive.fahrenheit_to_celsius(90.0), 70.0, adjustments=True
        )
        assert derive.celsius_to_fahrenheit(hi_c) == pytest.approx(105.0, abs=1.5)

    def test_monotone_in_humidity_when_hot(self):
        rh = np.array([40.0, 60.0, 80.0])
        hi = derive.heat_index(np.full(3, 33.0), rh)
        assert np.all(np.diff(hi) > 0)

    def test_adjustments_flag_changes_dry_result(self):
        plain = derive.heat_index(35.0, 8.0, adjustments=False)
        adjusted = derive.heat_index(35.0, 8.0, adjustments=True)
        assert plain != pytest.approx(adjusted)

    def test_default_is_unadjusted_for_reproducibility(self):
        # The Ghana exposure files were built with the bare Rothfusz form.
        # If this test fails, previously published estimates no longer
        # reproduce from this package.
        assert derive.heat_index(35.0, 8.0) == pytest.approx(
            derive.heat_index(35.0, 8.0, adjustments=False)
        )


class TestWindSpeed:
    def test_pythagorean(self):
        assert derive.wind_speed(3.0, 4.0) == pytest.approx(5.0)

    def test_sign_independent(self):
        assert derive.wind_speed(-3.0, -4.0) == pytest.approx(5.0)

    def test_calm(self):
        assert derive.wind_speed(0.0, 0.0) == pytest.approx(0.0)
