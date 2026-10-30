"""Synthetic data so the tutorial runs with no account and no download.

Real survey microdata is licensed and real reanalysis archives are large,
so neither can ship with a package. These generators produce small,
realistic stand-ins with known structure: a seasonal climate field over
southern Ghana and a survey whose interviews are spread across months.

Because the signal is planted deliberately, the tutorial can state what
the right answer looks like - which a tutorial on real data cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["make_gridded_data", "make_survey", "load_demo"]

# A box over southern Ghana, on a 0.25 degree grid.
_LAT = np.arange(4.75, 9.01, 0.25)
_LON = np.arange(-3.25, 1.01, 0.25)


def make_gridded_data(start="2022-01-01", end="2022-12-31", seed=0):
    """A daily temperature and rainfall field with a seasonal cycle.

    Temperature falls with latitude and peaks in the dry season; rainfall
    runs in antiphase, as it does in West Africa. That antiphase is the
    point: it is what makes naive single-exposure analyses misleading.
    """
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, end, freq="D")

    day = time.dayofyear.to_numpy()[:, None, None]
    lat = _LAT[None, :, None]

    # Warmest around day 60 (February), coolest in the wet season.
    seasonal = 3.0 * np.cos(2 * np.pi * (day - 60) / 365.0)
    t2m_c = 28.0 - 0.35 * (lat - 4.75) + seasonal
    t2m_c = t2m_c + rng.normal(0, 0.6, size=(len(time), len(_LAT), len(_LON)))

    # Rain peaks around day 210 (late July), well out of phase with the
    # temperature peak. At this offset the monthly temperature-rainfall
    # correlation is about -0.78, close to the -0.76 measured in the Ghana
    # DHS 2022 analysis this package was built for. That opposition is what
    # makes each exposure mask the other in a single-exposure model.
    wet = np.clip(np.cos(2 * np.pi * (day - 210) / 365.0), 0, None)
    rain_mm = 12.0 * wet * rng.gamma(1.2, 1.0, size=(len(time), len(_LAT), len(_LON)))

    return xr.Dataset(
        {
            "t2m_c": (("time", "latitude", "longitude"), t2m_c.astype("float32")),
            "rain_mm": (("time", "latitude", "longitude"), rain_mm.astype("float32")),
        },
        coords={"time": time, "latitude": _LAT, "longitude": _LON},
        attrs={"source": "pyclimextract.demo - synthetic, not real observations"},
    )


def make_survey(n_sites=40, n_per_site=25, seed=0):
    """A survey of `n_sites` sites, interviewed across several months.

    Some sites are deliberately visited in two different months, which is
    the situation that makes site-level exposure ambiguous. In the Ghana
    DHS 2022 this affected 69 of 618 clusters.
    """
    rng = np.random.default_rng(seed)

    sites = pd.DataFrame({
        "site_id": np.arange(1, n_sites + 1),
        "lat": rng.uniform(5.0, 8.8, n_sites).round(4),
        "lon": rng.uniform(-3.0, 0.8, n_sites).round(4),
    })

    # Most sites get one visit month; a quarter get two.
    months = rng.choice([3, 5, 7, 9, 11], size=n_sites)
    two_visits = rng.random(n_sites) < 0.25

    rows = []
    for i, site in sites.iterrows():
        visit_months = [months[i]] + ([min(months[i] + 3, 12)] if two_visits[i] else [])
        for m in visit_months:
            for _ in range(n_per_site // len(visit_months)):
                rows.append({
                    "site_id": int(site.site_id),
                    "lat": site.lat,
                    "lon": site.lon,
                    # Dated to mid-month, not to the day. Household surveys
                    # typically record the interview MONTH (DHS stores a
                    # century-month code), so everyone interviewed at a site
                    # in the same month shares one exposure window. That is
                    # what makes a site visited twice ambiguous, and a
                    # day-resolution demo would hide the problem by giving
                    # every person a window of their own.
                    "interview_date": pd.Timestamp(2022, m, 15),
                    "age": int(rng.integers(15, 50)),
                    "urban": int(rng.random() < 0.45),
                })

    people = pd.DataFrame(rows)
    people.insert(0, "person_id", np.arange(1, len(people) + 1))
    return people


def load_demo(seed=0):
    """Return ``(gridded, survey)`` ready for the tutorial."""
    return make_gridded_data(seed=seed), make_survey(seed=seed)
