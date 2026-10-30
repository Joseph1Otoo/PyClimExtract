"""PyClimExtract tutorial: link any gridded dataset to any georeferenced records.

Runs offline on synthetic data. No account, no download, no licensed
microdata:

    python examples/tutorial.py

The same five steps work for ERA5, CHIRPS, MODIS, a regional model run,
or any raster stack you already hold, and for DHS, MICS, LSMS, a clinical
cohort or a field study - anything with coordinates and a date.
"""

from __future__ import annotations

import pandas as pd

from pyclimextract import (
    add_derived,
    build_windows,
    extract_from_dataset,
    summarise_windows,
)
from pyclimextract.demo import load_demo

pd.set_option("display.width", 100)


def main():
    gridded, survey = load_demo()

    print("=" * 68)
    print("STEP 0 - what we start with")
    print("=" * 68)
    print(f"gridded : {dict(gridded.sizes)}, variables {list(gridded.data_vars)}")
    print(f"survey  : {len(survey)} people at {survey.site_id.nunique()} sites")
    print(survey.head(3).to_string(index=False))

    # ----------------------------------------------------------------
    # STEP 1 - exposure windows
    # ----------------------------------------------------------------
    # The window table is the contract every later step consumes. One row
    # per (place, time span). lookback_days=29 gives a 30-day window:
    # both endpoints are included.
    #
    # Build windows from the PEOPLE, not the sites. Two people at one site
    # interviewed in different months get different windows - which is the
    # whole point, and what site-level exposure silently destroys.
    print("\n" + "=" * 68)
    print("STEP 1 - build exposure windows")
    print("=" * 68)

    points = survey.rename(columns={"site_id": "point_id"})
    windows = build_windows(points, lookback_days=29, date_column="interview_date")

    print(f"{len(points)} people -> {len(windows)} distinct windows "
          f"across {windows.point_id.nunique()} sites")
    multi = windows.groupby("point_id").size()
    print(f"sites with more than one window: {int((multi > 1).sum())}")
    print(windows.head(3).to_string(index=False))

    # ----------------------------------------------------------------
    # STEP 2 - extract the daily series
    # ----------------------------------------------------------------
    # extract_from_dataset takes ONE dataset spanning everything. Declare
    # which variables accumulate within a day and must be summed; the rest
    # are averaged. Summing vs averaging rainfall is the single most common
    # error in this kind of work, and it fails silently.
    print("\n" + "=" * 68)
    print("STEP 2 - extract daily values at each point")
    print("=" * 68)

    daily = extract_from_dataset(windows, gridded, accumulated=["rain_mm"])
    print(f"{len(daily)} daily rows "
          f"({daily.point_id.nunique()} sites x {daily.window_end.nunique()} window ends)")
    print(daily.head(3).to_string(index=False))

    # ----------------------------------------------------------------
    # STEP 3 - derived metrics (optional)
    # ----------------------------------------------------------------
    # add_derived computes humidity, heat index, wind and unit conversions
    # when the inputs are present. Our synthetic data has no dewpoint, so
    # nothing is added here - it does not error, it simply has nothing to do.
    daily = add_derived(daily)

    # ----------------------------------------------------------------
    # STEP 4 - summarise over exposure windows
    # ----------------------------------------------------------------
    # by=["point_id", "window_end"] keeps a site's two visits apart.
    # Without it they pool into one series and the trailing-days selection
    # spans both - assigning everyone the later visit's weather.
    print("\n" + "=" * 68)
    print("STEP 4 - summarise, 7 / 14 / 30 days")
    print("=" * 68)

    exposures = summarise_windows(
        daily,
        windows_days=(7, 14, 30),
        mean_vars=("t2m_c",),
        sum_vars=("rain_mm",),
        by=["point_id", "window_end"],
    )
    print(exposures.head(3).to_string(index=False))

    # ----------------------------------------------------------------
    # STEP 5 - join back to the people
    # ----------------------------------------------------------------
    print("\n" + "=" * 68)
    print("STEP 5 - attach exposure to each person")
    print("=" * 68)

    survey = survey.assign(window_end=survey["interview_date"].dt.date)
    linked = survey.merge(
        exposures.rename(columns={"point_id": "site_id"}),
        on=["site_id", "window_end"],
        how="left",
    )

    unmatched = int(linked["t2m_c_mean_30d"].isna().sum())
    print(f"{len(linked)} people linked, {unmatched} unmatched")
    print(linked[["person_id", "site_id", "interview_date",
                  "t2m_c_mean_30d", "rain_mm_sum_30d"]].head(5).to_string(index=False))

    # ----------------------------------------------------------------
    # Why the assignment choice matters
    # ----------------------------------------------------------------
    # Compare per-person exposure against one value per site. For people at
    # a site visited twice, these differ - by a whole season in the demo,
    # exactly as they did in the real Ghana data.
    print("\n" + "=" * 68)
    print("WHY THIS MATTERS - per-person vs per-site exposure")
    print("=" * 68)

    site_level = summarise_windows(
        daily, windows_days=(30,), mean_vars=("t2m_c",), sum_vars=("rain_mm",),
    ).rename(columns={"point_id": "site_id",
                      "t2m_c_mean_30d": "t2m_site",
                      "rain_mm_sum_30d": "rain_site"})

    cmp = linked.merge(site_level[["site_id", "t2m_site", "rain_site"]], on="site_id")
    cmp["t2m_gap"] = (cmp["t2m_c_mean_30d"] - cmp["t2m_site"]).abs()
    cmp["rain_gap"] = (cmp["rain_mm_sum_30d"] - cmp["rain_site"]).abs()

    affected = cmp[cmp["t2m_gap"] > 0.01]
    print(f"people whose exposure changes: {len(affected)} of {len(cmp)} "
          f"({100 * len(affected) / len(cmp):.0f}%)")
    if len(affected):
        print(f"  temperature gap : mean {affected.t2m_gap.mean():.2f} C, "
              f"max {affected.t2m_gap.max():.2f} C")
        print(f"  rainfall gap    : mean {affected.rain_gap.mean():.1f} mm, "
              f"max {affected.rain_gap.max():.1f} mm")
    print("\nIn the Ghana DHS 2022 this affected 555 women: up to 33 points of")
    print("humidity and 440 mm of rainfall - dry season versus wet season.")

    print("\n" + "=" * 68)
    print("Done. Swap load_demo() for your own dataset and records;")
    print("every step above is unchanged.")
    print("=" * 68)


if __name__ == "__main__":
    main()
