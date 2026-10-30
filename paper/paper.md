---
title: 'PyClimExtract: time-anchored climate exposure windows for georeferenced survey data'
tags:
  - Python
  - climate and health
  - exposure assessment
  - reanalysis
  - household surveys
  - reproducible research
authors:
  - name: Joseph Otoo
    orcid: 0000-0000-0000-0000   # TODO: your ORCID
    affiliation: 1
    corresponding: true
affiliations:
  - name: Department of Computer Science and Information Systems, Ashesi University, Berekuso, Ghana
    index: 1
date: 30 October 2026
bibliography: paper.bib
---

# Summary

Studies of climate and health routinely need to know what the weather was
doing at a particular place during a particular span of days: the month
before a child was weighed, the week before a woman's haemoglobin was
measured. The ingredients are freely available — gridded reanalysis and
satellite products on one side, georeferenced household surveys on the
other — but joining them is left to bespoke scripts written afresh for
each study.

`PyClimExtract` is a Python package that performs that join. Given a
table of locations and dates, it constructs an exposure window for each
record, extracts the matching cells from a gridded product, aggregates
sub-daily data to daily values with the correct operation for each
variable, derives humidity and heat-index metrics, and reduces the result
to one row of exposures per record. It ships adapters for ERA5
[@hersbach2020] through the Copernicus Climate Data Store and for CHIRPS
daily rainfall [@funk2015] over OPeNDAP, and a source-agnostic path that
accepts any `xarray` dataset [@hoyer2017], so products it has never heard
of can be used without modification.

# Statement of need

Contextual climate data already accompanies some survey programmes. The
DHS Program distributes geospatial covariate datasets that attach
temperature and rainfall to survey clusters [@mayala2018], and the `rdhs`
package makes DHS data itself straightforward to obtain [@watson2019].
These resources describe the *long-run* climate of a place: the DHS
covariates are released as snapshots for a handful of years.

They cannot answer the question that acute-exposure epidemiology asks,
which is what conditions were like in the days immediately before an
individual was observed. Answering that requires aligning a gridded time
series to each respondent's own interview date, and no published tool
does it. Researchers therefore write the pipeline themselves, and the
same mistakes recur:

- **Averaging accumulated variables.** Reanalysis products accumulate
  precipitation and radiation within each time step. A daily *mean* of
  hourly rainfall is one twenty-fourth of the true total, and nothing in
  the output looks wrong.
- **Assigning exposure to the wrong time.** Where a survey cluster is
  visited across two months, a single cluster-level exposure value is
  ambiguous. Implementations commonly pool the days and summarise the
  trailing window, which silently assigns every respondent the later
  visit's weather. In the Ghana DHS 2022, this affects 69 of 618 clusters
  and 12% of women, displacing some of them by a whole season.
- **Silent gaps.** Missing-value sentinels summed into an exposure total,
  or a window with no usable data reported as zero rainfall rather than
  as missing.

Each of these produces plausible numbers, so none is caught by inspection.

`PyClimExtract` addresses them structurally. A single *window table* is
the contract between every stage, so the download and the extraction
cannot disagree about which data belongs to which record. Variables are
declared as accumulated or instantaneous and aggregated accordingly.
Window identity is carried through extraction, so records at one place
observed at different times stay distinct. Totals over windows with no
usable data return missing rather than zero. The derived meteorological
quantities are unit-tested against physical identities and published
reference values rather than against the package's own output.

The package was developed for an analysis of climate and anaemia among
4,775 women in the Ghana DHS 2022, and reproduces that study's
independently-written pipeline to within $5 \times 10^{-5}$ across all 618
clusters — the floor set by `float32` storage in the source files.

# Functionality

The workflow is four steps, each consuming the window table:

```python
from pyclimextract import (
    build_windows, extract_from_dataset, add_derived, summarise_windows
)
from pyclimextract.sources import open_chirps

windows  = build_windows(records, lookback_days=29, date_column="visit_date")
daily    = add_derived(extract_from_dataset(windows, open_chirps(),
                                            accumulated=["rain_mm"]))
exposure = summarise_windows(daily, windows_days=(7, 14, 21, 30),
                             by=["point_id", "window_end"])
```

Exposures are produced over several window lengths at once, which makes
the sensitivity of a result to the assumed exposure period a routine
check rather than extra work. A survey adapter interprets DHS cluster
files and century-month interview codes, and discards clusters with no
GPS fix, whose null coordinates would otherwise draw data from the ocean.
Both the displacement applied to DHS coordinates [@burgert2013] and the
resolution of the gridded product bound how precisely exposure can be
known, and the package documents this rather than obscuring it.

A tutorial runs on bundled synthetic data with no account, download, or
licensed microdata, so the workflow can be learned and tested before any
real data is obtained.

# Acknowledgements

This work was developed alongside research on climate and anaemia in
Ghana using data from The DHS Program. We thank the Copernicus Climate
Change Service and the Climate Hazards Center for maintaining the data
services the package reads.

# References
