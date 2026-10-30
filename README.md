# PyClimExtract

Link gridded climate reanalysis to georeferenced survey data through
point-based exposure windows.

Given a table of locations and dates, PyClimExtract downloads ERA5 data for
each location's exposure window, aggregates hourly fields to daily, derives
humidity and heat-index metrics, and collapses the result into one row of
exposures per location. It knows nothing about any particular survey or
health outcome; DHS-specific helpers live in `pyclimextract.adapters.dhs`.

## Install

```bash
pip install -e ".[download,dev]"
```

The `download` extra pulls in `cdsapi`, needed only for fetching. Extraction
and summarising work on already-downloaded NetCDF files without it.
Downloading also requires a configured `~/.cdsapirc` and acceptance of the
ERA5 licence on the Copernicus Climate Data Store.

## Try it in one command

No account, no download, no licensed data — the tutorial runs on bundled
synthetic data:

```bash
python examples/tutorial.py
```

It walks through the whole workflow and ends by showing why the exposure
assignment matters, on a dataset built to mirror the real one.

## Data sources

| Source | Variables | Access |
|---|---|---|
| **ERA5** (Copernicus) | temperature, dewpoint, precipitation, wind, radiation, soil moisture | CDS API — needs a free account and `~/.cdsapirc` |
| **CHIRPS** (UCSB/IRI) | daily rainfall, 1981–present, 50°S–50°N | OPeNDAP — no account, no download |

```python
from pyclimextract.sources import extract_chirps

daily = extract_chirps(windows)          # that is the whole thing
```

Only the requested cells and days cross the network, so a 30-day window
at one point is kilobytes rather than the multi-gigabyte annual file.

Bring any other product through `extract_from_dataset` below.

## Any gridded dataset, any georeferenced records

`extract_from_dataset` takes **one** dataset spanning all your points and
dates — a CHIRPS stack, a MODIS composite, a regional model run, any
NetCDF or Zarr you already hold. Nothing in it is specific to ERA5 or to
any survey:

```python
from pyclimextract import build_windows, extract_from_dataset, summarise_windows

windows = build_windows(records, lookback_days=29, date_column="visit_date")
daily   = extract_from_dataset(windows, my_dataset, accumulated=["rainfall"])
exposure = summarise_windows(daily, windows_days=(7, 30),
                             by=["point_id", "window_end"])
```

`accumulated` names the variables that total within a day rather than
averaging. Declare it: a daily *mean* of hourly rainfall is 1/24 of the
true total and looks entirely plausible.

The ERA5-over-CDS route below is one convenience path on top of this.

## The ERA5 download pipeline

Four stages, each taking the same *window table* as its contract:

```python
import pandas as pd
from pyclimextract import (
    build_windows, fetch_windows, extract_daily, add_derived, summarise_windows
)

points = pd.DataFrame({
    "point_id": [1, 2],
    "lat": [5.02, 6.70],
    "lon": [-2.74, -1.62],
    "visit_date": pd.to_datetime(["2022-06-15", "2022-07-15"]),
})

windows = build_windows(points, lookback_days=29, date_column="visit_date")
fetch_windows(windows, out_dir="era5_nc", variables=("t2m", "d2m", "tp"))
daily = add_derived(extract_daily(windows, nc_dir="era5_nc"))
exposures = summarise_windows(daily, windows_days=(7, 14, 21, 30))
```

`exposures` has one row per `point_id` with columns like `t2m_c_mean_30d`,
`hi_c_mean_7d`, `tp_mm_sum_14d`, plus `n_days_{w}d` recording how many days
actually contributed.

### The window table

Every stage passes this same five-column frame:

| column | meaning |
|---|---|
| `point_id` | location identifier |
| `lat`, `lon` | decimal degrees, WGS84 |
| `start_date`, `end_date` | exposure window, both endpoints inclusive |

Because `fetch`, `extract` and `summarise` all consume it and derive
filenames through `window_key()`, the download and the extraction cannot
disagree about which file belongs to which point.

Note that `lookback_days=29` gives a **30-day** window: both endpoints are
included.

## DHS surveys

```python
import pandas as pd
from pyclimextract.adapters import dhs

gps = dhs.read_dhs_gps("GHGE8AFL.csv")
ind = pd.read_stata("GHIR8CFL.DTA", columns=["v001", "v008"])
windows = dhs.dhs_windows(gps, ind, lookback_days=29)
```

`read_dhs_gps` handles the type-suffixed headers DHS exports
(`LATNUM,N,19,11`) and drops clusters whose coordinates are (0, 0) — DHS's
code for "no GPS fix", which would otherwise pull climate data from the Gulf
of Guinea.

`cmc_to_date` converts DHS Century Month Codes, dating each interview to the
15th of the month; CMC carries no day component, so the error is bounded at
about two weeks.

## Things worth knowing

**Accumulated vs instantaneous variables.** ERA5 accumulates `tp` and `ssrd`
hourly; the rest are instantaneous. PyClimExtract sums the former and
averages the latter over each day. Averaging precipitation instead of
summing it yields 1/24 of the true total and looks entirely plausible —
this is the single most common ERA5 mistake.

**Grid cell selection.** Values are taken at the cell nearest the point's own
coordinates, not the centre of the downloaded bounding box. On a 0.25°
grid with a 0.25° pad those can be different cells.

**All-missing windows are NaN, not zero.** Summing a window with no usable
data would report zero rainfall, which no one can distinguish from a
genuinely dry month. Sums use `min_count=1` so the gap stays visible.

**Sources differ, and the difference is information.** Over 15 Ghana
clusters, 30-day ERA5 and CHIRPS rainfall totals correlate at r = 0.93
but differ by 24 mm on average, with ERA5 the drier of the two.
`examples/compare_sources.py` reproduces that comparison.

**Heat index.** `derive.heat_index` defaults to `adjustments=False`, the bare
Rothfusz regression. The full NWS algorithm — Steadman's simple form first,
plus low- and high-humidity corrections — is `adjustments=True` and differs
by up to about 1.5 °C in dry or very humid air. The default preserves
reproducibility of exposure files built with the earlier pipeline; prefer
`adjustments=True` for new work.

**Missing downloads are loud.** `extract_daily` warns about each window with
no file and raises an explanatory error if nothing was extracted, rather
than silently returning an empty frame.

## Tests

```bash
pytest
```

93 tests, no network and no CDS account required. Two further tests
reach the live CHIRPS service and are opt-in:

```bash
pytest -m live
```

Extraction is tested against synthetic NetCDF written by the fixtures — extraction is tested
against synthetic NetCDF files written by the fixtures. The derived-metric
tests check against physical identities (−40 °C = −40 °F, saturated air is
100% RH) and published NWS heat-index table values.

## Validation against the original pipeline

Run over the full Ghana DHS 2022 archive (687 windows, 618 clusters,
1,376 NetCDF files), PyClimExtract reproduces
`era5_exposures_7_14_21_30d_by_cluster.csv` — the file the published
analysis used — to within **5e-5** on all 16 exposure columns for all 618
clusters. That residual is float32 storage precision in the ERA5 files,
not a difference in method.

The same run also explained a long-standing discrepancy between the two
exposure files in the project. `03_build_exposures.py` omits the
`.tail(w)` step and averages a cluster's *entire* pooled daily series:

| cluster type | count | max difference in `t2m_mean_30d` |
|---|---|---|
| one interview window | 549 | 0.064 °C (31 days averaged instead of 30) |
| two interview windows | 69 | **1.49 °C** (62 days pooled instead of 30) |

Reproducing that script's logic with this package matches its output to
5e-5 as well, confirming the cause. Use
`examples/validate_against_original.py` to re-check after any change.

## Status

Version 0.1.0. The API is not yet stable.

Not yet covered: multi-point single-file downloads (one request per window
is wasteful for dense point sets), non-ERA5 reanalyses, and population-
weighted or area-averaged exposure as an alternative to nearest-cell.
