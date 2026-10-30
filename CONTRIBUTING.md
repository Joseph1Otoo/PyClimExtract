# Contributing to PyClimExtract

Contributions are welcome, including from people who have never
contributed to a research package before.

## Getting help or reporting a problem

Open an issue on GitHub. Useful reports usually include:

- what you ran (the smallest script that shows the problem)
- what you expected and what happened instead
- the versions from `python -c "import pyclimextract, xarray, pandas; print(pyclimextract.__version__, xarray.__version__, pandas.__version__)"`

If the problem involves a particular data source, say which one and which
dates and coordinates — exposure bugs are almost always specific to a
place and a time.

Questions about *how to use* the package are also welcome as issues.
If the answer is not obvious from the README, that is a documentation
bug and we would like to know.

## Development setup

```bash
git clone https://github.com/<your-account>/PyClimExtract.git
cd PyClimExtract
pip install -e ".[dev,download]"
pytest
```

The test suite runs offline in a few seconds. Two tests reach the live
CHIRPS service and are opt-in:

```bash
pytest -m live
```

## Making a change

1. Open an issue first for anything substantial, so the design can be
   discussed before you spend time on it.
2. Add a test that fails before your change and passes after.
3. Keep the offline suite offline. Tests that need a network service get
   the `@pytest.mark.live` marker, so CI stays green when a data
   provider has an outage.
4. Match the surrounding style. Comments explain *why*, not *what*.

## Adding a data source

New sources go in `src/pyclimextract/sources/`. A source module needs to
return a plain `xarray.Dataset` with `time`, `latitude` and `longitude`
coordinates — everything downstream then works unchanged.

`sources/chirps.py` is the reference implementation. When writing one,
pay attention to:

- **Time encoding.** Products use calendars and epochs that `xarray`
  cannot always decode. Decode explicitly and test the first and last
  dates against known values.
- **Missing-value sentinels.** Blank them, and blank them at a point
  where the operation stays cheap. Masking a lazily-opened remote array
  can force the entire archive across the network.
- **Accumulated variables.** Say which variables total within a period
  rather than averaging. Getting this wrong is silent: a daily mean of
  hourly rainfall is 1/24 of the true total and looks plausible.
- **Coverage limits.** Document the spatial and temporal extent and warn
  when requested windows fall outside it, so users get a warning rather
  than quiet `NaN`.

## Scientific correctness

This package produces exposure estimates that end up in health research,
so a silent numerical error is worse than a crash. Derived quantities
(`derive.py`) are tested against physical identities and published
reference values rather than against their own output. Please keep that
standard: if you add a formula, cite its source in the docstring and test
it against a value you did not compute with this code.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
