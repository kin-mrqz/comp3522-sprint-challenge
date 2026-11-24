# EDA - AQHI sprint challenge

Exploratory analysis of `data/raw/aqhi_data.csv` (EPD hourly AQHI, 2021-01-01 to 2026-06-30).
46 figures, 62 tables, 25 written findings. No model is fit.

Start with `results/EDA_REPORT.md`. The machine-readable list of findings is
`results/tables/t55_all_findings.csv`; headline numbers for slides are in
`results/headline_numbers.json`.

## Layout

```
EDA/
  code/
    eda_lib.py              shared paths, station metadata, plot style
    00_build_processed.py   raw -> data/processed (parse, flags, aggregates)
    01_profile.py           data trust: missingness, substitution, censoring
    02_temporal.py          trend, season, diurnal, weekday, ACF, transitions
    03_episodes.py          high days, episodes, persistence
    04_spatial.py           station levels, co-movement, spread, type, tail
    05_decision.py          month-hour risk, safe windows, day transitions
    06_report.py            findings sheet, inventories, completeness check
  results/
    figures/                F01-F46 PNG
    tables/                 t00-t55 CSV
    EDA_REPORT.md           the reading report
    EDA_FINDINGS.md         generated findings + inventories
    figure_inventory.csv
    table_inventory.csv
    headline_numbers.json
```

## Run

From the repository root:

```bash
.venv/bin/python EDA/code/00_build_processed.py
.venv/bin/python EDA/code/01_profile.py
.venv/bin/python EDA/code/02_temporal.py
.venv/bin/python EDA/code/03_episodes.py
.venv/bin/python EDA/code/04_spatial.py
.venv/bin/python EDA/code/05_decision.py
.venv/bin/python EDA/code/06_report.py
```

The venv is the repo `.venv` (pandas, numpy, matplotlib, pyarrow, scipy). Scripts take no arguments;
edit the constants at the top of a script if you need a different window.

## Conventions

- Hour 1..24 -> `hour_of_day` 0..23, hour beginning. Hong Kong local time.
- `Daily Max` rows are excluded from the hourly table; they equal the hourly maximum.
- `10+` is 10 with `censored = True`; `*` is `substituted = True`; `-` is missing. No imputation.
- Bands: Low 1-3, Moderate 4-6, High 7, Very High 8-10, Serious 10+.
- Station type: 15 general, 3 roadside (Causeway Bay, Central, Mong Kok); Tap Mun is the background station.

## Known data gaps

- 38 absent dates: 2023-08-25..31, 2024-12-01..31.
- 17 dates present but blank for all stations: 2023-01-07, 2023-06-27, 2023-10-01..15.
- Tap Mun additionally blank 2023-09-07..25 and all of 2023-10.
- EPD changed the AQHI compilation on 2025-03-22; pre/post is not strictly one instrument.

## Not covered here

Task 2(a) needs one or two weather variables from the course project. The repo's `context/hko/`
contains citation notes only - the weather tables were left out on purpose - so there is no weather
join in this EDA. That is the next step, not a missing chart.
