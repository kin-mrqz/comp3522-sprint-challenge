# COMP3522 Sprint Challenge — Submission Code

Code for the time-series analysis of Hong Kong hourly **AQHI** (2021–2025)
linked to **Open-Meteo weather** (2023–2025), answering the Smoking Baby
safe-day scenario. This folder contains only the data-pulling, processing, and
analysis code (no deck-building code, no data files).

## Contents

```
submission/
├── scripts/
│   ├── fetch_weather.py        # download the project weather covariate (Open-Meteo)
│   ├── run_analysis.py         # clean AQHI, Q1 trends, Q2 weather links, logistic risk model
│   └── advanced_timeseries.py  # STL, ACF/PACF, ADF/KPSS, Mann-Kendall, PELT, periodogram, forecast
├── src/
│   ├── paths.py                # repo paths
│   └── io.py                   # table loading + figure saving helpers
└── requirements.txt
```

## Setup

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install statsmodels ruptures
```

`requirements.txt` covers the core stack; `statsmodels` and `ruptures` are
needed for the advanced time-series step and are installed above.

## Data (place before running)

- `data/raw/aqhi_data.csv` — EPD hourly AQHI table (Moodle).
- `data/external/weather/om_territory_hourly.csv` — produced by
  `scripts/fetch_weather.py` from the Hugging Face dataset
  `NotASI/COMP3522-HKO`.

```powershell
.venv\Scripts\python.exe scripts\fetch_weather.py
```

## Reproduce the analysis

From the folder that contains `scripts/` and `src/` (this `submission/` folder),
run in order:

```powershell
.venv\Scripts\python.exe scripts\fetch_weather.py        # weather covariate (once)
.venv\Scripts\python.exe scripts\run_analysis.py         # cleaning + Q1 + Q2 + risk model
.venv\Scripts\python.exe scripts\advanced_timeseries.py  # advanced time-series diagnostics
```

## Outputs written

- `outputs/figures/` — all charts (`q1_*`, `q2_*`, `ats_*`).
- `outputs/tables/` — all summary CSVs.
- `outputs/results.json` — every number in the report.
- `outputs/answers.md` — one claim/number/chart/limitation per question.
- `outputs/models/safe_day_model.joblib` — fitted risk model.

## How the questions are answered

1. **Q1 (trends):** annual / seasonal / diurnal / spatial / category analysis of
   hourly AQHI 2021–2025.
2. **Q2 (weather & safe days):** Spearman correlations of 6 weather variables
   with daily AQHI, plus a logistic risk model for P(AQHI ≥ 7) trained 2023–24
   and validated 2025; yields the safest/riskiest day conditions and a simple
   safe-day rule.
3. **Advanced TS:** STL decomposition, stationarity (ADF/KPSS), Mann-Kendall
   trend test, PELT change-points, periodogram, and an ETS forecasting backtest.

> Associations are ecological, not causal. The risk score is a screening/ranking
> aid, not a medical device. "Poor" air quality is defined as AQHI ≥ 7 (EPD High+).