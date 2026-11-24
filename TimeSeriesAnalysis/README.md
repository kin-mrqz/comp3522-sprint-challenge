# TimeSeriesAnalysis - AQHI daily series

Deep time-series test of the daily Hong Kong AQHI series, built on the processed tables in
`data/processed/`. The analysis follows the STAT4601 method family (Box-Jenkins identification,
ARIMA/SARIMA, Ljung-Box, exponential smoothing, GARCH) and adds the tests the course notes do not
cover as explicit cross-checks (HAC trend tests, block bootstrap, Diebold-Mariano).

## Scope and splits

The AQHI export starts on 2021-01-01, so the requested "2020-2024" window cannot be satisfied at the
start. The modelling scope is 2021-2024; 2025 is the untouched test year.

| Window | Dates | Days | Role |
|---|---|---|---|
| train | 2021-01-01 to 2023-12-31 | 1,095 (1,071 observed) | identification, estimation, grid search |
| validation | 2024-01-01 to 2024-11-30 | 335 | model selection only |
| test | 2025-01-01 to 2025-12-31 | 365 | scored after selection |
| final fit | 2021-01-01 to 2024-11-30 | 1,430 (1,406 observed) | refit of the selected models before the test |

December 2024 is absent from the export, so the final fit stops on 2024-11-30. The 2025 forecast is
therefore 32 steps beyond the last observation at its first test date. That is the honest origin.

## Series

Daily territory mean AQHI: the mean of the 18 station daily means, one value per calendar day.
The 24 missing training days are kept as NaN and handled by the state-space Kalman filter for
SARIMAX; exponential smoothing uses linear interpolation and is reported as a benchmark only.

## Models

- Benchmarks: mean, naive, seasonal naive 7, seasonal naive 365, monthly climatology, linear trend,
  deterministic seasonal OLS.
- ARIMA(p,d,q), p,q = 0..3, d = 0 and d = 1.
- SARIMA(p,0,q)(P,D,Q)_7, p,q = 0..2, P,Q = 0..1, D = 0/1, plus a refinement around the leader.
- SARIMAX with Fourier annual terms (K = 1..4) or 11 month dummies, day-of-week dummies, and ARMA
  errors. This is the model that matches the actual seasonal structure.
- Exponential smoothing: SES, damped Holt, Holt-Winters weekly additive/multiplicative, plus an
  annual (s = 365) variant.
- GARCH family on the selected mean-model residuals: ARCH(1), GARCH(1,1), GARCH(1,2), GARCH(2,1),
  GARCH(1,1)-t, GJR-GARCH(1,1), EGARCH(1,1), with a 2025 one-step VaR filter.

## Selection rules

- AIC is compared only within one dependent variable, one differencing order and one regressor set.
  Cross-family choices use validation RMSE.
- A model is adequate only if Ljung-Box passes at K = 14, 21 and 30 with df = K - m, and no AR/MA
  parameter is pinned at the boundary (|estimate| > 0.98 or a root inside 1.02).
- The test year is never used for selection. The test metrics are the honest estimate for the
  selected models.

## Run

From the repository root, with the STAT4601 venv:

```bash
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/00_build_series.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/01_eda_stationarity.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/02_trend_tests.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/03_arima_grid.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/04_sarima_grid.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/05_sarimax_seasonal.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/06_exponential_smoothing.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/07_garch.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/08_forecast_eval.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/09_report.py
```

The SARIMA grid is the slow step (about 3-4 minutes). The rest is under a minute each.

## Layout

```
TimeSeriesAnalysis/
  code/                  tsa_lib.py + 00..09 numbered scripts
  results/
    data/                daily_series.csv, mean_resid.csv
    figures/             F01-F36 PNG
    tables/              t00-t34 CSV
    facts/               series.json, stationarity.json, trend.json, arima.json, ...
  report/
    AQHI_TimeSeries_Report.md
```

## Caveats carried into the report

- The AQHI compilation changed on 2025-03-22. Test errors are reported separately before and after
  that date.
- The 2024-12 gap and the 2023 gaps are real; no value is invented for them.
- `10+` cells are stored as 10, but the daily mean is far from the cap, so censoring does not bind.
- AIC across different exog sets, differencing orders or series is not a comparison. The tables say
  which comparisons are legal.
