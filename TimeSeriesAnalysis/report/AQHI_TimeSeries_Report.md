# AQHI daily series: ARIMA, SARIMA and GARCH test

Scope: the daily Hong Kong AQHI series, 2021-01-01 to 2024-11-30 for fitting, 2025 held out.
The AQHI export starts in 2021, so the requested 2020 start is not available; the modelling scope is
2021-2024. The 2025 test is untouched until the final section.

Deliverables: 35 figures and 35 tables in `results/`, code in `code/`, this report in `report/`.
The series is the mean of the 18 station daily means. 2024-12 is absent from the export, so the
final fit stops on 2024-11-30 and the first 2025 test date is 32 steps beyond the last observation.

## 1. Verdict first

**There is no trend.** The daily slope is +0.011 AQHI/year over 2021-2023 (HAC p = 0.93) and
-0.038/year over 2021-2024 (HAC p = 0.62). The sign flips only because 2024 is a low year; the
2021-2024 naive slope is not significant even with iid standard errors (p = 0.105). Monthly OLS gives
-0.036/year (HAC p = 0.66), Mann-Kendall p = 0.65, Sen slope -0.035/year, and a residual block
bootstrap 95% interval of -0.225 to +0.257. Annual means are 3.36, 3.29, 3.41, 3.21.

What the series has instead is a strong annual cycle (winter about 3.70, summer 2.54; STL seasonal
strength 0.53, trend strength 0.01). ADF with a constant rejects a unit root at every lag tested
(p < 0.001); KPSS does not reject stationarity (p = 0.10). The level is stationary around a
deterministic seasonal mean, so d = 0. The ACF decay is persistence (r1 = 0.831), not a unit root;
the effective sample is about 99 independent observations, not 1,071.

Model ranking on the 2024 validation year (RMSE, AQHI):

| Model | Validation RMSE | Adequate? |
|---|---|---|
| SARIMAX(1,0,2)(1,0,0)_7 + Fourier K=2 + DOW | 0.846 | yes |
| Deterministic seasonal OLS (Fourier K=2 + DOW) | 0.856 | no ARMA errors |
| Monthly climatology | 0.871 | benchmark |
| SARIMA(1,0,3)(2,0,0)_7 | 0.938 | yes |
| ARIMA(3,0,0) | 0.940 | no, Ljung-Box fails |
| Mean | 0.961 | benchmark |
| Seasonal naive 365 | 1.169 | benchmark |
| ETS Holt-Winters annual | 1.379 | no |

The plain ARIMA grid has no adequate model: every fit fails Ljung-Box at K = 14, 21 or 30 because the
annual cycle remains in the residuals. The SARIMA grid fixes that with a seasonal AR at lag 7 and
passes. The best model adds annual Fourier terms and a weekly seasonal AR to the ARIMA errors; its
validation RMSE is 10% lower than SARIMA and 12% lower than the mean.

At forecast horizons the result is more sober. A naive last-value forecast wins the first week
(RMSE 0.585 against 0.953 for SARIMA and 0.968 for ARIMA); the models win from 8 to 90 days; beyond
90 days monthly climatology is as good as anything. On the 2025 test the monthly climatology is
first (0.954), SARIMAX second (0.956), and SARIMA/ARIMA tie the mean (1.048). The seasonal structure
is the forecastable part; the year-to-year level is not.

## 2. Data and splits

| Window | Dates | Days | Observed | Role |
|---|---|---|---|---|
| train | 2021-01-01 to 2023-12-31 | 1,095 | 1,071 | identification and grids |
| validation | 2024-01-01 to 2024-11-30 | 335 | 335 | selection only |
| test | 2025-01-01 to 2025-12-31 | 365 | 365 | scored after selection |
| final fit | 2021-01-01 to 2024-11-30 | 1,430 | 1,406 | refit before the test |

The 24 missing training days are kept as missing; SARIMAX handles them through the Kalman filter.
Exponential smoothing needs a complete series and uses linear interpolation; it is a benchmark only.
No value is invented for the 38 absent dates or the 17 blank 2023 dates. F01 shows the splits and
the gaps; F02 and F03 show the annual and weekly profiles.

## 3. Trend tests

The user's prior was a likely trend. Six specifications are reported in `t09_trend_regressions.csv`
and F10:

| Specification | Slope/year | iid p | HAC p |
|---|---|---|---|
| daily, 2021-2023, trend only | +0.011 | 0.77 | 0.93 |
| daily, 2021-2023, + 11 month dummies | +0.022 | 0.52 | 0.77 |
| daily, 2021-2023, + Fourier K=2 + DOW | +0.018 | 0.58 | 0.80 |
| monthly, 2021-2024 | -0.036 | 0.67 | 0.66 |
| annual, 4 points (2024 Jan-Nov) | -0.033 | 0.50 | not reliable at n=4 |
| daily, 2021-2024 | -0.038 | 0.105 | 0.62 |

The 2021-2023 window is flat. The negative slope appears only when the low 2024 is appended, and it
is not significant under HAC or seasonal controls. The block bootstrap (F12) gives -0.225 to +0.257.
The rolling 365-day mean ranges only 0.33 AQHI (F13). The annual regression is shown for description
only: four points cannot identify a trend, and its HAC interval is not used as evidence.

F09 makes the endpoint effect visible: the left panel (2021-2023) is flat, the right panel
(2021-2024) slopes down because 2024 is included. A trend claim that depends on the endpoint is not
a trend.

## 4. Stationarity and identification

- ADF on the level with a constant rejects at every lag 1..14 (F04); the no-constant row fails
  because the mean is not zero, so that row is misspecified, not evidence against stationarity.
- KPSS p = 0.10 and Phillips-Perron agree; both are supplementary cross-checks.
- ADF on the first difference also rejects, but that is not evidence of a unit root: a series
  stationary around a seasonal mean has a stationary difference too.
- ACF (F05): r1 = 0.831, r7 = 0.328, r365 = 0.142. The slow decay is persistence plus the annual
  cycle, not a stochastic trend.
- Removing the calendar-month mean (F06) lowers r1 only to 0.772, because the annual cycle is smooth
  within months; a month dummy is too coarse. Fourier terms are the right representation.
- STL (F07) gives a seasonal strength of 0.53 and a trend strength of 0.01. The annual component
  spans about 1.5 AQHI.
- F08 compares the ACF under the level, month adjustment and first difference. The level's slow
  decay is the annual cycle; differencing removes it but also removes the forecastable seasonal mean.

Identification therefore uses d = 0, D = 0 for the weekly seasonal ARIMA, with the annual cycle
carried by deterministic Fourier terms in the SARIMAX extension.

## 5. ARIMA

Grid: p, q = 0..3, d = 0 and d = 1, constant included; 32 fits. Detail in `t13_arima_grid.csv`.

- The d = 0 AIC surface (F14) falls sharply from (0,0) to (3,0); the overall minimum is at (3,3),
  but that fit has a near-unit parameter, so it is unidentified and excluded.
- The lowest-AIC non-boundary fit is ARIMA(3,0,0), AIC 1802.3, BIC 1827.3.
- No ARIMA fit passes Ljung-Box at K = 14, 21 and 30. AR(3) has p = 0.007, 0.043, 0.034: it leaves
  annual structure in the residuals. It is carried as the ARIMA representative, not as an adequate
  model.
- d = 1 is worse: the best d = 1 validation RMSE is 1.147 against 0.940 for d = 0. Differencing
  removes the level that the model needs.
- Validation RMSE 0.940, MAE 0.790, 95% coverage 0.982 (the intervals are wide).
- Overparameterisation (F16, `t15`): adding AR(4) raises AIC; adding MA(1) lowers AIC only by moving
  to a near-unit fit. There is no parsimonious ARIMA extension that fixes the annual residual.

## 6. SARIMA

Grid: p, q = 0..2, P, Q = 0..1, D = 0 and 1, s = 7, plus a refinement around the leader; 78 fits.
Detail in `t16_sarima_grid.csv`.

The selected model is SARIMA(1,0,3)(2,0,0)_7:

- AIC 1791.0, BIC 1831.0.
- Ljung-Box p = 0.171 at K = 14, 0.350 at K = 21, 0.265 at K = 30. This is the first adequate model.
- Validation RMSE 0.938, MAE 0.788, 95% coverage 0.982.
- D = 1 is rejected decisively: AIC 2068.6 and validation RMSE 1.038. Seasonal differencing at lag 7
  over-differences a series whose weekly effect is deterministic and small.
- Overparameterisation (`t18`): adding a third seasonal AR raises AIC to 1792.9; adding a seasonal MA
  raises it to 1794.7 and puts a parameter near the boundary. The selected order is the parsimonious
  adequate one.
- The weekly seasonal AR is not a large calendar effect: the fitted day-of-week range is only 0.07
  AQHI. It is capturing the leftover negative lag-7 autocorrelation, which is why the model passes
  Ljung-Box where the plain ARIMA fails.

F17 shows the AIC landscape and the near-unit flags; F18 shows that AIC and validation RMSE disagree
across D; F19 shows the residual ACF and histogram.

## 7. SARIMAX with annual seasonality

This is the extension that matches the data. Regressors: Fourier terms for the annual cycle
(K = 1..4) or 11 month dummies, six day-of-week dummies, and ARMA errors; the best specification also
carries a weekly seasonal AR at lag 7. Detail in `t19_sarimax_grid.csv`.

Selected: SARIMAX(1,0,2)(1,0,0)_7 with Fourier K=2 and DOW.

- AIC 1777.6, BIC 1857.6.
- Ljung-Box p = 0.356, 0.489, 0.198 at K = 14, 21, 30.
- Validation RMSE 0.845, MAE 0.672, 95% coverage 0.958. This is the best adequate model.
- The fitted annual amplitude is 1.52 AQHI; the fitted day-of-week range is 0.069.
- A linear trend added to this model is +0.021/year with p = 0.80. The trend is absent even after
  the seasonal structure is controlled.
- Eight of 32 fits are adequate; all of them include the weekly seasonal AR. Models with Fourier and
  DOW but no weekly SARMA fail Ljung-Box, which is the evidence that the lag-7 term is needed.
- AIC is not comparable across Fourier K or month dummies; the cross-family choice uses validation
  RMSE. Within 0.01 of the validation leader, the lower AIC is chosen.

F20 and F21 show the AIC and validation landscape; F22 the residual diagnostics; F23 the fitted
annual and weekly components.

## 8. Exponential smoothing

Six configurations: SES, damped Holt, weekly Holt-Winters additive and multiplicative, and an annual
(s = 365) additive variant. Detail in `t23_ets_grid.csv`.

The best is the annual Holt-Winters with validation RMSE 1.379, but it is still worse than the mean
(0.961) and far worse than SARIMAX (0.845). The weekly-only ETS variants are worse still (1.70-1.90)
because the last observed level is extrapolated flat: the 2023 year-end is high and 2024 is low. ETS
is not a serious competitor once the annual cycle matters. F24 and F25 show the forecast and the
grid.

## 9. GARCH

The selected SARIMAX residuals have strong ARCH effects: ARCH-LM p = 0.00025 at lag 10, and the
squared-residual ACF is far outside the band (F26). Eight volatility models are fitted:
Constant, ARCH(1), GARCH(1,1), GARCH(1,2), GARCH(2,1), GARCH(1,1)-t, GJR-GARCH(1,1), EGARCH(1,1).
Detail in `t26_garch_grid.csv`.

Selected: GARCH(1,1)-t on zero-mean residuals.

- AIC 1587.4, BIC 1607.3.
- Persistence alpha + beta = 0.943, half-life 11.9 days; unconditional volatility 0.732 AQHI.
- Standardised squared residuals pass Ljung-Box at K = 14, 21 and 30 (p = 0.548, 0.943 for K = 14
  and 30); the GARCH term absorbs the clustering.
- F27 shows the conditional volatility against the AQHI episodes; F28 shows the volatility term
  structure converging to the unconditional level over about a month.

A 2025 one-step upper-tail VaR filter with fixed parameters is in F29 and `t28_var_backtest.csv`.
At 95%, GARCH-t exceeds 34 of 365 days (9.3%) against 18.25 expected; constant variance exceeds 66
(18.1%). GARCH halves the misses but does not reach nominal coverage, because the 2025 mean forecast
is biased and the residuals are fat-tailed. The volatility model is useful; the mean model is the
binding constraint.

## 10. Forecast comparison

Validation (2024 Jan-Nov, 335 days) and test (2025, 365 days) are in `t29` and `t30`; F30-F32 show
the rankings and forecasts.

Validation ranking (RMSE): SARIMAX 0.846, seasonal OLS 0.856, monthly climatology 0.871,
SARIMA 0.938, ARIMA 0.940, mean 0.961, seasonal naive 365 1.169, ETS 1.379, seasonal naive 7 1.502,
naive 1.732.

Test ranking (RMSE): monthly climatology 0.954, SARIMAX 0.956, SARIMA 1.048, ARIMA 1.049,
mean 1.049, seasonal naive 7 1.110, seasonal naive 365 1.246, ETS 1.314.

Two facts matter more than the ranking:

1. **The test is a long-horizon exercise.** December 2024 is missing, so the first 2025 forecast is
   32 steps ahead. At that range the annual climatology is the signal, and the sophisticated models
   do not beat it. This does not mean the models are useless; it means the test cannot show their
   short-horizon value.
2. **Persistence wins the first week.** From the 2023-12-31 origin, the 1-7 day RMSE is 0.585 for
   the naive forecast, 0.673 for ETS, 0.953 for SARIMA and 0.968 for ARIMA (F33). The models
   mean-revert too quickly at one-day horizons. From 8 to 90 days SARIMA is best (0.775-0.776);
   beyond 90 days monthly climatology or seasonal OLS is as good.

The Diebold-Mariano tests (`t34`) say every model beats seasonal naive 365 on the 2025 test
(p < 0.05), but the margin is small and the climatology benchmark wins outright. The monthly test
errors (F34, `t32`) are largest in April (RMSE 1.60), September (bias -1.08) and October
(bias -0.82), and smallest in August (0.34). The method-change split (F35, `t33`) is mixed:
SARIMAX is better after 2025-03-22 (0.941) than before (1.005); SARIMA and ARIMA are also better
after (1.032 vs 1.102), but the monthly climatology is better before (0.928 vs 0.962). No model is
uniformly better across the two regimes.

## 11. What this means for the sprint

- Do not put a trend line on the deck. The 2021-2024 slope is an endpoint artifact; the robust
  tests are flat. The interesting pattern is the annual cycle, not a trend.
- For a one-day decision, the best forecast is close to today's value: persistence has the lowest
  1-7 day error. A model that reverts to the mean is worse at that horizon.
- For a 1-3 month plan, the seasonal model (SARIMAX with Fourier and DOW) is the best of the tested
  models, but the gain over monthly climatology is about 0.03-0.09 RMSE. The annual cycle does most
  of the work.
- For a year ahead, monthly climatology is as good as the model; do not claim model skill at that
  horizon.
- Volatility clusters around episodes and is persistent (half-life 11.9 days). GARCH improves tail
  risk over constant variance but does not reach nominal coverage because the mean forecast is the
  weak part.
- For the Smoking Baby question, the time-series result reinforces the EDA: the annual and daily
  cycles are the reliable structure; the year-to-year level is not predictable from this series.

## 12. Limitations

- The export starts in 2021 and has 38 absent dates plus 17 blank 2023 dates. The 2024-12 gap makes
  the test a 32-396 step forecast, not a one-step test.
- The AQHI compilation changed on 2025-03-22. The 2025 test mixes two instruments; the split table
  reports both sides, but no pre/post causal claim is made.
- AIC is compared only within one dependent variable, one differencing order and one regressor set.
  Cross-family choices use validation RMSE, and the test is never used for selection.
- The 2023 spring episode is an outlier; the models fit it with ARMA persistence, not with an
  explicit intervention. A pulse specification was tested but did not remove the remaining lag-7
  autocorrelation; the weekly SARMA does.
- The GARCH VaR filter uses fixed parameters and one-step updates; it is a diagnostic, not a
  production risk system.
- The series is the territory mean. Station-level series can behave differently (the EDA showed
  Tuen Mun and Tap Mun have heavier afternoon tails).

## 13. Figure index

| Figure | What it shows |
|---|---|
| F01 | Daily series, frozen splits and data gaps |
| F02 | Monthly and annual means |
| F03 | Month and day-of-week profiles |
| F04 | ADF p-value grid on the level |
| F05 | ACF and PACF of the level |
| F06 | ACF and PACF after the month mean is removed |
| F07 | STL decomposition with a 365-day period |
| F08 | ACF under level, month adjustment and first difference |
| F09 | Trend fits 2021-2023 vs 2021-2024 |
| F10 | Trend slope under six specifications with iid and HAC intervals |
| F11 | Monthly means with OLS and Sen trends |
| F12 | Residual block bootstrap of the trend |
| F13 | Rolling 365-day mean |
| F14 | ARIMA(p,0,q) AIC heatmap |
| F15 | ARIMA(p,0,q) validation RMSE heatmap |
| F16 | Selected ARIMA residual ACF and histogram |
| F17 | Delta-AIC of the 20 lowest SARIMA fits |
| F18 | SARIMA AIC vs validation RMSE, by D |
| F19 | Selected SARIMA residual diagnostics |
| F20 | SARIMAX AIC by regressor set and ARMA order |
| F21 | SARIMAX validation RMSE ranking |
| F22 | Selected SARIMAX residual ACF and Q-Q |
| F23 | Fitted annual cycle and day-of-week effects |
| F24 | Best ETS validation forecast |
| F25 | ETS validation RMSE grid |
| F26 | Residual and squared-residual ACF |
| F27 | GARCH conditional volatility against AQHI |
| F28 | Volatility term structure |
| F29 | 2025 one-step upper VaR backtest |
| F30 | Validation RMSE ranking |
| F31 | Validation forecasts of the top models |
| F32 | 2025 test forecasts |
| F33 | RMSE by forecast horizon |
| F34 | Monthly test means and errors |
| F35 | Test error before and after the 2025-03-22 change |

## 14. Method-to-source table

| Method | Source | Where used |
|---|---|---|
| ADF, unit-root verdict | STAT4601 Ch. 4 | Section 4, F04 |
| ACF/PACF identification | STAT4601 Ch. 4 | Section 4, F05-F06 |
| ARIMA estimation and diagnostics | STAT4601 Ch. 5-6 | Section 5 |
| Ljung-Box at several K, overparameterisation | STAT4601 Ch. 6 | Sections 5-7 |
| AIC/BIC selection | STAT4601 Ch. 4, 6 | Sections 5-7 |
| Seasonal ARIMA | STAT4601 Ch. 8 | Section 6 |
| Forecasting, prediction limits | STAT4601 Ch. 7 | Section 10 |
| Exponential smoothing | STAT4601 Ch. 9 | Section 8 |
| GARCH/ARCH, squared-residual diagnostics | STAT4601 Ch. 10 | Section 9 |
| HAC trend test, block bootstrap, Mann-Kendall, Diebold-Mariano | supplementary cross-checks | Sections 3 and 10 |

## 15. Reproduction

```bash
cd comp3522-sprint-challenge
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/00_build_series.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/01_eda_stationarity.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/02_trend_tests.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/03_arima_grid.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/04_sarima_grid.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/05_sarimax_seasonal.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/06_exponential_smoothing.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/07_garch.py
~/.hermes/venvs/ts4601/bin/python TimeSeriesAnalysis/code/08_forecast_eval.py
```

The SARIMA grid is the slow step (about 4 minutes); the rest are under a minute each. Every number
in this report is printed by one of those scripts into `results/tables/` or `results/facts/`.
