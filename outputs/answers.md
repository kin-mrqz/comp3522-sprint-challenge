# Answers

## Question 1 - Key trends in hourly AQHI (2021-2025)

- **Claim:** Hong Kong's air quality is strongly seasonal and peaks in the dry
  winter. Territory mean AQHI is highest in 1 and lowest in
  6; the daily cycle peaks around 16:00 and bottoms
  near 6:00 (range 1.192 AQHI points).
- **Number:** Across 762,101 station-hours, 1.6% of hours
  are "poor" (AQHI >= 7, High or worse). The cleanest year was 2024
  (mean 3.214) and the worst 2023
  (mean 3.412). New Territories stations
  (e.g. Tuen Mun, Yuen Long, Tung Chung) read higher than Central/Western.
- **Chart:** `outputs/figures/q1_seasonal_month.png` (seasonal cycle) and
  `q1_daily_timeseries.png` (daily series with 30-day rolling mean).
- **Limitation:** Station hourly values are not population-weighted; the
  territory mean treats every station equally. Missing hours are dropped, not
  imputed, and `10+` is capped at 10 so extreme episodes are understated.

## Question 2 - Weather relationships and safe-day advice

### (a) Weather-AQHI relationships

- **Claim:** On daily data (1041 days, 2023-01-01 to
  2025-12-31), the strongest correlate of AQHI is
  **Precipitation (mm)** (Spearman rho = -0.585). Wind speed and relative humidity
  also matter; temperature has a weaker, season-confounded link.
- **Number:** Precipitation (mm) has the largest |Spearman rho| = 0.585 among
  the six project weather variables (see
  `outputs/tables/q2_weather_correlations.csv`). Correlations are computed on
  daily territory means over the weather-overlap window 2023-2025.
- **Chart:** `outputs/figures/q2_correlation_bars.png` and
  `q2_scatter_top2.png`.
- **Limitation:** Weather is a territory-wide Open-Meteo mean (2023-2025 only)
  and AQHI is station-based; both are ecological aggregates, so these are
  associations, not causal effects. Season confounds raw correlations.

### (b) Conditions that maximise a safe day for Smoking Baby

- **Claim:** The safest days are warm, humid and wet (summer, low pressure);
  the riskiest are cool, dry, high-pressure days (winter monsoon). Standardised
  logistic effects on P(poor day, AQHI>=7): precipitation -1.32, humidity
  -0.21, temperature -1.00, pressure -1.39, wind -0.18
  (negative = lowers risk). Rain washes out particulates and the summer SW
  monsoon brings cleaner air; the dry NE monsoon traps pollutants.
- **Number:** The model separates risk well (validation AUC =
  0.621). Safest-quintile days average a
  0.0 poor-day rate vs
  0.006 for the riskiest quintile.
- **Simple risk rule:** Treat a day as *caution* when AQHI forecast/nowcast is
  already >= 4 (Moderate) AND wind is low AND pressure is high; treat AQHI >= 7
  as *avoid outdoor exertion* (EPD "High"+ advice for at-risk people). A wet,
  humid, warm day is the practical safe-day signal for Smoking Baby.
- **Chart:** `outputs/figures/q2_risk_coefficients.png` and
  `q2_risk_calibration.png`.
- **Limitation:** Trained on 2023-2024 and validated on 2025 daily data with
  only 6570 validation days; logistic model is deliberately
  simple. Smoking Baby's personal sensitivity is not in the data.

## Advanced time-series analysis

- **STL decomposition (monthly, period=12):** the series is mostly seasonal
  (amplitude ~2.23 AQHI, winter high / summer low) with a small, near-flat
  trend and a low-noise residual (std 0.40). Visual: `outputs/figures/ats_stl_monthly.png`.
- **Stationarity (ADF + KPSS):** both the daily and monthly series are stationary
  (ADF p < 0.01; KPSS fails to reject), so no differencing is needed for the
  seasonality story.
- **Trend (Mann-Kendall + Sen's slope):** no statistically significant monotonic
  trend (monthly tau = -0.05, p = 0.57; daily p = 0.21). Multi-year AQHI is
  stable; the variation is seasonal, not a rising/falling trend.
- **Change-points (PELT on the STL trend):** two structural breaks in the
  deseasonalised level around **2022-08** and **2023-11** (e.g. post-COVID
  activity rebound), not a steady drift. Figure: `ats_changepoint.png`.
- **Spectral analysis:** the periodogram peaks near a **~325-day** period,
  confirming the dominant annual cycle. Figure: `ats_periodogram.png`.
- **Forecasting backtest (train ≤2023, test 2024-2025):** a seasonal
  Holt-Winter ETS model beats the seasonal-naive baseline
  (MAE 0.37 vs 0.51, RMSE 0.46 vs 0.58), showing the annual cycle is
  predictable but the level is noisy. Figure: `ats_forecast.png`.

## Graph gate

- **Decision:** No native graph. The AQHI table is a spatiotemporal panel keyed
  by (timestamp, station); stations have fixed identities but no edge list or
  shared real-world graph id in the provided files, and a kNN graph on features
  is disallowed. Time-series and regression are the right tools here.

## Project link

- The course project's rainfall graph (CSI ~0.37 at 1-hour lead) is a different
  task (rain warning, not AQHI), so project numbers are used only for the weather
  covariate, not as AQHI benchmarks. The project's Open-Meteo weather feed is the
  same weather source joined in Question 2.
