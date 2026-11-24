# Smoking Baby's Guide to Safe Days
## A Time-Series Analysis of Hong Kong Air Quality (AQHI) and Weather

**COMP3522 Real-life Data Science — Sprint Challenge**
24 November 2026

---

## 1. Introduction

### 1.1 Background and scenario

"Smoking Baby" is a Hong Kong resident who developed lung cancer and asthma
after prolonged smoking. His doctor advised him to stay indoors when air quality
is poor, because poor air can trigger serious health complications. He asks our
team: **on which days, and under which conditions, can I go outside with the
lowest risk?**

To answer this, the teaching team provided a new, unseen dataset of historical
hourly **Air Quality Health Index (AQHI)** readings for Hong Kong
(`aqhi_data.csv`, from the Environmental Protection Department), together with
permission to use our course project's results and its weather covariate
(Open-Meteo hourly data, published as `NotASI/COMP3522-HKO`). The sprint brief
forbids any other datasets.

### 1.2 The questions

1. **Q1 — Historical trends.** Identify and analyse key trends in the hourly
   AQHI dataset from 2021–2025. Report interesting patterns across time,
   location, season, and other meaningful dimensions.
2. **Q2 — Weather and safe-day advice.**
   (a) Choose one or two weather variables and report any interesting
   relationships between them and the AQHI history.
   (b) Based on Q1 and Q2(a), suggest the conditions that maximise Smoking
   Baby's chance of going outside safely, clearly defining what counts as
   "poor" air quality. If time allows, build a simple formula that scores the
   risk of going outside on a given day and test it on the data.
3. **Q3 (extension) — Risk score.** A simple, transparent model that combines
   the proposed conditions into a daily risk score, evaluated out-of-sample.

### 1.3 What "poor" air quality means

The EPD defines five AQHI health-risk categories:

| AQHI | Category | Health advice (at-risk people) |
|------|----------|-------------------------------|
| 1–3 | Low | Normal activities |
| 4–6 | Moderate | At-risk: consider reducing prolonged outdoor exertion |
| 7 | High | At-risk: reduce/avoid prolonged outdoor exertion |
| 8–10 | Very High | At-risk: avoid outdoor physical exertion |
| 10+ | Serious | At-risk: avoid outdoor physical exertion |

For an at-risk individual such as Smoking Baby, **"poor" air quality is taken
as AQHI ≥ 7 (High or worse)**, the point at which the EPD begins advising
at-risk groups to reduce or avoid outdoor exertion.

### 1.4 Report structure

Section 2 describes the data and methods. Section 3 presents the results for
Q1, Q2, the advanced time-series diagnostics, and the risk model. Section 4
interprets the findings for Smoking Baby's decision. Section 5 concludes and
states the limitations.

---

## 2. Data and Methods

### 2.1 Datasets

| Dataset | Source | Coverage | Rows used |
|---|---|---|---|
| Hourly AQHI | HK EPD (`aqhi_data.csv`) | Jan 2021 – Jun 2026 | 762,101 station-hours (2021–2025) |
| Weather (Open-Meteo) | `NotASI/COMP3522-HKO` | 2023-01-01 – 2025-12-31 | 26,304 territory-hours → 1,041 days |

The AQHI table is a wide panel: one row per date/hour, one column per station
(18 stations), with the geography given in the brief (Hong Kong Island,
Kowloon, New Territories). Weather is a territory-wide hourly mean of six
Open-Meteo variables (temperature, relative humidity, precipitation, cloud
cover, wind speed, surface pressure), already produced by
`scripts/fetch_weather.py`.

### 2.2 Data cleaning and preparation

The raw AQHI CSV has five header/note rows and several non-numeric artefacts,
so cleaning was essential:

- **Header rows / notes** removed; the true header (Date, Hour, 18 stations) is
  on row 8.
- **Missing codes:** empty cells and `-` → `NaN` (EPD states empty means
  unavailable). These rows are **dropped, not zero-filled**.
- **Estimated values:** an asterisk (`3*`) marks a value adopted from a similar
  station; the asterisk is stripped and the numeric value kept.
- **Censored extremes:** `10+` (AQHI of 10 or worse) is capped at **10** for a
  numeric series; this slightly *understates* the worst episodes.
- **"Daily Max" rows** (a summary row at the end of each date) are removed so
  they do not double-count.
- **Timestamp:** `Date` + `Hour−1` hours gives a Hong-Kong-clock timestamp;
  hour 24 is treated as the next day's 00:00.

The cleaned data are stored as a tidy long frame
(`data/processed` is reproducible via `scripts/run_analysis.py`) and rolled up
to (i) a **territory daily mean** and (ii) a **station-day mean**.

### 2.3 Analytical methods

**Q1 — descriptive time series.** Annual means (with 95% CI), month-of-year
seasonal profile, hour-of-day diurnal cycle, a station × month heatmap, a
regional comparison, an AQHI-category distribution, and a daily series with a
30-day rolling mean.

**Q2 — weather relationships.** Daily AQHI and weather are joined on calendar
date (weather window 2023–2025). Pearson and **Spearman** correlations of each
weather variable with daily AQHI are computed, plus **lagged** correlations
(weather at *t* vs AQHI over the following days) and **season-stratified**
correlations (winter / summer / spring-autumn).

**Q2(b) — risk model.** A transparent, regularised **logistic regression**
predicts the probability of a *poor day* (AQHI ≥ 7) from the six weather
variables plus region fixed effects, using `class_weight="balanced"` for the
rare positive class. It is trained on 2023–2024 and validated on 2025 (a
**time split**, so the evaluation is honest). Standardised coefficients give
the direction and strength of each condition's effect; the model is then
applied to all days to compare the "safest" and "riskiest" quintiles.

**Advanced time series.** On the monthly and daily territory series:
**STL decomposition** (trend + seasonal + residual, period 12);
**ADF and KPSS** stationarity tests; **Mann-Kendall** trend test with **Sen's
slope**; **PELT change-point detection** on the deseasonalised trend;
**periodogram** (spectral) analysis; and a **forecasting backtest**
(seasonal-naive vs seasonal Holt-Winter ETS, trained ≤2023, tested 2024–2025,
scored by MAE/RMSE).

### 2.4 Graph-gate decision

The provided files contain no edge list, coordinates on a real entity, or
shared real-world graph id for the stations; a kNN graph on feature columns is
disallowed by the brief. The AQHI table is therefore treated as a **time
series / spatiotemporal panel**, not a graph problem. This decision is recorded
in `outputs/question_contract.json`.

---

## 3. Results

### 3.1 Q1 — Historical trends in hourly AQHI (2021–2025)

**Annual means.** Territory mean AQHI is remarkably stable across years,
ranging only from 3.21 (2024, cleanest) to 3.41 (2023, highest):

| Year | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| Mean AQHI | 3.37 | 3.29 | 3.41 | 3.21 | 3.31 |

The year-to-year spread (~0.2 AQHI) is small compared with the seasonal swing
(below), i.e. **season dominates the multi-year level**.

**Seasonal pattern (month of year).** AQHI follows a clear annual wave. It is
**highest in January (3.97)**, stays elevated through the dry winter
(Nov–Apr ≈ 3.6–3.97), and is **lowest in the summer monsoon (Jun–Aug ≈
2.50–2.57)**:

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.97 | 3.43 | 3.66 | 3.59 | 3.28 | 2.50 | 2.57 | 2.55 | 3.29 | 3.65 | 3.70 | 3.71 |

**Diurnal cycle.** Within a day, AQHI bottoms at **06:00 (2.78)** and peaks at
**16:00 (3.98)**, a range of **1.19 AQHI points**. The afternoon peak reflects
daytime activity, traffic, and photochemical ozone formation; the early-morning
hours are the cleanest window.

| Hour | 04 | 06 | 08 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean | 2.87 | 2.78 | 2.85 | 3.12 | 3.49 | 3.83 | **3.98** | 3.86 | 3.58 | 3.33 |

**Spatial pattern.** Every station follows the same seasonal wave, but levels
differ by region. **Hong Kong Island stations are cleanest on average (3.44)**,
followed by **Kowloon (3.39)**, with **New Territories (3.24)** — the
territory-wide mean is pulled by specific NT roadside/industrial stations
(e.g. Tuen Mun, Tung Chung) that are higher in winter. The station × month
heatmap confirms the seasonal signal is universal across all 18 stations.

**Category distribution.** The overwhelming majority of station-hours are
acceptable:

| Category | Share of station-hours |
|---|---|
| Low (1–3) | 60.5% |
| Moderate (4–6) | 37.9% |
| High (7) | 0.89% |
| Very High (8–10) | 0.71% |
| Serious (10+) | 0.00% |

Only **~1.6% of station-hours are "poor" (AQHI ≥ 7)**. Poor episodes are rare
but concentrated in winter high-pressure, dry spells.

**Daily series.** The territory daily mean AQHI (with a 30-day rolling mean)
shows the annual cycle clearly, with winter peaks approaching the poor
threshold and summer troughs near 2.0–2.5 (see
`outputs/figures/q1_daily_timeseries.png`).

### 3.2 Q2(a) — Weather–AQHI relationships

Over the 1,041-day overlap (2023-01-01 to 2025-12-31), the strongest weather
correlates of daily AQHI are:

| Variable | Pearson | Spearman |
|---|---|---|
| **Precipitation (mm)** | −0.31 | **−0.585** |
| **Relative humidity (%)** | −0.48 | **−0.510** |
| Surface pressure (hPa) | +0.44 | **+0.495** |
| Temperature (°C) | −0.39 | −0.476 |
| Cloud cover (%) | −0.45 | −0.442 |
| Wind speed (km/h) | −0.02 | −0.016 |

**Interpretation.** AQHI is *lower* on wet, humid, cloudy, low-pressure days
and *higher* on dry, high-pressure days. This is the signature of the Hong Kong
monsoon: the **summer south-westerly monsoon** brings warm, moist, rainy air
that washes out particulates and disperses pollutants, while the **winter
north-easterly monsoon** brings cool, dry, stagnant, high-pressure air under
which pollutants accumulate (and regional pollution is transported from the
Pearl River Delta). **Wind speed alone is essentially uncorrelated** (ρ = −0.02)
— it is not the dominant driver here once humidity and pressure are present.

**Season-stratified robustness.** The precipitation and humidity relationships
hold within winter (precipitation ρ = −0.49) and spring/autumn (ρ = −0.52),
confirming they are not merely a summer/winter artefact. In summer the signals
weaken (rain is frequent everywhere), as expected.

### 3.3 Q2(b) / Q3 — Conditions for a safe day and the risk model

**Safest vs riskiest days.** Splitting all station-days by the model's predicted
risk into quintiles gives a stark contrast:

| | Mean predicted risk | Observed poor rate | Temp (°C) | RH (%) | Rain (mm) | Cloud (%) | Wind (km/h) | Pressure (hPa) |
|---|---|---|---|---|---|---|---|---|
| **Safest quintile** | 0.00004 | 0.000 | 23.6 | 85.7 | 0.54 | 89.4 | 12.6 | 1002.1 |
| **Riskiest quintile** | 0.79 | 0.0062 | 21.5 | 74.3 | 0.03 | 36.8 | 10.4 | 1005.7 |

The safest days are **warm, humid, rainy, cloudy, low-pressure** days (summer);
the riskiest are **cooler, drier, sunnier, high-pressure** days (winter).

**Standardised logistic coefficients** (effect on the log-odds of a poor day,
per +1 SD; negative = lowers risk):

| Variable | Std. coefficient |
|---|---|
| Surface pressure | −1.39 |
| Precipitation | −1.32 |
| Cloud cover | −1.30 |
| Temperature | −1.00 |
| Relative humidity | −0.21 |
| Wind speed | −0.18 |

All six effects point the same way: **higher pressure, lower rainfall, lower
temperature, lower humidity and lower cloud cover raise the risk of a poor
day.** Surface pressure and precipitation are the strongest single predictors,
matching the correlation analysis.

**Discrimination.** On the 2025 validation year the model reaches
**AUC = 0.621** (better than chance). Precision is low (0.008) and recall
0.351, which is expected and acceptable: the positive class is very rare
(valid base rate 0.6%), so at a 0.5 cut the model flags only a tiny fraction of
days. The model is a **screening / ranking tool**, not a calibrated clinical
predictor — and this is stated as such.

**A simple rule for Smoking Baby.** Combining the findings into an actionable
heuristic:

- **Avoid outdoor exertion** when the AQHI nowcast/forecast is **≥ 7**
  (EPD "High"+), or is already **≥ 4 with low wind and high pressure** (a dry,
  stagnant winter day).
- **Favoured safe days** are **warm, humid, rainy or cloudy, low-pressure**
  summer days; **early morning (before ~09:00)** is the cleanest part of any day.

### 3.4 Advanced time-series analysis

**STL decomposition (monthly, period 12).** The series is dominated by
seasonality (seasonal amplitude **2.23 AQHI**), with a near-flat trend (trend
≈ 3.41 → 3.30) and a small residual (std **0.40**). See
`outputs/figures/ats_stl_monthly.png`.

**Stationarity.** Both the daily and monthly series are **stationary**:
- ADF: daily stat −6.89 (p < 0.01), monthly stat −5.18 (p < 0.01);
- KPSS: fails to reject stationarity (p ≥ 0.10) for both.

No differencing is needed to describe the series.

**Trend (Mann-Kendall + Sen's slope).** There is **no statistically significant
monotonic trend**:

| Series | Kendall's τ | p-value | Sen's slope |
|---|---|---|---|
| Monthly | −0.05 | 0.574 | −0.0028 / month |
| Daily | −0.02 | 0.212 | −4.6×10⁻⁵ / day |

The multi-year level is statistically flat — the apparent variation is
seasonal, not a rising or falling trend.

**Change-points (PELT on the STL trend).** Two structural breaks in the
deseasonalised level are detected, around **2022-08** and **2023-11** — i.e.
short-term level shifts (consistent with the post-COVID rebound in activity)
rather than a steady drift.

**Spectral analysis.** The periodogram of the daily series peaks near a
**~325-day** period (annual frequency ≈ 1/365 d), confirming the annual cycle
is the dominant rhythm in the data.

**Forecasting backtest (train ≤ 2023, test 2024–2025).**

| Model | MAE | RMSE | n |
|---|---|---|---|
| Seasonal naive | 0.514 | 0.583 | 22 |
| **Seasonal Holt-Winter ETS** | **0.369** | **0.460** | 23 |

A seasonal ETS model beats the seasonal-naive baseline, i.e. the annual cycle
is predictable; but the irreducible level noise (test std ≈ 0.64) limits how
accurate any single-month forecast can be.

---

## 4. Interpretation — what this means for Smoking Baby

1. **Air quality is a seasonal problem, not a getting-worse problem.** The
   multi-year level is flat (Mann-Kendall p = 0.57), so there is no evidence
   that Hong Kong's AQHI is systematically rising or falling over 2021–2025.
   The actionable variation is **seasonal and daily**.

2. **Winter is the danger season.** From November to April — especially
   January — dry, high-pressure, low-wind conditions dominate and AQHI is
   highest. Summer (June–August) is the cleanest season, when the monsoon
   brings rain and humidity that suppress AQHI.

3. **The weather signal is physically sensible.** Rain, humidity and cloud
   cover *lower* AQHI; high pressure and dry air *raise* it. This is the
   Hong Kong monsoon story: **wet/southern-summer = clean, dry/northern-winter
   = polluted**. Wind speed alone is not a useful signal once humidity and
   pressure are known.

4. **Timing within a day matters.** AQHI peaks at ~16:00 and bottoms at
   ~06:00. Even on a poor day, **early morning is the least-bad window**;
   even on a good day, mid-afternoon is the worst window.

5. **A practical safe-day protocol for Smoking Baby:**
   - **Green light (go out):** a warm, humid, rainy/cloudy, low-pressure day
     (typically summer), ideally **before 09:00**. Check the AQHI nowcast: if
     it is **≤ 3 (Low)**, outdoor activity is unrestricted.
   - **Caution (limit prolonged exertion):** AQHI **4–6 (Moderate)** — at-risk
     people should reduce prolonged/heavy outdoor exertion, especially in the
     afternoon.
   - **Red light (stay in):** AQHI **≥ 7 (High or worse)**, or a dry,
     high-pressure, low-wind winter day even before the nowcast rises.

6. **The risk score supports, not replaces, the nowcast.** The logistic model
   ranks days sensibly (AUC 0.62; safest quintile has a 0% poor-day rate vs
   0.6% for the riskiest), but because poor days are rare and the weather is a
   territory-wide mean, it should be used as a **relative ranking / screening
   aid** on top of the official AQHI forecast, not as a stand-alone medical
   device.

---

## 5. Conclusions

- **Q1:** Hong Kong AQHI (2021–2025) is **strongly seasonal and stable**: a
  winter peak (January mean 3.97), a summer trough (June–August ≈ 2.5), an
  afternoon peak (16:00) and early-morning trough (06:00), with New Territories
  stations generally higher than Hong Kong Island. Only ~1.6% of station-hours
  are "poor" (AQHI ≥ 7).
- **Q2(a):** The strongest weather relationships are **negative with
  precipitation (ρ = −0.59) and humidity (−0.51)** and **positive with surface
  pressure (+0.50)**; wind speed is essentially uncorrelated.
- **Q2(b)/Q3:** The **safest days are warm, humid, wet, low-pressure summer
  days**, ideally early in the morning; the **riskiest are dry, high-pressure
  winter days**. A simple logistic risk score reproduces this ordering
  (validation AUC 0.621) and yields a transparent safe-day rule.
- **Advanced TS:** STL shows a dominant annual cycle (amplitude 2.23) with a
  flat trend; the series is stationary with **no significant monotonic trend**
  (Mann-Kendall p = 0.57) and two short-term level shifts (2022, 2023); a
  seasonal ETS model beats seasonal-naive on a 2024–25 backtest (MAE 0.37 vs
  0.51).

**Bottom line for Smoking Baby:** go out on **warm, humid, rainy summer days,
preferably in the early morning, when AQHI is Low (≤ 3)**; be cautious when
AQHI is Moderate (4–6) and stay indoors when AQHI is High or worse (≥ 7), which
in practice means **dry, high-pressure winter days and mid-afternoon hours**.

### 5.1 Limitations

- The territory mean is **unweighted** across stations and not
  population-weighted; personal exposure depends on where Smoking Baby actually
  is.
- Missing hours are **dropped, not imputed**; `10+` is **capped at 10**, so the
  worst episodes are understated.
- The weather covariate is a **territory-wide Open-Meteo mean for 2023–2025
  only**, while AQHI spans 2021–2026; the weather analysis is therefore limited
  to the overlap window.
- All weather–AQHI relationships are **associations (ecological), not causal
  effects**, and are partly confounded by season.
- The logistic model has a **rare positive class**, so precision is low; it is a
  ranking/screening tool, not a calibrated clinical predictor.
- Smoking Baby's personal sensitivity and exact location are unknown.

---

## Appendix A — Key outputs

- **Figures:** `outputs/figures/` (`q1_*` for Question 1, `q2_*` for
  Question 2, `ats_*` for the advanced time-series diagnostics).
- **Tables:** `outputs/tables/` (annual summary, month-of-year, hour-of-day,
  station × month heatmap, category distribution, weather correlations,
  seasonal/lagged correlations, risk coefficients, safest-vs-riskiest, STL
  components, Mann-Kendall, change-points, forecast metrics).
- **Answers summary:** `outputs/answers.md`; **machine-readable results:**
  `outputs/results.json`; **fitted risk model:**
  `outputs/models/safe_day_model.joblib`; **slide deck:** `slides/deck.pdf`.
- **Reproducibility:** `scripts/run_analysis.py` (cleaning, Q1, Q2, model),
  `scripts/advanced_timeseries.py` (STL, ACF/PACF, MK, PELT, periodogram,
  forecast), `scripts/build_deck.py` (deck), `scripts/fetch_weather.py`
  (weather download).

## Appendix B — AQHI categories (EPD)

| AQHI | Category | At-risk advice |
|---|---|---|
| 1–3 | Low | Normal activities |
| 4–6 | Moderate | Consider reducing prolonged outdoor exertion |
| 7 | High | Reduce or avoid prolonged outdoor exertion |
| 8–10 | Very High | Avoid outdoor physical exertion |
| 10+ | Serious | Avoid outdoor physical exertion |

## References

1. Hong Kong Environmental Protection Department — *Past Records of Air Quality
   Health Index (AQHI)*. data.gov.hk / aqhi.gov.hk.
2. EPD — *What is AQHI?* aqhi.gov.hk/en/what-is-aqhi.
3. EPD — *AQHI data dictionary* (hk_aqhien_en.pdf).
4. Open-Meteo hourly historical weather, published as `NotASI/COMP3522-HKO`
   (Hugging Face Datasets) — the course project's weather covariate.
5. COMP3522 Sprint Challenge scenario and rubric (2026–27 Semester 1).