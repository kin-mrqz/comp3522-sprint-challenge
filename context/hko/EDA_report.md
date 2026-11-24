# HKO-evaluator — Exploratory Data Analysis (Task 1 & Task 3)

*Month-1 EDA write-up (working document). All figures and numbers are regenerated
by the scripts in this folder.*

## 1. What we explored, and why it changed our plan

We started EDA expecting two clean tables: **station observations** (ground truth)
and **HKO forecasts** (the thing being judged). The very first surprise reshaped
everything: the station data we have collected contains **only temperature and
tide**, and the 9-day forecast feed contains **temperatures and wind but no rain
amount**. That means our stated Task 1 plan (verify rain forecasts)
cannot be executed yet, and Task 3's rain labels cannot come from stations. So
the EDA became partly an honest **data-readiness audit**, and partly the genuine
pattern-finding below on the data we do have (temperature everywhere; rainfall on
grids).

Data actually used:

| Source | Rows / span | What it is |
|---|---|---|
| HKO station history | 98,915 station-days, 46 stations, 2020–2025 | Daily min/mean/max temperature (ground truth) |
| HKO current forecasts (OCF) | 8 runs, 38 stations, ~67k hourly rows, 2026-10-04→05 | 9-day hourly temperature (the forecast) |
| HKO current observations (`rhrread`) | 8 snapshots × 32 stations, 2026-10-04→05 | "Live" temperature (sparse validation) |
| Open-Meteo ERA5 hourly grid | 5.79 M grid-hours, 2020–2025 | Rainfall + 40 weather variables |
| HKO gridded nowcast (daily) | 371 days, 2022-06-27→2023-07-02 | Half-hourly nowcast rainfall stats |

## 2. Task 1 — Reliability and validity of what we can actually measure

### 2.1 The ground truth has a strong, predictable shape

Temperature is stable, well covered (only ~3.3% of station-days missing) and
strongly seasonal: **27.8 °C in May–September vs 20.1 °C otherwise**, peaking in
**July (28.9 °C)** and bottoming in **January (16.3 °C)**.

![Seasonal temperature cycle](figures/fig_temp_seasonal.png)

Hot days are common but **not trending up in our window** — about **12% of
station-days** reach 33 °C every year (11.7% in 2020, 12.4% in 2025). A useful
target for a temperature forecast is therefore not "did it beat zero" but "does
it track a strongly varying, high baseline".

![Hot days by year](figures/fig_hot_days.png)

### 2.2 Temperature varies far more between places than we assumed

The mean daily maximum differs by **8.1 °C** across the network. The warmest
stations are inland/new-town sites (Yuen Long, Sheung Shui) and the coolest are
**high-elevation** sites (Tai Mo Shan, Ngong Ping) — see the cooler blue points.

![Station temperature map](figures/fig_station_map.png)

**Insight:** we cannot report one "Hong Kong forecast error". The contrast is
mostly **elevation/coast**, not a neat four-region split, so region-level results
must use an elevation-aware grouping (or simply the station's own baseline).

### 2.3 Reliability: with only hours between runs, the forecast barely moves

Across the eight captured runs (a few hours apart), the same valid hour's
temperature changed by only **0.07 °C on average** (max 6.2 °C). The one run we
have covers **nine days of hourly values**, and the diurnal shape survives even
far out.

![Forecast diurnal shape by lead day](figures/fig_forecast_diurnal.png)
![Forecast revisions](figures/fig_forecast_revisions.png)

**Insight (with a caveat):** the "jumpiness" that worries the public happens
between **daily** forecast issues, which we do **not** have yet. The historical
9-day archive is required to measure reliability by lead time; the current feed
only lets us show short-horizon stability.

### 2.4 Validity: a first real forecast-vs-observation comparison

Joining the latest forecast for each observed station-hour gives 104 pairs:
**bias +0.53 °C (forecasts ran slightly warm), MAE 1.04 °C.** It is a tiny,
same-day sample, but it proves the join works and gives a concrete baseline.

![Forecast vs observation](figures/fig_forecast_vs_obs.png)

## 3. Task 3 — Rainstorm rain is rare, seasonal, and time-of-day driven

Because stations carry no rain, we used two independent grids.

### 3.1 Class imbalance is severe (the single most important finding)

Over **5.79 M Open-Meteo grid-hours**, rain is a minority and heavy rain an
extreme one:

| Threshold | Grid-hours | Share | "1 in…" |
|---|---|---|---|
| Any rain ≥ 0.1 mm/h | 1,483,373 | 25.6% | 4 |
| ≥ 30 mm/h (Amber) | 339 | 0.0059% | 17,070 |
| ≥ 50 mm/h (Red) | 43 | 0.00074% | 134,579 |
| ≥ 70 mm/h (Black) | 3 | 0.00005% | ~1.9 M |

![Rainfall distribution](figures/fig_rain_om_hourly_dist.png)

**Consequence:** any model that says "no rainstorm" will score ~99.99% correct
and be useless. This directly validates the course rule to lead with **POD / FAR /
CSI versus persistence and climatology**, never overall accuracy.

### 3.2 A sharp wet season and a morning peak

Rain and rainstorm hours both concentrate in **June–August** (wettest month
**August**, driest **January**). The wettest hour of day is **09:00** and the
driest **20:00**.

![Seasonal rain](figures/fig_rain_om_seasonal.png)
![Diurnal rain](figures/fig_rain_om_diurnal.png)

**Insight:** the morning peak overlaps the **commute**, which is exactly when a
missed warning hurts most — an argument for evaluating warning lead time by hour
of day, not only by season.

### 3.3 Space matters, and the nowcast grid is not "Hong Kong"

Annual rain varies across the grid (≈1,770–2,400 mm). More importantly, the
nowcast **121 × 121** grid reaches from ~21.3–23.5 N and ~113–115 E — only
**5.6% of its cells fall in a Hong Kong bounding box**. A naive "wettest cell in
the grid" is often rain over the Pearl River Delta or the sea.

![Nowcast grid extent](figures/fig_nowcast_grid_extent.png)

Within the HK box, per-cell-day extremes are not as rare as the hourly grid
suggests (**≥ 15 mm/30 min on 8.9% of cell-days**) because each cell-day folds a
median of **384** snapshots and we take the daily maximum — a reminder that
"rarity" depends entirely on the unit of analysis (grid-hour vs cell-day).

![Nowcast daily maxima](figures/fig_nowcast_daily_max.png)

### 3.4 The two rain sources only loosely agree

Comparing daily HK-wide maxima from Open-Meteo and the nowcast gives
**r = 0.30** — they mark the *same* days as heavy only sometimes.

![Cross-source agreement](figures/fig_rain_crosscheck.png)

**Insight:** their definitions differ (ERA5 hourly area-average vs a 2-hour
nowcast accumulation), and the nowcast is itself a **forecast**, not an
observation. We must not treat either as the definitive label; the eventual
Task 3 label needs a clearly-chosen rain definition.

## 4. Surprises and contradictions (short)

- **No rain in the station table.** The intent was ground-truth rainfall; the
  feed returned only temperature and tide. Contradicts our Task 3 label plan.
- **No rain amount in the forecast feed.** Task 1 as written (rain-forecast
  accuracy) needs the archived 9-day text forecast, which we have not collected.
- **Forecast stability looked "too good" (0.07 °C).** Because the runs are hours
  apart, not days apart — an artefact of sampling, not a claim about reliability.
- **The nowcast grid is mostly outside Hong Kong**, so unfiltered extrema are
  meaningless; clipping changed the cross-source correlation from 0.00 to 0.30.
- **Warmest ≠ most urban.** The temperature extremes track elevation (Tai Mo
  Shan, Ngong Ping), not simply city vs country.
- **Minor glitches:** 3,492 forecast rows (≈5%) carry a *past* valid time, and one
  nowcast cell has longitude `14.688°E` (a dropped digit). Both are harmless once
  filtered but must be handled before modelling.

## 5. Assumptions we formed, refined, or discarded

| Assumption | Status | Evidence |
|---|---|---|
| Station history gives rainfall ground truth | **Discarded** | Only `Max/Min/Mean_Temp_C` and tide columns exist |
| OCF forecast provides a rain amount | **Discarded** | Forecast feed has temperature/wind/weather-code only; code is empty |
| All 61 stations report every day | **Refined** | 46 have temperature; 3 (BHD, TLS, YCT) start late or miss days |
| Temperature differs by region (Island/Kowloon/NT/Lantau) | **Refined** | Spread is driven mainly by elevation/coast |
| Nowcast covers Hong Kong | **Discarded** | Only ~5.6% of cells in the HK box |
| Forecasts change a lot as the target nears | **Unproven yet** | Hourly-spaced runs change little; needs the daily 9-day archive |
| Rainstorm hours are rare | **Confirmed** | 1 in ~17,070 grid-hours at 30 mm/h |

## 6. Non-numeric context that matters

- HKO's amber/red/black rules are **hourly** (30/50/70 mm in one hour); daily
  totals cannot reproduce them, which is why sub-daily rain (grids) is essential.
- Hong Kong's rain is driven by **monsoon/typhoon-season convection** and
  **terrain lift** — hence the sharp May–September peak and hilly-area extremes.
- Many warnings follow **rain already measured** (nowcast case), so a weak
  long-horizon (12 h) warning model is an *expected, reportable* result.

## 7. How this shaped our understanding of the two tasks

- **Task 1.** Temperature is the only variable we can currently score
  end-to-end, so the first reliability/validity charts will be for temperature by
  lead time, **after** we collect the historical 9-day forecast archive (rain +
  probability of precipitation). Error must be cut by **hour of day, season, and
  elevation**, with the morning peak in mind.
- **Task 3.** Rare-event scoring is non-negotiable. The nowcast gives fine
  spatial structure but must be clipped to Hong Kong and paired with a
  well-defined label; the ERA5 hourly grid quantifies just how rare the target is.

## 8. Immediate next steps from EDA

1. Collect the **data.gov.hk 9-day forecast archive** (rain amount, PoP) — the
   missing piece for Task 1.
2. Add **hourly station rainfall** (or adopt a documented grid label) for Task 3.
3. Standardise a **Hong Kong bounding box** and an **elevation-aware station
   grouping**.
4. Add EDA-time **data-quality filters**: drop past-valid forecast rows, drop
   out-of-range coordinates, and record late-starting stations.

## 9. Reproduce

```bash
.venv/Scripts/python.exe EDA/_inventory.py       # schemas & coverage
.venv/Scripts/python.exe EDA/task1_stations.py   # temperature ground truth
.venv/Scripts/python.exe EDA/task1_forecasts.py  # forecast structure & validity
.venv/Scripts/python.exe EDA/task3_rain.py       # rain rarity & seasonality
.venv/Scripts/python.exe EDA/_checks.py          # data-quality checks
```

Numbers are also written to `EDA/results/*.json`; figures to `EDA/figures/`.
