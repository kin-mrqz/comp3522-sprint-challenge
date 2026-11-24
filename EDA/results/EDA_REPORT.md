# COMP3522 sprint challenge - AQHI exploratory analysis

Source: `data/raw/aqhi_data.csv`, the EPD hourly AQHI export for 2021-01-01 to 2026-06-30.
Scope: Task 1 (historical patterns) plus the AQHI-only evidence needed for Task 2(b).
This is exploratory analysis. No predictive model is fit, and no weather variable is joined here
(see "What this cannot support").

Everything below is generated from the scripts in `EDA/code/` and stored in
`EDA/results/figures/`, `EDA/results/tables/` and `EDA/results/headline_numbers.json`.
The machine-readable finding list is `EDA/results/tables/t55_all_findings.csv`.

---

## 1. What was built

| Step | Script | Output |
|---|---|---|
| 00 parse raw, split Daily Max, flag `*` and `10+` | `00_build_processed.py` | `data/processed/*.parquet`, cleaning log, audit |
| 01 data trust | `01_profile.py` | F01-F06, t01-t07 |
| 02 temporal structure | `02_temporal.py` | F07-F20, t08-t23 |
| 03 episodes and persistence | `03_episodes.py` | F21-F27, t24-t31 |
| 04 spatial structure | `04_spatial.py` | F28-F40, t32-t44 |
| 05 decision layer | `05_decision.py` | F41-F46, t45-t54 |
| 06 report assembly | `06_report.py` | findings sheet, inventories, headline numbers |

Run order is 00, 01, 02, 03, 04, 05, 06. Total runtime is about three minutes in the repo venv.

### Conventions that change the numbers

- `Hour` 1..24 is mapped to `hour_of_day` 0..23 (hour beginning). All hour statements use that mapping.
- `Daily Max` rows are removed from the hourly table. They equal the maximum of the 24 hourly
  values in every station-day checked (share = 1.000000), so keeping them would double-count each day.
- `10+` is stored as 10 with `censored = True`. It is a lower bound. There are 304 such cells.
- `*` marks a value borrowed from the most similar station (`substituted = True`); 2.69% of cells.
  Values are kept and flagged.
- Missing cells are never imputed. The daily territory table keeps a full calendar with `in_file = False`
  on dates that are absent or blank.
- Bands follow EPD: Low 1-3, Moderate 4-6, High 7, Very High 8-10, Serious 10+.

---

## 2. Headline findings

| # | Finding | Number | Figure |
|---|---|---|---|
| 1 | The file is complete enough to use, but the gaps are block-shaped, not random | 1.25% of 850,608 station-hours missing; 38 absent dates; 17 blank dates | F01, F04 |
| 2 | Substitution is small overall and uneven by station-year | 2.69% of hours; Tuen Mun 2026 = 17.7% | F02, F06 |
| 3 | There is no five-year trend in the annual mean | 3.21-3.41; OLS -0.019/yr, p = 0.50 | F07, F13 |
| 4 | Season is the largest temporal signal | winter 3.70 vs summer 2.54 | F08, F11 |
| 5 | Every station peaks in the late afternoon | 15:00-17:00 holds 49% of daily peaks | F10, F38, F44 |
| 6 | High days are a spring and autumn phenomenon, not a winter one | Spring 31.4%, Autumn 29.7%, Winter 27.2%, Summer 11.7% of 401 high days | F21, F43 |
| 7 | One spring 2023 episode dominates the record | 2023-02-26 to 03-18, 21 days, peak 10; 2023 had 115 high days vs 56-70 | F22, F24 |
| 8 | A high day nearly triples tomorrow's odds | 57.3% vs 20.6% base; 11.0% after a non-high day | F25 |
| 9 | The highest-mean station is not the station with the heaviest tail | Causeway Bay 3.55 mean but 1.35% of hours >= 7; Tuen Mun 3.42 mean and 2.95% | F28, F37, F40 |
| 10 | The safest daytime window is early morning in every month | 06:00-11:59; starting at 12:00 raises the risk about 13x | F41, F45 |

---

## 3. Data trust

The raw export has 49,225 rows and 18 station columns: 47,256 hourly timestamps plus one Daily Max row
per day. After melting, the analysis frame is 850,608 station-hours over 1,969 dates.

The missingness is not random. Three shapes appear (F01, F04):

- 38 calendar dates are absent from the export: 2023-08-25..31 and the whole of 2024-12.
- 17 dates are present but blank for all 18 stations: 2023-01-07, 2023-06-27 and 2023-10-01..15.
- Tap Mun is additionally blank 2023-09-07..25 and all of October 2023.

That makes the longest continuous outage 31 days for every station (December 2024). Tap Mun has
90 days with no value in total, in several blocks. F05 shows that missingness is flat across hours
of the day: these are whole-day outages, not hourly dropouts.

Substitution (F02) is 2.69% of cells overall, but Tuen Mun 2026 is 17.7%, North 2026 6.4% and
Central 2026 6.2%. Substituted hours average 3.373 against 3.319 for own measurements, a +0.054
difference (F06). That is small but not zero: a station mean mixes two data-generating processes,
so any station ranking should carry this caveat.

The value distribution (F03) is concentrated: 59.7% of station-hours are Low 1-3, 37.5% are
Moderate 4-6, and only 1.53% are High or above. The `10+` cells are 304 of 850,608 (0.036%), so
censoring does not move a mean, but it does matter for statements about the peak.

## 4. Temporal structure

The annual mean is flat: 3.364 (2021), 3.290 (2022), 3.408 (2023), 3.214 (2024), 3.307 (2025).
An OLS slope over five points is -0.019 AQHI per year with p = 0.50; over 2021-2024 it is -0.033
with p = 0.50 (t23). With five annual means and a mid-series change in the index construction, this
is not evidence of a downward trend. The honest statement is "flat within year-to-year variability".

The seasonal cycle is much larger than the trend (F08, F19): winter (Dec-Feb) averages 3.70 and
summer (Jun-Aug) 2.54, a 1.16 AQHI swing. Every year repeats the shape. Winter 2025 (4.08) and
winter 2021 (3.98) are the worst seasons; autumn 2022 (4.05) is the worst non-winter season, driven
by the September 2022 episode.

The diurnal cycle is equally regular (F10, F11):
- minimum at 06:00-07:00,
- maximum at 16:00-17:00,
- amplitude largest in autumn and winter, smallest in summer.

The single highest month-hour cell is January 17:00 at 4.75; the lowest is around 06:00 in June-August
at about 2.1-2.3. Day of week is not a lever: weekdays 3.325 vs weekend 3.306, a -0.019 difference
(F12).

The band mix is stable across years: the low band is 57.6-66.0% every year and the high-or-above
share is 1.14-1.93% (F14). The daily maximum distribution shifts only slightly (F15).

Memory is short at the daily scale and long at the hourly scale (F16):
- hourly lag-1 autocorrelation 0.986, lag-24 0.748,
- daily lag-1 0.819, lag-7 0.332.

The hour-to-hour band transition matrix (F17) says a High hour stays High or worse 80.9% of the
time, but a Low hour stays Low 96.8% of the time. There is no fast escape from a bad afternoon,
and no fast entry into one from a clean morning.

One construction caveat sits inside the series. EPD updated the AQHI compilation on 2025-03-22
(ozone risk factor). A matched Apr-Dec comparison, 2025 against 2021-2024, gives -0.107 AQHI
(95% CI -0.463 to +0.249) for the mean (F18), and the >= 7 share is 1.51% in 2025 against 1.69%
before. So this sample does not show a level step. It still means the pre- and post-March-2025
index is not strictly one instrument, and a 2026 comparison inherits that.

## 5. Episodes and persistence

Using "daily maximum across the 18 stations >= 7" as a high day, 401 of 1,952 days with data are
high (20.5%). The yearly counts are 70 (2021), 69 (2022), 115 (2023), 56 (2024), 64 (2025), 27 in
the 181 days of 2026 (F21, t25).

The month pattern is not "winter is bad" (F21, F43):
- Spring 31.4% of high days, autumn 29.7%, winter 27.2%, summer 11.7%.
- April 2025 reaches 30% of days, January 2026 45%, March 2023 71%, April 2023 77%.
- September 2022 reaches 67%, October 2024 52%.
- June is 6% and August 8% of days.

The 2023 spring episode is the dominant single pattern (F22, F24): 2023-02-26 to 2023-03-18,
21 consecutive days, peak 10, up to 16 stations at or above 7. 2023 has 115 high days against
56-70 in the other full years. A five-year average hides this completely.

Most episodes are short (F22): 172 daily episodes, median 2 days, 45% last one day. Peak height
correlates only 0.38 with duration (F23), so a long episode is not automatically a severe one.
The hourly view is the same (F27): 399 episodes, median 6 hours, 58% at most 6 hours, longest 82 hours.

Persistence is useful but not decisive (F25):
- P(tomorrow high) = 20.6% base,
- 57.3% if today was high,
- 11.0% if today was not high,
- by today's band: 3% after 1-3, 14% after 4-6, 47% after 7, 59% after 8-9, 77% after 10+.

The day-to-day band matrix (F46) gives the same message from the other side: P(tomorrow Low | today
Low) = 69%, P(tomorrow Moderate | today Moderate) = 75%, and a Serious day is followed by another
Serious day 33% of the time.

High days are usually local (F26). Half of them (201 of 401) touch at most 3 of the 18 stations;
only 36 (9.0% of high days) have all 18 stations at or above 7, and 118 (29.4% of high days) have
a majority. "The territory had a high day" is therefore a weak statement; the district matters.

## 6. Spatial structure

Station means run from 2.97 (Tung Chung) to 3.55 (Causeway Bay) (F28). The three roadside stations
average 3.50 against 3.28 for the fifteen general stations, a +0.22 roadside premium. The premium
is not constant over the day (F36): it is smallest in the early morning and peaks at 21:00 at +0.32.

The mean ranking and the tail ranking are different (F37, F40). Tuen Mun has 2.95% of its hours at
or above 7 and Tap Mun 2.86%, while Causeway Bay - the highest mean - has 1.35%. The stations with
the largest diurnal amplitude also have the heaviest tail (r = 0.83): Tap Mun 1.62 AQHI, Tuen Mun
1.40, Tung Chung 1.36, against about 1.0 at most urban stations.

This is visible in the station-by-hour maps (F38, F39). Every station peaks at 16:00-17:00, but at
17:00 the share of hours at or above 7 is 8.3% at Tap Mun and 10.5% at Tuen Mun, against 4.1% at
Causeway Bay. The dataset carries no pollutant concentrations, so the mechanism is an inference,
not a measurement: the pattern is consistent with afternoon ozone formation at the rural and
western stations, while the urban roadside stations carry a higher baseline driven by traffic.

Co-movement is high (F30): the mean off-diagonal correlation of daily anomalies is 0.93. The
clustering (F31) still separates Tap Mun and Tung Chung from the other sixteen stations, which is
the same pair that leads the amplitude-tail relationship. The territory mean is a reasonable
summary, but it smooths a real two-regime structure.

The daily range across stations has a median of 0.96 AQHI and a maximum of 3.7, and it rises with
the territory mean (r = 0.46) (F32). On a bad day, one district can be two or three bands worse than
another. Regional means (F33) order Hong Kong Island 3.44, Kowloon 3.41, New Territories 3.23.

Station ranks are stable across years: Spearman 2021 vs 2025 = 0.90 (F34). Level and variability
are only weakly related (r = 0.24) (F35): a high mean does not imply a volatile station.

## 7. Decision layer for the scenario

The EPD definitions matter. At AQHI 7 (High), people with existing heart or respiratory illnesses
are advised to reduce outdoor physical exertion. At 8-10 the general public is advised to reduce,
and at 10+ to reduce to the minimum. For someone with lung cancer and asthma, 7 is the stop line,
not 8.

The month-by-hour risk surface (F41) is the core result:

- The worst daytime cell is September 16:00 at 16.4% of station-hours at or above 7.
- The best cells have zero observed exceedances: January 06:00-09:00, February 06:00-08:00,
  March 07:00.
- The spread between best and worst is more than 20-fold.

The season-hour curves (F42) put summer lowest at every hour and winter highest. The daily maximum
band mix (F43) gives the day-level version: 6% of June days and 8% of August days reach 7 somewhere,
against 29% in January, 31% in April and 29% in September.

The daily peak is a late-afternoon event (F44): median 15:00, 49% of peaks between 15:00 and 17:59,
77% between 12:00 and 18:59. Morning is not just lower on average; it is where the peak has not yet
arrived.

The six-hour window analysis (F45) is blunt: in every month the lowest-risk window starts at 06:00.
Moving the start to 12:00 multiplies the risk of a station-day hitting 7 by about 13. A practical
outing is 06:00-11:59, with the afternoon treated as the danger block.

### Conditions that maximise a safe outing

| Condition | Rule | Evidence |
|---|---|---|
| Month | Prefer June and August; treat January, April and September as high-risk months | F21, F43 |
| Hour | Go out 06:00-11:59; avoid 14:00-18:00 | F41, F44, F45 |
| Day after a high day | Postpone if today or yesterday reached 7 anywhere; risk is 57% after a high day | F25, F46 |
| Day after a low day | Risk is 11% (3% if today was 1-3) | F25 |
| District | In Tuen Mun, Yuen Long, Tung Chung or Tap Mun, the afternoon tail is 2-3x the urban tail | F37, F39, F40 |
| Station type | A roadside reading (Causeway Bay, Central, Mong Kok) has a higher mean but a thinner tail; do not use it as a proxy for a rural outing | F28, F36 |
| Threshold | Use 7 as the stop line for a sensitive person, 8 for the general public | EPD bands |

These are descriptive rules, not a validated decision model. They are the conditions the data
supports; a formula that combines them should be fitted and tested on a held-out period before it
is presented as a risk score.

## 8. What this analysis cannot support

1. **No weather join.** Task 2(a) asks for one or two weather variables from the project. The repo's
   `context/hko/` says the weather tables were left out on purpose, and `data/` contains only AQHI.
   A temperature, humidity, wind or rainfall analysis cannot be produced from this folder. The next
   step is to pull the project's weather tables (HKO/Open-Meteo) and join them at territory-day or
   station-day grain. Until then, Task 2(a) is unanswered.
2. **No causal claim about episodes.** The AQHI is a composite of four pollutants. The rural
   afternoon-tail pattern is consistent with ozone, but the data has no pollutant columns, so the
   mechanism is a hypothesis.
3. **No trend claim.** Five annual means, a mid-series method change and 2023's single episode do not
   separate a trend from multi-year variability.
4. **No instrument continuity across 2025-03-22.** The EPD compilation changed. The matched comparison
   shows no detectable mean step, but a 2026 comparison still carries the caveat.
5. **No station coordinates.** The file has station names only. Spatial statements use the EPD region
   and station-type classification, not distance. A map or a distance-based graph would require an
   external station table, which the sprint brief does not allow.
6. **Hourly `Hour` convention is assumed.** The export does not state whether Hour 1 is 00:00-00:59
   or 01:00-01:59. All hour-of-day statements use hour beginning; a one-hour shift would move every
   clock label but not the shape of the diurnal curve.

## 9. Figure index

Every generated figure is listed here. "Read" is the one-line takeaway; "Limit" is what the figure
does not establish.

| Fig | Shows | Read | Limit |
|---|---|---|---|
| F01 | Missing % by station-year heatmap | Only 2023 and Tap Mun move; the rest sit near 1% | Does not show which dates; F04 does |
| F02 | Substitution % by station-year | A few 2026 station-years are heavily borrowed | Substitution is flagged, not corrected |
| F03 | Hourly AQHI value distribution | 59.7% low, 37.5% moderate, 1.53% high+ | Discrete index; a mean is not a health score |
| F04 | Daily capture raster, date x station | Gaps are blocks: Aug 2023, Oct 2023, Dec 2024 | Blank cells are not low AQHI |
| F05 | Missing % by station and hour | Flat across hours: whole-day outages | Does not explain the outage cause |
| F06 | Own vs substituted mean | Substituted hours run +0.054 higher | Small; not a correction |
| F07 | Territory daily mean and 30-day rolling | Flat level, strong seasonal oscillation | No trend inference |
| F08 | Monthly mean by year, one line per year | Seasonal shape repeats every year | 2026 is H1 only |
| F09 | Year x month mean heatmap | January and autumn are the warm colours | Cell means mix years' weather |
| F10 | Month x hour mean heatmap | Afternoon peak, January 17:00 highest | Pooled across years |
| F11 | Diurnal curve by season | Min 06:00-07:00, max 16:00-17:00 | Averages hide episode days |
| F12 | Weekday profile with CI | No meaningful weekday effect | None |
| F13 | Annual mean, high days, high-hour share | Flat mean; 2023 stands out on high days | Five points only |
| F14 | Band mix by year | Mix is stable; low band 58-62% | Band cut points are EPD's |
| F15 | Daily maximum boxplots by year | Distributions overlap heavily | Shows no trend |
| F16 | ACF, hourly and daily | Hourly memory long, daily memory short | Unconditional; no covariates |
| F17 | Hour-to-hour band transition | 80.9% stay >= 7; 96.8% stay low | Station-level, pooled |
| F18 | Monthly series with 2025-03-22 line | No detectable level step | Wide CI; method change is real |
| F19 | Season x year mean heatmap | Winter 2025 and autumn 2022 are the peaks | Small cells |
| F20 | Share of hours >= 7 by year-month | Mirrors F21 at the hour level | Pooled stations |
| F21 | Share of days >= 7 by year-month | 2023 Feb-Apr, 2022 Sep, 2024 Oct stand out | Any-station definition is local |
| F22 | Daily episode duration histogram | 45% one day, median 2, max 21 | Episode definition uses any station |
| F23 | Peak vs duration scatter | r = 0.38; long does not mean severe | Small n |
| F24 | Five highest-peaking episodes, hourly | 2023 spring and 2022/2025 April dominate | Context windows only |
| F25 | Persistence bars and escalation by today's band | 57.3% after high vs 20.6% base; 77% after 10+ | One-day horizon only |
| F26 | Stations >= 7 per high day | Half of high days touch <= 3 stations | Any-station threshold |
| F27 | Hourly episode duration histogram | Median 6 h, 58% <= 6 h, max 82 h | Any-station threshold |
| F28 | Station mean ranking with CI | Causeway Bay 3.55 to Tung Chung 2.97 | Daily-mean CI ignores autocorrelation |
| F29 | Station x month mean heatmap | Urban stations high in winter, rural high in autumn | Pooled years |
| F30 | Daily-anomaly correlation heatmap | Mean off-diagonal 0.93 | Anomalies, not levels |
| F31 | Station dendrogram | Tap Mun and Tung Chung split from the core | Cut height is a choice |
| F32 | Spatial range, monthly and by season | Median 0.96, max 3.7; widens on dirty days | Name-level spread, no distance |
| F33 | Regional monthly means | HK Island 3.44, Kowloon 3.41, NT 3.23 | Regional averaging hides station tails |
| F34 | Station rank by year | Spearman 0.90; ranks are stable | Rank, not level |
| F35 | Mean vs daily SD scatter | r = 0.24; level is not volatility | None |
| F36 | Diurnal by station type | Roadside premium peaks 21:00 at +0.32 | Three roadside stations only |
| F37 | Exceedance share by station | Tuen Mun 2.95%, Tap Mun 2.86%, Causeway Bay 1.35% | Any-station hours, not days |
| F38 | Station x hour mean heatmap | All peak 16:00-17:00, amplitude differs | Pooled years |
| F39 | Station x hour exceedance heatmap | Afternoon tail is 2-3x higher in the west/rural | No pollutant data |
| F40 | Amplitude vs exceedance scatter | r = 0.83 | 18 points, mechanism inferred |
| F41 | P(>= 7) by month and hour | Worst 16.4% Sep 16:00; best cells 0% | Pooled years; month instability |
| F42 | P(>= 7) and P(>= 8) by season-hour | Summer lowest at every hour | Same pooling caveat |
| F43 | Daily max band mix by month | 6% Jun, 8% Aug, 29% Jan, 31% Apr, 29% Sep | Any-station daily max |
| F44 | Hour of the daily peak | Median 15:00; 49% in 15:00-17:59 | Territory mean peak |
| F45 | Six-hour window mean and risk by month | Every best window starts 06:00; noon start = 13x risk | Fixed 6 h length |
| F46 | Today band -> tomorrow band | 69% low->low, 75% moderate->moderate | One-day horizon |

## 10. Reproduction and file map

```bash
cd comp3522-sprint-challenge
.venv/bin/python EDA/code/00_build_processed.py
.venv/bin/python EDA/code/01_profile.py
.venv/bin/python EDA/code/02_temporal.py
.venv/bin/python EDA/code/03_episodes.py
.venv/bin/python EDA/code/04_spatial.py
.venv/bin/python EDA/code/05_decision.py
.venv/bin/python EDA/code/06_report.py
```

| Path | Contents |
|---|---|
| `data/raw/aqhi_data.csv` | unchanged EPD export |
| `data/processed/` | hourly, station-day, territory-day, wide tables; README with schema |
| `EDA/code/` | seven numbered scripts plus `eda_lib.py` |
| `EDA/results/figures/` | F01-F46 PNG |
| `EDA/results/tables/` | t00-t55 CSV, including the finding list |
| `EDA/results/EDA_FINDINGS.md` | machine-generated findings and inventories |
| `EDA/results/headline_numbers.json` | numbers for the deck and answers.md |
| `EDA/results/EDA_REPORT.md` | this report |
